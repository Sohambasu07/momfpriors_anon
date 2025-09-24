from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from momfpriors.benchmarks.mf_zdt import (
    _calc_pareto_front_zdt1,
    _calc_pareto_front_zdt6,
    mf_zdt1_fn,
    mf_zdt6_fn,
)

sns.set_theme(style="whitegrid")
sns.set_context("paper")


def test_rs(
    fn: str,
    seed: int = 42,
    n_configs: int = 10,
    n_fidelities: int = 10,
    *,
    test_optimum: bool = False
) -> np.ndarray:
    """Test MF ZDT benchmarks with Random Search, querying at multiple fidelities."""
    fids = np.linspace(1, 100, n_fidelities, dtype=int)
    if n_fidelities == 1:
        fids = [100]
    np.random.seed(seed)  # noqa: NPY002
    if not test_optimum:
        match fn:
            case "zdt1":
                configs = np.random.uniform(0, 1, (n_configs, 30)) # noqa: NPY002
            case "zdt6":
                configs = np.random.uniform(0, 1, (n_configs, 10))  # noqa: NPY002
    else:
        configs = [np.random.uniform(0, 1)]  # noqa: NPY002
        match fn:
            case "zdt1":
                configs.extend([0.0] * 29)
            case "zdt6":
                configs.extend([0.0] * 9)
        configs = np.array([configs] * n_configs)
    results = []
    for fid in fids:
        for config in configs:
            match fn:
                case "zdt1":
                    result = mf_zdt1_fn(config, fid / 100)
                case "zdt6":
                    result = mf_zdt6_fn(config, fid / 100)
                case _:
                    raise ValueError(f"Unknown function: {fn}")
            results.append((fid, result))
            print(f"FID: {fid}, Config: {config}, Result: {result}")
    return results
    return results


def plot(
    fn: str,
    num_points: int,
    *,
    show: bool = True
) -> np.ndarray:
    """Plot the true Pareto front of the MF ZDT-1 benchmark."""
    match fn:
        case "zdt1":
            pareto_front = _calc_pareto_front_zdt1(n_pareto_points=num_points)
        case "zdt6":
            pareto_front = _calc_pareto_front_zdt6(n_pareto_points=num_points)
        case _:
            raise ValueError(f"Unknown function: {fn}")

    if show:
        plt.figure(figsize=(8, 6))
        plt.scatter(pareto_front[:, 0], pareto_front[:, 1], label="True Pareto Front", color="blue")
        plt.title(f"True Pareto Front of {fn} Benchmark")
        plt.xlabel("Objective 1 (f1)")
        plt.ylabel("Objective 2 (f2)")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid()
        plt.legend()
        plt.show()
    return pareto_front


def plot_zdt1_rs(
    num_points: int,
    fn: str,
    seed: int = 42,
    n_configs: int = 10,
    n_fidelities: int = 10,
    *,
    test_optimum: bool = False
) -> None:
    """Plot the true Pareto front of the MF ZDT-1 benchmark with Random Search results."""
    match fn:
        case "zdt1":
            pareto_front = _calc_pareto_front_zdt1(n_pareto_points=num_points)
        case "zdt6":
            pareto_front = _calc_pareto_front_zdt6(n_pareto_points=num_points)
        case _:
            raise ValueError(f"Unknown function: {fn}")
    results = test_rs(
        fn=fn,
        seed=seed,
        n_configs=n_configs,
        n_fidelities=n_fidelities,
        test_optimum=test_optimum
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], label="True Pareto Front", color="blue")
    for i, (fid, result) in enumerate(results):
        plt.scatter(result[0], result[1], label=f"RS{i+1}_{fid}")
    plt.title("MF ZDT-1 Benchmark with Random Search Results")
    plt.xlabel("Objective 1 (f1)")
    plt.ylabel("Objective 2 (f2)")
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    plt.grid()
    if len(results) <= 10:  # noqa: PLR2004
        plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--function", "-fn",
        type=str,
        default="zdt1",
        choices=["zdt1", "zdt6"],
        help="The benchmark function to plot the true Pareto front for."
    )
    parser.add_argument(
        "--n_points", "-n",
        type=int,
        default=100,
        help="Number of points to plot on the Pareto front."
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--show", "-sh",
        action="store_true",
        help="Whether to show the plot."
    )
    parser.add_argument(
        "--rs", "-rs",
        action="store_true",
        help="Whether to plot the results of Random Search."
    )
    parser.add_argument(
        "--n_configs", "-nc",
        type=int,
        default=10,
        help="Number of configurations to sample for Random Search."
    )
    parser.add_argument(
        "--n_fidelities", "-nf",
        type=int,
        default=10,
        help="Number of fidelities to sample for Random Search."
    )
    parser.add_argument(
        "--test_optimum", "-optimum",
        action="store_true",
        help="Whether to test the optimum configuration in Random Search."
    )
    args = parser.parse_args()
    if args.rs:
        plot_zdt1_rs(
            num_points=args.n_points,
            fn=args.function,
            seed=args.seed,
            n_configs=args.n_configs,
            n_fidelities=args.n_fidelities,
            test_optimum=args.test_optimum,
        )
    else:
        pf = plot(num_points=args.n_points, fn=args.function, show=args.show)