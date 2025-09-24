from __future__ import annotations

import argparse
import hashlib
import logging
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from hpoglue.utils import configpriors_to_dict

from momfpriors._run import Run
from momfpriors.constants import DEFAULT_PRIORS_DIR, DEFAULT_RESULTS_DIR, DEFAULT_ROOT_DIR

GLOBAL_SEED = 42


root_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", category=DeprecationWarning)


def exp(  # noqa: C901, PLR0912, PLR0913
    optimizers: (
        tuple[str, Mapping[str, Any]] |
        list[tuple[str, Mapping[str, Any]]]
    ),
    benchmarks: (
        tuple[str, Mapping[str, str | None]] |
        list[tuple[str, Mapping[str, str | None]]]
    ),
    seeds: int | list[int ] | None = None,
    num_seeds: int = 1,
    num_iterations: int = 1000,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    priors_dir: Path = DEFAULT_PRIORS_DIR,
    prior_distribution: Literal["normal", "uniform", "beta"] = "normal",
    exp_name: str | None = None,
    *,
    core_verbose: bool = False,
    overwrite: bool = False,
    use_continuations_as_budget: bool = False,
    num_runs_only: bool = False,
    **kwargs: Any
) -> None:
    """Run experiments with specified optimizers and benchmarks.

    Args:
        optimizers: A tuple or list of tuples where each tuple contains
            the optimizer name and its parameters.

        benchmarks: A tuple or list of tuples where each tuple contains
            the benchmark name and its parameters.

        seeds: A single seed, a list of seeds, or None to generate seeds automatically.

        num_seeds: Number of seeds to generate if seeds are not provided.

        num_iterations: Number of iterations for each run. Default is 1000.

        results_dir: Directory to store the results. Default is DEFAULT_RESULTS_DIR.

        priors_dir: Directory to store the priors. Default is DEFAULT_PRIORS_DIR.

        prior_distribution: Type of prior distribution to use. Default is "normal".

        exp_name: Name of the experiment.

        core_verbose: Whether to log verbose information during the core loop.

        overwrite: Whether to overwrite the results if they already exist.

        use_continuations_as_budget: Whether to use continuations as the budget.

        num_runs_only: If set to True, only count the number of runs without
            executing them or writing the YAMLs.

        **kwargs: Additional keyword arguments to pass to the Run.generate_run method.

    Returns:
        None
    """
    if not isinstance(optimizers, list):
        optimizers = [optimizers]

    if not isinstance(benchmarks, list):
        benchmarks = [benchmarks]

    if seeds and not isinstance(seeds, list):
        seeds = [seeds]

    if not seeds:
        seeds = generate_seeds(num_seeds)
    else:
        num_seeds = len(seeds)

    name_parts: list[str] = []
    name_parts.append(";".join([f"{opt[0]}{opt[-1]}" for opt in optimizers]))
    name_parts.append(";".join([f"{bench[0]}{bench[-1]}" for bench in benchmarks]))
    name_parts.append(f"seeds={seeds}")
    name_parts.append(f"budget={num_iterations}")
    if not exp_name:
        exp_name = hashlib.sha256((".".join(name_parts)).encode()).hexdigest()
    exp_dir = results_dir / exp_name
    exp_yaml_path = exp_dir / "exp.yaml"

    runs: list[Run] = []
    run_names: list[str] = []
    for benchmark in benchmarks:
        for optimizer in optimizers:
            if optimizer[0] == "MOMFBO" and not check_gpu_momfbo():
                root_logger.warning(
                    "GPU recommended for MOMFBO! Skipping this optimizer."
                )
                continue
            for seed in seeds:
                try:
                    run = Run.generate_run(
                        optimizer=optimizer,
                        benchmark=benchmark,
                        seed=seed,
                        num_iterations=num_iterations,
                        exp_dir=exp_dir,
                        priors_dir=priors_dir,
                        prior_distribution=prior_distribution,
                        **kwargs
                    )
                    if run.name in run_names:
                        continue
                    runs.append(run)
                    run_names.append(run.name)
                except Exception as e:  # noqa: BLE001
                    logging.error(
                        "Error in generating runs for "
                        f"{optimizer[0]} on {benchmark[0]} with seed {seed}"
                    )
                    logging.error(e)

    root_logger.info(f"Generated {len(runs)} runs.")

    if not num_runs_only:

        write_yaml(
            yaml_path=exp_yaml_path,
            runs=runs,
            exp_dir=exp_dir,
            seeds=seeds,
            num_seeds=num_seeds,
            budget=num_iterations,
        )

        root_logger.info("Running the experiments")

        for i, run in enumerate(runs):
            root_logger.info(f"Running {i + 1}/{len(runs)} runs.")
            run.run(
                core_verbose=core_verbose,
                overwrite=overwrite,
                use_continuations_as_budget=use_continuations_as_budget,
            )

        root_logger.info(f"Completed {len(runs)} runs.")

def generate_seeds(
    num_seeds: int,
    offset: int = 0, # To offset number of seeds
):
    """Generate a set of seeds using a Global Seed."""
    _rng = np.random.default_rng(GLOBAL_SEED)
    _num_seeds = num_seeds + offset
    _seeds = _rng.integers(0, 2 ** 32, size=_num_seeds)
    return _seeds[offset:].tolist()


def write_yaml(
    yaml_path: Path,
    runs: list[Run],
    exp_dir: Path,
    seeds: int | list[int],
    num_seeds: int,
    budget: int,
) -> None:
    """Write the experiment config to a YAML file."""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w") as file:
        yaml.dump(
            to_dict(
                runs=runs,
                exp_dir=exp_dir,
                seeds=seeds,
                num_seeds=num_seeds,
                budget=budget,
            ), file, sort_keys=False)


def to_dict(
    runs: list[Run],
    exp_dir: Path,
    seeds: int | list[int],
    num_seeds: int,
    budget: int,
) -> dict[str, Any]:
    """Convert the Experiment to a dictionary."""
    optimizers = []
    benchmarks = []
    continuations = 0
    bench_keys = {}
    opt_keys = []
    for run in runs:
        run.write_yaml()
        continuations += run.problem.continuations
        opt_name = run.name.split("benchmark")[0]
        if opt_name not in opt_keys:
            optimizers.append(
                (
                    run.optimizer,
                    run.optimizer_hyperparameters or {},
                )
            )
            opt_keys.append(opt_name)

        _store_val = {
            "objectives": run.problem.get_objectives(),
            "fidelities": run.problem.get_fidelities(),
            "costs": run.problem.get_costs(),
            "priors": run.problem.priors,
        }

        bench = bench_keys.setdefault(run.benchmark.name, (run.benchmark, []))

        if _store_val not in bench[1]:
            for variant in bench[1]:
                if variant["objectives"] == _store_val["objectives"]:
                    for key in ["fidelities", "costs", "priors"]:
                        variant[key] = variant[key] or _store_val[key]
                    break
            else:
                bench[1].append(_store_val)

    for _, v in bench_keys.items():
        [benchmarks.append(
            (
                v[0],
                var,
            )
        ) for var in v[1]]
    _optimizers = [{"name": opt[0].name, "hyperparameters": opt[1]} for opt in optimizers]
    _benchmarks = []
    for bench in benchmarks:
        _priors = None
        if bench[1]["priors"]:
            _priors = configpriors_to_dict(bench[1]["priors"])
            _priors = list(_priors)
        _benchmarks.append(
            {
                "name": bench[0].name,
                "objectives": bench[1]["objectives"],
                "fidelities": bench[1]["fidelities"],
                "costs": bench[1]["costs"],
                "priors": _priors,
            }
        )

    return {
        "name": exp_dir.name,
        "output_dir": str(exp_dir),
        "optimizers": _optimizers,
        "benchmarks": _benchmarks,
        "seeds": seeds,
        "num_seeds": num_seeds,
        "budget": budget,
        "continuations": continuations > 0,
    }


def check_gpu_momfbo() -> bool:
    """Check if the GPU is available for MOMFBO."""
    import torch
    return bool(torch.cuda.is_available())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", "-r",
        type=Path,
        default=DEFAULT_ROOT_DIR,
        help="Absolute path to the root directory."
    )
    parser.add_argument(
        "--results_dir", "-rs",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Path to the directory where the results will be stored relative to the root dir."
    )
    parser.add_argument(
        "--yaml_path", "-y",
        type=Path,
        help="Path to the YAML file containing the experiment config relative to the root dir."
    )
    parser.add_argument(
        "--exp_name", "-e",
        type=str,
        default=None,
        help="Name of the experiment."
    )
    parser.add_argument(
        "--optimizers", "-o",
        type=str,
        nargs="+",
        help="List of optimizers to run."
    )
    parser.add_argument(
        "--benchmarks", "-b",
        type=str,
        nargs="+",
        help="List of benchmarks to run."
    )
    parser.add_argument(
        "--seeds", "-s",
        type=int,
        nargs="+",
        help="List of seeds to run."
    )
    parser.add_argument(
        "--num_seeds", "-n",
        type=int,
        default=1,
        help="Number of seeds to run."
    )
    parser.add_argument(
        "--num_iterations", "-i",
        type=int,
        default=10,
        help="Number of iterations to run."
    )
    parser.add_argument(
        "--objectives", "-ob",
        type=str,
        nargs="+",
        help="List of objectives to run."
    )
    parser.add_argument(
        "--priors", "-p",
        type=str,
        nargs="+",
        help="List of priors for each objective."
    )
    parser.add_argument(
        "--prior_distribution", "-d",
        type=str,
        default="normal",
        help="Prior distribution to use for the benchmark priors."
    )
    parser.add_argument(
        "--priors_dir", "-pd",
        type=Path,
        default=DEFAULT_PRIORS_DIR,
        help="Path to the directory containing the priors relative to the root dir."
    )
    parser.add_argument(
        "--core_verbose", "-cv",
        action="store_true",
        help="Whether to log verbose information during the core loop."
    )
    parser.add_argument(
        "--overwrite", "-ow",
        action="store_true",
        help="Whether to overwrite the results if they already exist."
    )
    parser.add_argument(
        "--use_continuations_as_budget", "-c",
        action="store_true",
        help="Whether to use continuations as the budget."
    )
    parser.add_argument(
        "--num_runs_only", "-nruns",
        action="store_true",
        help="If set, only count the number of runs without executing them or writing the YAMLs."
    )

    args = parser.parse_args()

    if not isinstance(args.root_dir, Path):
        args.root_dir = Path(args.root_dir)

    if not isinstance(args.results_dir, Path):
        args.results_dir = Path(args.results_dir)
        args.results_dir = args.root_dir / args.results_dir

    if not isinstance(args.priors_dir, Path):
        args.priors_dir = Path(args.priors_dir)
        args.priors_dir = args.root_dir / args.priors_dir

    if args.yaml_path:
        yaml_path = args.root_dir / args.yaml_path

        with Path(yaml_path).open() as file:
            config = yaml.safe_load(file)

        _optimizers = []
        for opt in config["optimizers"]:
            match opt:
                case Mapping():
                    assert "name" in opt, f"Optimizer name not found in {opt}"
                    if len(opt) == 1 or not opt.get("hyperparameters"):
                        _optimizers.append((opt["name"], {}))
                    else:
                        _optimizers.append(tuple(opt.values()))
                case str():
                    _optimizers.append((opt, {}))
                case tuple():
                    assert len(opt) <= 2, "Each Optimizer must only have a name and hyperparameters"  # noqa: PLR2004
                    assert isinstance(opt[0], str), "Expected str for optimizer name"
                    if len(opt) == 1:
                        _optimizers.append((opt[0], {}))
                    else:
                        assert isinstance(opt[1], Mapping), (
                            "Expected Mapping for Optimizer hyperparameters"
                        )
                        _optimizers.append(opt)
                case _:
                    raise ValueError(
                        f"Invalid type for optimizer: {type(opt)}. "
                        "Expected Mapping, str or tuple"
                    )

        _benchmarks = []
        for bench in config["benchmarks"]:
            match bench:
                case Mapping():
                    assert "name" in bench, f"Benchmark name not found in {bench}"
                    assert "objectives" in bench, f"Benchmark objectives not found in {bench}"
                    _benchmarks.append(
                        (
                            bench["name"],
                            {
                                "objectives": bench["objectives"],
                                "fidelities": bench.get("fidelities"),
                            }
                        )
                    )
                case tuple():
                    assert len(bench) == 2, "Each Benchmark must only have a name and objectives"  # noqa: PLR2004
                    assert isinstance(bench[0], str), "Expected str for benchmark name"
                    assert isinstance(bench[1], Mapping), (
                        "Expected Mapping for Benchmark objectives and fidelities"
                    )
                    _benchmarks.append(bench)
                case _:
                    raise ValueError(
                        f"Invalid type for benchmark: {type(bench)}. "
                        "Expected Mapping or tuple"
                    )


        exp(
            optimizers=_optimizers,
            benchmarks=_benchmarks,
            seeds=config.get("seeds"),
            num_seeds=config.get("num_seeds", 1),
            num_iterations=config.get("num_iterations", 10),
            results_dir=args.results_dir or config.get("results_dir", DEFAULT_RESULTS_DIR),
            priors_dir=args.priors_dir or config.get("priors_dir", DEFAULT_PRIORS_DIR),
            prior_distribution=config.get("prior_distribution", "normal"),
            exp_name=args.exp_name,
            core_verbose=args.core_verbose,
            overwrite=args.overwrite,
            use_continuations_as_budget=True,
            num_runs_only=args.num_runs_only,
            **config.get("kwargs", {})
        )