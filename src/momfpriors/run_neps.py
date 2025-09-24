from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

import neps
import numpy as np
import yaml
from hpoglue import BenchmarkDescription
from neps.space.parsing import convert_configspace

from momfpriors.benchmarks.neps_benchmark_wrappers import mfpbench_benches
from momfpriors.constants import DEFAULT_PRIORS_DIR, DEFAULT_RESULTS_DIR, DEFAULT_ROOT_DIR
from momfpriors.utils import set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

GLOBAL_SEED = 42


def run_exps(
    *,
    optimizer: str,
    benchmark: tuple[str, dict],
    max_evaluations: int,
    results_dir: Path,
    priors_dir: Path,
    num_seeds: int | None = None,
) -> None:
    """Run parallel experiments using NEPS."""
    num_seeds = num_seeds if num_seeds is not None else 1
    seeds = generate_seeds(num_seeds)
    benchmark_name, objs_priors_fids = benchmark
    assert benchmark_name in mfpbench_benches, f"Benchmark {benchmark_name} not found."
    assert "objectives" in objs_priors_fids, (
        "Objectives must be specified in the benchmark config."
    )
    bench = mfpbench_benches[benchmark_name]


    _priors: Mapping[str, tuple[str, Mapping[str, Any]]] = {}
    objectives = list(objs_priors_fids["objectives"].keys())

    _prior_name = []

    for obj, prior_annot in objs_priors_fids["objectives"].items():
        _prior_name.append(f"{obj}.{prior_annot}")
        if prior_annot is not None:
            prior_path = priors_dir / f"{benchmark_name}_{obj}_{prior_annot}.yaml"
            with prior_path.open("r") as file:
                _priors[obj] = (
                    prior_annot,
                    yaml.safe_load(file)["config"]
                )
        else:
            _priors[obj] = (
                None,
                None
            )

    _prior_name = ";".join(_prior_name)
    priors: tuple[str, Mapping[str, Mapping[str, Any]]] = (
        _prior_name,
        {obj: prior[1] for obj, prior in _priors.items() if prior[1] is not None}
    )

    assert isinstance(bench, BenchmarkDescription), (
        f"Benchmark {benchmark_name} is not of type BenchmarkDescription."
        f"Got {type(bench)} instead."
    )
    pipeline_space = convert_configspace(
        bench.config_space,
    )

    if len(bench.fidelities.keys()) > 0:
        bench_fids = list(bench.fidelities.keys())
        fidelity = bench.fidelities[bench_fids[0]]

        fid_name = bench_fids[0]
        min_fidelity = fidelity.min
        max_fidelity = fidelity.max
        match fidelity.kind:
            case _ if fidelity.kind is int:
                pipeline_space.fidelities = {
                    f"{fid_name}": neps.Integer(
                        lower=min_fidelity, upper=max_fidelity, is_fidelity=True
                    )
                }
            case _ if fidelity.kind is float:
                pipeline_space.fidelities = {
                    f"{fid_name}": neps.Float(
                        lower=min_fidelity, upper=max_fidelity, is_fidelity=True
                    )
                }

    evaluate_pipeline = partial(
        bench.load(
            bench
        ).query,
        objectives=objectives
    )

    prior_centers = priors[1]

    prior_confidences = {
        obj: dict.fromkeys(
            prior.keys(),
            0.75
        )
        for obj, prior in priors[1].items()
    }

    opt = (
        optimizer,
        {
            "prior_centers": prior_centers,
            "prior_confidences": prior_confidences,
        }
    )


    for seed in seeds:
        name_parts: list[str] = [
            f"optimizer={optimizer}",
            f"benchmark={benchmark_name}",
            f"objectives={','.join(objectives)}",
            f"fidelities={fid_name if len(bench.fidelities) > 0 else 'None'}",
            f"prior={_prior_name if priors[1] else 'None'}",
            f"maxevals={max_evaluations}",
            f"seed={seed}",
        ]

        exp_name = ".".join(name_parts)

        exp_results_dir = results_dir / exp_name

        set_seed(seed)

        neps_runtime(
            optimizer=opt,
            evaluate_pipeline=evaluate_pipeline,
            pipeline_space=pipeline_space,
            results_dir=exp_results_dir,
            max_evaluations=max_evaluations,
        )


def neps_runtime(
    optimizer: str | tuple[str, Mapping[str, Any]],
    evaluate_pipeline,
    pipeline_space,
    results_dir: Path,
    max_evaluations: int,
) -> None:
    """Run a single NEPS experiment."""
    logging.basicConfig(level=logging.DEBUG)
    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=pipeline_space,
        root_directory=results_dir,
        overwrite_working_directory=False,
        continue_until_max_evaluation_completed=True,
        max_evaluations_total=max_evaluations,
        optimizer=optimizer,
        post_run_summary=True,
    )


def generate_seeds(
    num_seeds: int,
    offset: int = 0, # To offset number of seeds
):
    """Generate a set of seeds using a Global Seed."""
    _rng = np.random.default_rng(GLOBAL_SEED)
    _num_seeds = num_seeds + offset
    _seeds = _rng.integers(0, 2 ** 32, size=_num_seeds)
    return _seeds[offset:].tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir", "-root",
        type=Path,
        default=DEFAULT_ROOT_DIR,
        help="Absolute path to the root directory."
    )
    parser.add_argument(
        "--results-dir", "-res",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Path to the NEPS output results directory."
    )
    parser.add_argument(
        "--priors_dir", "-pd",
        type=Path,
        default=DEFAULT_PRIORS_DIR,
        help="Path to the directory containing the priors relative to the root dir."
    )
    parser.add_argument(
        "--exp_name", "-name",
        type=str,
        default="neps_parallel",
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--max-evaluations", "-evals",
        type=int,
        default=20,
        help="Maximum number of evaluations to perform.",
    )
    parser.add_argument(
        "--num_seeds", "-s",
        type=int,
        default=25,
        help="Number of random seeds to use.",
    )
    parser.add_argument(
        "--exp_config", "-cfg",
        type=str,
        default=None,
        help="Path to experiment configuration file.",
    )
    parser.add_argument(
        "--num_workers", "-n",
        type=int,
        default=1,
        help="Number of parallel workers being used."
        " Here, it is only used to identify the experiment directory."
        " Actual multiple workers must be handled externally.",
    )
    args = parser.parse_args()

    if isinstance(args.root_dir, str):
        args.root_dir = Path(args.root_dir)
    if isinstance(args.results_dir, str):
        args.results_dir = args.root_dir / args.results_dir
    if isinstance(args.priors_dir, str):
        args.priors_dir = args.root_dir / args.priors_dir

    assert args.exp_config, "Experiment configuration file must be specified."
    yaml_path = args.root_dir / args.exp_config

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

    args.exp_name = config.get("exp_name", args.exp_name)
    args.exp_name += f"_n_workers_{args.num_workers}"

    run_exps(
        optimizer=_optimizers[0][0],
        benchmark=_benchmarks[0],
        max_evaluations=config.get("max_evaluations", args.max_evaluations),
        results_dir=args.results_dir / args.exp_name,
        priors_dir=args.priors_dir,
        num_seeds=config.get("num_seeds", args.num_seeds)
    )
