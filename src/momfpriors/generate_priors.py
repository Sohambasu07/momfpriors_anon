from __future__ import annotations

import argparse
import copy
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path

import yaml
from hpoglue import FunctionalBenchmark, Result

from momfpriors.benchmarks import BENCHMARKS

# from momfpriors.benchmarks.bbob_mo import bbob_function_definitions, create_bbob_mo_desc
from momfpriors.constants import DEFAULT_PRIORS_DIR
from momfpriors.utils import bench_first_fid, cs_random_sampling, get_prior_configs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_priors_wrt_obj(  # noqa: C901, PLR0912
    seed: int,
    nsamples: int,
    prior_spec: Iterable[tuple[str, int, float | None, float | None]],
    to: Path,
    benchmarks: (
        Mapping[str, str | list[str]]
        | list[Mapping[str, str | list[str]]]
    ),
    fidelity: int | float | None = None,
    *,
    clean: bool = False,
) -> None:
    """Generate priors for the given benchmarks.

    Args:
        seed: The seed to use for generating the priors.

        nsamples: The number of samples to generate.

        prior_spec: The prior specification to use for generating the priors.

        to: The path to save the priors.

        benchmarks: The benchmarks to generate priors for.
            Each Mapping should have the benchmark as the key and the objective to
            generate the priors for as the value.

        fidelity: The fidelity to use for the benchmarks.
            Defaults to None.
            Only uses the first available fidelity for the benchmark.

        clean: Whether to clean the priors directory before generating the priors.
            Defaults to False.
    """
    if to.exists() and clean:
        for child in filter(lambda path: path.is_file(), to.iterdir()):
            child.unlink()

    to.mkdir(exist_ok=True)

    logger.info(f"Priors generation with {seed=}, {nsamples=} and saving to {to.resolve()}")

    for benchmark, objectives in benchmarks.items():

        if benchmark.startswith("bbob"):
            # _benchmark = create_bbob_mo_desc(func=benchmark)
            pass
        else:
            assert benchmark in BENCHMARKS, f"Unknown benchmark: {benchmark}"
            _benchmark = BENCHMARKS[benchmark]

        if isinstance(_benchmark, FunctionalBenchmark):
            _benchmark = _benchmark.desc


        log_info = (
            f"Generating priors for benchmark: {_benchmark.name}"
            f" and objective(s): {objectives}"
            f" for spec: {prior_spec}"
        )
        if fidelity is not None:
            log_info += f" with fidelity: {fidelity}"

        logger.info(log_info)


        max_fidelity = bench_first_fid(_benchmark).max

        if fidelity is not None:
            if (
                isinstance(max_fidelity, int)
                and isinstance(fidelity, float)
                and fidelity.is_integer()
            ):
                fidelity = int(fidelity)

            if type(fidelity) is not type(max_fidelity):
                raise ValueError(
                    f"Cannot use fidelity {fidelity} (type={type(fidelity)}) with"
                    f" benchmark {_benchmark.name}",
                )
            at = fidelity
        else:
            at = max_fidelity

        results: list[Result] = []
        bench = _benchmark.load(_benchmark)
        for query in cs_random_sampling(
            benchmark=_benchmark,
            nsamples=nsamples,
            seed=seed,
            at=at,
        ):
            results.append(bench.query(query))

        logger.info(" - Finished results")

        print("==========================================================")

        for objective in objectives:

            logger.info(f" - Objective: {objective}")

            processed_results = []
            for result in results:
                if objective in _benchmark.metrics:
                    res = _benchmark.metrics[objective].as_minimize(result.values[objective])
                elif objective in _benchmark.test_metrics:
                    res = _benchmark.test_metrics[objective].as_minimize(result.values[objective])
                elif objective in _benchmark.costs:
                    res = _benchmark.costs[objective].as_minimize(result.values[objective])
                else:
                    raise ValueError(f"Unknown objective: {objective}")

                _result = copy.deepcopy(result)
                _result.values[objective] = res
                processed_results.append(_result)

            prior_configs = get_prior_configs(
                results=processed_results,
                space=_benchmark.config_space,
                objective=objective,
                seed=seed,
                prior_spec=prior_spec,
            )

            logger.info(f" - Priors: {prior_configs}")

            for name, config in prior_configs.items():
                save_path = to / f"{_benchmark.name}_{objective}_{name}.yaml"
                with save_path.open("w") as f:
                    yaml.dump(
                        {
                            "benchmark": _benchmark.name,
                            "prior_name": name,
                            "objective": objective,
                            "config": config.values,
                        }, f
                    )
                logger.info(f"Priors saved to: {save_path.resolve()}")
    logger.info("Done generating priors.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=0
    )
    parser.add_argument(
        "--nsamples", "-n",
        type=int,
        default=10
    )
    parser.add_argument(
        "--prior_spec", "-p",
        nargs="+",
        type=str,
        default=[
            "good:0:0.01:None",
            "medium:0:0.125:None",
            "bad:-1:0:None",
        ],
    )
    parser.add_argument(
        "--to", "-t",
        type=str,
        default=DEFAULT_PRIORS_DIR,
    )
    parser.add_argument(
        "--benchmarks", "-b",
        nargs="+",
        type=str,
        default=[
            "MOMFPark:value1",
            "MOMFPark:value2",
        ],
    )
    parser.add_argument(
        "--fidelity", "-f",
        type=float,
        default=None
    )
    parser.add_argument(
        "--clean", "-c",
        action="store_true"
    )
    parser.add_argument(
        "--yaml", "-y",
        type=str,
        default=None,
        help="Path to a yaml file containing the prior specification."
    )
    args = parser.parse_args()

    if args.yaml:
        with Path(args.yaml).open() as f:
            prior_gen = yaml.safe_load(f)
        _benchmarks = prior_gen["benchmarks"]
        _prior_spec = prior_gen["prior_spec"]
        _to = prior_gen.get("to", DEFAULT_PRIORS_DIR)
        _fidelity = prior_gen.get("fidelity", args.fidelity)
        _clean = prior_gen.get("clean", args.clean)
        _seed = prior_gen.get("seed", args.seed)
        _nsamples = prior_gen.get("nsamples", args.nsamples)

        if not isinstance(_prior_spec, list):
            _prior_spec = [_prior_spec]
        if isinstance(_to, str):
            _to = Path(_to)

    else:

        if isinstance(args.benchmarks, str):
            args.benchmarks = [args.benchmarks]

        _benchmarks: list[Mapping[str, str]] = []
        for benchmark in args.benchmarks:
            if ":" not in benchmark:
                raise ValueError(
                    "Invalid benchmark specification!"
                    "Expected format: 'benchmark:objective'"
                    f"Got: '{benchmark}'"
                )
            bench_name, objective = benchmark.split(":")
            _benchmarks.append({bench_name.strip(): objective.strip()}
            )

        if isinstance(args.prior_spec, str):
            args.prior_spec = [args.prior_spec]

        _prior_spec: list[tuple[str, int, float | None, float | None]] = []
        for prior in args.prior_spec:
            if ":" not in prior:
                raise ValueError(
                    "Invalid prior specification!"
                    "Expected format: 'name:index:std:categorical_swap_chance'"
                    f"Got: '{prior}'"
                )
            name, index, std, categorical_swap_chance = prior.split(":")
            _prior_spec.append(
                (
                    name.strip(),
                    int(index.strip()),
                    float(std.strip()) if std != "None" else None,
                    float(categorical_swap_chance.strip())
                    if categorical_swap_chance != "None"
                    else None,
                )
            )

        if isinstance(args.to, str):
            args.to = Path(args.to)
        _seed = args.seed
        _nsamples = args.nsamples
        _to = args.to
        _fidelity = args.fidelity
        _clean = args.clean


    generate_priors_wrt_obj(
        seed=_seed,
        nsamples=_nsamples,
        prior_spec=_prior_spec,
        to=_to,
        benchmarks=_benchmarks,
        fidelity=_fidelity,
        clean=_clean,
    )
