"""Script to interface with the HPOBench library."""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hpobench.abstract_benchmark
import hpobench.benchmarks
import numpy as np
from ConfigSpace import UniformIntegerHyperparameter
from hpoglue.benchmark import BenchmarkDescription, SurrogateBenchmark
from hpoglue.env import Env
from hpoglue.fidelity import ContinuousFidelity, RangeFidelity
from hpoglue.measure import Measure
from hpoglue.result import Result

if TYPE_CHECKING:
    import hpobench
    from hpoglue.query import Query

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

openml_logger = logging.getLogger("openml")
openml_logger.setLevel(logging.ERROR)

xgboost_logger = logging.getLogger("xgboost")
xgboost_logger.setLevel(logging.ERROR)

def _get_surrogate_benchmark(
    description: BenchmarkDescription,
    *,
    benchmark: hpobench.abstract_benchmark.AbstractBenchmark,
    datadir: Path | str | None = None,
    **kwargs: Any,
) -> SurrogateBenchmark:

    if datadir is not None:
        datadir = Path(datadir).absolute().resolve()
        kwargs["datadir"] = datadir
    query_function = partial(_hpobench_surrogate_query_function, benchmark=benchmark)
    return SurrogateBenchmark(
        desc=description,
        benchmark=benchmark,
        config_space=description.config_space,
        query=query_function,
    )


def _hpobench_surrogate_query_function(
        query: Query,
        benchmark: hpobench.abstract_benchmark.AbstractBenchmark
) -> Result:
    if query.fidelity is not None:
        assert isinstance(query.fidelity, tuple)
        _, fid_value = query.fidelity
    else:
        fid_value = None
    return Result(
        query=query,
        values=dict(benchmark.objective_function(
            configuration=query.config.values,
            fidelity=fid_value,
        )),
        fidelity=query.fidelity,
    )


openml_task_ids = [
    "10101",
    "53",
    "146818",
    "146821",
    "9952",
    "146822",
    "31",
    "3917",
    "168912",
    "3",
    "167119",
    "12",
    "146212",
    "168911",
    "9981",
    "167120",
    "14965",
    "146606",
    "7592",
    "9977",
]


def xgb(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    """Generates benchmark descriptions for the HPOBench OpenML XGBoost Benchmarks.

    Args:
        datadir (Path | None): The directory where the data is stored.
        If None, the default directory is used.

    Yields:
        Iterator[BenchmarkDescription]: An iterator over BenchmarkDescription objects
        for each XGBoost benchmark.
    """
    import hpobench.benchmarks.ml.xgboost_benchmark as xgb_bench
    env = Env(
        name="py310-hpobench-0.0.10-xgb",
        python_version="3.10",
        requirements=("hpobench[xgboost]==0.0.10",),
        post_install=None,
    )
    for task_id in openml_task_ids:
        bench = xgb_bench.XGBoostBenchmark(task_id=task_id, rng=1)
        fidelity_space = list(bench.get_fidelity_space().values())
        yield BenchmarkDescription(
            name=f"hpobench_xgb_{task_id}",
            config_space=bench.get_configuration_space(),
            load=partial(
                _get_surrogate_benchmark, benchmark=bench, datadir=datadir
            ),
            metrics={"function_value": Measure.metric(bounds=(0, 1), minimize=True)},
            test_metrics=None,
            costs={"cost": Measure.cost(bounds=(0, np.inf), minimize=True)},
            fidelities={
                hp.name: (
                    RangeFidelity.from_tuple(
                        [hp.lower, hp.upper, hp.size - hp.upper - hp.lower]
                    ) if isinstance(hp, UniformIntegerHyperparameter)
                    else ContinuousFidelity.from_tuple(
                        [hp.lower, hp.upper]
                    )
                ) for hp in fidelity_space
            },
            is_tabular=False,
            has_conditionals=False,
            env=env,
            mem_req_mb=24576,
        )


def hpobench_benchmarks(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    """Generates benchmark descriptions for various HPOBench benchmarks.

    Args:
        datadir (Path | None): The directory where the data is stored.
        If None, the default directory is used.

    Yields:
        Iterator[BenchmarkDescription]: An iterator over BenchmarkDescription objects
        for each benchmark.
    """
    yield from xgb()
