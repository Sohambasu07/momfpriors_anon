from __future__ import annotations

from collections.abc import Callable
from functools import partial

import cocoex as ex
import numpy as np
from ConfigSpace import ConfigurationSpace
from hpoglue import BenchmarkDescription, Measure, Query, Result, SurrogateBenchmark
from hpoglue.env import Env

bbob_function_definitions = {
    "f1": "Sphere/Sphere",
    "f2": "Sphere/Ellipsoid separable",
    "f3": "Sphere/Attractive sector",
    "f4": "Sphere/Rosenbrock original",
    "f5": "Sphere/Sharp ridge",
    "f6": "Sphere/Sum of different powers",
    "f7": "Sphere/Rastrigin",
    "f8": "Sphere/Schaffer F7, condition 10",
    "f9": "Sphere/Schwefel",
    "f10": "Sphere/Gallagher 101 peaks",
    "f11": "Ellipsoid separable/Ellipsoid separable",
    "f12": "Ellipsoid separable/Attractive sector",
    "f13": "Ellipsoid separable/Rosenbrock original",
    "f14": "Ellipsoid separable/Sharp ridge",
    "f15": "Ellipsoid separable/Sum of different powers",
    "f16": "Ellipsoid separable/Rastrigin",
    "f17": "Ellipsoid separable/Schaffer F7, condition 10",
    "f18": "Ellipsoid separable/Schwefel",
    "f19": "Ellipsoid separable/Gallagher 101 peaks",
    "f20": "Attractive sector/Attractive sector",
    "f21": "Attractive sector/Rosenbrock original",
    "f22": "Attractive sector/Sharp ridge",
    "f23": "Attractive sector/Sum of different powers",
    "f24": "Attractive sector/Rastrigin",
    "f25": "Attractive sector/Schaffer F7, condition 10",
    "f26": "Attractive sector/Schwefel",
    "f27": "Attractive sector/Gallagher 101 peaks",
    "f28": "Rosenbrock original/Rosenbrock original",
    "f29": "Rosenbrock original/Sharp ridge",
    "f30": "Rosenbrock original/Sum of different powers",
    "f31": "Rosenbrock original/Rastrigin",
    "f32": "Rosenbrock original/Schaffer F7, condition 10",
    "f33": "Rosenbrock original/Schwefel",
    "f34": "Rosenbrock original/Gallagher 101 peaks",
    "f35": "Sharp ridge/Sharp ridge",
    "f36": "Sharp ridge/Sum of different powers",
    "f37": "Sharp ridge/Rastrigin",
    "f38": "Sharp ridge/Schaffer F7, condition 10",
    "f39": "Sharp ridge/Schwefel",
    "f40": "Sharp ridge/Gallagher 101 peaks",
    "f41": "Sum of different powers/Sum of different powers",
    "f42": "Sum of different powers/Rastrigin",
    "f43": "Sum of different powers/Schaffer F7, condition 10",
    "f44": "Sum of different powers/Schwefel",
    "f45": "Sum of different powers/Gallagher 101 peaks",
    "f46": "Rastrigin/Rastrigin",
    "f47": "Rastrigin/Schaffer F7, condition 10",
    "f48": "Rastrigin/Schwefel",
    "f49": "Rastrigin/Gallagher 101 peaks",
    "f50": "Schaffer F7, condition 10/Schaffer F7, condition 10",
    "f51": "Schaffer F7, condition 10/Schwefel",
    "f52": "Schaffer F7, condition 10/Gallagher 101 peaks",
    "f53": "Schwefel/Schwefel",
    "f54": "Schwefel/Gallagher 101 peaks",
    "f55": "Gallagher 101 peaks/Gallagher 101 peaks",
    "f56": "Sphere/Rastrigin separable",
    "f57": "Sphere/Rastrigin-Büche",
    "f58": "Sphere/Linear slope",
    "f59": "Separable Ellipsoid/Separable Rastrigin",
    "f60": "Separable Ellipsoid/Büche-Rastrigin",
    "f61": "Separable Ellipsoid/Linear Slope",
    "f62": "Separable Rastrigin/Büche-Rastrigin",
    "f63": "Separable Rastrigin/Linear Slope",
    "f64": "Büche-Rastrigin/Linear slope",
    "f65": "Attractive Sector/Step-ellipsoid",
    "f66": "Attractive Sector/rotated Rosenbrock",
    "f67": "Step-ellipsoid/separable Rosenbrock",
    "f68": "Step-ellipsoid/rotated Rosenbrock",
    "f69": "Separable Rosenbrock/rotated Rosenbrock",
    "f70": "Ellipsoid/Discus",
    "f71": "Ellipsoid/Bent Cigar",
    "f72": "Ellipsoid/Sharp Ridge",
    "f73": "Ellipsoid/Sum of different powers",
    "f74": "Discus/Bent Cigar",
    "f75": "Discus/Sharp Ridge",
    "f76": "Discus/Sum of different powers",
    "f77": "Bent Cigar/Sharp Ridge",
    "f78": "Bent Cigar/Sum of different powers",
    "f79": "Rastrigin/Schaffer F7 with conditioning of 1000",
    "f80": "Rastrigin/Griewank-Rosenbrock",
    "f81": "Schaffer F7/Schaffer F7 with conditioning 1000",
    "f82": "Schaffer F7/Griewank-Rosenbrock",
    "f83": "Schaffer F7 with conditioning 1000/Griewank-Rosenbrock",
    "f84": "Schwefel/Gallagher 21",
    "f85": "Schwefel/Katsuuras",
    "f86": "Schwefel/Lunacek bi-Rastrigin",
    "f87": "Gallagher 101/Gallagher 21",
    "f88": "Gallagher 101/Katsuuras",
    "f89": "Gallagher 101/Lunacek bi-Rastrigin",
    "f90": "Gallagher 21/Katsuuras",
    "f91": "Gallagher 21/Lunacek bi-Rastrigin",
    "f92": "Katsuuras/Lunacek bi-Rastrigin"
}



def _get_bbob_space(
    bbob_function: Callable,
    seed: int = 0
) -> ConfigurationSpace:
    lower_bounds = bbob_function.lower_bounds
    upper_bounds = bbob_function.upper_bounds
    return ConfigurationSpace(
        seed=seed,
        space={
            f"x{i}": (lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))
        }
    )


def _get_bbob_fn(func: str) -> Callable:
    f = func.split("_")[1]
    dims = func.split("_")[-1][:-1]
    suite_opts = f"function_indices:{f} dimensions:{dims}"
    return ex.Suite("bbob-biobj", "", suite_opts)[0]


def _bbob_mo_query_fn(
    query: Query,
    bbob_function: Callable
) -> Result:
    config = query.config.values
    assert query.fidelity is None, "Fidelity is not supported for BBOB Benchmark Suite."
    out = bbob_function(np.array([config[f"x{i}"] for i in range(2)]))

    return Result(
        query=query,
        fidelity=query.fidelity,
        values={
            f"value{i}": -out[i-1] for i, _ in enumerate(out, start=1)
        }
    )


def create_bbob_mo_desc(func: str) -> BenchmarkDescription:
    env = Env(
        name="bbob_biobj",
        python_version="3.10",
        requirements=("coco-experiment", "cocopp"),
        post_install=None
    )
    func_name = func
    if ":" in func:
        func = func.split(":")[0]
    bbob_function = _get_bbob_fn(func)
    return BenchmarkDescription(
            name=func_name,
            config_space=_get_bbob_space(bbob_function=bbob_function),
            load=partial(create_bbob_mo_bench, bbob_function=bbob_function),
            metrics={
                "value1": Measure.metric((-np.inf, np.inf), minimize=True),
                "value2": Measure.metric((-np.inf, np.inf), minimize=True),
            },
            test_metrics=None,
            costs=None,
            fidelities=None,
            has_conditionals=False,
            is_tabular=False,
            env=env,
            mem_req_mb=4096,
        )



def create_bbob_mo_bench(
    desc: BenchmarkDescription,
    bbob_function: Callable
) -> SurrogateBenchmark:
    query_function = partial(_bbob_mo_query_fn, bbob_function=bbob_function)
    return SurrogateBenchmark(
        desc=desc,
        config_space=desc.config_space,
        benchmark=bbob_function,
        query=query_function,
    )


def bbob_mo_benchmarks():
    for (func_num, _) in bbob_function_definitions.items():
        func_name = f"bbob_{func_num}_2D"
        yield create_bbob_mo_desc(func=func_name)