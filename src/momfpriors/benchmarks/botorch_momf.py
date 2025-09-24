from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from botorch.test_functions.multi_objective_multi_fidelity import MOMFBraninCurrin, MOMFPark
from ConfigSpace import ConfigurationSpace
from hpoglue import FunctionalBenchmark, Measure, Query, Result
from hpoglue.fidelity import RangeFidelity


# MOMF Branin Currin
def _get_MOMFBC_space(seed: int = 0) -> ConfigurationSpace:
    return ConfigurationSpace({
            f"x{i}": (0.0, 1.0) for i in range(2)
        }
    )

def wrapped_MOMFBC(
    query: Query,
) -> Result:
    config = query.config.values
    fidelity = float(query.fidelity[-1]/100) if query.fidelity else 1.0
    X = torch.Tensor(np.array([config["x0"], config["x1"], fidelity]))
    out = MOMFBraninCurrin()(X).tolist()

    return Result(
        query=query,
        fidelity=query.fidelity,
        values={
            f"value{i}": -out[i-1] for i, _ in enumerate(out, start=1)
        }
    )

MOMFBC_Bench = FunctionalBenchmark(
    name="MOMFBraninCurrin",
    config_space=_get_MOMFBC_space(),
    metrics={
        "value1": Measure.metric((-np.inf, np.inf), minimize=True),
        "value2": Measure.metric((-np.inf, np.inf), minimize=True)
    },
    fidelities={
        "s": RangeFidelity.from_tuple((1, 100, 1))
    },
    query=wrapped_MOMFBC
)


# MOMF Park

def _get_MOMFPark_space() -> ConfigurationSpace:
    return ConfigurationSpace({
            f"x{i}": (0.0, 1.0) for i in range(4)
        }
    )

def wrapped_MOMFPark(
    query: Query,
) -> Result:
    config = query.config.values
    fidelity = float(query.fidelity[-1]/100) if query.fidelity else 1.0
    X = torch.Tensor(np.array([config[f"x{i}"] for i in range(4)] + [fidelity]))
    out = MOMFPark()(X).tolist()

    return Result(
        query=query,
        fidelity=query.fidelity,
        values={
            f"value{i}": -out[i-1] for i, _ in enumerate(out, start=1)
        }
    )

MOMFPark_Bench = FunctionalBenchmark(
    name="MOMFPark",
    config_space=_get_MOMFPark_space(),
    metrics={
        "value1": Measure.metric((-np.inf, np.inf), minimize=True),
        "value2": Measure.metric((-np.inf, np.inf), minimize=True)
    },
    fidelities={
        "s": RangeFidelity.from_tuple((1, 100, 1))
    },
    query=wrapped_MOMFPark
)


def run_rs(
    fn: Literal["MOMFBC", "MOMFPark"] = "MOMFPark",
    n_samples: int = 1000,
    seed: int = 0,
) -> list[Result]:
    """Run random search on the specified MOMF function."""
    from hpoglue import Config
    results = []
    bench_space = None
    bench = None
    match fn:
        case "MOMFBC":
            bench_space = _get_MOMFBC_space(seed)
            bench = wrapped_MOMFBC
        case "MOMFPark":
            bench_space = _get_MOMFPark_space()
            bench = wrapped_MOMFPark
        case _:
            raise ValueError(f"Unknown function: {fn}")

    for i in range(n_samples):
        config = Config(
            config_id=str(i),
            values=dict(bench_space.sample_configuration()),
        )
        query = Query(config=config, fidelity=None)
        results.append(list(bench(query).values.values()))
    return results
