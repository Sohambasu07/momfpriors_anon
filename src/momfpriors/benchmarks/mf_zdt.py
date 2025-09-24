from __future__ import annotations

import numpy as np
from ConfigSpace import ConfigurationSpace
from hpoglue import FunctionalBenchmark, Measure, Query, Result
from hpoglue.fidelity import RangeFidelity

# MFZDT1

def mf_zdt1_fn(x: np.ndarray, s: float) -> list[float]:
    """Compute the value of the MF ZDT-1 function.

    The MF ZDT-1 function is the multi-fidelity augmentation of the
    multi-objective zdt-1 test function for optimization algorithms.
    It is defined as:
        f1(x) = x[0] + 0.1 * (1 - z)
        g(x) = 1 + 9 * sum(x[i] for i in range(1, len(x))) / (len(x) - 1)
        f2(x) = (1 - sqrt(f1(x) / g(x)) + 0.5 * (1 - z))

    where:
        g(x) = 1 + 9 * sum(x[i] for i in range(1, len(x))) / (len(x) - 1)

    Args:
        x: A 2-dimensional input array where x[0] is the first variable and
        the rest are additional variables.
        s: The fidelity parameter.

    Returns:
        The list of computed values of the MF ZDT-1 function.
    """
    f1 = x[0] + 0.1 * (1 - s)
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    f2 = (1 - np.sqrt(f1 / g) + 0.5 * (1 - s))
    return [f1, f2]

def _get_mfzdt1_space() -> ConfigurationSpace:
    """Creates and returns the configuration space for the MFZDT1 benchmark problem.

    The configuration space consists of 30 continuous variables named 'x0' to 'x29',
    each ranging from 0.0 to 1.0.
    """
    return ConfigurationSpace({
            f"x{i}": (0.0, 1.0) for i in range(30)
        }
    )

def wrapped_mfzdt1(
    query: Query,
) -> Result:
    """Wraps the MF ZDT-1 function and queries it with the given
    configuration and fidelity.
    """
    config = query.config.values
    fidelity = float(query.fidelity[-1]/100) if query.fidelity else 1.0
    X = np.array([config[f"x{i}"] for i in range(30)])
    out = mf_zdt1_fn(X, fidelity)

    return Result(
        query=query,
        fidelity=query.fidelity,
        values={
            f"f{i}": out[i-1] for i, _ in enumerate(out, start=1)
        }
    )

MFZDT1Bench = FunctionalBenchmark(
    name="MFZDT1",
    config_space=_get_mfzdt1_space(),
    metrics={
        "f1": Measure.metric((0, 1), minimize=True),
        "f2": Measure.metric((0, 1), minimize=True)
    },
    fidelities={
        "s": RangeFidelity.from_tuple((1, 100, 1))
    },
    query=wrapped_mfzdt1
)


def _calc_pareto_front_zdt1(n_pareto_points=100):
    x = np.linspace(0, 1, n_pareto_points)
    return np.array([x, 1 - np.sqrt(x)]).T




# MFZDT6



def mf_zdt6_fn(x: np.ndarray, s: float) -> list[float]:
    """Compute the value of the MF ZDT-6 function.

    The MF ZDT-6 function is the multi-fidelity augmentation of the
    multi-objective zdt-6 test function for optimization algorithms.
    It is defined as:

    The ordinary ZDT-6 function is defined as:
        f1(x) = 1 - np.exp(-4 * x[0]) * np.sin(6 * np.pi * x[0]) ** 6
        g(x) = 1 + 9 * (np.sum(x[1:]) / 9) ** 0.25
        f2(x) = 1 - (f1(x) / g(x)) ** 2

    We modify it to include a fidelity parameter `s` that scales the
    contributions of the variables and the function values.
    The modified function is defined as:
        f1(x) = 1 - np.exp(-4 * x[0]) * np.sin(6 * np.pi * x[0]) ** 6 + 0.1 * (1 - s)
        g(x) = 1 + 9 * (np.sum(x[1:]) / 9) ** 0.25
        f2(x) = (1 - (f1(x) / g(x)) ** 2) + 0.5 * (1 - s)


    Args:
        x: A 2-dimensional input array where x[0] is the first variable and
        the rest are additional variables.
        s: The fidelity parameter.

    Returns:
        The list of computed values of the MF ZDT-6 function.
    """
    f1 = 1 - np.exp(-4 * x[0] - 0.9 * (1 - s)) * np.sin(6 * np.pi * x[0]) ** 6
    g = 1 + 9 * (np.sum(x[1:]) / 9) ** 0.25
    f2 = (1 - (f1 / g) ** 2) + 0.5 * (1 - s)
    return [f1, f2]


def _get_mfzdt6_space() -> ConfigurationSpace:
    """Creates and returns the configuration space for the MFZDT6 benchmark problem.

    The configuration space consists of 10 continuous variables named 'x0' to 'x9',
    each ranging from 0.0 to 1.0.
    """
    return ConfigurationSpace({
            f"x{i}": (0.0, 1.0) for i in range(10)
        }
    )

def wrapped_mfzdt6(
    query: Query,
) -> Result:
    """Wraps the MF ZDT-6 function and queries it with the given
    configuration and fidelity.
    """
    config = query.config.values
    fidelity = float(query.fidelity[-1]/100) if query.fidelity else 1.0
    X = np.array([config[f"x{i}"] for i in range(10)])
    out = mf_zdt6_fn(X, fidelity)

    return Result(
        query=query,
        fidelity=query.fidelity,
        values={
            f"f{i}": out[i-1] for i, _ in enumerate(out, start=1)
        }
    )

MFZDT6Bench = FunctionalBenchmark(
    name="MFZDT6",
    config_space=_get_mfzdt6_space(),
    metrics={
        "f1": Measure.metric((0, 1), minimize=True),
        "f2": Measure.metric((0, 1), minimize=True)
    },
    fidelities={
        "s": RangeFidelity.from_tuple((1, 100, 1))
    },
    query=wrapped_mfzdt6
)

def _calc_pareto_front_zdt6(n_pareto_points=100):
    x = np.linspace(0.2807753191, 1, n_pareto_points)
    return np.array([x, 1 - np.power(x, 2)]).T
