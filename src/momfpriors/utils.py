from __future__ import annotations

import importlib.metadata
import logging
import os
import site
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import neps
import numpy as np
from hpoglue import BenchmarkDescription, Config, Query, Result
from packaging import version

if TYPE_CHECKING:
    import pandas as pd
    from ConfigSpace import ConfigurationSpace


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DEFAULT_CONFIDENCE_SCORES: Mapping[str, float] = {
    "low": 0.5,
    "medium": 0.25,
    "high": 0.125,
}


def find_incumbent(
    df: pd.DataFrame,
    results_col: str,
    objective: str,
    *,
    minimize: bool = True,
) -> float:
    """Find the best value of the objective in the results column of the dataframe.

    Args:
        df : The dataframe containing the results.

        results_col : The name of the column containing the results.

        objective : The name of the objective to minimize.

        minimize : Whether to minimize the objective. Defaults to True.

    Returns:
        The best value of the objective.
    """
    if df.empty:
        return np.nan
    vals = []
    for i in df[results_col]:
        vals.append(i[objective])
    if not minimize:
        return max(vals)
    return min(vals)

def get_prior_configs(
    results: list[Result],
    space: ConfigurationSpace,
    objective: str,
    seed: int,
    prior_spec: Iterable[tuple[str, int, float | None, float | None]]
) -> Mapping[str, Config]:
    """Generate prior configurations based on the given results and specifications.

    Args:
        results: A list of Result objects to be used for generating prior configurations.

        space: The configuration space within which the configurations are defined.

        objective: The objective key used to sort the results.

        seed: The seed for random number generation to ensure reproducibility.

        prior_spec: An iterable of tuples specifying the prior configurations.
            Each tuple contains:
                - name: The name of the prior configuration.
                - index: The index of the result to be used.
                - std: The standard deviation for perturbation.
                - categorical_swap_chance: The probability of swapping categorical values
                    during perturbation.

    Returns:
        A mapping of prior configuration names to their perturbed configurations.
    """
    match results:
        case list():
            pass
        case _:
            raise TypeError("results must be a DataFrame or a list of Results"
        )

    prior_spec = list(prior_spec)
    results = sorted(results, key=lambda r: r.values[objective])
    logger.info(" - Finished sorting")

    prior_configs = {
            name: (results[index].query.config, std, categorical_swap_chance)
            for name, index, std, categorical_swap_chance in prior_spec
        }

    return {
            name: perturb(
                config,
                space,
                seed=seed,
                std=std,
                categorical_swap_chance=categorical_swap_chance,
            )
            for name, (config, std, categorical_swap_chance) in prior_configs.items()
        }


def bench_first_fid(benchmark: BenchmarkDescription) -> int:
    """Get the first fidelity value of the benchmark."""
    return benchmark.fidelities[
        next(iter(benchmark.fidelities.keys()))
    ]


def cs_random_sampling(
    benchmark: BenchmarkDescription,
    nsamples: int,
    seed: int,
    at: int,
    ) -> list[Config]:
    """Sample configurations from the benchmark's configuration space.

    Args:
        benchmark: The benchmark description.

        nsamples: The number of samples to draw from the configuration space.

        seed: The seed for random number generation.

        at: The fidelity value to use for the sampled configurations.

    Returns:
        The sampled configurations as a list of Config objects.
    """
    benchmark.config_space.seed(seed)
    configs = benchmark.config_space.sample_configuration(nsamples)
    return [
        Query(
            config=Config(
                config_id=None,
                values=dict(config),
            ),
            fidelity=(bench_first_fid(benchmark), at)
        )
            for config in configs
    ]


def perturb(  # noqa: C901, PLR0912, PLR0915
    config: Config,
    space: ConfigurationSpace,
    seed: int,
    std: float | None = None,
    categorical_swap_chance: float | None = None,
) -> Config:
    """Perturb a configuration.

    Args:
        config: The configuration to perturb.

        space: The configuration space.

        seed: The seed for random number generation.

        std: The standard deviation of the perturbation.

        categorical_swap_chance: The chance of swapping a categorical value.

    Returns:
        Config: The perturbed configuration.
    """
    from ConfigSpace import (  # noqa: PLC0415
        CategoricalHyperparameter,
        Constant,
        NormalFloatHyperparameter,
        NormalIntegerHyperparameter,
        OrdinalHyperparameter,
        UniformFloatHyperparameter,
        UniformIntegerHyperparameter,
    )
    perturbed_val: dict[str, int | float] = {}

    for name, value in config.values.items():
        hp = space.get(name)
        assert 0 <= std <= 1, "Noise must be between 0 and 1"
        rng: np.random.RandomState
        if seed is None:
            rng = np.random.RandomState()
        elif isinstance(seed, int):
            rng = np.random.RandomState(seed)
        else:
            rng = seed

        match hp:
            case CategoricalHyperparameter():
                 # We basically with (1 - std) choose the same value,
                 # otherwise uniformly select at random

                if categorical_swap_chance:
                    std = categorical_swap_chance
                if std:
                    if rng.uniform() < 1 - std:
                        _val = value
                    else:
                        choices = set(hp.choices) - {value}
                        _val = rng.choice(list(choices))

                else:
                    _val = value

            case OrdinalHyperparameter():
                # We build a normal centered at the value

                if rng.uniform() < 1 - std:
                    _val = value
                else:

                    # [0, 1,  2, 3]                             # noqa: ERA001
                    #       ^  mean
                    index_value = hp.sequence.index(value)
                    index_std = std * len(hp.sequence)
                    normal_value = rng.normal(index_value, index_std)
                    index = int(np.rint(np.clip(normal_value, 0, len(hp.sequence))))
                    _val = hp.sequence[index]

            case Constant():
                _val = value

            case (
                NormalIntegerHyperparameter()
                | NormalFloatHyperparameter()
                | UniformFloatHyperparameter()
                | UniformIntegerHyperparameter()
            ):
                # TODO:
                # * https://github.com/automl/ConfigSpace/issues/287
                # * https://github.com/automl/ConfigSpace/issues/290
                # * https://github.com/automl/ConfigSpace/issues/291
                # Doesn't act as intended
                assert hp.upper is not None
                assert hp.lower is not None
                # assert hp.q is None
                assert isinstance(value, int | float)

                match hp:
                    case UniformIntegerHyperparameter():
                        if hp.log:
                            _lower = np.log(hp.lower)
                            _upper = np.log(hp.upper)
                        else:
                            _lower = hp.lower
                            _upper = hp.upper
                    case NormalIntegerHyperparameter():
                        if hp.log:
                            _lower = np.log(hp.lower)
                            _upper = np.log(hp.upper)
                        else:
                            _lower = hp.lower
                            _upper = hp.upper
                    case UniformFloatHyperparameter() | NormalFloatHyperparameter():
                        _lower = hp.lower_vectorized
                        _upper = hp.upper_vectorized
                    case _:
                        raise TypeError("Wut")

                space_length = std * (_upper - _lower)
                rescaled_std = std * space_length

                if not hp.log:
                    sample = np.clip(rng.normal(value, rescaled_std), _lower, _upper)
                else:
                    logged_value = np.log(value)
                    sample = rng.normal(logged_value, rescaled_std)
                    sample = np.clip(np.exp(sample), hp.lower, hp.upper)

                if isinstance(hp, UniformIntegerHyperparameter | NormalIntegerHyperparameter):
                    _val = int(np.rint(sample))  # type: ignore

                if isinstance(hp, UniformFloatHyperparameter | NormalFloatHyperparameter):
                    _val = float(sample)  # type: ignore

            case _:
                raise ValueError(f"Can't perturb hyperparameter: {hp}")

        perturbed_val[name] = sanitize_numpy_prior_vals(_val)

    return Config(
        config_id=config.config_id,
        values=perturbed_val,
    )


def sanitize_numpy_prior_vals(
    val: Any,
) -> Any:
    """Convert numpy types to native Python types for readable YAMLs."""
    if isinstance(val, np.generic):
        return val.item()
    return val


def objective_fn_wrapper(
    objective_fn: Callable,
    **config: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Wraps an objective function to be called with a configuration.

    Args:
        objective_fn: The objective function to be called.
            It should accept a Query object as its argument.

        **config: Hyperparameter config to be passed to the objective function.

    Returns:
        The result values from the objective function.
    """
    query = Query(
        config=Config(config_id=None, values=config),
        fidelity=None,
    )
    result: Result = objective_fn(query)
    return result.values


def pipeline_space_to_cs(
    pipeline_space: Mapping[str, Any],
)-> ConfigurationSpace:
    """Convert a Pipeline Space to ConfigSpace."""
    from ConfigSpace import (  # noqa: PLC0415
        CategoricalHyperparameter,
        ConfigurationSpace,
        Constant,
        NormalFloatHyperparameter,
        NormalIntegerHyperparameter,
    )
    cs = ConfigurationSpace()
    for name, param in pipeline_space.items():
        match param:
            case neps.Float():
                cs.add(
                    NormalFloatHyperparameter(
                        name=name,
                        mu= param.default,
                        sigma=DEFAULT_CONFIDENCE_SCORES[param.default_confidence_choice],
                        lower=param.lower,
                        upper=param.upper,
                        default_value=param.default,
                        log=param.log,
                    )
                )
            case neps.Integer():
                cs.add(
                    NormalIntegerHyperparameter(
                        name=name,
                        mu= param.default,
                        sigma=DEFAULT_CONFIDENCE_SCORES[param.default_confidence_choice],
                        lower=param.lower,
                        upper=param.upper,
                        default_value=param.default,
                        log=param.log,
                    )
                )
            case neps.Categorical():
                cs.add(
                    CategoricalHyperparameter(
                        name=name,
                        choices=param.choices,
                        default_value=param.default,
                    )
                )
            case neps.Constant():
                cs.add(
                    Constant(
                        name=name,
                        value=param.value,
                    )
                )
            case _:
                raise ValueError(f"Unsupported parameter type: {type(param)}")
    return cs


class HiddenPrints:  # noqa: D101
    def __enter__(self):
        self._original_stdout = sys.stdout
        from pathlib import Path  # noqa: PLC0415
        sys.stdout = Path(os.devnull).open("w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def is_package_installed(package_name: str) -> bool:
    """Check if a package is installed in the current virtual environment."""
    # Extract version constraint if present
    version_constraints = ["==", ">=", "<=", ">", "<"]
    version_check = None
    for constraint in version_constraints:
        if constraint in package_name:
            package_name, version_spec = package_name.split(constraint)
            version_check = (constraint, version_spec)
            break

    # Normalize package name (replace hyphens with underscores)
    package_name = package_name.replace("-", "_")

    # Get the site-packages directory of the current virtual environment
    venv_site_packages = site.getsitepackages() if hasattr(site, "getsitepackages") else []
    venv_prefix = sys.prefix  # Virtual environment root

    # Check if the package is installed in the virtual environment
    for site_package_path in venv_site_packages:
        package_path = Path(site_package_path) / package_name

        # Check if the package exists in the site-packages directory
        if package_path.exists() and venv_prefix in str(package_path):
            installed_version = importlib.metadata.version(package_name)
            if version_check:
                constraint, required_version = version_check
                return _check_package_version(installed_version, required_version, constraint)
            return True

        # Check if package is installed as different name (e.g., .dist-info or .egg-info)
        dist_info_pattern = f"{package_name}*"
        dist_info_paths = list(Path(site_package_path).glob(dist_info_pattern))
        if dist_info_paths:
            dist_info_name = dist_info_paths[0].name.replace(".dist-info", "") \
                .replace(".egg-info", "")
            installed_version = dist_info_name.split("-")[-1]
            if version_check:
                constraint, required_version = version_check
                return _check_package_version(installed_version, required_version, constraint)
            return True

    return False

def _check_package_version(
    installed_version: str,
    required_version: str,
    check_key: str,
):
    """Check if the installed package version satisfies the required version."""
    installed_version = version.parse(installed_version)
    required_version = version.parse(required_version)
    match check_key:
        case "==":
            return installed_version == required_version
        case ">=":
            return installed_version >= required_version
        case "<=":
            return installed_version <= required_version
        case ">":
            return installed_version > required_version
        case "<":
            return installed_version < required_version


def set_seed(seed: int) -> None:
    """Set the seed for the optimizer."""
    import random  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002