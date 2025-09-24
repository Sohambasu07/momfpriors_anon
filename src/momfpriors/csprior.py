"""A module for creating prior distributions over ConfigSpace hyperparameters."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from ConfigSpace import (
    BetaFloatHyperparameter,
    BetaIntegerHyperparameter,
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace,
    Constant,
    Float,
    Integer,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

if TYPE_CHECKING:
    from hpoglue import Config


class CSPrior:
    """A base class for creating prior distributions over ConfigSpace hyperparameters."""

    prior_config: Mapping[str, float]

    distribution: ConfigurationSpace

    seed: int

    def __init__(  # noqa: D107
        self,
        config_space: ConfigurationSpace,
        prior_config: Mapping[str, float],
        seed: int = 0,
    )-> None:
        self.prior_config = prior_config
        self.seed = seed
        self.distribution = self._create_prior_distribution(config_space)

    @abstractmethod
    def _create_prior_distribution(
        self,
        config_space: ConfigurationSpace,
    ) -> ConfigurationSpace:
        ...


    def pdf(self, config: Configuration) -> list[float]:
        """Compute the probability density of a configuration.

        Args:
            config (Configuration): A configuration to compute the probability density.

        Returns:
            float: The probability density of the configuration.
        """
        return [hp.pdf_values(config[hp.name]) for hp in self.hyperparameters]


    @abstractmethod
    def log_pdf(self, config: Configuration) -> float:
        """Compute the log probability density of a configuration.

        Args:
            config (Configuration): A configuration to compute the log probability density.

        Returns:
            float: The log probability density of the configuration.
        """
        ...


    def sample(self) -> Configuration:
        """Sample a configuration from the distribution.

        Returns:
            Configuration: A sampled configuration from the distribution.
        """
        return self.distribution.sample_configuration()


    def meta(self) -> Mapping[str, Any]:
        """Return metadata about the prior distribution.

        Returns:
            Mapping[str, Any]: Metadata about the prior distribution.
        """
        _prior_type: str
        _kwargs: Mapping[str, float] = {}
        match type(self).__name__:
            case "CSNormalPrior":
                _prior_type = "normal"
                _kwargs["sigma"] = self.sigma
            case "CSUniformPrior":
                _prior_type = "uniform"
            case "CSBetaPrior":
                _prior_type = "beta"
                _kwargs["alpha"] = self.alpha
                _kwargs["beta"] = self.beta
        return {
            "prior_type": _prior_type,
            "distribution": self.distribution,
            "prior_config": self.prior_config,
            "seed": self.seed,
            "kwargs": _kwargs,
        }


class CSNormalPrior(CSPrior):
    """A class to represent a normal prior distribution for hyperparameters
    in a configuration space.


    Attributes:
    -----------
    config_space : ConfigurationSpace
        A configuration space to create the prior distribution.

    prior_config : Mapping[str, float]
        A mapping of hyperparameter names to their default values.

    seed : int, optional
        A seed for random number generation (default is 0).

    sigma : float, optional
        The standard deviation of the normal distribution (default is 0.25).


    Methods:
    --------
    _create_prior_distribution(self, config_space: ConfigurationSpace) -> ConfigurationSpace:
        Creates and returns a configuration space with the specified normal prior distribution
        for the hyperparameters.
    """

    def __init__(  # noqa: D107
        self,
        config_space: ConfigurationSpace,
        prior_config: Mapping[str, float],
        seed: int = 0,
        sigma: float = 0.25,
    ) -> None:
        self.sigma = sigma
        super().__init__(config_space, prior_config, seed)

    def _create_prior_distribution(
        self,
        config_space: ConfigurationSpace,
    ) -> ConfigurationSpace:
        distribution = ConfigurationSpace(seed=self.seed)
        for hp in list(config_space.values()):
            _default = self.prior_config[hp.name]
            match hp:
                case UniformFloatHyperparameter() | BetaFloatHyperparameter():
                    distribution.add(
                        NormalFloatHyperparameter(
                            name=hp.name,
                            mu=_default,
                            sigma=self.sigma,
                            lower=hp.lower,
                            upper=hp.upper,
                            default_value=_default,
                            log=hp.log,
                        )
                    )
                case UniformIntegerHyperparameter() | BetaIntegerHyperparameter():
                    distribution.add(
                        NormalIntegerHyperparameter(
                            name=hp.name,
                            mu=_default,
                            sigma=self.sigma,
                            lower=hp.lower,
                            upper=hp.upper,
                            default_value=_default,
                            log=hp.log,
                        )
                    )
                case CategoricalHyperparameter() | OrdinalHyperparameter():
                    hp.default_value = _default
                    distribution.add(hp)
                case Constant():
                    distribution.add(hp)
                case NormalFloatHyperparameter() | NormalIntegerHyperparameter():
                    hp.mu = _default
                    hp.sigma = self.sigma
                    hp.default_value = _default
                    distribution.add(hp)
                case _:
                    raise ValueError(
                        f"Unsupported hyperparameter type: {type(hp).__name__}"
                    )
        return distribution


class CSUniformPrior(CSPrior):
    """A class to represent a uniform prior distribution for hyperparameters
    in a configuration space.


    Attributes:
    -----------
    config_space : ConfigurationSpace
        A configuration space to create the prior distribution.

    prior_config : Mapping[str, float]
        A mapping of hyperparameter names to their default values.

    seed : int, optional
        A seed for random number generation (default is 0).


    Methods:
    --------
    _create_prior_distribution(self, config_space: ConfigurationSpace) -> ConfigurationSpace:
        Creates a configuration space with the specified uniform prior distribution
        for each hyperparameter.
    """

    def __init__(  # noqa: D107
        self,
        config_space: ConfigurationSpace,
        prior_config: Mapping[str, float],
        seed: int = 0,
    ) -> None:
        super().__init__(config_space, prior_config, seed)

    def _create_prior_distribution(
        self,
        config_space: ConfigurationSpace,
    ) -> ConfigurationSpace:
        distribution = ConfigurationSpace(seed=self.seed)
        for hp in list(config_space.values()):
            _default = self.prior_config[hp.name]
            match hp:
                case NormalFloatHyperparameter() | BetaFloatHyperparameter():
                    distribution.add(
                        UniformFloatHyperparameter(
                            name=hp.name,
                            lower=hp.lower,
                            upper=hp.upper,
                            default_value=_default,
                            log=hp.log,
                        )
                    )
                case NormalIntegerHyperparameter() | BetaIntegerHyperparameter():
                    distribution.add(
                        UniformIntegerHyperparameter(
                            name=hp.name,
                            lower=hp.lower,
                            upper=hp.upper,
                            default_value=_default,
                            log=hp.log,
                        )
                    )
                case CategoricalHyperparameter() | OrdinalHyperparameter():
                    hp.default_value = _default
                    distribution.add(hp)
                case Constant() | Float() | Integer():
                    distribution.add(hp)
                case UniformFloatHyperparameter() | UniformIntegerHyperparameter():
                    hp.default_value = _default
                    distribution.add(hp)
                case _:
                    raise ValueError(
                        f"Unsupported hyperparameter type: {type(hp).__name__}"
                    )
        return distribution


class CSBetaPrior(CSPrior):
    """A class to represent a Beta prior distribution for hyperparameters
    in a configuration space.


    Attributes:
    -----------
    config_space : ConfigurationSpace
        A configuration space to create the prior distribution.

    prior_config : Mapping[str, float]
        A mapping of hyperparameter names to their default values.

    seed : int, optional
        A seed for random number generation (default is 0).

    alpha : float, optional
        The alpha parameter of the Beta distribution (default is 2).

    beta : float, optional
        The beta parameter of the Beta distribution (default is 2).


    Methods:
    --------
    _create_prior_distribution(self, config_space: ConfigurationSpace) -> ConfigurationSpace:
        Creates a configuration space with the specified Beta prior distribution
        for each hyperparameter.
    """

    def __init__(  # noqa: D107
        self,
        config_space: ConfigurationSpace,
        prior_config: Mapping[str, float],
        seed: int = 0,
        alpha: float = 2,
        beta: float = 2,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        super().__init__(config_space, prior_config, seed)

    def _create_prior_distribution(
        self,
        config_space: ConfigurationSpace,
    ) -> ConfigurationSpace:
        distribution = ConfigurationSpace(seed=self.seed)
        for hp in list(config_space.values()):
            _default = self.prior_config[hp.name]
            match hp:
                case NormalFloatHyperparameter() | UniformFloatHyperparameter():
                    distribution.add(
                        BetaFloatHyperparameter(
                            name=hp.name,
                            alpha=self.alpha,
                            beta=self.beta,
                            lower=hp.lower,
                            upper=hp.upper,
                            default_value=_default,
                            log=hp.log,
                        )
                    )
                case NormalIntegerHyperparameter() | UniformIntegerHyperparameter():
                    distribution.add(
                        BetaIntegerHyperparameter(
                            name=hp.name,
                            alpha=self.alpha,
                            beta=self.beta,
                            lower=hp.lower,
                            upper=hp.upper,
                            default_value=_default,
                            log=hp.log,
                        )
                    )
                case CategoricalHyperparameter() | OrdinalHyperparameter():
                    hp.default_value = _default
                    distribution.add(hp)
                case Constant():
                    distribution.add(hp)
                case BetaFloatHyperparameter() | BetaIntegerHyperparameter():
                    hp.alpha = self.alpha
                    hp.beta = self.beta
                    hp.default_value = _default
                    distribution.add(hp)
                case _:
                    raise ValueError(
                        f"Unsupported hyperparameter type: {type(hp).__name__}"
                    )
        return distribution



def construct_prior(
    priors: Mapping[str, Mapping[str, Any] | Config],
    config_space: ConfigurationSpace,
    prior_distribution: Literal["normal", "uniform", "beta"] = "normal",
    seed: int = 0,
    **kwargs: Any,
) -> Mapping[str, CSPrior]:
    """Constructs a prior distribution for a given configuration space and prior values.

    Parameters:
    -----------
    priors:
        A mapping of prior configurations. The keys are the names of the objects,
        and the values are either dictionaries of prior parameters or Config objects.
    config_space:
        The configuration space for which the prior is being constructed.
    prior_distribution:
        The type of prior distribution to construct. Default is "normal".
    seed:
        The random seed for reproducibility. Default is 0.
    **kwargs:
        Additional keyword arguments for specific prior distributions.

    Returns:
    --------
        A dictionary where the keys are the names of the objects and the values
        are instances of the constructed prior distributions.
    """
    _prior_type: type[CSPrior]
    _prior_kwargs: Mapping[str, Any] = {}
    match prior_distribution:
        case "normal":
            _prior_type = CSNormalPrior
            _prior_kwargs["sigma"] = kwargs.get("sigma", 0.25)
        case "uniform":
            _prior_type = CSUniformPrior
        case "beta":
            _prior_type = CSBetaPrior
            _prior_kwargs["alpha"] = kwargs.get("alpha", 2)
            _prior_kwargs["beta"] = kwargs.get("beta", 2)
        case _:
            raise ValueError(f"Invalid value for `prior_distribution`: {prior_distribution}")

    return {
        obj: _prior_type(
                config_space=config_space,
                prior_config=prior if isinstance(prior, dict) else prior.values,
                seed=seed,
                **_prior_kwargs,
            )
        for obj, prior in priors.items()
    }