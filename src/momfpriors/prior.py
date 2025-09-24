"""A module for defining prior distributions over Pipeline Space Hyperparameters."""
#TODO: Update when Eddie releases the new version of neps

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import neps
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    FloatHyperparameter,
    IntegerHyperparameter,
    OrdinalHyperparameter,
)
from neps.search_spaces import SearchSpace

from momfpriors.utils import pipeline_space_to_cs

if TYPE_CHECKING:
    from neps.search_spaces import Parameter


CONFIDENCE_SCORES: Mapping[float, str] = {
    0.5: "low",
    0.25: "medium",
    0.125: "high",
}

class Prior:
    """A base class for creating prior distributions over Pipeline Space hyperparameters."""

    prior_config: Mapping[str, float]

    prior_space: SearchSpace

    seed: int

    def __init__(  # noqa: D107
        self,
        input_space: ConfigurationSpace | Mapping[str, Parameter],
        prior_config: Mapping[str, float],
        seed: int = 0,
        std: float = 0.125,
    )-> None:
        self.prior_config = prior_config
        self.seed = seed
        self.confidence = CONFIDENCE_SCORES.get(std, 0.125)
        # np.random.seed(seed)  # noqa: NPY002
        import torch
        torch.manual_seed(seed)
        match input_space:
            case ConfigurationSpace():
                self.prior_space = self._cs_to_pipeline_space_with_priors(input_space)
            case Mapping():
                self.prior_space = SearchSpace(**input_space)
            case _:
                raise ValueError(
                    "hyperparameters must be a list[Hyperparameter], ConfigurationSpace,"
                    "or Mapping[Parameter]. "
                    f"Got {type(input_space).__name__}."
                )


    def _cs_to_pipeline_space_with_priors(
        self,
        input_space: ConfigurationSpace
    ) -> SearchSpace:
        """Convert a ConfigSpace to Pipeline Space with priors."""
        _pipeline_space: Mapping[str, Parameter] = {}
        for hp in list(input_space.values()):
            _default = self.prior_config[hp.name]
            match hp:
                case FloatHyperparameter():
                    _pipeline_space[hp.name] = neps.Float(
                        lower=hp.lower,
                        upper=hp.upper,
                        log=hp.log,
                        default=_default,
                        default_confidence=self.confidence,
                    )
                case IntegerHyperparameter():
                    _pipeline_space[hp.name] = neps.Integer(
                        lower=hp.lower,
                        upper=hp.upper,
                        log=hp.log,
                        default=_default,
                        default_confidence=self.confidence,
                    )
                case CategoricalHyperparameter() | OrdinalHyperparameter():
                    _pipeline_space[hp.name] = neps.Categorical(
                        choices=hp.choices,
                        default=_default,
                        default_confidence=self.confidence,
                    )
                case Constant():
                    _pipeline_space[hp.name] = neps.Constant(
                        value=hp.value,
                    )
                case _:
                    raise ValueError(
                        f"Unsupported hyperparameter type: {type(hp).__name__}"
                    )
        return SearchSpace(**_pipeline_space)

    def sample(self) -> SearchSpace:
        """Sample a Pipeline Space Configuration from the prior distribution."""
        return self.prior_space.sample(user_priors=True)


    def sample_cs(self) -> Configuration:
        """Sample a ConfigSpace Configuration from the prior distribution."""
        _config = self.sample()
        _config = {
                hp: _config[hp].value
                for hp in _config
            }
        return Configuration(
            configuration_space=pipeline_space_to_cs(self.prior_space),
            values=_config,
        )

