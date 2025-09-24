from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from hpoglue import Config, Problem, Query, Result

from momfpriors.csprior import construct_prior
from momfpriors.optimizer import Abstract_AskTellOptimizer

if TYPE_CHECKING:
    from momfpriors.csprior import CSPrior


class RandomSearch(Abstract_AskTellOptimizer):
    """Random search optimizer."""

    name = "RandomSearch"

    support = Problem.Support(
        objectives=("single", "many"),
        fidelities=(None),
        cost_awareness=(None),
        tabular=False,
    )

    def __init__(  # noqa: D107
        self,
        problem: Problem,
        working_directory: str | Path,  # noqa: ARG002
        seed: int = 0,
        **kwargs: Any  # noqa: ARG002
    ):
        self.config_space = problem.config_space
        self.config_space.seed(seed)
        self.problem = problem
        self._optmizer_unique_id = 0

    def ask(self) -> Query:  # noqa: D102
        self._optmizer_unique_id += 1
        config = Config(
            config_id=str(self._optmizer_unique_id),
            values=dict(self.config_space.sample_configuration()),
        )
        return Query(config=config, fidelity=None)

    def tell(self, result: Result) -> None:  # noqa: ARG002, D102
        return


class RandomSearchWithPriors(Abstract_AskTellOptimizer):
    """Random search optimizer that incorporates priors for multi-objective optimization."""

    name = "RandomSearchWithPriors"

    support = Problem.Support(
        objectives=("single", "many"),
        fidelities=(None),
        cost_awareness=(None),
        tabular=False,
        priors=True,
    )

    mem_req_mb = 1024

    def __init__(  # noqa: D107
        self,
        problem: Problem,
        working_directory: str | Path,  # noqa: ARG002
        mo_prior_sampling: Literal["random", "equal", "scalarization"] = "random",
        seed: int = 0,
        **kwargs: Any
    ) -> None:
        self.config_space = problem.config_space
        self.config_space.seed(seed)
        self.problem = problem
        self.priors: Mapping[str, CSPrior] = construct_prior(
            priors=problem.priors[1],
            config_space=self.config_space,
            seed=seed,
            **kwargs,
        )
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        if mo_prior_sampling == "equal":
            assert len(self.problem.get_objectives()) <= self.problem.budget.total, (
                "When using `mo_prior_sampling='equal'` the number of objectives "
                "should be less than or equal to the total budget."
            )
        self.mo_prior_sampling = mo_prior_sampling
        self._optmizer_unique_id = 0
        self._priors_used = dict.fromkeys(self.priors, 0)

    def ask(self) -> Query:  # noqa: D102
        self._optmizer_unique_id += 1
        if len(self.priors.items()) > 1:
            match self.mo_prior_sampling:
                case "random":
                    prior = self._rng.choice(list(self.priors.values()))
                case "equal":
                    # raise NotImplementedError
                    min_usage = min(self._priors_used.values())
                    eligible_priors = [
                        key for key, count in self._priors_used.items()
                        if count == min_usage
                    ]
                    selected_prior_key = self._rng.choice(eligible_priors)
                    prior = self.priors[selected_prior_key]
                    self._priors_used[selected_prior_key] += 1
                case "scalarization":
                    weights = self._rng.random(len(self.priors))
                    weights /= weights.sum()

                    # Scalarizing the priors with prob weights
                    # to get a single prior
                    config_keys = list(self.priors[next(iter(self.priors))].prior_config.keys())
                    prior_mean = {}
                    for key in config_keys:
                        prior_mean[key] = sum(
                            [
                                v.prior_config[key] * weights[i]
                                for i, (k, v) in enumerate(self.priors.items())
                            ]
                        )
                    new_sigma = np.sqrt(
                        sum(0.25**2 * weights[i]**2 for i in range(len(self.priors)))
                    )
                    prior = next(iter(construct_prior(
                        priors={"scalarized_prior": prior_mean},
                        config_space=self.config_space,
                        prior_distribution="normal",
                        sigma=new_sigma,
                        seed=self.seed,
                    ).values()))
                case _:
                    raise ValueError(
                        "Invalid value for `mo_prior_sampling`. "
                        "Expected one of ['random', 'equal']."
                        f"Got {self.mo_prior_sampling}."
                    )
        else:
            prior = next(iter(self.priors.values()))
        config = Config(
            config_id=str(self._optmizer_unique_id),
            values=dict(prior.sample()),
        )
        return Query(config=config, fidelity=None)

    def tell(self, result: Result) -> None:  # noqa: ARG002, D102
        return