from __future__ import annotations

# https://syne-tune.readthedocs.io/en/latest/examples.html#ask-tell-interface
# https://syne-tune.readthedocs.io/en/latest/examples.html#ask-tell-interface-for-hyperband
import datetime
from abc import abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import ConfigSpace as CS  # noqa: N817
from hpoglue import Config, Problem, Query
from hpoglue.env import Env

from momfpriors.optimizer import Abstract_AskTellOptimizer

if TYPE_CHECKING:
    from hpoglue import Result
    from syne_tune.config_space import (
        Domain,
    )
    from syne_tune.optimizer.scheduler import TrialScheduler


class SyneTuneOptimizer(Abstract_AskTellOptimizer):
    """Base class for SyneTune Optimizers."""

    name = "SyneTune_base"

    env = Env(
        name="syne_tune-0.13.0",
        python_version="3.10.12",
        requirements=("syne_tune==0.13.0",),
    )

    @abstractmethod
    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: Path,
        optimizer: TrialScheduler,
    ):
        """Create a SyneTune Optimizer instance for a given problem."""
        working_directory.mkdir(parents=True, exist_ok=True)
        self.problem = problem
        self.optimizer = optimizer
        self._counter = 0
        import numpy as np
        self.brackets = np.arange(
            0,
            np.floor(
                np.log(self.problem.fidelities[1].max/ self.problem.fidelities[1].min)),
            1
        )
        self.fidelity_rungs = np.linspace(
            self.problem.fidelities[1].min,
            self.problem.fidelities[1].max,
            len(self.brackets)
        )
        self.fidelity_counter = 0

    def ask(self) -> Query:
        """Get a configuration from the optimizer."""
        from syne_tune.backend.trial_status import Trial

        self._counter += 1
        trial_suggestion = self.optimizer.suggest(self._counter)
        assert trial_suggestion is not None
        assert trial_suggestion.config is not None
        name = str(self._counter)
        trial = Trial(
            trial_id=self._counter,
            config=trial_suggestion.config,
            creation_time=datetime.datetime.now(),
        )

        fidelity = None
        match self.problem.fidelities:
            case None:
                raise ValueError(
                    "SyneTuneMOASHA is a multi-fidelity optimizer. "
                    "The Problem must define a fidelity."
                )
            case (fidelity_name, _):
                # fidelity = (fidelity_name, trial_suggestion.config.pop(fidelity_name))
                fidelity = (fidelity_name, self.fidelity_rungs[self.fidelity_counter])
                trial_suggestion.config.pop(fidelity_name)
                if self.fidelity_counter == len(self.brackets) - 1:
                    self.fidelity_counter = 0
            case Mapping():
                raise NotImplementedError("Many-fidelity not yet implemented for SyneTuneMOASHA.")
            case _:
                raise TypeError("Fidelity must be None, a tuple or a Mapping")


        # TODO: How to get the fidelity??
        return Query(
            config=Config(config_id=name, values=trial.config),
            fidelity=fidelity,
            optimizer_info=(self._counter, trial, trial_suggestion, self.fidelity_counter),
        )

    def tell(self, result: Result) -> None:
        """Update the SyneTune Optimizer with the result of a query."""
        match self.problem.objectives:
            case Mapping():
                results_obj_dict = {
                    name: metric.as_minimize(result.values[name])
                    for name, metric in self.problem.objectives.items()
                }
                # results_obj_dict["trial_counter"] = result.query.optimizer_info[1]
                results_obj_dict["fidelity_counter"] = result.query.optimizer_info[3]
            case (metric_name, metric):
                results_obj_dict = {metric_name: metric.as_minimize(result.values[metric_name])}
            case _:
                raise TypeError("Objective must be a string or a list of strings")

        if self.problem.get_fidelities() is not None:
            self.optimizer.on_trial_add(trial=result.query.optimizer_info[1])  # type: ignore

        decision = self.optimizer.on_trial_result(
            trial=result.query.optimizer_info[1],
            result=results_obj_dict,
        )
        from syne_tune.optimizer.scheduler import SchedulerDecision
        if decision == SchedulerDecision.STOP:

            self.optimizer.on_trial_complete(
                trial=result.query.optimizer_info[1],  # type: ignore
                result=results_obj_dict,
            )


class SyneTuneMOASHA(SyneTuneOptimizer):
    """SyneTune MOASHA."""

    name = "SyneTuneMOASHA"
    support = Problem.Support(
        fidelities=("single"),
        objectives=("many",),
        cost_awareness=(None,),
        tabular=False,
    )

    mem_req_mb = 1024

    def __init__(  # noqa: C901, PLR0912
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: Path,
        **kwargs: Any,  # noqa: ARG002
    ):
        """Create a SyneTune MOASHA instance for a given problem.

        Args:
            problem: The problem to optimize.
            seed: The random seed.
            working_directory: The working directory to store the results.
            **kwargs: Additional arguments for the BayesianOptimization.
        """
        from syne_tune.optimizer.baselines import MOASHA


        match problem.costs:
            case None:
                pass
            case tuple() | Mapping():
                clsname = self.__class__.__name__
                raise ValueError(f"{clsname} does not support cost-awareness")
            case _:
                raise TypeError("cost_awareness must be a list, dict or None")


        mode: Literal["min", "max"]
        match problem.objectives:
            case (name, metric):
                metric_name = name
                mode = "min" if metric.minimize else "max"
            case Mapping():
                mode = []
                for _, metric in problem.objectives.items():
                    mode.append("min" if metric.minimize else "max")
            case _:
                raise TypeError("Objective must be a string or a list of strings")

        synetune_cs: dict[str, Domain]

        config_space = problem.config_space
        match config_space:
            case CS.ConfigurationSpace():
                synetune_cs = configspace_to_synetune_configspace(config_space)
            case list():
                raise ValueError("SyneTuneBO does not support tabular benchmarks")
            case _:
                raise TypeError("config_space must be of type ConfigSpace.ConfigurationSpace")

        match problem.fidelities:
            case None:
                raise ValueError(
                    "SyneTuneMOASHA is a multi-fidelity optimizer. "
                    "The Problem must define a fidelity."
                )
            case (fidelity_name, fidelity):
                synetune_cs[fidelity_name] = fidelity.max
            case Mapping():
                raise NotImplementedError("Many-fidelity not yet implemented for SyneTuneMOASHA.")
            case _:
                raise TypeError("Fidelity must be None, a tuple or a Mapping")

        super().__init__(
            problem=problem,
            seed=seed,
            working_directory=working_directory,
            optimizer=MOASHA(
                config_space = synetune_cs,
                metrics = problem.get_objectives(),
                mode = mode,
                time_attr = "fidelity_counter",
                # random_seed = seed,
                max_t = problem.fidelities[1].max,
                # resource_attr = problem.fidelities[0],
                reduction_factor = 3,
            )
        )


def configspace_to_synetune_configspace(  # noqa: C901
    config_space: CS.ConfigurationSpace,
) -> dict[str, Domain | Any]:
    """Convert ConfigSpace to SyneTune config_space."""
    from syne_tune.config_space import (
        choice,
        lograndint,
        loguniform,
        ordinal,
        randint,
        uniform,
    )

    if any(config_space.get_conditions()):
        raise NotImplementedError("ConfigSpace with conditions not supported")

    if any(config_space.get_forbiddens()):
        raise NotImplementedError("ConfigSpace with forbiddens not supported")

    synetune_cs: dict[str, Domain | Any] = {}
    for hp in config_space.get_hyperparameters():
        match hp:
            case CS.OrdinalHyperparameter():
                synetune_cs[hp.name] = ordinal(hp.sequence)
            case CS.CategoricalHyperparameter() if hp.weights is not None:
                raise NotImplementedError("CategoricalHyperparameter with weights not supported")
            case CS.CategoricalHyperparameter():
                synetune_cs[hp.name] = choice(hp.choices)
            case CS.UniformIntegerHyperparameter() if hp.log:
                synetune_cs[hp.name] = lograndint(hp.lower, hp.upper)
            case CS.UniformIntegerHyperparameter():
                synetune_cs[hp.name] = randint(hp.lower, hp.upper)
            case CS.UniformFloatHyperparameter() if hp.log:
                synetune_cs[hp.name] = loguniform(hp.lower, hp.upper)
            case CS.UniformFloatHyperparameter():
                synetune_cs[hp.name] = uniform(hp.lower, hp.upper)
            case CS.Constant():
                synetune_cs[hp.name] = hp.value
            case _:
                raise ValueError(f"Hyperparameter {hp.name} of type {type(hp)} is not supported")

    return synetune_cs
