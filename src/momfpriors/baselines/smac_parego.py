from __future__ import annotations

from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from ConfigSpace import ConfigurationSpace
from hpoglue import Config, Problem, Query
from hpoglue.budget import CostBudget, TrialBudget
from hpoglue.env import Env

from momfpriors.optimizer import Abstract_AskTellOptimizer
from momfpriors.utils import objective_fn_wrapper

if TYPE_CHECKING:
    from hpoglue import Result
    from smac.runhistory import TrialInfo


def _dummy_target_function(*args: Any, budget: int | float, seed: int) -> NoReturn:
    raise RuntimeError("This should never be called!")


class SMAC_ParEGO(Abstract_AskTellOptimizer):

    name = "SMAC_ParEGO"

    support = Problem.Support(
        fidelities=(None,),
        objectives=("single", "many"),
        cost_awareness=(None,),
        tabular=False,
    )

    mem_req_mb = 1024

    env = Env(
        name="SMAC-2.1",
        python_version="3.10",
        requirements=("smac==2.1",)
    )

    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: Path,
        **kwargs: Any
    ):
        """Create a SMAC Optimizer instance for a given problem statement.

        Args:
            problem: Problem statement.
            seed: Random seed for the optimizer.
            working_directory: Working directory to store SMAC run.
            config_space: Configuration space to optimize over.
            optimizer: SMAC optimizer instance.
        """
        config_space = problem.config_space

        working_directory.mkdir(parents=True, exist_ok=True)

        self.problem = problem
        self.working_directory = working_directory
        self.config_space = config_space
        self._trial_lookup: dict[Hashable, TrialInfo] = {}
        self._seed = seed

        from smac import (
            HyperparameterOptimizationFacade as HPOFacade,
            Scenario,
        )
        from smac.multi_objective.parego import ParEGO

        scenario = Scenario(
            configspace=self.config_space,
            deterministic=True,
            objectives=problem.get_objectives(),
            n_trials=problem.budget.total,
            seed=seed,
            output_directory=self.working_directory / "smac-output",
        )
        self.optimizer = HPOFacade(
            scenario=scenario,
            logging_level=False,
            target_function=_dummy_target_function,
            multi_objective_algorithm=ParEGO(scenario=scenario),
            intensifier=HPOFacade.get_intensifier(scenario),
            overwrite=True,
        )

    def ask(self) -> Query:
        """Ask SMAC for a new config to evaluate."""
        smac_info = self.optimizer.ask()
        assert smac_info.instance is None, "We don't do instance benchmarks!"

        config = smac_info.config
        raw_config = dict(config)
        config_id = str(self.optimizer.intensifier.runhistory.config_ids[config])

        return Query(
            config=Config(config_id=config_id, values=raw_config),
            fidelity=None,
            optimizer_info=smac_info,
        )

    def tell(self, result: Result) -> None:
        """Tell SMAC the result of the query."""
        from smac.runhistory import StatusType, TrialValue

        match self.problem.objectives:
            case Mapping():
                cost = [
                    obj.as_minimize(result.values[key])
                    for key, obj in self.problem.objectives.items()
                ]
            case _:
                raise TypeError("Objective must be a Mapping for multi-objective problems")

        self.optimizer.tell(
            result.query.optimizer_info,  # type: ignore
            TrialValue(
                cost=cost,
                time=0.0,
                starttime=0.0,
                endtime=0.0,
                status=StatusType.SUCCESS,
                additional_info={},
            ),
            save=True,
        )
