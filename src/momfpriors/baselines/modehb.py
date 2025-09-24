"""hposuite interface for the DEHB Optimizer."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ConfigSpace import ConfigurationSpace
from dehb import DEHB
from hpoglue import Config, Problem, Query
from hpoglue.env import Env

if TYPE_CHECKING:
    from hpoglue import Result


from momfpriors.optimizer import Abstract_AskTellOptimizer


class MO_DEHB(Abstract_AskTellOptimizer):
    """The MO_DEHB Optimizer."""

    name = "MO_DEHB"

    env = Env(
        name="dehb-0.1.2",
        python_version="3.10",
        requirements=("dehb==0.1.2"),
    )

    support = Problem.Support(
        fidelities=("single"),
        objectives=("many",),
        cost_awareness=(None),
        tabular=False,
        continuations=False,
    )

    mem_req_mb = 1024

    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: Path,
        eta: int = 3,
        verbose: bool = False,
    ):
        """Create a MO-DEHB Optimizer instance for a given problem.

        Args:
            problem: The problem to optimize.

            seed: The random seed to use.

            working_directory: The directory to store MO-DEHB's output.

            eta: Sets the aggressiveness of Hyperband's aggressive
                early stopping by retaining 1/eta configurations every round.
                Defaults to 3.

            verbose: Whether to enable verbose logging
        """
        config_space = problem.config_space
        match config_space:
            case ConfigurationSpace():
                pass
            case list():
                raise NotImplementedError("# TODO: Tabular not yet implemented for MO-DEHB!")
            case _:
                raise TypeError("Config space must be a list or a ConfigurationSpace!")

        match problem.fidelities:
            case None:
                raise ValueError("Fidelity must be defined for MO-DEHB!")
            case (_, fidelity):
                min_fidelity = fidelity.min
                max_fidelity = fidelity.max
            case Mapping():
                raise NotImplementedError("# TODO: Manyfidelity not yet implemented for MO-DEHB!")
            case _:
                raise TypeError("Fidelity must be a string or a list of strings!")

        working_directory.mkdir(parents=True, exist_ok=True)

        self.problem = problem

        # TODO(eddiebergman): Clarify if DEHB is actually cost-aware in
        # terms of how it optimizes or does it just track cost for the
        # sake of `run()`? We only use `ask()` and `tell()` but it seems
        # to require cost in `tell()`.
        self.dehb = DEHB(
            cs=config_space,
            min_budget=min_fidelity,
            max_budget=max_fidelity,
            verbose=verbose,
            seed=seed,
            eta=eta,
            n_workers=1,
            output_path=working_directory,
        )

        # HACK(eddiebergman): DEHB doesn't have an option to disable logging
        if not verbose:
            self.dehb.logger.disable("dehb.optimizers.dehb")

        self._info_lookup: dict[str, dict[str, Any]] = {}


    def ask(self) -> Query:
        """Ask DEHB for a new config to evaluate."""
        info = self.dehb.ask()
        assert isinstance(info, dict)

        match self.problem.fidelities:
            case None:
                fidelity = None
            case (key, fidelity_def):
                _val = info["fidelity"]
                value = int(_val) if fidelity_def.kind is int else _val
                fidelity = (key, value)
            case Mapping():
                raise NotImplementedError("# TODO: many-fidelity not yet implemented for MO-DEHB!")
            case _:
                raise TypeError("Fidelity must be None, a tuple, or a mapping!")

        config_id = info["config_id"]
        raw_config = info["config"]
        name = f"trial_{config_id}"

        return Query(
            config=Config(config_id=name, values=raw_config),
            fidelity=fidelity,
            optimizer_info=info,
        )


    def tell(self, result: Result) -> None:
        """Tell DEHB the result of the query."""
        fitness: float
        match self.problem.objectives:
            case Mapping():
                fitness = {
                    name: metric.as_minimize(result.values[name])
                    for name, metric in self.problem.objectives.items()
                }
                fitness = list(fitness.values())
            case _:
                raise TypeError("Objective must be a list of strings!")

        match self.problem.costs:
            case None:
                cost = None
            case tuple():
                raise NotImplementedError("# TODO: Cost-aware not yet implemented for MO-DEHB!")
            case Mapping():
                raise NotImplementedError("# TODO: Cost-aware not yet implemented for MO-DEHB!")
            case _:
                raise TypeError("Cost must be None or a mapping!")

        assert isinstance(result.query.optimizer_info, dict)
        if cost is None:
            self.dehb.tell(
                result.query.optimizer_info,
                # NOTE(eddiebergman): DEHB requires a cost value, even though
                # we never specify anything related to cost.
                {"fitness": fitness, "cost": result.budget_cost},
            )
        else:
            raise NotImplementedError("# TODO: Cost-aware not yet implemented for MO-DEHB!")
