from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import neps
import numpy as np
from hpoglue import Config, Problem, Query, Result
from hpoglue.env import Env
from neps import AskAndTell, algorithms
from neps.space.parsing import convert_configspace

from momfpriors.constants import DEFAULT_RESULTS_DIR
from momfpriors.optimizer import Abstract_AskTellOptimizer

if TYPE_CHECKING:
    from hpoglue.fidelity import Fidelity

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NepsOptimizer(Abstract_AskTellOptimizer):
    """Base class for Neps Optimizers."""
    name = "NepsOptimizer"

    mem_req_mb = 1024

    def __init__(
        self,
        *,
        problem: Problem,
        space: neps.SearchSpace,
        optimizer: str,
        seed: int = 0,
        working_directory: str | Path = DEFAULT_RESULTS_DIR,
        fidelities: tuple[str, Fidelity] | None = None,
        random_weighted_opt: bool = False,
        constant_weights: bool = True,
        scalarization_weights: Literal["equal", "random"] | Mapping[str, float] = "random",
        **kwargs: Any,
    ) -> None:
        """Initialize the optimizer."""
        self.problem = problem
        self.space = space

        match fidelities:
            case None:
                pass
            case (fid_name, fidelity):
                _fid = fidelity
                min_fidelity = fidelity.min
                max_fidelity = fidelity.max
                match _fid.kind:
                    case _ if _fid.kind is int:
                        space.fidelities = {
                            f"{fid_name}": neps.Integer(
                                lower=min_fidelity, upper=max_fidelity, is_fidelity=True
                            )
                        }
                    case _ if _fid.kind is float:
                        space.fidelities = {
                            f"{fid_name}": neps.Float(
                                lower=min_fidelity, upper=max_fidelity, is_fidelity=True
                            )
                        }
                    case _:
                        raise TypeError(
                            f"Invalid fidelity type: {type(_fid.kind).__name__}. "
                            "Expected int or float."
                        )
            case _:
                raise TypeError("Fidelity must be a tuple or None.")


        if not isinstance(working_directory, Path):
            working_directory = Path(working_directory)
        self.seed = seed
        self.working_dir = working_directory

        self.optimizer = AskAndTell(
            algorithms.PredefinedOptimizers[optimizer](
                space = space,
                **kwargs,
            )
        )
        self.trial_counter = 0

        self.objectives = self.problem.get_objectives()

        self._rng = np.random.default_rng(seed=self.seed)

        self.constant_weights = constant_weights
        self.random_weighted_opt = random_weighted_opt
        self.scalarization_weights = None

        if self.constant_weights and self.random_weighted_opt:
            match scalarization_weights:
                case Mapping():
                    self.scalarization_weights = scalarization_weights
                case "equal":
                    self.scalarization_weights = (
                        dict.fromkeys(self.objectives, 1.0 / len(self.objectives))
                    )
                case "random":
                    weights = self._rng.uniform(size=len(self.objectives))
                    self.scalarization_weights = dict(zip(self.objectives, weights, strict=True))
                case _:
                    raise ValueError(
                        f"Invalid scalarization_weights: {scalarization_weights}. "
                        "Expected 'equal', 'random', or a Mapping."
                    )

    @overload
    def ask(self) -> Query:
        ...

    @overload
    def tell(self, result: Result) -> None:
        ...


def set_seed(seed: int) -> None:
    """Set the seed for the optimizer."""
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002


class NepsMOASHA_RS(NepsOptimizer):
    """NepsMOASHA_RS."""

    name = "NepsMOASHA_RS"

    support = Problem.Support(
        fidelities=("single",),
        objectives=("many"),
        cost_awareness=(None,),
        tabular=False,
        continuations=True,
    )

    env = Env(
        name="Neps-0.12.2",
        python_version="3.10",
        requirements=("neural-pipeline-search==0.12.2",)
    )

    mem_req_mb = 1024

    def __init__(
        self,
        problem: Problem,
        seed: int = 0,
        working_directory: str | Path = DEFAULT_RESULTS_DIR,
        mo_selector: Literal["nsga2", "epsnet"] = "epsnet",
        **kwargs: Any # noqa: ARG002
    ) -> None:
        """Initialize the optimizer."""
        space = convert_configspace(problem.config_space)
        self.config_space = problem.config_space
        self.config_space.seed(seed)
        self.initial_design = 10

        _fid = None
        match problem.fidelities:
            case None:
                raise ValueError("NepsMOASHA_RS requires a fidelity.")
            case Mapping():
                raise NotImplementedError("Many-fidelity not yet implemented for NepsMOASHA_RS.")
            case (fid_name, fidelity):
                _fid = (fid_name, fidelity)
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")
        set_seed(seed)

        self.fidelity = _fid

        super().__init__(
            problem=problem,
            space=space,
            optimizer="moasha",
            seed=seed,
            working_directory=working_directory,
            fidelities=_fid,
            mo_selector=mo_selector,
        )


    def ask(self) -> Query:
        """Ask the optimizer for a new trial."""
        import copy
        fidelity = None
        trial = None
        trials = self.optimizer.trials
        if not self.threshold_reached(trials, self.initial_design):
            trial = self.optimizer.ask() # TODO: Figure out fidelity
            _config = copy.deepcopy(trial.config)
            match self.problem.fidelities:
                case None:
                    raise ValueError("NepsMOASHA_RS requires a fidelity.")
                case Mapping():
                    raise NotImplementedError("Many-fidelity not yet implemented for NepsOptimizer.")
                case (fid_name, _):
                    _fid_value = _config.pop(fid_name)
                    fidelity = (fid_name, _fid_value)
                case _:
                    raise TypeError(
                        "Fidelity must be a tuple or a Mapping. \n"
                        f"Got {type(self.problem.fidelities)}."
                    )
        else:
            _config = dict(self.config_space.sample_configuration())
            fidelity = (self.fidelity[0], self.fidelity[1].max)
        self.trial_counter += 1
        return Query(
            config = Config(config_id=self.trial_counter, values=_config),
            fidelity=fidelity,
            optimizer_info=trial
        )

    def threshold_reached(
        self,
        trials: Mapping[str, neps.state.Trial],
        threshold: float,
    ) -> bool:
        """Check if the threshold is reached."""
        used_fidelity = [
            t.config[self.fidelity[0]] for t in trials.values() if t.report is not None
        ]
        fidelity_units_used = sum(used_fidelity) / self.fidelity[-1].max
        return fidelity_units_used >= threshold


    def tell(self, result: Result) -> None:
        """Tell the optimizer about the result of a trial."""
        costs = {
            key: obj.as_minimize(result.values[key])
            for key, obj in self.problem.objectives.items()
        }
        if result.query.optimizer_info is None:
            return

        self.optimizer.tell(
            trial=result.query.optimizer_info,
            result=list(costs.values()),
        )