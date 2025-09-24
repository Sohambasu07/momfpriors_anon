from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import neps
import numpy as np
from hpoglue import Config, Problem, Query, Result
from hpoglue.env import Env
from neps import AskAndTell, algorithms
from neps.space.parsing import convert_configspace

from momfpriors.constants import DEFAULT_RESULTS_DIR
from momfpriors.optimizer import Abstract_AskTellOptimizer
from momfpriors.utils import set_seed

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


    def ask(self) -> Query:
        """Ask the optimizer for a new trial."""
        import copy
        trial = self.optimizer.ask() # TODO: Figure out fidelity
        fidelity = None
        _config = copy.deepcopy(trial.config)
        match self.problem.fidelities:
            case None:
                pass
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
        self.trial_counter += 1
        return Query(
            config = Config(config_id=self.trial_counter, values=_config),
            fidelity=fidelity,
            optimizer_info=trial
        )

    def tell(self, result: Result) -> None:
        """Tell the optimizer about the result of a trial."""
        costs = {
            key: obj.as_minimize(result.values[key])
            for key, obj in self.problem.objectives.items()
        }
        if self.random_weighted_opt:
            if not self.constant_weights:
                weights = self._rng.uniform(size=len(self.objectives))
                self.scalarization_weights = dict(zip(self.objectives, weights, strict=True))

            costs = sum(
                self.scalarization_weights[obj] * costs[obj] for obj in self.objectives
            )
        else:
            costs = list(costs.values())

        self.optimizer.tell(
            trial=result.query.optimizer_info,
            result=costs
        )


class NepsRW(NepsOptimizer):
    """Random Weighted Scalarization of objectives using Bayesian Optimization in Neps."""

    name = "NepsRW"

    support = Problem.Support(
        fidelities=(None,),
        objectives=("many"),
        cost_awareness=(None,),
        tabular=False,
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
        scalarization_weights: Literal["equal", "random"] | Mapping[str, float] = "random",
        searcher: str = "bayesian_optimization",
        **kwargs: Any,
    ) -> None:
        """Initialize the optimizer."""
        self.searcher = searcher
        space = convert_configspace(problem.config_space)
        optimizer = searcher


        match problem.fidelities:
            case None:
                pass
            case Mapping() | tuple():
                raise ValueError("NepsRW does not support fidelities.")
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")

        set_seed(seed)

        super().__init__(
            problem=problem,
            space=space,
            optimizer=optimizer,
            seed=seed,
            working_directory=working_directory,
            random_weighted_opt=True,
            constant_weights=True,
            scalarization_weights=scalarization_weights,
            initial_design_size=kwargs.get("initial_design_size", "ndim"),
        )


class NepsHyperbandRW(NepsOptimizer):
    """Random Weighted Scalarization of objectives using Hyperband for budget allocation in Neps."""

    name = "NepsHyperbandRW"

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
        scalarization_weights: Literal["equal", "random"] | Mapping[str, float] = "random",
        eta: int = 3,
    ) -> None:
        """Initialize the optimizer."""
        space = convert_configspace(problem.config_space)

        _fid = None
        match problem.fidelities:
            case None:
                raise ValueError("NepsHyperbandRW requires a fidelity.")
            case Mapping():
                raise NotImplementedError("Many-fidelity not yet implemented for NepsHyperbandRW.")
            case (fid_name, fidelity):
                _fid = (fid_name, fidelity)
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")

        set_seed(seed)

        super().__init__(
            problem=problem,
            space=space,
            optimizer="hyperband",
            seed=seed,
            working_directory=working_directory,
            fidelities=_fid,
            random_weighted_opt=True,
            constant_weights=True,
            scalarization_weights=scalarization_weights,
            eta=eta,
        )


class NepsMOASHA(NepsOptimizer):
    """NepsMOASHA."""

    name = "NepsMOASHA"

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

        _fid = None
        match problem.fidelities:
            case None:
                raise ValueError("NepsMOASHA requires a fidelity.")
            case Mapping():
                raise NotImplementedError("Many-fidelity not yet implemented for NepsMOASHA.")
            case (fid_name, fidelity):
                _fid = (fid_name, fidelity)
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")
        set_seed(seed)

        super().__init__(
            problem=problem,
            space=space,
            optimizer="moasha",
            seed=seed,
            working_directory=working_directory,
            fidelities=_fid,
            mo_selector=mo_selector,
        )


class NepsMOHyperband(NepsOptimizer):
    """NepsMOHyperband."""

    name = "NepsMOHyperband"

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
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the optimizer."""
        space = convert_configspace(problem.config_space)

        _fid = None
        match problem.fidelities:
            case None:
                raise ValueError("NepsMOHyperband requires a fidelity.")
            case Mapping():
                raise NotImplementedError("Many-fidelity not yet implemented for NepsMOHyperband.")
            case (fid_name, fidelity):
                _fid = (fid_name, fidelity)
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")
        set_seed(seed)

        super().__init__(
            problem=problem,
            space=space,
            optimizer="mo_hyperband",
            seed=seed,
            working_directory=working_directory,
            fidelities=_fid,
            mo_selector=mo_selector,
        )


class NepsMOPriorband(NepsOptimizer):
    """NepsMOPriorband."""

    name = "NepsMOPriorband"

    support = Problem.Support(
        fidelities=("single",),
        objectives=("many"),
        cost_awareness=(None,),
        tabular=False,
        priors=True,
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
        eta: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initialize the optimizer."""
        space = convert_configspace(problem.config_space)

        _fid = None
        match problem.fidelities:
            case None:
                raise ValueError("NepsMOPriorband requires a fidelity.")
            case Mapping():
                raise NotImplementedError("Many-fidelity not yet implemented for NepsMOPriorband.")
            case (fid_name, fidelity):
                _fid = (fid_name, fidelity)
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")
        set_seed(seed)

        prior_centers = {
            obj: prior.values
            for obj, prior in problem.priors[1].items()
        }

        prior_confidences = {
            obj: dict.fromkeys(
                prior.keys(),
                0.75
            )
            for obj, prior in problem.priors[1].items()
        }

        super().__init__(
            problem=problem,
            space=space,
            optimizer="mopriorband",
            seed=seed,
            working_directory=working_directory,
            fidelities=_fid,
            mo_selector=mo_selector,
            prior_centers=prior_centers,
            prior_confidences=prior_confidences,
            incumbent_type=kwargs.get("incumbent_type", "scalarized"),
            base="hyperband",
            eta=eta,
        )


class NepsPiBORW(NepsOptimizer):
    """Random Weighted Scalarization of objectives using PiBO in Neps
    with random choice of priors.
    """

    name = "NepsPiBORW"

    support = Problem.Support(
        fidelities=(None,),
        objectives=("many"),
        cost_awareness=(None,),
        tabular=False,
        priors=True,
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
        scalarization_weights: Literal["equal", "random"] | Mapping[str, float] = "random",
        **kwargs: Any,
    ) -> None:
        """Initialize the optimizer."""
        space = convert_configspace(problem.config_space)

        match problem.fidelities:
            case None:
                pass
            case Mapping() | tuple():
                raise ValueError("NepsPiBORW does not support fidelities.")
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")

        set_seed(seed)

        prior_centers = {
            obj: prior.values
            for obj, prior in problem.priors[1].items()
        }

        prior_confidences = {
            obj: dict.fromkeys(
                prior.keys(),
                0.75
            )
            for obj, prior in problem.priors[1].items()
        }

        super().__init__(
            problem=problem,
            space=space,
            optimizer="pibo",
            seed=seed,
            working_directory=working_directory,
            random_weighted_opt=True,
            constant_weights=False,
            initial_design_size=kwargs.get("initial_design_size", "ndim"),
            scalarization_weights=scalarization_weights,
            mo_prior_centers=prior_centers,
            mo_prior_confidences=prior_confidences,
        )

# Ablation: PriMO without MO-priors
class NepsMOASHABO(NepsOptimizer):
    """NePS PriMO without MO-priors, only vanilla BO with MOMF initial design."""

    name = "NepsMOASHABO"

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
        eta: int = 3,
    ) -> None:
        """Initialize the optimizer."""
        space = convert_configspace(problem.config_space)

        _fid = None
        match problem.fidelities:
            case None:
                raise ValueError("NepsMOASHABO requires a fidelity.")
            case Mapping():
                raise NotImplementedError("Many-fidelity not yet implemented for NepsMOASHABO.")
            case (fid_name, fidelity):
                _fid = (fid_name, fidelity)
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")
        set_seed(seed)

        super().__init__(
            problem=problem,
            space=space,
            optimizer="primo",
            sampler="uniform",
            seed=seed,
            working_directory=working_directory,
            fidelities=_fid,
            mo_selector=mo_selector,
            eta=eta,
            initial_design_size=5,
            epsilon=1.0,
            bo_type="vanilla",
        )


class NepsPriMO(NepsOptimizer):
    """The PriMO optimizer from NePS."""

    name = "NepsPriMO"

    support = Problem.Support(
        fidelities=("single",),
        objectives=("many"),
        cost_awareness=(None,),
        tabular=False,
        priors=True,
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
        eta: int = 3,
        sampler: Literal["uniform", "mopriorband"] = "uniform",
        initial_design_size: int = 5,
        epsilon=0.25,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the optimizer."""
        space = convert_configspace(problem.config_space)

        _fid = None
        match problem.fidelities:
            case None:
                raise ValueError("NepsPriMO requires a fidelity.")
            case Mapping():
                raise NotImplementedError("Many-fidelity not yet implemented for NepsPriMO.")
            case (fid_name, fidelity):
                _fid = (fid_name, fidelity)
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")
        set_seed(seed)

        prior_centers = {
            obj: prior.values
            for obj, prior in problem.priors[1].items()
        }

        prior_confidences = {
            obj: dict.fromkeys(
                prior.keys(),
                0.75
            )
            for obj, prior in problem.priors[1].items()
        }

        super().__init__(
            problem=problem,
            space=space,
            optimizer="primo",
            seed=seed,
            working_directory=working_directory,
            fidelities=_fid,
            mo_selector=mo_selector,
            eta=eta,
            sampler=sampler,
            initial_design_size= initial_design_size,
            prior_centers=prior_centers,
            prior_confidences=prior_confidences,
            epsilon=epsilon,
        )

# Ablation: PriMO with initial design and PiBO's BO
class NepsInitPiBORW(NepsOptimizer):
    """PriMO ablation with initial design and PiBO's BO."""

    name = "NepsInitPiBORW"

    support = Problem.Support(
        fidelities=("single",),
        objectives=("many"),
        cost_awareness=(None,),
        tabular=False,
        priors=True,
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
        eta: int = 3,
        sampler: Literal["uniform", "mopriorband"] = "uniform",
        initial_design_size: int = 5,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the optimizer."""
        space = convert_configspace(problem.config_space)

        _fid = None
        match problem.fidelities:
            case None:
                raise ValueError("NepsInitPiBORW requires a fidelity.")
            case Mapping():
                raise NotImplementedError("Many-fidelity not yet implemented for NepsInitPiBORW.")
            case (fid_name, fidelity):
                _fid = (fid_name, fidelity)
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")
        set_seed(seed)

        prior_centers = {
            obj: prior.values
            for obj, prior in problem.priors[1].items()
        }

        prior_confidences = {
            obj: dict.fromkeys(
                prior.keys(),
                0.75
            )
            for obj, prior in problem.priors[1].items()
        }

        super().__init__(
            problem=problem,
            space=space,
            optimizer="primo",
            seed=seed,
            working_directory=working_directory,
            fidelities=_fid,
            mo_selector=mo_selector,
            eta=eta,
            sampler=sampler,
            initial_design_size= initial_design_size,
            prior_centers=prior_centers,
            prior_confidences=prior_confidences,
            epsilon=0.0,
            bo_type="pibo",
        )


class NepsNoInitPriMO(NepsOptimizer):
    """The PriMO optimizer from NePS without the init design."""

    name = "NepsNoInitPriMO"

    support = Problem.Support(
        fidelities=("single",),
        objectives=("many"),
        cost_awareness=(None,),
        tabular=False,
        priors=True,
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
        eta: int = 3,
        sampler: Literal["uniform", "mopriorband"] = "uniform",
        initial_design_size: int = 5,
        epsilon=0.25,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the optimizer."""
        space = convert_configspace(problem.config_space)

        _fid = None
        match problem.fidelities:
            case None:
                raise ValueError("NepsNoInitPriMO requires a fidelity.")
            case Mapping():
                raise NotImplementedError("Many-fidelity not yet implemented for NepsNoInitPriMO.")
            case (fid_name, fidelity):
                _fid = (fid_name, fidelity)
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")
        set_seed(seed)

        prior_centers = {
            obj: prior.values
            for obj, prior in problem.priors[1].items()
        }

        prior_confidences = {
            obj: dict.fromkeys(
                prior.keys(),
                0.75
            )
            for obj, prior in problem.priors[1].items()
        }

        super().__init__(
            problem=problem,
            space=space,
            optimizer="primo",
            seed=seed,
            working_directory=working_directory,
            fidelities=_fid,
            mo_selector=mo_selector,
            eta=eta,
            sampler=sampler,
            initial_design_size= initial_design_size,
            prior_centers=prior_centers,
            prior_confidences=prior_confidences,
            epsilon=epsilon,
            init_design_type="random",
        )


# Naive Priors Optimizers

class NepsPriorMOASHA(NepsOptimizer):
    """NepsPriorMOASHA."""

    name = "NepsPriorMOASHA"

    support = Problem.Support(
        fidelities=("single",),
        objectives=("many"),
        cost_awareness=(None,),
        tabular=False,
        priors=True,
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
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the optimizer."""
        space = convert_configspace(problem.config_space)

        _fid = None
        match problem.fidelities:
            case None:
                raise ValueError("NepsPriorMOASHA requires a fidelity.")
            case Mapping():
                raise NotImplementedError("Many-fidelity not yet implemented for NepsPriorMOASHA.")
            case (fid_name, fidelity):
                _fid = (fid_name, fidelity)
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")
        set_seed(seed)

        prior_centers = {
            obj: prior.values
            for obj, prior in problem.priors[1].items()
        }

        prior_confidences = {
            obj: dict.fromkeys(
                prior.keys(),
                0.75
            )
            for obj, prior in problem.priors[1].items()
        }

        super().__init__(
            problem=problem,
            space=space,
            optimizer="priormoasha",
            seed=seed,
            working_directory=working_directory,
            fidelities=_fid,
            mo_selector=mo_selector,
            prior_centers=prior_centers,
            prior_confidences=prior_confidences,
        )


class NepsPriorRSMOASHA(NepsOptimizer):
    """NepsPriorRSMOASHA."""

    name = "NepsPriorRSMOASHA"

    support = Problem.Support(
        fidelities=("single",),
        objectives=("many"),
        cost_awareness=(None,),
        tabular=False,
        priors=True,
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
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the optimizer."""
        space = convert_configspace(problem.config_space)

        _fid = None
        match problem.fidelities:
            case None:
                raise ValueError("NepsPriorRSMOASHA requires a fidelity.")
            case Mapping():
                raise NotImplementedError(
                    "Many-fidelity not yet implemented for NepsPriorRSMOASHA."
                )
            case (fid_name, fidelity):
                _fid = (fid_name, fidelity)
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")
        set_seed(seed)

        prior_centers = {
            obj: prior.values
            for obj, prior in problem.priors[1].items()
        }

        prior_confidences = {
            obj: dict.fromkeys(
                prior.keys(),
                0.75
            )
            for obj, prior in problem.priors[1].items()
        }

        super().__init__(
            problem=problem,
            space=space,
            optimizer="priormoasha",
            seed=seed,
            working_directory=working_directory,
            fidelities=_fid,
            mo_selector=mo_selector,
            prior_centers=prior_centers,
            prior_confidences=prior_confidences,
            mix_random=True,
        )


# Ablation: PriMO with 1/eta prior sampling in the initial design.
class NepsEtaPriorPriMO(NepsOptimizer):
    """PriMO with 1/eta prior sampling in the initial design."""

    name = "NepsEtaPriorPriMO"

    support = Problem.Support(
        fidelities=("single",),
        objectives=("many"),
        cost_awareness=(None,),
        tabular=False,
        priors=True,
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
        eta: int = 3,
        initial_design_size: int = 5,
        epsilon=0.25,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the optimizer."""
        space = convert_configspace(problem.config_space)

        _fid = None
        match problem.fidelities:
            case None:
                raise ValueError("NepsEtaPriorPriMO requires a fidelity.")
            case Mapping():
                raise NotImplementedError(
                    "Many-fidelity not yet implemented for NepsEtaPriorPriMO."
                )
            case (fid_name, fidelity):
                _fid = (fid_name, fidelity)
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")
        set_seed(seed)

        prior_centers = {
            obj: prior.values
            for obj, prior in problem.priors[1].items()
        }

        prior_confidences = {
            obj: dict.fromkeys(
                prior.keys(),
                0.75
            )
            for obj, prior in problem.priors[1].items()
        }

        super().__init__(
            problem=problem,
            space=space,
            optimizer="primo",
            seed=seed,
            working_directory=working_directory,
            fidelities=_fid,
            mo_selector=mo_selector,
            eta=eta,
            sampler="mopriorsampler",
            mopriors_sampler_type="etaprior",
            initial_design_size= initial_design_size,
            prior_centers=prior_centers,
            prior_confidences=prior_confidences,
            epsilon=epsilon,
        )


# Ablation: MO-ASHA with initally random and then PriMO's BO for sampling.
class NepsMFPriMO(NepsOptimizer):
    """Multi-fidelity optimizer using PriMO's BO sampler."""

    name = "NepsMFPriMO"

    support = Problem.Support(
        fidelities=("single",),
        objectives=("many"),
        cost_awareness=(None,),
        tabular=False,
        priors=True,
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

        _fid = None
        match problem.fidelities:
            case None:
                raise ValueError("NepsMFPriMO requires a fidelity.")
            case Mapping():
                raise NotImplementedError("Many-fidelity not yet implemented for NepsMFPriMO.")
            case (fid_name, fidelity):
                _fid = (fid_name, fidelity)
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")
        set_seed(seed)


        prior_centers = {
            obj: prior.values
            for obj, prior in problem.priors[1].items()
        }

        prior_confidences = {
            obj: dict.fromkeys(
                prior.keys(),
                0.75
            )
            for obj, prior in problem.priors[1].items()
        }

        super().__init__(
            problem=problem,
            space=space,
            optimizer="moasha",
            seed=seed,
            working_directory=working_directory,
            fidelities=_fid,
            mo_selector=mo_selector,
            primo_initial_design_size=5,
            primo_prior_centers=prior_centers,
            primo_prior_confidences=prior_confidences,
        )


# Ablation: MO-ASHA with initally EtaPrior sampling and then PriMO's BO for sampling.
class NepsEtaPriorMFPriMO(NepsOptimizer):
    """Multi-fidelity optimizer using PriMO's BO sampler."""

    name = "NepsEtaPriorMFPriMO"

    support = Problem.Support(
        fidelities=("single",),
        objectives=("many"),
        cost_awareness=(None,),
        tabular=False,
        priors=True,
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

        _fid = None
        match problem.fidelities:
            case None:
                raise ValueError("NepsEtaPriorMFPriMO requires a fidelity.")
            case Mapping():
                raise NotImplementedError(
                    "Many-fidelity not yet implemented for NepsEtaPriorMFPriMO."
                )
            case (fid_name, fidelity):
                _fid = (fid_name, fidelity)
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")
        set_seed(seed)

        prior_centers = {
            obj: prior.values
            for obj, prior in problem.priors[1].items()
        }

        prior_confidences = {
            obj: dict.fromkeys(
                prior.keys(),
                0.75
            )
            for obj, prior in problem.priors[1].items()
        }

        super().__init__(
            problem=problem,
            space=space,
            optimizer="priormoasha",
            seed=seed,
            working_directory=working_directory,
            fidelities=_fid,
            prior_centers=prior_centers,
            prior_confidences=prior_confidences,
            mo_selector=mo_selector,
            sampler_type="etaprior",
            primo_initial_design_size=5,
        )


# Ablation: PriMO with initial design size of 10
class NepsPriMO_Init10(NepsOptimizer):
    """PriMO with initial design size of 10."""

    name = "NepsPriMO_Init10"

    support = Problem.Support(
        fidelities=("single",),
        objectives=("many"),
        cost_awareness=(None,),
        tabular=False,
        priors=True,
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
        eta: int = 3,
        sampler: Literal["uniform", "mopriorband"] = "uniform",
        epsilon=0.25,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the optimizer."""
        space = convert_configspace(problem.config_space)

        _fid = None
        match problem.fidelities:
            case None:
                raise ValueError("NepsPriMO_Init10 requires a fidelity.")
            case Mapping():
                raise NotImplementedError("Many-fidelity not yet implemented for NepsPriMO_Init10.")
            case (fid_name, fidelity):
                _fid = (fid_name, fidelity)
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")
        set_seed(seed)

        prior_centers = {
            obj: prior.values
            for obj, prior in problem.priors[1].items()
        }

        prior_confidences = {
            obj: dict.fromkeys(
                prior.keys(),
                0.75
            )
            for obj, prior in problem.priors[1].items()
        }

        super().__init__(
            problem=problem,
            space=space,
            optimizer="primo",
            seed=seed,
            working_directory=working_directory,
            fidelities=_fid,
            mo_selector=mo_selector,
            eta=eta,
            sampler=sampler,
            initial_design_size= 10,
            prior_centers=prior_centers,
            prior_confidences=prior_confidences,
            epsilon=epsilon,
        )