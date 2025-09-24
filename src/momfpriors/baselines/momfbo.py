"""Implementation of the algorithm from the paper:
Leveraging Trust for Joint Multi-Objective and Multi-Fidelity Optimization (
https://arxiv.org/pdf/2112.13901).
GitHub: https://github.com/PULSE-ML/MOMFBO-Algorithm .
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import neps
import neps.space
import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.multi_objective import MOMF
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.transforms import (
    normalize,
    unnormalize,
)
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from hpoglue import Config, Problem, Query, Result
from hpoglue.env import Env
from neps.space.parsing import convert_configspace
from torch import Tensor

from momfpriors.constants import DEFAULT_RESULTS_DIR
from momfpriors.optimizer import Abstract_AskTellOptimizer

if TYPE_CHECKING:
    from botorch.acquisition import AcquisitionFunction
    from botorch.models.model import Model
    from botorch.sampling import MCSampler

MC_SAMPLES = 128  # Number of Monte Carlo samples


def set_seed(seed: int) -> None:
    """Set the seed for the optimizer."""
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002


@dataclass
class Trial:
    """Represents a trial in an optimization process, containing information about
    the trial's input values, fidelity, and objective values.
    """
    trial_id: str

    x: dict[str, Any]  # input values

    fidelity: int | float   # fidelity value

    y: Tensor | None = field(default=None, init=False)  # objective values

    complete: bool = field(default=False, init=False)

    def _set_as_complete(self, y: Tensor) -> None:
        self.y = y
        self.complete = True


@dataclass
class MOMF_BO:
    """MOMF_BO class for MOMFBO algorithm."""

    acquisition_function: AcquisitionFunction = field(init=False)

    space: neps.SearchSpace

    min_fidelity: float

    max_fidelity: float

    config_encoder: neps.space.ConfigEncoder\

    initial_design_size: int

    trials: Mapping[str, Trial] = field(default_factory=dict, init=False)

    dim_x: int = field(init=False)

    tkwargs: dict = field(default_factory=dict, init=False)

    exp_arg_init: float | None = field(default=4.0)

    model: Model = field(init=False)

    mll: ExactMarginalLogLikelihood = field(init=False)

    sampler: MCSampler = field(init=False)

    mc_samples: int | None = field(default=MC_SAMPLES)

    rng: torch.Generator = field(init=False)

    seed: int | None = field(default=0)


    def __post_init__(self) -> None:
        self.dim_x = len(self.space.searchables)
        self.tkwargs = {}
        if torch.cuda.is_available():
            self.tkwargs = {
                "dtype": torch.double,
                "device": torch.cuda.current_device(),
            }
        else:
            self.tkwargs = (
                {  # Tkwargs is a dictionary contaning data about data type and data device
                    "dtype": torch.double,
                    "device": torch.device("cpu"),
                }
            )

        self.rng = torch.Generator(device=self.tkwargs["device"])
        self.rng.manual_seed(self.seed)

        self.exp_arg = torch.tensor(4,**self.tkwargs)

        self.sampler = SobolQMCNormalSampler(  # Initialize Sampler
            sample_shape=torch.Size([self.mc_samples])
        )
        self.acquisition_function = MOMF


    @classmethod
    def _bo(
        cls,
        space: neps.SearchSpace,
        min_fidelity: float,
        max_fidelity: float,
        exp_arg_init: float = 4.0,
        mc_samples: int = MC_SAMPLES,
        initial_design_size: int | None = None,
    ) -> MOMF_BO:
        if initial_design_size is None:
            initial_design_size = len(space.searchables)
        parameters = space.searchables
        config_encoder = neps.space.ConfigEncoder.from_parameters(parameters)
        return cls(
            space=space,
            config_encoder=config_encoder,
            min_fidelity=min_fidelity,
            max_fidelity=max_fidelity,
            exp_arg_init=exp_arg_init,
            mc_samples=mc_samples,
            initial_design_size=initial_design_size,
        )


    @staticmethod
    def _get_reference_point(loss_vals: np.ndarray) -> np.ndarray:
        """Get the reference point from the completed Trials.
        Source: https://github.com/optuna/optuna/blob/master/optuna/samplers/_tpe/sampler.py#L609 .
        """
        worst_point = np.max(loss_vals, axis=0)
        reference_point = np.maximum(1.1 * worst_point, 0.9 * worst_point)
        reference_point[reference_point == 0] = 1e-12
        return reference_point


    @staticmethod
    def cost_func(x, A):   # noqa: N803
        """A simple exponential cost function."""
        return torch.exp(A * x)


    @classmethod
    def cost_callable(cls, X, exp_arg) -> Tensor:  # noqa: D417, N803
        r"""Wrapper for the cost function that takes care of shaping input
        and output arrays for interfacing with cost_func.
        This is passed as a callable function to MOMF.

        Args:
            X: A `batch_shape x q x d`-dim Tensor
        Returns:
            Cost `batch_shape x q x m`-dim Tensor of cost generated
            from fidelity dimension using cost_func.
        """
        cost = cls.cost_func(torch.flatten(X), exp_arg).reshape(X.shape)
        return cost[..., -1].unsqueeze(-1)


    def _add_constants(self, config: dict[str, Any]) -> dict[str, Any]:
        """Adds constant values from the search space to the config."""
        for hp, value in self.space.constants.items():
            config[hp] = value
        return config


    def gen_init_data(self, num_candidates = 1) -> tuple[Tensor, Tensor]:
        """Generates training data with Fidelity dimension sampled from
        a probability distribution that depends on Cost function.
        """
        init_x = torch.rand( # Randomly generating initial points
            size=(num_candidates, self.dim_x),
            generator=self.rng,
            **self.tkwargs
        )
        init_x = unnormalize(init_x, bounds=self.bounds[..., :-1])

        fid_samples = torch.linspace( # Array from which fidelity values are sampled
            self.min_fidelity,
            self.max_fidelity,
            101,
            **self.tkwargs
        )
        prob = 1 / self.cost_func(
            fid_samples, self.exp_arg)  # Probability calculated from the Cost function
        prob = prob / torch.sum(prob)  # Normalizing
        idx = prob.multinomial(
            num_samples=1,
            replacement=True)  # Generating indices to choose fidelity samples

        return init_x, fid_samples[idx]


    def initialize_gp_model(self, train_x, train_obj):
        """Initializes a SingleTaskGP with Matern 5/2 Kernel and returns the model and its MLL."""
        ref_point = self._get_reference_point(
                np.vstack(
                    [trial.y.cpu().numpy() for trial in self.trials.values() if trial.complete],
                    dtype=np.float64
                )
            )
        ref_point = np.append(ref_point, self.max_fidelity)
        self.model = SingleTaskGP(train_x,
                            train_obj,
                            outcome_transform=Standardize(m=train_obj.shape[-1]))
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        partitioning = FastNondominatedPartitioning(
            ref_point=torch.tensor(
                ref_point,**self.tkwargs), Y=train_obj
        )

        self.acquisition_function = MOMF(
            model=self.model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=self.sampler,
            cost_call=partial(self.cost_callable, exp_arg=self.exp_arg),
        )


    def _encode_trials_for_gp(
        self,
        trials: list[Trial],
    ) -> Tensor:
        """Encodes the trials in the format required for the GP model."""
        configs: list[dict[str, Any]] = []
        for trial in trials:
            config:Tensor = self.config_encoder.encode([trial.x]).squeeze(0)
            config = config.to(**self.tkwargs)
            # print(f"{torch.tensor([trial.fidelity])=}")
            normalized_fidelity = normalize(
                torch.tensor([trial.fidelity], **self.tkwargs),
                bounds=torch.tensor(
                    [[self.min_fidelity], [self.max_fidelity]], **self.tkwargs
                ),
            )
            config = torch.cat((config, normalized_fidelity), dim=-1)
            # print(f"{config=}")
            configs.append(config)
        return torch.stack(configs)


    @property
    def bounds(self) -> Tensor:
        """Returns the bounds of the search space."""
        _lower_bounds = []
        _upper_bounds = []
        for hp_name, param in self.space.searchables.items():
            match param:
                case neps.Float() | neps.Integer():
                    _lower_bounds.append(
                        self.config_encoder.transformers[hp_name].domain.lower
                    )
                    _upper_bounds.append(
                        self.config_encoder.transformers[hp_name].domain.upper
                    )
                case neps.Categorical():
                    _lower_bounds.append(
                        self.config_encoder.transformers[hp_name].domain.lower
                    )
                    _upper_bounds.append(
                        self.config_encoder.transformers[hp_name].domain.upper
                    )
                case neps.Constant():
                    raise ValueError("Constants should not be part of the Neps Search Space.")
                case _:
                    raise ValueError(f"Unknown parameter type: {type(param)}")
        _lower_bounds.append(0.0)   # Lower bound for fidelity
        _upper_bounds.append(1.0)   # Upper bound for fidelity
        _bounds = [_lower_bounds, _upper_bounds]
        return torch.tensor(_bounds, **self.tkwargs)


    def ask(
        self,
        num_candidates: int | None = None,
    ) -> Trial:
        """Wrapper to call MOMF and optimizes it in a sequential greedy fashion
        returning a new candidate and evaluation.
        """
        # Get initial candidates
        n_evaulated = sum(
            1
            for trial in self.trials.values()
            if trial.complete
        )
        if n_evaulated < self.initial_design_size:
            init_x, fid = self.gen_init_data()
            init_config = self.config_encoder.decode_one(init_x)
            init_config = self._add_constants(init_config)
            trial_id = len(self.trials) + 1
            new_trial = Trial(
                trial_id=str(trial_id),
                x=init_config,
                fidelity=type(self.min_fidelity)(fid.item()),
            )
            self.trials[trial_id] = new_trial
            return new_trial

        # Get the training data from the completed trials by
        # stacking the input values and fidelity values
        train_x = self._encode_trials_for_gp(
            [trial for trial in self.trials.values() if trial.complete]
        )


        # Stack objective values and fidelity values from the completed trials
        train_obj = torch.stack(
            [
                torch.cat((trial.y, torch.tensor([trial.fidelity], **self.tkwargs)))
                for trial in self.trials.values() if trial.complete
            ]
        )

        # Initialize the GP model
        self.initialize_gp_model(train_x, train_obj)

        # Fit the model
        fit_gpytorch_mll(self.mll)

        # Optimization
        candidates, _ = optimize_acqf(
            acq_function=self.acquisition_function,
            bounds=self.bounds,
            q=num_candidates or 1,
            num_restarts=20,
            raw_samples=1024,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
            sequential=True,
        )

        # observe new values
        # print(f"{candidates=}")
        fid = unnormalize(
            candidates.detach()[..., -1],
            bounds=torch.tensor([[self.min_fidelity], [self.max_fidelity]], **self.tkwargs)
        )
        new_x = candidates.detach()[..., :-1]
        new_x = unnormalize(new_x, bounds=self.bounds[..., :-1]).squeeze(0)
        # print(f"{new_x=}")
        new_config = self.config_encoder.decode_one(new_x)
        new_config = self._add_constants(new_config)
        trial_id = len(self.trials) + 1
        new_trial = Trial(
            trial_id=str(trial_id),
            x=new_config,
            fidelity=type(self.min_fidelity)(fid.item()),
        )
        self.trials[trial_id] = new_trial
        return new_trial


    def tell(
        self,
        trial: Trial,
        y: list[float],
    ) -> None:
        """Update the trial with the new evaluation."""
        y_tensor = torch.tensor(y, **self.tkwargs)
        trial._set_as_complete(y_tensor)


class MOMFBO_Optimizer(Abstract_AskTellOptimizer):
    """Base class for the MOMFBO Optimizer."""

    name = "MOMFBO"

    support = Problem.Support(
        fidelities=("single",),
        objectives=("many"),
        cost_awareness=(None,),
        tabular=False,
        continuations=True,  # TODO: Check for correctness
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
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the optimizer."""
        self.problem = problem
        self.space = convert_configspace(problem.config_space)

        if not isinstance(working_directory, Path):
            working_directory = Path(working_directory)
        self.seed = seed
        self.working_dir = working_directory

        match problem.fidelities:
            case None:
                raise ValueError("MOMFBO only supports multi-fidelity optimization.")
            case (fid_name, fidelity):
                _fid = fidelity
                min_fidelity = fidelity.min
                max_fidelity = fidelity.max
                match _fid.kind:
                    case _ if _fid.kind is int:
                        self.space.fidelities = {
                            f"{fid_name}": neps.Integer(
                                lower=min_fidelity, upper=max_fidelity, is_fidelity=True
                            )
                        }
                    case _ if _fid.kind is float:
                        self.space.fidelities = {
                            f"{fid_name}": neps.Float(
                                lower=min_fidelity, upper=max_fidelity, is_fidelity=True
                            )
                        }
                    case _:
                        raise TypeError(
                            f"Invalid fidelity type: {type(_fid.kind).__name__}. "
                            "Expected int or float."
                        )
            case Mapping():
                raise NotImplementedError("Manyfidelity not yet implemented for MOMFBO.")
            case _:
                raise TypeError(
                    "Fidelity must be a tuple or a Mapping. \n"
                    f"Got {type(self.problem.fidelities)}."
                )

        self.optimizer: MOMF_BO = MOMF_BO._bo(
            space=self.space,
            min_fidelity=min_fidelity,
            max_fidelity=max_fidelity,
            initial_design_size=5,
        )
        set_seed(self.seed)

        self.trial_counter = 0


    def ask(self) -> Query:
        """Ask the optimizer for a new trial."""
        trial: Trial = self.optimizer.ask()
        assert isinstance(self.problem.fidelities, tuple)
        fid_name, _ = self.problem.fidelities

        _fid_value = trial.fidelity
        fidelity = (fid_name, _fid_value)
        self.trial_counter += 1
        return Query(
            config = Config(
                config_id=trial.trial_id,
                values=trial.x,
            ),
            fidelity=fidelity,
            optimizer_info=trial
        )


    def tell(self, result: Result) -> None:
        """Tell the optimizer about the result of a trial."""
        costs = {
            key: obj.as_minimize(result.values[key])
            for key, obj in self.problem.objectives.items()
        }
        trial = result.query.optimizer_info
        self.optimizer.tell(trial, list(costs.values()))