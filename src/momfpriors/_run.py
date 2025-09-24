from __future__ import annotations

import logging
import shutil
import traceback
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import pandas as pd
import yaml
from hpoglue import FunctionalBenchmark, Problem
from hpoglue._run import _run
from hpoglue.env import (
    GLUE_PYPI,
    Env,
    Venv,
    get_current_installed_hpoglue_version,
)

from momfpriors._core import _core
from momfpriors.baselines import OPTIMIZERS
from momfpriors.benchmarks import BENCHMARKS
from momfpriors.constants import DEFAULT_PRIORS_DIR, DEFAULT_RESULTS_DIR
from momfpriors.optimizer import Abstract_AskTellOptimizer, Abstract_NonAskTellOptimizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


smac_logger = logging.getLogger("smac")
smac_logger.setLevel(logging.ERROR)



if TYPE_CHECKING:
    from hpoglue import BenchmarkDescription, Config, Result

    from momfpriors.optimizer import Abstract_AskTellOptimizer, Abstract_NonAskTellOptimizer


T = TypeVar("T", bound=Hashable)


logger = logging.getLogger(__name__)

GLOBAL_SEED = 42


def _try_delete_if_exists(path: Path) -> None:
    if path.exists():
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

        except Exception as e:
            logger.exception(e)
            logger.error(f"Error deleting {path}: {e}")



@dataclass
class Run:
    """A run of a benchmark."""

    name: str = field(init=False)
    """The name of the run."""

    problem: Problem
    """The problem that was run."""

    seed: int
    """The seed used for the run."""

    exp_dir: Path
    """The experiment directory for all the runs in this experiment."""

    optimizer: type[Abstract_AskTellOptimizer | Abstract_NonAskTellOptimizer] = field(init=False)
    """The optimizer to use for the Problem"""

    rootdir: Path = field(init=False)
    """Default root directory."""

    benchmark: BenchmarkDescription = field(init=False)
    """The benchmark that the Problem was run on."""

    priors: Mapping[str, tuple[str, Mapping[str, Any]]] = field(default_factory=dict)
    """The priors to use for the run.
        Usage: {objective: (prior_annotation, prior_config)}
        Example: {
            "obj1": ("good", Prior config dict)
            }
    """

    prior_distribution: Literal["normal", "uniform", "beta"] = "normal"
    """The distribution in use for the priors."""

    env: Env = field(init=False)
    """The environment to setup the optimizer in for `isolated` mode."""

    working_dir: Path = field(init=False)
    """The working directory for the run."""

    complete_flag: Path = field(init=False)
    """The flag to indicate the run is complete."""

    error_file: Path = field(init=False)
    """The file to store the error traceback if the run crashed."""

    running_flag: Path = field(init=False)
    """The flag to indicate the run is currently running."""

    queue_flag: Path = field(init=False)
    """The flag to indicate the run is queued."""

    df_path: Path = field(init=False)
    """The path to the dataframe for the run."""

    env_path: Path = field(init=False)
    """The path to the environment for the run."""

    mem_req_mb: int = field(init=False)
    """The memory requirement for the run in mb.
    Calculated as the sum of the memory requirements of the optimizer and the benchmark.
    """

    def __post_init__(self) -> None:
        self.benchmark = self.problem.benchmark
        self.optimizer = self.problem.optimizer
        self.optimizer_hyperparameters = self.problem.optimizer_hyperparameters
        self.mem_req_mb = self.problem.mem_req_mb

        name_parts: list[str] = [
            self.problem.name,
            f"seed={self.seed}",
        ]

        self.name = ".".join(name_parts)

        match self.benchmark.env, self.optimizer.env:
            case (None, None):
                self.env = Env.empty()
            case (None, Env()):
                self.env = self.optimizer.env
            case (Env(), None):
                self.env = self.benchmark.env
            case (Env(), Env()):
                self.env = Env.merge(self.benchmark.env, self.optimizer.env)
            case _:
                raise ValueError("Invalid combination of benchmark and optimizer environments")

        self._set_paths(self.exp_dir)


    def _set_paths(self, expdir: Path) -> None:
        self.working_dir = expdir.absolute().resolve() / self.name
        self.complete_flag = self.working_dir / "complete.flag"
        self.error_file = self.working_dir / "error.txt"
        self.running_flag = self.working_dir / "running.flag"
        self.df_path = self.working_dir / f"{self.name}.parquet"
        self.venv_requirements_file = self.working_dir / "venv_requirements.txt"
        self.queue_flag = self.working_dir / "queue.flag"
        self.requirements_ran_with_file = self.working_dir / "requirements_ran_with.txt"
        self.env_description_file = self.working_dir / "env.yaml"
        self.post_install_steps = self.working_dir / "venv_post_install.sh"
        self.run_yaml_path = self.working_dir / "run_config.yaml"
        self.env_path = expdir / "envs" / self.env.identifier


    @property
    def venv(self) -> Venv:
        return Venv(self.env_path)

    @property
    def conda(self) -> Venv:
        raise NotImplementedError("Conda not implemented yet.")

    def run(
        self,
        *,
        on_error: Literal["raise", "continue"] = "raise",
        core_verbose: bool = False, # noqa: ARG002
        overwrite: Run.State | bool | Iterable[Run.State | str] = False,
        use_continuations_as_budget: bool = False,
        **kwargs: Any, # noqa: ARG002
    ) -> None:
        """Run the Run.

        Args:
            on_error: How to handle errors. In any case, the error will be written
                into the [`working_dir`][hpoglue.run.Run.working_dir]
                of the problem.

                * If "raise", raise an error.
                * If "continue", log the error and continue.

            core_verbose: Whether to log the core loop at INFO level.

            overwrite: Overwrite existing runs.

            use_continuations_as_budget: Whether to use continuations as budget.
                If True, the budget will be set to the fractional continuations cost.

            **kwargs: Additional keyword arguments to pass to the optimizer.
                Usage example: Scalarization weights for Neps Random Scalarization MO.
        """
        if on_error not in ("raise", "continue"):
            raise ValueError(f"Invalid value for `on_error`: {on_error}")

        overwrites = Run.State.collect(overwrite)

        state = self.state()
        if state in overwrites:
            logger.info(f"Overwriting {self.name} in `{state=}` at {self.working_dir}.")
            self.set_state(Run.State.PENDING)

        if self.df_path.exists():
            logger.info(f"Found run results for {self.name} in {self.working_dir}")
            return

        logger.info(f"Collecting {self.name} in state {state}")
        self.set_state(Run.State.PENDING)
        _hist: list[Result] = []
        try:

            self.set_state(self.State.RUNNING)
            _hist = _run(
                problem=self.problem,
                seed=self.seed,
                run_name=self.name,
                on_error=on_error,
                use_continuations_as_budget=use_continuations_as_budget,
            )
        except Exception as e:
            self.set_state(Run.State.CRASHED, err_tb=(e, traceback.format_exc()))
            logger.exception(e)
            logger.error(f"Error in Run {self.name}: {e}")
            match on_error:
                case "raise":
                    raise e
                case "continue":
                    raise NotImplementedError("Continue not yet implemented!") from e
                case _:
                    raise RuntimeError(f"Invalid value for `on_error`: {on_error}") from e
        logger.info(f"COMPLETED running {self.name}")
        logger.info(f"Saving {self.name} at {self.working_dir}")
        if _hist:
            self.create_df(history=_hist)
            logger.info(f"Results dumped at {self.df_path.absolute()}\n")


    def create_df(  # noqa: C901
        self,
        *,
        history: list[Result],
    ) -> pd.DataFrame:
        """Create a dataframe from the history."""
        _df = pd.DataFrame([res._to_dict() for res in history])
        opt_hps = "default"
        if len(self.optimizer_hyperparameters) > 0:
            opt_hps = ",".join(
                f"{k}={v}" for k, v in self.optimizer_hyperparameters.items()
            )
        _prior_annots = ",".join(
            f"{obj}={prior[0]}" for obj, prior in self.priors.items()
        )
        if "fidelity" in _df.columns and isinstance(_df["fidelity"].iloc[0], tuple):
            _df["fidelity"] = _df["fidelity"].apply(lambda x: x[1])

        minimize = {
            obj: metric.minimize for obj, metric in self.problem.objectives.items()
        }

        _df = _df.assign(
            seed=self.seed,
            budget_total=self.problem.budget.total,
            optimizer=self.optimizer.name,
            optimizer_hyperparameters=opt_hps,
            benchmark=self.benchmark.name,
            prior_annotations = _prior_annots,
            objectives=[self.problem.get_objectives()]*len(_df),
            minimize=[minimize]*len(_df),
        )

        match self.problem.fidelities:
            case None:
                _df["problem.fidelity.count"] = 0
            case (name, fid):
                _df["problem.fidelity.count"] = 1
                _df["problem.fidelity.1.name"] = name
                _df["problem.fidelity.1.min"] = fid.min
                _df["problem.fidelity.1.max"] = fid.max
            case Mapping():
                _df["problem.fidelity.count"] = len(self.problem.get_fidelities())
                for i, (name, fid) in enumerate(self.problem.fidelities.items(), start=1):
                    _df[f"problem.fidelity.{i}.name"] = name
                    _df[f"problem.fidelities.{i}.min"] = fid.min
                    _df[f"problem.fidelities.{i}.max"] = fid.max

        match self.benchmark.fidelities:
            case None:
                _df["benchmark.fidelity.count"] = 0
            case (name, fid):
                _df["benchmark.fidelity.count"] = 1
                _df["benchmark.fidelity.1.name"] = name
                _df["benchmark.fidelity.1.min"] = fid.min
                _df["benchmark.fidelity.1.max"] = fid.max
            case Mapping():
                list(self.benchmark.fidelities)
                _df["benchmark.fidelity.count"] = len(self.benchmark.fidelities)
                for i, (k, v) in enumerate(self.benchmark.fidelities.items(), start=1):
                    _df[f"benchmark.fidelity.{i}.name"] = k
                    _df[f"benchmark.fidelity.{i}.min"] = v.min
                    _df[f"benchmark.fidelity.{i}.max"] = v.max
            case _:
                raise TypeError("Must be a tuple (name, fidelitiy) or a mapping")

        self.set_state(Run.State.COMPLETE, df=_df)


    def create_env(
        self,
        *,
        how: Literal["venv", "conda"] = "venv",
        hpoglue: Literal["current_version"] | str,
    ) -> None:
        """Set up the isolation for the experiment."""
        if hpoglue == "current_version":
            raise NotImplementedError("Not implemented yet.")

        match hpoglue:
            case "current_version":
                _version = get_current_installed_hpoglue_version()
                req = f"{GLUE_PYPI}=={_version}"
            case str():
                req = hpoglue
            case _:
                raise ValueError(f"Invalid value for `hpoglue`: {hpoglue}")

        requirements = [req, *self.env.requirements]

        logger.info(f"Installing deps: {self.env.identifier}")
        self.working_dir.mkdir(parents=True, exist_ok=True)
        with self.venv_requirements_file.open("w") as f:
            f.write("\n".join(requirements))

        if self.env_path.exists():
            return

        self.env_path.parent.mkdir(parents=True, exist_ok=True)

        env_dict = self.env.to_dict()
        env_dict.update({"env_path": str(self.env_path), "hpoglue_source": req})

        logger.info(f"Installing env: {self.env.identifier}")
        match how:
            case "venv":
                logger.info(f"Creating environment {self.env.identifier} at {self.env_path}")
                self.venv.create(
                    path=self.env_path,
                    python_version=self.env.python_version,
                    requirements_file=self.venv_requirements_file,
                    exists_ok=False,
                )
                if self.env.post_install:
                    logger.info(f"Running post install for {self.env.identifier}")
                    with self.post_install_steps.open("w") as f:
                        f.write("\n".join(self.env.post_install))
                    self.venv.run(self.env.post_install)
            case "conda":
                raise NotImplementedError("Conda not implemented yet.")
            case _:
                raise ValueError(f"Invalid value for `how`: {how}")


    def to_dict(self) -> dict[str, Any]:
        _problem = self.problem.to_dict()
        _problem["priors"] = list(_problem["priors"]) if _problem["priors"] else None
        return {
            "problem": _problem,
            "seed": self.seed,
            "prior_distribution": self.prior_distribution,
            "exp_dir": str(self.exp_dir),
        }

    @classmethod
    def from_yaml(cls, path: Path) -> Run:
        with path.open("r") as file:
            return Run.from_dict(yaml.safe_load(file))

    def write_yaml(self) -> None:
        self.run_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with self.run_yaml_path.open("w") as file:
            yaml.dump(self.to_dict(), file, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Run:

        _problem: Problem = Problem.from_dict(data["problem"])
        _priors: tuple[str, Mapping[str, Config]] = _problem.priors
        seed = data["seed"]

        _prior_obj_annots = _priors[0].split(";")
        _priors = _priors[1]

        for obj_annot in _prior_obj_annots:
            obj, annot = obj_annot.split(".")
            _priors[obj] = (
                annot if annot != "None" else None,
                _priors[obj]
            )

        return Run(
            problem=_problem,
            seed=seed,
            prior_distribution=data["prior_distribution"],
            priors=_priors,
            exp_dir=Path(data["expdir"]),
        )


    @classmethod
    def generate_run(
        cls,
        optimizer: tuple[str, Mapping[str, Any]],
        benchmark: tuple[str, Mapping[str, str | None]],
        seed: int = 0,
        num_iterations: int = 1000,
        exp_dir: Path = DEFAULT_RESULTS_DIR,
        priors_dir: Path = DEFAULT_PRIORS_DIR,
        prior_distribution: Literal["normal", "uniform", "beta"] = "normal",
        **kwargs: Any  # noqa: ARG003
        ) -> Run:
        """Generates a Run instance configured with the specified optimizer, benchmark, and priors.

        Args:
            optimizer: A tuple containing the optimizer name and its hyperparameters.

            benchmark: A tuple containing the benchmark name and a mapping of
                objectives to prior annotations and optionally, the names of fidelties to use.

            seed: The random seed for reproducibility.

            num_iterations: The number of iterations for the optimization process.

            exp_dir: The directory where experiment results will be stored.

            priors_dir: The directory where prior configurations are stored.

            prior_distribution: The type of prior distribution to use.
                Available options are: "normal", "uniform", "beta".

            **kwargs: Additional keyword arguments.

        Returns:
            An instance of the Run class configured with the specified parameters.

        Notes:
            - `Run.generate_run()` reads prior configurations from YAML files
                located in the priors_dir.
            - The priors are constructed using the specified prior_distribution and
                additional keyword arguments.
            - `Run.generate_run()` creates a Problem instance with the specified
                optimizer, benchmark and priors.
            - The Run instance is returned with the configured problem, seed, priors,
                prior distribution, and experiment directory.
            - NOTE: `Run.generate_run()` only generates a Run object but does not write
                anything to the disk. That can be done using the `Run.write_yaml()` method.
        """
        optimizer_name, optimizer_kwargs = optimizer
        benchmark_name, objs_priors_fids = benchmark

        optimizer_kwargs = optimizer_kwargs or {}

        assert "objectives" in objs_priors_fids, (
            "Objectives must be specified in the benchmark config."
        )

        _priors: Mapping[str, tuple[str, Mapping[str, Any]]] = {}
        objectives = list(objs_priors_fids["objectives"].keys())
        fidelities = objs_priors_fids.get("fidelities", None)

        _prior_name = []

        for obj, prior_annot in objs_priors_fids["objectives"].items():
            _prior_name.append(f"{obj}.{prior_annot}")
            if prior_annot is not None:
                prior_path = priors_dir / f"{benchmark_name}_{obj}_{prior_annot}.yaml"
                with prior_path.open("r") as file:
                    _priors[obj] = (
                        prior_annot,
                        yaml.safe_load(file)["config"]
                    )
            else:
                _priors[obj] = (
                    None,
                    None
                )

        _prior_name = ";".join(_prior_name)
        priors: tuple[str, Mapping[str, Mapping[str, Any]]] = (
            _prior_name,
            {obj: prior[1] for obj, prior in _priors.items() if prior[1] is not None}
        )


        optimizer = OPTIMIZERS[optimizer_name]
        benchmark = BENCHMARKS[benchmark_name]
        if isinstance(benchmark, FunctionalBenchmark):
            benchmark = benchmark.desc

        optimizer_kwargs.pop("priors", None)

        match fidelities, benchmark.fidelities, optimizer.support.fidelities[0]:
            case _, None, _:
                fidelities = None
            case _, _, None:
                fidelities = None
            case _, _, "many":
                raise ValueError("Many-fidelity not yet implemented in momfpriors.")
            case None, _, "single":
                fidelities = 1
            case str() | int() | list(), _, "single":
                if isinstance(fidelities, list):
                    assert len(fidelities) == 1, (
                        "Many-fidelity not yet implemented in momfpriors."
                    )
                    fidelities = fidelities[0]
            case _:
                raise ValueError(
                    f"Invalid fidelity type: {type(fidelities)}. "
                    f"Expected None, str, int or list"
                )


        _problem = Problem.problem(
            optimizer = optimizer,
            optimizer_hyperparameters=optimizer_kwargs,
            benchmark=benchmark,
            objectives=objectives,
            fidelities=fidelities,
            budget=num_iterations,
            multi_objective_generation="mix_metric_cost",
            priors=priors,
            # NOTE: This Doesn't comply with Problem's Prior
            # typing of Mapping[str, Config], but removes the
            # need to create Prior objects everytime inside an
            # Optimizer script. # There are no checks for the
            # validity of the priors in Problem, so we can get
            #  away with this.
        )

        return Run(
            problem=_problem,
            seed=seed,
            priors=_priors if _problem.priors else {},
            prior_distribution=prior_distribution if _problem.priors else None,
            exp_dir=exp_dir,
        )


    def state(self) -> Run.State:
        """Return the state of the run.

        Args:
            run: The run to get the state for.
        """
        if self.complete_flag.exists():
            return Run.State.COMPLETE

        if self.error_file.exists():
            return Run.State.CRASHED

        if self.running_flag.exists():
            return Run.State.RUNNING

        if self.queue_flag.exists():
            return Run.State.QUEUED

        return Run.State.PENDING

    def set_state(  # noqa: C901, PLR0912
        self,
        state: Run.State,
        *,
        df: pd.DataFrame | None = None,
        err_tb: tuple[Exception, str] | None = None,
    ) -> None:
        """Set the run to a certain state.

        Args:
            state: The state to set the problem to.
            df: Optional dataframe to save if setting to [`Run.State.COMPLETE`].
            err_tb: Optional error traceback to save if setting to [`Run.State.CRASHED`].
        """
        _flags = (self.complete_flag, self.error_file, self.running_flag, self.queue_flag)
        match state:
            case Run.State.PENDING:
                for _file in (*_flags, self.df_path, self.requirements_ran_with_file):
                    _try_delete_if_exists(_file)

                with self.run_yaml_path.open("w") as f:
                    yaml.dump(self.to_dict(), f, sort_keys=False)

            case Run.State.QUEUED:
                for _file in (*_flags, self.df_path, self.requirements_ran_with_file):
                    _try_delete_if_exists(_file)

                self.queue_flag.touch()

            case Run.State.RUNNING:
                for _file in (*_flags, self.df_path, self.requirements_ran_with_file):
                    _try_delete_if_exists(_file)

                self.working_dir.mkdir(parents=True, exist_ok=True)
                self.df_path.parent.mkdir(parents=True, exist_ok=True)

                # lines = subprocess.run(
                #     [self.venv.pip, "freeze"],  # noqa: S603
                #     check=True,
                #     capture_output=True,
                #     text=True,
                # )
                # with self.requirements_ran_with_file.open("w") as f:
                #     f.write(lines.stdout)

                self.running_flag.touch()

            case Run.State.CRASHED:
                for _file in (*_flags, self.df_path):
                    _try_delete_if_exists(_file)

                with self.error_file.open("w") as f:
                    if err_tb is None:
                        f.write("None")
                    else:
                        exc, tb = err_tb
                        f.write(f"{tb}\n{exc}")

            case Run.State.COMPLETE:
                for _file in (*_flags, self.df_path):
                    _try_delete_if_exists(_file)

                self.complete_flag.touch()

                if df is not None:
                    df.to_parquet(self.df_path)
            case _:
                raise ValueError(f"Unknown state {state}")


    class State(str, Enum):
        """The state of a problem."""

        PENDING = "PENDING"
        QUEUED = "QUEUED"
        RUNNING = "RUNNING"
        CRASHED = "CRASHED"
        COMPLETE = "COMPLETE"

        @classmethod
        def collect(
            cls,
            state: str | Run.State | bool | Iterable[Run.State | str],
        ) -> list[Run.State]:
            """Collect state requested."""
            match state:
                case True:
                    return list(cls)
                case False:
                    return []
                case Run.State():
                    return [state]
                case str():
                    return [cls(state)]
                case _:
                    return [cls(s) if isinstance(s, str) else s for s in state]
