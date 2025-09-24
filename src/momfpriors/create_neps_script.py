from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path

import yaml

from momfpriors.constants import DEFAULT_RESULTS_DIR, DEFAULT_ROOT_DIR, DEFAULT_SCRIPTS_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

GLOBAL_SEED = 42
GEN_CONFIGS_DIR = "neps_parallel_gen_configs"


def split_yamls(
    optimizers: list[tuple[str, dict]],
    benchmarks: list[tuple[str, dict]],
    max_evaluations: int,
    exp_name: str = "neps_parallel",
    root_dir: Path = DEFAULT_ROOT_DIR,
    num_seeds: int = 1,
) -> None:
    """Split experiment configurations into separate YAML files
    for each optimizer-benchmark pair.
    """
    generated_configs_dir = root_dir / GEN_CONFIGS_DIR
    generated_configs_dir.mkdir(parents=True, exist_ok=True)
    array_task_id = 0
    for optimizer in optimizers:
        for benchmark in benchmarks:
            config_dict = {
                "exp_name": exp_name,
                "optimizers": [{
                    "name": optimizer[0],
                    "hyperparameters": optimizer[1],
                }],
                "benchmarks": [{
                    "name": benchmark[0],
                    "objectives": benchmark[1]["objectives"],
                }],
                "max_evaluations": max_evaluations,
                "num_seeds": num_seeds,
            }
            config_path = (
                generated_configs_dir / f"{exp_name}_{array_task_id}_config.yaml"
            )
            with config_path.open("w") as f:
                yaml.dump(config_dict, f)
            logger.info(f"Generated config file: {config_path}")
            array_task_id += 1


def create_script(
    *,
    partition: str,
    script_path: Path,
    cpus_total: int = 30,
    memory: str = "30000M",
    num_workers: int = 1,
    exp_name: str = "neps_parallel",
    job_name: str = "neps_parallel_n=1",
    time: str = "1-00:00:00",
    array: bool = True,
) -> str:
    """Create a SLURM script for running NEPS experiments in parallel."""
    generated_configs_dir = Path(GEN_CONFIGS_DIR)
    assert generated_configs_dir.exists(), (
        f"Generated configs directory {generated_configs_dir} does not exist. "
        "Please run the script with --do split first."
    )
    config_yaml = generated_configs_dir / f"{exp_name}_${{SLURM_ARRAY_TASK_ID}}_config.yaml"
    cmd_parts = [
        "python",
        "-m",
        "momfpriors.run_neps",
        "--exp_config",
        f"{config_yaml}",
        "--num_workers",
        str(num_workers)
    ]
    cmd = " ".join(cmd_parts)
    script = []
    cpus_per_task = max(1, cpus_total // num_workers)
    script.append("#!/bin/bash")
    script.append(f"#SBATCH --job-name={job_name}")
    script.append(f"#SBATCH --partition={partition}")
    script.append(f"#SBATCH --ntasks={num_workers}")
    script.append(f"#SBATCH --cpus-per-task={cpus_per_task}")
    script.append(f"#SBATCH --time={time}")
    script.append("#SBATCH --output=logs/%x_%A_%a.out")
    script.append("#SBATCH --error=logs/%x_%A_%a.err")
    if num_workers > 1:
        script.append(f"#SBATCH --mem-per-cpu={memory}")
    else:
        script.append(f"#SBATCH --mem={memory}")
    if array:
        num_configs = len(list(generated_configs_dir.glob("*_config.yaml")))
        script.append(f"#SBATCH --array=0-{num_configs - 1}")
        script.append("")

    script.append("source ~/repos/momfp_env/bin/activate")
    script.append('echo "Starting job $SLURM_JOB_ID on $SLURM_NODELIST"')
    script.append(f"srun --exclusive {cmd}")
    script_content = "\n".join(script)

    script_path = script_path / f"{job_name.replace('=', '_')}.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)

    with script_path.open("w") as f:
        f.write(script_content)
    logger.info(f"Generated SLURM script: {script_path}")

    # TODO: Array for multi-opt-bench combinations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir", "-root",
        type=Path,
        default=DEFAULT_ROOT_DIR,
        help="Path to the root directory."
    )
    parser.add_argument(
        "--results-dir", "-res",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Path to the results directory."
    )
    parser.add_argument(
        "--script_path",
        type=Path,
        default=DEFAULT_SCRIPTS_DIR,
        help="Path to save the generated SLURM script.",
    )
    parser.add_argument(
        "--max-evaluations", "-evals",
        type=int,
        default=20,
        help="Maximum number of evaluations to perform.",
    )
    parser.add_argument(
        "--exp_name", "-name",
        type=str,
        default="neps_primo_parallel_default",
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--num_workers", "-n",
        type=int,
        default=1,
        help="Number of parallel workers to use.",
    )
    parser.add_argument(
        "--num_seeds", "-s",
        type=int,
        default=1,
        help="Number of random seeds to run.",
    )
    parser.add_argument(
        "--exp_config", "-cfg",
        type=str,
        default=None,
        help="Path to experiment configuration file.",
    )
    parser.add_argument(
        "--partition", "-p",
        type=str,
        default="dev",
        help="SLURM partition to use.",
    )
    parser.add_argument(
        "--cpus_total", "-c",
        type=int,
        default=30,
        help="Total number of CPUs available for a run."
        " This will be divided among the num_workers.",
    )
    parser.add_argument(
        "--memory", "-m",
        type=str,
        default="30000M",
        help="Total memory allocated for a single worker task.",
    )
    parser.add_argument(
        "--time", "-t",
        type=str,
        default="1-00:00:00",
        help="Time limit for each job.",
    )
    parser.add_argument(
        "--no_array",
        action="store_true",
        help="Whether to not use SLURM array jobs.",
    )
    parser.add_argument(
        "--do",
        choices=["split", "slurm"],
    )
    args = parser.parse_args()

    if isinstance(args.root_dir, str):
        args.root_dir = Path(args.root_dir)

    if isinstance(args.results_dir, str):
        args.results_dir = args.root_dir / args.results_dir

    if isinstance(args.script_path, str):
        args.script_path = args.root_dir / args.script_path

    if args.exp_config:
        yaml_path = args.root_dir / args.exp_config

        with Path(yaml_path).open() as file:
            config = yaml.safe_load(file)

        _optimizers = []
        for opt in config["optimizers"]:
            match opt:
                case Mapping():
                    assert "name" in opt, f"Optimizer name not found in {opt}"
                    if len(opt) == 1 or not opt.get("hyperparameters"):
                        _optimizers.append((opt["name"], {}))
                    else:
                        _optimizers.append(tuple(opt.values()))
                case str():
                    _optimizers.append((opt, {}))
                case tuple():
                    assert len(opt) <= 2, "Each Optimizer must only have a name and hyperparameters"  # noqa: PLR2004
                    assert isinstance(opt[0], str), "Expected str for optimizer name"
                    if len(opt) == 1:
                        _optimizers.append((opt[0], {}))
                    else:
                        assert isinstance(opt[1], Mapping), (
                            "Expected Mapping for Optimizer hyperparameters"
                        )
                        _optimizers.append(opt)
                case _:
                    raise ValueError(
                        f"Invalid type for optimizer: {type(opt)}. "
                        "Expected Mapping, str or tuple"
                    )

        _benchmarks = []
        for bench in config["benchmarks"]:
            match bench:
                case Mapping():
                    assert "name" in bench, f"Benchmark name not found in {bench}"
                    assert "objectives" in bench, f"Benchmark objectives not found in {bench}"
                    _benchmarks.append(
                        (
                            bench["name"],
                            {
                                "objectives": bench["objectives"],
                                "fidelities": bench.get("fidelities"),
                            }
                        )
                    )
                case tuple():
                    assert len(bench) == 2, "Each Benchmark must only have a name and objectives"  # noqa: PLR2004
                    assert isinstance(bench[0], str), "Expected str for benchmark name"
                    assert isinstance(bench[1], Mapping), (
                        "Expected Mapping for Benchmark objectives and fidelities"
                    )
                    _benchmarks.append(bench)
                case _:
                    raise ValueError(
                        f"Invalid type for benchmark: {type(bench)}. "
                        "Expected Mapping or tuple"
                    )

    match args.do:
        case "split":
            split_yamls(
                optimizers=_optimizers,
                benchmarks=_benchmarks,
                max_evaluations=config.get("max_evaluations", args.max_evaluations),
                exp_name=config.get("exp_name", args.exp_name),
                root_dir=args.root_dir,
                num_seeds=config.get("num_seeds", args.num_seeds),
            )
        case "slurm":
            exp_name = config.get("exp_name", args.exp_name)
            num_workers = config.get("num_workers", args.num_workers)
            job_name = f"{exp_name}_n={num_workers}"
            create_script(
                partition=config.get("partition", args.partition),
                script_path=args.script_path,
                cpus_total=config.get("cpus_total", args.cpus_total),
                memory=config.get("memory", args.memory),
                num_workers=num_workers,
                exp_name=exp_name,
                job_name=job_name,
                time=config.get("time", args.time),
                array=config.get("array", not args.no_array),
            )
        case _:
            raise ValueError(f"Invalid value for --do: {args.do}")