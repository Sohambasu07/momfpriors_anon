"""Script to generate a set of plots."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

match_plot_files = {
    "merge": "merge_rank_plots",
    "merge_allfids": "merge_rank_plots_allfids",
    "main_mo": "plot_main",
    "main_mo_allfids": "plot_main_allfids",
    "main_so": "plot_so",
    "main_so_allfids": "plot_so_allfids",
}

intro = {
    "main_mo": [
        "intro_good.yaml",
    ]
}

main_exps = {
    "merge": [
        "exp_naive_all_3_rankings",
        "exp_good_ranking",
        "exp_robust_ov_bad_ranking",
    ],
    "main_mo": [
        "exp_subset1_goodgood",
        "exp_subset2_goodgood",
        "robust_subset1_ov_bad",
        "robust_subset2_ov_bad",
        "robust_subset12_ov_all"
    ],
}

app_extra = {
    "merge": [
        "addexp_badgood_ranking",
        "addexp_badbad_ranking",
        "addexp_subset1_100_all_rankings",
    ],
    "merge_allfids": [
        "addexp_momfbo_all_ranks",
    ],
    "main_mo": [
        "addexp_subset1_badgood",
        "addexp_subset2_badgood",
        "addexp_subset1_badbad",
        "addexp_subset2_badbad",
        "addexp_subset1_100_goodgood",
    ],
    "main_mo_allfids": [
        "addexp_momfbo_goodgood",
    ],
}

app_ablation = {
    "merge": [
        "ablation_init_merge_all_ranks",
        "ablation_algos_merge_all_ranks",
    ],
    "main_mo": [
        "ablation_init_goodgood",
        "ablation_init_ov_bad",
        "ablation_init_ov_all",
        "ablation_algos_ov_all",
    ]
}

match_jobs = {
    "intro": (intro, "intro"),
    "main_exps": (main_exps, "exp"),
    "app_extra": (app_extra, "appendix"),
    "app_ablation": (app_ablation, "appendix"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots for MOMF-Priors experiments."
    )
    parser.add_argument(
        "--job",
        nargs="+",
        type=str,
        default=["all"],
        choices=[*match_jobs.keys(), "all"],
        help="The job to run."
    )
    parser.add_argument(
        "--extra_common_args", "-ex_args",
        nargs="+",
        type=str,
        default=[],
        help="Extra common arguments to pass to the plotting scripts."
    )

    args = parser.parse_args()

    plot_configs_dir = Path("/home/XXXX/XXXX/code/mo_mf_priors/configs/plot_configs")

    if isinstance(args.job, str):
        args.job = [args.job]

    if "all" in args.job:
        args.job = list(match_jobs.keys())

    for job in args.job:
        assert job in match_jobs, f"Job {job} not found in match_jobs."

        for j, yamls in match_jobs[job][0].items():
            yaml_config_subdir = match_jobs[job][1]
            _plot_file = match_plot_files[j]
            for y in yamls:
                _yaml = plot_configs_dir / yaml_config_subdir / f"{y}.yaml"
                print(f"Running {j} for {_yaml}...")
                if not _yaml.exists():
                    raise FileNotFoundError(f"YAML file {_yaml} does not exist.")
                import sys
                subprocess.run(  # noqa: S603
                    [
                        sys.executable,
                        "-m",
                        f"momfpriors.plotting.{_plot_file}",
                        "--from_yaml",
                        _yaml,
                        *args.extra_common_args
                    ],
                    check=False
                )

