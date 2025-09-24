from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from momfpriors.constants import DEFAULT_RESULTS_DIR
from momfpriors.plotting.plot_styles import (
    RC_PARAMS,
    XTICKS,
    other_fig_params,
)
from momfpriors.plotting.plot_utils import (
    avg_seed_dfs_for_ranking,
    edit_legend_labels,
    get_style,
    regret_bounds,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


sns.set_theme(style="whitegrid")
sns.set_context("paper")

SEED_COL = "run.seed"
OPTIMIZER_COL = "optimizer.name"
BENCHMARK_COL = "benchmark.name"
HP_COL = "optimizer.hp_str"
OBJECTIVE1_COL = "result.objective.1.value"
BUDGET_USED_COL = "result.budget_used_total"
BUDGET_TOTAL_COL = "problem.budget.total"
FIDELITY_COL = "query.fidelity.1.value"
BENCHMARK_COUNT_FIDS = "benchmark.fidelity.count"
BENCH_FIDELITY_NAME = "benchmark.fidelity.1.name"
BENCH_FIDELITY_MIN_COL = "benchmark.fidelity.1.min"
BENCH_FIDELITY_MAX_COL = "benchmark.fidelity.1.max"
CONTINUATIONS_COL = "result.continuations_cost.1"
CONTINUATIONS_BUDGET_USED = "result.continuations_budget_used_total"


def normalized_regret(
    benchmark: str,
    results: pd.Series,
) -> pd.Series:
    """Calculate the normalized regret for a given benchmark and set of results."""
    bounds = regret_bounds[benchmark]
    return (results - bounds[0]) / (bounds[1] - bounds[0])


def plot_average_rank(
    ax: plt.Axes,
    ranks: dict,
    budget: int,
) -> float:
    """Plots the average rank of optimizers over iterations with standard error."""
    def _mean(_dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
            return pd.concat(_dfs).reset_index().groupby("index").mean()

    def _sem(_dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(_dfs).reset_index().groupby("index").sem()

    # Get mean across all benchmarks, for each seed
    ranks_per_seed_averaged_over_benchmarks = {
        seed: _mean(ranks[seed].values()) for seed in ranks
    }

    # Average over all seeds
    mean_ranks = _mean(ranks_per_seed_averaged_over_benchmarks.values())
    sem_ranks = _sem(ranks_per_seed_averaged_over_benchmarks.values())

    # Sort by the last iteration
    mean_ranks = mean_ranks.sort_values(by=budget, axis=1, ascending=False)
    sem_ranks = sem_ranks[mean_ranks.columns]

    # Plot settings
    plt.figure(figsize=(12, 6))

    # Plotting
    for instance in mean_ranks.columns:
        means = mean_ranks[instance]
        sems = sem_ranks[instance]

        marker, color, _, _, _ = get_style(instance)
        ax.plot(
            means,
            linestyle="-",
            label=instance,
            color=color,
        )
        ax.fill_between(
            means.index,
            means - sems,
            means + sems,
            alpha=0.1,
            color=color,
            edgecolor=None,
        )

    return np.ceil(np.max(mean_ranks))


def create_plots(  # noqa: C901, PLR0912, PLR0915
    *,
    agg_data: Mapping[str, Mapping[str, Any]],
    benchmark: str,
    budget: int,
    fidelity: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Function to plot the performance over
    iterations from a pandas Series
    of Results, i.e., Mapping[str, float] objects.
    """
    print("================================================")
    logger.info(f"Plots for runs on benchmark: {benchmark}")

    seed_means_dict = {}
    rank_means_dict = {}
    for instance, instance_data in agg_data.items():

        # Get the marker, color, optimizer name and prior annotations for the optimizer instance
        marker, color, opt, hps, prior_annot = get_style(instance)

        continuations = False
        logger.info(f"Plots for instance: {instance}")
        seed_perf_dict = {}
        seed_cont_dict = {}
        for seed, data in instance_data.items():
            results: pd.DataFrame = data["_df"]
            perf_vals: list[dict[str, Any]] = results[OBJECTIVE1_COL].values.astype(np.float64)
            is_fid_opt = FIDELITY_COL in results.columns
            total_study_budget = results[BUDGET_TOTAL_COL].iloc[0]

            budget_list = results[BUDGET_USED_COL].values.astype(np.float64)

            if not results[CONTINUATIONS_COL].isna().all():
                continuations = True

            num_full_evals = 0
            if is_fid_opt:
                final_perfs = []
                for i, perf in enumerate(perf_vals, start=1):
                    fidelity_queried = results[FIDELITY_COL].iloc[i-1]
                    bench_max_fid = results[BENCH_FIDELITY_MAX_COL].iloc[0]
                    budget_used = results[BUDGET_USED_COL].iloc[i-1]
                    if continuations:
                        budget_used = results[CONTINUATIONS_BUDGET_USED].iloc[i-1]
                    max_fid_flag = float(fidelity_queried) == float(bench_max_fid)

                    if not max_fid_flag:
                        if int(budget_used) > int(num_full_evals):
                            if num_full_evals + 1 > budget:
                                break
                            num_full_evals += 1
                            if len(final_perfs) > 0:
                                final_perfs.append(final_perfs[-1])
                            else:
                                final_perfs.append(np.nan)
                        continue
                    if num_full_evals + 1 > budget:
                        break
                    num_full_evals += 1
                    final_perfs.append(perf)

                perf_vals = final_perfs
                budget_list = np.arange(1, num_full_evals + 1, 1)


            # For MF Opts that support continuations
            if continuations:
                seed_cont_dict[seed] = normalized_regret(
                    benchmark,
                    pd.Series(perf_vals, index=budget_list)
                )
                seed_incumbents = seed_cont_dict[seed].cummin()
                instance_name = (
                    f"{opt}_w_continuations" +
                    (";" + hps if hps is not None else "") +
                    (";priors=" + prior_annot if prior_annot is not None else "")
                )

            # For MF Opts that do not support continuations
            elif is_fid_opt:
                seed_perf_dict[seed] = normalized_regret(
                    benchmark,
                    pd.Series(perf_vals, index=budget_list)
                )
                seed_incumbents = seed_perf_dict[seed].cummin()
                instance_name = instance

            # For Blackbox Optimizers
            else:
                seed_perf_dict[seed] = normalized_regret(
                    benchmark,
                    pd.Series(perf_vals, index=budget_list)
                )
                seed_incumbents = seed_perf_dict[seed].cummin()
                instance_name = instance

            seed_incumbents[total_study_budget] = seed_incumbents.iloc[-1]

            if seed not in seed_means_dict:
                seed_means_dict[seed] = {}
            seed_means_dict[seed][instance_name] = seed_incumbents


        # Aggregating Performance - calculating means, cumulative max and std_error

        _color = None

        if not continuations:
            print(instance_name)
            seed_perf_df = pd.DataFrame(seed_perf_dict)
            seed_perf_df = seed_perf_df.ffill(axis=0)
            means = pd.Series(seed_perf_df.mean(axis=1), name=f"means_{instance_name}")
            sem = pd.Series(seed_perf_df.sem(axis=1), name=f"ste_{instance_name}")
            means = means.cummin()
            means = means.drop_duplicates()
            sem = sem.loc[means.index]
            means[total_study_budget] = means.iloc[-1]
            sem[total_study_budget] = sem.iloc[-1]

        # For plotting continuations
        else:

            _, _color, _, _, _ = get_style(instance_name)
            seed_cont_df = pd.DataFrame(seed_cont_dict)
            seed_cont_df = seed_cont_df.ffill(axis=0)
            means_cont = pd.Series(seed_cont_df.mean(axis=1), name=f"means_{instance_name}")
            sem_cont = pd.Series(seed_cont_df.sem(axis=1), name=f"sem_{instance_name}")
            means_cont = means_cont.cummin()
            means_cont = means_cont.drop_duplicates()
            sem_cont = sem_cont.loc[means_cont.index]
            means_cont[total_study_budget] = means_cont.iloc[-1]
            sem_cont[total_study_budget] = sem_cont.iloc[-1]


    seed_means_dict = avg_seed_dfs_for_ranking(seed_means_dict)
    # Ranking every optimizer instance per seed per benchmark
    for seed, instances in seed_means_dict.items():
        _rankdf = pd.DataFrame(instances)
        _rankdf = _rankdf.ffill(axis=0)
        _rankdf = _rankdf.rank(axis=1, method="average", ascending=True)
        rank_means_dict[seed] = _rankdf
    instances = list(agg_data.keys())

    # if "cifar" in benchmark:
        # breakpoint()
    return rank_means_dict


def agg_data(  # noqa: C901
    exp_dir: Path,
    skip_opt: list[str] | None = None,
    skip_benchmarks: list[str] | None = None,
) -> tuple[
    Mapping[str, Mapping[tuple[str, str], list[pd.DataFrame]]],
    int,
    int
]:
    """Function to aggregate data from all runs in the experiment directory
    into a dictionary mapping benchmark names to a dictionary of run configs and dataframes.
    """
    benchmarks_in_dir = [
        (f.name.split("benchmark=")[-1].split(".")[0])
        for f in exp_dir.iterdir() if f.is_dir() and "benchmark=" in f.name]
    benchmarks_in_dir = list(set(benchmarks_in_dir))
    benchmarks_in_dir.sort()
    logger.info(f"Found benchmarks: {benchmarks_in_dir}")

    with (exp_dir / "study_config.yaml").open("r") as f:
        exp_config = yaml.safe_load(f)

    all_benches = [(bench.pop("name"), bench) for bench in exp_config["benchmarks"]]

    total_budget = int(exp_config.get("budget"))

    benchmarks_dict: Mapping[str, Mapping[tuple[str, str], list[pd.DataFrame]]] = {}

    for benchmark in benchmarks_in_dir:
        if skip_benchmarks and benchmark in skip_benchmarks:
            continue
        objectives = None
        for file in exp_dir.rglob("*.parquet"):
            if benchmark not in file.name:
                continue
            opt_name = file.name.split("optimizer=")[-1].split(".")[0]
            if skip_opt and opt_name in skip_opt:
                continue
            _df = pd.read_parquet(file)


            with (file.parent / "run_config.yaml").open("r") as f:
                run_config = yaml.safe_load(f)
            objectives = run_config["problem"]["objectives"]
            assert objectives is not None
            if not isinstance(objectives, str):
                assert len(objectives) == 1, (
                    "Plotting only single objective runs is supported. "
                    f"Found {len(objectives)} objectives in run config for {benchmark}."
                )
                objectives = objectives[0]
            fidelities = run_config["problem"]["fidelities"]
            priors = run_config["problem"]["priors"]
            if priors is not None:
                assert len(priors[1]) == 1
                priors = priors[0].split("_")[-1]
            _df["prior_annotations"] = [priors] * len(_df)
            if fidelities and not isinstance(fidelities, str) and len(fidelities) > 1:
                raise NotImplementedError("Plotting not yet implemented for many-fidelity runs.")

            # Add default benchmark fidelity to a blackbox Optimizer to compare it
            # alongside MF optimizers if the latter exist in the study
            if fidelities is None:
                fid = next(
                    bench[1]["fidelities"]
                    for bench in all_benches
                    if bench[0] == benchmark
                )
                if fid == _df[BENCH_FIDELITY_NAME].iloc[0]:
                # Study config is saved in such a way that if Blackbox Optimizers
                # are used along with MF optimizers on MF benchmarks, the "fidelities"
                # key in the benchmark instance in the study config is set to the fidelity
                # being used by the MF optimizers. In that case, there is no benchmark
                # instance with fidelity as None. In case of multiple fidelities being used
                # for the same benchmark, separate benchmark instances are created
                # for each fidelity.
                # If only Blackbox Optimizers are used in the study, there is only one
                # benchmark instance with fidelity as None.
                # When a problem with a Blackbox Optimizer is used on a MF benchmark,
                # each config is queried at the highest available 'first' fidelity in the
                # benchmark. Hence, we only set `fidelities` to `fid` if the benchmark instance
                # is the one with the default fidelity, else it would be incorrect.
                    fidelities = fid

            all_plots_dict = benchmarks_dict.setdefault(benchmark, {})
            conf_tuple = (objectives, fidelities)
            if conf_tuple not in all_plots_dict:
                all_plots_dict[conf_tuple] = [_df]
            else:
                all_plots_dict[conf_tuple].append(_df)

    return benchmarks_dict, total_budget


def gen_plots_per_bench(
    total_budget: int,
    *,
    benchmark: str,
    conf_tuple: tuple[str, str],
    _all_dfs: list[pd.DataFrame],
    priors_to_avg: list[str] | None = None,
    skip_non_avg: bool = False,
    skip_priors: bool = False,
    skip_opt: list[str] | None = None,
    avg_prior_label: str = "all",
) -> dict[int, pd.DataFrame]:
    """Function to generate plots for a given benchmark and its config dict."""
    agg_dict: Mapping[str, Mapping[str, Any]] = {}
    fidelity = conf_tuple[1]
    for _df in _all_dfs:
        if _df.empty:
            continue

        annotations = None

        if _df["prior_annotations"][0] is not None:
            annotations = "-".join(
                [a.split("=")[-1] for a in _df["prior_annotations"][0].split(",")]
            )

        instance = (
            _df[OPTIMIZER_COL][0] +
            (
                ";" + _df[HP_COL][0]
                if "default" not in _df[HP_COL][0]
                else ""
            ) +
            (f";priors={annotations}" if annotations else "")
        )

        seed = int(_df[SEED_COL][0])

        if priors_to_avg and annotations in priors_to_avg:
            instance = (
                _df[OPTIMIZER_COL][0] +
                (
                    ";" + _df[HP_COL][0]
                    if "default" not in _df[HP_COL][0]
                    else ""
                ) + f";priors={avg_prior_label}"
            )
            if skip_opt and _df[OPTIMIZER_COL][0] in skip_opt:
                continue
            if instance not in agg_dict:
                agg_dict[instance] = {}
            seed = (
                f"{seed}_{_df[OPTIMIZER_COL][0]}" +
                (
                    ";" + _df[HP_COL][0]
                    if "default" not in _df[HP_COL][0]
                    else ""
                ) + f"_{annotations}"
            )
            agg_dict[instance][seed] = {
                "_df": _df,
            }
            continue
        if annotations and skip_non_avg and priors_to_avg and annotations not in priors_to_avg:
            continue
        if skip_priors and annotations in skip_priors:
            continue
        if instance not in agg_dict:
            agg_dict[instance] = {}
        agg_dict[instance][seed] = {
            "_df": _df,
        }


    return create_plots(
        agg_data=agg_dict,
        benchmark=benchmark,
        budget=total_budget,
        fidelity=fidelity,
    )


def make_subplots(  # noqa: PLR0913
    exp_dir: Path,
    ax: plt.Axes,
    *,
    cut_off_iteration: int | None = None,
    skip_opt: list[str] | None = None,
    skip_bench: list[str] | None = None,
    priors_to_avg: list[str] | None = None,
    skip_non_avg: bool = False,
    skip_priors: bool = False,
    avg_prior_label: str = "all",
    ax_title: str | None = None,
    hide_ylabel: bool = False,
) -> None:
    """Function to make subplots for all plots in the same experiment directory."""
    xylabelsize = other_fig_params["stitched_xylabel_fontsize"]
    xytick_labelsize = other_fig_params["xytick_labelsize"]
    benchmarks_dict, total_budget = agg_data(exp_dir, skip_opt=skip_opt, skip_benchmarks=skip_bench)
    if cut_off_iteration:
        total_budget = cut_off_iteration

    plt.rcParams.update(RC_PARAMS)

    means_dict = {}
    bench_dict = {}
    for _, (benchmark, conf_dict) in enumerate(benchmarks_dict.items()):
        assert len(conf_dict) == 1, (
            "Plotting more than 1 config for a benchmark not yet implemented."
            f" Found {len(conf_dict)} configs for benchmark {benchmark}."
        )
        for conf_tuple, _all_dfs in conf_dict.items():
            seed_dict_per_bench = gen_plots_per_bench(
                benchmark=benchmark,
                conf_tuple=conf_tuple,
                _all_dfs=_all_dfs,
                total_budget=total_budget,
                priors_to_avg=priors_to_avg,
                skip_non_avg=skip_non_avg,
                skip_priors=skip_priors,
                skip_opt=skip_opt,
                avg_prior_label=avg_prior_label,
            )

            for _seed, rank_df in seed_dict_per_bench.items():
                if _seed not in means_dict:
                    means_dict[_seed] = {}
                means_dict[_seed][benchmark] = rank_df
                if _seed not in bench_dict:
                    bench_dict[_seed] = {}
                bench_dict[_seed][benchmark] = rank_df


            bench_dict = {}

    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(1.0)
        ax.spines[side].set_color("black")

    # Plotting average ranks over all benchmarks and seeds
    max_avg_rank = plot_average_rank(
        ax=ax,
        ranks=means_dict,
        budget=total_budget,
    )
    ax.set_xlabel("Full Evaluations", fontsize=xylabelsize)
    ax.set_xticks(XTICKS[(1, total_budget)])
    ax.set_yticks(np.arange(1, max_avg_rank + 1, 1))
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=xytick_labelsize,
    )
    if not hide_ylabel:
        ax.set_ylabel("Relative Rank", fontsize=xylabelsize)
    ax.grid(visible=True)
    ax.set_xlim(1, total_budget)
    if ax_title is not None:
        ax.set_title(ax_title, fontsize=other_fig_params["title_fontsize"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=None,
        help="Absolute path to the output directory where the plots will be saved. "
            "If not provided, the plots will be saved in the 'plots' subdirectory"
            "of the experiment directory."
    )
    parser.add_argument(
        "--from_yaml", "-yaml",
        type=str,
        default=None,
        required=True,
        help="YAML containing the plot configs to stitch ranking plots from. "
    )
    parser.add_argument(
        "--save_suffix", "-s",
        type=str,
        default="",
        help="Suffix to add to the saved plot filenames. "
            "Useful for distinguishing between different runs or configurations."
    )
    parser.add_argument(
        "--cut_off_iteration", "-i",
        type=int,
        default=None,
        help="Cut off iteration for the plots. "
            "If provided, only the first `cut_off_iteration` iterations will be plotted."
    )
    parser.add_argument(
        "--no_save", "-ns",
        action="store_true",
        default=False,
        help="If set, the plots will not be saved to disk."
    )
    parser.add_argument(
        "--file_type", "-file",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg", "jpg"],
        help="File type to save the plots. "
            "Can be 'pdf', 'png', 'svg', 'jpg. Default is 'pdf'."
    )
    parser.add_argument(
        "--figsize", "-size",
        type=int,
        nargs=2,
        default=(14, 6),
        help="Figure size in inches (width height)."
    )
    parser.add_argument(
        "--sub_labels", "-slabel",
        nargs="+",
        type=str,
        help="Labels for the subplots. "
    )
    parser.add_argument(
        "--remove_prior_annots", "-rm_annot",
        action="store_true",
        default=False,
        help="If set, the prior annotations will be removed from the optimizer names in the plots."
    )
    args = parser.parse_args()

    yaml_paths = []

    if args.from_yaml:
        with Path(args.from_yaml).open("r") as f:
            yaml_config = yaml.safe_load(f)
            for yaml_file in yaml_config.get("yaml_files", []):
                yaml_paths.append(Path(yaml_file))
        args.output_dir = yaml_config.get("output_dir", args.output_dir)
        args.cut_off_iteration = yaml_config.get("cut_off_iteration", args.cut_off_iteration)
        args.no_save = yaml_config.get("no_save", args.no_save)
        args.save_suffix = yaml_config.get("save_suffix", args.save_suffix)
        args.file_type = yaml_config.get("file_type", args.file_type)
        args.figsize = yaml_config.get("figsize", args.figsize)
        args.sub_labels = yaml_config.get("sub_labels", args.sub_labels)
        args.remove_prior_annots = yaml_config.get(
            "remove_prior_annots", args.remove_prior_annots
        )

    num_plots = len(yaml_paths)
    if args.sub_labels is not None:
        assert len(args.sub_labels) == num_plots


    if args.output_dir is not None:
        output_dir = Path(args.output_dir)

    # Overall Relative Ranking plot
    nrows, ncols = other_fig_params["n_rows_cols"][num_plots]
    fig_ov_rank, axs_ov_rank = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(
            args.figsize[0],
            args.figsize[1]
        ),
    )

    axs_ov_rank: list[plt.Axes] = axs_ov_rank.flatten() if num_plots > 1 else [axs_ov_rank]

    all_labels = []
    all_handles = []

    for i, yaml_path in enumerate(yaml_paths):
        with yaml_path.open("r") as f:
            yaml_config = yaml.safe_load(f)
        exp_dir = yaml_config.get("exp_dir")
        skip_opt = yaml_config.get("skip_opt")
        skip_bench = yaml_config.get("skip_bench")
        priors_to_avg = yaml_config.get("priors_to_avg")
        skip_non_avg = yaml_config.get("skip_non_avg", False)
        skip_priors = yaml_config.get("skip_priors")
        avg_prior_label = yaml_config.get("avg_prior_label", "all")
        which_labels = yaml_config.get("which_labels", "1")
        specific_rc_params = yaml_config.get("specific_rc_params")
        specific_fig_params = yaml_config.get("specific_fig_params")


        assert exp_dir is not None, "Experiment directory must be specified."
        exp_dir: Path = DEFAULT_RESULTS_DIR / exp_dir

        if specific_rc_params:
            for param in specific_rc_params:
                key, value = param.split("=")
                RC_PARAMS[key] = value

        if specific_fig_params:
            import ast
            for param in specific_fig_params:
                key, value = param.split("=")
                if key in other_fig_params:
                    other_fig_params[key] = ast.literal_eval(value)
                else:
                    logger.warning(f"Unknown figure parameter: {key}. Skipping.")

        logger.info(f"Picking plots from: {exp_dir}")

        make_subplots(
            exp_dir=exp_dir,
            ax=axs_ov_rank[i],
            cut_off_iteration=args.cut_off_iteration,
            skip_opt=skip_opt,
            skip_bench=skip_bench,
            priors_to_avg=priors_to_avg,
            skip_non_avg=skip_non_avg,
            skip_priors=skip_priors,
            avg_prior_label=avg_prior_label,
            ax_title=args.sub_labels[i] if args.sub_labels else None,
            hide_ylabel=i!=0,
        )

        ov_rank_handles, ov_rank_labels = axs_ov_rank[i].get_legend_handles_labels()
        ov_rank_labels = edit_legend_labels(ov_rank_labels, which_labels=which_labels)

        all_labels.extend(ov_rank_labels)
        all_handles.extend(ov_rank_handles)

    # Remove prior annotations from labels if specified
    if args.remove_prior_annots:
        all_labels = [
            label.split(" (")[0]
            for label in all_labels
        ]

    # Remove duplicates from all_labels and all_handles
    seen_labels = set()
    all_labels, all_handles = zip(
        *[(label, handle) for label, handle in zip(all_labels, all_handles, strict=True)
        if label not in seen_labels and not seen_labels.add(label)],
        strict=False
    )

    # Remove empty labels and handles
    all_labels = [label for label in all_labels if label]
    all_handles = [handle for handle in all_handles if handle]

    # Sort all labels and handles by label
    all_labels, all_handles = zip(
        *sorted(
            zip(all_labels, all_handles, strict=False),
            key=lambda x: len(x[0])
        ),
        strict=False
    )

    num_opts = len(all_labels)
    tight_layout_pads = other_fig_params["tight_layout_pads"]
    stitched_cols = other_fig_params["stitched_cols"][num_opts]
    leg_fontsize = other_fig_params["stitched_leg_fontsize"]


    # Legend for overall rank plot
    ov_rank_leg = fig_ov_rank.legend(
        all_handles,
        all_labels,
        fontsize=leg_fontsize,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=stitched_cols,
        frameon=True,
        markerscale=2,
    )
    for item in ov_rank_leg.legend_handles:
        item.set_linewidth(2)

    logger.info(f"Final Optimizer labels: {', '.join(all_labels)}")

    fig_ov_rank.tight_layout(**tight_layout_pads)

    save_dir = exp_dir/ "plots" / "subplots"
    if output_dir is not None:
        save_dir = output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.save_suffix:
        save_suffix = f"_{args.save_suffix}"
    if not args.no_save:
        fig_ov_rank.savefig(
            save_dir / f"rank_overall_subplots{save_suffix}.{args.file_type}",
            dpi=300, bbox_inches="tight"
        )
        logger.info("Saved overall rank plot")
