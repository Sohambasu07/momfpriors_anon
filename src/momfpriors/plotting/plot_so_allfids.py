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
    normalized_regret = (results - bounds[0]) / (bounds[1] - bounds[0])
    # breakpoint()
    return normalized_regret


def calc_eqv_full_evals(
    results: pd.Series,
    budget_total: float,
    benchmark: str
) -> pd.Series:
    """Calculate equivalent full evaluations for fractional costs."""
    evals = np.arange(1, budget_total + 1)
    _df = results.reset_index()
    _df.columns = ["index", "performance"]
    bins = pd.cut(_df["index"], bins=[-np.inf, *evals], right=False, labels=evals)
    _df["group"] = bins
    group_min = _df.groupby("group", observed=True)["performance"].min()
    result = group_min.reindex(evals)
    return normalized_regret(
        benchmark=benchmark,
        results=result,
    )

def plot_average_rank(
    ax: plt.Axes,
    ranks: dict,
    budget: int,
) -> None:
    # breakpoint()
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

    # # Sort by the last iteration
    # mean_ranks = mean_ranks.sort_values(by=budget, axis=1, ascending=True)
    # sem_ranks = sem_ranks[mean_ranks.columns]


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


def create_plots(  # noqa: PLR0915
    *,
    ax: plt.Axes,
    agg_data: Mapping[str, Mapping[str, Any]],
    benchmark: str,
    budget: int,
    fidelity: str | None = None,
    is_single_opt: bool = False,
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

            # Check if continuations col is not NA
            if not results[CONTINUATIONS_COL].isna().all():
                continuations = True
                continuations_list = results[CONTINUATIONS_BUDGET_USED].values.astype(np.float64)


            # For MF Opts that support continuations
            if continuations:
                seed_cont_dict[seed] = pd.Series(perf_vals, index=continuations_list)
                seed_cont_dict[seed] = calc_eqv_full_evals(
                    seed_cont_dict[seed],
                    total_study_budget,
                    benchmark=benchmark,
                )
                seed_incumbents = seed_cont_dict[seed].cummin()
                instance_name = (
                    f"{opt}_w_continuations" +
                    (";" + hps if hps is not None else "") +
                    (";priors=" + prior_annot if prior_annot is not None else "")
                )

            # For MF Opts that do not support continuations
            elif is_fid_opt:
                seed_perf_dict[seed] = pd.Series(perf_vals, index=budget_list)
                seed_perf_dict[seed] = calc_eqv_full_evals(
                    seed_perf_dict[seed],
                    total_study_budget,
                    benchmark=benchmark,
                )
                seed_incumbents = seed_perf_dict[seed].cummin()
                instance_name = instance

            # For Blackbox Optimizers
            else:
                seed_perf_dict[seed] = pd.Series(perf_vals, index=budget_list)
                seed_incumbents = seed_perf_dict[seed].cummin()
                instance_name = instance

            seed_incumbents[total_study_budget] = seed_incumbents.iloc[-1]

            if seed not in seed_means_dict:
                seed_means_dict[seed] = {}
            seed_means_dict[seed][instance_name] = seed_incumbents

            # if "MFBO" in instance and "cifar" in benchmark:
                # breakpoint()


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

            # Plotting Performance

            ax.plot(
                means,
                label=instance_name,
                color=color if not is_single_opt else None,
            )
            ax.fill_between(
                means.index,
                means - sem,
                means + sem,
                alpha=0.1,
                color=color if not is_single_opt else None,
                edgecolor=None,
            )

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

            ax.plot(
                means_cont,
                label=f"{instance_name}_w_continuations",
                color=_color if not is_single_opt else None,
            )
            ax.fill_between(
                means_cont.index,
                means_cont - sem_cont,
                means_cont + sem_cont,
                alpha=0.1,
                color=_color if not is_single_opt else None,
                edgecolor=None,
            )

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


def gen_plots_per_bench(  # noqa: C901
    ax: plt.Axes,
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
    is_single_opt = False
    if len(agg_dict) == 1:
        is_single_opt = True


    return create_plots(
        ax=ax,
        agg_data=agg_dict,
        benchmark=benchmark,
        budget=total_budget,
        fidelity=fidelity,
        is_single_opt=is_single_opt,
    )



def make_subplots(  # noqa: C901, PLR0912, PLR0913, PLR0915
    exp_dir: Path,
    *,
    cut_off_iteration: int | None = None,
    no_save: bool = False,
    save_suffix: str = "",
    skip_opt: list[str] | None = None,
    skip_benchmarks: list[str] | None = None,
    priors_to_avg: list[str] | None = None,
    skip_non_avg: bool = False,
    skip_priors: bool = False,
    avg_prior_label: str = "all",
    file_type: str = "pdf",
    output_dir: Path | None = None,
    which_labels: str = "1",
    which_plots: list[str] | None = None,
    turn_off_legends: list[str] | None = None,
    plot_title: str | None = None,
) -> None:
    """Function to make subplots for all plots in the same experiment directory."""
    if which_plots is None:
        which_plots = ["all"]
    if which_plots is None:
        which_plots = ["perf", "rank", "ov_rank"]
    fig_size = other_fig_params["fig_size"]
    benchmarks_dict, total_budget = agg_data(
        exp_dir,
        skip_opt=skip_opt,
        skip_benchmarks=skip_benchmarks)
    if cut_off_iteration:
        total_budget = cut_off_iteration
    num_plots = len(benchmarks_dict)
    nrows, ncols = other_fig_params["n_rows_cols"][num_plots]

    # Performance plot
    fig_perf, axs_perf = plt.subplots(
        nrows = nrows,
        ncols = ncols,
        figsize=fig_size,
    )

    # Relative Ranking per benchmark plot
    fig_rank, axs_rank = plt.subplots(
        nrows = nrows,
        ncols = ncols,
        figsize=fig_size,
    )

    # Overall Relative Ranking plot
    fig_ov_rank, axs_ov_rank = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(
            other_fig_params["ovrank_xsize"],
            other_fig_params["ovrank_ysize"],
        ),
    )
    suptitle_x = other_fig_params["suptitle_bbox"][0]
    suptitle_y = other_fig_params["suptitle_bbox"][1]

    # Set the title of the plots
    if plot_title is not None:
        fig_perf.suptitle(
            plot_title,
            x=suptitle_x,
            y=suptitle_y,
            fontsize=other_fig_params["title_fontsize"])
        fig_rank.suptitle(
            plot_title,
            x=suptitle_x,
            y=suptitle_y,
            fontsize=other_fig_params["title_fontsize"])
        fig_ov_rank.suptitle(
            plot_title,
            x=suptitle_x,
            y=suptitle_y,
            fontsize=other_fig_params["title_fontsize"])

    plt.rcParams.update(RC_PARAMS)

    axs_perf: list[plt.Axes] = axs_perf.flatten() if num_plots > 1 else [axs_perf]
    axs_rank: list[plt.Axes] = axs_rank.flatten() if num_plots > 1 else [axs_rank]
    axs_ov_rank: list[plt.Axes] = [axs_ov_rank]

    xylabel_fontsize = other_fig_params["xylabel_fontsize"]
    xlabel_i = other_fig_params["xlabel_start_i"][num_plots]
    ylabel_i_inc = other_fig_params["ylabel_i_inc"][num_plots]
    ylabel_i_counter = 0

    means_dict = {}
    bench_dict = {}
    for i, (benchmark, conf_dict) in enumerate(benchmarks_dict.items()):
        assert len(conf_dict) == 1, (
            "Plotting more than 1 config for a benchmark not yet implemented."
            f" Found {len(conf_dict)} configs for benchmark {benchmark}."
        )
        for conf_tuple, _all_dfs in conf_dict.items():
            seed_dict_per_bench = gen_plots_per_bench(
                ax=axs_perf[i],
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

            axs_perf[i].set_xticks(XTICKS[(1, total_budget)])
            axs_perf[i].grid(visible=True, which="both", ls="-")
            axs_perf[i].set_title(benchmark)
            axs_perf[i].set_xlim(1, total_budget)
            axs_perf[i].set_ylim(top=1)
            axs_perf[i].set_yscale("log")

            for _seed, rank_df in seed_dict_per_bench.items():
                if _seed not in means_dict:
                    means_dict[_seed] = {}
                means_dict[_seed][benchmark] = rank_df
                if _seed not in bench_dict:
                    bench_dict[_seed] = {}
                bench_dict[_seed][benchmark] = rank_df

            # Plotting average ranks per benchmark
            plot_average_rank(
                ax=axs_rank[i],
                ranks=bench_dict,
                budget=total_budget,
            )
            axs_rank[i].set_xticks(XTICKS[(1, total_budget)])
            axs_rank[i].set_title(benchmark)
            axs_rank[i].grid(visible=True)
            axs_rank[i].set_xlim(1, total_budget)

            if i >= xlabel_i:
                axs_perf[i].set_xlabel("Full Evaluations", fontsize=xylabel_fontsize)
                axs_rank[i].set_xlabel("Full Evaluations", fontsize=xylabel_fontsize)

            if i == ylabel_i_counter:
                axs_perf[i].set_ylabel("Normalized Regret", fontsize=xylabel_fontsize)
                axs_rank[i].set_ylabel("Relative Rank", fontsize=xylabel_fontsize)
                ylabel_i_counter += ylabel_i_inc


            bench_dict = {}

    # Plotting average ranks over all benchmarks and seeds
    plot_average_rank(
        ax=axs_ov_rank[0],
        ranks=means_dict,
        budget=total_budget,
    )
    axs_ov_rank[0].set_xlabel("Full Evaluations")
    axs_ov_rank[0].set_xticks(XTICKS[(1, total_budget)])
    axs_ov_rank[0].set_ylabel("Relative Rank")
    axs_ov_rank[0].grid(visible=True)
    axs_ov_rank[0].set_xlim(1, total_budget)

    perf_handles, perf_labels = axs_perf[0].get_legend_handles_labels()
    rank_handles, rank_labels = axs_rank[0].get_legend_handles_labels()
    ov_rank_handles, ov_rank_labels = axs_ov_rank[0].get_legend_handles_labels()

    perf_labels = edit_legend_labels(perf_labels, which_labels=which_labels)
    rank_labels = edit_legend_labels(rank_labels, which_labels=which_labels)
    ov_rank_labels = edit_legend_labels(ov_rank_labels, which_labels=which_labels)

    logger.info(f"Final Optimizer labels: {perf_labels}")

    # Sort all labels and handles by label

    perf_labels, perf_handles = zip(
        *sorted(
            zip(perf_labels, perf_handles, strict=False),
            key=lambda x: len(x[0])
        ),
        strict=False
    )
    rank_labels, rank_handles = zip(
        *sorted(
            zip(rank_labels, rank_handles, strict=False),
            key=lambda x: len(x[0])
        ),
        strict=False
    )
    ov_rank_labels, ov_rank_handles = zip(
        *sorted(
            zip(ov_rank_labels, ov_rank_handles, strict=False),
            key=lambda x: len(x[0])
        ),
        strict=False
    )

    num_opts = len(perf_labels)

    bbox_to_anchor = other_fig_params["bbox_to_anchor"]
    tight_layout_pads = other_fig_params["tight_layout_pads"]
    legend_fontsize = other_fig_params["legend_fontsize"]

    multi_cols = other_fig_params["multi_fig_leg_cols"][num_opts]
    single_cols = other_fig_params["single_fig_leg_cols"][num_opts]

    # Remove empty subplots for performance plot
    for j in range(i + 1, len(axs_perf)):
        fig_perf.delaxes(axs_perf[j])

    # Legend for performance plot
    if "all" not in turn_off_legends or "perf" not in turn_off_legends:
        perf_leg = fig_perf.legend(
            perf_handles,
            perf_labels,
            fontsize=legend_fontsize,
            loc="lower center",
            bbox_to_anchor=bbox_to_anchor,
            ncol=multi_cols,
            frameon=True,
            markerscale=2,
        )
        for item in perf_leg.legend_handles:
            item.set_linewidth(2)

    fig_perf.tight_layout(**tight_layout_pads)

    # Remove empty subplots for rank plot
    for j in range(i + 1, len(axs_rank)):
        fig_rank.delaxes(axs_rank[j])

    # Legend for rank plot
    if "all" not in turn_off_legends or "rank" not in turn_off_legends:
        rank_leg = fig_rank.legend(
            rank_handles,
            rank_labels,
            fontsize=legend_fontsize,
            loc="lower center",
            bbox_to_anchor=bbox_to_anchor,
            ncol=multi_cols,
            frameon=True,
            markerscale=2,
        )
        for item in rank_leg.legend_handles:
            item.set_linewidth(2)

    fig_rank.tight_layout(**tight_layout_pads)

    # Legend for overall rank plot
    if "all" not in turn_off_legends or "ov_rank" not in turn_off_legends:
        ov_rank_leg = fig_ov_rank.legend(
            ov_rank_handles,
            ov_rank_labels,
            fontsize=10,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=single_cols,
            frameon=True,
            markerscale=2,
        )
        for item in ov_rank_leg.legend_handles:
            item.set_linewidth(2)

    fig_ov_rank.tight_layout(**tight_layout_pads)

    save_dir = exp_dir/ "plots" / "subplots"
    if output_dir is not None:
        save_dir = output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    if save_suffix:
        save_suffix = f"_{save_suffix}"
    if not no_save:
        if "all" in which_plots or "perf" in which_plots:
            fig_perf.savefig(
                save_dir / f"performance_subplots{save_suffix}.{file_type}",
                dpi=300, bbox_inches="tight"
            )
            logger.info("Saved perf plot")
        if "all" in which_plots or "rank" in which_plots:
            fig_rank.savefig(
                save_dir / f"rank_subplots{save_suffix}.{file_type}",
                dpi=300, bbox_inches="tight"
            )
            logger.info("Saved rank plot")
        if "all" in which_plots or "ov_rank" in which_plots:
            fig_ov_rank.savefig(
                save_dir / f"rank_overall_subplots{save_suffix}.{file_type}",
                dpi=300, bbox_inches="tight"
            )
            logger.info("Saved overall rank plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir", "-e",
        type=str,
        default=None,
        help="Main experiment directory containing the runs to plot."
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=None,
        help="Absolute path to the output directory where the plots will be saved. "
            "If not provided, the plots will be saved in the 'plots' subdirectory"
            "of the experiment directory."
    )
    parser.add_argument(
        "--cut_off_iteration", "-i",
        type=int,
        default=None,
        help="Cut off all plots at the given iteration."
    )
    parser.add_argument(
        "--no_save", "-ns",
        action="store_true",
        help="Do not save the plots."
    )
    parser.add_argument(
        "--save_suffix", "-suffix",
        type=str,
        default="",
        help="Suffix to add to the saved plots."
    )
    parser.add_argument(
        "--skip_opt", "-skip",
        nargs="+",
        type=str,
        default=None,
        help="Skip the given optimizers."
    )
    parser.add_argument(
        "--priors_to_avg", "-avg",
        nargs="+",
        type=str,
        default=None,
        help="Priors to average over."
    )
    parser.add_argument(
        "--skip_non_avg", "-skipna",
        action="store_true",
        help="Skip the non-averaged priors."
    )
    parser.add_argument(
        "--skip_priors", "-skp",
        nargs="+",
        type=str,
        default=None,
        help="Skip the given priors."
    )
    parser.add_argument(
        "--avg_prior_label", "-avg_label",
        type=str,
        default="all",
        help="Label for the averaged priors."
    )
    parser.add_argument(
        "--file_type", "-file",
        type=str,
        choices=["pdf", "png", "svg", "jpg"],
        default="pdf",
        help="File type to save the plots."
    )
    parser.add_argument(
        "--which_labels", "-labels",
        type=str,
        default="1",
        help="Which labels to use for the plots."
    )
    parser.add_argument(
        "--which_plots", "-plots",
        nargs="+",
        type=str,
        default="all",
        help="Which plots to generate. 'all' generates all plots, "
            "'perf' generates only performance plots, "
            "'rank' generates only rank plots, "
            "'ov_rank' generates only overall rank plots."
    )
    parser.add_argument(
        "--from_yaml", "-yaml",
        type=str,
        default=None,
        help="Path to a YAML file containing the plotting configuration."
    )
    parser.add_argument(
        "--specific_rc_params", "-rc",
        nargs="+",
        type=str,
        default=None,
        help="Specific rcParams to set for the plots. "
            "Format: 'param1=value1 param2=value2 ...'. "
            "If not provided, default rcParams will be used."
    )
    parser.add_argument(
        "--specific_fig_params", "-figparams",
        nargs="+",
        type=str,
        default=None,
        help="Specific figure parameters to set for the plots. "
            "Format: 'param1=value1 param2=value2 ...'. "
            "If not provided, default figure parameters will be used."
    )
    parser.add_argument(
        "--turn_off_legends", "-nolegends",
        nargs="+",
        type=str,
        default="all",
        help="Turn off legend for the plots. "
            "If 'all' is provided, legend will be turned off for all plots. "
            "perf, rank, ov_rank can be specified to turn off legend for specific plots."
    )
    parser.add_argument(
        "--plot_title", "-title",
        type=str,
        default="",
        help="Title for the plots. If not provided, no title will be set."
    )
    parser.add_argument(
        "--skip_benchmarks", "-skip_bench",
        nargs="+",
        type=str,
        default=None,
        help="Skip the given benchmarks. "
            "Useful for skipping benchmarks that are not relevant for the current analysis."
    )
    args = parser.parse_args()

    if args.from_yaml:
        with Path(args.from_yaml).open("r") as f:
            yaml_config = yaml.safe_load(f)
        args.exp_dir = yaml_config.get("exp_dir", args.exp_dir)
        args.output_dir = yaml_config.get("output_dir", args.output_dir)
        args.cut_off_iteration = yaml_config.get("cut_off_iteration", args.cut_off_iteration)
        args.no_save = yaml_config.get("no_save", args.no_save)
        args.save_suffix = yaml_config.get("save_suffix", args.save_suffix)
        args.skip_opt = yaml_config.get("skip_opt", args.skip_opt)
        args.priors_to_avg = yaml_config.get("priors_to_avg", args.priors_to_avg)
        args.skip_non_avg = yaml_config.get("skip_non_avg", args.skip_non_avg)
        args.skip_priors = yaml_config.get("skip_priors", args.skip_priors)
        args.avg_prior_label = yaml_config.get("avg_prior_label", args.avg_prior_label)
        args.file_type = yaml_config.get("file_type", args.file_type)
        args.which_labels = yaml_config.get("which_labels", args.which_labels)
        args.which_plots = yaml_config.get("which_plots", args.which_plots)
        args.specific_rc_params = yaml_config.get("specific_rc_params", args.specific_rc_params)
        args.specific_fig_params = yaml_config.get("specific_fig_params", args.specific_fig_params)
        args.turn_off_legends = yaml_config.get("turn_off_legends", args.turn_off_legends)
        args.plot_title = yaml_config.get("plot_title", args.plot_title)
        args.skip_benchmarks = yaml_config.get("skip_benchmarks", args.skip_benchmarks)

    if args.specific_rc_params:
        for param in args.specific_rc_params:
            key, value = param.split("=")
            RC_PARAMS[key] = value

    if args.specific_fig_params:
        import ast
        for param in args.specific_fig_params:
            key, value = param.split("=")
            if key in other_fig_params:
                other_fig_params[key] = ast.literal_eval(value)
            else:
                logger.warning(f"Unknown figure parameter: {key}. Skipping.")

    assert args.exp_dir is not None, "Experiment directory must be specified."
    exp_dir: Path = DEFAULT_RESULTS_DIR / args.exp_dir

    agg_dict: Mapping[str, Mapping[str, Any]] = {}
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)

    if isinstance(args.which_plots, str):
        args.which_plots = [args.which_plots]

    if isinstance(args.turn_off_legends, str):
        args.turn_off_legends = [args.turn_off_legends]

    print(other_fig_params["fig_size"])

    make_subplots(
        exp_dir=exp_dir,
        cut_off_iteration=args.cut_off_iteration,
        no_save=args.no_save,
        save_suffix=args.save_suffix,
        skip_opt=args.skip_opt,
        priors_to_avg=args.priors_to_avg,
        skip_non_avg=args.skip_non_avg,
        skip_priors=args.skip_priors,
        avg_prior_label=args.avg_prior_label,
        file_type=args.file_type,
        output_dir=output_dir if args.output_dir else None,
        which_labels=args.which_labels,
        which_plots=args.which_plots,
        turn_off_legends=args.turn_off_legends,
        plot_title=args.plot_title,
        skip_benchmarks=args.skip_benchmarks,
    )
