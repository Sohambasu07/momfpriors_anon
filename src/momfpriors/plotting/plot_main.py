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
from pymoo.indicators.hv import Hypervolume

from momfpriors.baselines import OPTIMIZERS
from momfpriors.benchmarks.botorch_momf import run_rs
from momfpriors.benchmarks.mf_zdt import _calc_pareto_front_zdt1, _calc_pareto_front_zdt6
from momfpriors.constants import DEFAULT_RESULTS_DIR
from momfpriors.plotting.plot_styles import (
    RC_PARAMS,
    XTICKS,
    other_fig_params,
)
from momfpriors.plotting.plot_utils import (
    avg_seed_dfs_for_ranking,
    change_opt_names,
    edit_axis_labels,
    edit_bench_labels,
    edit_legend_labels,
    fid_perc_momfbo,
    get_style,
    hv_low_cutoffs,
    pareto_front,
    reference_points_dict,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


sns.set_theme(style="whitegrid")
sns.set_context("paper")

SEED_COL = "seed"
OPTIMIZER_COL = "optimizer"
BENCHMARK_COL = "benchmark"
HP_COL = "optimizer_hyperparameters"
OBJECTIVES_COL = "objectives"
OBJECTIVES_MINIMIZE_COL = "minimize"
RESULTS_COL = "results"
BUDGET_USED_COL = "budget_used_total"
BUDGET_TOTAL_COL = "budget_total"
FIDELITY_COL = "fidelity"
BENCHMARK_COUNT_FIDS = "benchmark.fidelity.count"
BENCH_FIDELITY_NAME = "benchmark.fidelity.1.name"
BENCH_FIDELITY_MIN_COL = "benchmark.fidelity.1.min"
BENCH_FIDELITY_MAX_COL = "benchmark.fidelity.1.max"
CONTINUATIONS_COL = "continuations_cost"
CONTINUATIONS_BUDGET_USED = "continuations_budget_used_total"


def plot_average_rank(
    ax: plt.Axes,
    ranks: dict,
    budget: int,
) -> None:
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


def create_plots(  # noqa: C901, PLR0912, PLR0915
    *,
    ax_hv: plt.Axes,
    ax_pareto: plt.Axes,
    agg_data: Mapping[str, Mapping[str, Any]],
    benchmark: str,
    objectives: list[str],
    budget: int,
    fidelity: str | None = None,
    is_single_opt: bool = False,
    plot_true_pareto: bool = False,
    fixed_pareto_seed: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Function to plot the dominated hypervolume over
    iterations and pareto fronts from a pandas Series
    of Results, i.e., Mapping[str, float] objects.
    """
    print("================================================")
    logger.info(f"Plots for runs on benchmark: {benchmark}")
    fixed_pareto = None

    seed_means_dict = {}
    rank_means_dict = {}

    true_pareto_front = plot_true_pareto

    reference_point = np.array([
        reference_points_dict[benchmark][obj]
        for obj in objectives
    ])
    for instance, instance_data in agg_data.items():

        # Get the marker, color, optimizer name and prior annotations for the optimizer instance
        marker, color, opt, hps, prior_annot = get_style(instance)

        assert opt in OPTIMIZERS

        continuations = False
        agg_pareto_costs = []
        agg_pareto_front = []
        logger.info(f"Plots for instance: {instance}")
        seed_hv_dict = {}
        seed_cont_dict = {}
        for seed, data in instance_data.items():
            _df = data["_df"]
            results: list[dict[str, Any]] = _df[RESULTS_COL].values.tolist()
            acc_costs = []
            pareto = None
            hv_vals = []
            budget_type = "TrialBudget" if fidelity is None else "FidelityBudget"
            match budget_type:
                case "FidelityBudget":
                    assert FIDELITY_COL in _df.columns
                    if _df[FIDELITY_COL].iloc[0] is None:
                        budget_list = _df[BUDGET_USED_COL].values.astype(np.float64)
                case "TrialBudget":
                    budget_list = _df[BUDGET_USED_COL].values.astype(np.float64)
                case _:
                    raise NotImplementedError(f"Budget type {budget_type} not implemented")


            if OPTIMIZERS[opt].support.continuations:
                continuations = True

            num_full_evals = 0
            for i, costs in enumerate(results, start=1):
                if budget_type == "FidelityBudget" and _df[FIDELITY_COL][0] is not None:
                    fidelity_queried = _df[FIDELITY_COL].iloc[i-1]
                    bench_max_fid = _df[BENCH_FIDELITY_MAX_COL].iloc[0]
                    continuations_budget_used = _df[CONTINUATIONS_BUDGET_USED].iloc[i-1]
                    max_fid_flag = True
                    match instance:
                        case "MOMFBO":
                            max_fid_flag = (
                                float(fidelity_queried) / float(bench_max_fid)
                                > float(fid_perc_momfbo[benchmark])
                            )
                        case str():
                            max_fid_flag = float(fidelity_queried) == float(bench_max_fid)
                        case _:
                            print("Huh?")

                    if not max_fid_flag:
                        if int(continuations_budget_used) > int(num_full_evals):
                            if num_full_evals + 1 > budget:
                                break
                            num_full_evals += 1
                            if len(hv_vals) > 0:
                                hv_vals.append(hv_vals[-1])
                            else:
                                hv_vals.append(np.nan)
                        continue
                # Compute hypervolume
                if num_full_evals + 1 > budget:
                    break
                num_full_evals += 1
                acc_costs.append(costs)
                pareto = pareto_front(acc_costs)
                pareto = np.array([list(ac.values()) for ac in acc_costs])[pareto]
                pareto = pareto[pareto[:, 0].argsort()]
                agg_pareto_costs.extend(pareto)
                hv = Hypervolume(ref_point=reference_point)
                hypervolume = hv.do(pareto)
                hv_vals.append(hypervolume)

            budget_list = np.arange(1, num_full_evals + 1, 1)

            if continuations:
                seed_cont_dict[seed] = pd.Series(hv_vals, index=budget_list)
                seed_incumbents = seed_cont_dict[seed].cummax()
                instance_name = (
                    f"{opt}_w_continuations" +
                    (";" + hps if hps is not None else "") +
                    (";priors=" + prior_annot if prior_annot is not None else "")
                )
            else:
                seed_hv_dict[seed] = pd.Series(hv_vals, index=budget_list)
                seed_incumbents = seed_hv_dict[seed].cummax()
                instance_name = instance

            seed_incumbents[budget] = seed_incumbents.iloc[-1]

            if seed not in seed_means_dict:
                seed_means_dict[seed] = {}
            seed_means_dict[seed][instance_name] = seed_incumbents

            if fixed_pareto_seed is not None and seed == fixed_pareto_seed:
                fixed_pareto = pareto


        # Aggregating Hypervolumes - calculating means, cumulative max and std_error

        _color = None

        if not continuations:
            seed_hv_df = pd.DataFrame(seed_hv_dict)
            seed_hv_df = seed_hv_df.ffill(axis=0)
            means = pd.Series(seed_hv_df.mean(axis=1), name=f"means_{instance}")
            sem = pd.Series(seed_hv_df.sem(axis=1), name=f"ste_{instance}")
            means = means.cummax()
            means = means.drop_duplicates()
            sem = sem.loc[means.index]
            means[budget] = means.iloc[-1]
            sem[budget] = sem.iloc[-1]

            # Plotting Hypervolumes

            ax_hv.plot(
                means,
                label=instance,
                color=color if not is_single_opt else None,
            )
            ax_hv.fill_between(
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
            means_cont = pd.Series(seed_cont_df.mean(axis=1), name=f"means_{instance}")
            sem_cont = pd.Series(seed_cont_df.sem(axis=1), name=f"sem_{instance}")
            means_cont = means_cont.cummax()
            means_cont = means_cont.drop_duplicates()
            sem_cont = sem_cont.loc[means_cont.index]
            means_cont[budget] = means_cont.iloc[-1]
            sem_cont[budget] = sem_cont.iloc[-1]

            ax_hv.plot(
                means_cont,
                label=f"{instance}_w_continuations",
                color=_color if not is_single_opt else None,
            )
            ax_hv.fill_between(
                means_cont.index,
                means_cont - sem_cont,
                means_cont + sem_cont,
                alpha=0.1,
                color=_color if not is_single_opt else None,
                edgecolor=None,
            )

        if fixed_pareto is None:

            # Aggregated Pareto front over all seeds

            agg_pareto_costs = np.array(agg_pareto_costs)
            agg_pareto_front = pareto_front(agg_pareto_costs)
            agg_pareto_front = np.array(agg_pareto_costs)[agg_pareto_front]
            agg_pareto_front = agg_pareto_front[agg_pareto_front[:, 0].argsort()]
            ax_pareto.step(
                agg_pareto_front[:, 0],
                agg_pareto_front[:, 1],
                where="post",
                marker=marker,
                color=color if not continuations else _color,
                label=instance if not continuations else f"{instance}_w_continuations",
            )
        else:
            # Plotting the fixed Pareto front for the given seed
            ax_pareto.step(
                fixed_pareto[:, 0],
                fixed_pareto[:, 1],
                where="post",
                marker=marker,
                color=color if not continuations else _color,
                label=instance if not continuations else f"{instance}_w_continuations",
            )

    # Plotting the true Pareto front if available
    if true_pareto_front:
        match benchmark:
            case "MFZDT1":
                true_pareto = _calc_pareto_front_zdt1(n_pareto_points=20)
            case "MFZDT6":
                true_pareto = _calc_pareto_front_zdt6(n_pareto_points=20)
            case "MOMFPark":
                rs_results = run_rs(
                    fn="MOMFPark",
                    n_samples=100000,
                    seed=fixed_pareto_seed,
                )
                true_pareto = pareto_front(
                    np.array(rs_results),
                )
                true_pareto = np.array(rs_results)[true_pareto]
                true_pareto = true_pareto[true_pareto[:, 0].argsort()]
            case _:
                logger.warning(
                    f"True Pareto front not available for benchmark {benchmark}. "
                )
        ax_pareto.step(
            true_pareto[:, 0],
            true_pareto[:, 1],
            where="post",
            color="black",
            label="True Pareto Front",
            marker="s",
        )

    seed_means_dict = avg_seed_dfs_for_ranking(seed_means_dict)
    # Ranking every optimizer instance per seed per benchmark
    for seed, instances in seed_means_dict.items():
        _rankdf = pd.DataFrame(instances)
        _rankdf = _rankdf.ffill(axis=0)
        _rankdf = _rankdf.rank(axis=1, method="average", ascending=False)
        rank_means_dict[seed] = _rankdf
    instances = list(agg_data.keys())
    return rank_means_dict


def agg_data(  # noqa: C901
    exp_dir: Path,
    skip_opt: list[str] | None = None,
    skip_bench: list[str] | None = None,
    which_benchmarks: list[str] | None = None,
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
    if skip_bench:
        benchmarks_in_dir = [
            bench for bench in benchmarks_in_dir if bench not in skip_bench
        ]
    benchmarks_in_dir.sort()

    with (exp_dir / "exp.yaml").open("r") as f:
        exp_config = yaml.safe_load(f)

    if which_benchmarks is not None:
        print(which_benchmarks)
        benchmarks_in_dir = list(set(which_benchmarks))
    logger.info(f"Found benchmarks: {benchmarks_in_dir}")

    all_benches = [
        (bench.pop("name"), bench)
        for bench in exp_config["benchmarks"]
        if bench["name"] in benchmarks_in_dir
    ]

    total_budget = int(exp_config.get("budget"))

    benchmarks_dict: Mapping[str, Mapping[tuple[str, str], list[pd.DataFrame]]] = {}

    for benchmark in benchmarks_in_dir:
        objectives = []
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
            fidelities = run_config["problem"]["fidelities"]
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
            conf_tuple = (tuple(objectives), fidelities)
            if conf_tuple not in all_plots_dict:
                all_plots_dict[conf_tuple] = [_df]
            else:
                all_plots_dict[conf_tuple].append(_df)

    return benchmarks_dict, total_budget


def gen_plots_per_bench(  # noqa: C901, PLR0913
    ax_hv: plt.Axes,
    ax_pareto: plt.Axes,
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
    plot_true_pareto: bool = False,
    fixed_pareto_seed: int | None = None,
) -> dict[int, pd.DataFrame]:
    """Function to generate plots for a given benchmark and its config dict."""
    agg_dict: Mapping[str, Mapping[str, Any]] = {}
    objectives = conf_tuple[0]
    fidelity = conf_tuple[1]
    for _df in _all_dfs:
        if _df.empty:
            continue
        assert objectives is not None


        assert len(objectives) == 2, ( # noqa: PLR2004
            "More than 2 objectives found in results file: "
            f"{objectives}. "
            "Can only plot pareto front for 2D cost space."
        )

        _results = _df[RESULTS_COL].apply(
            lambda x, objectives=objectives: {
                k: x[k] for k in objectives
            }
        )

        _df[RESULTS_COL] = _results

        annotations = None

        optimizer_name = change_opt_names(
            _df[OPTIMIZER_COL][0],
        )

        if _df["prior_annotations"][0] is not None:
            annotations = "-".join(
                [a.split("=")[-1] for a in _df["prior_annotations"][0].split(",")]
            )

        instance = (
            optimizer_name +
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
                optimizer_name +
                (
                    ";" + _df[HP_COL][0]
                    if "default" not in _df[HP_COL][0]
                    else ""
                ) + f";priors={avg_prior_label}"
            )
            if skip_opt and optimizer_name in skip_opt:
                continue
            if instance not in agg_dict:
                agg_dict[instance] = {}
            seed = (
                f"{seed}_{optimizer_name}" +
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
    assert len(objectives) > 0, "Objectives not found in results file."


    return create_plots(
        ax_hv=ax_hv,
        ax_pareto=ax_pareto,
        agg_data=agg_dict,
        benchmark=benchmark,
        budget=total_budget,
        fidelity=fidelity,
        is_single_opt=is_single_opt,
        objectives=objectives,
        plot_true_pareto=plot_true_pareto,
        fixed_pareto_seed=fixed_pareto_seed,
    )



def make_subplots(  # noqa: C901, PLR0912, PLR0913, PLR0915
    exp_dir: Path,
    *,
    cut_off_iteration: int | None = None,
    no_save: bool = False,
    save_suffix: str = "",
    skip_opt: list[str] | None = None,
    skip_bench: list[str] | None = None,
    which_benchmarks: list[str] | None = None,
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
    hv_cut_off: bool = False,
    prior_annotations: str | None = None,
    plot_true_pareto: bool = False,
    fixed_pareto_seed: int | None = None,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Function to make subplots for all plots in the same experiment directory."""
    if which_benchmarks is not None and not isinstance(which_benchmarks, list):
        which_benchmarks = [which_benchmarks]
    if which_plots is None:
        which_plots = ["all"]
    if which_plots is None:
        which_plots = ["hv", "pareto", "rank", "ov_rank"]
    fig_size = figsize or other_fig_params["fig_size"]
    benchmarks_dict, total_budget = agg_data(
        exp_dir,
        skip_opt=skip_opt,
        skip_bench=skip_bench,
        which_benchmarks=which_benchmarks,
    )
    if cut_off_iteration:
        total_budget = cut_off_iteration
    num_plots = len(benchmarks_dict)
    nrows, ncols = other_fig_params["n_rows_cols"][num_plots]

    if which_benchmarks is not None and not isinstance(which_benchmarks, Iterable):
        which_benchmarks = [which_benchmarks]

    # Hypervolume plot
    fig_hv, axs_hv = plt.subplots(
        nrows = nrows,
        ncols = ncols,
        figsize=fig_size,
    )

    # Pareto plot
    fig_pareto, axs_pareto = plt.subplots(
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
        fig_hv.suptitle(
            plot_title,
            x=suptitle_x,
            y=suptitle_y,
            fontsize=other_fig_params["title_fontsize"])
        fig_pareto.suptitle(
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

    axs_hv: list[plt.Axes] = axs_hv.flatten() if num_plots > 1 else [axs_hv]
    axs_pareto: list[plt.Axes] = axs_pareto.flatten() if num_plots > 1 else [axs_pareto]
    axs_rank: list[plt.Axes] = axs_rank.flatten() if num_plots > 1 else [axs_rank]
    axs_ov_rank: list[plt.Axes] = [axs_ov_rank]

    xylabel_fontsize = other_fig_params["xylabel_fontsize"]
    ovrank_xylabelsize = other_fig_params["stitched_xylabel_fontsize"]
    xytick_labelsize = other_fig_params["xytick_labelsize"]
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
                ax_hv=axs_hv[i],
                ax_pareto=axs_pareto[i],
                benchmark=benchmark,
                conf_tuple=conf_tuple,
                _all_dfs=_all_dfs,
                total_budget=total_budget,
                priors_to_avg=priors_to_avg,
                skip_non_avg=skip_non_avg,
                skip_priors=skip_priors,
                skip_opt=skip_opt,
                avg_prior_label=avg_prior_label,
                plot_true_pareto=plot_true_pareto,
                fixed_pareto_seed=fixed_pareto_seed,
            )

            bench_label = edit_bench_labels(benchmark)

            axs_hv[i].set_xticks(XTICKS[(1, total_budget)])
            axs_hv[i].tick_params(
                direction="out",
                axis="both",
                which="major",
                labelsize=xytick_labelsize,
            )
            axs_hv[i].grid(visible=True)
            axs_hv[i].set_title(bench_label)
            axs_hv[i].set_xlim(1, total_budget)
            if hv_cut_off:
                axs_hv[i].set_ylim(hv_low_cutoffs[benchmark])

            axs_pareto[i].grid(visible=True)
            axs_pareto[i].tick_params(
                axis="both",
                which="major",
                labelsize=xytick_labelsize,
            )
            axs_pareto[i].set_title(bench_label)

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
            axs_rank[i].set_title(bench_label)
            axs_rank[i].grid(visible=True)
            axs_rank[i].set_xlim(1, total_budget)

            pareto_xlabel = edit_axis_labels("x")
            pareto_ylabel = edit_axis_labels("y")

            if i >= xlabel_i:
                axs_hv[i].set_xlabel("Evaluations", fontsize=xylabel_fontsize)
                axs_pareto[i].set_xlabel(pareto_xlabel, fontsize=xylabel_fontsize)
                axs_rank[i].set_xlabel("Evaluations", fontsize=xylabel_fontsize)

            if i == ylabel_i_counter:
                axs_hv[i].set_ylabel("Hypervolume", fontsize=xylabel_fontsize)
                axs_pareto[i].set_ylabel(pareto_ylabel, fontsize=xylabel_fontsize)
                axs_rank[i].set_ylabel("Relative Rank", fontsize=xylabel_fontsize)
                ylabel_i_counter += ylabel_i_inc

            for side in ["left", "bottom"]:
                axs_hv[i].spines[side].set_color("black")
                axs_rank[i].spines[side].set_color("black")
                axs_pareto[i].spines[side].set_color("black")


            bench_dict = {}

    # Plotting average ranks over all benchmarks and seeds
    plot_average_rank(
        ax=axs_ov_rank[0],
        ranks=means_dict,
        budget=total_budget,
    )
    axs_ov_rank[0].set_xlabel("Evaluations", fontsize=ovrank_xylabelsize)
    axs_ov_rank[0].set_ylabel("Relative Rank", fontsize=ovrank_xylabelsize)
    axs_ov_rank[0].tick_params(
        axis="both",
        which="major",
        labelsize=xytick_labelsize,
    )
    axs_ov_rank[0].spines["left"].set_color("black")
    axs_ov_rank[0].spines["bottom"].set_color("black")
    axs_ov_rank[0].set_xticks(XTICKS[(1, total_budget)])
    axs_ov_rank[0].grid(visible=True)
    axs_ov_rank[0].set_xlim(1, total_budget)

    hv_handles, hv_labels = axs_hv[0].get_legend_handles_labels()
    pareto_handles, pareto_labels = axs_pareto[0].get_legend_handles_labels()
    rank_handles, rank_labels = axs_rank[0].get_legend_handles_labels()
    ov_rank_handles, ov_rank_labels = axs_ov_rank[0].get_legend_handles_labels()

    hv_labels = edit_legend_labels(
        hv_labels,
        which_labels=which_labels,
        prior_annotations=prior_annotations
    )
    pareto_labels = edit_legend_labels(
        pareto_labels,
        which_labels=which_labels,
        prior_annotations=prior_annotations
    )
    rank_labels = edit_legend_labels(
        rank_labels,
        which_labels=which_labels,
        prior_annotations=prior_annotations
    )
    ov_rank_labels = edit_legend_labels(
        ov_rank_labels,
        which_labels=which_labels,
        prior_annotations=prior_annotations
    )

    logger.info(f"Final Optimizer labels: {hv_labels}")

    # Sort all labels and handles by label

    hv_labels, hv_handles = zip(
        *sorted(
            zip(hv_labels, hv_handles, strict=False),
            key=lambda x: len(x[0])
        ),
        strict=False
    )
    pareto_labels, pareto_handles = zip(
        *sorted(
            zip(pareto_labels, pareto_handles, strict=False),
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

    num_opts = len(hv_labels)

    bbox_to_anchor = other_fig_params["bbox_to_anchor"]
    tight_layout_pads = other_fig_params["tight_layout_pads"]
    legend_fontsize = other_fig_params["legend_fontsize"]

    multi_cols = other_fig_params["multi_fig_leg_cols"][num_opts]
    single_cols = other_fig_params["single_fig_leg_cols"][num_opts]

    # Remove empty subplots for hypervolume plot
    for j in range(i + 1, len(axs_hv)):
        fig_hv.delaxes(axs_hv[j])

    # Legend for hypervolume plot
    if "all" not in turn_off_legends or "hv" not in turn_off_legends:
        hv_leg = fig_hv.legend(
            hv_handles,
            hv_labels,
            fontsize=legend_fontsize,
            loc="lower center",
            bbox_to_anchor=bbox_to_anchor,
            ncol=multi_cols,
            frameon=True,
            markerscale=2,
        )
        for item in hv_leg.legend_handles:
            item.set_linewidth(2)

    fig_hv.tight_layout(**tight_layout_pads)

    # Remove empty subplots for pareto plot
    for j in range(i + 1, len(axs_pareto)):
        fig_pareto.delaxes(axs_pareto[j])

    # Legend for pareto plot
    if "all" not in turn_off_legends or "pareto" not in turn_off_legends:
        pareto_leg = fig_pareto.legend(
            pareto_handles,
            pareto_labels,
            fontsize=legend_fontsize,
            loc="lower center",
            bbox_to_anchor=bbox_to_anchor,
            ncol=multi_cols,
            frameon=True,
            markerscale=2,
        )
        for item in pareto_leg.legend_handles:
            item.set_linewidth(2)

    fig_pareto.tight_layout(**tight_layout_pads)

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

    ovrank_leg_fontsize = other_fig_params["stitched_leg_fontsize"]

    # Legend for overall rank plot
    if "all" not in turn_off_legends or "ov_rank" not in turn_off_legends:
        ov_rank_leg = fig_ov_rank.legend(
            ov_rank_handles,
            ov_rank_labels,
            fontsize=14,
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
        if "all" in which_plots or "hv" in which_plots:
            fig_hv.savefig(
                save_dir / f"hypervolume_subplots{save_suffix}.{file_type}",
                dpi=300, bbox_inches="tight"
            )
            logger.info("Saved hv plot")
        if "all" in which_plots or "pareto" in which_plots:
            fig_pareto.savefig(
                save_dir / f"pareto_subplots{save_suffix}.{file_type}",
                dpi=300, bbox_inches="tight"
            )
            logger.info("Saved pareto plot")
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
        help="Plot pareto fronts and Hypervolumes for the given iterations."
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
        "--skip_bench", "-skipb",
        nargs="+",
        type=str,
        default=None,
        help="Skip the given benchmarks."
    )
    parser.add_argument(
        "--which_benchmarks", "-bench",
        nargs="+",
        type=str,
        default=None,
        help="Which benchmarks to plot. If not provided, all benchmarks in the experiment directory"
        "will be plotted."
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
            "'hv' generates only hypervolume plots, "
            "'pareto' generates only pareto plots, "
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
            "hv, pareto, rank, ov_rank can be specified to turn off legend for specific plots."
    )
    parser.add_argument(
        "--plot_title", "-title",
        type=str,
        default="",
        help="Title for the plots. If not provided, no title will be set."
    )
    parser.add_argument(
        "--hv_cut_off", "-hv_cut",
        action="store_true",
        help="Cut off the hypervolume plot at the specified ylim."
    )
    parser.add_argument(
        "--prior_annotations", "-annots",
        type=str,
        default="default",
        help="Custom prior annotations to overwrite the existing ones. "
            "Only supports single prior combination."
    )
    parser.add_argument(
        "--plot_true_pareto", "-true_pareto",
        action="store_true",
        help="Plot the true Pareto front if available for the benchmark."
    )
    parser.add_argument(
        "--fixed_pareto_seed", "-fpseed",
        type=int,
        default=None,
        help="fixed seed for the Pareto front. "
            "If not provided, the Pareto front will be aggregated over all seeds."
    )
    parser.add_argument(
        "--figsize", "-figsize",
        nargs=2,
        type=float,
        default=None,
        help="Figure size for the plots. "
            "If not provided, default figure size will be used."
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
        args.skip_bench = yaml_config.get("skip_bench", args.skip_bench)
        args.which_benchmarks = yaml_config.get("which_benchmarks", args.which_benchmarks)
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
        args.hv_cut_off = yaml_config.get("hv_cut_off", args.hv_cut_off)
        args.prior_annotations = yaml_config.get("prior_annotations", args.prior_annotations)
        args.plot_true_pareto = yaml_config.get("plot_true_pareto", args.plot_true_pareto)
        args.fixed_pareto_seed = yaml_config.get("fixed_pareto_seed", args.fixed_pareto_seed)
        args.figsize = yaml_config.get("figsize", args.figsize)

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

    make_subplots(
        exp_dir=exp_dir,
        cut_off_iteration=args.cut_off_iteration,
        no_save=args.no_save,
        save_suffix=args.save_suffix,
        skip_opt=args.skip_opt,
        skip_bench=args.skip_bench,
        which_benchmarks=args.which_benchmarks,
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
        hv_cut_off=args.hv_cut_off,
        prior_annotations=args.prior_annotations,
        plot_true_pareto=args.plot_true_pareto,
        fixed_pareto_seed=args.fixed_pareto_seed,
        figsize=args.figsize,
    )
