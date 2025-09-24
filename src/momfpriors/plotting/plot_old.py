from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from pymoo.indicators.hv import Hypervolume

from momfpriors.baselines import OPTIMIZERS
from momfpriors.constants import DEFAULT_RESULTS_DIR, DEFAULT_ROOT_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


sns.set_theme(style="whitegrid")

SEED_COL = "seed"
OPTIMIZER_COL = "optimizer"
BENCHMARK_COL = "benchmark"
HP_COL = "optimizer_hyperparameters"
OBJECTIVES_COL = "objectives"
OBJECTIVES_MINIMIZE_COL = "minimize"
BUDGET_USED_COL = "budget_used_total"
BUDGET_TOTAL_COL = "budget_total"
FIDELITY_COL = "fidelity"
BENCHMARK_COUNT_FIDS = "benchmark.fidelity.count"
BENCH_FIDELITY_NAME = "benchmark.fidelity.1.name"
BENCH_FIDELITY_MIN_COL = "benchmark.fidelity.1.min"
BENCH_FIDELITY_MAX_COL = "benchmark.fidelity.1.max"
CONTINUATIONS_COL = "continuations_cost"


reference_points_dict = {

    # PD1
    "pd1-cifar100-wide_resnet-2048": {"valid_error_rate": 1, "train_cost": 100},
    "pd1-imagenet-resnet-512": {"valid_error_rate": 1, "train_cost": 5000},
    "pd1-lm1b-transformer-2048": {"valid_error_rate": 1, "train_cost": 1000},
    "pd1-translate_wmt-xformer_translate-64": {"valid_error_rate": 1, "train_cost": 20000},

    # JAHSBench
    "jahs-CIFAR10": {"valid_acc": 0, "runtime": 200000},
    "jahs-ColorectalHistology": {"valid_acc": 0, "runtime": 200000},
    "jahs-FashionMNIST": {"valid_acc": 0, "runtime": 200000},

    # MOMFPark
    "MOMFPark": {"value1": 1, "value2": 1},

    # YAHPO-LCBench
    "yahpo-lcbench-126026": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 200
    },
    "yahpo-lcbench-167190": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 200
    },
    "yahpo-lcbench-168330": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 20000
    },
    "yahpo-lcbench-168910": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189906": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 1000
    },
    "yahpo-lcbench-3945": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-7593": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-34539": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-126025": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-126029": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-146212": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167104": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167149": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167152": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167161": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167168": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167181": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167184": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167185": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167200": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167201": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-168329": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-168331": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-168335": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-168868": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-168908": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189354": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189862": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189865": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189866": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189873": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189905": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189908": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189909": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    }
}



with (DEFAULT_ROOT_DIR / "configs" / "plotting_styles.yaml").open("r") as f:
    style_dict = yaml.safe_load(f)


def dynamic_reference_point(
    loss_vals: pd.Series | list[Mapping[str, float]]
) -> np.array:
    """Function to calculate the pareto front from a pandas Series
    of Results, i.e., Mapping[str, float] objects.
    """
    if isinstance(loss_vals, pd.Series):
        loss_vals = loss_vals.to_numpy()
    loss_vals = np.array(
        [
            list(cost.values())
            for cost in loss_vals
        ]
    )
    worst_point = np.max(loss_vals, axis=0)
    reference_point = np.maximum(1.1 * worst_point, 0.9 * worst_point)
    reference_point[reference_point == 0] = 1e-12
    return reference_point


def pareto_front(
    costs: pd.Series | list[Mapping[str, float]] | np.array,
) -> np.array:
    """Function to calculate the pareto front from a pandas Series
    of Results, i.e., Mapping[str, float] objects.
    """
    if not isinstance(costs, np.ndarray):
        costs: np.array = np.array([list(cost.values()) for cost in costs])
    is_pareto = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(costs[is_pareto] < c, axis=1)
            is_pareto[i] = True
    return is_pareto


def _get_style(instance: str) -> tuple[str, str, str, str | None]:
    """Function to get the plotting style for a given instance."""
    prior_annot = instance.split("priors=")[-1] if "priors=" in instance else None
    opt = instance.split(";")[0]
    color = style_dict["colors"].get(opt)
    if prior_annot:
        color = style_dict["colors"].get(f"{opt}-{prior_annot}")
    marker = style_dict["markers"].get(prior_annot, "s")
    return marker, color, opt, prior_annot


def plot_average_rank(
    ranks: dict,
    budget: int,
    budget_type: str,
    exp_dir: Path,
    *,
    no_save: bool = False,
    benchmark: str | None = None,
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

        marker, color, _, _ = _get_style(instance)
        plt.plot(
            means,
            linestyle="-",
            label=instance,
            # marker=marker,
            color=color,
        )
        plt.fill_between(
            means.index,
            means - sems,
            means + sems,
            alpha=0.2,
            linewidth=2,
            color=color,
            edgecolor=color,
        )

    # Labels and legend
    plot_title = "Average_Rank_of_Optimizers_over_Iterations_across_all_Benchmarks_and_seeds"
    if benchmark:
        plot_title = plot_title.replace(
            "across_all_Benchmarks_and_seeds",
            f"for_{benchmark}_and_all_seeds"
        )
    match budget_type:
        case "TrialBudget":
            plt.xlabel("Number of Iterations")
            plt.xticks(np.arange(5, budget + 1, step=5))
        case "FidelityBudget":
            plt.xlabel("Fidelity Budget")
            plt.xticks(np.arange(means.index[0], budget + 1, step=500))
    plt.ylabel("Average Rank (Lower is Better)")
    plt.title(plot_title)
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(0, -0.05),
        fontsize=12
    )
    plt.tight_layout()
    plt.tight_layout()
    plt.grid(linestyle="--", alpha=0.6)

    # Save the plot
    if not no_save:
        plt.savefig(exp_dir / "plots" / f"{plot_title}.png")
    plt.show()


def create_plots(  # noqa: C901, PLR0912, PLR0913, PLR0915
    agg_data: Mapping[str, Mapping[str, Any]],
    exp_dir: Path,
    benchmark: str,
    objectives: list[str],
    seed_for_pareto: int,
    cut_off_iteration: int | None = None,
    fidelity: str | None = None,
    plot_opt: str | None = None,
    *,
    is_single_opt: bool = False,
    no_save: bool = False,
    aggregate_pareto_fronts: bool = False,
) -> None:
    """Function to plot the dominated hypervolume over
    iterations and pareto fronts from a pandas Series
    of Results, i.e., Mapping[str, float] objects.
    """
    print("================================================")
    logger.info(f"Plots for runs on benchmark: {benchmark}")
    plt.figure(1, figsize=(10, 10))
    plt.figure(2, figsize=(10, 10))

    seed_means_dict = {}
    rank_means_dict = {}

    reference_point = np.array([
        reference_points_dict[benchmark][obj]
        for obj in objectives
    ])
    if plot_opt:
        plt.figure(3, figsize=(10, 10))
    if aggregate_pareto_fronts:
        plt.figure(4, figsize=(10, 10))
    for instance, instance_data in agg_data.items():

        # Get the marker, color, optimizer name and prior annotations for the optimizer instance
        marker, color, opt, prior_annot = _get_style(instance)

        assert opt in OPTIMIZERS

        continuations = False
        agg_pareto_costs = []
        agg_pareto_front = []
        budget=0
        logger.info(f"Plots for instance: {instance}")
        seed_hv_dict = {}
        seed_cont_dict = {}
        for seed, data in instance_data.items():
            results: dict = data["results"]
            _df = data["_df"]
            plot_title = data["plot_title"]
            keys = list(results[0].keys())
            acc_costs = []
            pareto = None
            hv_vals = []
            minimize: dict[str, bool] = _df["minimize"][0]
            budget_type = "TrialBudget" if fidelity is None else "FidelityBudget"
            match budget_type:
                case "FidelityBudget":
                    assert FIDELITY_COL in _df.columns
                    # hposuite FidelityBudget code
                    if _df[FIDELITY_COL].iloc[0] is not None:
                        budget_list = _df[FIDELITY_COL].values.astype(np.float64)
                        budget_list = np.cumsum(budget_list)
                        budget_type = "FidelityBudget"
                    else:
                        budget_list = np.cumsum(
                            _df[BENCH_FIDELITY_MAX_COL].values.astype(np.float64)
                        )
                case "TrialBudget":
                    budget_list = _df[BUDGET_USED_COL].values.astype(np.float64)
                case _:
                    raise NotImplementedError(f"Budget type {budget_type} not implemented")
            budget = max(budget, budget_list[-1])

            # print(_df[FIDELITY_COL])
            # print(_df[CONTINUATIONS_COL])

            if "single" in OPTIMIZERS[opt].support.fidelities:
                continuations = True
                continuations_list = _df[CONTINUATIONS_COL].values.astype(np.float64)
                continuations_list = np.cumsum(continuations_list)

            # print(continuations)

            results = _df["results"].apply(
                lambda x, objectives=objectives, minimize=minimize: {
                    k: x[k] if minimize[k] else -x[k] for k in objectives
                }
            )
            for i, costs in enumerate(results, start=1):
                # Compute hypervolume
                acc_costs.append(costs)
                pareto = pareto_front(acc_costs)
                pareto = np.array([list(ac.values()) for ac in acc_costs])[pareto]
                pareto = pareto[pareto[:, 0].argsort()]
                if aggregate_pareto_fronts:
                    agg_pareto_costs.extend(pareto)
                if budget_type == "FidelityBudget" and _df[FIDELITY_COL][0] is not None:
                    fidelity_queried = _df[FIDELITY_COL].iloc[i-1]
                    if float(fidelity_queried) != float(_df[BENCH_FIDELITY_MAX_COL].iloc[0]):
                        if hv_vals:
                            hv_vals.append(hv_vals[-1])
                        else:
                            hv_vals.append(np.nan)
                        continue
                hv = Hypervolume(ref_point=reference_point)
                hypervolume = hv.do(pareto)
                hv_vals.append(hypervolume)

                # if i == cut_off_iteration and budget_type == "TrialBudget":
                #     break
            # budget_list = _df[BUDGET_USED_COL].values.astype(np.float64)[:budget]
            seed_hv_dict[seed] = pd.Series(hv_vals, index=budget_list)

            if continuations:
                # print(len(hv_vals), len(continuations_list))
                # print(f"{seed_hv_dict[seed]=}")
                seed_cont_dict[seed] = pd.Series(hv_vals, index=continuations_list)

            # Accumulating incumbents for every optimizer instance per seed per benchmark
            seed_incumbents = seed_hv_dict[seed].cummax()
            if seed not in seed_means_dict:
                seed_means_dict[seed] = {}
            seed_means_dict[seed][instance] = seed_incumbents

            if budget_type == "FidelityBudget":
                plot_dir = exp_dir / "plots" / budget_type
            else:
                plot_dir = exp_dir / "plots" / budget_type / str(budget)

            match plot_opt:
                case None:
                    pass
                case "all":
                # Plotting Pareto fronts for each seed
                    plt.figure(3)
                    plt.step(
                        pareto[:, 0],
                        pareto[:, 1],
                        where="post",
                        marker=marker,
                        label=f"{instance}_{seed}",
                        markersize=7,
                        linewidth=1,
                    )
                    plt.legend([f"{instance}_{seed}" for seed in instance_data])
                    plt.xlabel(keys[0])
                    plt.ylabel(keys[1])
                    plt.grid(visible=True)
                    plt.title(f"Multiple seeds pareto plot for \n{instance} on {benchmark}")
                    plt.tight_layout()
                case str():
                    if plot_opt in instance:
                        plt.figure(3)
                        plt.step(
                            pareto[:, 0],
                            pareto[:, 1],
                            where="post",
                            marker=marker,
                            label=f"{instance}_{seed}",
                            markersize=7,
                            linewidth=1,
                        )
                        plt.legend()
                        plt.xlabel(keys[0])
                        plt.ylabel(keys[1])
                        plt.grid(visible=True)
                        plt.title(f"Multiple seeds pareto plot for \n{instance} on {benchmark}")
                case _:
                    raise ValueError(f"Invalid plot_opt: {plot_opt}.")
            if seed == seed_for_pareto:
                plt.figure(1)
                plt.step(
                    pareto[:, 0],
                    pareto[:, 1],
                    where="post",
                    marker=marker,
                    color=color if not is_single_opt else None,
                    label=f"{instance}_{seed}",
                    markersize=7,
                    linewidth=1,
                )

        if plot_opt and (plot_opt == "all" or plot_opt in instance):
            pareto_save_dir = plot_dir /"pareto"
            pareto_save_dir.mkdir(parents=True, exist_ok=True)
            plt.figure(3)
            if not no_save:
                plt.savefig(
                    pareto_save_dir / "multiple" /
                    f"Multiple seeds pareto plot for {instance} on {benchmark}.png"
                )
            plt.clf()

        if len(agg_data) > 1:
            plot_title = benchmark

        plt.figure(1)
        plt.legend(
            loc="upper left",
            bbox_to_anchor=(0, -0.05),
            fontsize=12
        )
        plt.tight_layout()
        plt.xlabel(keys[0])
        plt.ylabel(keys[1])
        plt.grid(visible=True)
        plt.title(benchmark)
        pareto_save_dir = plot_dir / "pareto"
        pareto_save_dir.mkdir(parents=True, exist_ok=True)

        # Aggregating Hypervolumes - calculating means, cumulative max and std_error


        plt.figure(2)

        if not continuations:
            seed_hv_df = pd.DataFrame(seed_hv_dict)
            seed_hv_df = seed_hv_df.ffill(axis=0)
            # exit(0)
            # print(seed_hv_df)
            means = pd.Series(seed_hv_df.mean(axis=1), name=f"means_{instance}")
            ste = pd.Series(seed_hv_df.sem(axis=1), name=f"ste_{instance}")
            means = means.cummax()
            means = means.drop_duplicates()
            ste = ste.loc[means.index]
            means[budget] = means.iloc[-1]
            ste[budget] = ste.iloc[-1]

            # print(means)
            # print(ste)

            # Plotting Hypervolumes
            plt.plot(
                means,
                # marker=marker,
                color=color if not is_single_opt else None,
                linestyle="-",
                label=instance,
            )
            plt.fill_between(
                means.index,
                means - ste,
                means + ste,
                alpha=0.2,
                color=color if not is_single_opt else None,
                edgecolor=color if not is_single_opt else None,
                linewidth=2,
            )

        #For plotting continuations
        if continuations:

            _marker, _color, _, _ = _get_style(f"{opt}_w_continuations")
            seed_cont_df = pd.DataFrame(seed_cont_dict)
            seed_cont_df = seed_cont_df.ffill(axis=0)
            # seed_cont_df = seed_cont_df.dropna(axis=0)
            means_cont = pd.Series(seed_cont_df.mean(axis=1), name=f"means_{instance}")
            sem_cont = pd.Series(seed_cont_df.sem(axis=1), name=f"sem_{instance}")
            means_cont = means_cont.cummax()
            means_cont = means_cont.drop_duplicates()
            sem_cont = sem_cont.loc[means_cont.index]

            # print(len(means_cont), len(sem_cont))
            plt.plot(
                means_cont,
                label=f"{instance}_w_continuations",
                # marker=marker,
                # markersize=5,
                color=_color,
                linewidth=2,
            )
            plt.fill_between(
                means_cont.index,
                means_cont - sem_cont,
                means_cont + sem_cont,
                alpha=0.2,
                step="post",
                color=_color,
                edgecolor=_color,
                linewidth=2,
            )


        plt.legend(
            loc="upper left",
            bbox_to_anchor=(0, -0.05),
            fontsize=12
        )
        plt.tight_layout()
        plt.xlabel(budget_type)
        if budget_type == "TrialBudget":
            plt.xticks(np.arange(1, budget + 1, 1))
        plt.ylabel("Hypervolume")
        plt.grid(visible=True)
        plt.title(benchmark)
        hv_save_dir = plot_dir / "hypervolume"
        hv_save_dir.mkdir(parents=True, exist_ok=True)

        # Aggregated Pareto front over all seeds

        if aggregate_pareto_fronts:
            agg_pareto_costs = np.array(agg_pareto_costs)
            agg_pareto_front = pareto_front(agg_pareto_costs)
            agg_pareto_front = np.array(agg_pareto_costs)[agg_pareto_front]
            agg_pareto_front = agg_pareto_front[agg_pareto_front[:, 0].argsort()]
            plt.figure(4)
            plt.step(
                agg_pareto_front[:, 0],
                agg_pareto_front[:, 1],
                where="post",
                marker=marker,
                color=color,
                label=instance,
                markersize=5,
                linewidth=1,
            )
            plt.xlabel(keys[0])
            plt.ylabel(keys[1])
            plt.grid(visible=True)
            plt.title(f"Aggregated Pareto front for \n{plot_title}")
            plt.legend()


    plot_title = plot_title.replace("\n", "")
    if not no_save:
        plt.figure(1)
        plt.savefig(pareto_save_dir / "seeds" / f"Pareto_plot_{plot_title.replace(' ', '_')}.png")

        plt.figure(2)
        plt.savefig(hv_save_dir / f"Hypervolume_plot_{plot_title.replace(' ', '_')}.png")

        plt.figure(4)
        plt.savefig(
            pareto_save_dir / "agg" / f"Aggregated_Pareto_plot_{plot_title.replace(' ', '_')}.png"
        )
    plt.show()

    plt.close("all")

    # Ranking every optimizer instance per seed per benchmark
    for seed, instances in seed_means_dict.items():
        _rankdf = pd.DataFrame(instances)
        _rankdf = _rankdf.ffill(axis=0)
        _rankdf = _rankdf.dropna(axis=0)
        # print(_rankdf)
        _rankdf = _rankdf.rank(axis=1, method="average", ascending=False)
        # print(_rankdf)
        # exit(0)
        rank_means_dict[seed] = _rankdf
    # exit(0)
    return rank_means_dict, budget_type, budget


def agg_data(  # noqa: C901, PLR0912, PLR0915
    exp_dir: Path,
    plot_opt: str | None = None,
    plot_iters: list[int] | None = None,
    *,
    no_save: bool = False,
    agg_pareto: bool = False,
)-> None:
    """Function to aggregate data from all runs in the experiment directory."""
    agg_dict: Mapping[str, Mapping[str, Any]] = {}
    means_dict = {}

    benchmarks_in_dir = [
        (f.name.split("benchmark=")[-1].split(".")[0])
        for f in exp_dir.iterdir() if f.is_dir() and "benchmark=" in f.name]
    benchmarks_in_dir = list(set(benchmarks_in_dir))
    benchmarks_in_dir.sort()
    logger.info(f"Found benchmarks: {benchmarks_in_dir}")

    with (exp_dir / "exp.yaml").open("r") as f:
        exp_config = yaml.safe_load(f)

    all_benches = [(bench.pop("name"), bench) for bench in exp_config["benchmarks"]]

    seed_for_pareto = exp_config.get("seeds")[0]
    iterations = exp_config.get("budget")


    if plot_iters and isinstance(plot_iters, int):
        plot_iters = [plot_iters]
    plot_iters = plot_iters if plot_iters else []

    if iterations not in plot_iters:
        assert any(plot_iters) < iterations, (
            "Iterations to plot should be less than the total budget."
        )
        plot_iters.append(iterations)


    benchmarks_dict: Mapping[str, Mapping[tuple[str, str], list[pd.DataFrame]]] = {}

    for benchmark in benchmarks_in_dir:
        bench_dict= {}
        objectives = []
        for file in exp_dir.rglob("*.parquet"):
            if benchmark not in file.name:
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

            seed = int(run_config["seed"])
            all_plots_dict = benchmarks_dict.setdefault(benchmark, {})
            conf_tuple = (tuple(objectives), fidelities)
            if conf_tuple not in all_plots_dict:
                all_plots_dict[conf_tuple] = [_df]
            else:
                all_plots_dict[conf_tuple].append(_df)


    for benchmark, conf_dict in benchmarks_dict.items():
        for conf_tuple, _all_dfs in conf_dict.items():
            df_agg = {}
            objectives = conf_tuple[0]
            fidelity = conf_tuple[1]
            for _df in _all_dfs:
                if _df.empty:
                    continue
                instance = _df[OPTIMIZER_COL].iloc[0]
                if _df[HP_COL].iloc[0] is not None:
                    instance = f"{instance}_{_df[HP_COL].iloc[0]}"
                seed = _df[SEED_COL].iloc[0]
                if instance not in df_agg:
                    df_agg[instance] = {}
                if int(seed) not in df_agg[instance]:
                    df_agg[instance][int(seed)] = {"results": _df}
                assert objectives is not None


                assert len(objectives) == 2, ( # noqa: PLR2004
                    "More than 2 objectives found in results file: "
                    f"{objectives}. "
                    "Can only plot pareto front for 2D cost space."
                )

                minimize: dict[str, bool] = _df["minimize"][0]

                _results = _df["results"].apply(
                    lambda x, objectives=objectives, minimize=minimize: {
                        k: x[k] if minimize[k] else -x[k] for k in objectives
                    }
                )

                if _df["prior_annotations"][0] is not None:
                    annotations = "-".join(
                        [a.split("=")[-1] for a in _df["prior_annotations"][0].split(",")]
                    )

                instance = (
                    _df["optimizer"][0] +
                    (
                        ";" + _df[HP_COL][0]
                        if "default" not in _df[HP_COL][0]
                        else ""
                    ) +
                    (f";priors={annotations}" if annotations else "")
                )
                seed = int(_df[SEED_COL][0])
                if instance not in agg_dict:
                    agg_dict[instance] = {}
                agg_dict[instance][seed] = {
                    "_df": _df,
                    "results": _results,
                    "plot_title": (
                        f"{instance}"
                        f" on \n{benchmark}"
                    ),
                }
            is_single_opt = False
            if len(agg_dict) == 1:
                is_single_opt = True
            assert len(objectives) > 0, "Objectives not found in results file."


            seed_dict_per_bench, budget_type, total_budget = create_plots(
                agg_data=agg_dict,
                exp_dir=exp_dir,
                benchmark=benchmark,
                # cut_off_iteration=iteration,
                fidelity=fidelity,
                is_single_opt=is_single_opt,
                plot_opt=plot_opt,
                seed_for_pareto=seed_for_pareto,
                objectives=objectives,
                no_save=no_save,
                aggregate_pareto_fronts=agg_pareto,
            )
            agg_dict = {}
            for _seed, rank_df in seed_dict_per_bench.items():
                if _seed not in means_dict:
                    means_dict[_seed] = {}
                means_dict[_seed][benchmark] = rank_df
                if _seed not in bench_dict:
                    bench_dict[_seed] = {}
                bench_dict[_seed][benchmark] = rank_df
            plot_average_rank(
                bench_dict,
                budget=total_budget,
                budget_type=budget_type,
                exp_dir=exp_dir,
                no_save=no_save,
                benchmark=benchmark,
            )
            bench_dict = {}
            # Per benchmark ranking plots
    # Plotting average ranks over all benchmarks and seeds
    plot_average_rank(
        means_dict,
        budget=total_budget,
        budget_type=budget_type,
        exp_dir=exp_dir,
        no_save=no_save
    )


def make_subplots(  # noqa: C901
    exp_dir: Path,
    iteration: int | None = None,
    budget_type: str | None = None,
    *,
    no_save: bool = False,
) -> None:
    """Function to make subplots for all plots in the same experiment directory."""
    if isinstance(iteration, list):
        iteration = iteration[0]
    if budget_type == "FidelityBudget":
        pareto_plots_dir = exp_dir / "plots" / "FidelityBudget" / "pareto" / "seeds"
        hv_plots_dir = exp_dir / "plots" / "FidelityBudget" / "hypervolume"
    else:
        pareto_plots_dir = exp_dir / "plots" / "TrialBudget" /str(iteration) / "pareto" / "seeds"
        hv_plots_dir = exp_dir / "plots" / "TrialBudget" / str(iteration) / "hypervolume"
    str_to_match = "*.png"
    num_plots = len(list(pareto_plots_dir.rglob(str_to_match)))
    print(f"Found {num_plots} plots.")
    assert num_plots == len(list(hv_plots_dir.rglob(str_to_match))), "Number of plots do not match."


    def plot_subplots(dir: Path, type: Literal["pareto", "hypervolume"]) -> None:
        image_paths = sorted(dir.rglob(str_to_match))
        images = [mpimg.imread(img) for img in image_paths]
        import math
        if math.sqrt(num_plots) % 1 == 0:
            nrows = ncols = int(math.sqrt(num_plots))
        else:
            nrows, ncols = 1, 2
            while num_plots > nrows * ncols:
                ncols += 1
                if ncols > 4:   # noqa: PLR2004
                    nrows += 1
                    ncols = 4
        if ncols <=1 or nrows <=1:
            logger.error("Just one plot found. Exiting.")
            return
        fig, axs = plt.subplots(
            nrows = nrows,
            ncols = ncols,
            figsize=(20, 10),
        )
        axs = axs.flatten() if num_plots > 1 else [axs]
        # print(len(axs))
        for i, img in enumerate(images):
            # print(i)
            axs[i].imshow(img)
            axs[i].axis("off")

        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        save_dir = dir.parent / "subplots"
        save_dir.mkdir(parents=True, exist_ok=True)
        if not no_save:
            plt.savefig(save_dir / f"all_{type}_subplots_{iteration=}.png", dpi = 300)
        plt.show()

    plot_subplots(pareto_plots_dir, "pareto")
    plot_subplots(hv_plots_dir, "hypervolume")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir", "-e",
        type=str,
        required=True,
        help="Main experiment directory containing the runs to plot."
    )
    parser.add_argument(
        "--make_subplots", "-s",
        action="store_true",
        help="make subplot for all plots in the same experiment directory."
    )
    parser.add_argument(
        "--plot_opt", "-o",
        type=str,
        default=None,
        help="Plot all pareto fronts only for the given optimizer."
    )
    parser.add_argument(
        "--plot_for_iterations", "-i",
        type=int,
        nargs="+",
        default=None,
        help="Plot pareto fronts and Hypervolumes for the given iterations."
    )
    parser.add_argument(
        "--no_save", "-ns",
        action="store_true",
        help="Do not save the plots."
    )
    parser.add_argument(
        "--agg_pareto", "-agg",
        action="store_true",
        help="Plot aggregated pareto front over all seeds."
    )
    parser.add_argument(
        "--subplot_budget_type", "-sb_type",
        type=str,
        default=None,
        help="Budget type for subplots. Either TrialBudget or FidelityBudget."
    )
    args = parser.parse_args()
    exp_dir: Path = DEFAULT_RESULTS_DIR / args.exp_dir

    agg_dict: Mapping[str, Mapping[str, Any]] = {}

    if args.make_subplots:
        assert args.plot_for_iterations or args.subplot_budget_type, (
            "Provide iterations to make subplots."
        )
        if args.plot_for_iterations:
            plots_dir = exp_dir / "plots" / str(args.plot_for_iterations[0])
        elif args.subplot_budget_type:
            plots_dir = exp_dir / "plots" / args.subplot_budget_type
        else:
            raise ValueError("Provide either iterations or budget type for subplots.")
        # print(len(list((plots_dir/"hypervolume").rglob("*.png"))))
        match (
            plots_dir.exists(),
            len(list((plots_dir/"hypervolume").rglob("*.png")))
        ):
            case True, 1:
                logger.info("Only one plot found. Exiting.")
            case _, 0:
                agg_data(
                    exp_dir=exp_dir,
                    plot_opt=args.plot_opt,
                    plot_iters=args.plot_for_iterations,
                    no_save=args.no_save,
                    agg_pareto=args.agg_pareto
                )
                make_subplots(
                    exp_dir,
                    args.plot_for_iterations,
                    no_save=args.no_save,
                    budget_type=args.subplot_budget_type
                )
            case _:
                make_subplots(
                    exp_dir,
                    args.plot_for_iterations,
                    no_save=args.no_save,
                    budget_type=args.subplot_budget_type
                )
    else:
        agg_data(
            exp_dir=exp_dir,
            plot_opt=args.plot_opt,
            plot_iters=args.plot_for_iterations,
            no_save=args.no_save,
            agg_pareto=args.agg_pareto
        )
