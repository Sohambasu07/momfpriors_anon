from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from pymoo.indicators.hv import Hypervolume
from significance_analysis import (
    benchmark_information_check,
    cd_diagram,
    dataframe_validator,
    glrt,
    model,
    seed_dependency_check,
)

from momfpriors.baselines import OPTIMIZERS
from momfpriors.constants import DEFAULT_RESULTS_DIR
from momfpriors.plotting.plot_styles import (
    RC_PARAMS,
    other_fig_params,
)
from momfpriors.plotting.plot_utils import (
    avg_seed_dfs_for_ranking,
    change_opt_names,
    edit_legend_labels,
    fid_perc_momfbo,
    get_style,
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


best_hv_dict = {
    "MOMFPark": 0.0,
    "pd1-cifar100-wide_resnet-2048": 0.0,
    "pd1-imagenet-resnet-512": 0.0,
    "pd1-lm1b-transformer-2048": 0.0,
    "pd1-translate_wmt-xformer_translate-64": 0.0,
    "yahpo-lcbench-126026": 0.0,
    "yahpo-lcbench-146212": 0.0,
    "yahpo-lcbench-168330": 0.0,
    "yahpo-lcbench-168868": 0.0,
}

worst_hv_dict = {
    "MOMFPark": np.inf,
    "pd1-cifar100-wide_resnet-2048": np.inf,
    "pd1-imagenet-resnet-512": np.inf,
    "pd1-lm1b-transformer-2048": np.inf,
    "pd1-translate_wmt-xformer_translate-64": np.inf,
    "yahpo-lcbench-126026": np.inf,
    "yahpo-lcbench-146212": np.inf,
    "yahpo-lcbench-168330": np.inf,
    "yahpo-lcbench-168868": np.inf,
}


def calc_hv_norm_regret(
    *,
    best: float,
    worst: float,
    hv: float,
) -> float:
    """Function to calculate the normalized regret of the hypervolume values."""
    if best == worst:
        return 0.0
    if pd.isna(hv):
        return 1.0
    return (best - hv) / (best - worst)


def prep_results_for_cd(  # noqa: C901, PLR0912, PLR0915
    *,
    agg_data: Mapping[str, Mapping[str, Any]],
    benchmark: str,
    objectives: list[str],
    budget: int,
    fidelity: str | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Function to calculate the dominated hypervolume over
    iterations and pareto fronts from a pandas Series
    of Results, i.e., Mapping[str, float] objects.
    """
    print("================================================")
    logger.info(f"Benchmark: {benchmark}")

    seed_means_dict = {}

    reference_point = np.array([
        reference_points_dict[benchmark][obj]
        for obj in objectives
    ])
    for instance, instance_data in agg_data.items():

        # Get the marker, color, optimizer name and prior annotations for the optimizer instance
        marker, color, opt, hps, prior_annot = get_style(instance)

        assert opt in OPTIMIZERS

        continuations = False
        logger.info(f"Calculating Hypervolume for instance: {instance}")
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
                hv = Hypervolume(ref_point=reference_point)
                hypervolume = hv.do(pareto)
                hv_vals.append(hypervolume)

            budget_list = np.arange(1, num_full_evals + 1, 1)

            best_hv_dict[benchmark] = max(best_hv_dict[benchmark], *hv_vals)
            worst_hv_dict[benchmark] = min(worst_hv_dict[benchmark], *hv_vals)

            if continuations:
                seed_cont_dict[seed] = pd.Series(hv_vals, index=budget_list)
                seed_cont_dict[seed] = seed_cont_dict[seed].cummax()
                seed_cont_dict[budget] = seed_cont_dict[seed].iloc[-1]
                seed_cont_dict[seed] = seed_cont_dict[seed].ffill()
                instance_name = (
                    f"{opt}_w_continuations" +
                    (";" + hps if hps is not None else "") +
                    (";priors=" + prior_annot if prior_annot is not None else "")
                )
            else:
                seed_hv_dict[seed] = pd.Series(hv_vals, index=budget_list)
                seed_hv_dict[seed] = seed_hv_dict[seed].cummax()
                seed_hv_dict[budget] = seed_hv_dict[seed].iloc[-1]
                seed_hv_dict[seed] = seed_hv_dict[seed].ffill()
                instance_name = instance

            if seed not in seed_means_dict:
                seed_means_dict[seed] = {}
            if continuations:
                seed_means_dict[seed][instance_name] = seed_cont_dict[seed]
            else:
                seed_means_dict[seed][instance_name] = seed_hv_dict[seed]

    seed_means_dict = avg_seed_dfs_for_ranking(
        seed_means_dict
    )
    rank_means_dict: dict[str, pd.DataFrame] = {}
    seed_dict: dict[str, pd.DataFrame] = {}
    # Create a DataFrame from the seed means dict
    for seed, instances in seed_means_dict.items():
        _rankdf = pd.DataFrame(instances)
        _rankdf = _rankdf.ffill(axis=0)
        _seeddf = _rankdf.copy(deep=True)
        seed_dict[seed] = _seeddf
        _rankdf = _rankdf.rank(axis=1, method="average", ascending=True)
        rank_means_dict[seed] = _rankdf

    return seed_dict, rank_means_dict


def extract_iteration_data(
    *,
    means_dict: Mapping[str, Mapping[str, pd.DataFrame]],
    iteration: int | None = None,
    normalize_hv: bool = False,
    which_labels: str = "sig",
):
    """Extracts data for a specific iteration from the nested dict."""
    rows = []

    for seed, benchmark_dict in means_dict.items():

        for benchmark, df in benchmark_dict.items():
            if iteration:
                assert iteration < len(df)
                iteration_rows = [df.iloc[iteration]]
            else:
                iteration_rows = df.to_dict(orient="records")

            for row in iteration_rows:

                for optimizer in df.columns:
                    prior_annot = optimizer.split("priors=")[-1] if "priors=" in optimizer else None
                    hypervolume = row[optimizer]

                    opt = edit_legend_labels(
                        labels=[optimizer],
                        which_labels=which_labels,
                    )[0]

                    hv_regret = calc_hv_norm_regret(
                        best=best_hv_dict[benchmark],
                        worst=worst_hv_dict[benchmark],
                        hv=hypervolume
                    )
                    hv = hv_regret if normalize_hv else hypervolume

                    rows.append({
                        "algorithm": opt,
                        "benchmark": benchmark,
                        "value": hv,
                        "seed": str(seed),
                        "prior": prior_annot,
                    })

    return pd.DataFrame(rows)



def significance_analysis(
    data: pd.DataFrame
) -> None:
    """Function to perform significance analysis on the data."""
    # 1. Generate/import dataset
    data = dataframe_validator(data)[0]

    # 2. Run the preconfigured sanity checks
    seed_dependency_check(data)
    benchmark_information_check(data)

    mod=model("value ~ algorithm + (1|benchmark)", data)
    post_hoc_results=mod.post_hoc("algorithm")

    # Plotting the results
    cd_diagram(
        result = post_hoc_results,
    )

    plt.show()

    # 3. Run a custom hypoXXXX test, comparing model_1 and model_2
    model_1 = model("value ~ algorithm", data)
    model_2 = model("value ~ 1", data)
    glrt_res = glrt(model_1, model_2)
    logger.info(f"GLRT results: {glrt_res}")

def agg_data(
    exp_dir: Path,
    skip_opt: list[str] | None = None,
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

    with (exp_dir / "exp.yaml").open("r") as f:
        exp_config = yaml.safe_load(f)

    all_benches = [(bench.pop("name"), bench) for bench in exp_config["benchmarks"]]

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
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
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
            _df[OPTIMIZER_COL][0]
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
                "results": _results,
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
            "results": _results,
        }
    assert len(objectives) > 0, "Objectives not found in results file."


    return prep_results_for_cd(
        agg_data=agg_dict,
        benchmark=benchmark,
        budget=total_budget,
        fidelity=fidelity,
        objectives=objectives,
    )



def make_subplots(  # noqa: C901, PLR0913
    exp_dir: Path,
    *,
    at_iteration: int | None = None,
    no_save: bool = False,
    save_suffix: str = "",
    skip_opt: list[str] | None = None,
    priors_to_avg: list[str] | None = None,
    skip_non_avg: bool = False,
    skip_priors: bool = False,
    avg_prior_label: str = "all",
    file_type: str = "pdf",
    output_dir: Path | None = None,
    which_labels: str = "sig",
    plot_title: str | None = None,
    normalize_hv: bool = False,
) -> None:
    """Function to make subplots for all plots in the same experiment directory."""
    fig_size = other_fig_params["fig_size"]
    benchmarks_dict, total_budget = agg_data(exp_dir, skip_opt=skip_opt)
    num_plots = len(benchmarks_dict)
    nrows, ncols = other_fig_params["n_rows_cols"][num_plots]

    if at_iteration and at_iteration > total_budget:
        raise ValueError(
            f"Iteration for Critical Difference plots ({at_iteration}) "
            f"cannot be greater than total budget ({total_budget})."
        )

    # CD plot
    fig_cd, axs_cd = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=fig_size,
    )
    suptitle_x = other_fig_params["suptitle_bbox"][0]
    suptitle_y = other_fig_params["suptitle_bbox"][1]

    # Set the title of the plots
    if plot_title is not None:
        fig_cd.suptitle(
            plot_title,
            x=suptitle_x,
            y=suptitle_y,
            fontsize=other_fig_params["title_fontsize"])

    plt.rcParams.update(RC_PARAMS)

    axs_cd: list[plt.Axes] = axs_cd.flatten() if num_plots > 1 else [axs_cd]

    means_dict = {}
    bench_dict = {}
    for i, (benchmark, conf_dict) in enumerate(benchmarks_dict.items()):
        assert len(conf_dict) == 1, (
            "Plotting more than 1 config for a benchmark not yet implemented."
            f" Found {len(conf_dict)} configs for benchmark {benchmark}."
        )
        for conf_tuple, _all_dfs in conf_dict.items():
            seed_dict_per_bench, ranked_seed_dict_per_bench = gen_plots_per_bench(
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

            axs_cd[i].set_title(benchmark)

            for _seed, rank_df in seed_dict_per_bench.items():
                if _seed not in means_dict:
                    means_dict[_seed] = {}
                means_dict[_seed][benchmark] = rank_df
                if _seed not in bench_dict:
                    bench_dict[_seed] = {}
                bench_dict[_seed][benchmark] = rank_df


            bench_dict = {}

    # Aggregate the data into a single DataFrame for the given iteration
    agg_df = extract_iteration_data(
        means_dict=means_dict,
        iteration=at_iteration,
        normalize_hv=normalize_hv,
        which_labels=which_labels,
    )
    # Perform significance analysis on the aggregated data
    significance_analysis(agg_df)

    # Remove empty subplots for CD plot
    for j in range(i + 1, len(axs_cd)):
        fig_cd.delaxes(axs_cd[j])

    tight_layout_pads = other_fig_params["tight_layout_pads"]

    fig_cd.tight_layout(**tight_layout_pads)

    save_dir = exp_dir/ "plots" / "subplots"
    if output_dir is not None:
        save_dir = output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    if save_suffix:
        save_suffix = f"_{save_suffix}"
    if not no_save:
        fig_cd.savefig(
            save_dir / f"cd_subplots{save_suffix}.{file_type}",
            dpi=300, bbox_inches="tight"
        )
        logger.info("Saved CD plot")


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
        "--at_iteration", "-i",
        type=int,
        default=None,
        help="Plot CD at a specific iteration."
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
        default="sig",
        help="Which labels to use for the plots."
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
        "--plot_title", "-title",
        type=str,
        default="",
        help="Title for the plots. If not provided, no title will be set."
    )
    parser.add_argument(
        "--normalize_hv", "-norm",
        action="store_true",
        help="Normalize regret for the hypervolume values for the LMEM significance analysis."
    )
    args = parser.parse_args()

    if args.from_yaml:
        with Path(args.from_yaml).open("r") as f:
            yaml_config = yaml.safe_load(f)
        args.exp_dir = yaml_config.get("exp_dir", args.exp_dir)
        args.output_dir = yaml_config.get("output_dir", args.output_dir)
        args.at_iteration = yaml_config.get("at_iteration", args.at_iteration)
        args.no_save = yaml_config.get("no_save", args.no_save)
        args.save_suffix = yaml_config.get("save_suffix", args.save_suffix)
        args.skip_opt = yaml_config.get("skip_opt", args.skip_opt)
        args.priors_to_avg = yaml_config.get("priors_to_avg", args.priors_to_avg)
        args.skip_non_avg = yaml_config.get("skip_non_avg", args.skip_non_avg)
        args.skip_priors = yaml_config.get("skip_priors", args.skip_priors)
        args.avg_prior_label = yaml_config.get("avg_prior_label", args.avg_prior_label)
        args.file_type = yaml_config.get("file_type", args.file_type)
        args.which_labels = yaml_config.get("which_labels", args.which_labels)
        args.specific_rc_params = yaml_config.get("specific_rc_params", args.specific_rc_params)
        args.specific_fig_params = yaml_config.get("specific_fig_params", args.specific_fig_params)
        args.plot_title = yaml_config.get("plot_title", args.plot_title)
        args.normalize_hv = yaml_config.get("normalize_hv", args.normalize_hv)

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

    make_subplots(
        exp_dir=exp_dir,
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
        plot_title=args.plot_title,
        at_iteration=args.at_iteration,
        normalize_hv=args.normalize_hv
    )
