# Multi-objective Multi-fidelity Optimization with Priors

#### Package name: `momfpriors`

## Minimal example to Run experiments

We use `YAML` configuration files to run the experiments using `momfpriors`. [Here](configs/exp_configs/minimal.yaml) we provide a minimal example for an experiment configuration.

The following `bash` command runs the minimal example:

```bash

python3 -m momfpriors.run -y "configs/exp_configs/minimal.yaml" -e "minimal_ex"
```

## Installation

### Create a Virtual Environment using Venv
```bash
python -m venv momfp_env
source momfp_env/bin/activate
```

### Installation from source

```bash
git clone XXXX
cd XXXX

pip install -e . # -e for editable install
```

> [!TIP]
> * `pip install .["all"]` - To install momfpriors with all optimizers and benchmarks
> * `pip install .["optimizers"]` - To install momfpriors with all available optimizers only
> * `pip install .["benchmarks"]` - To install momfpriors with all available benchmarks only
> * Additionally do `pip install .[""significance-analysis"]` to run Statistical Significance analysis using LMEMs.


## Generating Priors

For generating priors, we use the following CLI command:

```bash
python -m momfpriors.generate_priors \
    --yaml <abspath to prior generation config> \
    --to <abspath of dir to store the priors> \
    --nsamples <number of random samples to use> \ # optional, default=10
    --seed <random seed for reproducibility> \

```

Here's an example for a prior generation `YAML` config:

```yaml
benchmarks:
  pd1-cifar100-wide_resnet-2048:
    - valid_error_rate
    - train_cost
  pd1-imagenet-resnet-512:
    - valid_error_rate
    - train_cost
  pd1-lm1b-transformer-2048:
    - valid_error_rate
    - train_cost
  pd1-translate_wmt-xformer_translate-64:
    - valid_error_rate
    - train_cost
prior_spec:
  - [good, 0, 0.01, null]
  - [medium, 0, 0.125, null]
  - [bad, -1, 0, null]
```

## Plotting

### Main Plots

We generate 4 different plots from our experiments:
* Dominated Hypervolume against function evaluations: `"hv"`
* Pareto front: `"pareto"`
* Relative rankings per benchmark: `"rank"`
* Overall relative rankings: `"ovrank"`

The following CLI command runs the main plotting script:

```bash
python -m momfpriors.plotting.plot_main \
    --from_yaml <yaml config abspath> \
    # The following major arguments can also be provided in the plot config yaml
    --exp_dir <abspath of the exp folder>
    --output_dir <abspath of dir to save the plots> \ # optional
    --file_type <file_type for saving the plot> \ # optional, choices: ["pdf", "png", "jpg"].
    --which_plots <list of plots to generate> \ # optional, choices: ["hv", "pareto", "rank", "ovrank"]. If not provided, all plots are generated
```

`--output_dir` by default is `exp_dir/ "plots" / "subplots"`

### Critical Difference diagrams

```bash
python -m "momfpriors.plotting.cd_plots" \
    --from_yaml <yaml config abspath> \
    --at_iteration <iteration to use for significance testing>
    --normalize_hv <whether to calculate normalized hypervolume regret>
    # The following major arguments can also be provided in the plot config yaml
    --exp_dir <abspath of the exp folder>
    --output_dir <abspath of dir to save the plots> \ # optional
    --file_type <file_type for saving the plot> \ # optional, choices: ["pdf", "png", "jpg"].
```



> [!TIP]
> * We also use `YAML` configs to make plotting a lot easier than writing cumbersome CLI commands (which is also possible, but not the preferred way).
> * Check the `momfpriors/plotting/plot_main.py` for more information about the arguments for the main plotting script.
> * We also have a `merge_rank_plots.py` script which merges relative ranking plots from multiple different single plot configs.

Here is an example plot config:
```yaml
exp_dir: example_plots
output_dir: /home/user/momfpriors_plots/
skip_priors:
  - bad-good
  - bad-bad
save_suffix: ex_plots_good-good
skip_opt:
  - RandomSearchWithPriors
which_labels: "1"
which_plots:
  - hv
  - pareto
specific_fig_params:
  - xylabel_fontsize=20
  - ovrank_xsize=6
  - ovrank_ysize=4
  - bbox_to_anchor=(0.5, -0.08)
```


# Reproducibility Guide

## Experimental Seeds

To ensure reproducible results across experiments, our code uses a deterministic seed generation function. All experiments use a global seed value of `GLOBAL_SEED = 42` to generate the required number of seeds to actually run the experiments.

### Generated Seeds (n=25)

The following 25 seeds are automatically generated for the experiments in the XXXX:

| Seed Number | Seed Value | Seed Number | Seed Value | Seed Number | Seed Value |
|-------|-----------|-------|-----------|-------|-----------|
| 1     | 383329928 | 10    | 404488629 | 19    | 3606691182 |
| 2     | 3324115917| 11    | 2261209996| 20    | 1934392873 |
| 3     | 2811363265| 12    | 4190266093| 21    | 2148995113 |
| 4     | 1884968545| 13    | 3160032369| 22    | 1592565387 |
| 5     | 1859786276| 14    | 3269070127| 23    | 784044719 |
| 6     | 3687649986| 15    | 3081541440| 24    | 3980425318 |
| 7     | 369133709 | 16    | 3376120483| 25    | 3356806662 |
| 8     | 2995172878| 17    | 2204291347|       |           |
| 9     | 865305067 | 18    | 550243862 |       |           |

## Output Data Structure

Each optimization run generates a `pandas.DataFrame` containing comprehensive experimental results and metadata. The DataFrame follows a standardized schema with the following columns:

| Column | Description |
|--------|-------------|
| `budget_cost` | Optimization budget cost. For BB optimizers, this is 1 every iteration; for MF optimizers we take a *fractional* cost: `z/z_max`, where `z` is the fidelity at which a config is evaluated and `z_max` is the *maximum fidelity* available in the benchmark. |
| `budget_used_total` | Cumulative addition of `budget_cost` at every iteration. |
| `continuations_budget_cost` | Optimization budget cost in a *continual* setup, i.e., for configs at a higher fidelity (`z'`) than already evaluated (`z`), the effective cost is `(z'-z)/z_max`. |
| `continuations_budget_used_total` | Cumulative addition of `continuations_budget_cost` at every iteration. |
| `continuations_cost` | Direct fidelity cost in a *continual* optimization run: `z'-z`. |
| `fidelity` | Fidelity at which the current config is evaluated.|
| `config_id` | Unique identifier for the configuration |
| `config` | Complete configuration dictionary. |
| `results` | Dictionary containing the result values of the evaluated config. |
| `seed` | Random seed used for this specific run. |
| `budget_total` | Maximum available budget for the run. |
| `optimizer` | Name of the optimizer used. |
| `optimizer_hyperparameters` | Hyperparameter settings for the optimizer. |
| `benchmark` | Benchmark name. |
| `prior_annotations` | Any prior knowledge or annotations, e.g.: `good-good` signifies good priors for a 2-objective MO `hpoglue` `Problem`. |
| `objectives` | Optimization objectives. |
| `minimize` | Dict of booleans indicating minimization vs maximization for each objective. |
| `problem.fidelity.count` | Number of fidelity dimensions in the `Problem`. |
| `problem.fidelity.1.name` | Name of the first fidelity dimension in the `Problem`. |
| `problem.fidelity.1.min` | Minimum value for first fidelity dimension in the `Problem`. |
| `problem.fidelity.1.max` | Maximum value for first fidelity dimension in the `Problem`. |
| `benchmark.fidelity.count` | Number of benchmark fidelity dimensions. |
| `benchmark.fidelity.1.name` | Name of the first benchmark fidelity dimension. |
| `benchmark.fidelity.1.min` | Minimum value for the first fidelity dimension in the benchmark.|
| `benchmark.fidelity.1.max` | Maximum value for the first fidelity dimension in the benchmark. |

---