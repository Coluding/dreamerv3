# How to use the Experimental Framework
The experimental framework is designed to serve as a single point of entry for running experiments in a well-documented and structured way - to avoid that information gets lost. It also allows to create aggregated tables for use in a paper.
<br>
<br>

## Running Experiments
The experiments/experiment_definitions.py package serves as a CLI for running experiments.
The first argument is the name of the experiment function and the following arguments can be function arguments that should be passed to the experiment functions defined in experiment_definitions.py. The name of the run config is case insensitive. The structure works as follows:
```
python experiments/experiment_definitions.py experiment_function_name --optional_function_argument value
```
For instance, to run the standard experiment from the DreamerV3 Readme page:
```
python experiments/experiment_definitions.py run_standard_dreamer --name "Test Run to check functionality" --description "Just a run with 2 seeds for testing purposes" --num_seeds 2
```
<br>

## Accessing the Results
All results are stored in `dreamerv3/artifacts/results.csv`. It contains the content of the config file, the run config (preset) and all training metrics, logged at every single step.
<br>
<br>

## Creating Tables
To create tables that are aggregated over several runs of the same experiment using different seeds, you can use the tables CLI. To create a table from the results CLI, run:
```
python experiments/tables.py
```
To include/exclude metrics from the table, modify the default argument of the `process_experiment_results` function in `tables.py`. To include experiments, add/remove the names of the experiments from the `experiment_names` default argument set. The result is printed to the commandline.
<br>

# Custom Plotting Tool 

The `custom_plot.py` script provides visualization capabilities for experiment results, supporting both score metrics and training losses.

### Basic Usage

```bash
python custom_plot.py --logdir path/to/logs/ --outdir plots/
```

### Key Features

- Automatically discovers and groups runs by method, game, and seed
- Plots individual runs and statistical aggregates (mean, median)
- Supports multiple metrics visualization (scores and various loss types)
- Auto-scales y-axis based on data range (log scale for loss metrics)

### Options

```bash
# Filter by specific methods 
python custom_plot.py --method_filter default latent_reward_disagreement

# Specify custom metrics to plot
python custom_plot.py --metrics train/loss/rew train/loss/value

# Include self-normalized statistics
python custom_plot.py --stats mean self_mean
```

### Tips

- When using `--method_filter`, separate values with spaces (not commas)
- Use `--auto_log_scale False` to disable automatic log scaling for loss metrics
- Use `--method_filter all` to include all methods (default behavior)
