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
