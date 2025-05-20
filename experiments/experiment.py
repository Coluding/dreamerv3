from experiments.base import Experiment
from experiments.presets import *
from dreamerv3.main import main
from typing import Iterable


RESULTS_CSV_PATH = "artifacts/results.csv"
DEFAULT_DATASETS = {"atari100k"}
ATARI_TASKS = {"atari100k_boxing", "atari100k_krull"}


def run_experiment(
        run_config: dict = DEFAULT_RUN_CFG, 
        name: str = "Unnamed Experiment", 
        description: str = "Undescriped Experiment",
        num_seeds: int = 1,
        results_csv_path: str = RESULTS_CSV_PATH,
        datasets: Iterable = DEFAULT_DATASETS,
        tasks: Iterable = ATARI_TASKS,
    ) -> None:
    """
    Run an experiment.
    """

    for seed in range(num_seeds):
        for dataset in datasets:
            if tasks is not None:
                for task in tasks:
                    make_run(
                        run_config,
                        name,
                        description,
                        seed,
                        results_csv_path,
                        dataset,
                        task
                    )

def make_run(
        run_config: dict = DEFAULT_RUN_CFG, 
        name: str = "Unnamed Experiment", 
        description: str = "Undescriped Experiment",
        seed: int = 1,
        results_csv_path: str = RESULTS_CSV_PATH,
        dataset: str = None,
        task: str = None
    ) -> None:

    print(f"Starting run {seed+1} for dataset {dataset}")
    run_config["configs"] = dataset
    if task:
        run_config["task"] = task
    
    # First setup experiment instance
    experiment = Experiment(
        run_cfg=run_config,
        experiment_name=name,
        experiment_description=description,
    )

    # Convert the flat dictionary to a list of command-line style arguments
    argv = []
    for key, value in run_config.items():
        argv.extend([f'--{key}', str(value)])
    argv.extend(['--seed', str(seed)])

    # Now call the main function
    main(argv=argv, experiment=experiment)

    # Save results
    experiment.store(csv_file=results_csv_path)
    print(f"Finished run {seed+1} for dataset {dataset}")



def run_standard_dreamer(
        run_config: dict = DEFAULT_RUN_CFG, 
        name: str = "Standard Dreamer", 
        description: str = "Standard Dreamer Experiment from the Readme but with Atari100k instead.",
        num_seeds: int = 1,
        datasets: Iterable = DEFAULT_DATASETS,
    ) -> None:
    """
    Run the standard DreamerV3 on Atari.
    """

    run_experiment(run_config, name, description, num_seeds, datasets=datasets)


def run_replay_buffer_experiment(
        run_config: dict = REPLAY_BUFFER_RUN_CFG,
        name: str = "Optimized Replay Buffer", 
        description: str = "Using a prioritized replay buffer.",
        num_seeds: int = 2,
        datasets: Iterable = DEFAULT_DATASETS,
    ) -> None:
    """
    Running the Dreamer with optimized replay buffer.
    """

    run_experiment(run_config, name, description, num_seeds, datasets=datasets)


def run_latent_disagreement_experiment(
        run_config: dict = REPLAY_LATENT_DISAGREEMENT_CFG,
        name: str = "Latent Disagreement", 
        description: str = "Using the latent disagreement method to guide exploration - the mean+variance variant.",
        num_seeds: int = 2,
        datasets: Iterable = DEFAULT_DATASETS,
    ) -> None:
    """
    Running the Dreamer with latent disagreement.
    """

    run_experiment(run_config, name, description, num_seeds, datasets=datasets)



if __name__ == "__main__":
    import argparse
    import inspect
    
    # Get all experiment functions
    experiment_functions = {
        name: func for name, func in globals().items() 
        if name.startswith('run_') and callable(func)
    }
    
    # Create parser
    parser = argparse.ArgumentParser(description='Run DreamerV3 experiments')
    parser.add_argument('experiment', choices=list(experiment_functions.keys()),
                        help='Name of experiment function to run')
    
    # Add arguments for the selected experiment
    args, unknown = parser.parse_known_args()
    selected_func = experiment_functions[args.experiment]
    sig = inspect.signature(selected_func)
    
    # Add parameters for the selected function
    for param_name, param in sig.parameters.items():
        if param_name == 'run_config':
            parser.add_argument('--config', dest='run_config',
                               help='Name of config from presets.py (case insensitive)')
        else:
            parser.add_argument(f'--{param_name}', 
                               type=type(param.default) if param.default is not inspect.Parameter.empty else str,
                               default=param.default if param.default is not inspect.Parameter.empty else None)
    
    # Parse all arguments
    args = parser.parse_args()

    # Handle run_config specially - match against presets
    if hasattr(args, 'run_config') and args.run_config:
        # Get all uppercase configs from presets
        preset_configs = {k.upper(): v for k, v in globals().items() 
                         if k.endswith('_CFG') and isinstance(v, dict)}
        
        # Try to match the provided config name
        config_name = args.run_config.upper()
        matched_config = None
        
        for preset_name, preset_value in preset_configs.items():
            if preset_name.startswith(config_name) or config_name in preset_name:
                matched_config = preset_value
                print(f"Using config: {preset_name}")
                break
        
        if matched_config:
            args.run_config = matched_config
        else:
            print(f"Warning: No matching config found for '{args.run_config}'")
            print(f"Available configs: {list(preset_configs.keys())}")
    else:
        # If no config provided, use the default from the function signature
        func_defaults = {
            name: func.__defaults__[list(inspect.signature(func).parameters.keys()).index('run_config')] 
            for name, func in experiment_functions.items()
        }
        args.run_config = func_defaults[args.experiment]
        print(f"Using default config for {args.experiment}")

    # Call the selected experiment function with the parsed arguments
    kwargs = {k: v for k, v in vars(args).items() if k != 'experiment'}
    experiment_functions[args.experiment](**kwargs)
