import time

from base import Experiment, DEFAULT_RUN_CFG
from dreamerv3.main import main


def run_standard_dreamer(run_config: dict = DEFAULT_RUN_CFG) -> None:
    """
    Run the standard DreamerV3 on Atari.
    """

    # First setup experiment instance
    experiment = Experiment(run_cfg=run_config)

    # Convert the flat dictionary to a list of command-line style arguments
    argv = []
    for key, value in run_config.items():
        argv.extend([f'--{key}', str(value)])

    # Now call the main function
    main(argv=argv, experiment=experiment)

    # Save results
    experiment.store(csv_file=f"artifacts/standard_dreamer/results_{time.time()}.csv")

if __name__ == "__main__":
    run_standard_dreamer()
