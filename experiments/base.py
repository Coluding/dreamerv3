"""
Base class used to collect and store experimental results for each run.
"""

import os
import matplotlib.pyplot as plt
from datetime import datetime
import csv
        
        
DEFAULT_RUN_CFG = {
    "logdir": r"~/logdir/{timestamp}", 
    "configs": "atari100k", 
    "run.train_ratio": 32,
}


class Experiment:
    # Class-level variable to keep track of the experiment count across runtimes
    _experiment_count = 0

    def __init__(self, run_cfg: dict = DEFAULT_RUN_CFG, model_name: str = "DreamerV3"):
        """
        Class used to collect and store information on one run.
        """

        # Increment the experiment count to generate a unique ID
        self.ID = Experiment._experiment_count
        Experiment._experiment_count += 1

        self.train_loss_mse = []
        self.test_loss_mse = []
        self.run_cfg = run_cfg
        self.config = {} # Later added once the experiment is initialized.
        self.model_name = model_name

    def flatten(self):
        # Combine all necessary experiment information into a flattened dictionary
        flattened_dict = {
            "ID": self.ID,
            "train_loss_mse": str(self.train_loss_mse),
            "test_loss_mse": str(self.test_loss_mse),
            "final_epoch_train_mse": self.train_loss_mse[-1],
            "final_epoch_test_mse": self.test_loss_mse[-1],
            "run_config": str(self.run_cfg),
            "config": str(self.config),
            "model_name": self.model_name,
            "PID": os.getpid(),
            "timestamp": datetime.now(),
        }
        return flattened_dict

    def store(self, csv_file):
        # Append the flattened experiment data to a CSV file
        flattened_data = self.flatten()
        file_exists = os.path.exists(csv_file)
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=flattened_data.keys())
            if not file_exists:
                writer.writeheader()  # Write header if the file is empty
            writer.writerow(flattened_data)

    def plot_loss(self):
        # Plot the training and test loss over epochs using matplotlib
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_loss, label="Training Loss", marker="o")
        plt.plot(self.test_loss, label="Test Loss", marker="x")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Plot for Experiment {self.ID}")
        plt.legend()
        plt.grid(True)
        plt.show()


