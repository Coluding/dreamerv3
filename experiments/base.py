"""
Base class used to collect and store experimental results for each run.
"""

import os
import matplotlib.pyplot as plt
from datetime import datetime
import csv
from collections import defaultdict

        
DEFAULT_RUN_CFG = {
    "logdir": r"~/logdir/{timestamp}", 
    "configs": "atari100k", 
    "run.train_ratio": 32,
    "run.duration": 60,
    "run.steps": 1000,
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

        self.run_cfg = run_cfg
        self.config = {} # Later added once the experiment is initialized.
        self.train_step_metrics = []
        self.model_name = model_name

    def flatten(self):
        concatenated_metrics = defaultdict(list) # Will contain a list of the metrics from each step.
        for metric_set in self.train_step_metrics:
            for metric_key, value in metric_set:
                concatenated_metrics[metric_key].append(value)

        # Combine all necessary experiment information into a flattened dictionary
        flattened_dict = {
            "ID": self.ID,
            "run_config": str(self.run_cfg),
            "config": str(self.config),
            "model_name": self.model_name,
            "PID": os.getpid(),
            "timestamp": datetime.now(),
            **concatenated_metrics,
        }
        # TODO: Check why the collected metrics are not stored in the CSV
        return flattened_dict

    def store(self, csv_file: str) -> None:
        """
        Stores the flattened experiment data in a CSV file. If the file does not exist, it creates it.
        If the data contains new metrics (columns not in the existing file), it updates the header and
        rewrites the file with the additional columns.
        """

        # Ensure the directory exists
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)

        flattened_data = self.flatten()
        new_fields = list(flattened_data.keys())
        rows = []

        if os.path.exists(csv_file):
            with open(csv_file, mode="r", newline="") as file:
                reader = csv.DictReader(file)
                existing_fields = reader.fieldnames or []
                rows = list(reader)
        else:
            existing_fields = []

        # Merge fieldnames and ensure order
        all_fields = list(dict.fromkeys(existing_fields + new_fields))

        # Add the new data
        rows.append(flattened_data)

        # Rewrite the file with updated fieldnames and all data
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=all_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in all_fields})

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

