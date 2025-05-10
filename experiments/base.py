"""
Base class used to collect and store experimental results for each run.
"""

import os
import matplotlib.pyplot as plt
from datetime import datetime
import csv
from collections import defaultdict
import numpy as np

        
DEFAULT_RUN_CFG = {
    "logdir": r"~/logdir/{timestamp}", 
    "configs": "atari100k", 
    "run.train_ratio": 32,
    "run.duration": 120,
    "run.steps": 2000,
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
            for metric_key, value in metric_set.items():
                # print(f"Now processing: {metric_key}")
                if (isinstance(value, np.ndarray) and len(value.shape) == 0) or isinstance(value, np.floating):
                    parsed_value = float(value)
                elif isinstance(value, np.ndarray) and all(len(item) == 1 for item in value):
                    parsed_value = map(float, value)
                elif isinstance(value, np.ndarray):
                    parsed_value = self._to_list_recursive(value)
                else:
                    print(f"I could not parse {value}")
                    parsed_value = np.asarray(value)
                concatenated_metrics[metric_key].append(parsed_value)

        print(f"The length of the collected metric sets: {len(self.train_step_metrics)}")
        # is_all_empty = all(len(metric_set.keys()) == 0 for metric_set in self.train_step_metrics)
        # print(f"The metric sets are all empty: {is_all_empty}")

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
        return flattened_dict

    def _to_list_recursive(self, arr: np.ndarray | np.floating) -> list | float:
        if isinstance(arr, np.ndarray):
            return [self._to_list_recursive(item) for item in arr]
        elif isinstance(arr, np.floating):
            return float(arr)
        return arr

    def store(self, csv_file: str, delimiter: str = ";") -> None:
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
                reader = csv.DictReader(file, delimiter=delimiter)
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
            writer = csv.DictWriter(file, fieldnames=all_fields, delimiter=delimiter)
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

