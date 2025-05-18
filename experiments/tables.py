import pandas as pd
import ast
from typing import Set, Union


def process_experiment_results(
    csv_path: str = "artifacts/results.csv",
    experiment_names: Set[str] = {"Test Run to check functionality"},
    metrics: Set[str] = {"train/loss/policy", "train/loss/value", "train/opt/loss", "train/ret", "train/rew", "duration (s)"},
) -> pd.DataFrame:
    """
    Loads a CSV containing experiment results, filters by experiment names and metrics,
    normalizes metric values (extracting last from lists or using float directly),
    and computes the mean and standard deviation for each metric grouped by experiment name.

    Args:
        csv_path (str): Path to the results.csv file.
        experiment_names (Set[str]): Set of experiment names to filter.
        metrics (Set[str]): Set of metric column names to include in the output.

    Returns:
        pd.DataFrame: DataFrame containing the mean and standard deviation of each metric
                      grouped by experiment name.
    """
    df = pd.read_csv(csv_path, delimiter=";")

    # Filter by experiment names
    df = df[df['experiment_name'].isin(experiment_names)]

    # Keep only experiment name and the relevant metrics
    df = df[['experiment_name'] + list(metrics)]

    # Normalize metric values: take last of list or float
    def normalize(value: Union[str, float]) -> float:
        if isinstance(value, float):
            return value
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list) and parsed:
                return float(parsed[-1])
            else:
                return float(parsed)
        except (ValueError, SyntaxError):
            return float('nan')

    for metric in metrics:
        df[metric] = df[metric].apply(normalize)

    # Group by experiment name and compute mean and std
    grouped = df.groupby('experiment_name').agg(['mean', 'std'])

    # Flatten MultiIndex columns
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped.reset_index(inplace=True)

    grouped.to_csv(csv_path.replace(".csv", "_agg.csv"))
    return grouped


if __name__ == "__main__":
    agg_df = process_experiment_results()
    print("The first 10 lines of the aggregated result Dataframe:\n")
    print(agg_df.head(10))
