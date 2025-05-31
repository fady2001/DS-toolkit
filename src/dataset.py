from pathlib import Path
from typing import Dict

import pandas as pd


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load a dataset from a CSV file.

    Args:
        dataset_path (str): The path to the dataset CSV file.

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    dataset_path: Path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    return pd.read_csv(dataset_path)


def handle_missing_date_indicies(
    df: pd.DataFrame,
    date_column: str = "date",
    fill_strategy: Dict[str, Dict[str, Dict[str,str | float]]] = {
        "dcoilwtico": {"interpolate": {"method": "polynomial", "order": 2}}
    },
) -> pd.DataFrame:
    """
    Handle missing date indices in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame with a 'date' column.
        date_column (str): The name of the date column in the DataFrame.
        fill_strategy (Dict[str, Dict[str, Dict[str,str | float]]]): A dictionary defining how to fill missing values
            for each column. The keys are column names and the values are dictionaries with strategies like 'interpolate',
            'bfill', 'ffill', or 'fillna'.
    Raises:
        ValueError: If the DataFrame does not contain the specified date column.
    Returns:
        pd.DataFrame: The DataFrame with missing date indices filled.
    """
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column.")

    all_dates: pd.DatetimeIndex = pd.date_range(start=df[date_column].min(), end=df[date_column].max())
    df: pd.DataFrame = df.set_index(date_column).reindex(all_dates).rename_axis(date_column).reset_index()

    for column, strategy in fill_strategy.items():
        if column in df.columns:
            if strategy == "interpolate":
                df[column] = df[column].interpolate(
                    **strategy.get("interpolate", {"method": "polynomial", "order": 2})
                )
            elif strategy == "bfill":
                df[column] = df[column].bfill()
            elif strategy == "ffill":
                df[column] = df[column].ffill()
            elif strategy == "fillna":
                df[column] = df[column].fillna(strategy.get("fillna", 0))
            else:
                raise ValueError(f"Unknown fill strategy: {strategy}")
    return df

def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """
    Save a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        output_path (str): The path where the DataFrame will be saved as a CSV file.
    """
    output_path: Path = Path(output_path)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")