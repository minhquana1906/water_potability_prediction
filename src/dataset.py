from pathlib import Path
import typer
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from loguru import logger
from tqdm import tqdm

from src.config import (
    RAW_DATA_DIR,
    DATA_DISK,
    PARAMS_FILE,
)

app = typer.Typer()


def load_params(filepath: Path) -> dict:
    """Load configuration parameters from a YAML file."""
    try:
        with open(filepath, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filepath} not found! Cannot load parameters.")


def load_data(filepath: Path) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filepath} not found! Cannot load data.")


def split_data(
    data: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and test sets."""
    try:
        return train_test_split(data, test_size=test_size, random_state=random_state)
    except ValueError:
        raise ValueError("Invalid parameters for train_test_split!")


def save_data(df: pd.DataFrame, filepath: Path) -> None:
    """Save DataFrame to a CSV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    df.to_csv(filepath, index=False)


@app.command()
def main(
    config_path: Path = PARAMS_FILE,
    input_path: Path = DATA_DISK,
    train_output_path: Path = RAW_DATA_DIR / "train.csv",
    test_output_path: Path = RAW_DATA_DIR / "test.csv",
):
    """Load dataset, split into train/test, and save outputs."""
    logger.info("Loading parameters...")
    params = load_params(config_path)

    logger.info("Loading dataset...")
    data = load_data(input_path)

    logger.info("Splitting dataset into train/test...")
    train_data, test_data = split_data(data, params["data_ingestion"]["test_size"], 42)

    logger.info("Saving train/test datasets...")
    for filepath, df in tqdm(
        zip([train_output_path, test_output_path], [train_data, test_data]), total=2, desc="Saving"
    ):
        save_data(df, filepath)

    logger.success("Data splitting complete!")


if __name__ == "__main__":
    app()
