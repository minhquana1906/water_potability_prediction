from pathlib import Path
import typer
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def load_data(filepath: Path) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filepath} not found! Cannot load data.")


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with column mean."""
    return df.fillna(df.mean())


def save_data(df: pd.DataFrame, filepath: Path) -> None:
    """Save DataFrame to a CSV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    df.to_csv(filepath, index=False)


def preprocess(
    train_input_path: Path = RAW_DATA_DIR / "train.csv",
    test_input_path: Path = RAW_DATA_DIR / "test.csv",
    train_output_path: Path = PROCESSED_DATA_DIR / "train_processed.csv",
    test_output_path: Path = PROCESSED_DATA_DIR / "test_processed.csv",
):
    """Preprocess dataset: handle missing values and save outputs."""
    logger.info("Loading datasets...")
    train_data = load_data(train_input_path)
    test_data = load_data(test_input_path)

    logger.info("Filling missing values...")
    train_processed = fill_missing_values(train_data)
    test_processed = fill_missing_values(test_data)

    logger.info("Saving preprocessed datasets...")
    for filepath, df in tqdm(
        zip([train_output_path, test_output_path], [train_processed, test_processed]),
        total=2,
        desc="Saving",
    ):
        save_data(df, filepath)

    logger.success("Data preprocessing complete!")


@app.command()
def main():
    """Main pipeline: Preprocess data and generate features."""
    logger.info("Starting full pipeline...")
    preprocess()
    logger.success("Pipeline execution complete!")


if __name__ == "__main__":
    app()
