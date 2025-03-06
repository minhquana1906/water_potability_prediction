from pathlib import Path

import pandas as pd
import typer
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# from src.config import RAW_DATA_DIR, DATA_DISK, PARAMS_FILE

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DISK = PROJ_ROOT / "datasets/water_potability.csv"
PARAMS_FILE = PROJ_ROOT / "params.yaml"
RAW_DATA_DIR = PROJ_ROOT / "data/raw"

app = typer.Typer()


def load_yaml(filepath: Path) -> dict:
    """Load configuration parameters from a YAML file."""
    if not filepath.exists():
        logger.error(f"File {filepath} not found!")
        raise FileNotFoundError(f"File {filepath} not found!")
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def load_csv(filepath: Path) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    if not filepath.exists():
        logger.error(f"File {filepath} not found!")
        raise FileNotFoundError(f"File {filepath} not found!")
    return pd.read_csv(filepath)


def split_data(
    data: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and test sets."""
    if not 0 < test_size < 1:
        logger.error(f"Invalid test_size: {test_size}. It must be between 0 and 1.")
        raise ValueError("test_size must be between 0 and 1")
    return train_test_split(data, test_size=test_size, random_state=random_state)


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """Save DataFrame to a CSV file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.success(f"File saved: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save file {filepath}: {str(e)}")
        raise


@app.command()
def main(
    config_path: Path = PARAMS_FILE,
    input_path: Path = DATA_DISK,
    train_output_path: Path = RAW_DATA_DIR / "train.csv",
    test_output_path: Path = RAW_DATA_DIR / "test.csv",
):
    """Load dataset, split into train/test, and save outputs."""
    logger.info("Loading parameters...")
    params = load_yaml(config_path)

    logger.info("Loading dataset...")
    data = load_csv(input_path)

    logger.info("Splitting dataset into train/test...")
    train_data, test_data = split_data(data, params["test_size"], 42)

    logger.info("Saving train/test datasets...")
    for filepath, df in tqdm(
        zip([train_output_path, test_output_path], [train_data, test_data]), total=2, desc="Saving"
    ):
        save_csv(df, filepath)

    logger.success("Data splitting complete!")


if __name__ == "__main__":
    app()
