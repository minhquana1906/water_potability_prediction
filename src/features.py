from pathlib import Path

import mlflow
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

# from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

PROJ_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJ_ROOT / "data/raw"
PROCESSED_DATA_DIR = PROJ_ROOT / "data/processed"

app = typer.Typer()


def load_csv(filepath: Path) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    if not filepath.exists():
        logger.error(f"File {filepath} not found!")
        raise FileNotFoundError(f"File {filepath} not found!")
    return pd.read_csv(filepath)


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with column means."""
    missing_before = df.isnull().sum().sum()
    df_filled = df.fillna(df.mean())
    missing_after = df_filled.isnull().sum().sum()

    logger.info(f"Missing values before: {missing_before}, after: {missing_after}")
    return df_filled


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """Save DataFrame to a CSV file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.success(f"File saved: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save file {filepath}: {str(e)}")
        raise


def log_dataset_info(df: pd.DataFrame, dataset_name: str) -> None:
    """Log dataset statistics to MLflow."""
    with mlflow.start_run(nested=True):
        mlflow.log_param(f"{dataset_name}_num_rows", df.shape[0])
        mlflow.log_param(f"{dataset_name}_num_cols", df.shape[1])
        mlflow.log_metric(f"{dataset_name}_missing_values", df.isnull().sum().sum())
        logger.info(f"Logged {dataset_name} dataset info to MLflow.")


def preprocess(
    train_input_path: Path = RAW_DATA_DIR / "train.csv",
    test_input_path: Path = RAW_DATA_DIR / "test.csv",
    train_output_path: Path = PROCESSED_DATA_DIR / "train_processed.csv",
    test_output_path: Path = PROCESSED_DATA_DIR / "test_processed.csv",
):
    """Preprocess dataset: handle missing values and save outputs."""
    logger.info("Loading datasets...")
    train_data = load_csv(train_input_path)
    test_data = load_csv(test_input_path)

    logger.info("Logging dataset info to MLflow...")
    log_dataset_info(train_data, "train")
    log_dataset_info(test_data, "test")

    logger.info("Filling missing values...")
    train_processed = fill_missing_values(train_data)
    test_processed = fill_missing_values(test_data)

    logger.info("Saving preprocessed datasets...")
    for filepath, df in tqdm(
        zip([train_output_path, test_output_path], [train_processed, test_processed]),
        total=2,
        desc="Saving",
    ):
        save_csv(df, filepath)

    logger.success("Data preprocessing complete!")


@app.command()
def main():
    """Main pipeline: Preprocess data and generate features."""
    logger.info("Starting full pipeline...")
    preprocess()
    logger.success("Pipeline execution complete!")


if __name__ == "__main__":
    app()
