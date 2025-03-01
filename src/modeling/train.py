from pathlib import Path
import os
import pickle
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
import typer
from loguru import logger
from tqdm import tqdm

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, PARAMS_FILE

app = typer.Typer()


def load_params(filepath: Path) -> dict:
    """Load hyperparameters from a YAML file."""
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


def prepare_data(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataset into features and target."""
    try:
        return df.drop(columns=[target_col]), df[target_col]
    except KeyError:
        raise KeyError(f"Column '{target_col}' not found in the dataset!")


def train_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> RandomForestClassifier:
    """Train a Random Forest model."""
    logger.info("Training model...")
    model = RandomForestClassifier(
        n_estimators=params["model_train"]["n_estimators"],
        max_depth=params["model_train"]["max_depth"],
        random_state=params["model_train"].get("random_state", 42),
    )
    model.fit(X_train, y_train)
    logger.success("Model training complete.")
    return model


def save_model(model, filepath: Path) -> None:
    """Save trained model to a file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
        logger.success(f"Model saved to {filepath.absolute()}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise


def train(
    train_data_path: Path = PROCESSED_DATA_DIR / "train_processed.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    params_path: Path = PARAMS_FILE,
):
    """Train and save a machine learning model."""
    logger.info("Loading parameters and data...")
    params = load_params(params_path)
    train_data = load_data(train_data_path)

    X_train, y_train = prepare_data(train_data, "Potability")

    model = train_model(X_train, y_train, params)

    save_model(model, model_path)


@app.command()
def main():
    """Execute full pipeline: train and save model."""
    logger.info("Starting full training pipeline...")
    train()
    logger.success("Pipeline execution complete!")


if __name__ == "__main__":
    app()
