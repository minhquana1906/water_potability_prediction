from pathlib import Path
import json
import pickle
import typer
import pandas as pd
import numpy as np
import yaml
from loguru import logger
from dvclive import Live
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, PARAMS_FILE

app = typer.Typer()


def load_params(filepath: Path) -> dict:
    """Load hyperparameters from a YAML file."""
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} not found! Cannot load parameters.")
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def load_data(filepath: Path) -> pd.DataFrame:
    """Load test data from a CSV file."""
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} not found! Cannot load data.")
    return pd.read_csv(filepath)


def prepare_data(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataset into features and target."""
    if target_col not in df.columns:
        raise KeyError(f"Column {target_col} not found in dataframe!")
    return df.drop(target_col, axis=1), df[target_col]


def load_model(filepath: Path) -> object:
    """Load a trained model from a pickle file."""
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} not found! Cannot load model.")
    with open(filepath, "rb") as f:
        return pickle.load(f)


def predict(model: object, X_test: pd.DataFrame) -> np.ndarray:
    """Perform predictions using the trained model."""
    return model.predict(X_test)


def evaluate(y_test: pd.Series, y_pred: np.ndarray, params: dict) -> dict:
    """Evaluate model performance and log metrics with dvclive."""
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    # Ensure dvclive syncs with params.yaml
    with Live(save_dvc_exp=True) as live:
        live.log_params(params)

        for metric_name, metric_value in metrics.items():
            live.log_metric(metric_name, metric_value)

    return metrics


def save_metrics(metrics: dict, filepath: Path) -> None:
    """Save evaluation metrics to a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=4)


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_processed.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    metrics_path: Path = REPORTS_DIR / "metrics.json",
):
    """Perform inference and evaluation on the test dataset."""
    logger.info("Loading hyperparameters...")
    params = load_params(PARAMS_FILE)  # Load params once for consistency

    logger.info("Loading test data...")
    test_data = load_data(features_path)
    X_test, y_test = prepare_data(test_data, "Potability")

    logger.info("Loading model...")
    model = load_model(model_path)

    logger.info("Performing inference...")
    y_pred = predict(model, X_test)

    logger.info("Evaluating model...")
    metrics = evaluate(y_test, y_pred, params)

    logger.info("Saving metrics...")
    save_metrics(metrics, metrics_path)

    logger.success("Inference and evaluation complete.")


if __name__ == "__main__":
    app()
