from pathlib import Path
import os
import json
import pickle
import typer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import mlflow
import dagshub

from mlflow.models import infer_signature
from loguru import logger
from dvclive import Live
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

# from src.config import (
#     MODELS_DIR,
#     PROCESSED_DATA_DIR,
#     PARAMS_FILE,
#     METRICS_DIR,
#     CONFUSION_MATRIX_DIR,
#     ROC_CURVE_DIR,
# )

PROJ_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJ_ROOT / "models"
PROCESSED_DATA_DIR = PROJ_ROOT / "data/processed"
PARAMS_FILE = PROJ_ROOT / "params.yaml"
METRICS_DIR = PROJ_ROOT / "reports/metrics"
CONFUSION_MATRIX_DIR = PROJ_ROOT / "reports/figures/confusion_matrix"
ROC_CURVE_DIR = PROJ_ROOT / "reports/figures/roc_curve"

MODEL_NAME = "RandomForestClassifier"

# This code is only used with browser-based DAGs, in CI pipeline, we need to use the key-based authentication
# dagshub.init(repo_owner="minhquana1906", repo_name="water_potability_prediction", mlflow=True)
# mlflow.set_tracking_uri("https://dagshub.com/minhquana1906/water_potability_prediction.mlflow")
# mlflow.set_experiment("Final model")


dagshub_token = os.getenv("DAGSHUB_TOKEN")
logger.info(f"DAGSHUB_TOKEN: {dagshub_token}")
if not dagshub_token:
    logger.error("DAGSHUB_TOKEN is not set!")
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set!")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_uri = "https://dagshub.com"
repo_owner = "minhquana1906"
repo_name = "water_potability_prediction"
mlflow.set_tracking_uri(f"{dagshub_uri}/{repo_owner}/{repo_name}.mlflow")
mlflow.set_experiment("Final model")

app = typer.Typer()


def load_yaml(filepath: Path) -> dict:
    """Load YAML file safely."""
    try:
        with open(filepath, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"File {filepath} not found! Cannot load parameters.")
        raise
    except yaml.YAMLError as e:
        logger.exception(f"Error parsing YAML file {filepath}: {e}")
        raise


def load_csv(filepath: Path) -> pd.DataFrame:
    """Load CSV file safely."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        logger.error(f"File {filepath} not found! Cannot load data.")
        raise
    except pd.errors.ParserError as e:
        logger.exception(f"Error reading CSV file {filepath}: {e}")
        raise


def load_pickle(filepath: Path):
    """Load a pickle file safely."""
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.error(f"File {filepath} not found! Cannot load model.")
        raise
    except pickle.UnpicklingError as e:
        logger.exception(f"Error loading pickle file {filepath}: {e}")
        raise


def prepare_data(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataset into features and target."""
    if target_col not in df.columns:
        raise KeyError(f"Column {target_col} not found in dataframe!")
    return df.drop(columns=[target_col]), df[target_col]


def predict(model, X_test: pd.DataFrame, run_id: str) -> np.ndarray:
    """Perform predictions and log model to MLflow using an existing run."""
    try:
        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=MODEL_NAME,
                signature=infer_signature(X_test, model.predict(X_test)),
                input_example=X_test.head(),
            )
            logger.info(f"Logged model to MLflow in run {run_id}")

        return model.predict(X_test)

    except Exception as e:
        logger.exception(f"Error during prediction or MLflow logging: {e}")
        raise


def evaluate(y_test: pd.Series, y_pred: np.ndarray, params: dict, run_id: str) -> dict:
    """Evaluate model performance and log everything to the same MLflow run."""
    try:
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred),
        }

        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            logger.info("Logged metrics & params to MLflow.")

            # Log artifacts
            cm = confusion_matrix(y_test, y_pred)
            cm_path = CONFUSION_MATRIX_DIR / "confusion_matrix.png"
            plot_and_save_confusion_matrix(cm, cm_path)
            mlflow.log_artifact(str(cm_path))

            fpr, tpr, _ = roc_curve(y_test, y_pred)
            roc_path = ROC_CURVE_DIR / "roc_curve.png"
            plot_and_save_roc_curve(fpr, tpr, roc_path)
            mlflow.log_artifact(str(roc_path))

            logger.info("Logged artifacts (confusion matrix, ROC curve).")

        return metrics

    except Exception as e:
        logger.exception(f"Error occurred during evaluation: {e}")
        raise


def save_metrics(metrics: dict, filepath: Path) -> None:
    """Save evaluation metrics to a JSON file."""
    logger.info(f"Saving metrics to {filepath}...")
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("Metrics successfully saved.")
    except Exception as e:
        logger.exception(f"Failed to save metrics: {e}")
        raise


def plot_and_save_confusion_matrix(cm: np.ndarray, filepath: Path) -> None:
    """Plot and save the confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=True, yticklabels=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(filepath)
    plt.close()


def plot_and_save_roc_curve(fpr, tpr, filepath: Path) -> None:
    """Plot and save the ROC curve."""
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(filepath)
    plt.close()


def save_metrics(metrics: dict, filepath: Path) -> None:
    """Save evaluation metrics to a JSON file."""
    logger.info(f"Saving metrics to {filepath}...")

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("Metrics successfully saved.")
    except Exception as e:
        logger.error(f"Failed to save metrics: {str(e)}", exc_info=True)


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_processed.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    metrics_path: Path = METRICS_DIR / "metrics.json",
):
    """Perform inference and evaluation on the test dataset."""
    try:
        logger.info("Loading hyperparameters...")
        params = load_yaml(PARAMS_FILE)

        logger.info("Loading test data...")
        test_data = load_csv(features_path)
        X_test, y_test = prepare_data(test_data, "Potability")

        logger.info("Loading model...")
        model = load_pickle(model_path)

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run {run_id}")

            logger.info("Performing inference...")
            y_pred = predict(model, X_test, run_id)

            logger.info("Evaluating model...")
            metrics = evaluate(y_test, y_pred, params, run_id)

            logger.info("Saving metrics...")
            save_metrics(metrics, metrics_path)

            logger.info("Save run id and model name...")
            run_info = {"run_id": run.info.run_id, "model_name": MODEL_NAME}
            reports_path = "reports/run_info.json"
            with open(reports_path, "w") as file:
                json.dump(run_info, file, indent=4)

        logger.success("Inference and evaluation complete.")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    app()
