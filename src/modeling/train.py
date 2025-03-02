from pathlib import Path
import pickle
import pandas as pd
import yaml
import typer
from loguru import logger
from sklearn.ensemble import RandomForestClassifier

# from src.config import PROCESSED_DATA_DIR, MODELS_DIR, PARAMS_FILE

PROJ_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJ_ROOT / "data/processed"
MODELS_DIR = PROJ_ROOT / "models"
PARAMS_FILE = PROJ_ROOT / "params.yaml"
logger.info(f"PARAM: {PARAMS_FILE}")

app = typer.Typer()


def load_yaml(filepath: Path) -> dict:
    """Load a YAML file."""
    if not filepath.exists():
        logger.error(f"File {filepath} not found!")
        raise FileNotFoundError(f"File {filepath} not found!")
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def load_csv(filepath: Path) -> pd.DataFrame:
    """Load a CSV file."""
    if not filepath.exists():
        logger.error(f"File {filepath} not found!")
        raise FileNotFoundError(f"File {filepath} not found!")
    return pd.read_csv(filepath)


def prepare_data(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataset into features and target."""
    if target_col not in df.columns:
        logger.error(f"Column '{target_col}' not found in dataset!")
        raise KeyError(f"Column '{target_col}' not found in dataset!")
    return df.drop(columns=[target_col]), df[target_col]


def train_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> RandomForestClassifier:
    """Train a Random Forest model."""
    logger.info("Training Random Forest model...")
    model_params = params.get("model_train", {})
    model = RandomForestClassifier(
        n_estimators=model_params.get("n_estimators", 100),
        max_depth=model_params.get("max_depth", None),
        random_state=model_params.get("random_state", 42),
    )
    model.fit(X_train, y_train)
    logger.success("Model training complete.")
    return model


def save_model(model, filepath: Path) -> None:
    """Save the trained model to a file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
        logger.success(f"Model saved to {filepath.absolute()}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise


def train_pipeline(
    train_data_path: Path = PROCESSED_DATA_DIR / "train_processed.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    params_path: Path = PARAMS_FILE,
):
    """Train a model and save it."""
    logger.info("Loading hyperparameters and training data...")
    params = load_yaml(params_path)
    train_data = load_csv(train_data_path)

    X_train, y_train = prepare_data(train_data, "Potability")

    model = train_model(X_train, y_train, params)

    save_model(model, model_path)


@app.command()
def main():
    """Execute the full training pipeline."""
    logger.info("Starting full training pipeline...")
    train_pipeline()
    logger.success("Pipeline execution complete!")


if __name__ == "__main__":
    app()
