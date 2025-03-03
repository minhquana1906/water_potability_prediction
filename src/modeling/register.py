import mlflow
from mlflow.tracking import MlflowClient
import dagshub
import json
from pathlib import Path
from loguru import logger
import os

# from src.config import REPORTS_DIR

PROJ_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJ_ROOT / "reports"

# Initialize MLflow with DagsHub
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

# Create MLflow client
client = MlflowClient()


def load_run_info(filepath: Path) -> dict:
    """Load run_id and model_name from JSON file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File {filepath} not found! Cannot proceed.")
        raise
    except json.JSONDecodeError as e:
        logger.exception(f"Error parsing JSON file {filepath}: {e}")
        raise


def get_production_model_version(model_name: str) -> str:
    """Get the model version currently marked as 'Champion' (Production)."""
    try:
        aliases = client.get_model_version_by_alias(model_name, "Champion")
        if aliases:
            logger.info(f"Current Production model: {model_name} (version {aliases.version})")
            return aliases.version
    except Exception as e:
        logger.warning(f"No Production model found for {model_name}. Error: {e}")
    return None


def register_model(run_id: str, model_name: str) -> str:
    """Register a new model version in MLflow Model Registry."""
    model_uri = f"runs:/{run_id}/artifacts/{model_name}"
    reg = mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.info(f"Model {model_name} (version {reg.version}) registered successfully.")
    return reg.version


def deploy_model(model_name: str, version: str):
    """Deploy a model only if no existing model is already in Production."""
    current_prod_version = get_production_model_version(model_name)

    if current_prod_version:
        logger.warning(
            f"Model {model_name} version {current_prod_version} is already in Production."
        )
        logger.info(f"New model version {version} will be set as 'pending validation'.")
        client.set_model_version_tag(model_name, version, "validation_status", "pending")
        client.set_registered_model_alias(model_name, "Challenger", version)
    else:
        client.set_registered_model_alias(model_name, "Champion", version)
        client.set_model_version_tag(model_name, version, "deployment", "Production")
        client.set_model_version_tag(model_name, version, "validation_status", "approved")
        logger.success(
            f"Model {model_name} version {version} is now set as 'Champion' and ready for deployment!"
        )


def main():
    """Pipeline to register and manage model deployment."""
    run_info = load_run_info(REPORTS_DIR / "run_info.json")
    run_id = run_info["run_id"]
    model_name = run_info["model_name"]

    # Register model
    model_version = register_model(run_id, model_name)

    # Handle deployment logic based on Production status
    deploy_model(model_name, model_version)


if __name__ == "__main__":
    main()
