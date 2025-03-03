import json
import os
from pathlib import Path
import dagshub
import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient

# dagshub.init(repo_owner="minhquana1906", repo_name="water_potability_prediction", mlflow=True)
# mlflow.set_tracking_uri("https://dagshub.com/minhquana1906/water_potability_prediction.mlflow")
# mlflow.set_experiment("Final model")

dagshub_token = os.getenv("DAGSHUB_TOKEN")

if not dagshub_token:
    logger.error("MLflow authentication credentials are not set!")
    raise EnvironmentError("MLflow credentials environment variables are missing!")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_uri = "https://dagshub.com"
repo_owner = "minhquana1906"
repo_name = "water_potability_prediction"

mlflow.set_tracking_uri(f"{dagshub_uri}/{repo_owner}/{repo_name}.mlflow")
mlflow.set_experiment("Final model")

client = MlflowClient()

PROJ_ROOT = Path(__file__).resolve().parents[2]
RUN_INFO_PATH = PROJ_ROOT / "reports/run_info.json"


def load_run_info(filepath: Path) -> dict:
    """Load run_id and model_name from JSON."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading run info file {filepath}: {e}")
        raise


def register_model(run_id: str, model_name: str) -> str:
    """Register model to MLflow."""
    model_uri = f"runs:/{run_id}/artifacts/{model_name}"
    reg = mlflow.register_model(model_uri=model_uri, name=model_name)

    logger.info(f"Model {model_name} (version {reg.version}) registered successfully.")

    client.set_model_version_tag(model_name, reg.version, "validation_status", "pending")
    client.set_registered_model_alias(model_name, "staging", reg.version)

    return reg.version


def main():
    """Pipeline to register the model to MLflow."""
    logger.info("Registering the model to MLflow...")
    run_info = load_run_info(RUN_INFO_PATH)
    run_id = run_info["run_id"]
    model_name = run_info["model_name"]

    register_model(run_id, model_name)


if __name__ == "__main__":
    main()
