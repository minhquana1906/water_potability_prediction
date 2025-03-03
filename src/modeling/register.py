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
    with open(filepath, "r") as file:
        run_info = json.load(file)
    logger.info("Loaded run information from JSON file.")
    return run_info


def register_and_transition_model(
    client: MlflowClient, run_id: str, model_name: str, new_stage: str = "Staging"
):
    """Register model to MLflow."""
    model_uri = f"runs:/{run_id}/artifacts/{model_name}"
    reg = mlflow.register_model(model_uri, model_name)
    model_version = reg.version

    client.transition_model_version_stage(
        name=model_name, version=model_version, stage=new_stage, archive_existing_versions=True
    )
    logger.info(
        f"Model '{model_name}' version {model_version} transitioned to '{new_stage}' stage."
    )
    return model_version


def main():
    """Main function to execute the model registration and transition process."""
    logger.info("Registering the model to MLflow...")
    # Load run information
    reports_path = "reports/run_info.json"
    run_info = load_run_info(reports_path)
    run_id = run_info["run_id"]
    model_name = run_info["model_name"]

    # Register model and transition to Staging
    client = MlflowClient()
    model_version = register_and_transition_model(client, run_id, model_name)

    print(f"Model {model_name} version {model_version} transitioned to Staging stage.")


if __name__ == "__main__":
    main()
