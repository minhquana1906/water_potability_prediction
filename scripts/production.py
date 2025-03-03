import os
import mlflow
from mlflow.tracking import MlflowClient
from loguru import logger

dagshub_token = os.getenv("DAGSHUB_TOKEN")

if not dagshub_token:
    logger.error("MLflow authentication credentials are not set!")
    raise EnvironmentError("MLflow credentials environment variables are missing!")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_uri = "https://dagshub.com"
repo_owner = "minhquana1906"
repo_name = "water_potability_prediction"


def promote_model_to_production(model_name: str, version: str):
    """Promote the model to production."""
    client = MlflowClient()
    client.set_registered_model_alias(model_name, "Champion", version)
    client.set_model_version_tag(model_name, version, "deployment", "Production")
    client.set_model_version_tag(model_name, version, "validation_status", "approved")
    logger.success(f"Model {model_name} version {version} is now deployed as 'Champion'!")
