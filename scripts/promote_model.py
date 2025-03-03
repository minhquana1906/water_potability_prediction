import os
import mlflow
from mlflow.tracking import MlflowClient
from loguru import logger

MODEL_NAME = "RandomForest"
dagshub_token = os.getenv("DAGSHUB_TOKEN")

if not dagshub_token:
    logger.error("MLflow authentication credentials are not set!")
    raise EnvironmentError("MLflow credentials environment variables are missing!")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_uri = "https://dagshub.com"
repo_owner = "minhquana1906"
repo_name = "water_potability_prediction"


def promote_model_to_production():
    """Promote the model to production using alias and tag instead of stage."""
    client = MlflowClient()

    # Get the latest model in staging by alias
    try:
        staging_model = client.get_model_version_by_alias(MODEL_NAME, "staging")
    except Exception:
        logger.error(f"No model found with alias 'staging' for {MODEL_NAME}!")
        return

    staging_version = staging_model.version
    logger.info(f"Found staging model: {MODEL_NAME} version {staging_version}")

    # Get the current production model by alias
    try:
        production_model = client.get_model_version_by_alias(MODEL_NAME, "production")
        production_version = production_model.version

        # Remove production alias from the current production model
        client.delete_model_version_alias(MODEL_NAME, "production", production_version)
        logger.info(f"Removed alias 'production' from model version {production_version}.")
    except Exception:
        logger.info(f"No current production model found for {MODEL_NAME}.")

    # Promote staging model to production
    client.set_registered_model_alias(MODEL_NAME, "production", staging_version)
    client.set_model_version_tag(MODEL_NAME, staging_version, "deployment", "Production")

    logger.success(f"Model {MODEL_NAME} version {staging_version} promoted to production!")


if __name__ == "__main__":
    promote_model_to_production()
