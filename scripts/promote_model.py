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
    """Promote the model to production."""
    client = MlflowClient()
    staging_versions = client.get_latest_versions(MODEL_NAME, stages=["staging"])
    if not staging_versions:
        logger.error(f"No staging versions found for model {MODEL_NAME}!")
        return

    latest_staging_version = staging_versions[0]
    staging_version_number = latest_staging_version.version

    production_versions = client.get_latest_versions(MODEL_NAME, stages=["production"])

    if production_versions:
        current_production_version = production_versions[0]
        production_version_number = current_production_version.version
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=production_version_number,
            stage="archived",
            archive_existing_versions=False,
        )
        logger.info(f"Model {MODEL_NAME} version {production_version_number} archived.")
    else:
        logger.info(f"No production versions found for model {MODEL_NAME}.")

    # Transition the latest staging version to production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=staging_version_number,
        stage="production",
        archive_existing_versions=False,
    )
    logger.success(f"Model {MODEL_NAME} version {staging_version_number} promoted to production!")


if __name__ == "__main__":
    promote_model_to_production()
