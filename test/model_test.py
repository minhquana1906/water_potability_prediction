import os
import unittest
from pathlib import Path

import mlflow
import pandas as pd
from loguru import logger
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Project-related configurations
PROJ_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = PROJ_ROOT / "data/processed/test_preprocessed.csv"
MODEL_NAME = "RandomForestClassifier"

# Initialize MLflow Client
client = MlflowClient()


def get_latest_model_by_tag(model_name: str, tag_key: str, tag_value: str):
    """Retrieve the latest model version with the specified tag."""
    versions = client.search_model_versions(
        f"name='{model_name}' and tags.{tag_key}='{tag_value}'"
    )
    return max(versions, key=lambda v: int(v.version)) if versions else None


def compare_models(model_name: str, new_version: str, prod_version: str) -> bool:
    """Compare the newly registered model against the Production model based on accuracy."""
    new_metrics = client.get_model_version(model_name, new_version).tags
    prod_metrics = client.get_model_version(model_name, prod_version).tags

    # Extract accuracy metrics
    new_acc = float(new_metrics.get("metric_accuracy", 0))
    prod_acc = float(prod_metrics.get("metric_accuracy", 0))

    logger.info(f"New Model Accuracy: {new_acc}, Production Model Accuracy: {prod_acc}")
    return new_acc > prod_acc


class TestModelEvaluation(unittest.TestCase):
    """Unit test class for model validation and comparison."""

    def test_model_performance(self):
        """Validate the performance of the newly registered model."""
        latest_model = get_latest_model_by_tag(MODEL_NAME, "validation_status", "pending")
        if not latest_model:
            self.fail("No models found with 'validation_status' = 'pending', skipping the test!")

        # Load model from MLflow
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{latest_model.run_id}/{MODEL_NAME}")

        # Check for test data file existence
        if not TEST_DATA.exists():
            self.fail(f"Test data file {TEST_DATA} does not exist!")

        test_data = pd.read_csv(TEST_DATA)
        X_test = test_data.drop(columns=["Potability"])
        y_test = test_data["Potability"]

        # Make predictions
        y_pred = loaded_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logger.info(
            f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
        )

        # Compare with the Production model (if exists)
        current_prod = get_latest_model_by_tag(MODEL_NAME, "deployment", "Production")
        if current_prod:
            is_better = compare_models(MODEL_NAME, latest_model.version, current_prod.version)
            if is_better:
                logger.success(
                    f"Model {MODEL_NAME} version {latest_model.version} outperforms Production!"
                )
                client.set_registered_model_alias(MODEL_NAME, "Champion", latest_model.version)
                client.set_model_version_tag(
                    MODEL_NAME, latest_model.version, "deployment", "Production"
                )
                client.set_model_version_tag(
                    MODEL_NAME, latest_model.version, "validation_status", "approved"
                )

                # Remove alias from the previous Production model
                client.delete_model_version_alias(MODEL_NAME, "Champion", current_prod.version)
            else:
                logger.warning(
                    f"Model {MODEL_NAME} version {latest_model.version} does not outperform the Production model."
                )
        else:
            # If no Production model exists, promote the current model to Production
            logger.success(
                f"No Production model found. Promoting model {MODEL_NAME} version {latest_model.version} to Production!"
            )
            client.set_registered_model_alias(MODEL_NAME, "Champion", latest_model.version)
            client.set_model_version_tag(
                MODEL_NAME, latest_model.version, "deployment", "Production"
            )
            client.set_model_version_tag(
                MODEL_NAME, latest_model.version, "validation_status", "approved"
            )


if __name__ == "__main__":
    unittest.main()
