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
MODEL_NAME = "RandomForest"

# Initialize MLflow Client
client = MlflowClient()


def get_latest_model_by_alias(model_name: str, alias: str):
    """Retrieve the latest model version associated with a specific alias."""
    try:
        version = client.get_model_version_by_alias(model_name, alias)
        return version
    except Exception:
        return None


def compare_models(model_name: str, new_version: str, reference_version: str) -> bool:
    """Compare the newly registered model against a reference model based on accuracy."""
    new_metrics = client.get_model_version(model_name, new_version).tags
    ref_metrics = client.get_model_version(model_name, reference_version).tags

    # Extract accuracy metrics
    new_acc = float(new_metrics.get("metric_accuracy", 0))
    ref_acc = float(ref_metrics.get("metric_accuracy", 0))

    logger.info(f"New Model Accuracy: {new_acc}, Reference Model Accuracy: {ref_acc}")
    return new_acc > ref_acc


class TestModelEvaluation(unittest.TestCase):
    """Unit test class for model validation and comparison."""

    def list_models(self):
        """List all registered models."""
        from pprint import pprint

        client = MlflowClient()
        for rm in client.search_registered_models():
            pprint(dict(rm), indent=4)

    # def test_model_performance(self):
    #     """Validate the performance of the newly registered model."""

    #     latest_model = client.get_model_version_by_alias(MODEL_NAME, "staging")
    #     logger.info(f"Latest model: {latest_model}")
    #     if not latest_model:
    #         self.fail("No models found with alias 'staging', skipping the test!")

    #     # Load model from MLflow
    #     loaded_model = mlflow.pyfunc.load_model(f"runs:/{latest_model.run_id}/{MODEL_NAME}")

    #     # Check for test data file existence
    #     if not TEST_DATA.exists():
    #         self.fail(f"Test data file {TEST_DATA} does not exist!")

    #     test_data = pd.read_csv(TEST_DATA)
    #     X_test = test_data.drop(columns=["Potability"])
    #     y_test = test_data["Potability"]

    #     # Make predictions
    #     y_pred = loaded_model.predict(X_test)
    #     acc = accuracy_score(y_test, y_pred)
    #     prec = precision_score(y_test, y_pred)
    #     recall = recall_score(y_test, y_pred)
    #     f1 = f1_score(y_test, y_pred)

    #     self.assertGreaterEqual(acc, 0.3, "Accuracy should be at least 30%")
    #     self.assertGreaterEqual(prec, 0.3, "Precision should be at least 30%")
    #     self.assertGreaterEqual(recall, 0.3, "Recall should be at least 30%")
    #     self.assertGreaterEqual(f1, 0.3, "F1 Score should be at least 30%")

    #     logger.info(
    #         f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
    #     )

    #     # Compare with staging and production models
    #     staging_model = get_latest_model_by_alias(MODEL_NAME, "staging")
    #     production_model = get_latest_model_by_alias(MODEL_NAME, "production")

    #     if staging_model:
    #         is_better_than_staging = compare_models(
    #             MODEL_NAME, latest_model.version, staging_model.version
    #         )
    #         if is_better_than_staging:
    #             logger.success(
    #                 f"Model {MODEL_NAME} version {latest_model.version} outperforms staging!"
    #             )
    #             client.set_registered_model_alias(MODEL_NAME, "staging", latest_model.version)

    #     if production_model:
    #         is_better_than_production = compare_models(
    #             MODEL_NAME, latest_model.version, production_model.version
    #         )
    #         if is_better_than_production:
    #             logger.success(
    #                 f"Model {MODEL_NAME} version {latest_model.version} outperforms production!"
    #             )
    #             client.set_registered_model_alias(MODEL_NAME, "production", latest_model.version)
    #     else:
    #         logger.success(
    #             f"No production model found. Promoting model {MODEL_NAME} version {latest_model.version} to production!"
    #         )
    #         client.set_registered_model_alias(MODEL_NAME, "production", latest_model.version)


if __name__ == "__main__":
    unittest.main()
