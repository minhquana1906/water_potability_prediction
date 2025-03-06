import unittest
from pathlib import Path
import os

import mlflow
import pandas as pd
from loguru import logger
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import dagshub


# Project-related configurations
PROJ_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = PROJ_ROOT / "data/processed/test_processed.csv"
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

mlflow.set_tracking_uri(f"{dagshub_uri}/{repo_owner}/{repo_name}.mlflow")
mlflow.set_experiment("Final model")

# Initialize MLflow with DagsHub
# dagshub.init(repo_owner="minhquana1906", repo_name="water_potability_prediction", mlflow=True)
# mlflow.set_tracking_uri("https://dagshub.com/minhquana1906/water_potability_prediction.mlflow")
# mlflow.set_experiment("Final model")


class TestModelLoading(unittest.TestCase):
    """Unit test class to verify MLflow model loading from Staging status."""

    def test_model_in_staging(self):
        """Test if the model is in 'Staging' stage."""
        client = MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])

        if not versions:
            self.fail("No models found in 'Staging' stage, skipping the loading test!")

        latest_version = versions[0].version
        logger.success(f"Model {MODEL_NAME} (version {latest_version}) is in 'Staging' stage!")

    def test_model_loading(self):
        """Test the loading of the model in 'Staging' stage."""
        client = MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])

        if not versions:
            self.fail("No models found in 'Staging' stage, skipping the loading test!")

        try:
            run_id = versions[0].run_id
            model = mlflow.pyfunc.load_model(f"runs:/{run_id}/{MODEL_NAME}")
        except Exception as e:
            self.fail(f"Model loading failed with error: {e}")

        self.assertIsNotNone(model, "Model loading failed!")
        logger.success(f"Model {MODEL_NAME} (version {versions[0].version}) loaded successfully!")

    def test_model_performance(self):
        """Test the performance of the model in 'Staging' stage."""
        client = MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])

        if not versions:
            self.fail("No models found in 'Staging' stage, skipping the performance test!")

        run_id = versions[0].run_id
        model = mlflow.pyfunc.load_model(f"runs:/{run_id}/{MODEL_NAME}")

        # Load the test data
        if not TEST_DATA.exists():
            self.fail(f"Test data file {TEST_DATA} not found!")

        test_data = pd.read_csv(TEST_DATA)
        X_test = test_data.drop(columns=["Potability"])
        y_test = test_data["Potability"]

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"Precision: {prec:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")

        self.assertGreaterEqual(acc, 0.3, "Accuracy is below threshold!")
        self.assertGreaterEqual(prec, 0.3, "Precision is below threshold!")
        self.assertGreaterEqual(recall, 0.3, "Recall is below threshold!")
        self.assertGreaterEqual(f1, 0.3, "F1 Score is below threshold!")


if __name__ == "__main__":
    unittest.main()
