import os
import unittest

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from mllfow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

MODEL_NAME = "RandomForestClassifier"

mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

if not mlflow_username or not mlflow_password:
    logger.error("MLflow authentication credentials are not set!")
    raise EnvironmentError("MLflow credentials environment variables are missing!")

os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password

dagshub_uri = "https://dagshub.com"
repo_owner = "minhquana1906"
repo_name = "water_potability_prediction"

mlflow.set_tracking_uri(f"{dagshub_uri}/{repo_owner}/{repo_name}.mlflow")
mlflow.set_experiment("Final model")


class TestModelLoading(unittest.TestCase):
    """Unit test class to verify MLflow model loading from pending status."""
