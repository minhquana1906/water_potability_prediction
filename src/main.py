import pickle
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger
from mlflow.tracking import MlflowClient

from data_model import WaterTest

# from src.config import MODELS_DIR

MODEL_NAME = "RandomForest"

app = FastAPI(
    title="Water Potability Prediction API",
    description="API to predict whether water is potable(safe to drink) or not.",
    version="0.1",
)

# Load model
# try:
#     model_path = MODELS_DIR / "model.pkl"
#     with open(model_path, "rb") as file:
#         model = pickle.load(file)
#     logger.success(f"Model loaded from {model_path}")
# except Exception as e:
#     logger.error(f"Failed to load model: {str(e)}")
#     raise RuntimeError("Model loading failed!")

dagshub_url = "https://dagshub.com"
repo_owner = "minhquana1906"
repo_name = "water_potability_prediction"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")


def load_registered_model():
    """Load registered model from MLflow."""
    try:
        versions = MlflowClient().get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            raise ValueError(f"Model '{MODEL_NAME}' not found in Production stage!")

        run_id = versions[0].run_id
        model = mlflow.pyfunc.load_model(f"runs:/{run_id}/{MODEL_NAME}")
        logger.success(f"Model {MODEL_NAME} loaded from MLflow.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {MODEL_NAME}: {str(e)}")
        raise RuntimeError(f"Model {MODEL_NAME} loading failed!")


model = load_registered_model()


@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Welcome to the Water Potability Prediction API!"}


@app.post("/predict")
def predict(water: WaterTest):
    """Predict water potability from input features."""
    try:
        data_dict = water.dict()
        data_df = pd.DataFrame([data_dict])

        logger.info(f"Received request: {data_dict}")

        prediction = model.predict(data_df)[0]
        result = "Potable" if prediction == 1 else "Not Potable"

        # Log prediction to MLflow
        # with mlflow.start_run(nested=True):
        #     mlflow.log_params(data_dict)
        #     mlflow.log_metric("prediction", int(prediction))
        #     logger.info("Logged prediction to MLflow")

        logger.success(f"Prediction: {result}")
        return {"prediction": result}

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction error")


if __name__ == "__main__":
    app.run()
