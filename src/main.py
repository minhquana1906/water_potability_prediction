from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
import mlflow
from loguru import logger
from data_model import WaterTest
from config import MODELS_DIR


app = FastAPI(
    title="Water Potability Prediction API",
    description="API to predict water potability",
    version="0.1",
)

# Load model
try:
    model_path = MODELS_DIR / "model.pkl"
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    logger.success(f"Model loaded from {model_path}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError("Model loading failed!")


@app.get("/")
def index():
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
        with mlflow.start_run(nested=True):
            mlflow.log_params(data_dict)
            mlflow.log_metric("prediction", int(prediction))
            logger.info("Logged prediction to MLflow")

        logger.success(f"Prediction: {result}")
        return {"prediction": result}

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction error")
