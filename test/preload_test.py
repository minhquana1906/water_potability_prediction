import json

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

# Set MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/minhquana1906/water_potability_prediction.mlflow")

# Initialize MLflow Client
client = MlflowClient()

# Define model name
MODEL_NAME = "RandomForestClassifier"

# Get the model version assigned to alias 'Champion'
champion_model = client.get_model_version_by_alias(MODEL_NAME, "Champion")

if champion_model:
    model_version = champion_model.version
    model_uri = f"models:/{MODEL_NAME}/{model_version}"
    print(f"Loading Champion model: {MODEL_NAME} (version {model_version})")
else:
    raise ValueError(f"No Champion model found for {MODEL_NAME}")

# Load the model
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Load input data
with open("./test/sample_input.json") as f:
    json_data = json.load(f)

data = pd.DataFrame(json_data["data"], columns=json_data["columns"])

# Run prediction
predictions = loaded_model.predict(data)
print("Predictions:", predictions)
