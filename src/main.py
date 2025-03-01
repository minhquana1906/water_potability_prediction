from fastapi import FastAPI
import pickle
import pandas as pd
from data_model import Water, WaterTest

app = FastAPI(
    title="Water Potability Prediction API",
    description="API to predict water potability",
    version="0.1",
)

model = pickle.load(open("../models/model.pkl", "rb"))


@app.get("/")
def index():
    return "Welcome to the Water Potability Prediction API!"


@app.post("/predict")
def predict(water: WaterTest):
    data = dict(water)  # Convert pydantic model to dictionary

    data_df = pd.DataFrame(data, index=[0])
    # Convert dictionary to dataframe

    prediction = model.predict(data_df)[0]
    # Make prediction and get the first element of the result (predict returns a list)

    return "Potable" if prediction == 1 else "Not Potable"
