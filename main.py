from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load models and scaler
scaler = joblib.load("scaler_final.pkl")       # For classification features (5 features)
lr_model = joblib.load("lr_final.pkl")
rf_model = joblib.load("rf_final.pkl")
reg_model = joblib.load("reg_model.pkl")
logistic_model = joblib.load("logistic_model.pkl")

alpha = 0.5        # weight between lr_model and rf_model
threshold = 0.5    # probability threshold

app = FastAPI()

# Define input schema including time.full for regression
class InputData(BaseModel):
    latitude: float
    longitude: float
    depth: float
    mag: float
    gap: float
    time_full: float    # include this as float


@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Prepare classification features (5 features)
        features_classification = np.array([[input_data.latitude, input_data.longitude, input_data.depth,
                                             input_data.mag, input_data.gap]])
        features_scaled = scaler.transform(features_classification)

        # Predict probability from classification models
        prob_lr = lr_model.predict_proba(features_scaled)[:, 1]
        prob_rf = rf_model.predict_proba(features_scaled)[:, 1]
        combined_proba = alpha * prob_lr + (1 - alpha) * prob_rf
        earthquake_likely = combined_proba[0] >= threshold

        # Prepare regression features (5 features including time_full)
        features_regression = np.array([[input_data.mag, input_data.depth, input_data.latitude,
                                         input_data.longitude, input_data.time_full]])

        # Predict time-to-event if earthquake is likely and cap between 0 and 24 hours
        if earthquake_likely:
            hours_until_event = reg_model.predict(features_regression)[0]
            hours_until_event = max(0, min(hours_until_event, 24))
        else:
            hours_until_event = None

        return {
            "earthquake_likely": bool(earthquake_likely),
            "predicted_hours_until_event": hours_until_event
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
