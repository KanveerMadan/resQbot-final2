from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load models and scaler
scaler = joblib.load("scaler_final.pkl")       # scaler for classification (5 features)
lr_model = joblib.load("lr_final.pkl")
rf_model = joblib.load("rf_final.pkl")
reg_model = joblib.load("reg_model.pkl")
logistic_model = joblib.load("logistic_model.pkl")

alpha = 0.5        # weight between lr_model and rf_model
threshold = 0.5    # probability threshold

app = FastAPI()

class InputData(BaseModel):
    latitude: float
    longitude: float
    depth: float
    mag: float
    gap: float
    time_full: str   # ISO8601 string, e.g. "2025-05-15T12:34:56Z"

@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Prepare classification features (5 features)
        features_class = np.array([[input_data.mag, input_data.depth, input_data.latitude, input_data.longitude, input_data.gap]])
        features_class_scaled = scaler.transform(features_class)

        # Predict probabilities from classification models
        prob_lr = lr_model.predict_proba(features_class_scaled)[:, 1]
        prob_rf = rf_model.predict_proba(features_class_scaled)[:, 1]
        combined_proba = alpha * prob_lr + (1 - alpha) * prob_rf
        earthquake_likely = combined_proba[0] >= threshold

        # Prepare features for regression (6 features including time.full)
        time_numeric = pd.to_datetime(input_data.time_full).timestamp()
        features_reg = np.array([[input_data.mag, input_data.depth, input_data.latitude, input_data.longitude, input_data.gap, time_numeric]])

        # Scale regression features if your scaler supports 6 features,
        # otherwise skip scaling or retrain scaler with 6 features.
        try:
            features_reg_scaled = scaler.transform(features_reg)
        except Exception:
            features_reg_scaled = features_reg  # fallback: no scaling for regression features

        # Predict time-to-event using regression model if earthquake likely
        hours_until_event = reg_model.predict(features_reg_scaled)[0] if earthquake_likely else None

        return {
            "earthquake_likely": bool(earthquake_likely),
            "predicted_hours_until_event": hours_until_event
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
