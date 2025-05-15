from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from datetime import datetime

# Load models and scaler
scaler = joblib.load("scaler_final.pkl")        # For classification features (5)
lr_model = joblib.load("lr_final.pkl")
rf_model = joblib.load("rf_final.pkl")
reg_model = joblib.load("reg_model.pkl")        # Takes 6 features, including time_full as relative hours
logistic_model = joblib.load("logistic_model.pkl")

alpha = 0.5        # Weight between lr_model and rf_model
threshold = 0.5    # Probability threshold

app = FastAPI()

# Define input schema
class InputData(BaseModel):
    latitude: float
    longitude: float
    depth: float
    mag: float
    gap: float
    time_full: float    # Expecting Unix timestamp in seconds


@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Prepare classification features (5 features)
        features_clf = np.array([[input_data.latitude, input_data.longitude, input_data.depth,
                                  input_data.mag, input_data.gap]])
        features_clf_scaled = scaler.transform(features_clf)

        # Predict earthquake likelihood using classification models
        prob_lr = lr_model.predict_proba(features_clf_scaled)[:, 1]
        prob_rf = rf_model.predict_proba(features_clf_scaled)[:, 1]
        combined_proba = alpha * prob_lr + (1 - alpha) * prob_rf
        earthquake_likely = combined_proba[0] >= threshold

        # Convert time_full (absolute timestamp) to relative hours difference from now
        now_ts = datetime.utcnow().timestamp()
        time_diff_hours = (input_data.time_full - now_ts) / 3600.0

        # Prepare regression features (6 features including time_diff_hours)
        features_reg = np.array([[input_data.mag, input_data.depth, input_data.latitude,
                                  input_data.longitude, input_data.gap, time_diff_hours]])

        # Predict time to event if earthquake likely
        if earthquake_likely:
            hours_until_event = reg_model.predict(features_reg)[0]
        else:
            hours_until_event = None

        return {
            "earthquake_likely": bool(earthquake_likely),
            "predicted_hours_until_event": hours_until_event
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
