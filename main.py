from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load models and scaler
scaler = joblib.load("scaler_final.pkl")
lr_model = joblib.load("lr_final.pkl")
rf_model = joblib.load("rf_final.pkl")
reg_model = joblib.load("reg_model.pkl")
logistic_model = joblib.load("logistic_model.pkl")

alpha = 0.5        # weight between lr_model and rf_model
threshold = 0.5    # probability threshold
mag_threshold = 4.0  # minimum magnitude to consider earthquake likely

app = FastAPI()

# Define input schema
class InputData(BaseModel):
    latitude: float
    longitude: float
    depth: float
    mag: float
    gap: float
    time_full: float  # keep this to pass to reg_model

@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Prepare features for classification (5 features)
        features = np.array([[input_data.latitude, input_data.longitude, input_data.depth,
                              input_data.mag, input_data.gap]])
        features_scaled = scaler.transform(features)

        # Predict probability using classification models
        prob_lr = lr_model.predict_proba(features_scaled)[:, 1]
        prob_rf = rf_model.predict_proba(features_scaled)[:, 1]
        combined_proba = alpha * prob_lr + (1 - alpha) * prob_rf

        # Apply magnitude threshold first to reduce false positives
        if input_data.mag < mag_threshold:
            earthquake_likely = False
        else:
            earthquake_likely = combined_proba[0] >= threshold

        # Predict time-to-event if earthquake is likely
        if earthquake_likely:
            # Prepare regression features (5 features including time_full)
            reg_features = np.array([[input_data.mag, input_data.depth, input_data.latitude,
                                      input_data.longitude, input_data.time_full]])
            hours_until_event = reg_model.predict(reg_features)[0]
            # Clamp hours to max 24 and min 0
            hours_until_event = max(0, min(hours_until_event, 24))
        else:
            hours_until_event = None

        return {
            "earthquake_likely": bool(earthquake_likely),
            "predicted_hours_until_event": hours_until_event
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
