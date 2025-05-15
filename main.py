
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load models
scaler = joblib.load("scaler_final.pkl")
lr_model = joblib.load("lr_final.pkl")
rf_model = joblib.load("rf_final.pkl")
reg_model = joblib.load("reg_model.pkl")

alpha = 0.5        # weight between lr_model and rf_model
threshold = 0.5    # probability threshold

app = FastAPI()

# Define input schema
class InputData(BaseModel):
    latitude: float
    longitude: float
    depth: float
    mag: float
    gap: float
    

@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Prepare and scale features
        features = np.array([[input_data.latitude, input_data.longitude, input_data.depth,
                              input_data.mag, input_data.gap ]])
        features_scaled = scaler.transform(features)

        # Predict probability
        prob_lr = lr_model.predict_proba(features_scaled)[:, 1]
        prob_rf = rf_model.predict_proba(features_scaled)[:, 1]
        combined_proba = alpha * prob_lr + (1 - alpha) * prob_rf
        earthquake_likely = combined_proba[0] >= threshold

        # Predict time-to-event if earthquake is likely
        hours_until_event = reg_model.predict(features_scaled)[0] if earthquake_likely else None

        return {
            "earthquake_likely": bool(earthquake_likely),
            "predicted_hours_until_event": hours_until_event
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
