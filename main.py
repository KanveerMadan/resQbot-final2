from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load models
scaler = joblib.load("scaler_final.pkl")       # For classification features (5)
lr_model = joblib.load("lr_final.pkl")
rf_model = joblib.load("rf_final.pkl")
reg_model = joblib.load("reg_model.pkl")
logistic_model = joblib.load("logistic_model.pkl")

alpha = 0.5        # weight between lr_model and rf_model
threshold = 0.5    # probability threshold

app = FastAPI()

# Define input schema, add time.full here as optional if you want
class InputData(BaseModel):
    latitude: float
    longitude: float
    depth: float
    mag: float
    gap: float
    time_full: float = None   # Make it optional in case not provided


@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Convert input to DataFrame for easy column selection
        data_dict = {
            "mag": input_data.mag,
            "depth": input_data.depth,
            "latitude": input_data.latitude,
            "longitude": input_data.longitude,
            "gap": input_data.gap,
            "time.full": input_data.time_full if input_data.time_full is not None else 0  # Provide default if missing
        }
        df_sample = pd.DataFrame([data_dict])

        # Features for classification (5 features)
        classification_features = ['mag', 'depth', 'latitude', 'longitude', 'gap']
        X_class = df_sample[classification_features]
        X_class_scaled = scaler.transform(X_class)

        # Classification prediction
        prob_lr = lr_model.predict_proba(X_class_scaled)[:, 1]
        prob_rf = rf_model.predict_proba(X_class_scaled)[:, 1]
        combined_proba = alpha * prob_lr + (1 - alpha) * prob_rf
        earthquake_likely = combined_proba[0] >= threshold

        # Features for regression (6 features including time.full)
        regression_features = ['mag', 'depth', 'latitude', 'longitude', 'gap', 'time.full']
        X_reg = df_sample[regression_features]
        X_reg_scaled = scaler_reg.transform(X_reg)

        # Regression prediction only if earthquake likely
        hours_until_event = reg_model.predict(X_reg_scaled)[0] if earthquake_likely else None

        return {
            "earthquake_likely": bool(earthquake_likely),
            "predicted_hours_until_event": hours_until_event
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
