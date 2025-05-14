# Earthquake Prediction API

A FastAPI-based earthquake prediction service that:

- Predicts whether an earthquake is likely (binary classification)
- Estimates how many hours until it may occur (regression)

## 🔧 Input Parameters

Send a POST request to `/predict` with this JSON:

```json
{
  "latitude": 34.05,
  "longitude": -118.25,
  "depth": 12.5,
  "mag": 4.8,
  "gap": 30.0,
  "dmin": 0.2,
  "rms": 0.6
}
