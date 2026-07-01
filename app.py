"""
app.py
-------
Flask web application for the Rising Waters flood prediction system.

Routes:
  /            Home page
  /predict     Prediction input form (GET) + handles submission (POST)
  /result      Shows Flood Chance / No Flood Chance result page

Run:
  python app.py
Then open http://127.0.0.1:5000/ in your browser.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "flood_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model", "metadata.json")

app = Flask(__name__)

# ---- Load trained model, scaler and metadata at startup -------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(METADATA_PATH) as f:
    metadata = json.load(f)

FEATURE_NAMES = metadata["feature_names"]
MODEL_NAME = metadata["best_model_name"]
MODEL_ACCURACY = round(metadata["accuracy"] * 100, 2)

# Human-friendly labels + sensible default values for the form
FEATURE_CONFIG = {
    "ANNUAL_RAINFALL": {"label": "Annual Rainfall (mm)", "default": 2200, "min": 0, "max": 6000, "step": 1},
    "SEASONAL_RAINFALL": {"label": "Seasonal Rainfall (mm)", "default": 1100, "min": 0, "max": 3000, "step": 1},
    "CLOUD_VISIBILITY": {"label": "Cloud Visibility (km)", "default": 6, "min": 0, "max": 15, "step": 0.1},
    "HUMIDITY": {"label": "Humidity (%)", "default": 70, "min": 0, "max": 100, "step": 1},
    "TEMPERATURE": {"label": "Temperature (\u00b0C)", "default": 28, "min": 0, "max": 50, "step": 0.1},
    "RIVER_WATER_LEVEL": {"label": "River Water Level (m)", "default": 5, "min": 0, "max": 20, "step": 0.1},
    "WIND_SPEED": {"label": "Wind Speed (km/h)", "default": 12, "min": 0, "max": 100, "step": 0.1},
    "DRAINAGE_QUALITY": {"label": "Drainage Quality (1=Poor .. 5=Excellent)", "default": 3, "min": 1, "max": 5, "step": 1},
    "ELEVATION": {"label": "Elevation (m above sea level)", "default": 120, "min": 0, "max": 1000, "step": 1},
    "HISTORICAL_FLOOD_OCCURRENCES": {"label": "Historical Flood Occurrences (last 10 yrs)", "default": 1, "min": 0, "max": 10, "step": 1},
}


@app.route("/")
def home():
    return render_template("home.html", model_name=MODEL_NAME, accuracy=MODEL_ACCURACY)


@app.route("/predict", methods=["GET"])
def predict_form():
    fields = [(name, FEATURE_CONFIG.get(name, {"label": name, "default": 0, "min": 0, "max": 100, "step": 1}))
              for name in FEATURE_NAMES]
    return render_template("predict.html", fields=fields)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_values = []
        readable_inputs = {}
        for name in FEATURE_NAMES:
            raw = request.form.get(name, FEATURE_CONFIG.get(name, {}).get("default", 0))
            val = float(raw)
            input_values.append(val)
            readable_inputs[FEATURE_CONFIG.get(name, {}).get("label", name)] = val

        X = pd.DataFrame([input_values], columns=FEATURE_NAMES)
        X_scaled = scaler.transform(X)

        prediction = int(model.predict(X_scaled)[0])

        probability = None
        if hasattr(model, "predict_proba"):
            probability = round(float(model.predict_proba(X_scaled)[0][1]) * 100, 2)

        if prediction == 1:
            return render_template(
                "flood_result.html",
                probability=probability,
                inputs=readable_inputs,
                model_name=MODEL_NAME,
            )
        else:
            return render_template(
                "no_flood_result.html",
                probability=probability,
                inputs=readable_inputs,
                model_name=MODEL_NAME,
            )
    except Exception as e:
        return render_template("predict.html",
                                fields=[(name, FEATURE_CONFIG.get(name, {"label": name, "default": 0, "min": 0, "max": 100, "step": 1}))
                                        for name in FEATURE_NAMES],
                                error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
