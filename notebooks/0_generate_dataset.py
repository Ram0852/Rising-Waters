"""
0_generate_dataset.py
----------------------
Generates a realistic synthetic flood-prediction dataset.

NOTE ON DATA SOURCE:
The original project instructions point to open-source flood datasets on
Kaggle (e.g. "Flood Prediction Dataset"). Since this environment has no
internet access, this script SIMULATES a dataset with the same structure,
feature ranges, and statistical relationships that a real Kaggle flood
dataset would have, so the rest of the pipeline (EDA, preprocessing,
modelling, Flask app) runs end-to-end exactly the way it would on the
real data.

If you have internet access, simply download a Kaggle flood dataset,
rename the columns to match the ones used below (or update the scripts),
place it at data/flood_data.csv, and skip running this script.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

np.random.seed(42)

N = 3000

# ---- Base meteorological features -----------------------------------
annual_rainfall = np.random.normal(2200, 550, N).clip(400, 4500)          # mm/year
seasonal_rainfall = annual_rainfall * np.random.uniform(0.35, 0.65, N)     # mm (monsoon season)
cloud_visibility = np.random.normal(6, 3, N).clip(0.1, 15)                 # km
humidity = np.random.normal(70, 15, N).clip(20, 100)                       # %
temperature = np.random.normal(28, 5, N).clip(10, 45)                      # deg C
river_water_level = np.random.normal(5, 2, N).clip(0.5, 15)                # meters
wind_speed = np.random.normal(12, 6, N).clip(0, 60)                        # km/h
drainage_quality = np.random.randint(1, 6, N)                              # 1 (poor) - 5 (excellent)
elevation = np.random.normal(120, 90, N).clip(0, 600)                      # meters above sea level
historical_flood_occurrences = np.random.poisson(1.5, N).clip(0, 10)       # count in last 10 years

# ---- Construct a realistic flood-risk score, then threshold it -------
risk_score = (
    0.0011 * annual_rainfall +
    0.0016 * seasonal_rainfall +
    -0.18 * cloud_visibility +
    0.02 * humidity +
    0.12 * river_water_level +
    -0.35 * drainage_quality +
    -0.006 * elevation +
    0.55 * historical_flood_occurrences +
    0.01 * wind_speed
)

# add noise
risk_score += np.random.normal(0, 1.3, N)

# normalize to 0-1 probability using a sigmoid, then threshold
prob = 1 / (1 + np.exp(-(risk_score - risk_score.mean()) / risk_score.std()))
flood_occurred = (prob > 0.5).astype(int)

df = pd.DataFrame({
    "ANNUAL_RAINFALL": annual_rainfall.round(2),
    "SEASONAL_RAINFALL": seasonal_rainfall.round(2),
    "CLOUD_VISIBILITY": cloud_visibility.round(2),
    "HUMIDITY": humidity.round(2),
    "TEMPERATURE": temperature.round(2),
    "RIVER_WATER_LEVEL": river_water_level.round(2),
    "WIND_SPEED": wind_speed.round(2),
    "DRAINAGE_QUALITY": drainage_quality,
    "ELEVATION": elevation.round(2),
    "HISTORICAL_FLOOD_OCCURRENCES": historical_flood_occurrences,
    "FLOOD_OCCURRED": flood_occurred
})

# introduce a few missing values (to make preprocessing step meaningful)
for col in ["ANNUAL_RAINFALL", "CLOUD_VISIBILITY", "HUMIDITY"]:
    idx = np.random.choice(df.index, size=int(0.02 * N), replace=False)
    df.loc[idx, col] = np.nan

# introduce a few outliers
idx = np.random.choice(df.index, size=15, replace=False)
df.loc[idx, "ANNUAL_RAINFALL"] = df["ANNUAL_RAINFALL"].max() * np.random.uniform(1.5, 2.0, 15)

df.to_csv(os.path.join(DATA_DIR, "flood_data.csv"), index=False)
print("Dataset generated:", df.shape)
print(df["FLOOD_OCCURRED"].value_counts(normalize=True))
