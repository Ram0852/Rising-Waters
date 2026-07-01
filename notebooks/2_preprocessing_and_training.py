"""
2_preprocessing_and_training.py
---------------------------------
Full pipeline:
  1. Data pre-processing (missing values, outliers, scaling, split)
  2. Model building: Decision Tree, Random Forest, KNN, XGBoost
  3. Model evaluation: confusion matrix, classification report, accuracy
  4. Best model selection & saving (.pkl) for Flask deployment
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score
)

# XGBoost is preferred (per project spec). Falls back to sklearn's
# GradientBoostingClassifier (API-compatible) if xgboost isn't installed
# in the current environment, so the pipeline always runs end-to-end.
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[INFO] 'xgboost' package not found in this environment — "
          "using sklearn's GradientBoostingClassifier as a drop-in "
          "substitute. Install xgboost (`pip install xgboost`) and "
          "re-run this script to use real XGBoost.")

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_PATH = os.path.join(BASE_DIR, "data", "flood_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
IMG_DIR = os.path.join(BASE_DIR, "static", "images")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ============================================================
# 1. DATA PRE-PROCESSING
# ============================================================
df = pd.read_csv(DATA_PATH)
print("Loaded data:", df.shape)

# ---- 1a. Handle missing values (median imputation) --------------------
for col in df.columns:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Imputed {col} missing values with median={median_val:.2f}")

# ---- 1b. Detect & treat outliers using IQR capping ---------------------
def cap_outliers_iqr(series, factor=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return series.clip(lower, upper)

feature_cols = [c for c in df.columns if c != "FLOOD_OCCURRED"]
for col in feature_cols:
    if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 5:
        before_max = df[col].max()
        df[col] = cap_outliers_iqr(df[col])
        after_max = df[col].max()
        if before_max != after_max:
            print(f"Capped outliers in {col}: max {before_max:.2f} -> {after_max:.2f}")

# ---- 1c. Encoding categorical data --------------------------------------
# (This synthetic dataset is fully numeric already; DRAINAGE_QUALITY is an
#  ordinal 1-5 scale so no additional one-hot encoding is required. If your
#  real Kaggle dataset has text categories, use pd.get_dummies() here.)
categorical_cols = df.select_dtypes(include="object").columns.tolist()
if categorical_cols:
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print("One-hot encoded columns:", categorical_cols)
else:
    print("No categorical columns to encode.")

# ---- 1d. Split dependent / independent variables ------------------------
X = df.drop(columns=["FLOOD_OCCURRED"])
y = df["FLOOD_OCCURRED"]
FEATURE_NAMES = X.columns.tolist()
print("\nFeatures used for modelling:", FEATURE_NAMES)

# ---- 1e. Train / test split ----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ---- 1f. Feature scaling --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 2. MODEL BUILDING
# ============================================================
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7),
}

if XGB_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=RANDOM_STATE
    )
else:
    models["XGBoost"] = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.08, random_state=RANDOM_STATE
    )

results = {}
trained_models = {}

print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    results[name] = {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    trained_models[name] = model

    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, preds))

# ---- Confusion matrix visualizations -------------------------------------
fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
if len(models) == 1:
    axes = [axes]
for ax, (name, res) in zip(axes, results.items()):
    sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Flood", "Flood"], yticklabels=["No Flood", "Flood"])
    ax.set_title(f"{name}\nAccuracy: {res['accuracy']*100:.2f}%")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/06_confusion_matrices.png", dpi=110)
plt.close()

# ---- Accuracy comparison bar chart ----------------------------------------
plt.figure(figsize=(8, 5))
names = list(results.keys())
accs = [results[n]["accuracy"] * 100 for n in names]
bars = plt.bar(names, accs, color=["#4299e1", "#48bb78", "#ed8936", "#9f7aea"][:len(names)])
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 100)
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width() / 2, acc + 1, f"{acc:.2f}%", ha="center")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/07_accuracy_comparison.png", dpi=110)
plt.close()

# ============================================================
# 3. BEST MODEL SELECTION
# ============================================================
best_name = max(results, key=lambda n: results[n]["accuracy"])
best_model = trained_models[best_name]
best_acc = results[best_name]["accuracy"]

print("\n" + "=" * 60)
print(f"BEST MODEL: {best_name}  |  Accuracy: {best_acc*100:.2f}%")
print("=" * 60)

# ============================================================
# 4. SAVE MODEL, SCALER, METADATA FOR FLASK DEPLOYMENT
# ============================================================
joblib.dump(best_model, f"{MODEL_DIR}/flood_model.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

metadata = {
    "best_model_name": best_name,
    "accuracy": best_acc,
    "feature_names": FEATURE_NAMES,
    "xgboost_used": XGB_AVAILABLE,
    "model_scores": {n: r["accuracy"] for n, r in results.items()},
}
with open(f"{MODEL_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved:")
print(f"  {MODEL_DIR}/flood_model.pkl")
print(f"  {MODEL_DIR}/scaler.pkl")
print(f"  {MODEL_DIR}/metadata.json")
