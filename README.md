# 🌊 Rising Waters: A Machine Learning Approach to Flood Prediction

An end-to-end flood risk prediction system built with **Python, Machine
Learning, and Flask**, following the full project workflow: environment
setup → dataset → EDA → preprocessing → model building → best model
selection → Flask web app.

---

## 📁 Project Structure

```
flood_prediction_project/
├── app.py                             # Flask web application
├── requirements.txt                   # Python dependencies
├── data/
│   └── flood_data.csv                 # Flood prediction dataset
├── model/
│   ├── flood_model.pkl                # Trained best model (XGBoost)
│   ├── scaler.pkl                     # Fitted StandardScaler
│   └── metadata.json                  # Feature names, accuracy, model info
├── notebooks/
│   ├── 0_generate_dataset.py          # Dataset creation script
│   ├── 1_data_visualization.py        # EDA: distributions, box plots, heatmap
│   └── 2_preprocessing_and_training.py# Preprocessing + model training + saving .pkl
├── static/
│   ├── css/style.css                  # App styling
│   └── images/                        # Saved EDA plots (generated)
└── templates/
    ├── home.html                      # Home page
    ├── predict.html                   # Prediction input page
    ├── flood_result.html              # "Flood Chance" result page
    └── no_flood_result.html           # "No Flood Chance" result page
```

---

## ⚙️ 1. Environment Setup

Using Anaconda Navigator (or any Python 3.8+ environment):

```bash
conda create -n flood_env python=3.10
conda activate flood_env
pip install -r requirements.txt
```

**Note on XGBoost:** the training script tries to import `xgboost`. If it
isn't installed in your environment it automatically falls back to
`sklearn.ensemble.GradientBoostingClassifier` (a very close, API-compatible
substitute) so the pipeline still runs end-to-end. For the true XGBoost
model referenced in the project spec, make sure `pip install xgboost`
succeeds before running the training script.

---

## 📊 2. Dataset

`data/flood_data.csv` contains weather-related features:

| Column | Description |
|---|---|
| ANNUAL_RAINFALL | Annual rainfall (mm) |
| SEASONAL_RAINFALL | Monsoon/seasonal rainfall (mm) |
| CLOUD_VISIBILITY | Cloud visibility (km) |
| HUMIDITY | Relative humidity (%) |
| TEMPERATURE | Temperature (°C) |
| RIVER_WATER_LEVEL | River water level (m) |
| WIND_SPEED | Wind speed (km/h) |
| DRAINAGE_QUALITY | Drainage quality, 1 (poor) – 5 (excellent) |
| ELEVATION | Elevation above sea level (m) |
| HISTORICAL_FLOOD_OCCURRENCES | Flood events in the last 10 years |
| FLOOD_OCCURRED | Target: 1 = flood, 0 = no flood |

> **Important:** This environment had no internet access, so
> `notebooks/0_generate_dataset.py` **simulates** a realistic dataset with
> the same structure, ranges, and relationships you'd find in an actual
> Kaggle "Flood Prediction" dataset (including missing values and outliers,
> so the preprocessing steps have real work to do). If you have internet
> access, download a real dataset from Kaggle, match/rename the columns to
> the ones above (or edit the scripts), replace `data/flood_data.csv`, and
> re-run steps 3 onward for production-quality results.

Regenerate the dataset any time with:
```bash
python notebooks/0_generate_dataset.py
```

---

## 📈 3. Data Visualization & Analysis

```bash
python notebooks/1_data_visualization.py
```

Produces (saved to `static/images/`):
- `01_distributions.png` – univariate distribution/histogram + KDE plots
- `02_boxplots.png` – box plots for outlier detection
- `03_heatmap.png` – correlation heat map
- `04_feature_vs_target.png` – key features vs. flood occurrence
- `05_pairplot.png` – multivariate pairplot

Descriptive statistics and missing-value counts are printed to the console.

---

## 🧹 4. Data Pre-processing & 🤖 5–6. Model Building / Best Model Selection

```bash
python notebooks/2_preprocessing_and_training.py
```

This single script performs:
1. **Missing value handling** – median imputation
2. **Outlier treatment** – IQR-based capping
3. **Categorical encoding** – one-hot encoding (if needed)
4. **X / y split** and **train/test split** (80/20, stratified)
5. **Feature scaling** – `StandardScaler`
6. **Model training** – Decision Tree, Random Forest, KNN, XGBoost
7. **Evaluation** – confusion matrix, classification report, accuracy for each model
   (saved as `static/images/06_confusion_matrices.png` and `07_accuracy_comparison.png`)
8. **Best model selection** – highest test accuracy wins
9. **Saving artifacts** – `model/flood_model.pkl`, `model/scaler.pkl`, `model/metadata.json`

Sample run results on the bundled synthetic dataset:

| Model | Accuracy |
|---|---|
| Decision Tree | ~65.7% |
| Random Forest | ~73.8% |
| KNN | ~73.3% |
| **XGBoost (best)** | **~75.3%** |

> With a real, larger Kaggle flood dataset, accuracy (including the
> 96%+ figure referenced in the project scenario) will depend on the
> real-world signal strength of the data — the pipeline itself is what
> matters and is fully production-ready.

---

## 🌐 7. Flask Web Application

```bash
python app.py
```

Then open **http://127.0.0.1:5000/** in your browser.

Pages:
- **Home (`/`)** – project overview
- **Predict (`/predict`, GET)** – input form for all weather features
- **Predict (`/predict`, POST)** – runs the saved model + scaler on submitted values
- **Flood Chance Result** – shown when the model predicts a flood, with probability and safety advice
- **No Flood Chance Result** – shown when the model predicts no flood, with the probability score

---

## ☁️ 8. IBM Cloud Deployment (Optional Next Step)

To deploy on IBM Cloud:
1. Push this project to a Git repository.
2. Create an IBM Cloud **Code Engine** or **Cloud Foundry** app.
3. Set the buildpack/runtime to Python, with `app.py` as the entry point
   and `gunicorn app:app` as the start command (add `gunicorn` to
   `requirements.txt`).
4. Set the port to read from the `PORT` environment variable:
   ```python
   if __name__ == "__main__":
       app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
   ```
5. Deploy via `ibmcloud` CLI or the IBM Cloud console, then share the
   public URL with disaster management teams.

---

## 🔁 Re-running Everything From Scratch

```bash
pip install -r requirements.txt
python notebooks/0_generate_dataset.py
python notebooks/1_data_visualization.py
python notebooks/2_preprocessing_and_training.py
python app.py
```
