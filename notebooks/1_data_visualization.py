"""
1_data_visualization.py
------------------------
Univariate & multivariate analysis, distribution plots, box plots,
heat maps, and descriptive statistics for the flood dataset.

Run: python 1_data_visualization.py
Outputs are saved as PNG files under ../static/images/
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_PATH = os.path.join(BASE_DIR, "data", "flood_data.csv")
OUT_DIR = os.path.join(BASE_DIR, "static", "images")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("SHAPE:", df.shape)
print("=" * 60)
print("\nDESCRIPTIVE STATISTICS\n")
print(df.describe(include="all").T)

print("\nMISSING VALUES\n")
print(df.isnull().sum())

print("\nCLASS BALANCE (FLOOD_OCCURRED)\n")
print(df["FLOOD_OCCURRED"].value_counts())

numeric_cols = df.select_dtypes(include="number").columns.tolist()
numeric_cols.remove("FLOOD_OCCURRED")

import math
n_cols = len(numeric_cols)
grid_cols = 4
grid_rows = math.ceil(n_cols / grid_cols)

# ---------- Univariate analysis: distribution plots -------------------
fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(20, 4.5 * grid_rows))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#2b6cb0")
    axes[i].set_title(f"Distribution of {col}")
for j in range(n_cols, len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_distributions.png", dpi=110)
plt.close()

# ---------- Box plots (outlier detection) ------------------------------
fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(20, 4.5 * grid_rows))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.boxplot(x=df[col], ax=axes[i], color="#63b3ed")
    axes[i].set_title(f"Box plot of {col}")
for j in range(n_cols, len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_boxplots.png", dpi=110)
plt.close()

# ---------- Multivariate analysis: correlation heat map ---------------
plt.figure(figsize=(11, 9))
corr = df[numeric_cols + ["FLOOD_OCCURRED"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heat Map")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_heatmap.png", dpi=110)
plt.close()

# ---------- Class-wise comparison (multivariate) ------------------------
key_features = ["ANNUAL_RAINFALL", "SEASONAL_RAINFALL", "CLOUD_VISIBILITY",
                 "RIVER_WATER_LEVEL", "HISTORICAL_FLOOD_OCCURRENCES"]
fig, axes = plt.subplots(1, len(key_features), figsize=(22, 5))
for i, col in enumerate(key_features):
    sns.boxplot(x="FLOOD_OCCURRED", y=col, data=df, ax=axes[i], palette="Set2")
    axes[i].set_title(f"{col} vs Flood")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_feature_vs_target.png", dpi=110)
plt.close()

# ---------- Pairplot on key features -----------------------------------
sample = df.sample(min(500, len(df)), random_state=42)
g = sns.pairplot(sample[key_features + ["FLOOD_OCCURRED"]],
                  hue="FLOOD_OCCURRED", palette="Set1", corner=True)
g.savefig(f"{OUT_DIR}/05_pairplot.png", dpi=110)
plt.close()

print("\nAll visualizations saved to:", OUT_DIR)
