#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import json

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "video_metrics_for_ML__comments_only.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
METRIC_DIR = RESULTS_DIR / "metrics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE)

# Define "success" = top 25% of videos by view_count
threshold = df["view_count"].quantile(0.75)
df["success"] = (df["view_count"] >= threshold).astype(int)

print("Success threshold (views):", threshold)
print(df["success"].value_counts())

# Features (from your prep)
features = [
    "log_owner_engagement_share",
    "log_response_rate",
    "owner_comments_total",
    "audience_comments_total"
]

X = df[features].fillna(0)
y = df["success"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

print("\nClassification Report:")
rep = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm.tolist())

# Save metrics
with open(METRIC_DIR / "rf_metrics.json", "w") as f:
    json.dump({
        "threshold_views_q75": float(threshold),
        "classification_report": rep,
        "confusion_matrix": cm.tolist(),
        "features": features
    }, f, indent=2)

# Feature Importances
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(6,4))
importances.plot(kind="bar")
plt.title("Feature Importance (Random Forest)")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(FIG_DIR / "rf_feature_importance.png", dpi=150)
plt.close()

print(f"\nSaved metrics → {METRIC_DIR/'rf_metrics.json'}")
print(f"Saved feature importance plot → {FIG_DIR/'rf_feature_importance.png'}")