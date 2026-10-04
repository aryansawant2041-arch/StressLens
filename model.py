"""
model.py
--------
This file does 3 jobs:
1. Load the raw CSV and clean/preprocess it (encode text columns, bin the
   Stress Level number into Low/Medium/High, scale numeric columns).
2. Train three models: Decision Tree, Random Forest, Gradient Boosting.
3. Evaluate them and save everything (models + encoders + scaler + metrics)
   to disk so app.py can load them instantly without retraining every time.

Run this file directly to train everything:
    python model.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

DATA_PATH = os.path.join("dataset", "sleep_health_lifestyle.csv")
MODELS_DIR = "models"

# Columns we will actually feed into the model
FEATURE_COLUMNS = [
    "Gender",
    "Age",
    "Occupation",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "BMI Category",
    "Heart Rate",
    "Daily Steps",
]
CATEGORICAL_COLUMNS = ["Gender", "Occupation", "BMI Category"]
NUMERIC_COLUMNS = [
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Heart Rate",
    "Daily Steps",
]
TARGET_COLUMN = "Stress Level"
TARGET_CLASS_COLUMN = "Stress Class"  # Low / Medium / High


def bin_stress_level(value):
    """Turn the numeric 1-10 Stress Level into 3 buckets."""
    if value <= 4:
        return "Low"
    elif value <= 6:
        return "Medium"
    else:
        return "High"


def load_and_preprocess_data(path=DATA_PATH):
    """
    Loads the CSV, creates the Low/Medium/High target column,
    label-encodes categorical columns, and scales numeric columns.

    Returns:
        X (DataFrame): processed features
        y (Series): target labels (Low/Medium/High)
        df_raw (DataFrame): original untouched dataframe (for EDA)
        encoders (dict): fitted LabelEncoders for each categorical column
        scaler (StandardScaler): fitted scaler for numeric columns
        target_encoder (LabelEncoder): encodes Low/Medium/High -> 0/1/2
    """
    df_raw = pd.read_csv(path)

    df = df_raw.copy()
    df[TARGET_CLASS_COLUMN] = df[TARGET_COLUMN].apply(bin_stress_level)

    X = df[FEATURE_COLUMNS].copy()

    # Encode categorical (text) columns into numbers
    encoders = {}
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Scale numeric columns so all features are on a similar range
    scaler = StandardScaler()
    X[NUMERIC_COLUMNS] = scaler.fit_transform(X[NUMERIC_COLUMNS])

    # Encode the target Low/Medium/High into 0/1/2
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df[TARGET_CLASS_COLUMN])

    return X, y, df, encoders, scaler, target_encoder


def train_models(X_train, y_train):
    """Train Decision Tree, Random Forest, and Gradient Boosting."""
    dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)

    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    return {"Decision Tree": dt, "Random Forest": rf, "Gradient Boosting": gb}


def evaluate_models(models, X_test, y_test, target_encoder):
    """Return accuracy, confusion matrix, and classification report per model."""
    class_names = list(target_encoder.classes_)
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds).tolist()
        report = classification_report(
            y_test, preds, target_names=class_names, output_dict=True
        )
        results[name] = {
            "accuracy": acc,
            "confusion_matrix": cm,
            "classification_report": report,
        }
    return results, class_names


def run_full_pipeline():
    """Loads data, trains all models, evaluates them, and saves everything."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    X, y, df_raw, encoders, scaler, target_encoder = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = train_models(X_train, y_train)
    results, class_names = evaluate_models(models, X_test, y_test, target_encoder)

    # Feature importance from Random Forest
    rf_importances = dict(
        zip(FEATURE_COLUMNS, models["Random Forest"].feature_importances_.tolist())
    )

    # Save models + preprocessing objects so app.py doesn't need to retrain
    joblib.dump(models["Decision Tree"], os.path.join(MODELS_DIR, "decision_tree.joblib"))
    joblib.dump(models["Random Forest"], os.path.join(MODELS_DIR, "random_forest.joblib"))
    joblib.dump(models["Gradient Boosting"], os.path.join(MODELS_DIR, "gradient_boosting.joblib"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(target_encoder, os.path.join(MODELS_DIR, "target_encoder.joblib"))

    metrics_to_save = {
        "results": {
            name: {"accuracy": r["accuracy"], "confusion_matrix": r["confusion_matrix"],
                   "classification_report": r["classification_report"]}
            for name, r in results.items()
        },
        "class_names": class_names,
        "feature_importance": rf_importances,
        "feature_columns": FEATURE_COLUMNS,
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    return results, rf_importances


if __name__ == "__main__":
    print("Loading data and training models...\n")
    results, importances = run_full_pipeline()

    print("=== MODEL ACCURACY ===")
    for name, r in results.items():
        print(f"{name}: {r['accuracy']*100:.2f}%")

    print("\n=== RANDOM FOREST FEATURE IMPORTANCE ===")
    for feat, score in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"{feat}: {score:.4f}")

    print("\nAll models and preprocessing objects saved to the 'models/' folder.")
