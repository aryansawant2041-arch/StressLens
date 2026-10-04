"""
app.py
------
Flask web application for StressLens.

Routes:
  GET  /            - Home page
  GET  /predict      - Predict page (form)
  POST /api/predict  - JSON API used by the Predict page's JavaScript
  GET  /insights     - EDA page
  GET  /models       - Model comparison page

Run with:  python app.py
Then open: http://localhost:5000
"""

import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

from model import (
    load_and_preprocess_data,
    run_full_pipeline,
    bin_stress_level,
    FEATURE_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    TARGET_CLASS_COLUMN,
    DATA_PATH,
    MODELS_DIR,
)
from eda import (
    plot_correlation_heatmap,
    plot_stress_distribution,
    plot_sleep_vs_stress,
    plot_activity_vs_stress,
    plot_confusion_matrix,
    plot_feature_importance,
)

app = Flask(__name__)

MODEL_FILES = {
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "Gradient Boosting": "gradient_boosting.joblib",
}
MODEL_META = {
    "Decision Tree": {"tag": "BASELINE", "tag_color": "#b6c6ed", "subtitle": "Standard Baseline", "icon": "🌳"},
    "Random Forest": {"tag": "CORE MODEL", "tag_color": "#44d8f1", "subtitle": "Ensemble Learning (Bagging)", "icon": "🌲"},
    "Gradient Boosting": {"tag": "ENSEMBLE BOOSTING", "tag_color": "#38debb", "subtitle": "High Precision Optimized", "icon": "⚡"},
}
RESULT_COLORS = {"Low": "#4CAF50", "Medium": "#FFC107", "High": "#F44336"}
RESULT_DESCRIPTIONS = {
    "Low": "Your current data indicates minimal physiological strain. Your lifestyle patterns align with the dataset's lowest-stress group.",
    "Medium": "Your current data indicates moderate physiological strain. This level often suggests manageable workload with slight disruptions in recovery cycles.",
    "High": "Your current data indicates significant physiological strain. Sleep, activity, or heart rate patterns closely resemble the dataset's highest-stress group.",
}


def ensure_models_trained():
    if not os.path.exists(os.path.join(MODELS_DIR, "metrics.json")):
        print("First run detected - training models...")
        run_full_pipeline()


def load_all():
    ensure_models_trained()
    models = {name: joblib.load(os.path.join(MODELS_DIR, fname)) for name, fname in MODEL_FILES.items()}
    encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    target_encoder = joblib.load(os.path.join(MODELS_DIR, "target_encoder.joblib"))
    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    return models, encoders, scaler, target_encoder, metrics


# Load everything once at startup
MODELS, ENCODERS, SCALER, TARGET_ENCODER, METRICS = load_all()


def get_raw_df():
    df = pd.read_csv(DATA_PATH)
    df[TARGET_CLASS_COLUMN] = df[TARGET_COLUMN].apply(bin_stress_level)
    return df


@app.route("/")
def home():
    df = get_raw_df()
    best_model_name = max(METRICS["results"], key=lambda m: METRICS["results"][m]["accuracy"])
    best_acc = round(METRICS["results"][best_model_name]["accuracy"] * 100, 1)
    preview_columns = list(df.columns[:10])
    preview_rows = df.head(8).to_dict(orient="records")
    return render_template(
        "home.html", active="home",
        best_acc=best_acc, best_model_name=best_model_name,
        dataset_size=len(df), num_features=len(FEATURE_COLUMNS),
        preview_columns=preview_columns, preview_rows=preview_rows,
    )


@app.route("/predict")
def predict():
    return render_template(
        "predict.html", active="predict",
        bmi_classes=list(ENCODERS["BMI Category"].classes_),
        occupation_classes=list(ENCODERS["Occupation"].classes_),
        gender_classes=list(ENCODERS["Gender"].classes_),
        model_names=list(MODELS.keys()),
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        input_dict = {
            "Gender": data["gender"], "Age": data["age"], "Occupation": data["occupation"],
            "Sleep Duration": data["sleep_duration"], "Quality of Sleep": data["quality_of_sleep"],
            "Physical Activity Level": data["physical_activity"], "BMI Category": data["bmi_category"],
            "Heart Rate": data["heart_rate"], "Daily Steps": data["daily_steps"],
        }
        input_df = pd.DataFrame([input_dict])[FEATURE_COLUMNS]
        for col in CATEGORICAL_COLUMNS:
            input_df[col] = ENCODERS[col].transform(input_df[col].astype(str))
        input_df[NUMERIC_COLUMNS] = SCALER.transform(input_df[NUMERIC_COLUMNS])

        model_choice = data.get("model_choice", "Gradient Boosting")
        model = MODELS[model_choice]
        pred_encoded = model.predict(input_df)[0]
        pred_label = TARGET_ENCODER.inverse_transform([pred_encoded])[0]

        proba = None
        confidence = 100.0
        stress_index = {"Low": 15, "Medium": 50, "High": 85}[pred_label]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            classes = list(TARGET_ENCODER.classes_)
            confidence = float(proba[list(classes).index(pred_label)]) * 100
            label_to_score = {"Low": 0, "Medium": 50, "High": 100}
            stress_index = sum(p * label_to_score[c] for p, c in zip(proba, classes))

        importance = METRICS["feature_importance"]
        top_feats = sorted(importance.items(), key=lambda x: -x[1])[:3]
        max_imp = top_feats[0][1] if top_feats else 1
        top_factors = [
            {"name": name, "level": ("Elevated" if pred_label == "High" else "Moderate" if pred_label == "Medium" else "Normal"),
             "pct": round((imp / max_imp) * 100, 1)}
            for name, imp in top_feats
        ]

        recommendations = []
        if data["sleep_duration"] < 6.5:
            recommendations.append({"icon": "😴", "title": "Improve Sleep Duration", "text": "You're sleeping under 6.5 hours — aim for 7-8 hours to help lower stress."})
        if data["quality_of_sleep"] <= 5:
            recommendations.append({"icon": "🌙", "title": "Improve Sleep Quality", "text": "Low sleep quality is one of the strongest predictors of higher stress in this model."})
        if data["physical_activity"] < 40:
            recommendations.append({"icon": "🏃", "title": "Increase Physical Activity", "text": "More daily activity is linked with lower predicted stress in this dataset."})
        if data["heart_rate"] > 85:
            recommendations.append({"icon": "❤️", "title": "Monitor Heart Rate", "text": "Your heart rate is on the higher side — consider relaxation techniques."})
        if not recommendations:
            recommendations.append({"icon": "✅", "title": "Great Habits", "text": "Your inputs align with lower-stress patterns in this dataset. Keep it up!"})

        return jsonify({
            "prediction": pred_label,
            "confidence": confidence,
            "stress_index": stress_index,
            "description": RESULT_DESCRIPTIONS[pred_label],
            "top_factors": top_factors,
            "recommendations": recommendations[:3],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/insights")
def insights():
    df = get_raw_df()
    return render_template(
        "insights.html", active="insights",
        plot_correlation=plot_correlation_heatmap(df),
        plot_distribution=plot_stress_distribution(df),
        plot_sleep=plot_sleep_vs_stress(df),
        plot_activity=plot_activity_vs_stress(df),
    )


@app.route("/models")
def models_page():
    model_cards = []
    for name in MODELS:
        r = METRICS["results"][name]
        macro = r["classification_report"]["macro avg"]
        meta = MODEL_META[name]
        model_cards.append({
            "name": name, "accuracy": round(r["accuracy"] * 100, 1),
            "f1": round(macro["f1-score"], 2), "precision": round(macro["precision"], 2),
            "recall": round(macro["recall"], 2), **meta,
        })

    sorted_models = sorted(model_cards, key=lambda x: -x["accuracy"])
    bar_colors = ["#38debb", "#44d8f1", "#b6c6ed"]
    for m, c in zip(sorted_models, bar_colors):
        m["bar_color"] = c

    best_model_name = max(METRICS["results"], key=lambda m: METRICS["results"][m]["accuracy"])
    cm = METRICS["results"][best_model_name]["confusion_matrix"]
    class_names = METRICS["class_names"]

    return render_template(
        "models.html", active="models",
        model_cards=model_cards, sorted_models=sorted_models,
        confusion_model=best_model_name,
        confusion_matrix_img=plot_confusion_matrix(cm, class_names),
        feature_importance_img=plot_feature_importance(METRICS["feature_importance"]),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
