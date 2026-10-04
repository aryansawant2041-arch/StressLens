# StressLens — Prediction of Stress Levels Using Ensemble Learning

A full custom web application (Flask + HTML/Tailwind CSS) that predicts a
person's stress level (**Low / Medium / High**) from lifestyle and sleep
data, using **Decision Tree**, **Random Forest**, and **Gradient Boosting**
models — styled to match the StressLens design system.

## Dataset

- **Name:** Sleep Health and Lifestyle Dataset
- **Source:** [Kaggle](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)
- Place the downloaded CSV at `dataset/sleep_health_lifestyle.csv`

## Project Structure

```
stresslens-web/
├── app.py                  # Flask app (routes + prediction API)
├── model.py                 # Data preprocessing + model training/evaluation
├── eda.py                   # Exploratory data analysis plots (base64 PNGs)
├── requirements.txt
├── templates/
│   ├── base.html             # Shared navbar/footer layout
│   ├── home.html
│   ├── predict.html
│   ├── insights.html
│   └── models.html
├── static/
│   └── style.css             # Glass-card / gradient-text / glow CSS
├── dataset/
│   └── sleep_health_lifestyle.csv
└── models/                  # Auto-generated: saved trained models
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

The first time you run it, it will automatically train all three models
(since `models/` doesn't exist yet) — this takes a few seconds. After that
it loads instantly from the saved model files.

## Pages

- **Home** (`/`) — hero section, real accuracy stats, model overview cards, data preview
- **Predict** (`/predict`) — live form with sliders/dropdowns, circular stress gauge, confidence, contributing factors, and suggestions — powered by a `/api/predict` JSON endpoint
- **Insights** (`/insights`) — correlation heatmap, stress distribution, sleep vs stress, activity vs stress
- **Models** (`/models`) — accuracy/F1/precision/recall cards for all 3 models, accuracy comparison bars, confusion matrix, feature importance

## Tech Stack

Python · Flask · pandas · scikit-learn · matplotlib · seaborn · Tailwind CSS (via CDN)
