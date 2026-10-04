"""
eda.py
------
Same plots as before, but returns base64-encoded PNG strings instead of
matplotlib Figure objects, since Flask templates need <img> src data.
"""

import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

DARK_BG = "#0d1c32"
TEXT_COLOR = "#d6e3ff"
MUTED = "#8892B0"

sns.set_style("darkgrid", {"axes.facecolor": DARK_BG, "figure.facecolor": DARK_BG})

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": DARK_BG,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlecolor": TEXT_COLOR,
    "grid.color": "#1c2a41",
})


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=["int64", "float64"]).drop(columns=["Person ID"], errors="ignore")
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap of Numeric Features")
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_stress_distribution(df: pd.DataFrame, stress_class_col="Stress Class"):
    fig, ax = plt.subplots(figsize=(5, 4))
    order = ["Low", "Medium", "High"]
    sns.countplot(data=df, x=stress_class_col, order=order, hue=stress_class_col, legend=False,
                  palette={"Low": "#4CAF50", "Medium": "#FFC107", "High": "#F44336"}, ax=ax)
    ax.set_title("Distribution of Stress Levels")
    ax.set_xlabel("Stress Level")
    ax.set_ylabel("Number of People")
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_sleep_vs_stress(df: pd.DataFrame, stress_class_col="Stress Class"):
    fig, ax = plt.subplots(figsize=(5, 4))
    order = ["Low", "Medium", "High"]
    sns.boxplot(data=df, x=stress_class_col, y="Sleep Duration", order=order, hue=stress_class_col, legend=False,
                palette={"Low": "#4CAF50", "Medium": "#FFC107", "High": "#F44336"}, ax=ax)
    ax.set_title("Sleep Duration vs Stress Level")
    ax.set_xlabel("Stress Level")
    ax.set_ylabel("Sleep Duration (hours)")
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_activity_vs_stress(df: pd.DataFrame, stress_class_col="Stress Class"):
    fig, ax = plt.subplots(figsize=(5, 4))
    order = ["Low", "Medium", "High"]
    sns.stripplot(data=df, x=stress_class_col, y="Physical Activity Level", order=order, hue=stress_class_col, legend=False,
                  palette={"Low": "#4CAF50", "Medium": "#FFC107", "High": "#F44336"},
                  jitter=0.25, size=6, alpha=0.7, ax=ax)
    ax.set_title("Physical Activity vs Stress Level")
    ax.set_xlabel("Stress Level")
    ax.set_ylabel("Physical Activity Level")
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="mako", xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar=False, linewidths=1, linecolor=DARK_BG)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_feature_importance(importance_dict, color="#44d8f1"):
    df = pd.DataFrame(importance_dict.items(), columns=["Feature", "Importance"]).sort_values("Importance", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df, x="Importance", y="Feature", ax=ax, color=color)
    fig.tight_layout()
    return _fig_to_base64(fig)
