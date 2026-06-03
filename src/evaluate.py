"""
evaluate.py
───────────
Model evaluation and visualization script.

Generates and saves:
  - Confusion matrix          -> model/confusion_matrix.png
  - ROC curve                 -> model/roc_curve.png
  - Metrics bar chart         -> model/metrics_bar.png
  - Feature importance chart  -> model/feature_importance.png
"""

import os
import sys
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.preprocess import load_dataset, clean_data, get_train_test_split

MODEL_DIR  = os.path.join(ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
DATA_PATH  = os.path.join(ROOT, "data", "heart.csv")

# ── Shared plot style ──────────────────────────────────────────────────────────
DARK_BG   = "#0f1117"
CARD_BG   = "#1a1f2e"
ACCENT    = "#4fc3f7"
ACCENT2   = "#81c784"
ACCENT3   = "#ef5350"
TEXT_CLR  = "#e0e0e0"
GRID_CLR  = "#2a2f3e"


def _apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    CARD_BG,
        "axes.edgecolor":    GRID_CLR,
        "axes.labelcolor":   TEXT_CLR,
        "axes.titlecolor":   TEXT_CLR,
        "xtick.color":       TEXT_CLR,
        "ytick.color":       TEXT_CLR,
        "text.color":        TEXT_CLR,
        "grid.color":        GRID_CLR,
        "grid.linestyle":    "--",
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
    })


def plot_confusion_matrix(y_test, y_pred, save_path):
    """Save a stylized confusion matrix heatmap."""
    _apply_dark_style()
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        ax=ax, linewidths=0.5, linecolor=DARK_BG,
        annot_kws={"size": 16, "weight": "bold", "color": "white"}
    )
    ax.set_title("Confusion Matrix", pad=14)
    ax.set_xlabel("Predicted Label", labelpad=10)
    ax.set_ylabel("True Label", labelpad=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Confusion matrix saved -> {save_path}")


def plot_roc_curve(y_test, y_proba, save_path):
    """Save ROC curve."""
    _apply_dark_style()
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=ACCENT, lw=2.5, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#555", linestyle="--", lw=1.5, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.15, color=ACCENT)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", labelpad=10)
    ax.set_ylabel("True Positive Rate", labelpad=10)
    ax.set_title("ROC Curve -- Logistic Regression", pad=14)
    ax.legend(loc="lower right", facecolor=CARD_BG, edgecolor=GRID_CLR)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] ROC curve saved         -> {save_path}")


def plot_metrics_bar(metrics: dict, save_path: str):
    """Save a horizontal bar chart of main metrics."""
    _apply_dark_style()
    keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    values = [metrics[k] for k in keys]
    colors = [ACCENT, ACCENT2, "#ffb74d", "#ce93d8", ACCENT3]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(labels, values, color=colors, height=0.55, edgecolor=DARK_BG)
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left",
            color=TEXT_CLR, fontsize=12, fontweight="bold"
        )
    ax.set_xlim(0, 1.1)
    ax.set_title("Model Performance Metrics -- Test Set", pad=14)
    ax.set_xlabel("Score", labelpad=8)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Metrics bar chart saved -> {save_path}")


def plot_feature_importance(pipeline, save_path: str):
    """
    Extract and plot coefficient magnitudes from the Logistic Regression model
    as a proxy for feature importance.
    """
    _apply_dark_style()
    from src.preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES

    classifier   = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # Get feature names from the ColumnTransformer
    num_features = NUMERIC_FEATURES
    try:
        ohe_features = list(
            preprocessor
            .named_transformers_["cat"]
            .named_steps["encoder"]
            .get_feature_names_out(CATEGORICAL_FEATURES)
        )
    except Exception:
        ohe_features = [f"cat_{i}" for i in range(classifier.coef_.shape[1] - len(num_features))]

    all_features = num_features + ohe_features
    coefs = classifier.coef_[0]

    # Keep only top-N by absolute value
    n = min(15, len(all_features))
    idx = np.argsort(np.abs(coefs))[-n:]
    top_features = [all_features[i] for i in idx]
    top_coefs    = coefs[idx]
    colors = [ACCENT2 if c > 0 else ACCENT3 for c in top_coefs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_features, top_coefs, color=colors, height=0.65, edgecolor=DARK_BG)
    ax.axvline(0, color=TEXT_CLR, linewidth=0.8, linestyle="--")
    ax.set_title(f"Top {n} Feature Coefficients (Logistic Regression)", pad=14)
    ax.set_xlabel("Coefficient Value", labelpad=8)
    ax.grid(axis="x", alpha=0.3)

    pos_patch = mpatches.Patch(color=ACCENT2, label="Increases Disease Risk")
    neg_patch = mpatches.Patch(color=ACCENT3, label="Decreases Disease Risk")
    ax.legend(handles=[pos_patch, neg_patch], facecolor=CARD_BG, edgecolor=GRID_CLR)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Feature importance saved -> {save_path}")


def evaluate(data_path: str = DATA_PATH, model_path: str = MODEL_PATH):
    """Load model, evaluate on test set, and generate all visualizations."""
    print("=" * 60)
    print("  Heart Disease Prediction -- Model Evaluation")
    print("=" * 60)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")

    pipeline = joblib.load(model_path)
    print(f"[LOAD] Model loaded from {model_path}")

    df = load_dataset(data_path)
    df = clean_data(df)
    _, X_test, _, y_test = get_train_test_split(df)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    plot_confusion_matrix(y_test, y_pred,
                          os.path.join(MODEL_DIR, "confusion_matrix.png"))
    plot_roc_curve(y_test, y_proba,
                   os.path.join(MODEL_DIR, "roc_curve.png"))
    plot_metrics_bar(metrics,
                     os.path.join(MODEL_DIR, "metrics_bar.png"))
    plot_feature_importance(pipeline,
                            os.path.join(MODEL_DIR, "feature_importance.png"))

    print("\n[METRICS]", json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    evaluate()
