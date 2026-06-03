"""
train.py
────────
Model training script for Heart Disease Prediction MLOps pipeline.

Workflow:
  1. Load and clean data
  2. Split into train/test
  3. Build a scikit-learn Pipeline (preprocessor + LogisticRegression)
  4. Train with cross-validation
  5. Print metrics
  6. Serialize the pipeline to model/model.pkl
  7. Save training metrics to model/metrics.json
"""

import os
import sys
import json
import time
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.preprocess import (
    load_dataset, clean_data, build_preprocessor, get_train_test_split
)

MODEL_DIR  = os.path.join(ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
DATA_PATH = os.path.join(ROOT, "data", "heart.csv")


def train(data_path: str = DATA_PATH) -> dict:
    """Full training run. Returns metrics dict."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── 1. Load & clean ───────────────────────────────────────────────────────
    print("=" * 60)
    print("  Heart Disease Prediction -- Model Training")
    print("=" * 60)
    df = load_dataset(data_path)
    df = clean_data(df)
    print(f"\n[DATA] Shape after cleaning : {df.shape}")
    print(f"[DATA] Class distribution  : {df['target'].value_counts().to_dict()}")

    # ── 2. Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    print(f"\n[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")

    # ── 3. Build pipeline ─────────────────────────────────────────────────────
    preprocessor = build_preprocessor()
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   model)
    ])

    # ── 4. Cross-validation ───────────────────────────────────────────────────
    print("\n[TRAIN] Running 5-fold cross-validation …")
    t0 = time.time()
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    print(f"[CV] Fold accuracies : {np.round(cv_scores, 4)}")
    print(f"[CV] Mean ± Std       : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── 5. Final fit on full training set ─────────────────────────────────────
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"\n[TRAIN] Training completed in {elapsed:.2f}s")

    # ── 6. Evaluate on test set ───────────────────────────────────────────────
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    roc  = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS (Test Set)")
    print("=" * 60)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc:.4f}")
    print("=" * 60)
    print("\n[REPORT]\n", classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    metrics = {
        "accuracy":         round(acc,  4),
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "f1_score":         round(f1,   4),
        "roc_auc":          round(roc,  4),
        "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std_accuracy":  round(float(cv_scores.std()),  4),
        "train_samples":    int(len(X_train)),
        "test_samples":     int(len(X_test)),
        "training_time_s":  round(elapsed, 2),
    }

    # ── 7. Save model & metrics ───────────────────────────────────────────────
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n[SAVE] Model serialized -> {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics saved   -> {METRICS_PATH}")

    return metrics


if __name__ == "__main__":
    train()
