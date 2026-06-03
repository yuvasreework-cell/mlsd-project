"""
preprocess.py
─────────────
Data preprocessing module for Heart Disease Prediction.

Responsibilities:
  - Load the UCI Cleveland Heart Disease dataset (or a local CSV)
  - Rename columns to human-readable names
  - Handle missing values
  - Split features / target
  - Return X_train, X_test, y_train, y_test, and the fitted ColumnTransformer
"""

import os
import pandas as pd
import numpy as np
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ── Column names for the UCI Cleveland Heart Disease dataset ──────────────────
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

# ── Feature groups ────────────────────────────────────────────────────────────
NUMERIC_FEATURES  = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

TARGET_COLUMN = "target"

# ── UCI dataset URL ────────────────────────────────────────────────────────────
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)


def load_dataset(data_path: str = "data/heart.csv") -> pd.DataFrame:
    """Load dataset from local CSV or download from UCI repository."""
    if os.path.exists(data_path):
        print(f"[INFO] Loading dataset from local file: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print(f"[INFO] Downloading Cleveland Heart Disease dataset from UCI …")
        import urllib.request
        raw_path = data_path
        os.makedirs(os.path.dirname(raw_path) if os.path.dirname(raw_path) else ".", exist_ok=True)
        try:
            urllib.request.urlretrieve(UCI_URL, raw_path.replace(".csv", "_raw.csv"))
            df = pd.read_csv(
                raw_path.replace(".csv", "_raw.csv"),
                names=COLUMN_NAMES,
                na_values="?"
            )
            df.to_csv(raw_path, index=False)
            print(f"[INFO] Dataset saved to {raw_path}")
        except Exception as e:
            print(f"[WARN] Download failed ({e}). Generating synthetic data …")
            df = _generate_synthetic_data()
            os.makedirs(os.path.dirname(raw_path) if os.path.dirname(raw_path) else ".", exist_ok=True)
            df.to_csv(raw_path, index=False)
    return df


def _generate_synthetic_data(n: int = 303, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic dataset that mirrors the UCI Cleveland schema."""
    rng = np.random.default_rng(seed)
    data = {
        "age":      rng.integers(29, 77, n),
        "sex":      rng.integers(0, 2, n),
        "cp":       rng.integers(0, 4, n),
        "trestbps": rng.integers(94, 200, n),
        "chol":     rng.integers(126, 564, n),
        "fbs":      rng.integers(0, 2, n),
        "restecg":  rng.integers(0, 3, n),
        "thalach":  rng.integers(71, 202, n),
        "exang":    rng.integers(0, 2, n),
        "oldpeak":  np.round(rng.uniform(0, 6.2, n), 1),
        "slope":    rng.integers(0, 3, n),
        "ca":       rng.integers(0, 4, n),
        "thal":     rng.choice([0, 1, 2, 3], n),
    }
    df = pd.DataFrame(data)
    # Create a semi-realistic target (more attrition for higher oldpeak / age)
    prob = 0.3 + 0.005 * (df["age"] - 50) + 0.05 * df["oldpeak"]
    prob = prob.clip(0.1, 0.9)
    df["target"] = rng.binomial(1, prob, n)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe:
      1. Ensure correct column names
      2. Replace '?' with NaN, drop rows with missing values
      3. Binarize target: 0 -> 0 (no disease), >0 -> 1 (disease)
    """
    if list(df.columns) != COLUMN_NAMES:
        if len(df.columns) == len(COLUMN_NAMES):
            df.columns = COLUMN_NAMES
    # Replace '?' with NaN and drop
    df = df.replace("?", np.nan)
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna()
    after  = len(df)
    if before != after:
        print(f"[INFO] Dropped {before - after} rows with missing values.")
    # Binarize target
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] > 0).astype(int)
    return df.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer for numeric scaling + categorical encoding."""
    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline,  NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    """Split dataframe into X_train, X_test, y_train, y_test."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


if __name__ == "__main__":
    df = load_dataset()
    df = clean_data(df)
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    print(f"\nTarget distribution:\n{df['target'].value_counts()}")
