"""
test_pipeline.py
────────────────
Automated tests for the Heart Disease MLOps pipeline.

Run with: pytest tests/test_pipeline.py -v
"""

import os
import sys
import json
import pytest
import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.preprocess import (
    load_dataset, clean_data, build_preprocessor,
    get_train_test_split, _generate_synthetic_data,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN
)

MODEL_PATH   = os.path.join(ROOT, "model", "model.pkl")
METRICS_PATH = os.path.join(ROOT, "model", "metrics.json")
DATA_PATH    = os.path.join(ROOT, "data",  "heart.csv")


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def raw_df():
    """Load or generate synthetic dataset once per test module."""
    return _generate_synthetic_data(n=303, seed=42)


@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean_data(raw_df.copy())


@pytest.fixture(scope="module")
def split_data(clean_df):
    return get_train_test_split(clean_df)


@pytest.fixture(scope="module")
def trained_model():
    """Load trained model if available; skip tests requiring it if not."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"model.pkl not found at {MODEL_PATH}. Run train.py first.")
    return joblib.load(MODEL_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestDataset:

    def test_synthetic_data_shape(self, raw_df):
        assert raw_df.shape[0] == 303, "Expected 303 rows in synthetic dataset"
        assert len(raw_df.columns) == 14, "Expected 14 columns (13 features + target)"

    def test_required_columns(self, raw_df):
        required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
        for col in required:
            assert col in raw_df.columns, f"Missing column: {col}"

    def test_clean_data_no_nulls(self, clean_df):
        assert clean_df.isnull().sum().sum() == 0, "Cleaned dataset should have no nulls"

    def test_target_binary(self, clean_df):
        unique_vals = set(clean_df[TARGET_COLUMN].unique())
        assert unique_vals.issubset({0, 1}), f"Target should be binary, got: {unique_vals}"

    def test_numeric_features_dtype(self, clean_df):
        for col in NUMERIC_FEATURES:
            assert pd.api.types.is_numeric_dtype(clean_df[col]), f"{col} should be numeric"

    def test_train_test_split_ratio(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total
        assert abs(test_ratio - 0.2) < 0.02, f"Test ratio {test_ratio:.3f} not ≈ 0.2"

    def test_no_label_leakage(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        assert TARGET_COLUMN not in X_train.columns
        assert TARGET_COLUMN not in X_test.columns


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSOR TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessor:

    def test_preprocessor_fits(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        preprocessor = build_preprocessor()
        X_transformed = preprocessor.fit_transform(X_train)
        assert X_transformed.shape[0] == len(X_train)
        assert X_transformed.shape[1] > len(NUMERIC_FEATURES), "OHE should expand feature count"

    def test_preprocessor_transforms_test(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        preprocessor = build_preprocessor()
        preprocessor.fit(X_train)
        X_test_t = preprocessor.transform(X_test)
        assert X_test_t.shape[0] == len(X_test)

    def test_preprocessor_no_nan_output(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        preprocessor = build_preprocessor()
        X_t = preprocessor.fit_transform(X_train)
        assert not np.any(np.isnan(X_t)), "Preprocessed output should have no NaN values"


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestModel:

    def test_model_loads(self, trained_model):
        assert trained_model is not None

    def test_model_has_predict(self, trained_model):
        assert hasattr(trained_model, "predict"), "Model must have predict method"
        assert hasattr(trained_model, "predict_proba"), "Model must have predict_proba method"

    def test_model_predict_shape(self, trained_model, split_data):
        _, X_test, _, y_test = split_data
        preds = trained_model.predict(X_test)
        assert preds.shape == y_test.shape, "Prediction shape must match y_test shape"

    def test_model_predict_binary(self, trained_model, split_data):
        _, X_test, _, _ = split_data
        preds = trained_model.predict(X_test)
        assert set(preds).issubset({0, 1}), f"Predictions must be binary, got: {set(preds)}"

    def test_model_probability_bounds(self, trained_model, split_data):
        _, X_test, _, _ = split_data
        proba = trained_model.predict_proba(X_test)
        assert proba.shape[1] == 2, "predict_proba should return 2 columns"
        assert (proba >= 0).all(), "Probabilities must be >= 0"
        assert (proba <= 1).all(), "Probabilities must be <= 1"
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6,
                                   err_msg="Probabilities must sum to 1")

    def test_model_minimum_accuracy(self, trained_model):
        """Test that model meets minimum accuracy using the same data it was trained on."""
        from sklearn.metrics import accuracy_score
        data_path = DATA_PATH if os.path.exists(DATA_PATH) else None
        if data_path is None:
            pytest.skip("Real dataset not available")
        df = clean_data(load_dataset(data_path))
        _, X_test, _, y_test = get_train_test_split(df)
        preds = trained_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        assert acc >= 0.70, f"Model accuracy {acc:.3f} is below minimum threshold of 0.70"

    def test_metrics_json_exists(self):
        assert os.path.exists(METRICS_PATH), f"metrics.json not found at {METRICS_PATH}"

    def test_metrics_json_keys(self):
        if not os.path.exists(METRICS_PATH):
            pytest.skip("metrics.json not found")
        with open(METRICS_PATH) as f:
            m = json.load(f)
        required_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        for k in required_keys:
            assert k in m, f"Missing key in metrics.json: {k}"

    def test_metrics_values_in_range(self):
        if not os.path.exists(METRICS_PATH):
            pytest.skip("metrics.json not found")
        with open(METRICS_PATH) as f:
            m = json.load(f)
        for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            v = m[k]
            assert 0.0 <= v <= 1.0, f"Metric {k}={v} out of range [0, 1]"

    def test_single_prediction(self, trained_model):
        """Test end-to-end prediction with a single patient record."""
        sample = pd.DataFrame([{
            "age": 54, "sex": 1, "cp": 2, "trestbps": 132, "chol": 254,
            "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
            "oldpeak": 1.4, "slope": 1, "ca": 0, "thal": 2
        }])
        pred   = trained_model.predict(sample)
        proba  = trained_model.predict_proba(sample)
        assert pred[0] in [0, 1]
        assert 0.0 <= proba[0][1] <= 1.0
