"""
streamlit_app.py
────────────────
Professional Streamlit application for Heart Disease Prediction.

Features:
  - Prediction interface with all 13 clinical features
  - Model performance dashboard with metrics & charts
  - Dataset explorer
  - Monitoring log viewer
"""

import os
import sys
import json
import time
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.preprocess import (
    load_dataset, clean_data,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN
)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(ROOT, "model", "model.pkl")
METRICS_PATH = os.path.join(ROOT, "model", "metrics.json")
DATA_PATH    = os.path.join(ROOT, "data",  "heart.csv")
LOG_PATH     = os.path.join(ROOT, "monitoring", "prediction_log.json")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor | MLOps",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #252b3e 100%);
    border: 1px solid #2a2f3e;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.metric-card h3 { color: #a0aec0; font-size: 0.85rem; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 1px; }
.metric-card h2 { color: #4fc3f7; font-size: 2rem; font-weight: 700; margin: 0; }

.predict-card {
    background: linear-gradient(135deg, #1a2744 0%, #1e2d50 100%);
    border: 1px solid #4fc3f7;
    border-radius: 16px;
    padding: 30px;
    margin: 10px 0;
}
.result-positive {
    background: linear-gradient(135deg, #3d1a1a 0%, #4d2020 100%);
    border: 2px solid #ef5350;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.result-negative {
    background: linear-gradient(135deg, #1a3d1a 0%, #204d20 100%);
    border: 2px solid #81c784;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #4fc3f7;
    border-bottom: 2px solid #4fc3f7;
    padding-bottom: 8px;
    margin-bottom: 20px;
}
.tag {
    display: inline-block;
    background: #4fc3f7;
    color: #0f1117;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ── Auto-train on startup if model is missing (Streamlit Cloud support) ───────
def _ensure_model_exists():
    """Train the model automatically if model.pkl doesn't exist."""
    if not os.path.exists(MODEL_PATH):
        import subprocess
        os.makedirs(os.path.join(ROOT, "model"), exist_ok=True)
        with st.spinner("🏋️ First launch: training model... (this takes ~5 seconds)"):
            result = subprocess.run(
                [sys.executable, os.path.join(ROOT, "src", "train.py")],
                capture_output=True, text=True, cwd=ROOT
            )
        if result.returncode != 0:
            st.error(f"Model training failed:\n{result.stderr[-500:]}")
            return False
        st.success("Model trained successfully!")
        st.rerun()
    return True

# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_data():
    df = load_dataset(DATA_PATH)
    return clean_data(df)


def log_prediction(input_data: dict, prediction: int, probability: float):
    """Append prediction to a JSON-lines log file for monitoring."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    entry = {
        "timestamp":   datetime.datetime.utcnow().isoformat() + "Z",
        "input":       input_data,
        "prediction":  int(prediction),
        "probability": round(float(probability), 4),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 Heart Disease\nPredictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔮 Predict", "📊 Model Dashboard", "🗄️ Dataset", "📋 Monitoring"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <small>
    <b>MLOps Pipeline</b><br>
    Dataset &rarr; Preprocessing &rarr; Training &rarr; Evaluation &rarr; Docker &rarr; Kubernetes &rarr; Streamlit<br><br>
    <b>Stack:</b> Python &middot; Scikit-Learn &middot; Streamlit &middot; Docker &middot; GitHub Actions &middot; Kubernetes
    </small>
    """, unsafe_allow_html=True)

# ── Ensure model exists (auto-train on Streamlit Cloud first boot) ─────────────
_ensure_model_exists()


model   = load_model()
metrics = load_metrics()
df      = load_data()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("# 🫀 Heart Disease Prediction")
    st.markdown("#### An End-to-End MLOps Pipeline Demonstration")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### About This Project
        This application demonstrates a **complete MLOps pipeline** built as part of a
        university Machine Learning System Design (MLSD) course project.

        **Dataset:** UCI Cleveland Heart Disease Dataset  
        **Model:** Logistic Regression with balanced class weighting  
        **Pipeline:** sklearn Pipeline (ColumnTransformer → LogisticRegression)

        ---

        ### MLOps Pipeline
        ```
        Dataset → Preprocessing → Training → Evaluation
             ↓
        Model Serialization (model.pkl)
             ↓
        Flask REST API + Streamlit UI
             ↓
        Git → GitHub → GitHub Actions CI
             ↓
        Docker Build → DockerHub Push
             ↓
        Kubernetes Deployment → Service
             ↓
        Monitoring & Logging
        ```
        """)

    with col2:
        if metrics:
            st.markdown("### 📈 Quick Stats")
            for label, key in [("Accuracy", "accuracy"), ("F1-Score", "f1_score"), ("ROC-AUC", "roc_auc")]:
                val = metrics.get(key, 0)
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom:12px">
                    <h3>{label}</h3>
                    <h2>{val:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Dataset Size", f"{len(df)} records")
    with c2: st.metric("Features", "13")
    with c3: st.metric("Model", "Logistic Regression")
    with c4: st.metric("CV Folds", "5")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown("# 🔮 Heart Disease Prediction")
    st.markdown("Enter patient clinical data to predict heart disease risk.")
    st.markdown("---")

    if model is None:
        st.warning("Model not loaded. Please train the model first.")
    else:
        with st.form("prediction_form"):
            st.markdown('<div class="section-header">Patient Clinical Data</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                age      = st.slider("Age", 20, 80, 54)
                trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)
                chol     = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 245)
                thalach  = st.slider("Max Heart Rate Achieved", 60, 210, 150)
                oldpeak  = st.slider("ST Depression (Oldpeak)", 0.0, 7.0, 1.0, step=0.1)

            with c2:
                sex      = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
                cp       = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                                        format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina",
                                                                2: "Non-Anginal Pain", 3: "Asymptomatic"}[x])
                fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                                        format_func=lambda x: "No" if x == 0 else "Yes")
                restecg  = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                                        format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality",
                                                                2: "Left Ventricular Hypertrophy"}[x])

            with c3:
                exang    = st.selectbox("Exercise Induced Angina", options=[0, 1],
                                        format_func=lambda x: "No" if x == 0 else "Yes")
                slope    = st.selectbox("Slope of Peak Exercise ST", options=[0, 1, 2],
                                        format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
                ca       = st.selectbox("Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
                thal     = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                                        format_func=lambda x: {0: "Normal", 1: "Fixed Defect",
                                                                2: "Reversible Defect", 3: "Unknown"}[x])

            submitted = st.form_submit_button("🔮 Predict", use_container_width=True, type="primary")

        if submitted:
            input_data = {
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
                "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
                "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
            }
            input_df  = pd.DataFrame([input_data])
            t0 = time.time()
            prediction  = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            elapsed_ms  = (time.time() - t0) * 1000

            log_prediction(input_data, prediction, probability)

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="result-positive">
                        <h1>⚠️ High Risk</h1>
                        <h3>Heart Disease Detected</h3>
                        <p>Confidence: <b>{probability:.1%}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-negative">
                        <h1>✅ Low Risk</h1>
                        <h3>No Heart Disease Detected</h3>
                        <p>Confidence: <b>{1 - probability:.1%}</b></p>
                    </div>
                    """, unsafe_allow_html=True)

            with c2:
                st.markdown("**Prediction Details**")
                st.json({
                    "prediction":        int(prediction),
                    "label":             "Disease" if prediction == 1 else "No Disease",
                    "disease_prob":      f"{probability:.4f}",
                    "no_disease_prob":   f"{1-probability:.4f}",
                    "inference_time_ms": f"{elapsed_ms:.2f}",
                })

            # Probability gauge
            fig, ax = plt.subplots(figsize=(6, 0.6))
            ax.set_facecolor("#1a1f2e")
            fig.patch.set_facecolor("#0f1117")
            color = "#ef5350" if probability > 0.5 else "#81c784"
            ax.barh([0], [probability], color=color, height=0.5)
            ax.barh([0], [1 - probability], left=[probability], color="#2a2f3e", height=0.5)
            ax.set_xlim(0, 1)
            ax.axis("off")
            ax.set_title(f"Disease Probability: {probability:.1%}", color="#e0e0e0", fontsize=10)
            st.pyplot(fig, use_container_width=True)
            plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Dashboard":
    st.markdown("# 📊 Model Performance Dashboard")
    st.markdown("---")

    if not metrics:
        st.warning("No metrics found. Run `python src/train.py` first.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        cols = [c1, c2, c3, c4, c5]
        metric_items = [
            ("Accuracy",  "accuracy"),
            ("Precision", "precision"),
            ("Recall",    "recall"),
            ("F1-Score",  "f1_score"),
            ("ROC-AUC",   "roc_auc"),
        ]
        for col, (label, key) in zip(cols, metric_items):
            with col:
                val = metrics.get(key, 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{label}</h3>
                    <h2>{val:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs(["Confusion Matrix", "ROC Curve", "Metrics Bar", "Feature Importance"])

        vis_dir = os.path.join(ROOT, "model")
        with tab1:
            p = os.path.join(vis_dir, "confusion_matrix.png")
            if os.path.exists(p): st.image(p, use_container_width=True)
            else: st.info("Run `python src/evaluate.py` to generate visualizations.")

        with tab2:
            p = os.path.join(vis_dir, "roc_curve.png")
            if os.path.exists(p): st.image(p, use_container_width=True)
            else: st.info("Run `python src/evaluate.py` to generate visualizations.")

        with tab3:
            p = os.path.join(vis_dir, "metrics_bar.png")
            if os.path.exists(p): st.image(p, use_container_width=True)
            else: st.info("Run `python src/evaluate.py` to generate visualizations.")

        with tab4:
            p = os.path.join(vis_dir, "feature_importance.png")
            if os.path.exists(p): st.image(p, use_container_width=True)
            else: st.info("Run `python src/evaluate.py` to generate visualizations.")

        st.markdown("---")
        st.markdown("### Training Details")
        detail_cols = ["cv_mean_accuracy", "cv_std_accuracy", "train_samples", "test_samples", "training_time_s"]
        detail_data = {k: metrics.get(k, "N/A") for k in detail_cols}
        st.json(detail_data)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗄️ Dataset":
    st.markdown("# 🗄️ Dataset Explorer")
    st.markdown("**UCI Cleveland Heart Disease Dataset**  •  303 Records  •  13 Features")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total Records", len(df))
    with c2: st.metric("Disease Cases", int(df[TARGET_COLUMN].sum()))
    with c3: st.metric("Healthy Cases", int((df[TARGET_COLUMN] == 0).sum()))

    st.markdown("---")
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("### Statistical Summary")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("### Target Distribution")
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1f2e")
    counts = df[TARGET_COLUMN].value_counts()
    bars = ax.bar(["No Disease", "Disease"], counts.values, color=["#81c784", "#ef5350"], edgecolor="#0f1117", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, str(val),
                ha="center", va="bottom", color="#e0e0e0", fontweight="bold")
    ax.set_facecolor("#1a1f2e")
    ax.tick_params(colors="#e0e0e0")
    ax.spines[:].set_color("#2a2f3e")
    ax.set_title("Class Distribution", color="#e0e0e0")
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MONITORING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Monitoring":
    st.markdown("# 📋 Prediction Monitoring")
    st.markdown("---")

    if os.path.exists(LOG_PATH):
        logs = []
        with open(LOG_PATH) as f:
            for line in f:
                try: logs.append(json.loads(line))
                except: pass

        if logs:
            log_df = pd.DataFrame([{
                "Timestamp":   l["timestamp"],
                "Prediction":  "Disease" if l["prediction"] == 1 else "No Disease",
                "Probability": f"{l['probability']:.4f}",
            } for l in logs])
            st.metric("Total Predictions Served", len(logs))
            st.metric("Disease Predictions", sum(1 for l in logs if l["prediction"] == 1))
            st.markdown("### Recent Predictions")
            st.dataframe(log_df.tail(50), use_container_width=True)
        else:
            st.info("No predictions logged yet. Use the Predict page to generate logs.")
    else:
        st.info("No prediction log found. Logs will appear here after the first prediction.")

    st.markdown("---")
    st.markdown("### Monitoring Strategy")
    mon_path = os.path.join(ROOT, "monitoring", "monitoring.md")
    if os.path.exists(mon_path):
        with open(mon_path) as f:
            st.markdown(f.read())
