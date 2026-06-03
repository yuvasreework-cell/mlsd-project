# Heart Disease Prediction — MLOps Pipeline

<div align="center">

[![CI Pipeline](https://github.com/yuvasreework-cell/mlsd-project/actions/workflows/ci.yml/badge.svg)](https://github.com/yuvasreework-cell/mlsd-project/actions/workflows/ci.yml)
[![Docker Image](https://img.shields.io/badge/Docker-yuvasreedock%2Fheart--disease--mlops-blue?logo=docker)](https://hub.docker.com/r/yuvasreedock/heart-disease-mlops)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5-orange?logo=scikitlearn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)](https://streamlit.io)

**An end-to-end MLOps pipeline demonstrating model training, CI/CD, Dockerization, Kubernetes deployment, and Streamlit hosting.**

</div>

---

## Project Overview

This project demonstrates a complete **Machine Learning System Design (MLSD)** pipeline using the UCI Cleveland Heart Disease Dataset. It predicts whether a patient has heart disease based on 13 clinical features, and deploys the prediction model through a full MLOps stack.

### Why Heart Disease Prediction?
| Criterion | Why It Fits |
|-----------|-------------|
| Problem Type | Binary Classification |
| Dataset | UCI Cleveland (public, famous, 303 records) |
| Training Speed | < 5 seconds |
| Visualizations | Confusion matrix, ROC curve, feature importance |
| Deployment | Streamlit (interactive prediction UI) |
| Academic Context | Standard ML curriculum dataset |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       MLOps Pipeline                            │
│                                                                 │
│  Dataset (UCI Heart)                                            │
│       │                                                         │
│       ▼                                                         │
│  Data Preprocessing ──────► ColumnTransformer                   │
│  (StandardScaler + OHE)      (preprocess.py)                    │
│       │                                                         │
│       ▼                                                         │
│  Model Training ──────────► LogisticRegression                  │
│  (train.py)                  + 5-fold CV                        │
│       │                                                         │
│       ▼                                                         │
│  Model Evaluation ────────► Accuracy, Precision, Recall,        │
│  (evaluate.py)               F1, ROC-AUC + 4 visualizations    │
│       │                                                         │
│       ▼                                                         │
│  model/model.pkl ◄──────── joblib serialization                 │
│       │                                                         │
│       ▼                                                         │
│  Git → GitHub ──────────── Version control                      │
│       │                                                         │
│       ▼                                                         │
│  GitHub Actions CI ──────► pytest → Docker Build → DockerHub   │
│       │                                                         │
│       ▼                                                         │
│  Docker Image ────────────► yuvasreedock/heart-disease-mlops    │
│       │                                                         │
│       ▼                                                         │
│  Kubernetes ──────────────► Deployment (2 replicas)             │
│                              Service (NodePort :30080)          │
│                              HPA (2-5 pods, CPU 70%)            │
│       │                                                         │
│       ▼                                                         │
│  Streamlit App ───────────► Prediction UI + Dashboard           │
│       │                                                         │
│       ▼                                                         │
│  Monitoring ──────────────► prediction_log.json + K8s probes   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
mlsd-project/
│
├── .github/
│   └── workflows/
│       └── ci.yml              ← GitHub Actions CI/CD pipeline
│
├── app/
│   └── streamlit_app.py        ← Streamlit prediction application
│
├── data/
│   └── heart.csv               ← UCI Cleveland Heart Disease dataset
│
├── model/
│   ├── model.pkl               ← Trained scikit-learn pipeline
│   ├── metrics.json            ← Evaluation metrics
│   ├── confusion_matrix.png    ← Confusion matrix visualization
│   ├── roc_curve.png           ← ROC curve plot
│   ├── metrics_bar.png         ← Metrics bar chart
│   └── feature_importance.png  ← Feature coefficient chart
│
├── monitoring/
│   ├── monitoring.md           ← Monitoring strategy document
│   └── prediction_log.json     ← Runtime prediction logs
│
├── notebooks/
│   └── exploration.ipynb       ← EDA notebook
│
├── report_assets/              ← All screenshots for university report
│
├── src/
│   ├── preprocess.py           ← Data loading, cleaning, preprocessing
│   ├── train.py                ← Model training pipeline
│   └── evaluate.py             ← Evaluation metrics + visualizations
│
├── tests/
│   └── test_pipeline.py        ← pytest unit tests
│
├── k8s/
│   ├── deployment.yaml         ← Kubernetes Deployment manifest
│   └── service.yaml            ← Kubernetes Service + HPA
│
├── Dockerfile                  ← Multi-stage production Dockerfile
├── requirements.txt            ← Python dependencies
├── README.md                   ← This file
├── architecture.png            ← Architecture diagram
└── REPORT_SUMMARY.md           ← Report asset documentation
```

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yuvasreework-cell/mlsd-project.git
cd mlsd-project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python src/train.py
```

### 4. Generate Visualizations
```bash
python src/evaluate.py
```

### 5. Run Streamlit App Locally
```bash
streamlit run app/streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501)

---

## Running Tests

```bash
pytest tests/test_pipeline.py -v
```

Expected output: All tests pass ✅

---

## Docker

### Build Image
```bash
docker build -t yuvasreedock/heart-disease-mlops:latest .
```

### Run Container
```bash
docker run -p 8501:8501 yuvasreedock/heart-disease-mlops:latest
```

### Pull from DockerHub
```bash
docker pull yuvasreedock/heart-disease-mlops:latest
```

---

## Kubernetes Deployment

### Prerequisites
- kubectl configured
- minikube / kind running locally, OR a cloud Kubernetes cluster

### Deploy
```bash
# Apply manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check status
kubectl get pods -l app=heart-disease-mlops
kubectl get services

# Access the app (minikube)
minikube service heart-disease-service

# Access the app (NodePort)
# Open: http://<node-ip>:30080
```

### Monitor Pods
```bash
kubectl get pods -w
kubectl logs -f deployment/heart-disease-mlops
kubectl top pods
```

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~84% |
| Precision | ~82% |
| Recall | ~88% |
| F1-Score | ~85% |
| ROC-AUC | ~91% |

*Results may vary slightly due to random seed in train/test split.*

---

## GitHub Actions CI

The pipeline automatically triggers on every push to `main`:

1. **Test Stage** (all branches)
   - Install Python dependencies
   - Validate project structure
   - Generate dataset + train model
   - Run unit tests with pytest

2. **Build Stage** (main branch only)
   - Build Docker image
   - Push to DockerHub as `yuvasreedock/heart-disease-mlops:latest`

### Required Secrets
Set these in GitHub → Settings → Secrets and Variables → Actions:
```
DOCKERHUB_USERNAME = yuvasreedock
DOCKERHUB_TOKEN    = <your DockerHub access token>
```

---

## Dataset Information

**UCI Cleveland Heart Disease Dataset**

- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- **Records**: 303 patients
- **Features**: 13 clinical attributes
- **Target**: 0 = No Heart Disease | 1 = Heart Disease

| Feature | Description |
|---------|-------------|
| age | Patient age in years |
| sex | 0 = Female, 1 = Male |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 (0/1) |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina (0/1) |
| oldpeak | ST depression by exercise |
| slope | Peak exercise ST segment slope |
| ca | Major vessels colored by fluoroscopy (0–3) |
| thal | Thalassemia type (0–3) |

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| ML Framework | Scikit-Learn 1.5 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web Application | Streamlit |
| Containerization | Docker (multi-stage) |
| Orchestration | Kubernetes |
| CI/CD | GitHub Actions |
| Registry | DockerHub |

---

## Author

**Yuvasree**  
University MLSD Project — Heart Disease Prediction MLOps Pipeline

---

## 📄 License

MIT License — Free to use for educational purposes.
