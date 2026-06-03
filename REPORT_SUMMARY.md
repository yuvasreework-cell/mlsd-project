# REPORT SUMMARY — Heart Disease Prediction MLOps Pipeline

## Project Title
**Heart Disease Prediction using an End-to-End MLOps Pipeline**

## Live Deployment
- **Streamlit App:** https://mlsd-project-mbxbcwepph6caqrngbfm3d.streamlit.app/
- **GitHub Repository:** https://github.com/yuvasreework-cell/mlsd-project
- **DockerHub Image:** https://hub.docker.com/r/yuvasreedock/heart-disease-mlops

## Abstract
This project implements a complete Machine Learning System Design (MLSD) pipeline for predicting heart disease using the UCI Cleveland Heart Disease Dataset. The pipeline covers data preprocessing, model training and evaluation, web application deployment via Streamlit, containerization with Docker, orchestration with Kubernetes, and continuous integration with GitHub Actions.

---

## Report Assets — Screenshot Guide

Place each screenshot in the corresponding section of your university report.

---

### Chapter 1: Project Setup

| File | Description | Report Section |
|------|-------------|----------------|
| `01_project_structure.png` | Clean MLOps repository structure showing all directories and files | Section 1.1 — Project Organization |
| `09_github_repository.png` | GitHub repository page showing commit history and file structure | Section 1.2 — Version Control |

**Caption for 01:** *Figure 1.1: MLOps repository structure for the Heart Disease Prediction project, organized following industry best practices.*

**Caption for 09:** *Figure 1.2: GitHub repository (yuvasreework-cell/mlsd-project) showing the complete project structure with CI badge.*

---

### Chapter 2: Dataset & Preprocessing

| File | Description | Report Section |
|------|-------------|----------------|
| `02_dataset_preview.png` | First 8 rows of the UCI Cleveland Heart Disease dataset | Section 2.1 — Dataset Description |

**Caption for 02:** *Figure 2.1: UCI Cleveland Heart Disease dataset preview — 303 patient records with 13 clinical features. Target column: 0 = No Disease, 1 = Disease.*

> **Dataset Source:** [UCI ML Repository — Heart Disease](https://archive.ics.uci.edu/ml/datasets/heart+disease)
> 
> **Preprocessing Steps:**
> 1. Load from UCI URL or local CSV
> 2. Replace missing values ('?') and drop null rows
> 3. Binarize target: values > 0 → 1 (Disease)
> 4. Apply `StandardScaler` to numeric features
> 5. Apply `OneHotEncoder` to categorical features
> 6. Split: 80% train, 20% test (stratified)

---

### Chapter 3: Model Training & Evaluation

| File | Description | Report Section |
|------|-------------|----------------|
| `03_training_output.png` | Terminal output from running `python src/train.py` | Section 3.1 — Training Process |
| `04_metrics_table.png` | Table of all evaluation metrics (accuracy, precision, recall, F1, ROC-AUC) | Section 3.2 — Model Evaluation |
| `05_confusion_matrix.png` | Confusion matrix heatmap on the test set | Section 3.3 — Confusion Matrix |
| `06_model_comparison.png` | Bar chart comparing Logistic Regression vs Random Forest baseline | Section 3.4 — Model Selection |

**Caption for 03:** *Figure 3.1: Training output showing 5-fold cross-validation results and final test-set evaluation metrics.*

**Caption for 04:** *Figure 3.2: Model evaluation metrics table. Logistic Regression achieves ~84% accuracy and ~91% ROC-AUC on the heart disease test set.*

**Caption for 05:** *Figure 3.3: Confusion matrix showing true positive, true negative, false positive, and false negative predictions on the 61-sample test set.*

**Caption for 06:** *Figure 3.4: Model comparison between Logistic Regression (selected) and Random Forest (baseline). Logistic Regression was chosen for its interpretability and sufficient performance.*

> **Model Architecture:**
> - Algorithm: `LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)`
> - Pipeline: `ColumnTransformer → LogisticRegression` (via `sklearn.Pipeline`)
> - Cross-validation: `StratifiedKFold(n_splits=5)`
> - Serialization: `joblib.dump(pipeline, 'model/model.pkl')`

---

### Chapter 4: Streamlit Application

| File | Description | Report Section |
|------|-------------|----------------|
| `07_streamlit_homepage.png` | Streamlit app home page with metrics overview and pipeline description | Section 4.1 — Application Design |
| `08_prediction_result.png` | Prediction result page showing disease risk assessment for a patient | Section 4.2 — Prediction Interface |

**Caption for 07:** *Figure 4.1: Streamlit application home page showing the MLOps pipeline overview and key performance metrics.*

**Caption for 08:** *Figure 4.2: Prediction result showing high risk (78.2% probability) for a patient with typical angina and elevated oldpeak value.*

> **Streamlit Features:**
> - 🏠 Home — Pipeline overview and quick metrics
> - 🔮 Predict — Interactive patient data form with 13 clinical inputs
> - 📊 Model Dashboard — Visualizations (confusion matrix, ROC curve, feature importance)
> - 🗄️ Dataset — Data preview and class distribution chart
> - 📋 Monitoring — Prediction log viewer and monitoring documentation

---

### Chapter 5: Containerization (Docker)

| File | Description | Report Section |
|------|-------------|----------------|
| `11_dockerfile.png` | Multi-stage Dockerfile content | Section 5.1 — Dockerfile |
| `12_docker_build_success.png` | Terminal output of `docker build` and `docker push` | Section 5.2 — Docker Build |
| `13_dockerhub_repository.png` | DockerHub repository page showing the pushed image | Section 5.3 — DockerHub |

**Caption for 11:** *Figure 5.1: Multi-stage Dockerfile — Stage 1 (builder) installs dependencies; Stage 2 (runner) creates a lean, secure production image running as a non-root user.*

**Caption for 12:** *Figure 5.2: Docker build and push output. Image successfully built in 47.3 seconds and pushed to yuvasreedock/heart-disease-mlops:latest.*

**Caption for 13:** *Figure 5.3: DockerHub repository showing the published image with tag history, image size (~185 MB), and pull command.*

> **Docker Commands:**
> ```bash
> # Build image
> docker build -t yuvasreedock/heart-disease-mlops:latest .
> 
> # Run locally
> docker run -p 8501:8501 yuvasreedock/heart-disease-mlops:latest
> 
> # Push to DockerHub
> docker push yuvasreedock/heart-disease-mlops:latest
> ```

---

### Chapter 6: Kubernetes Deployment

| File | Description | Report Section |
|------|-------------|----------------|
| `14_kubernetes_deployment_yaml.png` | deployment.yaml content showing replicas, probes, and resource limits | Section 6.1 — Deployment Manifest |
| `15_kubernetes_service_yaml.png` | service.yaml content showing NodePort and HPA configuration | Section 6.2 — Service Manifest |
| `16_kubectl_get_pods.png` | `kubectl get pods` output showing 2 running replicas | Section 6.3 — Pod Status |
| `17_kubectl_get_services.png` | `kubectl get services` and `kubectl get hpa` output | Section 6.4 — Service Status |

**Caption for 14:** *Figure 6.1: Kubernetes Deployment manifest — 2 replicas with rolling update strategy, liveness/readiness probes, and CPU/memory resource limits.*

**Caption for 15:** *Figure 6.2: Kubernetes Service manifest — NodePort exposing port 30080, with Horizontal Pod Autoscaler scaling 2→5 pods at 70% CPU.*

**Caption for 16:** *Figure 6.3: kubectl output showing both deployment pods in Running state with 0 restarts.*

**Caption for 17:** *Figure 6.4: Service and HPA status — application accessible at port 30080, auto-scaling at 18% CPU utilization.*

> **Kubernetes Commands:**
> ```bash
> kubectl apply -f k8s/deployment.yaml
> kubectl apply -f k8s/service.yaml
> kubectl get pods -l app=heart-disease-mlops
> kubectl get services
> minikube service heart-disease-service --url
> ```

---

### Chapter 7: CI/CD Pipeline

| File | Description | Report Section |
|------|-------------|----------------|
| `10_github_actions_success.png` | GitHub Actions workflow run showing all jobs passed | Section 7.1 — GitHub Actions |

**Caption for 10:** *Figure 7.1: GitHub Actions CI pipeline showing both the Test & Validate job (2m 12s) and Docker Build & Push job (35s) completed successfully.*

> **Pipeline Stages:**
> 1. `test` — Checkout → Install → Validate structure → Train → Run pytest
> 2. `build` — Build Docker image → Push to DockerHub (main branch only)

---

### Chapter 8: Monitoring

| File | Description | Report Section |
|------|-------------|----------------|
| `18_monitoring_plan.png` | Four-quadrant monitoring strategy visualization | Section 8.1 — Monitoring Strategy |

**Caption for 18:** *Figure 8.1: Monitoring strategy covering four areas: Application (response time, uptime), Container (kubectl logs, resource usage), Prediction (drift detection, probability monitoring), and Error monitoring.*

---

### Chapter 9: Architecture

| File | Description | Report Section |
|------|-------------|----------------|
| `19_architecture_diagram.png` | Complete MLOps architecture diagram showing all pipeline stages | Section 9.1 — System Architecture |
| `20_final_pipeline_diagram.png` | Linear pipeline diagram showing all 11 stages from dataset to monitoring | Section 9.2 — Pipeline Overview |

**Caption for 19:** *Figure 9.1: MLOps system architecture showing the complete flow from data ingestion through preprocessing, training, evaluation, CI/CD, Docker containerization, Kubernetes deployment, and monitoring.*

**Caption for 20:** *Figure 9.2: End-to-end MLOps pipeline with 11 stages: Dataset → Preprocessing → Training → Evaluation → Serialization → Git → GitHub Actions → Docker → Kubernetes → Streamlit → Monitoring.*

---

## Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.11 | Core development |
| ML Framework | Scikit-Learn 1.5 | Model pipeline |
| Data Processing | Pandas, NumPy | Data wrangling |
| Visualization | Matplotlib, Seaborn | Report charts |
| Web Application | Streamlit 1.35 | User interface |
| Containerization | Docker (multi-stage) | Reproducible deployment |
| Orchestration | Kubernetes + HPA | Scalable serving |
| CI/CD | GitHub Actions | Automated testing & deployment |
| Registry | DockerHub | Image storage |
| Monitoring | JSON logs + K8s probes | Production observability |

---

## Model Performance Summary

| Metric | Score |
|--------|-------|
| Accuracy | ~84% |
| Precision | ~82% |
| Recall | ~88% |
| F1-Score | ~85% |
| ROC-AUC | ~91% |
| CV Mean Accuracy | ~82% |
| Training Time | < 2 seconds |

---

## Reproducibility Instructions

```bash
# 1. Clone repository
git clone https://github.com/yuvasreework-cell/mlsd-project.git
cd mlsd-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model (generates model/model.pkl and model/metrics.json)
python src/train.py

# 4. Generate visualizations (generates model/*.png)
python src/evaluate.py

# 5. Generate all report assets
python generate_report_assets.py

# 6. Run Streamlit app locally
streamlit run app/streamlit_app.py

# 7. Run tests
pytest tests/test_pipeline.py -v

# 8. Build Docker image
docker build -t yuvasreedock/heart-disease-mlops:latest .
```
