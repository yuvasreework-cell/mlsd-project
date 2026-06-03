# Monitoring Strategy — Heart Disease MLOps Pipeline

## Overview

This document describes the complete monitoring strategy for the Heart Disease Prediction MLOps application. Monitoring is divided into four layers: **Application**, **Container**, **Prediction**, and **Error** monitoring.

---

## 1. Application Monitoring

### Metrics to Track
| Metric | Tool | Threshold | Action |
|--------|------|-----------|--------|
| Response time | Streamlit logs | > 2 seconds | Alert |
| Memory usage | Container metrics | > 900 MB | Scale up |
| CPU usage | Container metrics | > 80% | Scale up |
| Uptime | Kubernetes probes | < 99.9% | Investigate |

### Health Endpoints
- **Streamlit health check**: `GET /_stcore/health`
- **Kubernetes liveness probe**: `/_stcore/health` (every 20s)
- **Kubernetes readiness probe**: `/_stcore/health` (every 10s)

### Logging Format
Application logs are written in structured JSON format:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "event": "prediction_served",
  "prediction": 1,
  "probability": 0.7823,
  "inference_time_ms": 12.4
}
```

---

## 2. Container Monitoring

### Docker / Kubernetes Commands

**View live container logs:**
```bash
# Docker
docker logs -f <container_id>

# Kubernetes
kubectl logs -f deployment/heart-disease-mlops

# All pods
kubectl logs -f -l app=heart-disease-mlops --all-containers
```

**Monitor resource usage:**
```bash
# Kubernetes resource stats
kubectl top pods -l app=heart-disease-mlops
kubectl top nodes

# Docker stats
docker stats <container_id>
```

**Check pod health:**
```bash
kubectl get pods -l app=heart-disease-mlops
kubectl describe pod <pod_name>
kubectl get events --sort-by=.metadata.creationTimestamp
```

**Deployment status:**
```bash
kubectl rollout status deployment/heart-disease-mlops
kubectl get deployment heart-disease-mlops -o wide
```

---

## 3. Prediction Monitoring

All predictions are logged to `monitoring/prediction_log.json` in JSON-Lines format.

### Prediction Log Schema
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "input": { "age": 54, "sex": 1, "cp": 2, ... },
  "prediction": 1,
  "probability": 0.7823
}
```

### Monitoring Dashboard (Streamlit)
The **Monitoring** page in the Streamlit app shows:
- Total predictions served
- Disease vs. No-Disease ratio
- Last 50 prediction records
- Probability distribution over time

### Model Drift Detection
Run this check weekly to detect data drift:
```python
# Load recent predictions
import json, pandas as pd
logs = [json.loads(l) for l in open("monitoring/prediction_log.json")]
df = pd.DataFrame([l["input"] for l in logs])

# Compare to training data distribution
# If mean differs by >2 standard deviations → alert
```

### Key Metrics to Watch
| Metric | Healthy Range | Action if Exceeded |
|--------|--------------|-------------------|
| Disease prediction rate | 40% – 60% | Investigate data drift |
| Mean probability score | 0.4 – 0.7 | Review model calibration |
| Predictions per hour | > 0 | Alert if no traffic |

---

## 4. Error Monitoring

### Common Errors and Responses

| Error | Likely Cause | Response |
|-------|-------------|----------|
| `FileNotFoundError: model.pkl` | Model not trained | Run `python src/train.py` |
| `KeyError` in prediction | Input schema mismatch | Validate input format |
| OOMKilled pod | Memory exceeded | Increase resource limits |
| Pod `CrashLoopBackOff` | App startup error | Check `kubectl logs` |
| Image pull error | Wrong tag / auth | Re-push to DockerHub |

### Error Log Parsing
```bash
# Find errors in Kubernetes logs
kubectl logs deployment/heart-disease-mlops | grep -i "error\|exception\|traceback"

# Count errors by type
kubectl logs deployment/heart-disease-mlops | python -c "
import sys, collections
lines = sys.stdin.readlines()
errors = [l for l in lines if 'ERROR' in l]
print(f'Total errors: {len(errors)}')
"
```

### Alerting Strategy (Manual / CI-based)
1. **Daily**: Check `kubectl get pods` — all pods must be `Running`
2. **Weekly**: Review prediction log distribution for drift
3. **On Deploy**: Check `kubectl rollout status` passes
4. **On Merge**: GitHub Actions CI must pass (green checkmark)

---

## 5. Kubernetes Self-Healing

The deployment is configured for automatic recovery:

```yaml
# Liveness probe — auto-restart on crash
livenessProbe:
  httpGet:
    path: /_stcore/health
    port: 8501
  initialDelaySeconds: 30
  periodSeconds: 20
  failureThreshold: 3

# Readiness probe — remove from LB if not ready
readinessProbe:
  httpGet:
    path: /_stcore/health
    port: 8501
  initialDelaySeconds: 15
  periodSeconds: 10
```

**HPA (Horizontal Pod Autoscaler):** Automatically scales from 2 to 5 pods when CPU > 70%.

---

## 6. Monitoring Checklist (for Report)

- [x] Structured JSON logging implemented in `app/streamlit_app.py`
- [x] Kubernetes liveness probe configured
- [x] Kubernetes readiness probe configured
- [x] HPA configured for auto-scaling
- [x] Prediction logs stored in `monitoring/prediction_log.json`
- [x] Monitoring dashboard in Streamlit app
- [x] GitHub Actions CI validates model quality on every push
- [x] Docker healthcheck configured in Dockerfile
