# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Heart Disease Prediction MLOps
# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage build for a lean, secure production image
# Final image: python:3.11-slim (~180MB)
#
# Build:  docker build -t yuvasreedock/heart-disease-mlops:latest .
# Run:    docker run -p 8501:8501 yuvasreedock/heart-disease-mlops:latest
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

# Install build essentials (only in builder stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install/pkg --no-cache-dir -r requirements.txt


# ── Stage 2: Production runner ────────────────────────────────────────────────
FROM python:3.11-slim AS runner

LABEL maintainer="yuvasreedock"
LABEL version="1.0"
LABEL description="Heart Disease Prediction MLOps — Streamlit Application"

# Copy installed packages from builder
COPY --from=builder /install/pkg /usr/local

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy project files (order: rarely-changing first for layer caching)
COPY requirements.txt .
COPY src/ ./src/
COPY app/ ./app/
COPY model/ ./model/
COPY data/ ./data/
COPY monitoring/ ./monitoring/

# Create directories that may be written at runtime
RUN mkdir -p monitoring && \
    chown -R appuser:appuser /app

USER appuser

# Streamlit config — disable browser auto-open, set headless mode
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
