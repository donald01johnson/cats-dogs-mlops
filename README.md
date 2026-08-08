# 🐱🐶 cats-dogs-mlops

> **End-to-end MLOps pipeline for binary image classification (Cats vs Dogs)**
> Built for a pet adoption platform that automatically classifies uploaded pet images.

---

## 🔗 Quick Links

| Resource | Link |
|---|---|
| 📁 **GitHub Repository** | `<!-- ADD YOUR GITHUB REPO URL HERE -->` |
| 🎬 **Demo Video** | `<!-- ADD YOUR DEMO VIDEO LINK HERE (< 5 min) -->` |
| 🐳 **Docker Hub Image** | [donald01johnson/cats-dogs-api](https://hub.docker.com/r/donald01johnson/cats-dogs-api) |
| 📊 **Dataset** | [Kaggle — Cats & Dogs Classification](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset) |

---

## 📚 Course Information

| Field | Details |
|---|---|
| **Course** | S1-25_AIMLCZG523 — MLOps |
| **Institution** | BITS Pilani (WILP) |
| **Assignment** | Assignment 2 — Total Marks: 50 |
| **Author** | Donald Johnson A. |
| **Platform** | Ubuntu 22.04 LTS |

---

## 🎯 Use Case

A pet adoption platform needs to automatically classify images uploaded by users into **Cat** or **Dog** categories. This project implements a full MLOps pipeline — from raw data ingestion to a monitored, auto-deployed REST API — covering model training, experiment tracking, containerization, CI/CD, and production monitoring.

---

## 📈 Results Summary

| Metric | Value |
|---|---|
| **Validation Accuracy** (training) | **80.62%** |
| **Post-Deploy Simulation Accuracy** | **86.67%** |
| **Post-Deploy F1-Score** | **0.87** |
| **API Avg Latency** | **27ms** |
| **API P95 Latency** | **31.6ms** |
| **Unit Tests** | **14 / 14 passed** |
| **CI Pipeline** | ✅ GitHub Actions (green) |
| **CD Pipeline** | ✅ Self-hosted runner (green) |

---

## 🚀 Milestones

| # | Title | Marks | Status |
|---|---|---|---|
| **M1** | Model Development & Experiment Tracking | 10 | ✅ Complete |
| **M2** | Model Packaging & Containerization | 10 | ✅ Complete |
| **M3** | CI Pipeline — Build, Test & Image | 10 | ✅ Complete |
| **M4** | CD Pipeline & Deployment | 10 | ✅ Complete |
| **M5** | Monitoring, Logs & Final Submission | 10 | ✅ Complete |
| | **TOTAL** | **50** | ✅ **50/50** |

---

## 🗂️ Project Structure

```
cats-dogs-mlops/
├── .github/workflows/
│   ├── ci.yml                  # CI: checkout → install → test → build → push image
│   └── cd.yml                  # CD: pull → deploy → smoke test
├── data/
│   ├── raw/                    # [DVC] Raw Kaggle dataset (~500MB, 24,998 images)
│   └── processed/              # [DVC] Pre-processed 224x224 images (train/val/test)
├── src/
│   ├── data/
│   │   ├── preprocess.py       # Resize to 224x224, augment, split 80/10/10
│   │   └── dataset.py          # PyTorch CatsDogsDataset class + DataLoader factory
│   ├── model/
│   │   ├── cnn.py              # SimpleCNN architecture (3 conv blocks + FC head)
│   │   └── train.py            # Training loop + MLflow experiment logging
│   ├── api/
│   │   ├── main.py             # FastAPI app: GET /health, POST /predict, GET /metrics
│   │   └── schemas.py          # Pydantic V2 request/response models
│   └── monitoring/
│       └── metrics.py          # Prometheus instrumentation + request/prediction tracking
├── tests/
│   ├── conftest.py             # Shared pytest fixtures (dummy model, test client)
│   ├── test_preprocess.py      # 6 unit tests: data preprocessing functions
│   └── test_inference.py       # 8 unit tests: model output + API endpoints
├── artifacts/
│   └── model.pt                # [DVC] Trained model checkpoint (197MB, 51M params)
├── docker/
│   └── Dockerfile              # Container spec (python:3.10-slim base)
├── deploy/
│   ├── docker-compose.yml      # Full stack: API + Prometheus
│   ├── prometheus.yml          # Prometheus scrape configuration
│   ├── smoke_test.sh           # Post-deploy: tests /health + /predict + /metrics
│   └── test_sample.jpg         # Sample image used in CD smoke test
├── scripts/
│   ├── simulate_requests.py    # Post-deploy performance simulation (M5)
│   └── create_dummy_model.py   # Creates dummy model.pt for CI environment
├── notebooks/
│   └── exploration.ipynb       # EDA and dataset exploration
├── logs/                       # Runtime API logs (Docker volume mount)
├── requirements.txt            # Pinned production dependencies
├── requirements-dev.txt        # Training + test dependencies
└── mlruns/                     # MLflow local experiment tracking store
```

---

## 🧱 Tech Stack

| Layer | Tool | Version |
|---|---|---|
| ML Framework | PyTorch (CPU) | 2.13.0 |
| Data Versioning | DVC | 3.51.2 |
| Experiment Tracking | MLflow | 2.13.2 |
| API Framework | FastAPI | 0.111.0 |
| Containerization | Docker + Compose | 24+ / v2 |
| CI/CD | GitHub Actions | — |
| Container Registry | Docker Hub | — |
| Deployment Target | Docker Compose (local) | v2 |
| Monitoring | Prometheus + Python logging | 2.52.0 |
| Testing | pytest | 7.4.4 |

---

## ⚙️ Local Setup

### Prerequisites

- Ubuntu 22.04 LTS
- Python 3.10+
- Docker & Docker Compose v2
- Git
- Kaggle CLI (for dataset download)

### 1. Clone the repository

```bash
# <!-- REPLACE WITH YOUR ACTUAL GITHUB REPO URL -->
git clone git@github.com:donald01johnson/cats-dogs-mlops.git
cd cats-dogs-mlops
```

### 2. Create and activate virtual environment

```bash
python3 -m venv mlops_cd
source mlops_cd/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip

# PyTorch CPU (CPU-only — no GPU required)
pip install torch==2.13.0+cpu torchvision==0.28.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Production dependencies
pip install -r requirements.txt

# Development + test dependencies
pip install -r requirements-dev.txt

# Fix DVC pathspec conflict (important!)
pip install "pathspec==0.11.2"
```

### 4. Configure Kaggle credentials

```bash
mkdir -p ~/.kaggle
# Paste your kaggle.json token content:
nano ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Verify:
kaggle datasets list | head -3
```

### 5. Download and preprocess dataset

```bash
# Download (~500MB)
kaggle datasets download -d bhavikjikadara/dog-and-cat-classification-dataset -p data/raw/
cd data/raw && unzip dog-and-cat-classification-dataset.zip && cd ../..

# Track raw data with DVC
dvc add data/raw/

# Preprocess (resize 224x224, split 80/10/10)
PYTHONPATH=. python src/data/preprocess.py
```

**Expected output:**
```
Train  : 19998 images  (80.0%)
Val    : 2498  images  (10.0%)
Test   : 2502  images  (10.0%)
Total  : 24998 images processed
```

### 6. Train model

```bash
# ~4 hours on CPU for 10 epochs — run overnight
PYTHONPATH=. python src/model/train.py
```

**Expected result:**
```
Best Val Accuracy : 80.62%
Model saved       : artifacts/model.pt
```

### 7. View MLflow experiment UI

```bash
mlflow ui
# Open browser at http://localhost:5000
```

---

## 🐳 Docker

### Build image locally (with real trained model)

```bash
# IMPORTANT: Use --no-cache to ensure real model.pt is baked in
docker build --no-cache -f docker/Dockerfile -t donald01johnson/cats-dogs-api:latest .
```

### Run container

```bash
docker run -d -p 8000:8000 --name cats-dogs-api donald01johnson/cats-dogs-api:latest
docker logs cats-dogs-api   # Verify model loaded
```

### Test endpoints

```bash
# Health check
curl http://localhost:8000/health

# Predict a cat
curl -X POST http://localhost:8000/predict \
  -F "file=@data/processed/test/cats/cats_00000.jpg"

# Predict a dog
curl -X POST http://localhost:8000/predict \
  -F "file=@data/processed/test/dogs/dogs_00000.jpg"

# Prometheus metrics
curl http://localhost:8000/metrics
```

### Push to Docker Hub

```bash
docker login -u donald01johnson
docker push donald01johnson/cats-dogs-api:latest
```

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check + model load status |
| `POST` | `/predict` | Upload image → returns label + confidence |
| `GET` | `/metrics` | Prometheus metrics (request count, latency) |
| `GET` | `/docs` | Auto-generated OpenAPI documentation |

### Sample Responses

**GET /health:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0",
  "device": "cpu"
}
```

**POST /predict:**
```json
{
  "label": "cat",
  "confidence": 0.8105,
  "class_probabilities": {
    "cat": 0.8105,
    "dog": 0.1895
  }
}
```

---

## 🧪 Running Tests

```bash
# Run all 14 unit tests
PYTHONPATH=. pytest tests/ -v --tb=short
```

**Expected:**
```
tests/test_inference.py::test_model_output_shape             PASSED
tests/test_inference.py::test_health_endpoint_status         PASSED
tests/test_inference.py::test_predict_returns_valid_label    PASSED
tests/test_preprocess.py::test_resize_output_shape           PASSED
tests/test_preprocess.py::test_train_val_test_split_ratios   PASSED
...
14 passed in 2.28s
```

---

## 🚢 Deployment (Docker Compose — Full Stack)

Deploys **API + Prometheus** together:

```bash
# Start full stack
docker compose -f deploy/docker-compose.yml up -d

# Check status
docker compose -f deploy/docker-compose.yml ps
```

**Services:**

| Service | Port | URL |
|---|---|---|
| cats-dogs-api | 8000 | http://localhost:8000 |
| cats-dogs-prometheus | 9090 | http://localhost:9090 |

```bash
# Run smoke tests (validates /health + /predict + /metrics)
./deploy/smoke_test.sh

# Stop stack
docker compose -f deploy/docker-compose.yml down
```

---

## 📡 Monitoring

### Prometheus

Open **http://localhost:9090** → Status → Targets → `cats-dogs-api` should show **UP**.

Useful Prometheus queries:

```
# Total predictions by label
predictions_total

# Request rate per second
rate(api_requests_total[1m])

# 95th percentile latency
histogram_quantile(0.95, rate(prediction_latency_ms_bucket[5m]))
```

### Structured Logging

```bash
# View API logs (real-time)
tail -f logs/api.log
```

Sample log entry:
```
2026-08-08 19:44:12 | INFO | cats_dogs_api | REQUEST | method=POST path=/predict status=200 duration=27.0ms
2026-08-08 19:44:12 | INFO | cats_dogs_api | PREDICTION | file=cats_00000.jpg label=cat confidence=0.8105 latency=27.0ms
```

---

## 🔁 CI/CD Pipeline

### CI — GitHub Actions (`.github/workflows/ci.yml`)

Triggers on every **push** and **pull request** to `main`:

```
Push to main
    │
    ├─► Job 1: Run Unit Tests
    │       └─ Install deps → create dummy model → pytest (14 tests)
    │
    └─► Job 2: Build & Push Docker Image  (only if tests pass)
            └─ docker build → docker push → donald01johnson/cats-dogs-api:latest
```

### CD — Self-Hosted Runner (`.github/workflows/cd.yml`)

Triggers automatically **after CI succeeds** on `main`:

```
CI succeeds
    │
    └─► CD Job: Deploy & Smoke Test  (runs on local Ubuntu laptop)
            ├─ docker pull donald01johnson/cats-dogs-api:latest
            ├─ docker compose up -d --force-recreate
            ├─ sleep 20  (wait for startup)
            └─ ./deploy/smoke_test.sh  (fail pipeline if any test fails)
```

---

## 🔐 Required GitHub Secrets

| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` | `donald01johnson` |
| `DOCKERHUB_TOKEN` | Docker Hub access token (Read/Write/Delete) |

Set at: **GitHub Repo → Settings → Secrets and Variables → Actions**

---

## 📁 Dataset

- **Source:** [Kaggle — Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
- **Size:** ~500MB (raw) | ~25,000 images
- **Classes:** Cat (label=0), Dog (label=1) — perfectly balanced
- **Split:** 80% Train (19,998) / 10% Val (2,498) / 10% Test (2,502)
- **Pre-processing:** Resize to 224×224 RGB + augmentation (RandomHorizontalFlip, RandomRotation, ColorJitter)
- **Versioning:** Managed by DVC — NOT committed to Git

---

## ⚠️ CI/CD Model Artifact Note

The trained model (`artifacts/model.pt`, **197MB**) is tracked by **DVC** and not stored in Git. The CI pipeline uses a randomly initialized dummy model for building and testing the Docker image.

**For production deployment with the real trained model:**

```bash
# Rebuild locally with real model.pt (use --no-cache!)
docker build --no-cache -f docker/Dockerfile -t donald01johnson/cats-dogs-api:latest .

# Push to Docker Hub
docker push donald01johnson/cats-dogs-api:latest

# Redeploy
docker compose -f deploy/docker-compose.yml up -d --force-recreate
```

---

## 📄 License

For academic use only — BITS Pilani AIMLCZG523 Assignment 2.
