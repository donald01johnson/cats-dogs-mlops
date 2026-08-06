# 🐱🐶 cats-dogs-mlops

> **End-to-end MLOps pipeline for binary image classification (Cats vs Dogs)**  
> Built for a pet adoption platform that automatically classifies uploaded pet images.

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

A pet adoption platform needs to automatically classify images uploaded by users into **Cat** or **Dog** categories. This project implements a full MLOps pipeline — from raw data ingestion to a monitored, auto-deployed REST API.

---

## 🗂️ Project Structure

```
cats-dogs-mlops/
├── .github/workflows/
│   ├── ci.yml                  # CI: test → build → push Docker image
│   └── cd.yml                  # CD: pull → deploy → smoke test
├── data/
│   ├── raw/                    # [DVC] Raw Kaggle dataset
│   └── processed/              # [DVC] Pre-processed 224x224 images
├── src/
│   ├── data/
│   │   ├── preprocess.py       # Resize, augment, split (80/10/10)
│   │   └── dataset.py          # PyTorch CatsDogsDataset class
│   ├── model/
│   │   ├── cnn.py              # SimpleCNN architecture
│   │   └── train.py            # Training loop + MLflow logging
│   ├── api/
│   │   ├── main.py             # FastAPI app (/health, /predict, /metrics)
│   │   └── schemas.py          # Pydantic request/response models
│   └── monitoring/
│       └── metrics.py          # Prometheus instrumentation + logging
├── tests/
│   ├── conftest.py             # Shared pytest fixtures
│   ├── test_preprocess.py      # Unit tests: data preprocessing
│   └── test_inference.py       # Unit tests: model + API endpoints
├── artifacts/
│   └── model.pt                # [DVC] Trained model checkpoint
├── docker/
│   └── Dockerfile              # Container spec (python:3.10-slim)
├── deploy/
│   ├── docker-compose.yml      # Deployment manifest
│   └── smoke_test.sh           # Post-deploy health + predict test
├── scripts/
│   └── simulate_requests.py    # Batch prediction simulation (M5)
├── notebooks/
│   └── exploration.ipynb       # EDA and initial experiments
├── logs/                       # Runtime API logs (Docker volume)
├── requirements.txt            # Pinned production dependencies
├── requirements-dev.txt        # Test/dev dependencies
├── dvc.yaml                    # DVC pipeline stages
└── mlruns/                     # MLflow local tracking store
```

---

## 🧱 Tech Stack

| Layer | Tool | Version |
|---|---|---|
| ML Framework | PyTorch (CPU) | 2.13.0 |
| Data Versioning | DVC | 3.51.2 |
| Experiment Tracking | MLflow | 2.13.2 |
| API Framework | FastAPI | 0.111.0 |
| Containerization | Docker + Compose | 24+ |
| CI/CD | GitHub Actions | — |
| Container Registry | Docker Hub | — |
| Deployment | Docker Compose | v2 |
| Monitoring | Prometheus + Python logging | — |
| Testing | pytest | 7.4.4 |

---

## 🚀 Milestones

| # | Title | Marks | Status |
|---|---|---|---|
| M1 | Model Development & Experiment Tracking | 10 | ⬜ |
| M2 | Model Packaging & Containerization | 10 | ⬜ |
| M3 | CI Pipeline — Build, Test & Image | 10 | ⬜ |
| M4 | CD Pipeline & Deployment | 10 | ⬜ |
| M5 | Monitoring, Logs & Final Submission | 10 | ⬜ |

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
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 4. Download dataset (requires Kaggle API key)
```bash
kaggle datasets download -d bhavikjikadara/dog-and-cat-classification-dataset -p data/raw/
cd data/raw && unzip dog-and-cat-classification-dataset.zip && cd ../..
```

### 5. Run preprocessing
```bash
PYTHONPATH=. python src/data/preprocess.py
```

### 6. Train model
```bash
PYTHONPATH=. python src/model/train.py
```

### 7. View MLflow experiment UI
```bash
mlflow ui
# Open browser at http://localhost:5000
```

---

## 🐳 Docker

### Build image locally
```bash
docker build -f docker/Dockerfile -t cats-dogs-api:latest .
```

### Run container
```bash
docker run -d -p 8000:8000 --name cats-dogs-api cats-dogs-api:latest
```

### Test endpoints
```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict -F "file=@sample.jpg"

# Prometheus metrics
curl http://localhost:8000/metrics
```

---

## 🧪 Running Tests

```bash
PYTHONPATH=. pytest tests/ -v --tb=short
```

---

## 🚢 Deployment (Docker Compose)

```bash
docker compose -f deploy/docker-compose.yml up -d
```

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check + model status |
| `POST` | `/predict` | Upload image → returns label + confidence |
| `GET` | `/metrics` | Prometheus metrics (request count, latency) |

### Sample Response — /predict
```json
{
  "label": "cat",
  "confidence": 0.923,
  "class_probabilities": {
    "cat": 0.923,
    "dog": 0.077
  }
}
```

---

## 📁 Dataset

- **Source:** [Kaggle — Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
- **Size:** ~500MB (raw)
- **Split:** 80% Train / 10% Validation / 10% Test
- **Pre-processing:** Resize to 224×224 RGB + augmentation
- **Tracking:** Managed by DVC (NOT committed to Git)

---

## 🔐 Required GitHub Secrets

| Secret | Description |
|---|---|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token (Read/Write) |

Set at: `GitHub Repo → Settings → Secrets and Variables → Actions`

---

## 📄 License

For academic use only — BITS Pilani AIMLCZG523 Assignment 2.
