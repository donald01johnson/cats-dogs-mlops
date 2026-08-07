"""
src/api/main.py
===============
FastAPI inference service for Cats vs Dogs binary classification.

Endpoints:
    GET  /health   — Service health check + model status
    POST /predict  — Upload image -> returns label + confidence
    GET  /metrics  — Prometheus metrics (request count, latency)

Usage (local):
    PYTHONPATH=. uvicorn src.api.main:app --reload --port 8000

Usage (Docker):
    docker run -p 8000:8000 cats-dogs-api:latest
"""

import io
import os
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from src.api.schemas import HealthResponse, PredictResponse, ErrorResponse
from src.model.cnn import SimpleCNN
from src.monitoring.metrics import (
    setup_metrics,
    track_prediction,
    track_request,
)

# ── Logging setup ──────────────────────────────────────────────────────────────
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),                          # stdout (docker logs)
        logging.FileHandler(log_dir / "api.log"),         # persistent file
    ]
)
logger = logging.getLogger("cats_dogs_api")

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH   = Path(os.getenv("MODEL_PATH", "artifacts/model.pt"))
API_VERSION  = "1.0.0"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IDX_TO_CLASS = {0: "cat", 1: "dog"}

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
MAX_FILE_SIZE_MB      = 10

# ── Inference transforms (same as val/test — NO augmentation) ──────────────────
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ── App state ──────────────────────────────────────────────────────────────────
app_state = {"model": None, "model_loaded": False}


# ── Lifespan — load model at startup, cleanup at shutdown ─────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("=" * 55)
    logger.info("  Cats vs Dogs API — Starting up")
    logger.info(f"  Device     : {DEVICE}")
    logger.info(f"  Model path : {MODEL_PATH}")
    logger.info("=" * 55)

    if not MODEL_PATH.exists():
        logger.error(f"Model file not found: {MODEL_PATH}")
        logger.error("Run 'python src/model/train.py' first!")
        app_state["model_loaded"] = False
    else:
        try:
            model = SimpleCNN(num_classes=2).to(DEVICE)
            model.load_state_dict(
                torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            )
            model.eval()
            app_state["model"] = model
            app_state["model_loaded"] = True
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            app_state["model_loaded"] = False

    yield  # App runs here

    # Shutdown
    logger.info("Cats vs Dogs API — Shutting down")
    app_state["model"] = None


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cats vs Dogs Classifier API",
    description=(
        "MLOps Assignment 2 — Binary image classification REST API. "
        "Upload an image and get back a cat/dog prediction with confidence score."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

# Setup Prometheus metrics
setup_metrics(app)


# ── Middleware — request logging ───────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response   = await call_next(request)
    duration   = (time.time() - start_time) * 1000  # ms

    logger.info(
        f"REQUEST | method={request.method} path={request.url.path} "
        f"status={response.status_code} duration={duration:.1f}ms "
        f"client={request.client.host if request.client else 'unknown'}"
    )
    track_request(request.method, request.url.path, response.status_code, duration)
    return response


# ── GET /health ────────────────────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns service status and whether the model is loaded.",
    tags=["Health"],
)
async def health():
    return HealthResponse(
        status="ok" if app_state["model_loaded"] else "degraded",
        model_loaded=app_state["model_loaded"],
        version=API_VERSION,
        device=str(DEVICE),
    )


# ── POST /predict ──────────────────────────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict Cat or Dog",
    description=(
        "Upload a JPEG or PNG image. "
        "Returns the predicted label (cat/dog), confidence score, "
        "and full class probabilities."
    ),
    tags=["Inference"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict(file: UploadFile = File(..., description="Image file (JPEG or PNG)")):
    # ── Guard: model loaded? ───────────────────────────────────────────────
    if not app_state["model_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check server logs."
        )

    # ── Guard: valid content type ──────────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. "
                   f"Only JPEG and PNG images are accepted."
        )

    # ── Read image bytes ───────────────────────────────────────────────────
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    # ── Guard: file size ───────────────────────────────────────────────────
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Maximum allowed: {MAX_FILE_SIZE_MB}MB."
        )

    # ── Open and preprocess image ──────────────────────────────────────────
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {e}")

    try:
        tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {e}")

    # ── Inference ──────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        model = app_state["model"]
        with torch.no_grad():
            logits       = model(tensor)                          # (1, 2)
            probs        = F.softmax(logits, dim=1).squeeze(0)   # (2,)
            pred_idx     = probs.argmax().item()
            confidence   = probs[pred_idx].item()
            label        = IDX_TO_CLASS[pred_idx]
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    latency_ms = (time.time() - t0) * 1000

    # ── Build class probabilities dict ─────────────────────────────────────
    class_probs = {
        IDX_TO_CLASS[i]: round(probs[i].item(), 4)
        for i in range(len(IDX_TO_CLASS))
    }

    # ── Log prediction (NO image bytes logged — privacy safe) ─────────────
    logger.info(
        f"PREDICTION | file={file.filename} label={label} "
        f"confidence={confidence:.4f} latency={latency_ms:.1f}ms"
    )
    track_prediction(label, latency_ms)

    return PredictResponse(
        label=label,
        confidence=round(confidence, 4),
        class_probabilities=class_probs,
    )


# ── Root redirect ──────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse(content={
        "message": "Cats vs Dogs Classifier API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
        "metrics": "/metrics",
    })
