# =============================================================
# Dockerfile — Cats vs Dogs Inference API
# Base  : python:3.10-slim (minimal footprint)
# Port  : 8000
# =============================================================

FROM python:3.10-slim

# ── Metadata ───────────────────────────────────────────────────
LABEL maintainer="Donald Johnson A."
LABEL description="Cats vs Dogs binary image classifier — FastAPI inference service"
LABEL version="1.0.0"

# ── System dependencies ────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ─────────────────────────────────────────
# Copy requirements first (Docker layer cache — only rebuilds if req changes)
COPY requirements.txt .

RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir \
        torch==2.13.0+cpu \
        torchvision==0.28.0+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application source code ────────────────────────────────
COPY src/ ./src/

# ── Copy trained model artifact ─────────────────────────────────
COPY artifacts/model.pt ./artifacts/model.pt

# ── Create logs directory ───────────────────────────────────────
RUN mkdir -p /app/logs

# ── Environment variables ───────────────────────────────────────
ENV MODEL_PATH=/app/artifacts/model.pt
ENV LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Expose API port ─────────────────────────────────────────────
EXPOSE 8000

# ── Health check (Docker built-in) ──────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start the API server ────────────────────────────────────────
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
