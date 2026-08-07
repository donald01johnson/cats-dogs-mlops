"""
src/monitoring/metrics.py
=========================
Prometheus instrumentation and request/prediction tracking.

- Exposes /metrics endpoint via prometheus-fastapi-instrumentator
- Tracks: request count, request latency, prediction count by label
"""

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

# ── Custom Prometheus metrics ──────────────────────────────────────────────────

# Count predictions broken down by label (cat / dog)
PREDICTION_COUNTER = Counter(
    name="predictions_total",
    documentation="Total number of predictions made, by label",
    labelnames=["label"],
)

# Count errors
ERROR_COUNTER = Counter(
    name="prediction_errors_total",
    documentation="Total number of prediction errors",
)

# Latency histogram for inference only (milliseconds)
PREDICTION_LATENCY = Histogram(
    name="prediction_latency_ms",
    documentation="Inference latency in milliseconds",
    buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
)

# Request counter by method + path + status
REQUEST_COUNTER = Counter(
    name="api_requests_total",
    documentation="Total API requests by method, path and status code",
    labelnames=["method", "path", "status_code"],
)

# Request latency histogram (milliseconds)
REQUEST_LATENCY = Histogram(
    name="api_request_latency_ms",
    documentation="Request latency in milliseconds",
    buckets=[5, 10, 25, 50, 100, 250, 500, 1000, 2500],
    labelnames=["path"],
)


# ── Setup function called from main.py ────────────────────────────────────────

def setup_metrics(app):
    """
    Attach the Prometheus instrumentator to the FastAPI app.
    Exposes the /metrics endpoint automatically.
    """
    Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics", "/"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# ── Tracking helpers called from main.py ──────────────────────────────────────

def track_prediction(label: str, latency_ms: float):
    """Record a successful prediction."""
    PREDICTION_COUNTER.labels(label=label).inc()
    PREDICTION_LATENCY.observe(latency_ms)


def track_request(method: str, path: str, status_code: int, latency_ms: float):
    """Record every API request."""
    REQUEST_COUNTER.labels(
        method=method,
        path=path,
        status_code=str(status_code)
    ).inc()
    REQUEST_LATENCY.labels(path=path).observe(latency_ms)


def track_error():
    """Record a prediction error."""
    ERROR_COUNTER.inc()
