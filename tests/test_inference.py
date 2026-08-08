"""
tests/test_inference.py
=======================
Unit tests for model utility and API inference functions.

Tests:
    1. test_model_output_shape          - CNN forward pass returns (1, 2) logits
    2. test_model_output_is_tensor      - Output is a torch.Tensor
    3. test_health_endpoint_status      - GET /health returns 200
    4. test_health_endpoint_body        - GET /health body has correct fields
    5. test_predict_returns_valid_label - POST /predict returns cat or dog
    6. test_predict_confidence_range    - Confidence is between 0.0 and 1.0
    7. test_predict_probabilities_sum   - Class probabilities sum to ~1.0
    8. test_predict_rejects_non_image   - POST /predict rejects invalid file type
"""

import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np
from PIL import Image


# ── Test 1: Model output shape ────────────────────────────────────────────────
def test_model_output_shape(loaded_model, sample_tensor):
    """
    GIVEN a SimpleCNN model and a (1, 3, 224, 224) input tensor
    WHEN  a forward pass is performed
    THEN  the output shape must be (1, 2) — one logit per class
    """
    with torch.no_grad():
        output = loaded_model(sample_tensor)

    assert output.shape == (1, 2), \
        f"Expected output shape (1, 2), got {output.shape}"


# ── Test 2: Model output is tensor ───────────────────────────────────────────
def test_model_output_is_tensor(loaded_model, sample_tensor):
    """
    GIVEN a SimpleCNN model
    WHEN  a forward pass is performed
    THEN  the output must be a torch.Tensor with no NaN or Inf values
    """
    with torch.no_grad():
        output = loaded_model(sample_tensor)

    assert isinstance(output, torch.Tensor), \
        "Model output must be a torch.Tensor"
    assert not torch.isnan(output).any(), \
        "Model output contains NaN values"
    assert not torch.isinf(output).any(), \
        "Model output contains Inf values"


# ── Test 3: Health endpoint returns 200 ──────────────────────────────────────
def test_health_endpoint_status(test_client):
    """
    GIVEN the FastAPI app is running
    WHEN  GET /health is called
    THEN  HTTP status code must be 200
    """
    response = test_client.get("/health")
    assert response.status_code == 200, \
        f"Expected status 200, got {response.status_code}"


# ── Test 4: Health endpoint body ─────────────────────────────────────────────
def test_health_endpoint_body(test_client):
    """
    GIVEN the FastAPI app is running with model loaded
    WHEN  GET /health is called
    THEN  response body must contain status, model_loaded, version, device
    """
    response = test_client.get("/health")
    body     = response.json()

    assert "status"       in body, "Missing 'status' field in health response"
    assert "model_loaded" in body, "Missing 'model_loaded' field in health response"
    assert "version"      in body, "Missing 'version' field in health response"
    assert "device"       in body, "Missing 'device' field in health response"
    assert body["status"] == "ok",  f"Expected status='ok', got '{body['status']}'"
    assert body["model_loaded"] is True, "Expected model_loaded=True"


# ── Test 5: Predict returns valid label ──────────────────────────────────────
def test_predict_returns_valid_label(test_client):
    """
    GIVEN a random 224x224 RGB image uploaded to POST /predict
    WHEN  the inference runs
    THEN  the response label must be either 'cat' or 'dog'
    """
    # Create a random dummy image as bytes
    random_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img          = Image.fromarray(random_array, mode="RGB")
    buf          = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = test_client.post(
        "/predict",
        files={"file": ("test_image.jpg", buf, "image/jpeg")}
    )

    assert response.status_code == 200, \
        f"Expected status 200, got {response.status_code}. Body: {response.text}"

    body = response.json()
    assert "label" in body, "Missing 'label' field in predict response"
    assert body["label"] in ["cat", "dog"], \
        f"Expected label 'cat' or 'dog', got '{body['label']}'"


# ── Test 6: Predict confidence in range [0, 1] ───────────────────────────────
def test_predict_confidence_range(test_client):
    """
    GIVEN an image uploaded to POST /predict
    WHEN  the inference runs
    THEN  confidence score must be between 0.0 and 1.0 inclusive
    """
    random_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img          = Image.fromarray(random_array, mode="RGB")
    buf          = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = test_client.post(
        "/predict",
        files={"file": ("test_image.jpg", buf, "image/jpeg")}
    )
    body = response.json()

    assert "confidence" in body, "Missing 'confidence' field in predict response"
    confidence = body["confidence"]
    assert 0.0 <= confidence <= 1.0, \
        f"Confidence {confidence} is out of range [0.0, 1.0]"


# ── Test 7: Class probabilities sum to 1.0 ───────────────────────────────────
def test_predict_probabilities_sum_to_one(test_client):
    """
    GIVEN an image uploaded to POST /predict
    WHEN  the inference runs
    THEN  class_probabilities for cat + dog must sum to ~1.0
    """
    random_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img          = Image.fromarray(random_array, mode="RGB")
    buf          = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = test_client.post(
        "/predict",
        files={"file": ("test_image.jpg", buf, "image/jpeg")}
    )
    body  = response.json()
    probs = body["class_probabilities"]

    assert "cat" in probs, "Missing 'cat' in class_probabilities"
    assert "dog" in probs, "Missing 'dog' in class_probabilities"

    total = probs["cat"] + probs["dog"]
    assert abs(total - 1.0) < 0.01, \
        f"Probabilities sum to {total}, expected ~1.0"


# ── Test 8: Predict rejects invalid file type ────────────────────────────────
def test_predict_rejects_invalid_file_type(test_client):
    """
    GIVEN a text file uploaded to POST /predict (invalid type)
    WHEN  the request is processed
    THEN  the API must return HTTP 400 Bad Request
    """
    fake_text = io.BytesIO(b"this is not an image")

    response = test_client.post(
        "/predict",
        files={"file": ("test.txt", fake_text, "text/plain")}
    )

    assert response.status_code == 400, \
        f"Expected 400 for invalid file type, got {response.status_code}"
