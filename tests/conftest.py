"""
tests/conftest.py
=================
Shared pytest fixtures for Cats vs Dogs MLOps test suite.

Fixtures:
    - dummy_model_path  : Creates a dummy model.pt if real one doesn't exist
    - sample_image_pil  : A random 224x224 RGB PIL Image
    - sample_tensor     : A (1, 3, 224, 224) torch.Tensor
    - loaded_model      : SimpleCNN with weights loaded
    - test_client       : FastAPI TestClient with app fully initialized
"""

import sys
import os
from pathlib import Path

# ── Ensure project root is on PYTHONPATH ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from src.model.cnn import SimpleCNN


# ── Fixture: Dummy model ───────────────────────────────────────────────────────
@pytest.fixture(scope="session", autouse=True)
def dummy_model_path():
    """
    Ensures artifacts/model.pt exists before any test runs.
    - If real trained model exists  → uses it (local dev)
    - If model does NOT exist       → saves a randomly initialized model (CI)
    This allows ALL tests to run in CI without needing DVC pull.
    """
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    model_path = artifacts_dir / "model.pt"

    if not model_path.exists():
        print(f"\n[conftest] model.pt not found — creating dummy model for CI at {model_path}")
        model = SimpleCNN(num_classes=2)
        torch.save(model.state_dict(), model_path)
    else:
        print(f"\n[conftest] Using existing model.pt at {model_path}")

    return model_path


# ── Fixture: Sample PIL Image ──────────────────────────────────────────────────
@pytest.fixture
def sample_image_pil():
    """
    Returns a random 224x224 RGB PIL Image.
    Used to test preprocessing transforms without needing real dataset images.
    """
    random_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(random_array, mode="RGB")


# ── Fixture: Sample Tensor ─────────────────────────────────────────────────────
@pytest.fixture
def sample_tensor():
    """
    Returns a random (1, 3, 224, 224) float tensor.
    Simulates a pre-processed image batch for model forward pass tests.
    """
    return torch.randn(1, 3, 224, 224)


# ── Fixture: Loaded model ──────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def loaded_model(dummy_model_path):
    """
    Returns a SimpleCNN with state_dict loaded from artifacts/model.pt.
    Set to eval() mode ready for inference tests.
    """
    model = SimpleCNN(num_classes=2)
    model.load_state_dict(
        torch.load(dummy_model_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


# ── Fixture: FastAPI TestClient ───────────────────────────────────────────────
@pytest.fixture(scope="session")
def test_client(dummy_model_path):
    """
    Returns a FastAPI TestClient with the full app initialized.
    The lifespan context (model loading) runs automatically.
    """
    from src.api.main import app
    with TestClient(app) as client:
        yield client


# ── Fixture: Real test image paths ────────────────────────────────────────────
@pytest.fixture
def sample_cat_image_path():
    """Path to first cat image in test set (if processed data exists)."""
    p = PROJECT_ROOT / "data" / "processed" / "test" / "cats"
    if p.exists():
        images = sorted(p.glob("*.jpg"))
        if images:
            return str(images[0])
    return None


@pytest.fixture
def sample_dog_image_path():
    """Path to first dog image in test set (if processed data exists)."""
    p = PROJECT_ROOT / "data" / "processed" / "test" / "dogs"
    if p.exists():
        images = sorted(p.glob("*.jpg"))
        if images:
            return str(images[0])
    return None
