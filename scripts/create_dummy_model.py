"""
scripts/create_dummy_model.py
==============================
Creates a randomly initialized (untrained) model.pt for use in CI environments
where the real trained model is not available (DVC not pulled).

This script is called automatically by the GitHub Actions CI pipeline
BEFORE running pytest and BEFORE building the Docker image.

NOTE: This dummy model produces random predictions — it is NOT the trained model.
      For real inference, always use the model.pt produced by src/model/train.py.

Usage:
    python scripts/create_dummy_model.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.model.cnn import SimpleCNN


def create_dummy_model(force: bool = False):
    """
    Save a randomly initialized SimpleCNN to artifacts/model.pt.

    Args:
        force: If True, overwrites existing model.pt.
               If False, skips if model.pt already exists.
    """
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "model.pt"

    if model_path.exists() and not force:
        print(f"[create_dummy_model] model.pt already exists at {model_path} — skipping.")
        print("[create_dummy_model] Use --force to overwrite.")
        return

    print("[create_dummy_model] Creating dummy model (randomly initialized SimpleCNN)...")
    model = SimpleCNN(num_classes=2)
    torch.save(model.state_dict(), model_path)

    size_kb = model_path.stat().st_size / 1024
    print(f"[create_dummy_model] Saved to {model_path}  ({size_kb:.1f} KB)")
    print("[create_dummy_model] Done. This model is for CI/testing only.")


if __name__ == "__main__":
    force = "--force" in sys.argv
    create_dummy_model(force=force)
