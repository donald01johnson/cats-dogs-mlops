"""
tests/test_preprocess.py
========================
Unit tests for data preprocessing functions.

Tests:
    1. test_resize_output_shape         - Transform produces correct tensor shape
    2. test_train_val_test_split_ratios - Split logic produces correct proportions
    3. test_augmentation_does_not_crash - All augmentations run without error
    4. test_normalize_range             - Normalized tensor has correct stats
    5. test_val_transform_no_augment    - Val transform differs from train transform
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np
from PIL import Image

from src.data.dataset import get_transforms, CLASS_TO_IDX


# ── Test 1: Resize output shape ───────────────────────────────────────────────
def test_resize_output_shape(sample_image_pil):
    """
    GIVEN a PIL image of any size
    WHEN  the training transform is applied
    THEN  the output tensor shape must be (3, 224, 224)
    """
    transform = get_transforms("train")
    tensor    = transform(sample_image_pil)

    assert isinstance(tensor, torch.Tensor), \
        "Transform output must be a torch.Tensor"
    assert tensor.shape == (3, 224, 224), \
        f"Expected shape (3, 224, 224), got {tensor.shape}"


# ── Test 2: Train/val/test split ratios ───────────────────────────────────────
def test_train_val_test_split_ratios():
    """
    GIVEN a list of 1000 dummy file paths
    WHEN  the split function is applied (80/10/10)
    THEN  the resulting splits must have correct proportions
    """
    # Import the split helper from preprocess.py
    from src.data.preprocess import split_files

    dummy_files = [Path(f"image_{i}.jpg") for i in range(1000)]
    train, val, test = split_files(dummy_files, seed=42)

    total = len(train) + len(val) + len(test)
    assert total == 1000, f"Total split count mismatch: {total} != 1000"

    # Allow ±1% tolerance for rounding
    assert 790 <= len(train) <= 810, \
        f"Train size {len(train)} not in expected range [790, 810]"
    assert 95  <= len(val)   <= 105, \
        f"Val size {len(val)} not in expected range [95, 105]"
    assert 95  <= len(test)  <= 105, \
        f"Test size {len(test)} not in expected range [95, 105]"


# ── Test 3: Augmentation does not crash ───────────────────────────────────────
def test_augmentation_does_not_crash(sample_image_pil):
    """
    GIVEN a random PIL image
    WHEN  all training augmentations are applied
    THEN  no exception should be raised and output must be a valid tensor
    """
    transform = get_transforms("train")

    try:
        tensor = transform(sample_image_pil)
    except Exception as e:
        pytest.fail(f"Augmentation transform raised an exception: {e}")

    assert tensor is not None
    assert not torch.isnan(tensor).any(), \
        "Tensor contains NaN values after augmentation"
    assert not torch.isinf(tensor).any(), \
        "Tensor contains Inf values after augmentation"


# ── Test 4: Normalize produces correct tensor type ────────────────────────────
def test_normalize_output_is_float_tensor(sample_image_pil):
    """
    GIVEN a PIL image
    WHEN  val transform (resize + normalize) is applied
    THEN  output must be a float32 tensor with 3 channels
    """
    transform = get_transforms("val")
    tensor    = transform(sample_image_pil)

    assert tensor.dtype == torch.float32, \
        f"Expected float32 tensor, got {tensor.dtype}"
    assert tensor.shape[0] == 3, \
        f"Expected 3 channels (RGB), got {tensor.shape[0]}"


# ── Test 5: Val transform has no random augmentations ─────────────────────────
def test_val_transform_deterministic(sample_image_pil):
    """
    GIVEN the same PIL image
    WHEN  val transform is applied twice
    THEN  both outputs must be identical (no random augmentation in val)
    """
    transform = get_transforms("val")

    tensor1 = transform(sample_image_pil)
    tensor2 = transform(sample_image_pil)

    assert torch.allclose(tensor1, tensor2), \
        "Val transform is not deterministic — augmentation should not apply to val set"


# ── Test 6: Class label mapping is correct ────────────────────────────────────
def test_class_label_mapping():
    """
    GIVEN the CLASS_TO_IDX mapping
    THEN  cats must map to 0 and dogs must map to 1
    """
    assert CLASS_TO_IDX["cats"] == 0, \
        f"Expected cats=0, got cats={CLASS_TO_IDX['cats']}"
    assert CLASS_TO_IDX["dogs"] == 1, \
        f"Expected dogs=1, got dogs={CLASS_TO_IDX['dogs']}"
    assert len(CLASS_TO_IDX) == 2, \
        f"Expected exactly 2 classes, got {len(CLASS_TO_IDX)}"
