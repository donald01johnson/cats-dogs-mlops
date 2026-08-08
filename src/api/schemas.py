"""
src/api/schemas.py
==================
Pydantic request/response models for the Cats vs Dogs inference API.
Updated for Pydantic V2 — uses model_config and json_schema_extra.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Dict


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "model_loaded": True,
                "version": "1.0.0",
                "device": "cpu"
            }
        }
    )

    status: str
    model_loaded: bool
    version: str
    device: str


class PredictResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "label": "cat",
                "confidence": 0.923,
                "class_probabilities": {"cat": 0.923, "dog": 0.077}
            }
        }
    )

    label: str
    confidence: float
    class_probabilities: Dict[str, float]


class ErrorResponse(BaseModel):
    error: str
    detail: str
