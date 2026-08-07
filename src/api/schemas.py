"""
src/api/schemas.py
==================
Pydantic request/response models for the Cats vs Dogs inference API.
"""

from pydantic import BaseModel, Field
from typing import Dict


class HealthResponse(BaseModel):
    """Response model for GET /health endpoint."""
    status: str = Field(..., example="ok")
    model_loaded: bool = Field(..., example=True)
    version: str = Field(..., example="1.0.0")
    device: str = Field(..., example="cpu")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "model_loaded": True,
                "version": "1.0.0",
                "device": "cpu"
            }
        }


class PredictResponse(BaseModel):
    """Response model for POST /predict endpoint."""
    label: str = Field(..., example="cat",
                       description="Predicted class: 'cat' or 'dog'")
    confidence: float = Field(..., example=0.923,
                               description="Confidence score of the predicted class (0.0 - 1.0)")
    class_probabilities: Dict[str, float] = Field(
        ...,
        example={"cat": 0.923, "dog": 0.077},
        description="Softmax probability for each class"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "label": "cat",
                "confidence": 0.923,
                "class_probabilities": {
                    "cat": 0.923,
                    "dog": 0.077
                }
            }
        }


class ErrorResponse(BaseModel):
    """Response model for error cases."""
    error: str = Field(..., example="Invalid image format")
    detail: str = Field(..., example="Only JPEG and PNG images are supported")
