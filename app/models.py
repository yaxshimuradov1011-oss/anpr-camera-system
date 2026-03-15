from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
    timestamp: datetime


class DetectionBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    score: float = Field(ge=0.0, le=1.0)


class PlatePrediction(BaseModel):
    plate: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: DetectionBox


class ANPRResponse(BaseModel):
    request_id: str
    predictions: list[PlatePrediction]
    elapsed_ms: int


class ErrorResponse(BaseModel):
    detail: str
