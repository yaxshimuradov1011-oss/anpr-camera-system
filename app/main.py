from __future__ import annotations

from datetime import datetime, timezone
import logging

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from app.config import settings
from app.detectors.opencv_detector import OpenCVPlateDetector
from app.logging_setup import configure_logging
from app.models import ANPRResponse, ErrorResponse, HealthResponse
from app.ocr.tesseract_engine import TesseractOCREngine
from app.pipeline import ANPRPipeline
from app.security import api_key_dependency, require_api_key

configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="ANPR Camera System", version="1.0.0")
pipeline = ANPRPipeline(
    detector=OpenCVPlateDetector(),
    ocr_engine=TesseractOCREngine(settings.plate_pattern),
    plate_pattern=settings.plate_pattern,
    min_confidence=settings.min_confidence,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(service=settings.app_name, timestamp=datetime.now(timezone.utc))


@app.post(
    "/v1/anpr/detect",
    response_model=ANPRResponse,
    responses={401: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
async def detect(
    file: UploadFile = File(...),
    x_api_key: str | None = Depends(api_key_dependency),
) -> ANPRResponse:
    require_api_key(settings.api_key, x_api_key)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Uploaded file exceeds max size")

    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Unable to decode image")

    result = pipeline.run(frame)
    logger.info("request_id=%s detections=%d", result.request_id, len(result.predictions))
    return result
