from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "anpr-camera-system")
    environment: str = os.getenv("ENVIRONMENT", "production")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    api_key: str | None = os.getenv("ANPR_API_KEY")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "8"))
    plate_pattern: str = os.getenv("PLATE_PATTERN", r"[A-Z0-9]{5,10}")
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.45"))
    request_timeout_s: float = float(os.getenv("REQUEST_TIMEOUT_S", "8"))
    ocr_engine: str = os.getenv("OCR_ENGINE", "tesseract")
    detector_engine: str = os.getenv("DETECTOR_ENGINE", "opencv")


settings = Settings()
