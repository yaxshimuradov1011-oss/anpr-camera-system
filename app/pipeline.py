from __future__ import annotations

import logging
import re
import time
from uuid import uuid4

import numpy as np

from app.detectors.base import PlateDetector
from app.models import ANPRResponse, DetectionBox, PlatePrediction
from app.ocr.base import OCREngine

logger = logging.getLogger(__name__)


class ANPRPipeline:
    def __init__(self, detector: PlateDetector, ocr_engine: OCREngine, plate_pattern: str, min_confidence: float) -> None:
        self._detector = detector
        self._ocr = ocr_engine
        self._pattern = re.compile(plate_pattern)
        self._min_confidence = min_confidence

    def run(self, image: np.ndarray) -> ANPRResponse:
        start = time.perf_counter()
        predictions: list[PlatePrediction] = []

        for candidate in self._detector.detect(image):
            roi = image[candidate.y1:candidate.y2, candidate.x1:candidate.x2]
            if roi.size == 0:
                continue

            plate_text, ocr_conf = self._ocr.read_text(roi)
            if not plate_text:
                continue

            normalized = plate_text.replace(" ", "")
            if not self._pattern.fullmatch(normalized):
                logger.debug("Rejected OCR text that does not match pattern: %s", normalized)
                continue

            total_conf = min(1.0, (candidate.score * 0.4) + (ocr_conf * 0.6))
            if total_conf < self._min_confidence:
                continue

            predictions.append(
                PlatePrediction(
                    plate=normalized,
                    confidence=round(total_conf, 3),
                    box=DetectionBox(
                        x1=candidate.x1,
                        y1=candidate.y1,
                        x2=candidate.x2,
                        y2=candidate.y2,
                        score=round(candidate.score, 3),
                    ),
                )
            )

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        deduped: dict[str, PlatePrediction] = {}
        for pred in predictions:
            current = deduped.get(pred.plate)
            if current is None or pred.confidence > current.confidence:
                deduped[pred.plate] = pred

        ordered = sorted(deduped.values(), key=lambda x: x.confidence, reverse=True)
        return ANPRResponse(request_id=str(uuid4()), predictions=ordered, elapsed_ms=elapsed_ms)
