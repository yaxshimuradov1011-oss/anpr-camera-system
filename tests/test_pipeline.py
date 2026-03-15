from __future__ import annotations

import numpy as np

from app.detectors.base import CandidateRegion, PlateDetector
from app.ocr.base import OCREngine
from app.pipeline import ANPRPipeline


class StubDetector(PlateDetector):
    def detect(self, image: np.ndarray) -> list[CandidateRegion]:
        return [CandidateRegion(0, 0, image.shape[1], image.shape[0], 0.9)]


class StubOCR(OCREngine):
    def __init__(self, text: str, conf: float) -> None:
        self.text = text
        self.conf = conf

    def read_text(self, roi: np.ndarray) -> tuple[str, float]:
        return self.text, self.conf


def test_pipeline_returns_prediction_for_valid_plate() -> None:
    pipeline = ANPRPipeline(
        detector=StubDetector(),
        ocr_engine=StubOCR("ABC1234", 0.95),
        plate_pattern=r"[A-Z0-9]{5,10}",
        min_confidence=0.5,
    )

    image = np.zeros((40, 120, 3), dtype=np.uint8)
    result = pipeline.run(image)

    assert len(result.predictions) == 1
    assert result.predictions[0].plate == "ABC1234"


def test_pipeline_rejects_text_not_matching_pattern() -> None:
    pipeline = ANPRPipeline(
        detector=StubDetector(),
        ocr_engine=StubOCR("***", 0.95),
        plate_pattern=r"[A-Z0-9]{5,10}",
        min_confidence=0.1,
    )

    image = np.zeros((40, 120, 3), dtype=np.uint8)
    result = pipeline.run(image)

    assert result.predictions == []
