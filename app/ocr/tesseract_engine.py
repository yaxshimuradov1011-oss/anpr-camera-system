from __future__ import annotations

import re
import cv2
import numpy as np
import pytesseract

from app.ocr.base import OCREngine


class TesseractOCREngine(OCREngine):
    def __init__(self, plate_pattern: str) -> None:
        self._pattern = re.compile(plate_pattern)

    def read_text(self, roi: np.ndarray) -> tuple[str, float]:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        proc = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        raw = pytesseract.image_to_string(proc, config=config)

        cleaned = re.sub(r"[^A-Z0-9]", "", raw.upper())
        if not cleaned:
            return "", 0.0

        match = self._pattern.search(cleaned)
        if not match:
            return cleaned, 0.3

        text = match.group(0)
        length_boost = min(0.3, len(text) * 0.03)
        confidence = 0.6 + length_boost
        return text, min(confidence, 0.98)
