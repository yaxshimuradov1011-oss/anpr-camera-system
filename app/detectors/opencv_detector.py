from __future__ import annotations

import logging
import cv2
import numpy as np

from app.detectors.base import CandidateRegion, PlateDetector

logger = logging.getLogger(__name__)


class OpenCVPlateDetector(PlateDetector):
    """Heuristic plate detector based on contour geometry.

    It is not model-based but provides a deterministic fallback detector that works
    reasonably for high-contrast plate regions.
    """

    def detect(self, image: np.ndarray) -> list[CandidateRegion]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(blur, 30, 200)

        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        results: list[CandidateRegion] = []
        h, w = gray.shape

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
            if len(approx) not in (4, 5):
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            aspect = bw / max(bh, 1)
            area = bw * bh
            area_ratio = area / float(w * h)

            if 2.0 <= aspect <= 6.5 and 0.01 <= area_ratio <= 0.30:
                score = min(1.0, 0.4 + area_ratio + (0.2 if len(approx) == 4 else 0.1))
                results.append(CandidateRegion(x, y, x + bw, y + bh, score))

        if not results:
            logger.debug("No contours matched plate heuristics; returning whole frame fallback")
            results.append(CandidateRegion(0, 0, w, h, 0.2))

        return sorted(results, key=lambda item: item.score, reverse=True)[:3]
