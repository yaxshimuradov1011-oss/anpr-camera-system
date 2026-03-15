from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CandidateRegion:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


class PlateDetector:
    def detect(self, image: np.ndarray) -> list[CandidateRegion]:
        raise NotImplementedError
