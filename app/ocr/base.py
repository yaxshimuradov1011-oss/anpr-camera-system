from __future__ import annotations

import numpy as np


class OCREngine:
    def read_text(self, roi: np.ndarray) -> tuple[str, float]:
        raise NotImplementedError
