from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_detect_rejects_non_image_payload() -> None:
    client = TestClient(app)
    response = client.post(
        "/v1/anpr/detect",
        files={"file": ("bad.txt", b"not-image", "text/plain")},
    )
    assert response.status_code == 400


def test_detect_accepts_image_payload() -> None:
    client = TestClient(app)
    image = np.zeros((80, 160, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    response = client.post(
        "/v1/anpr/detect",
        files={"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "request_id" in payload
    assert "predictions" in payload
