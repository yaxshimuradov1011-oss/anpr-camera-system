# ANPR Camera System (Python)

A production-oriented **Automatic Number Plate Recognition (ANPR)** backend built with FastAPI.

## Features
- HTTP API for health checks and ANPR inference.
- Image validation and size limits.
- Pluggable detection/OCR layers with concrete OpenCV + Tesseract implementation.
- Confidence fusion and regex-based plate validation.
- API key support for protected deployments.
- Unit tests for pipeline behavior and API contract.

## Architecture
1. `OpenCVPlateDetector` finds candidate regions from contours.
2. `TesseractOCREngine` extracts plate text from each candidate region.
3. `ANPRPipeline` merges detector + OCR confidence, filters invalid plates, and returns ranked predictions.

## API
### `GET /health`
Returns service metadata and UTC timestamp.

### `POST /v1/anpr/detect`
Multipart upload:
- `file`: image file (`jpg`, `png`, etc.)
- Optional header: `x-api-key` when `ANPR_API_KEY` is configured.

Response:
```json
{
  "request_id": "uuid",
  "predictions": [
    {
      "plate": "ABC1234",
      "confidence": 0.91,
      "box": {"x1": 11, "y1": 28, "x2": 220, "y2": 86, "score": 0.85}
    }
  ],
  "elapsed_ms": 17
}
```

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Configuration
Use environment variables:
- `ANPR_API_KEY`: enable API key auth if set.
- `MAX_UPLOAD_MB` (default `8`)
- `PLATE_PATTERN` (default `[A-Z0-9]{5,10}`)
- `MIN_CONFIDENCE` (default `0.45`)
- `LOG_LEVEL` (default `INFO`)

## Docker
```bash
docker build -t anpr-camera-system .
docker run --rm -p 8000:8000 -e ANPR_API_KEY=secret anpr-camera-system
```

## Test
```bash
pytest
```
