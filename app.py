"""
Minimal FastAPI app for the W30D4 demo.

Exposes:
- GET  /health      → returns {"status": "ok"}
- POST /predict     → returns a stubbed prediction
- GET  /metadata    → returns model version info

This is intentionally minimal — enough to make the CI pipeline pass
and demonstrate a working API without needing a real trained model.
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="W30D4 Demo API", version="0.1.0")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    prediction: str
    confidence: float


@app.get("/health")
def health():
    """Health endpoint used by CI to verify the app starts correctly."""
    return {"status": "ok", "version": "0.1.0"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Stub prediction. Replace with real model inference in production."""
    # Simple stub: label anything with "error" or "fail" as negative, else positive.
    text_lower = request.text.lower()
    if "error" in text_lower or "fail" in text_lower or "bad" in text_lower:
        return PredictResponse(prediction="negative", confidence=0.87)
    return PredictResponse(prediction="positive", confidence=0.92)


@app.get("/metadata")
def metadata():
    """Metadata endpoint for model version tracking."""
    return {
        "model_name": "demo-classifier",
        "model_version": "0.1.0",
        "trained_on": "2026-04-01",
    }
