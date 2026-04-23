"""
Tests for the W30D4 demo API.

Run locally: pytest tests/ -v
These tests are executed automatically by .github/workflows/ml-prod.yaml on every push.
"""

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_returns_200():
    """Blocker: /health must return HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_status_ok():
    """Blocker: /health response must include status=ok."""
    response = client.get("/health")
    assert response.json()["status"] == "ok"


def test_predict_accepts_valid_input():
    """Blocker: /predict must accept valid input and return a prediction."""
    response = client.post("/predict", json={"text": "this is a great product"})
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confidence" in response.json()


def test_predict_returns_positive_for_good_text():
    """Warning: /predict should classify positive text as positive."""
    response = client.post("/predict", json={"text": "excellent service"})
    assert response.json()["prediction"] == "positive"


def test_predict_returns_negative_for_bad_text():
    """Warning: /predict should classify negative text as negative."""
    response = client.post("/predict", json={"text": "this is a bad error"})
    assert response.json()["prediction"] == "negative"


def test_predict_rejects_invalid_input():
    """Warning: /predict should return 422 for malformed requests, not crash."""
    response = client.post("/predict", json={"wrong_field": "oops"})
    assert response.status_code == 422


def test_metadata_returns_model_version():
    """Monitoring: /metadata should return model version info."""
    response = client.get("/metadata")
    assert response.status_code == 200
    assert "model_version" in response.json()
