"""Pruebas de contrato y flujo integrado entrenamiento-inferencia."""

from pathlib import Path

from fastapi.testclient import TestClient

from mlops_platform.api import app, load_model
from mlops_platform.train import train_model

SAMPLE = {
    "age": 38,
    "tenure_months": 12,
    "monthly_charges": 89.9,
    "support_tickets": 3,
    "month_to_month": True,
    "autopay_enabled": False,
}


def test_health_without_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "not_ready"}
        response = client.post("/predict", json=SAMPLE)
        assert response.status_code == 503


def test_input_validation_without_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(app) as client:
        response = client.post("/predict", json={**SAMPLE, "age": -2})
        assert response.status_code == 422


def test_train_and_predict(tmp_path: Path, monkeypatch) -> None:
    model_file = tmp_path / "model.joblib"
    tracking_uri = (tmp_path / "mlruns").resolve().as_uri()
    metrics = train_model(output_path=model_file, tracking_uri=tracking_uri, n_samples=250)
    assert model_file.exists()
    assert (tmp_path / "model_metadata.json").exists()
    assert 0 <= metrics["roc_auc"] <= 1

    monkeypatch.setenv("MODEL_PATH", str(model_file))
    load_model.cache_clear()
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ready"}
        response = client.post("/predict", json=SAMPLE)
        assert response.status_code == 200
        payload = response.json()
        assert 0 <= payload["churn_probability"] <= 1
        assert payload["threshold"] == 0.5
        assert isinstance(payload["predicted_churn"], bool)
