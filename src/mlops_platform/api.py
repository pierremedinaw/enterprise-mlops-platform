"""API de inferencia para el modelo sintético de abandono de clientes."""

import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlops_platform.data import FEATURES

app = FastAPI(
    title="Enterprise MLOps Platform",
    description="API demostrativa: predicción sobre datos sintéticos, no lista para producción.",
    version="0.1.0",
)


class CustomerFeatures(BaseModel):
    """Contrato de entrada documentado y validado por Pydantic."""

    age: int = Field(ge=18, le=100)
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=500)
    support_tickets: int = Field(ge=0, le=30)
    month_to_month: bool
    autopay_enabled: bool


class PredictionResponse(BaseModel):
    """Salida de inferencia con umbral explícito."""

    churn_probability: float = Field(ge=0, le=1)
    predicted_churn: bool
    threshold: float


def model_path() -> Path:
    """Permite sobrescribir la ruta del artefacto con MODEL_PATH."""
    return Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))


@lru_cache(maxsize=8)
def load_model(path: str):
    """Carga solo modelos generados por nosotros desde un directorio confiable."""
    return joblib.load(path)


@app.get("/health")
def health() -> dict[str, str]:
    """Indica si el artefacto requerido existe en disco."""
    return {"status": "ready" if model_path().is_file() else "not_ready"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    """Calcula la probabilidad de abandono de un registro sintético."""
    path = model_path()
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Model unavailable. Run training first.")

    try:
        model = load_model(str(path.resolve()))
        record = customer.model_dump()
        record["month_to_month"] = int(record["month_to_month"])
        record["autopay_enabled"] = int(record["autopay_enabled"])
        frame = pd.DataFrame([record], columns=FEATURES)
        probability = float(model.predict_proba(frame)[0, 1])
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(status_code=503, detail="Model could not perform inference.") from exc

    threshold = 0.5
    return PredictionResponse(
        churn_probability=probability,
        predicted_churn=probability >= threshold,
        threshold=threshold,
    )
