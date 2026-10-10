"""Entrenamiento reproducible, evaluación y registro de artefactos en MLflow."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_platform.data import FEATURES, TARGET, generate_dataset

MODEL_NAME = "synthetic-customer-churn"


def train_model(
    output_path: Path,
    tracking_uri: str,
    n_samples: int = 1200,
    seed: int = 42,
) -> dict[str, float]:
    """Entrena el baseline y devuelve métricas calculadas sobre datos de prueba."""
    data = generate_dataset(n_samples=n_samples, seed=seed)
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.2, stratify=data[TARGET], random_state=seed
    )

    # Pipeline evita fuga de información: el escalador se ajusta solo al train.
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    predicted = (probability >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    metadata_path = output_path.parent / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": MODEL_NAME,
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                "features": FEATURES,
                "samples_total": n_samples,
                "seed": seed,
                "classification_threshold": 0.5,
                "metrics": metrics,
                "dataset": "synthetic (not real-world validated)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("synthetic-customer-churn")
    with mlflow.start_run(run_name="logistic-regression-baseline"):
        mlflow.set_tags({"project": "enterprise-mlops-platform", "dataset": "synthetic"})
        mlflow.log_params({"model": "LogisticRegression", "samples": n_samples, "seed": seed})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(metadata_path), artifact_path="deployment")
        mlflow.sklearn.log_model(model, artifact_path="model")

    return metrics


def main() -> None:
    """Permite ejecutar el entrenamiento desde la terminal."""
    parser = argparse.ArgumentParser(description="Entrenar un modelo baseline de churn sintético")
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tracking-uri", type=str, default=None)
    args = parser.parse_args()

    # URI de archivos portable entre Windows y Linux.
    uri = args.tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or Path("mlruns").resolve().as_uri()
    metrics = train_model(args.output, uri, args.samples, args.seed)
    print("Entrenamiento terminado. Métricas en dataset sintético:")
    print(json.dumps(metrics, indent=2))
    print(f"Modelo guardado en: {args.output.resolve()}")


if __name__ == "__main__":
    main()
