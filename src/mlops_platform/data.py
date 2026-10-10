"""Generación determinista de datos sintéticos sin información personal real."""

import numpy as np
import pandas as pd

FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "month_to_month",
    "autopay_enabled",
]
TARGET = "churn"


def generate_dataset(n_samples: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Construye un ejemplo pedagógico de abandono de clientes.

    Las etiquetas se simulan con una distribución probabilística; no se usan
    datos empresariales reales ni se deben interpretar como un modelo validado.
    """
    if n_samples < 100:
        raise ValueError("n_samples debe ser >= 100")

    rng = np.random.default_rng(seed)
    age = rng.integers(18, 76, n_samples)
    tenure = rng.integers(0, 73, n_samples)
    charges = rng.uniform(20, 180, n_samples).round(2)
    tickets = rng.poisson(1.5, n_samples).clip(0, 15)
    monthly_contract = rng.binomial(1, 0.62, n_samples)
    autopay = rng.binomial(1, 0.55, n_samples)

    # Relación causal simulada: sirve para probar el pipeline, no para negocio.
    logit = (
        -0.45
        + 0.80 * monthly_contract
        - 0.035 * tenure
        + 0.28 * tickets
        + 0.011 * (charges - 70)
        - 0.65 * autopay
        + 0.008 * (age - 40)
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "age": age,
            "tenure_months": tenure,
            "monthly_charges": charges,
            "support_tickets": tickets,
            "month_to_month": monthly_contract,
            "autopay_enabled": autopay,
            TARGET: churn,
        }
    )
