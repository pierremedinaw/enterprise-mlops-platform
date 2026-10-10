"""Pruebas de calidad del generador sintético."""

import pytest

from mlops_platform.data import FEATURES, TARGET, generate_dataset


def test_dataset_schema_and_label() -> None:
    data = generate_dataset(n_samples=250, seed=42)
    assert len(data) == 250
    assert list(data.columns) == [*FEATURES, TARGET]
    assert data.isna().sum().sum() == 0
    assert set(data[TARGET].unique()) == {0, 1}


def test_dataset_is_reproducible() -> None:
    assert generate_dataset(200, 7).equals(generate_dataset(200, 7))


def test_rejects_small_sample() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        generate_dataset(99)
