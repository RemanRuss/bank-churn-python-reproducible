"""Tests for validation and deterministic preprocessing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn.functions import (  # noqa: E402
    clean_churn_data,
    clean_column_names,
    split_and_impute,
    validate_raw_data,
)


def make_example_data(n: int = 60) -> pd.DataFrame:
    """Construct a small balanced data frame with the expected schema."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "credit_score": rng.integers(500, 801, size=n),
            "country": rng.choice(["France", "Germany", "Spain"], size=n),
            "gender": rng.choice(["Female", "Male"], size=n),
            "age": rng.integers(20, 71, size=n),
            "tenure": rng.integers(0, 11, size=n),
            "balance": rng.uniform(0, 200_000, size=n),
            "products_number": rng.integers(1, 5, size=n),
            "credit_card": rng.integers(0, 2, size=n),
            "active_member": rng.integers(0, 2, size=n),
            "estimated_salary": rng.uniform(10_000, 200_000, size=n),
            "churn": np.tile([0, 1], n // 2),
        }
    )


def test_raw_data_validation_catches_structural_problems() -> None:
    """Validation should reject absent columns and non-binary churn labels."""
    valid = clean_column_names(make_example_data())
    validate_raw_data(valid)

    missing_balance = valid.drop(columns=["balance"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_raw_data(missing_balance)

    invalid_churn = valid.copy()
    invalid_churn.loc[0, "churn"] = 2
    with pytest.raises(ValueError, match="only 0/1"):
        validate_raw_data(invalid_churn)


def test_cleaning_and_split_are_deterministic_and_impute_from_training() -> None:
    """Repeated seeded splits should match and leave no predictor missingness."""
    example = make_example_data()
    example.loc[[1, 6], "balance"] = np.nan

    cleaned = clean_churn_data(example)
    assert not {"customer_id", "gender", "country"}.intersection(cleaned.columns)
    assert set(cleaned["churn"].unique()) == {0, 1}

    train_a, test_a, medians_a = split_and_impute(cleaned, seed=123, train_fraction=0.8)
    train_b, test_b, medians_b = split_and_impute(cleaned, seed=123, train_fraction=0.8)

    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)
    assert medians_a == medians_b
    assert not train_a.isna().any().any()
    assert not test_a.isna().any().any()
    assert len(train_a) + len(test_a) == len(cleaned)
