"""Stage 1: validate, clean, split, and impute the churn data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from churn.functions import clean_churn_data, split_and_impute  # noqa: E402

RAW_PATH = ROOT / "data" / "raw" / "Bank Customer Churn Prediction.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "churn_clean.csv"
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
MEDIANS_PATH = ROOT / "data" / "processed" / "training_medians.json"


def main() -> None:
    """Run the data-preparation stage."""
    print("[1/4] Reading and validating raw data...")
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Required raw data file was not found: {RAW_PATH}\n"
            "Place 'Bank Customer Churn Prediction.csv' in data/raw/. "
            "See data/raw/README.md."
        )

    raw_data = pd.read_csv(RAW_PATH)
    clean_data = clean_churn_data(raw_data)
    train, test, medians = split_and_impute(clean_data, seed=123, train_fraction=0.8)

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_data.to_csv(CLEAN_PATH, index=False)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    MEDIANS_PATH.write_text(json.dumps(medians, indent=2), encoding="utf-8")

    print(f"[1/4] Done: {len(train)} training rows and {len(test)} test rows.")


if __name__ == "__main__":
    main()
