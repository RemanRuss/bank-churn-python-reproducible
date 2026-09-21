"""Stage 2: train and persist the four churn-classification models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from churn.functions import fit_churn_models  # noqa: E402

TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
MODEL_PATH = ROOT / "artifacts" / "models" / "churn_models.joblib"
PARAMS_PATH = ROOT / "artifacts" / "models" / "best_params.json"


def main() -> None:
    """Fit all models using five-fold cross-validation and save artifacts."""
    print("[2/4] Loading processed training data...")
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Processed checkpoint is missing: {TRAIN_PATH}\n"
            "Run src/pipeline/01_prepare_data.py first (or run 'make reproduce')."
        )

    train = pd.read_csv(TRAIN_PATH)
    print("[2/4] Training Logistic Regression, Decision Tree, KNN, and SVM...")
    models = fit_churn_models(train, seed=123, n_jobs=-1)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, MODEL_PATH)
    best_params = {name: model.best_params_ for name, model in models.items()}
    PARAMS_PATH.write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    for name, model in models.items():
        print(f"[2/4] {name}: best CV AUC = {model.best_score_:.4f}; params = {model.best_params_}")
    print(f"[2/4] Done: models saved to {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
