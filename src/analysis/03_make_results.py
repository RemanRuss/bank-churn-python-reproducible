"""Stage 3: evaluate fitted models and generate final tables and figures."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from churn.functions import TARGET, evaluate_roc  # noqa: E402

TEST_PATH = ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = ROOT / "artifacts" / "models" / "churn_models.joblib"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"


def main() -> None:
    """Create model-comparison tables and exploratory/evaluation figures."""
    if not TEST_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Required processed data or model artifacts are missing. "
            "Run 'make reproduce' from the repository root."
        )

    print("[3/4] Loading fitted models and held-out test data...")
    test = pd.read_csv(TEST_PATH)
    models = joblib.load(MODEL_PATH)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("[3/4] Calculating held-out ROC/AUC results...")
    evaluations = {name: evaluate_roc(model, test) for name, model in models.items()}
    auc_table = pd.DataFrame(
        [{"model": name, "auc": result["auc"]} for name, result in evaluations.items()]
    ).sort_values("auc", ascending=False)
    auc_table.to_csv(TABLE_DIR / "model_auc.csv", index=False)

    churn_counts = (
        test[TARGET]
        .value_counts()
        .sort_index()
        .rename_axis("churn")
        .reset_index(name="count")
    )
    churn_counts["label"] = churn_counts["churn"].map({0: "no", 1: "yes"})
    churn_counts["percent"] = churn_counts["count"] / churn_counts["count"].sum()
    churn_counts[["label", "count", "percent"]].to_csv(
        TABLE_DIR / "churn_distribution.csv", index=False
    )

    plt.figure(figsize=(7.5, 6))
    for name, result in evaluations.items():
        plt.plot(result["fpr"], result["tpr"], label=f"{name} (AUC={result['auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves on the Held-Out Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "roc_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6.5, 4.5))
    plt.bar(churn_counts["label"], churn_counts["count"])
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.title("Distribution of Customer Churn")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "churn_distribution.png", dpi=150)
    plt.close()

    predictors = test.drop(columns=[TARGET])
    axes = predictors.hist(bins=30, figsize=(10, 8))
    fig = axes[0, 0].get_figure() if getattr(axes, "ndim", 1) > 1 else axes[0].get_figure()
    fig.suptitle("Histograms of Numeric Features")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURE_DIR / "numeric_histograms.png", dpi=150)
    plt.close(fig)

    print("[3/4] Done: tables and figures written under results/.")


if __name__ == "__main__":
    main()
