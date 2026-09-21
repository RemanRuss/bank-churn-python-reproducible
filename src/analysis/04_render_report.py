"""Stage 4: render a self-contained project summary as HTML."""

from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AUC_PATH = ROOT / "results" / "tables" / "model_auc.csv"
CHURN_PATH = ROOT / "results" / "tables" / "churn_distribution.csv"
OUTPUT_PATH = ROOT / "results" / "report" / "churn_report.html"


def dataframe_to_html(data: pd.DataFrame, digits: int = 4) -> str:
    """Format a data frame as a compact HTML table."""
    display = data.copy()
    for column in display.select_dtypes(include="number").columns:
        display[column] = display[column].round(digits)
    return display.to_html(index=False, border=0, classes="result-table")


def main() -> None:
    """Build the final HTML report from generated tables and figures."""
    if not AUC_PATH.exists() or not CHURN_PATH.exists():
        raise FileNotFoundError(
            "Result tables are missing. Run src/analysis/03_make_results.py "
            "first (or run 'make reproduce')."
        )

    print("[4/4] Rendering HTML report...")
    auc_table = pd.read_csv(AUC_PATH)
    churn_table = pd.read_csv(CHURN_PATH)

    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bank Customer Churn Prediction</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 920px; margin: 40px auto; padding: 0 22px; line-height: 1.55; }}
h1, h2 {{ margin-top: 1.4em; }}
img {{ display: block; max-width: 100%; margin: 18px auto 30px; }}
.result-table {{ border-collapse: collapse; margin: 16px 0 28px; width: 100%; }}
.result-table th, .result-table td {{ border-bottom: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
.meta {{ color: #555; }}
code {{ background: #f3f3f3; padding: 2px 4px; }}
</style>
</head>
<body>
<h1>Bank Customer Churn Prediction</h1>
<p class="meta">Yanze Jiang · generated {date.today().isoformat()}</p>

<h2>Introduction</h2>
<p>Customer churn occurs when a customer stops using a company's services. This project uses a bank-customer dataset and predicts the binary <code>churn</code> outcome from numeric account and financial characteristics. The reproducible Python pipeline compares Logistic Regression, Decision Tree, K-nearest neighbors (KNN), and an RBF support vector machine (SVM).</p>

<h2>Reproducible workflow</h2>
<p>The raw data are validated, the original exclusions (<code>customer_id</code>, <code>gender</code>, and <code>country</code>) are applied, and the data are split 80/20 with stratification and seed 123. Missing numeric values are filled using medians learned from the training set. Model selection uses five-fold stratified cross-validation scored by ROC AUC. KNN and SVM are standardized inside their model pipelines.</p>

<h2>Exploratory data analysis</h2>
<h3>Churn distribution</h3>
{dataframe_to_html(churn_table)}
<img src="../figures/churn_distribution.png" alt="Churn distribution">

<h3>Numeric features</h3>
<img src="../figures/numeric_histograms.png" alt="Numeric predictor histograms">

<h2>Model comparison</h2>
{dataframe_to_html(auc_table)}
<img src="../figures/roc_comparison.png" alt="ROC curve comparison">

<h2>Conclusion</h2>
<p>The model comparison above is generated directly from the current pipeline. No AUC values are hard-coded into the report, so the written output stays consistent with the latest reproduced results.</p>
</body>
</html>
"""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"[4/4] Done: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
