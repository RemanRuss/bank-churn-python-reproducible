# Reproducible Bank Customer Churn Project — Python Version

## Purpose

This repository converts the original single-file R Markdown bank-customer-churn analysis into a
structured, documented, testable, and reproducible **Python** workflow. The statistical scope is
kept the same: clean the customer data, create an 80/20 train/test split, fit Logistic Regression,
Decision Tree, K-nearest neighbors (KNN), and RBF-SVM models, compare held-out ROC AUC, and
produce figures, tables, and an HTML report.

The original analysis is preserved unchanged in `original/original_analysis.Rmd` for provenance.
It is **not required to run the Python pipeline**.

## Principal outputs

Running the project produces:

- `results/tables/model_auc.csv` — held-out AUC for each model
- `results/tables/churn_distribution.csv` — test-set churn counts and percentages
- `results/figures/roc_comparison.png` — ROC comparison
- `results/figures/churn_distribution.png` — churn class distribution
- `results/figures/numeric_histograms.png` — numeric predictor distributions
- `results/report/churn_report.html` — final reproducible report

## Data source

The dataset used in this project is the Bank Customer Churn Dataset by
Gaurav Topre on Kaggle.

A copy of `Bank Customer Churn Prediction.csv` is included in
`data/raw/`, so the full analysis can be reproduced directly after
cloning the repository.

Original source:
https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset
reproduction workflow.

## Computational environment

### System requirements

- Python 3.13 recommended
- GNU Make
- Internet access for the first package installation

### Windows note

On Windows, run the `make` commands from **Git Bash**. The Makefile uses Unix-style shell
commands and is not intended to be run directly from PowerShell or Command Prompt.

The active analysis uses only Python. A project-local virtual environment is stored in `.venv/`
and is excluded from Git. Exact direct package versions are recorded in `requirements.txt`.

### Create the environment on a fresh system

From the repository root:

```bash
make setup
```

This creates `.venv/` and installs the pinned packages in `requirements.txt`.

If GNU Make is unavailable, the Windows/Git-Bash equivalent is:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

On macOS/Linux, replace `.venv/Scripts/python.exe` with `.venv/bin/python`.

## One-command reproduction

After the CSV is present in `data/raw/` and the environment has been created, run:

```bash
make reproduce
```

The command is non-interactive and executes four stages:

1. validate, clean, split, and impute the data;
2. train the four model families using five-fold stratified cross-validation;
3. evaluate held-out AUC and generate figures/tables;
4. render the final HTML report.

Randomness is fixed with seed `123`. Missing numeric predictor values are imputed using medians
learned from the **training set only**. KNN and SVM use `StandardScaler` inside their scikit-learn
pipelines so scaling is learned within each cross-validation fold.

## Run the tests

```bash
make test
```

The tests cover two validation categories:

1. **Data validation** — required columns and valid binary churn values.
2. **Function correctness** — deterministic seeded splitting, expected column exclusions, and
   training-median imputation with no remaining predictor missingness.

## Repository structure

```text
bank-churn-python-reproducible/
├── original/
│   └── original_analysis.Rmd       # archived source analysis only
├── data/
│   ├── raw/                        # source CSV (not committed)
│   └── processed/                  # generated clean/train/test checkpoints
├── src/
│   ├── churn/                      # reusable Python functions
│   ├── pipeline/                   # data preparation + model training
│   └── analysis/                   # evaluation + report generation
├── artifacts/
│   ├── models/                     # fitted joblib models + tuning metadata
│   └── sufficient-stats/           # reserved shareable checkpoint location
├── results/
│   ├── tables/
│   ├── figures/
│   └── report/
├── tests/                          # pytest tests
├── docs/                           # reflection and requirement mapping
├── requirements.txt                # pinned direct Python dependencies
├── pytest.ini
├── Makefile                        # setup, test, reproduce, clean
├── README.md
└── .gitignore
```

## Modeling correspondence to the original analysis

The Python refactor retains the original predictor exclusions and model families. It uses
scikit-learn equivalents rather than `caret`:

| Original model | Python implementation |
|---|---|
| Logistic regression (`glm`) | `LogisticRegression` |
| Decision tree (`rpart`) | `DecisionTreeClassifier` with pruning-parameter CV |
| KNN (`knn`) | `StandardScaler` + `KNeighborsClassifier` |
| Radial SVM (`svmRadial`) | `StandardScaler` + `SVC(kernel="rbf")` |

Because R/caret and scikit-learn use different implementations and tuning grids, reproduced AUC
values may differ slightly from the old R report. The project therefore computes and displays the
current Python results rather than copying old numerical AUC values into the narrative.

## Expected grader workflow

```bash
make setup
make test
make reproduce
```

Then open:

```text
results/report/churn_report.html
```
