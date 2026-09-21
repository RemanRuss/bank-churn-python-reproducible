"""Core data validation, preprocessing, modeling, and evaluation utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

REQUIRED_COLUMNS = {
    "customer_id",
    "country",
    "gender",
    "credit_score",
    "age",
    "tenure",
    "balance",
    "products_number",
    "credit_card",
    "active_member",
    "estimated_salary",
    "churn",
}

DROP_COLUMNS = ["customer_id", "gender", "country"]
TARGET = "churn"


def clean_column_name(name: str) -> str:
    """Convert a raw column name to a compact snake_case representation."""
    text = re.sub(r"[^0-9A-Za-z]+", "_", str(name).strip()).strip("_")
    text = re.sub(r"_+", "_", text)
    return text.lower()


def clean_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *data* with normalized snake_case column names."""
    out = data.copy()
    out.columns = [clean_column_name(column) for column in out.columns]
    return out


def validate_raw_data(data: pd.DataFrame) -> None:
    """Validate required columns and the binary churn target.

    Parameters
    ----------
    data:
        Raw data after column-name normalization.

    Raises
    ------
    ValueError
        If required columns are absent, churn is not binary, or predictors
        cannot be interpreted as numeric after the original exclusions.
    """
    missing = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if missing:
        raise ValueError(f"Raw data are missing required columns: {missing}")

    churn_nonmissing = pd.to_numeric(data[TARGET], errors="coerce").dropna()
    observed = set(churn_nonmissing.unique().tolist())
    if not observed.issubset({0, 1}) or churn_nonmissing.empty:
        raise ValueError("The churn column must contain only 0/1 values.")

    if data[TARGET].isna().any():
        raise ValueError("The churn column contains missing values.")


def clean_churn_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataset while preserving the original modeling choices.

    The original analysis excludes customer_id, gender, and country, keeps
    the remaining predictors numeric, and uses churn as a binary outcome.
    """
    data = clean_column_names(raw_data)
    validate_raw_data(data)

    data = data.drop(columns=DROP_COLUMNS).copy()
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data[TARGET].isna().any():
        raise ValueError("The churn target became missing during numeric conversion.")

    data[TARGET] = data[TARGET].astype(int)
    return data


def split_and_impute(
    data: pd.DataFrame,
    seed: int = 123,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Create a deterministic stratified split and impute numeric missing values.

    Medians are learned from the training partition only and then applied to
    both training and test data, preventing leakage from the held-out set.
    """
    if TARGET not in data.columns:
        raise ValueError(f"Expected target column '{TARGET}'.")

    train, test = train_test_split(
        data,
        train_size=train_fraction,
        stratify=data[TARGET],
        random_state=seed,
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    predictor_columns = [column for column in train.columns if column != TARGET]
    medians = train[predictor_columns].median(numeric_only=True)
    all_missing = medians[medians.isna()].index.tolist()
    if all_missing:
        raise ValueError(
            "Cannot median-impute predictors that are entirely missing in the "
            f"training data: {all_missing}"
        )

    train[predictor_columns] = train[predictor_columns].fillna(medians)
    test[predictor_columns] = test[predictor_columns].fillna(medians)

    if train[predictor_columns].isna().any().any() or test[predictor_columns].isna().any().any():
        raise ValueError("Numeric missing values remain after training-median imputation.")

    return train, test, {key: float(value) for key, value in medians.items()}


def _cv(seed: int) -> StratifiedKFold:
    """Return the common five-fold stratified cross-validation splitter."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


def fit_churn_models(
    train_data: pd.DataFrame,
    seed: int = 123,
    n_jobs: int = -1,
) -> dict[str, GridSearchCV]:
    """Fit the four model families used in the original analysis.

    Model selection is based on mean cross-validated ROC AUC. KNN and SVM are
    standardized inside a scikit-learn Pipeline so scaling is learned only
    from each training fold.
    """
    x_train = train_data.drop(columns=[TARGET])
    y_train = train_data[TARGET]
    cv = _cv(seed)

    logistic_pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=np.inf, max_iter=2000, random_state=seed)),
        ]
    )
    logistic = GridSearchCV(
        logistic_pipeline,
        param_grid={},
        scoring="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )

    tree = GridSearchCV(
        DecisionTreeClassifier(random_state=seed),
        param_grid={
            "ccp_alpha": [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )

    knn_pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", KNeighborsClassifier()),
        ]
    )
    knn = GridSearchCV(
        knn_pipeline,
        param_grid={"model__n_neighbors": [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]},
        scoring="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )

    svm_pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=False, random_state=seed)),
        ]
    )
    svm = GridSearchCV(
        svm_pipeline,
        param_grid=[
            {"model__C": [0.25], "model__gamma": ["scale"]},
            {"model__C": [0.5], "model__gamma": ["scale"]},
            {"model__C": [1.0], "model__gamma": ["scale"]},
            {"model__C": [2.0], "model__gamma": ["scale"]},
            {"model__C": [4.0], "model__gamma": ["scale"]},
        ],
        scoring="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )

    models: dict[str, GridSearchCV] = {
        "Logistic Regression": logistic,
        "Decision Tree": tree,
        "KNN": knn,
        "SVM": svm,
    }

    for model in models.values():
        model.fit(x_train, y_train)

    return models


def evaluate_roc(model: BaseEstimator, test_data: pd.DataFrame) -> dict[str, Any]:
    """Calculate held-out ROC AUC and ROC coordinates for one fitted model."""
    x_test = test_data.drop(columns=[TARGET])
    y_test = test_data[TARGET]
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(x_test)
    else:
        raise TypeError("Model must provide predict_proba or decision_function for ROC evaluation.")

    fpr, tpr, thresholds = roc_curve(y_test, scores)
    return {
        "auc": float(roc_auc_score(y_test, scores)),
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }
