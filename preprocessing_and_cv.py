"""Shared data, preprocessing, model-construction, and evaluation helpers.

Expected input files:
  - headerless CSV files;
  - 18 columns in the order defined by COLUMN_NAMES;
  - the final column is Diagnosis, encoded as {-1, 1} or {0, 1}.

Preprocessing stays inside each fitted sklearn pipeline so data-dependent
transformations are learned only from the rows supplied to ``fit``. Experiment
modules provide their own cross-validation splits and selection protocols.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted


RANDOM_STATE = 42

COLUMN_NAMES = [
    "Age",
    "Gender",
    "Education Level",
    "Marital Status",
    "Occupation",
    "Income Level",
    "Living Area",
    "Hospitalizations",
    "Family History",
    "Substance Use",
    "Suicide Attempt",
    "Positive Symptom Score",
    "Negative Symptom Score",
    "GAF Score",
    "Social Support",
    "Stress Factors",
    "Medication Adherence",
    "Diagnosis",
]

NUMERICAL_COLUMNS = [
    "Age",
    "Hospitalizations",
    "Positive Symptom Score",
    "Negative Symptom Score",
    "GAF Score",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "Education Level",
    "Marital Status",
    "Occupation",
    "Income Level",
    "Living Area",
    "Family History",
    "Substance Use",
    "Suicide Attempt",
    "Social Support",
    "Stress Factors",
    "Medication Adherence",
]


def _read_headerless_csv(path: str | Path) -> pd.DataFrame:
    """Read one CSV and validate its basic structure."""
    path = Path(path)
    data = pd.read_csv(path, header=None)
    data.columns = COLUMN_NAMES
    return data


def _convert_binary_labels(labels: pd.Series) -> pd.Series:
    """Convert {-1, 1} labels to {0, 1}; preserve valid {0, 1} labels."""
    unique_labels = set(labels.unique().tolist())
    if unique_labels == {-1, 1}:
        return (labels == 1).astype(int)
    if unique_labels == {0, 1}:
        return labels.astype(int)
    raise ValueError(
        "Diagnosis must contain exactly {-1, 1} or {0, 1}; "
        f"found {sorted(unique_labels)}."
    )


def load_datasets(
    training_path: str | Path,
    testing_path: str | Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load the pre-split files and return features and binary labels."""
    training_data = _read_headerless_csv(training_path)
    testing_data = _read_headerless_csv(testing_path)

    X_train = training_data.drop(columns="Diagnosis")
    y_train = _convert_binary_labels(training_data["Diagnosis"])
    X_test = testing_data.drop(columns="Diagnosis")
    y_test = _convert_binary_labels(testing_data["Diagnosis"])

    return X_train, y_train, X_test, y_test


def _resolve_feature_columns(
    feature_columns: Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    """Validate a selected predictor list and split it by data type."""
    if feature_columns is None:
        selected = [*NUMERICAL_COLUMNS, *CATEGORICAL_COLUMNS]
    else:
        selected = list(feature_columns)

    allowed = set(NUMERICAL_COLUMNS) | set(CATEGORICAL_COLUMNS)
    unknown = [column for column in selected if column not in allowed]
    if unknown:
        raise ValueError(
            "Unknown predictor columns: "
            f"{sorted(set(unknown))}."
        )
    if len(selected) != len(set(selected)):
        raise ValueError("feature_columns must not contain duplicates.")
    if not selected:
        raise ValueError("feature_columns must contain at least one predictor.")

    selected_set = set(selected)
    numerical = [
        column for column in NUMERICAL_COLUMNS if column in selected_set
    ]
    categorical = [
        column for column in CATEGORICAL_COLUMNS if column in selected_set
    ]
    return numerical, categorical


def build_preprocessor(
    feature_columns: Sequence[str] | None = None,
) -> ColumnTransformer:
    """Create an unfitted preprocessor for the selected predictors."""
    numerical_columns, categorical_columns = _resolve_feature_columns(
        feature_columns
    )
    numerical_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            )
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_transformer, numerical_columns),
            ("categorical", categorical_transformer, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(
    model: object,
    feature_columns: Sequence[str] | None = None,
) -> Pipeline:
    """Connect a fresh preprocessor to one estimator."""
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(feature_columns=feature_columns),
            ),
            ("model", model),
        ]
    )


class MahalanobisKNNClassifier(ClassifierMixin, BaseEstimator):
    """k-NN that estimates its Mahalanobis precision matrix during fit."""

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        n_jobs: int | None = None,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X_checked = check_array(X, dtype=float)
        self.covariance_estimator_ = LedoitWolf(
            store_precision=True
        ).fit(X_checked)
        self.precision_matrix_ = self.covariance_estimator_.get_precision()

        self.knn_ = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric="mahalanobis",
            metric_params={"VI": self.precision_matrix_},
            algorithm="brute",
            n_jobs=self.n_jobs,
        )
        self.knn_.fit(X_checked, y)
        self.classes_ = self.knn_.classes_
        self.n_features_in_ = X_checked.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "knn_")
        return self.knn_.predict(check_array(X, dtype=float))

    def predict_proba(self, X):
        check_is_fitted(self, "knn_")
        return self.knn_.predict_proba(check_array(X, dtype=float))

    def score(self, X, y):
        check_is_fitted(self, "knn_")
        return self.knn_.score(check_array(X, dtype=float), y)


def build_estimator(
    model_key: str,
    model_params: dict[str, Any],
    random_state: int,
) -> object:
    """Create one estimator from a frozen configuration."""
    if model_key == "logistic":
        return LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            **model_params,
        )
    if model_key == "knn":
        params = model_params.copy()
        metric = params.pop("metric")
        if metric == "mahalanobis":
            return MahalanobisKNNClassifier(**params)
        return KNeighborsClassifier(metric=metric, **params)
    if model_key == "perceptron":
        return Perceptron(random_state=random_state, **model_params)
    if model_key == "mlp":
        return MLPClassifier(
            activation="relu",
            solver="adam",
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=random_state,
            **model_params,
        )
    if model_key == "random_forest":
        return RandomForestClassifier(
            random_state=random_state,
            n_jobs=1,
            **model_params,
        )
    raise ValueError(f"Unknown model key: {model_key}")


def build_configured_pipeline(
    model_key: str,
    model_params: dict[str, Any],
    scale_numeric: bool,
    random_state: int,
    feature_columns: Sequence[str] | None = None,
) -> Pipeline:
    """Build a preprocessing/model pipeline for one frozen configuration."""
    estimator = build_estimator(model_key, model_params, random_state)
    pipeline = build_pipeline(estimator, feature_columns=feature_columns)
    if not scale_numeric:
        pipeline.set_params(preprocessor__numerical__scaler="passthrough")
    return pipeline


def _prediction_scores(fitted_pipeline: Pipeline, X: pd.DataFrame) -> Any:
    """Return positive-class scores for ranking metrics."""
    if hasattr(fitted_pipeline, "predict_proba"):
        probabilities = fitted_pipeline.predict_proba(X)
        classes = fitted_pipeline.named_steps["model"].classes_.tolist()
        return probabilities[:, classes.index(1)]
    if hasattr(fitted_pipeline, "decision_function"):
        return fitted_pipeline.decision_function(X)
    raise TypeError("The fitted pipeline cannot produce ranking scores.")


def evaluate_test_set(
    fitted_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float | int]:
    """Compute holdout metrics for one already-fitted pipeline."""
    predictions = fitted_pipeline.predict(X_test)
    scores = _prediction_scores(fitted_pipeline, X_test)
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "Test Accuracy": float(accuracy_score(y_test, predictions)),
        "Test Precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "Test Recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "Test Specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "Test F1": float(f1_score(y_test, predictions, zero_division=0)),
        "Test ROC-AUC": float(roc_auc_score(y_test, scores)),
        "Test PR-AUC": float(average_precision_score(y_test, scores)),
        "Test TN": int(tn),
        "Test FP": int(fp),
        "Test FN": int(fn),
        "Test TP": int(tp),
    }


def inspect_encoded_features(
    X_train: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
) -> list[str]:
    """Fit a separate diagnostic preprocessor and return output names."""
    diagnostic_preprocessor = clone(
        build_preprocessor(feature_columns=feature_columns)
    )
    transformed = diagnostic_preprocessor.fit_transform(X_train)
    feature_names = diagnostic_preprocessor.get_feature_names_out().tolist()

    if transformed.shape[1] != len(feature_names):
        raise RuntimeError("Encoded matrix width does not match feature names.")
    return feature_names
