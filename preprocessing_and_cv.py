"""Build leakage-safe preprocessing pipelines and run preliminary CV.

Expected input files:
  - headerless CSV files;
  - 18 columns in the order defined by COLUMN_NAMES;
  - the final column is Diagnosis, encoded as {-1, 1} or {0, 1}.

This script does not use the testing set for model selection or evaluation.
It only verifies the preprocessing/model pipeline with cross-validation on
the training set. Formal hyperparameter tuning and final test evaluation
belong to later experimental stages.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def build_preprocessor() -> ColumnTransformer:
    """Create the unfitted numerical/categorical preprocessing object."""
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
            ("numerical", numerical_transformer, NUMERICAL_COLUMNS),
            ("categorical", categorical_transformer, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(model: object) -> Pipeline:
    """Connect a fresh preprocessor to one estimator."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )

def inspect_encoded_features(X_train: pd.DataFrame) -> list[str]:
    """Fit a separate diagnostic preprocessor and return output names."""
    diagnostic_preprocessor = clone(build_preprocessor())
    transformed = diagnostic_preprocessor.fit_transform(X_train)
    feature_names = diagnostic_preprocessor.get_feature_names_out().tolist()

    if transformed.shape[1] != len(feature_names):
        raise RuntimeError("Encoded matrix width does not match feature names.")
    return feature_names
