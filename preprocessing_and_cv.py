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

import argparse
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
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


def build_pipelines(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """Create preliminary, untuned pipelines for the four study models."""
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=random_state,
        ),
        "k-NN": KNeighborsClassifier(
            n_neighbors=21,
            metric="manhattan",
            weights="uniform",
        ),
        "Perceptron": Perceptron(
            penalty=None,
            max_iter=1000,
            tol=1e-3,
            random_state=random_state,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=random_state,
        ),
    }

    return {name: build_pipeline(model) for name, model in models.items()}


def build_scoring() -> dict[str, object]:
    """Define metrics, including specificity as recall of class 0."""
    return {
        "Accuracy": "accuracy",
        "Precision": make_scorer(precision_score, zero_division=0),
        "Recall": make_scorer(recall_score, zero_division=0),
        "Specificity": make_scorer(
            recall_score,
            pos_label=0,
            zero_division=0,
        ),
        "F1": make_scorer(f1_score, zero_division=0),
        "ROC-AUC": "roc_auc",
    }


def evaluate_pipelines(
    pipelines: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Evaluate complete pipelines using stratified cross-validation."""
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2.")

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )
    scoring = build_scoring()
    rows: list[dict[str, float | str]] = []

    for model_name, pipeline in pipelines.items():
        scores = cross_validate(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=n_jobs,
            error_score="raise",
        )

        row: dict[str, float | str] = {"Model": model_name}
        for metric_name in scoring:
            values = scores[f"test_{metric_name}"]
            row[f"{metric_name} Mean"] = float(values.mean())
            row[f"{metric_name} Std"] = float(values.std(ddof=1))
        row["Fit Time Mean (s)"] = float(scores["fit_time"].mean())
        rows.append(row)

    return pd.DataFrame(rows)


def inspect_encoded_features(X_train: pd.DataFrame) -> list[str]:
    """Fit a separate diagnostic preprocessor and return output names."""
    diagnostic_preprocessor = clone(build_preprocessor())
    transformed = diagnostic_preprocessor.fit_transform(X_train)
    feature_names = diagnostic_preprocessor.get_feature_names_out().tolist()

    if transformed.shape[1] != len(feature_names):
        raise RuntimeError("Encoded matrix width does not match feature names.")
    return feature_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one-hot/scaling pipelines and run preliminary stratified "
            "cross-validation on the training set."
        )
    )
    parser.add_argument(
        "--training",
        type=Path,
        default=Path("training_data.csv"),
        help="Headerless training CSV (default: training_data.csv).",
    )
    parser.add_argument(
        "--testing",
        type=Path,
        default=Path("testing_data.csv"),
        help="Headerless testing CSV (default: testing_data.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("preliminary_cv_results.csv"),
        help="Cross-validation summary CSV output path.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel CV jobs; -1 uses all available cores (default: -1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X_train, y_train, X_test, y_test = load_datasets(
        args.training,
        args.testing,
    )

    encoded_feature_names = inspect_encoded_features(X_train)
    pipelines = build_pipelines(random_state=args.random_state)

    print(f"Training feature shape: {X_train.shape}")
    print(f"Testing feature shape: {X_test.shape}")
    print(f"Training class counts: {y_train.value_counts().sort_index().to_dict()}")
    print(f"Testing class counts: {y_test.value_counts().sort_index().to_dict()}")
    print(f"Encoded feature count: {len(encoded_feature_names)}")
    print("Testing data are loaded for structural checks only; no test metrics are computed.")

    results = evaluate_pipelines(
        pipelines=pipelines,
        X_train=X_train,
        y_train=y_train,
        cv_splits=args.folds,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    with pd.option_context("display.max_columns", None, "display.width", 240):
        print("\nPreliminary cross-validation summary:")
        print(results.round(4).to_string(index=False))
    print(f"\nSaved results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
