"""Phase 2 feature-importance analysis.

This script uses only ``training_data.csv``. It performs five-fold stratified
cross-validation and produces two complementary original-feature rankings:

1. standardized Perceptron coefficient magnitude; and
2. Random Forest Tree SHAP importance using the Phase 1 configuration
   ``random_forest_021``.

The preprocessing object is fitted independently inside every training fold.
One-hot encoded columns are grouped back to their original variables before
the fold-level results are summarized.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_PATH = PROJECT_ROOT / "training_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "importance"

RANDOM_STATE = 42
CV_SPLITS = 5
TOP_K = 3

PERCEPTRON_PARAMS: dict[str, Any] = {
    "penalty": None,
    "max_iter": 200,
    "tol": 1e-3,
}

# Phase 1 configuration random_forest_021.
RANDOM_FOREST_PARAMS: dict[str, Any] = {
    "n_estimators": 30,
    "max_depth": 20,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": None,
}


def make_preprocessor(
    build_preprocessor: Callable[[], Any],
    scale_numeric: bool,
) -> Any:
    """Build one fresh preprocessor and set the numeric scaling policy."""
    preprocessor = clone(build_preprocessor())
    if not scale_numeric:
        preprocessor.set_params(numerical__scaler="passthrough")
    return preprocessor


def map_encoded_features(
    encoded_feature_names: Sequence[str],
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> list[str]:
    """Map transformed feature names back to their original variables."""
    numerical = set(numerical_columns)
    categorical = sorted(categorical_columns, key=len, reverse=True)
    mapped: list[str] = []

    for encoded_name in encoded_feature_names:
        name = str(encoded_name)
        if name in numerical:
            mapped.append(name)
            continue

        match = next(
            (
                column
                for column in categorical
                if name.startswith(f"{column}_")
            ),
            None,
        )
        if match is None:
            raise ValueError(
                "Could not map encoded feature back to an original column: "
                f"{name!r}."
            )
        mapped.append(match)

    return mapped


def group_coefficient_l2(
    coefficients: np.ndarray,
    original_feature_names: Sequence[str],
) -> dict[str, float]:
    """Aggregate encoded Perceptron coefficients with an L2 norm."""
    coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
    if coefficients.size != len(original_feature_names):
        raise ValueError("Coefficient count does not match feature-name count.")

    grouped: dict[str, list[float]] = {}
    for coefficient, feature in zip(
        coefficients,
        original_feature_names,
        strict=True,
    ):
        grouped.setdefault(feature, []).append(float(coefficient))

    return {
        feature: float(np.linalg.norm(values, ord=2))
        for feature, values in grouped.items()
    }


def group_shap_importance(
    shap_values: np.ndarray,
    original_feature_names: Sequence[str],
) -> dict[str, float]:
    """Group encoded SHAP contributions, then calculate mean absolute value."""
    values = np.asarray(shap_values, dtype=float)
    if values.ndim != 2:
        raise ValueError("SHAP values must be a two-dimensional matrix.")
    if values.shape[1] != len(original_feature_names):
        raise ValueError("SHAP width does not match feature-name count.")

    grouped_indices: dict[str, list[int]] = {}
    for index, feature in enumerate(original_feature_names):
        grouped_indices.setdefault(feature, []).append(index)

    return {
        feature: float(np.mean(np.abs(values[:, indices].sum(axis=1))))
        for feature, indices in grouped_indices.items()
    }


def extract_positive_class_shap(raw_values: Any) -> np.ndarray:
    """Normalize legacy and current binary-classification SHAP outputs."""
    if isinstance(raw_values, list):
        if len(raw_values) != 2:
            raise ValueError(
                "Expected two SHAP arrays for binary classification."
            )
        values = np.asarray(raw_values[1], dtype=float)
    else:
        values = np.asarray(raw_values, dtype=float)
        if values.ndim == 3:
            if values.shape[2] != 2:
                raise ValueError(
                    "Expected the final SHAP dimension to contain two classes."
                )
            values = values[:, :, 1]

    if values.ndim != 2:
        raise ValueError(
            "Expected positive-class SHAP values with shape "
            "(samples, features)."
        )
    return values


def summarize_fold_importance(
    fold_rows: pd.DataFrame,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Summarize fold-level importance and ranking stability."""
    required = {"Fold", "Feature", "Importance"}
    missing = required - set(fold_rows.columns)
    if missing:
        raise ValueError(f"Fold results are missing columns: {sorted(missing)}")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    ranked = fold_rows.copy()
    ranked["Rank"] = ranked.groupby("Fold")["Importance"].rank(
        method="min",
        ascending=False,
    )
    ranked["In Top K"] = ranked["Rank"] <= top_k

    summary = (
        ranked.groupby("Feature", as_index=False)
        .agg(
            **{
                "Mean Importance": ("Importance", "mean"),
                "Importance Std": ("Importance", "std"),
                "Mean Rank": ("Rank", "mean"),
                "Rank Std": ("Rank", "std"),
                "Top 3 Frequency": ("In Top K", "mean"),
            }
        )
        .sort_values(
            ["Mean Importance", "Feature"],
            ascending=[False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    summary["Overall Rank"] = np.arange(1, len(summary) + 1)
    return summary


def _validate_training_frame(
    training_data: pd.DataFrame,
    column_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series]:
    if training_data.shape[1] != len(column_names):
        raise ValueError(
            f"Expected {len(column_names)} columns, found "
            f"{training_data.shape[1]}."
        )
    if training_data.isna().any().any():
        raise ValueError("The training data contain missing values.")

    training_data = training_data.copy()
    training_data.columns = list(column_names)
    if "Diagnosis" not in training_data.columns:
        raise ValueError("COLUMN_NAMES must include Diagnosis.")

    labels = training_data["Diagnosis"]
    unique_labels = set(labels.unique().tolist())
    if unique_labels == {-1, 1}:
        labels = (labels == 1).astype(int)
    elif unique_labels == {0, 1}:
        labels = labels.astype(int)
    else:
        raise ValueError(
            "Diagnosis must contain exactly {-1, 1} or {0, 1}; "
            f"found {sorted(unique_labels)}."
        )

    features = training_data.drop(columns="Diagnosis")
    return features, labels


def load_training_data(
    training_path: str | Path,
    column_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series]:
    path = Path(training_path)
    if not path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {path.resolve()}")
    training_data = pd.read_csv(path, header=None)
    return _validate_training_frame(training_data, column_names)


def _importance_rows(
    fold: int,
    grouped_importance: dict[str, float],
) -> list[dict[str, float | int | str]]:
    return [
        {
            "Fold": fold,
            "Feature": feature,
            "Importance": importance,
        }
        for feature, importance in grouped_importance.items()
    ]


def run_cross_validated_importance(
    X: pd.DataFrame,
    y: pd.Series,
    build_preprocessor: Callable[[], Any],
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
    cv_splits: int = CV_SPLITS,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate fold-level Perceptron weights and Random Forest SHAP."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "The 'shap' package is required. Install it with: "
            "python.exe -m pip install shap"
        ) from exc

    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2.")

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )
    perceptron_rows: list[dict[str, float | int | str]] = []
    shap_rows: list[dict[str, float | int | str]] = []
    diagnostic_rows: list[dict[str, float | int]] = []

    for fold, (train_indices, validation_indices) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        print(f"Fold {fold}/{cv_splits}")
        X_train = X.iloc[train_indices]
        X_validation = X.iloc[validation_indices]
        y_train = y.iloc[train_indices]
        y_validation = y.iloc[validation_indices]

        perceptron_preprocessor = make_preprocessor(
            build_preprocessor,
            scale_numeric=True,
        )
        X_train_perceptron = np.asarray(
            perceptron_preprocessor.fit_transform(X_train),
            dtype=float,
        )
        X_validation_perceptron = np.asarray(
            perceptron_preprocessor.transform(X_validation),
            dtype=float,
        )
        perceptron_encoded_names = (
            perceptron_preprocessor.get_feature_names_out().tolist()
        )
        perceptron_original_names = map_encoded_features(
            encoded_feature_names=perceptron_encoded_names,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )

        perceptron = Perceptron(
            random_state=random_state,
            **PERCEPTRON_PARAMS,
        )
        perceptron.fit(X_train_perceptron, y_train)
        coefficients = np.asarray(perceptron.coef_, dtype=float).reshape(-1)
        perceptron_grouped = group_coefficient_l2(
            coefficients=coefficients,
            original_feature_names=perceptron_original_names,
        )
        perceptron_rows.extend(_importance_rows(fold, perceptron_grouped))

        forest_preprocessor = make_preprocessor(
            build_preprocessor,
            scale_numeric=False,
        )
        X_train_forest = np.asarray(
            forest_preprocessor.fit_transform(X_train),
            dtype=float,
        )
        X_validation_forest = np.asarray(
            forest_preprocessor.transform(X_validation),
            dtype=float,
        )
        forest_encoded_names = (
            forest_preprocessor.get_feature_names_out().tolist()
        )
        forest_original_names = map_encoded_features(
            encoded_feature_names=forest_encoded_names,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )

        random_forest = RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
            **RANDOM_FOREST_PARAMS,
        )
        random_forest.fit(X_train_forest, y_train)
        explainer = shap.TreeExplainer(
            random_forest,
            feature_perturbation="tree_path_dependent",
        )
        raw_shap_values = explainer.shap_values(
            X_validation_forest,
            check_additivity=True,
        )
        positive_shap_values = extract_positive_class_shap(raw_shap_values)
        shap_grouped = group_shap_importance(
            shap_values=positive_shap_values,
            original_feature_names=forest_original_names,
        )
        shap_rows.extend(_importance_rows(fold, shap_grouped))

        tree_depths = np.array(
            [tree.tree_.max_depth for tree in random_forest.estimators_],
            dtype=float,
        )
        diagnostic_rows.append(
            {
                "Fold": fold,
                "Perceptron Validation Accuracy": float(
                    accuracy_score(
                        y_validation,
                        perceptron.predict(X_validation_perceptron),
                    )
                ),
                "Random Forest Validation Accuracy": float(
                    accuracy_score(
                        y_validation,
                        random_forest.predict(X_validation_forest),
                    )
                ),
                "Random Forest Mean Tree Depth": float(tree_depths.mean()),
                "Random Forest Maximum Tree Depth": int(tree_depths.max()),
            }
        )

    return (
        pd.DataFrame(perceptron_rows),
        pd.DataFrame(shap_rows),
        pd.DataFrame(diagnostic_rows),
    )


def build_ranking_comparison(
    perceptron_summary: pd.DataFrame,
    shap_summary: pd.DataFrame,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Combine both rankings and calculate an average consensus rank."""
    perceptron_columns = {
        "Mean Importance": "Perceptron Mean Importance",
        "Importance Std": "Perceptron Importance Std",
        "Mean Rank": "Perceptron Mean Fold Rank",
        "Top 3 Frequency": "Perceptron Top 3 Frequency",
        "Overall Rank": "Perceptron Rank",
    }
    shap_columns = {
        "Mean Importance": "SHAP Mean Importance",
        "Importance Std": "SHAP Importance Std",
        "Mean Rank": "SHAP Mean Fold Rank",
        "Top 3 Frequency": "SHAP Top 3 Frequency",
        "Overall Rank": "SHAP Rank",
    }
    perceptron = perceptron_summary[
        ["Feature", *perceptron_columns]
    ].rename(columns=perceptron_columns)
    shap = shap_summary[["Feature", *shap_columns]].rename(
        columns=shap_columns
    )
    comparison = perceptron.merge(shap, on="Feature", validate="one_to_one")
    comparison["Consensus Rank Score"] = (
        comparison["Perceptron Rank"] + comparison["SHAP Rank"]
    ) / 2.0
    comparison["Both Top 3"] = (
        (comparison["Perceptron Rank"] <= top_k)
        & (comparison["SHAP Rank"] <= top_k)
    )
    comparison = comparison.sort_values(
        ["Consensus Rank Score", "SHAP Rank", "Perceptron Rank", "Feature"],
        kind="stable",
    ).reset_index(drop=True)
    comparison["Consensus Rank"] = np.arange(1, len(comparison) + 1)
    comparison["Consensus Top 3"] = comparison["Consensus Rank"] <= top_k
    return comparison


def plot_importance(
    summary: pd.DataFrame,
    title: str,
    output_path: str | Path,
) -> None:
    """Save a horizontal importance chart with fold-to-fold variation."""
    import matplotlib.pyplot as plt

    plotted = summary.sort_values("Mean Importance", ascending=True)
    figure_height = max(5.0, 0.36 * len(plotted))
    fig, ax = plt.subplots(figsize=(9.0, figure_height))
    ax.barh(
        plotted["Feature"],
        plotted["Mean Importance"],
        xerr=plotted["Importance Std"].fillna(0.0),
        color="#4C78A8",
        edgecolor="#222222",
        linewidth=0.5,
        capsize=2,
    )
    ax.set_title(title)
    ax.set_xlabel("Mean importance across CV folds")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # Import the established project schema and preprocessing pipeline only
    # when the experiment is run, keeping the helper functions testable.
    from preprocessing_and_cv import (
        CATEGORICAL_COLUMNS,
        COLUMN_NAMES,
        NUMERICAL_COLUMNS,
        build_preprocessor,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_training_data(TRAINING_PATH, COLUMN_NAMES)

    expected_features = set(NUMERICAL_COLUMNS) | set(CATEGORICAL_COLUMNS)
    if expected_features != set(X.columns):
        raise ValueError(
            "The numerical and categorical column lists do not exactly cover "
            "the predictor columns."
        )

    print(f"Training rows: {len(X)}")
    print(f"Original predictors: {X.shape[1]}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    perceptron_folds, shap_folds, diagnostics = (
        run_cross_validated_importance(
            X=X,
            y=y,
            build_preprocessor=build_preprocessor,
            numerical_columns=NUMERICAL_COLUMNS,
            categorical_columns=CATEGORICAL_COLUMNS,
        )
    )
    perceptron_summary = summarize_fold_importance(perceptron_folds)
    shap_summary = summarize_fold_importance(shap_folds)
    comparison = build_ranking_comparison(
        perceptron_summary=perceptron_summary,
        shap_summary=shap_summary,
    )

    perceptron_folds.to_csv(
        OUTPUT_DIR / "perceptron_fold_importance.csv",
        index=False,
    )
    perceptron_summary.to_csv(
        OUTPUT_DIR / "perceptron_importance_summary.csv",
        index=False,
    )
    shap_folds.to_csv(
        OUTPUT_DIR / "shap_fold_importance.csv",
        index=False,
    )
    shap_summary.to_csv(
        OUTPUT_DIR / "shap_importance_summary.csv",
        index=False,
    )
    comparison.to_csv(
        OUTPUT_DIR / "feature_ranking_comparison.csv",
        index=False,
    )
    diagnostics.to_csv(
        OUTPUT_DIR / "fold_diagnostics.csv",
        index=False,
    )

    plot_importance(
        perceptron_summary,
        "Perceptron coefficient importance",
        OUTPUT_DIR / "perceptron_importance.pdf",
    )
    plot_importance(
        shap_summary,
        "Random Forest Tree SHAP importance",
        OUTPUT_DIR / "shap_importance.pdf",
    )

    protocol = {
        "phase": "Phase 2 feature importance",
        "training_file": str(TRAINING_PATH.name),
        "testing_data_used": False,
        "cv_splits": CV_SPLITS,
        "random_state": RANDOM_STATE,
        "top_k": TOP_K,
        "perceptron": {
            "scale_numeric": True,
            "parameters": PERCEPTRON_PARAMS,
            "encoded_coefficient_aggregation": "L2 norm",
        },
        "random_forest": {
            "phase1_config_id": "random_forest_021",
            "scale_numeric": False,
            "parameters": RANDOM_FOREST_PARAMS,
            "shap_explainer": "TreeExplainer",
            "feature_perturbation": "tree_path_dependent",
            "one_hot_grouping": (
                "sum SHAP contributions within each original feature for "
                "each observation, then calculate mean absolute value"
            ),
        },
    }
    (OUTPUT_DIR / "phase2_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pd.option_context("display.max_columns", None, "display.width", 220):
        print("\nFeature ranking comparison:")
        print(comparison.round(6).to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
