"""Run the fixed-parameter feature ablation experiment.

The experiment compares the five representative Phase 1 configurations on
all 17 predictors and on the same rows after removing the three predictors
selected in Phase 2. Every fold receives a fresh preprocessing/model pipeline
and both feature conditions reuse one shared stratified split list.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from preprocessing_and_cv import (
    COLUMN_NAMES,
    build_configured_pipeline,
    evaluate_test_set,
    load_datasets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
CV_SPLITS = 5

PREDICTOR_COLUMNS = [
    column for column in COLUMN_NAMES if column != "Diagnosis"
]
DROPPED_FEATURES = (
    "Negative Symptom Score",
    "GAF Score",
    "Positive Symptom Score",
)
ABLATED_FEATURES = [
    column for column in PREDICTOR_COLUMNS if column not in DROPPED_FEATURES
]
PRIMARY_METRICS = (
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "ROC-AUC",
)
MODEL_KEYS = (
    "logistic",
    "knn",
    "perceptron",
    "mlp",
    "random_forest",
)

# These are the exact representative configurations recorded in
# results/full/best_by_cv_full.csv. They are frozen here so this experiment cannot
# silently re-select parameters from a new result file or from test metrics.
PHASE1_CONFIGS: dict[str, dict[str, Any]] = {
    "logistic": {
        "config_id": "logistic_002",
        "model_params": {"C": 0.01},
        "scale_numeric": False,
    },
    "knn": {
        "config_id": "knn_002",
        "model_params": {
            "n_neighbors": 69,
            "metric": "euclidean",
            "weights": "uniform",
        },
        "scale_numeric": False,
    },
    "perceptron": {
        "config_id": "perceptron_002",
        "model_params": {
            "penalty": None,
            "max_iter": 200,
            "tol": 0.001,
        },
        "scale_numeric": False,
    },
    "mlp": {
        "config_id": "mlp_016",
        "model_params": {
            "hidden_layer_sizes": (32,),
            "alpha": 0.001,
            "learning_rate_init": 0.01,
        },
        "scale_numeric": False,
    },
    "random_forest": {
        "config_id": "random_forest_001",
        "model_params": {
            "n_estimators": 10,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": None,
        },
        "scale_numeric": False,
    },
}


def _jsonable(value: Any) -> Any:
    """Convert tuples and nested values into JSON-compatible values."""
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def make_shared_splits(
    y: pd.Series | Sequence[int] | np.ndarray,
    cv_splits: int = CV_SPLITS,
    random_state: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create one deterministic fold list for both feature conditions."""
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2.")
    labels = np.asarray(y)
    splitter = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )
    placeholder_X = np.zeros((len(labels), 1), dtype=float)
    return [
        (
            np.asarray(train_indices, dtype=int),
            np.asarray(validation_indices, dtype=int),
        )
        for train_indices, validation_indices in splitter.split(
            placeholder_X,
            labels,
        )
    ]


def _evaluation_row(
    evaluation: dict[str, float | int],
    prefix: str = "",
) -> dict[str, float | int]:
    """Select the six primary metrics and confusion counts."""
    prefix_text = f"{prefix} " if prefix else ""
    row: dict[str, float | int] = {}
    for metric in PRIMARY_METRICS:
        source_key = f"Test {metric}"
        row[f"{prefix_text}{metric}".strip()] = float(evaluation[source_key])
    for confusion_name in ("TN", "FP", "FN", "TP"):
        row[f"{prefix_text}{confusion_name}".strip()] = int(
            evaluation[f"Test {confusion_name}"]
        )
    return row


def _feature_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Select predictors while preserving row indices and column names."""
    return frame.loc[:, list(feature_columns)]


def _build_pipeline(
    model_key: str,
    config: dict[str, Any],
    feature_columns: Sequence[str],
    random_state: int,
):
    return build_configured_pipeline(
        model_key=model_key,
        model_params=config["model_params"],
        scale_numeric=bool(config["scale_numeric"]),
        random_state=random_state,
        feature_columns=feature_columns,
    )


def _run_fold(
    model_key: str,
    config: dict[str, Any],
    feature_set_name: str,
    feature_columns: Sequence[str],
    fold: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> dict[str, float | int | str]:
    pipeline = _build_pipeline(
        model_key=model_key,
        config=config,
        feature_columns=feature_columns,
        random_state=random_state,
    )
    X_fit = _feature_frame(X_train.iloc[train_indices], feature_columns)
    X_validation = _feature_frame(
        X_train.iloc[validation_indices],
        feature_columns,
    )
    y_fit = y_train.iloc[train_indices]
    y_validation = y_train.iloc[validation_indices]
    pipeline.fit(X_fit, y_fit)
    evaluation = evaluate_test_set(pipeline, X_validation, y_validation)

    row: dict[str, float | int | str] = {
        "Model": model_key,
        "Config ID": config["config_id"],
        "Feature Set": feature_set_name,
        "Fold": fold,
        "Feature Count": len(feature_columns),
        "Train Rows": len(train_indices),
        "Validation Rows": len(validation_indices),
    }
    row.update(_evaluation_row(evaluation))
    return row


def _summarize_cv(fold_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    group_columns = ["Model", "Config ID", "Feature Set"]
    for group_values, group in fold_results.groupby(
        group_columns,
        sort=False,
    ):
        model, config_id, feature_set = group_values
        row: dict[str, float | int | str] = {
            "Model": model,
            "Config ID": config_id,
            "Feature Set": feature_set,
            "Feature Count": int(group["Feature Count"].iloc[0]),
            "Folds": int(len(group)),
        }
        for metric in PRIMARY_METRICS:
            row[f"CV {metric} Mean"] = float(group[metric].mean())
            row[f"CV {metric} Std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_differences(fold_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for model_key in MODEL_KEYS:
        model_rows = fold_results[fold_results["Model"] == model_key]
        if model_rows.empty:
            continue
        full = model_rows[model_rows["Feature Set"] == "full"].set_index("Fold")
        ablated = model_rows[
            model_rows["Feature Set"] == "ablated"
        ].set_index("Fold")
        shared_folds = full.index.intersection(ablated.index)
        if len(shared_folds) != len(full) or len(shared_folds) != len(ablated):
            raise ValueError(f"Full/ablated fold mismatch for {model_key}.")

        row: dict[str, float | int | str] = {
            "Model": model_key,
            "Config ID": str(full["Config ID"].iloc[0]),
            "Folds": int(len(shared_folds)),
        }
        for metric in PRIMARY_METRICS:
            full_values = full.loc[shared_folds, metric].to_numpy(dtype=float)
            ablated_values = ablated.loc[shared_folds, metric].to_numpy(
                dtype=float
            )
            deltas = ablated_values - full_values
            row[f"Full {metric} Mean"] = float(full_values.mean())
            row[f"Ablated {metric} Mean"] = float(ablated_values.mean())
            row[f"Delta {metric} Mean (Ablated - Full)"] = float(deltas.mean())
            row[f"Delta {metric} Std"] = float(deltas.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _test_results(
    model_keys: Sequence[str],
    configs: dict[str, dict[str, Any]],
    feature_sets: dict[str, Sequence[str]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for model_key in model_keys:
        config = configs[model_key]
        for feature_set_name, feature_columns in feature_sets.items():
            pipeline = _build_pipeline(
                model_key=model_key,
                config=config,
                feature_columns=feature_columns,
                random_state=random_state,
            )
            pipeline.fit(
                _feature_frame(X_train, feature_columns),
                y_train,
            )
            evaluation = evaluate_test_set(
                pipeline,
                _feature_frame(X_test, feature_columns),
                y_test,
            )
            row: dict[str, float | int | str] = {
                "Model": model_key,
                "Config ID": config["config_id"],
                "Feature Set": feature_set_name,
                "Feature Count": len(feature_columns),
                "Training Rows": len(X_train),
                "Testing Rows": len(X_test),
                "Evaluation Status": "exploratory",
            }
            row.update(_evaluation_row(evaluation, prefix="Test"))
            rows.append(row)
    return pd.DataFrame(rows)


def _interpret_results(
    paired_differences: pd.DataFrame,
    test_results: pd.DataFrame,
) -> str:
    lines = [
        "# Feature ablation interpretation",
        "",
        "The fixed comparison removes Negative Symptom Score, GAF Score, and "
        "Positive Symptom Score together. Deltas are ablated minus full; "
        "negative values mean the metric decreased after removal.",
        "",
        "## Largest paired CV changes",
        "",
    ]

    candidates: list[tuple[float, str, str, float, float]] = []
    for _, row in paired_differences.iterrows():
        for metric in PRIMARY_METRICS:
            delta = float(row[f"Delta {metric} Mean (Ablated - Full)"])
            candidates.append(
                (
                    abs(delta),
                    str(row["Model"]),
                    metric,
                    delta,
                    float(row[f"Full {metric} Mean"]),
                )
            )
    for _, model, metric, delta, full_mean in sorted(
        candidates,
        reverse=True,
    )[:5]:
        lines.append(
            f"- {model} - {metric}: full mean {full_mean:.6f}; "
            f"paired delta {delta:+.6f} (ablated - full)."
        )

    lines.extend(
        [
            "",
            "## Fixed-test comparison",
            "",
            "The test rows below were generated after fitting each pipeline on "
            "all training rows. They are exploratory because Phase 1 evaluated "
            "this same test set for all 146 candidate configurations; they were "
            "not used to select the ablation feature set or parameters.",
            "",
        ]
    )
    test_candidates: list[tuple[float, str, str, float, float]] = []
    for model_key in test_results["Model"].drop_duplicates():
        model_rows = test_results[test_results["Model"] == model_key].set_index(
            "Feature Set"
        )
        if not {"full", "ablated"}.issubset(model_rows.index):
            continue
        for metric in PRIMARY_METRICS:
            full_value = float(model_rows.loc["full", f"Test {metric}"])
            ablated_value = float(
                model_rows.loc["ablated", f"Test {metric}"]
            )
            delta = ablated_value - full_value
            test_candidates.append(
                (abs(delta), str(model_key), metric, delta, full_value)
            )
    for _, model, metric, delta, full_value in sorted(
        test_candidates,
        reverse=True,
    )[:5]:
        lines.append(
            f"- {model} - {metric}: full {full_value:.6f}; "
            f"test delta {delta:+.6f} (ablated - full)."
        )

    lines.extend(
        [
            "",
            "These changes describe predictive performance on this dataset. "
            "They do not establish causal effects, clinical utility, clinical "
            "diagnostic validity, or early-prediction capability.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_outputs(
    output_dir: Path,
    fold_results: pd.DataFrame,
    cv_summary: pd.DataFrame,
    paired_differences: pd.DataFrame,
    test_results: pd.DataFrame,
    protocol: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(output_dir / "ablation_fold_results.csv", index=False)
    cv_summary.to_csv(output_dir / "ablation_cv_summary.csv", index=False)
    paired_differences.to_csv(
        output_dir / "ablation_cv_paired_differences.csv",
        index=False,
    )
    test_results.to_csv(output_dir / "ablation_test_results.csv", index=False)
    (output_dir / "ablation_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "ablation_interpretation.md").write_text(
        _interpret_results(paired_differences, test_results),
        encoding="utf-8",
    )


def run_ablation(
    training_path: str | Path = PROJECT_ROOT / "training_data.csv",
    testing_path: str | Path = PROJECT_ROOT / "testing_data.csv",
    output_dir: str | Path = PROJECT_ROOT / "results" / "ablation",
    model_keys: Sequence[str] = MODEL_KEYS,
    cv_splits: int = CV_SPLITS,
    random_state: int = RANDOM_STATE,
) -> dict[str, pd.DataFrame]:
    """Run the ablation experiment and write reproducibility artifacts."""
    unknown_models = sorted(set(model_keys) - set(MODEL_KEYS))
    if unknown_models:
        raise ValueError(f"Unknown model keys: {unknown_models}")
    if not model_keys:
        raise ValueError("At least one model key is required.")

    X_train, y_train, X_test, y_test = load_datasets(
        training_path,
        testing_path,
    )
    feature_sets: dict[str, Sequence[str]] = {
        "full": PREDICTOR_COLUMNS,
        "ablated": ABLATED_FEATURES,
    }
    shared_splits = make_shared_splits(
        y_train,
        cv_splits=cv_splits,
        random_state=random_state,
    )

    fold_rows: list[dict[str, float | int | str]] = []
    configs = {model_key: PHASE1_CONFIGS[model_key] for model_key in model_keys}
    for model_key in model_keys:
        config = configs[model_key]
        for feature_set_name, feature_columns in feature_sets.items():
            for fold, (train_indices, validation_indices) in enumerate(
                shared_splits,
                start=1,
            ):
                fold_rows.append(
                    _run_fold(
                        model_key=model_key,
                        config=config,
                        feature_set_name=feature_set_name,
                        feature_columns=feature_columns,
                        fold=fold,
                        train_indices=train_indices,
                        validation_indices=validation_indices,
                        X_train=X_train,
                        y_train=y_train,
                        random_state=random_state,
                    )
                )

    fold_results = pd.DataFrame(fold_rows)
    cv_summary = _summarize_cv(fold_results)
    paired_differences = _paired_differences(fold_results)
    test_results = _test_results(
        model_keys=model_keys,
        configs=configs,
        feature_sets=feature_sets,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_state=random_state,
    )

    protocol = {
        "phase": "Ablation fixed-parameter feature comparison",
        "training_file": str(Path(training_path).name),
        "testing_file": str(Path(testing_path).name),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "predictor_count_full": len(PREDICTOR_COLUMNS),
        "predictor_count_ablated": len(ABLATED_FEATURES),
        "dropped_features": list(DROPPED_FEATURES),
        "feature_sets": {
            name: list(features) for name, features in feature_sets.items()
        },
        "models_run": list(model_keys),
        "phase1_representative_configurations": _jsonable(configs),
        "cv_splits": cv_splits,
        "random_state": random_state,
        "same_cv_indices_for_full_and_ablated": True,
        "metrics": list(PRIMARY_METRICS),
        "metric_difference_definition": "ablated minus full",
        "preprocessing": {
            "numeric": (
                "StandardScaler is passthrough for all five frozen Phase 1 "
                "configurations (scale_numeric=false); if enabled in another "
                "run, it is fitted within each training fold"
            ),
            "categorical": (
                "OneHotEncoder(handle_unknown='ignore', sparse_output=False), "
                "fitted within each training fold"
            ),
            "knn_distance": (
                "Euclidean for knn_002; no Mahalanobis covariance is fitted"
            ),
            "final_test_fit": "same pipeline fitted on all training rows",
        },
        "scale_numeric_by_model": {
            model_key: bool(configs[model_key]["scale_numeric"])
            for model_key in model_keys
        },
        "test_evaluation": {
            "performed": True,
            "used_for_ablation_selection": False,
            "status": (
                "exploratory; Phase 1 evaluated this same test set for all "
                "146 candidate configurations"
            ),
        },
    }
    output_path = Path(output_dir)
    _write_outputs(
        output_dir=output_path,
        fold_results=fold_results,
        cv_summary=cv_summary,
        paired_differences=paired_differences,
        test_results=test_results,
        protocol=protocol,
    )

    return {
        "fold_results": fold_results,
        "cv_summary": cv_summary,
        "paired_differences": paired_differences,
        "test_results": test_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training",
        type=Path,
        default=PROJECT_ROOT / "training_data.csv",
    )
    parser.add_argument(
        "--testing",
        type=Path,
        default=PROJECT_ROOT / "testing_data.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "ablation",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_KEYS,
        default=list(MODEL_KEYS),
    )
    parser.add_argument("--folds", type=int, default=CV_SPLITS)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_ablation(
        training_path=args.training,
        testing_path=args.testing,
        output_dir=args.output_dir,
        model_keys=args.models,
        cv_splits=args.folds,
        random_state=args.random_state,
    )
    print(f"Fold rows: {len(outputs['fold_results'])}")
    print(f"CV summary rows: {len(outputs['cv_summary'])}")
    print(f"Paired difference rows: {len(outputs['paired_differences'])}")
    print(f"Test rows: {len(outputs['test_results'])}")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print("\nCV summary:")
        print(outputs["cv_summary"].round(6).to_string(index=False))
        print("\nPaired differences:")
        print(
            outputs["paired_differences"]
            .round(6)
            .to_string(index=False)
        )
    print(f"\nResults saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
