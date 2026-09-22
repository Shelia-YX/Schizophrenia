"""Run Phase 1 full-feature hyperparameter sensitivity experiments.

Every frozen configuration receives:
  1. stratified cross-validation on the training set; and
  2. an evaluation on the fixed testing set after fitting all training rows.

All configurations are reported. The optional best-by-CV summary is selected
only with training-set cross-validation metrics, never with testing metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config_full import (MODEL_DISPLAY_NAMES, get_parameter_configs)
from preprocessing_and_cv import (
    build_configured_pipeline,
    evaluate_test_set,
    load_datasets,
)

RANDOM_STATE = 42


def load_data(
    training_path: str | Path,
    testing_path: str | Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load the fixed Full-feature development and testing sets."""
    return load_datasets(training_path, testing_path)


def build_scoring() -> dict[str, object]:
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
        "PR-AUC": "average_precision",
    }


def evaluate_configuration(
    model_key: str,
    config: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: StratifiedKFold,
    scoring: dict[str, object],
    random_state: int,
    n_jobs: int,
) -> dict[str, Any]:
    """Run CV and holdout evaluation for one frozen configuration."""
    pipeline = build_configured_pipeline(
        model_key=model_key,
        model_params=config["model_params"],
        scale_numeric=config["scale_numeric"],
        random_state=random_state,
    )
    cv_scores = cross_validate(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=n_jobs,
        error_score="raise",
    )

    result: dict[str, Any] = {
        "Model": MODEL_DISPLAY_NAMES[model_key],
        "Config ID": config["config_id"],
        "Scale Numeric": bool(config["scale_numeric"]),
        "Parameters JSON": json.dumps(
            config["model_params"],
            sort_keys=True,
        ),
    }
    for metric_name in scoring:
        values = cv_scores[f"test_{metric_name}"]
        result[f"CV {metric_name} Mean"] = float(values.mean())
        result[f"CV {metric_name} Std"] = float(values.std(ddof=1))
    result["CV Fit Time Mean (s)"] = float(cv_scores["fit_time"].mean())

    final_pipeline = clone(pipeline)
    start_time = time.perf_counter()
    final_pipeline.fit(X_train, y_train)
    result["Full Fit Time (s)"] = float(time.perf_counter() - start_time)
    result.update(evaluate_test_set(final_pipeline, X_test, y_test))
    return result


def run_model_experiments(
    model_key: str,
    configs: list[dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str | Path,
    cv_splits: int = 5,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Run and checkpoint every configuration for one model."""
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2.")
    if not configs:
        raise ValueError("At least one parameter configuration is required.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_key}_full_results.csv"
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )
    scoring = build_scoring()
    rows: list[dict[str, Any]] = []

    for index, config in enumerate(configs, start=1):
        print(
            f"[{MODEL_DISPLAY_NAMES[model_key]}] "
            f"configuration {index}/{len(configs)}: {config['config_id']}"
        )
        row = evaluate_configuration(
            model_key=model_key,
            config=config,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_path, index=False)

    return pd.DataFrame(rows)


def select_best_by_cv(results: pd.DataFrame) -> pd.DataFrame:
    """Choose one representative configuration per model using CV only."""
    required = {
        "Model",
        "Config ID",
        "CV ROC-AUC Mean",
        "CV F1 Mean",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Results are missing columns: {sorted(missing)}")

    sort_columns = ["Model", "CV ROC-AUC Mean", "CV F1 Mean"]
    ascending = [True, False, False]
    if "CV Accuracy Mean" in results.columns:
        sort_columns.append("CV Accuracy Mean")
        ascending.append(False)
    sort_columns.append("Config ID")
    ascending.append(True)

    ordered = results.sort_values(
        by=sort_columns,
        ascending=ascending,
        kind="stable",
    )
    return ordered.groupby("Model", as_index=False, sort=True).head(1).reset_index(drop=True)


def combine_available_results(output_dir: str | Path) -> pd.DataFrame:
    """Combine every completed per-model result file in the output folder."""
    output_dir = Path(output_dir)
    frames: list[pd.DataFrame] = []
    for model_key in MODEL_DISPLAY_NAMES:
        result_path = output_dir / f"{model_key}_full_results.csv"
        if result_path.is_file():
            frames.append(pd.read_csv(result_path))
    if not frames:
        raise FileNotFoundError(
            f"No per-model result files were found in {output_dir}."
        )
    return pd.concat(frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 1 Full-feature CV and holdout sensitivity experiments."
    )
    parser.add_argument(
        "--training",
        type=Path,
        default=Path("training_data.csv"),
    )
    parser.add_argument(
        "--testing",
        type=Path,
        default=Path("testing_data.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/full"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_DISPLAY_NAMES),
        default=list(MODEL_DISPLAY_NAMES),
        help="Models to run; default: all available models.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small grid to verify the environment and workflow.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X_train, y_train, X_test, y_test = load_data(
        args.training,
        args.testing,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Models: {args.models}")
    print(f"Quick grid: {args.quick}")

    protocol: dict[str, Any] = {
        "phase": "Phase 1 Full features",
        "folds": args.folds,
        "random_state": args.random_state,
        "quick": args.quick,
        "models_run_this_invocation": args.models,
        "selection_metric": "CV ROC-AUC Mean",
        "test_metrics_used_for_selection": False,
        "models": {
            model_key: get_parameter_configs(model_key, quick=args.quick)
            for model_key in MODEL_DISPLAY_NAMES
        },
    }

    for model_key in args.models:
        configs = get_parameter_configs(model_key, quick=args.quick)
        run_model_experiments(
            model_key=model_key,
            configs=configs,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            output_dir=args.output_dir,
            cv_splits=args.folds,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )

    combined = combine_available_results(args.output_dir)
    combined_path = args.output_dir / "full_all_results.csv"
    combined.to_csv(combined_path, index=False)

    best_by_cv = select_best_by_cv(combined)
    best_path = args.output_dir / "best_by_cv_full.csv"
    best_by_cv.to_csv(best_path, index=False)

    protocol_path = args.output_dir / "full_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pd.option_context("display.max_columns", None, "display.width", 260):
        print("\nRepresentative configurations selected by CV only:")
        print(best_by_cv.round(6).to_string(index=False))
    print(f"\nAll results: {combined_path.resolve()}")
    print(f"CV-selected summary: {best_path.resolve()}")
    print(f"Frozen protocol: {protocol_path.resolve()}")


if __name__ == "__main__":
    main()

"""
To run the code:
python.exe -m full.full --models logistic --output-dir results\full --folds 5 --n-jobs -1
python.exe -m full.full --models knn --output-dir results\full --folds 5 --n-jobs -1
python.exe -m full.full --models perceptron --output-dir results\full --folds 5 --n-jobs -1
python.exe -m full.full --models mlp --output-dir results\full --folds 5 --n-jobs 1
"""
