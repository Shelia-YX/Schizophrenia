"""Frozen hyperparameter grids for Phase 1 full-feature experiments."""

from __future__ import annotations

from itertools import product
from typing import Any


MODEL_DISPLAY_NAMES = {
    "logistic": "Logistic Regression",
    "knn": "k-NN",
    "perceptron": "Perceptron",
    "mlp": "MLP",
    "random_forest": "Random Forest",
}


def _finalize_configs(
    model_key: str,
    raw_configs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "config_id": f"{model_key}_{index:03d}",
            "scale_numeric": config.pop("scale_numeric"),
            "model_params": config,
        }
        for index, config in enumerate(raw_configs, start=1)
    ]


def _logistic_configs(quick: bool) -> list[dict[str, Any]]:
    c_values = [0.1, 1.0] if quick else [0.01, 0.1, 1.0, 10.0]
    scaling = [True] if quick else [True, False]
    return [
        {"C": c_value, "scale_numeric": scale_numeric}
        for c_value, scale_numeric in product(c_values, scaling)
    ]


def _knn_configs(quick: bool) -> list[dict[str, Any]]:
    neighbors = [89] if quick else [69,79,89,99,109]
    metrics = ["mahalanobis"] if quick else ["euclidean", "manhattan", "mahalanobis"]
    weights = ["uniform"]
    scaling = [True] if quick else [True, False]
    return [
        {
            "n_neighbors": n_neighbors,
            "metric": metric,
            "weights": weight,
            "scale_numeric": scale_numeric,
        }
        for n_neighbors, metric, weight, scale_numeric in product(
            neighbors,
            metrics,
            weights,
            scaling,
        )
    ]


def _perceptron_configs(quick: bool) -> list[dict[str, Any]]:
    if quick:
        return [
            {
                "penalty": None,
                "max_iter": 1000,
                "tol": 1e-3,
                "scale_numeric": True,
            },
            {
                "penalty": "l2",
                "alpha": 0.0001,
                "max_iter": 1000,
                "tol": 1e-3,
                "scale_numeric": True,
            },
        ]

    raw_configs: list[dict[str, Any]] = []

    for max_iter, tol, scale_numeric in product(
            [200, 300, 500],
            [1e-3, 1e-4],
            [True, False],
    ):
        raw_configs.append(
            {
                "penalty": None,
                "max_iter": max_iter,
                "tol": tol,
                "scale_numeric": scale_numeric,
            }
        )

    for penalty, alpha, max_iter, tol, scale_numeric in product(
        ["l1", "l2"],
        [1e-3, 1e-2],
        [200, 300, 500],
        [1e-3, 1e-4],
        [True, False],
    ):
        raw_configs.append(
            {
                "penalty": penalty,
                "alpha": alpha,
                "max_iter": max_iter,
                "tol": tol,
                "scale_numeric": scale_numeric,
            }
        )
    return raw_configs


def _mlp_configs(quick: bool) -> list[dict[str, Any]]:
    hidden_layers = [(16,)] if quick else [(16,), (32,), (32, 16)]
    alphas = [0.001] if quick else [0.0001, 0.001]
    learning_rates = [0.001] if quick else [0.001, 0.01]
    scaling = [True] if quick else [True, False]
    return [
        {
            "hidden_layer_sizes": hidden_layer_sizes,
            "alpha": alpha,
            "learning_rate_init": learning_rate,
            "scale_numeric": scale_numeric,
        }
        for hidden_layer_sizes, alpha, learning_rate, scale_numeric in product(
            hidden_layers,
            alphas,
            learning_rates,
            scaling,
        )
    ]

def _random_forest_configs(quick: bool) -> list[dict[str, Any]]:
    """Return Random Forest configurations."""
    if quick:
        return [
            {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "class_weight": None,
                "scale_numeric": False,
            }
        ]
    return [
        {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": "sqrt",
            "class_weight": class_weight,
            "scale_numeric": False,
        }
        for (
            n_estimators,
            max_depth,
            min_samples_leaf,
            class_weight,
        ) in product(
            [10, 30],
            [None, 10, 20],
            [1, 5],
            [None, "balanced"],
        )
    ]

def get_parameter_configs(
    model_key: str,
    quick: bool = False,
) -> list[dict[str, Any]]:
    """Return deterministic, JSON-serializable configurations."""
    builders = {
        "logistic": _logistic_configs,
        "knn": _knn_configs,
        "perceptron": _perceptron_configs,
        "mlp": _mlp_configs,
        "random_forest": _random_forest_configs,
    }
    if model_key not in builders:
        raise ValueError(
            f"Unknown model '{model_key}'. Choose from {sorted(builders)}."
        )
    raw_configs = builders[model_key](quick)
    return _finalize_configs(model_key, raw_configs)
