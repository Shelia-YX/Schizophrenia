"""Plot how the highest-ranked feature relates to diagnosis and peers.

The experiment uses training data only. It takes the six highest features
from the consensus importance table, uses the first-ranked feature as the
x-axis, and compares its density across diagnosis and ranks two through six.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from preprocessing_and_cv import COLUMN_NAMES


PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_PATH = PROJECT_ROOT / "training_data.csv"
RANKING_PATH = (
    PROJECT_ROOT / "results" / "importance" / "feature_ranking_comparison.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "relationships"
TOP_N = 6

PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
)
QUANTILE_COLORS = {
    "Low": PALETTE[0],
    "Medium": PALETTE[2],
    "High": PALETTE[1],
}


def select_ranked_features(
    ranking: pd.DataFrame,
    top_n: int = TOP_N,
) -> list[str]:
    """Return feature names ordered by the consensus rank column."""
    required = {"Feature", "Consensus Rank"}
    missing = required - set(ranking.columns)
    if missing:
        raise ValueError(f"Ranking table is missing columns: {sorted(missing)}")
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    ordered = ranking.sort_values(
        ["Consensus Rank", "Feature"],
        kind="stable",
    )
    if len(ordered) < top_n:
        raise ValueError(
            f"Ranking table has {len(ordered)} rows; {top_n} are required."
        )
    selected = ordered["Feature"].head(top_n).astype(str).tolist()
    if len(set(selected)) != len(selected):
        raise ValueError("Top-ranked feature names must be unique.")
    return selected


def group_hospitalizations(values: pd.Series) -> pd.Series:
    """Group hospitalization counts as 0, 1-5, and 6+."""
    numeric = pd.to_numeric(values, errors="raise")
    if (numeric.dropna() < 0).any():
        raise ValueError("Hospitalizations cannot be negative.")
    return pd.cut(
        numeric,
        bins=[-np.inf, 0, 5, np.inf],
        labels=["0", "1-5", "6+"],
        ordered=True,
    )


def _quantile_groups(values: pd.Series) -> tuple[pd.Series, list[float]]:
    numeric = pd.to_numeric(values, errors="raise")
    grouped, edges = pd.qcut(
        numeric,
        q=3,
        labels=["Low", "Medium", "High"],
        retbins=True,
    )
    return grouped, [float(edge) for edge in edges]


def _categorical_groups(values: pd.Series, variable: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if variable == "Diagnosis":
        labels = {
            -1: "Not schizophrenic",
            0: "Not schizophrenic",
            1: "Schizophrenic",
        }
    elif variable in {"Family History", "Substance Use"}:
        labels = {0: "No", 1: "Yes"}
    else:
        return numeric.astype(str)

    grouped = numeric.map(labels)
    if grouped.isna().any():
        unknown = sorted(numeric[grouped.isna()].dropna().unique().tolist())
        raise ValueError(f"Unknown {variable} values: {unknown}")
    return grouped


def build_groups(
    data: pd.DataFrame,
    variable: str,
) -> tuple[pd.Series, dict[str, Any]]:
    """Build plotting groups and return their reproducibility metadata."""
    if variable == "Hospitalizations":
        return group_hospitalizations(data[variable]), {
            "method": "fixed intervals",
            "groups": {"0": "0", "1-5": "1 through 5", "6+": "6 or more"},
        }
    if variable in {"Positive Symptom Score", "Negative Symptom Score"}:
        grouped, edges = _quantile_groups(data[variable])
        return grouped, {
            "method": "training-set tertiles",
            "labels": ["Low", "Medium", "High"],
            "edges": edges,
        }
    grouped = _categorical_groups(data[variable], variable)
    return grouped, {"method": "original categories"}


def _load_training_data(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    for column in COLUMN_NAMES:
        data[column] = pd.to_numeric(data[column], errors="raise")
    return data


def _ordered_groups(groups: pd.Series) -> list[str]:
    if isinstance(groups.dtype, pd.CategoricalDtype):
        return [
            str(value)
            for value in groups.cat.categories
            if (groups == value).any()
        ]
    return sorted(groups.dropna().astype(str).unique().tolist())


def _plot_density_panel(
    ax: Any,
    x_values: pd.Series,
    groups: pd.Series,
    variable: str,
    x_label: str,
) -> None:
    x_numeric = pd.to_numeric(x_values, errors="raise")
    x_grid = np.linspace(float(x_numeric.min()), float(x_numeric.max()), 400)
    group_text = groups.astype("object").astype(str)

    for index, group_name in enumerate(_ordered_groups(groups)):
        values = x_numeric[group_text == group_name].dropna().to_numpy(dtype=float)
        if len(values) < 2 or np.isclose(values.std(ddof=1), 0.0):
            continue
        density = gaussian_kde(values)(x_grid)
        color = QUANTILE_COLORS.get(
            group_name,
            PALETTE[index % len(PALETTE)],
        )
        ax.plot(x_grid, density, color=color, linewidth=1.8, label=group_name)
        ax.fill_between(x_grid, density, color=color, alpha=0.16)

    ax.set_title(variable)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.set_xlim(float(x_numeric.min()), float(x_numeric.max()))
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.2, linewidth=0.6)
    if variable == "Hospitalizations":
        _, y_max = ax.get_ylim()
        ax.set_ylim(0.0, y_max*1.1)
        legend_location = "upper left"
    else:
        legend_location = "best"
    ax.legend(
        frameon=False,
        fontsize=8,
        title=None,
        loc=legend_location,
    )


def _group_summary(
    data: pd.DataFrame,
    x_feature: str,
    grouped_variables: dict[str, pd.Series],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable, groups in grouped_variables.items():
        frame = pd.DataFrame({"x": data[x_feature], "group": groups}).dropna()
        for group_name, group in frame.groupby("group", observed=True, sort=False):
            values = group["x"].astype(float)
            rows.append(
                {
                    "X Feature": x_feature,
                    "Comparison Variable": variable,
                    "Group": str(group_name),
                    "Rows": int(len(values)),
                    "Mean": float(values.mean()),
                    "Std": float(values.std(ddof=1)),
                    "Median": float(values.median()),
                    "Minimum": float(values.min()),
                    "Maximum": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def run_relationship_analysis(
    training_path: str | Path = TRAINING_PATH,
    ranking_path: str | Path = RANKING_PATH,
    output_dir: str | Path = OUTPUT_DIR,
) -> dict[str, Path]:
    """Generate relationship plots and reproducibility artifacts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _load_training_data(training_path)
    ranking = pd.read_csv(ranking_path)
    top_features = select_ranked_features(ranking)
    x_feature = top_features[0]
    comparison_variables = ["Diagnosis", *top_features[1:]]

    grouped_variables: dict[str, pd.Series] = {}
    grouping_protocol: dict[str, dict[str, Any]] = {}
    for variable in comparison_variables:
        grouped, metadata = build_groups(data, variable)
        grouped_variables[variable] = grouped
        grouping_protocol[variable] = metadata

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, variable in zip(axes.flat, comparison_variables, strict=True):
        _plot_density_panel(
            ax=ax,
            x_values=data[x_feature],
            groups=grouped_variables[variable],
            variable=variable,
            x_label=x_feature,
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "top_feature_relationships.png"
    pdf_path = output_dir / "top_feature_relationships.pdf"
    summary_path = output_dir / "top_feature_relationship_summary.csv"
    protocol_path = output_dir / "relationship_protocol.json"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    _group_summary(data, x_feature, grouped_variables).to_csv(
        summary_path,
        index=False,
    )
    protocol = {
        "training_file": Path(training_path).name,
        "training_rows": int(len(data)),
        "testing_data_used": False,
        "ranking_file": str(Path(ranking_path).name),
        "ranking_column": "Consensus Rank",
        "top_features": top_features,
        "x_feature": x_feature,
        "comparison_variables": comparison_variables,
        "grouping": grouping_protocol,
        "plot": "Gaussian kernel density estimate on a shared x range",
    }
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "png": png_path,
        "pdf": pdf_path,
        "summary": summary_path,
        "protocol": protocol_path,
    }


def main() -> None:
    outputs = run_relationship_analysis()
    for name, path in outputs.items():
        print(f"{name}: {path.resolve()}")


if __name__ == "__main__":
    main()
