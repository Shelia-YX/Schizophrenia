"""Plot distributions of every variable in the SchizoHealth dataset.

Categorical variables are shown as percentage bar charts. Numerical variables
are shown as histograms. The script supports the 18-column, headerless CSV used
in the manuscript and also accepts the same columns with an English header.

Usage:
    python plot_dataset_distributions.py schizophrenia_dataset.csv
    python plot_dataset_distributions.py schizophrenia_dataset.csv --individual
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Rectangle


# The order of columns in the uploaded 18-column CSV.
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

# Arrange panels in the same a-r order as the current Figure 1.
PLOT_ORDER = [
    "Age",
    "Gender",
    "Occupation",
    "Education Level",
    "Marital Status",
    "Hospitalizations",
    "Income Level",
    "Living Area",
    "Family History",
    "Substance Use",
    "Medication Adherence",
    "Suicide Attempt",
    "Social Support",
    "Stress Factors",
    "GAF Score",
    "Diagnosis",
    "Positive Symptom Score",
    "Negative Symptom Score",
]

CATEGORICAL_VARIABLES = [
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
    "Diagnosis",
]

NUMERICAL_VARIABLES = [
    "Age",
    "Hospitalizations",
    "Positive Symptom Score",
    "Negative Symptom Score",
    "GAF Score",
]

# Edit these labels if your original codebook used different category names.
# The numerical ordering is retained even if a mapping is absent.
CATEGORY_LABELS = {
    "Gender": {0: "Female", 1: "Male"},
    "Education Level": {
        1: "Primary",
        2: "Middle school",
        3: "High school",
        4: "University",
        5: "Postgraduate",
    },
    "Marital Status": {0: "Single", 1: "Married", 2: "Divorced", 3: "Widowed"},
    "Occupation": {0: "Unemployed", 1: "Employed", 2: "Student", 3: "Retired"},
    "Income Level": {0: "Low", 1: "Medium", 2: "High"},
    "Living Area": {0: "Rural", 1: "Urban"},
    "Family History": {0: "No", 1: "Yes"},
    "Substance Use": {0: "No", 1: "Yes"},
    "Suicide Attempt": {0: "No", 1: "Yes"},
    "Social Support": {0: "Low", 1: "Medium", 2: "High"},
    "Stress Factors": {0: "Low", 1: "Medium", 2: "High"},
    "Medication Adherence": {0: "Poor", 1: "Moderate", 2: "Good"},
    # The uploaded CSV uses {-1, 1}; 0 is included for compatibility with
    # versions encoded as {0, 1}.
    "Diagnosis": {-1: "Not schizophrenic", 0: "Not schizophrenic", 1: "Schizophrenic"},
}

# Muted palettes adapted from the supplied reference figure. Categorical
# levels run from plum to peach; numerical bins run from navy to pale green.
CATEGORICAL_PALETTE = ["#5B4B6F", "#A85F79", "#D27A78", "#E8A27F", "#E8C8BB"]
NUMERICAL_PALETTE = ["#243B73", "#22918C", "#8BD3CF", "#DDECCF"]
GRID_COLOR = "#E3E3E3"


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the current headerless file without losing its first observation."""
    raw = pd.read_csv(csv_path, header=None)

    if raw.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"Expected {len(COLUMN_NAMES)} columns, but found {raw.shape[1]}. "
            "Check that this is the processed 18-column dataset."
        )

    # If a real header exists, pandas read it as the first data row because we
    # deliberately used header=None. Detect and remove that row.
    first_row = {str(value).strip().lower() for value in raw.iloc[0]}
    if "age" in first_row and "diagnosis" in first_row:
        raw = raw.iloc[1:].reset_index(drop=True)

    raw.columns = COLUMN_NAMES

    # Every field in the supplied file is numerically encoded. Invalid cells
    # become NaN and are omitted only from the affected plot.
    for column in COLUMN_NAMES:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    return raw


def ordered_category_counts(series: pd.Series) -> pd.Series:
    counts = series.dropna().value_counts().sort_index()
    return counts


def category_tick_labels(variable: str, values: list[float]) -> list[str]:
    mapping = CATEGORY_LABELS.get(variable, {})
    labels = []
    for value in values:
        integer_value = int(value) if float(value).is_integer() else value
        labels.append(mapping.get(integer_value, str(integer_value)))
    return labels


def plot_categorical(ax: plt.Axes, data: pd.Series, variable: str) -> None:
    counts = ordered_category_counts(data)
    percentages = counts / counts.sum() * 100
    x = np.arange(len(counts))

    colors = [CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i in range(len(counts))]
    bars = ax.bar(x, percentages.to_numpy(), color=colors, width=0.76)
    ax.set_xticks(x)
    ax.set_xticklabels(
        category_tick_labels(variable, counts.index.to_list()),
        rotation= 0,
        ha= "center",
    )
    ax.set_ylabel("Percentage")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylim(0, max(10, float(percentages.max()) * 1.20))

    for bar, value in zip(bars, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )


def histogram_bins(data: pd.Series, variable: str):
    clean = data.dropna()
    if variable == "Hospitalizations":
        return np.arange(clean.min() - 0.5, clean.max() + 1.5, 1)
    if variable == "Age":
        return np.arange(17.5, 82.5, 5)
    if variable in {"Positive Symptom Score", "Negative Symptom Score"}:
        return np.arange(-0.5, 110.5, 10)
    if variable == "GAF Score":
        return np.arange(9.5, 110.5, 10)
    return "auto"


def plot_numerical(ax: plt.Axes, data: pd.Series, variable: str) -> None:
    clean = data.dropna()
    counts, _, patches = ax.hist(
        clean,
        bins=histogram_bins(clean, variable),
        edgecolor="white",
        linewidth=0.45,
    )
    cmap = LinearSegmentedColormap.from_list("reference_numerical", NUMERICAL_PALETTE)
    denominator = max(1, len(patches) - 1)
    for index, patch in enumerate(patches):
        patch.set_facecolor(cmap(index / denominator))

    ax.bar_label(
        patches,
        labels=[
            f"{int(count)}" if count > 0 else ""
            for count in counts
        ],
        padding=2,
        fontsize=5.5,
        rotation=0,
    )
    ax.set_ylabel("Frequency")
    ax.set_ylim(0, max(counts) * 1.15)
    if variable == "Hospitalizations":
        ax.set_xticks(range(int(clean.min()), int(clean.max()) + 1))



def finish_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=9, pad=4)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.55, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9A9A9A")
    ax.spines["bottom"].set_color("#9A9A9A")
    ax.tick_params(axis="both", labelsize=7, length=2.5)
    ax.yaxis.label.set_size(7.5)


def draw_variable(ax: plt.Axes, df: pd.DataFrame, variable: str, panel: str | None = None) -> None:
    if variable in CATEGORICAL_VARIABLES:
        plot_categorical(ax, df[variable], variable)
    elif variable in NUMERICAL_VARIABLES:
        plot_numerical(ax, df[variable], variable)
    else:
        raise KeyError(f"No plot type configured for {variable!r}")

    title = f"({panel}) {variable}" if panel else variable
    finish_axis(ax, title)


def combined_grid_shape(panel_count: int, ncols: int = 3) -> tuple[int, int]:
    """Return the compact grid shape used by the manuscript figure."""
    return math.ceil(panel_count / ncols), ncols


def panel_grid_positions(panel_count: int, ncols: int = 3) -> list[tuple[int, int]]:
    """Place an incomplete final row in the horizontal center of the grid."""
    full_rows, remainder = divmod(panel_count, ncols)
    positions = [(row, col) for row in range(full_rows) for col in range(ncols)]
    if remainder:
        start_column = (ncols - remainder) // 2
        positions.extend((full_rows, start_column + offset) for offset in range(remainder))
    return positions


def save_combined_figure(df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    nrows, ncols = combined_grid_shape(len(PLOT_ORDER))
    positions = panel_grid_positions(len(PLOT_ORDER), ncols)
    fig = plt.figure(figsize=(14, 13))
    grid = fig.add_gridspec(nrows, ncols)
    axes = []

    for index, (variable, (row, col)) in enumerate(zip(PLOT_ORDER, positions)):
        # 这是固定大小的外框面板
        panel_ax = fig.add_subplot(grid[row, col])
        panel_ax.set_xticks([])
        panel_ax.set_yticks([])
        panel_ax.patch.set_alpha(0)

        for spine in panel_ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#999999")
            spine.set_linewidth(0.05)

        # 真正画图的区域，留出标题、坐标轴标题和刻度标签的空间
        ax = panel_ax.inset_axes([0.15, 0.15, 0.80, 0.75])

        draw_variable(ax, df, variable, chr(ord("a") + index))

    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.045, top=0.965, wspace=0.03, hspace=0.05)
    fig.savefig(output_dir / "figure_1.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "figure_1.pdf", bbox_inches="tight")
    plt.close(fig)


def save_individual_figures(df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    individual_dir = output_dir / "individual_plots"
    individual_dir.mkdir(parents=True, exist_ok=True)

    for variable in PLOT_ORDER:
        fig, ax = plt.subplots(figsize=(5.2, 4.1))
        draw_variable(ax, df, variable)
        fig.tight_layout()
        safe_name = variable.lower().replace(" ", "_")
        fig.savefig(individual_dir / f"{safe_name}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to schizophrenia_dataset.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("distribution_plots"),
        help="Directory for generated figures (default: distribution_plots)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300)")
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Also save one PNG file for each variable",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.csv)
    save_combined_figure(df, args.output_dir, args.dpi)
    if args.individual:
        save_individual_figures(df, args.output_dir, args.dpi)

    print(f"Loaded {len(df):,} observations and {df.shape[1]} variables.")
    print(f"Figures saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
