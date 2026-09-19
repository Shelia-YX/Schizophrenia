"""Split a headerless CSV dataset into stratified training and testing files.

The final column is treated as the target label (Diagnosis). The output files
remain headerless and contain the same columns as the input file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def split_csv(
    input_path: str | Path,
    train_path: str | Path,
    test_path: str | Path,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict[str, Any]:
    """Read, validate, stratify, split, and save a headerless CSV file."""
    input_path = Path(input_path)
    train_path = Path(train_path)
    test_path = Path(test_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV file not found: {input_path}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    data = pd.read_csv(input_path, header=None)
    if data.empty:
        raise ValueError("The input CSV file is empty.")
    if data.shape[1] < 2:
        raise ValueError("The dataset must contain at least one feature and one target column.")

    target = data.iloc[:, -1]
    if target.isna().any():
        raise ValueError("The target column contains missing values.")
    if target.nunique() < 2:
        raise ValueError("The target column must contain at least two classes.")
    if target.value_counts().min() < 2:
        raise ValueError("Each target class must contain at least two observations.")

    try:
        training_data, testing_data = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            stratify=target,
            shuffle=True,
        )
    except ValueError as exc:
        raise ValueError(f"Unable to create a stratified split: {exc}") from exc

    training_data = training_data.reset_index(drop=True)
    testing_data = testing_data.reset_index(drop=True)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    training_data.to_csv(train_path, index=False, header=False)
    testing_data.to_csv(test_path, index=False, header=False)

    return {
        "total_rows": len(data),
        "training_rows": len(training_data),
        "testing_rows": len(testing_data),
        "training_class_counts": training_data.iloc[:, -1].value_counts().to_dict(),
        "testing_class_counts": testing_data.iloc[:, -1].value_counts().to_dict(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a headerless CSV into stratified training and testing CSV files. "
            "The final column is used as the target label."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Path to the original CSV file.")
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("training_data.csv"),
        help="Training CSV output path (default: training_data.csv).",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path("testing_data.csv"),
        help="Testing CSV output path (default: testing_data.csv).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction assigned to the testing set (default: 0.20).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for reproducibility (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = split_csv(
        input_path=args.input_csv,
        train_path=args.train_output,
        test_path=args.test_output,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"Total rows: {summary['total_rows']}")
    print(f"Training rows: {summary['training_rows']}")
    print(f"Testing rows: {summary['testing_rows']}")
    print(f"Training class counts: {summary['training_class_counts']}")
    print(f"Testing class counts: {summary['testing_class_counts']}")
    print(f"Training file: {args.train_output.resolve()}")
    print(f"Testing file: {args.test_output.resolve()}")


if __name__ == "__main__":
    main()
