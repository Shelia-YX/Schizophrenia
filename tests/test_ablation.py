from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing_and_cv import build_preprocessor


ABLATION_FEATURES = [
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
    "Social Support",
    "Stress Factors",
    "Medication Adherence",
]


def _small_ablated_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age": [20, 40, 60, 30],
            "Gender": [0, 1, 0, 1],
            "Education Level": [1, 2, 3, 4],
            "Marital Status": [0, 1, 2, 3],
            "Occupation": [0, 1, 2, 3],
            "Income Level": [0, 1, 2, 0],
            "Living Area": [0, 1, 0, 1],
            "Hospitalizations": [0, 1, 2, 3],
            "Family History": [0, 1, 0, 1],
            "Substance Use": [0, 1, 0, 1],
            "Suicide Attempt": [0, 1, 0, 1],
            "Social Support": [0, 1, 2, 0],
            "Stress Factors": [0, 1, 2, 0],
            "Medication Adherence": [0, 1, 2, 0],
        }
    )


class PreprocessorSelectionTests(unittest.TestCase):
    def test_preprocessor_uses_only_selected_predictors(self) -> None:
        frame = _small_ablated_frame()

        preprocessor = build_preprocessor(feature_columns=ABLATION_FEATURES)
        transformed = preprocessor.fit_transform(frame)
        encoded_names = preprocessor.get_feature_names_out().tolist()

        self.assertEqual(transformed.shape[1], len(encoded_names))
        self.assertNotIn("Positive Symptom Score", encoded_names)
        self.assertNotIn("Negative Symptom Score", encoded_names)
        self.assertNotIn("GAF Score", encoded_names)

    def test_preprocessor_rejects_unknown_predictor(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown feature"):
            build_preprocessor(feature_columns=["Age", "Unknown feature"])


class AblationRunnerContractTests(unittest.TestCase):
    def test_ablation_uses_shared_pipeline_module(self) -> None:
        from ablation import run_ablation
        from full import full

        for module in (run_ablation, full):
            self.assertEqual(
                module.build_configured_pipeline.__module__,
                "preprocessing_and_cv",
            )
            self.assertEqual(
                module.evaluate_test_set.__module__,
                "preprocessing_and_cv",
            )

    def test_phase1_configurations_are_frozen(self) -> None:
        from ablation.run_ablation import PHASE1_CONFIGS

        expected_ids = {
            "logistic": "logistic_002",
            "knn": "knn_002",
            "perceptron": "perceptron_002",
            "mlp": "mlp_016",
            "random_forest": "random_forest_001",
        }
        self.assertEqual(
            {key: config["config_id"] for key, config in PHASE1_CONFIGS.items()},
            expected_ids,
        )
        self.assertTrue(
            all(not config["scale_numeric"] for config in PHASE1_CONFIGS.values())
        )

    def test_shared_splits_are_repeatable(self) -> None:
        from ablation.run_ablation import make_shared_splits

        labels = np.array([0] * 10 + [1] * 10)
        first = make_shared_splits(labels, cv_splits=5, random_state=42)
        second = make_shared_splits(labels, cv_splits=5, random_state=42)

        self.assertEqual(len(first), 5)
        for (first_train, first_validation), (second_train, second_validation) in zip(
            first,
            second,
            strict=True,
        ):
            np.testing.assert_array_equal(first_train, second_train)
            np.testing.assert_array_equal(first_validation, second_validation)

    def test_runner_writes_paired_metric_outputs(self) -> None:
        from ablation.run_ablation import run_ablation

        source = pd.read_csv("training_data.csv", header=None)
        small = pd.concat(
            [
                source[source.iloc[:, -1] == -1].head(30),
                source[source.iloc[:, -1] == 1].head(30),
            ],
            ignore_index=True,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            training_path = temporary_path / "training.csv"
            testing_path = temporary_path / "testing.csv"
            output_directory = temporary_path / "results"
            small.to_csv(training_path, index=False, header=False)
            pd.concat([small.iloc[:10], small.iloc[30:40]]).to_csv(
                testing_path,
                index=False,
                header=False,
            )

            outputs = run_ablation(
                training_path=training_path,
                testing_path=testing_path,
                output_dir=output_directory,
                model_keys=("logistic",),
                cv_splits=5,
                random_state=42,
            )

            fold_results = outputs["fold_results"]
            self.assertEqual(len(fold_results), 10)
            self.assertEqual(
                set(fold_results["Feature Set"]),
                {"full", "ablated"},
            )
            self.assertEqual(
                set(fold_results.columns)
                & {
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "Specificity",
                    "F1",
                    "ROC-AUC",
                },
                {
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "Specificity",
                    "F1",
                    "ROC-AUC",
                },
            )
            self.assertEqual(len(outputs["cv_summary"]), 2)
            self.assertEqual(len(outputs["paired_differences"]), 1)
            self.assertEqual(len(outputs["test_results"]), 2)
            self.assertEqual(
                set(outputs["test_results"]["Evaluation Status"]),
                {"exploratory"},
            )
            protocol = json.loads(
                (output_directory / "ablation_protocol.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                {path.name for path in output_directory.iterdir()},
                {
                    "ablation_fold_results.csv",
                    "ablation_cv_summary.csv",
                    "ablation_cv_paired_differences.csv",
                    "ablation_test_results.csv",
                    "ablation_protocol.json",
                    "ablation_interpretation.md",
                },
            )
            self.assertEqual(protocol["dropped_features"], [
                "Negative Symptom Score",
                "GAF Score",
                "Positive Symptom Score",
            ])
            self.assertEqual(protocol["cv_splits"], 5)
            self.assertEqual(protocol["random_state"], 42)
            self.assertTrue(protocol["same_cv_indices_for_full_and_ablated"])
            self.assertFalse(protocol["test_evaluation"]["used_for_ablation_selection"])
            self.assertEqual(
                protocol["test_evaluation"]["status"],
                "exploratory; Phase 1 evaluated this same test set for all 146 candidate configurations",
            )
            summary_columns = set(outputs["cv_summary"].columns)
            for metric in (
                "Accuracy",
                "Precision",
                "Recall",
                "Specificity",
                "F1",
                "ROC-AUC",
            ):
                self.assertIn(f"CV {metric} Mean", summary_columns)
                self.assertIn(f"CV {metric} Std", summary_columns)
            paired_columns = set(outputs["paired_differences"].columns)
            self.assertIn(
                "Delta Accuracy Mean (Ablated - Full)",
                paired_columns,
            )
            full_test = outputs["test_results"].query(
                "`Feature Set` == 'full'"
            ).iloc[0]
            self.assertEqual(int(full_test["Test TN"]), 10)
            self.assertEqual(int(full_test["Test FP"]), 0)
            self.assertEqual(int(full_test["Test FN"]), 0)
            self.assertEqual(int(full_test["Test TP"]), 10)
            self.assertAlmostEqual(float(full_test["Test Specificity"]), 1.0)
            self.assertEqual(
                set(fold_results[fold_results["Feature Set"] == "full"]["Fold"]),
                set(
                    fold_results[fold_results["Feature Set"] == "ablated"]["Fold"]
                ),
            )
