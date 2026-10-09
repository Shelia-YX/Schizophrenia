import unittest

import pandas as pd

from full.full import select_best_by_cv


class FullSelectionTests(unittest.TestCase):
    def test_appends_best_scaled_perceptron_to_model_winners(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "Model": "Logistic Regression",
                    "Config ID": "logistic_001",
                    "Scale Numeric": True,
                    "CV ROC-AUC Mean": 0.95,
                    "CV F1 Mean": 0.90,
                    "CV Accuracy Mean": 0.91,
                    "CV Fit Time Mean (s)": 0.03,
                },
                {
                    "Model": "Perceptron",
                    "Config ID": "perceptron_010",
                    "Scale Numeric": False,
                    "CV ROC-AUC Mean": 1.0,
                    "CV F1 Mean": 1.0,
                    "CV Accuracy Mean": 1.0,
                    "CV Fit Time Mean (s)": 0.01,
                },
                {
                    "Model": "Perceptron",
                    "Config ID": "perceptron_019",
                    "Scale Numeric": True,
                    "CV ROC-AUC Mean": 1.0,
                    "CV F1 Mean": 0.99,
                    "CV Accuracy Mean": 0.995,
                    "CV Fit Time Mean (s)": 0.03,
                },
                {
                    "Model": "Perceptron",
                    "Config ID": "perceptron_015",
                    "Scale Numeric": True,
                    "CV ROC-AUC Mean": 1.0,
                    "CV F1 Mean": 0.99,
                    "CV Accuracy Mean": 0.995,
                    "CV Fit Time Mean (s)": 0.02,
                },
            ]
        )

        selected = select_best_by_cv(results)

        self.assertEqual(
            selected["Config ID"].tolist(),
            ["logistic_001", "perceptron_010", "perceptron_015"],
        )
        scaled_perceptron = selected.iloc[-1]
        self.assertEqual(scaled_perceptron["Model"], "Perceptron")
        self.assertTrue(bool(scaled_perceptron["Scale Numeric"]))


if __name__ == "__main__":
    unittest.main()
