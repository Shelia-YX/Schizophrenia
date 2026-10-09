import unittest

import pandas as pd

from importance import summarize_fold_importance


class ImportanceRankingTests(unittest.TestCase):
    def test_equal_mean_importance_receives_equal_average_rank(self) -> None:
        fold_rows = pd.DataFrame(
            [
                {"Fold": 1, "Feature": "Strong", "Importance": 2.0},
                {"Fold": 1, "Feature": "Zero A", "Importance": 0.0},
                {"Fold": 1, "Feature": "Zero B", "Importance": 0.0},
                {"Fold": 2, "Feature": "Strong", "Importance": 2.0},
                {"Fold": 2, "Feature": "Zero A", "Importance": 0.0},
                {"Fold": 2, "Feature": "Zero B", "Importance": 0.0},
            ]
        )

        summary = summarize_fold_importance(fold_rows).set_index("Feature")

        self.assertEqual(summary.loc["Strong", "Overall Rank"], 1.0)
        self.assertEqual(summary.loc["Zero A", "Overall Rank"], 2.5)
        self.assertEqual(summary.loc["Zero B", "Overall Rank"], 2.5)


if __name__ == "__main__":
    unittest.main()
