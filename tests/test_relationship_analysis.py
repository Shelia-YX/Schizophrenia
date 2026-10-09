import unittest

import pandas as pd

from relationship_analysis import (
    group_hospitalizations,
    select_ranked_features,
)


class RelationshipAnalysisTests(unittest.TestCase):
    def test_hospitalizations_use_requested_three_groups(self) -> None:
        values = pd.Series([0, 1, 5, 6, 10])

        grouped = group_hospitalizations(values)

        self.assertEqual(
            grouped.astype("object").tolist(),
            ["0", "1-5", "1-5", "6+", "6+"],
        )

    def test_selects_features_by_consensus_rank(self) -> None:
        ranking = pd.DataFrame(
            {
                "Feature": ["Third", "First", "Second"],
                "Consensus Rank": [3, 1, 2],
            }
        )

        selected = select_ranked_features(ranking, top_n=2)

        self.assertEqual(selected, ["First", "Second"])


if __name__ == "__main__":
    unittest.main()
