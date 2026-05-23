import unittest

from src.experiments.depth_width_analysis import analyse, build_structural_index


class DepthWidthAnalysisTests(unittest.TestCase):
    def test_structural_index_is_scoped_by_dataset_split_linker_and_item(self) -> None:
        exp3_rows = [
            {
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "item_id": "shared",
                "condition": "q_to_a_2hop",
                "gold_min_path_length": 2,
                "num_retrieved_nodes": 25,
            },
            {
                "dataset": "medMCQA",
                "split": "test",
                "linker": "spacy",
                "item_id": "shared",
                "condition": "q_to_a_2hop",
                "gold_min_path_length": None,
                "num_retrieved_nodes": 120,
            },
        ]
        structural_index = build_structural_index(exp3_rows)
        pred_rows = [
            {
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "item_id": "shared",
                "condition": "chain_depth_2_width_medium_evidence",
                "is_correct": True,
            },
            {
                "dataset": "medMCQA",
                "split": "test",
                "linker": "spacy",
                "item_id": "shared",
                "condition": "chain_depth_2_width_medium_evidence",
                "is_correct": False,
            },
        ]

        result = analyse(pred_rows, structural_index, "chain_depth_2_width_medium_evidence")

        self.assertEqual(result["depth"]["2"], "1.000 (n=1)")
        self.assertEqual(result["depth"]["no_path"], "0.000 (n=1)")
        self.assertEqual(result["width"]["medium"], "1.000 (n=1)")
        self.assertEqual(result["width"]["full"], "0.000 (n=1)")


if __name__ == "__main__":
    unittest.main()
