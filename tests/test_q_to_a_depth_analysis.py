import unittest

from src.experiments.q_to_a_depth_analysis import analyse, parse_exact_depth


class QToADepthAnalysisTests(unittest.TestCase):
    def test_parse_exact_depth_condition(self) -> None:
        self.assertEqual(parse_exact_depth("q_to_a_exact_depth_3_evidence"), 3)
        self.assertIsNone(parse_exact_depth("q_to_a_path_evidence"))
        self.assertIsNone(parse_exact_depth("chain_depth_3_width_full_evidence"))

    def test_analyse_groups_exact_depth_rows_by_model_dataset_linker_and_depth(self) -> None:
        rows = [
            {
                "model": "Qwen/Qwen3.5-4B",
                "model_key": "Qwen/Qwen3.5-4B",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "mode": "non_thinking",
                "condition": "q_to_a_exact_depth_1_evidence",
                "prediction": "A",
                "is_correct": True,
                "malformed": False,
                "q_to_a_path_found": True,
                "q_to_a_path_edge_count": 1,
                "q_to_a_path_node_count": 2,
            },
            {
                "model": "Qwen/Qwen3.5-4B",
                "model_key": "Qwen/Qwen3.5-4B",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "mode": "non_thinking",
                "condition": "q_to_a_exact_depth_1_evidence",
                "is_correct": False,
                "prediction": None,
                "malformed": True,
                "q_to_a_path_found": False,
                "q_to_a_path_edge_count": 0,
                "q_to_a_path_node_count": 0,
            },
            {
                "model": "Qwen/Qwen3.5-4B",
                "model_key": "Qwen/Qwen3.5-4B",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "mode": "non_thinking",
                "condition": "q_to_a_exact_depth_2_evidence",
                "prediction": "A",
                "is_correct": True,
                "malformed": False,
                "q_to_a_path_found": True,
                "q_to_a_path_edge_count": 2,
                "q_to_a_path_node_count": 3,
            },
        ]

        summary = analyse(rows)

        by_depth = {row["depth"]: row for row in summary}
        self.assertEqual(by_depth[1]["condition"], "q_to_a_exact_depth_1_evidence")
        self.assertEqual(by_depth[1]["n"], 2)
        self.assertEqual(by_depth[1]["accuracy"], 0.5)
        self.assertEqual(by_depth[1]["strict_accuracy"], 0.5)
        self.assertEqual(by_depth[1]["malformed_rate"], 0.5)
        self.assertEqual(by_depth[1]["path_found_rate"], 0.5)
        self.assertEqual(by_depth[1]["mean_path_edge_count"], 0.5)
        self.assertEqual(by_depth[1]["mean_path_node_count"], 1.0)
        self.assertEqual(by_depth[2]["n"], 1)
        self.assertEqual(by_depth[2]["accuracy"], 1.0)
        self.assertEqual(by_depth[2]["path_found_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
