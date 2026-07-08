import unittest

from src.experiments.step3_mechanism_analysis import (
    build_markdown_report,
    paired_depth_comparisons,
    relation_label_counts,
    summarize_prediction_rows,
    wilson_interval,
)


class Step3MechanismAnalysisTests(unittest.TestCase):
    def test_wilson_interval_is_bounded_and_centered_on_observed_rate(self):
        low, high = wilson_interval(5, 10)

        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

    def test_summarize_prediction_rows_groups_depth_conditions(self):
        rows = [
            {
                "model": "Qwen-0.8B",
                "condition": "q_to_a_d4_shortcut_1_evidence",
                "item_id": "i1",
                "gold": "A",
                "prediction": "A",
                "is_correct": True,
                "malformed": False,
            },
            {
                "model": "Qwen-0.8B",
                "condition": "q_to_a_d4_shortcut_1_evidence",
                "item_id": "i2",
                "gold": "A",
                "prediction": "B",
                "is_correct": False,
                "malformed": False,
            },
            {
                "model": "Qwen-0.8B",
                "condition": "q_to_a_d4_shortcut_1_evidence",
                "item_id": "i3",
                "gold": "A",
                "prediction": "",
                "is_correct": False,
                "malformed": True,
            },
        ]

        summary = summarize_prediction_rows(rows, "q_to_a_d4_shortcut_")

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["model"], "Qwen-0.8B")
        self.assertEqual(summary[0]["depth"], 1)
        self.assertEqual(summary[0]["n"], 3)
        self.assertEqual(summary[0]["valid"], 2)
        self.assertEqual(summary[0]["malformed"], 1)
        self.assertAlmostEqual(summary[0]["accuracy"], 0.5)

    def test_paired_depth_comparisons_use_common_items(self):
        rows = [
            {
                "model": "Qwen-2B",
                "condition": "q_to_a_d4_shortcut_1_evidence",
                "item_id": "i1",
                "is_correct": True,
                "malformed": False,
            },
            {
                "model": "Qwen-2B",
                "condition": "q_to_a_d4_shortcut_4_evidence",
                "item_id": "i1",
                "is_correct": False,
                "malformed": False,
            },
            {
                "model": "Qwen-2B",
                "condition": "q_to_a_d4_shortcut_1_evidence",
                "item_id": "i2",
                "is_correct": True,
                "malformed": False,
            },
            {
                "model": "Qwen-2B",
                "condition": "q_to_a_d4_shortcut_4_evidence",
                "item_id": "i2",
                "is_correct": True,
                "malformed": False,
            },
            {
                "model": "Qwen-2B",
                "condition": "q_to_a_d4_shortcut_4_evidence",
                "item_id": "only-reference",
                "is_correct": True,
                "malformed": False,
            },
        ]

        comparisons = paired_depth_comparisons(
            rows,
            "q_to_a_d4_shortcut_",
            depths=(1,),
            reference_depth=4,
        )

        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons[0]["common_n"], 2)
        self.assertAlmostEqual(comparisons[0]["accuracy"], 1.0)
        self.assertAlmostEqual(comparisons[0]["reference_accuracy"], 0.5)
        self.assertEqual(comparisons[0]["discordant_b"], 1)
        self.assertEqual(comparisons[0]["discordant_c"], 0)

    def test_relation_label_counts_parse_evidence_text(self):
        cache_rows = [
            {
                "condition": "q_to_a_exact_depth_3_evidence",
                "evidence_text": (
                    "UMLS evidence:\n"
                    "- Proteinuria -- has allowed qualifier --> Severe\n"
                    "- Severe -- concept in subset --> Finding\n"
                    "- Finding -- related to --> Diagnosis\n"
                ),
            }
        ]

        counts = relation_label_counts(cache_rows, "q_to_a_exact_depth_")

        self.assertEqual(counts[3]["has allowed qualifier"], 1)
        self.assertEqual(counts[3]["concept in subset"], 1)
        self.assertEqual(counts[3]["related to"], 1)

    def test_report_renders_mechanism_claim(self):
        report = build_markdown_report(
            faithful_summary=[
                {
                    "model": "Qwen-9B",
                    "condition": "q_to_a_exact_depth_4_evidence",
                    "depth": 4,
                    "n": 100,
                    "valid": 100,
                    "malformed": 0,
                    "correct": 67,
                    "accuracy": 0.67,
                    "ci_low": 0.57,
                    "ci_high": 0.75,
                }
            ],
            shortcut_summary=[],
            shortcut_pairwise=[],
            relation_counts={4: {"has allowed qualifier": 20}},
            coverage_rows=[{"source": "candidate", "count": 301}],
        )

        self.assertIn("UMLS graph distance is not semantic reasoning depth", report)
        self.assertIn("| Qwen-9B | 4 | 100 | 0.670", report)
        self.assertIn("has allowed qualifier", report)


if __name__ == "__main__":
    unittest.main()
