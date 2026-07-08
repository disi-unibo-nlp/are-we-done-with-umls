import unittest

from src.experiments.graph_distance_mechanism import (
    build_markdown_report,
    classify_distance_signal,
    distance_margin,
    option_distance_row,
    select_stratified_items,
    summarize_prediction_rows,
)
from src.experiments.schemas import QAItem
from src.experiments.umls_index import RelationIndex


class GraphDistanceMechanismTests(unittest.TestCase):
    def test_option_distance_row_compares_raw_and_semantic_graphs(self) -> None:
        item = QAItem(
            item_id="i1",
            dataset="medQA",
            split="test",
            linker="spacy",
            question="Question",
            options={"A": "Alpha", "B": "Beta", "C": "Gamma"},
            gold="A",
            answer_text="Alpha",
            question_cuis=["C_Q"],
            option_cuis={"A": ["C_A"], "B": ["C_B"], "C": ["C_C"]},
        )
        raw = RelationIndex(
            {
                "C_Q": {"C_ADMIN", "C_SEM_B", "C_SEM_C"},
                "C_ADMIN": {"C_Q", "C_A"},
                "C_A": {"C_ADMIN"},
                "C_SEM_B": {"C_Q", "C_B"},
                "C_B": {"C_SEM_B"},
                "C_SEM_C": {"C_Q", "C_C"},
                "C_C": {"C_SEM_C"},
            },
            edge_labels={
                ("C_ADMIN", "C_Q"): ("AQ", ""),
                ("C_A", "C_ADMIN"): ("QB", ""),
                ("C_Q", "C_SEM_B"): ("RO", "clinically_associated_with"),
                ("C_B", "C_SEM_B"): ("RO", "may_treat"),
                ("C_Q", "C_SEM_C"): ("RO", "clinically_associated_with"),
                ("C_C", "C_SEM_C"): ("RO", "may_treat"),
            },
        )
        semantic = raw.filter_edges(lambda label: label in {"AQ", "QB"})

        row = option_distance_row(item, raw, semantic, max_hops=4)

        self.assertEqual(row["raw_distances"], {"A": 2, "B": 2, "C": 2})
        self.assertEqual(row["semantic_distances"], {"A": None, "B": 2, "C": 2})
        self.assertEqual(row["raw_signal"], "gold_tied_closest")
        self.assertEqual(row["semantic_signal"], "gold_missing")
        self.assertIsNone(row["semantic_margin"])

    def test_distance_margin_and_signal_require_complete_option_distances(self) -> None:
        self.assertEqual(distance_margin({"A": 1, "B": 3, "C": 2}, "A"), 1)
        self.assertEqual(classify_distance_signal({"A": 1, "B": 3, "C": 2}, "A"), "gold_unique_closest")
        self.assertEqual(classify_distance_signal({"A": 2, "B": 2, "C": 3}, "A"), "gold_tied_closest")
        self.assertEqual(classify_distance_signal({"A": 3, "B": 2, "C": 4}, "A"), "gold_not_closest")
        self.assertEqual(classify_distance_signal({"A": 1, "B": None, "C": 4}, "A"), "partial")
        self.assertIsNone(distance_margin({"A": 1, "B": None, "C": 4}, "A"))

    def test_select_stratified_items_is_exclusive_and_deterministic(self) -> None:
        rows = [
            {"item_id": "sg1", "semantic_margin": 2, "raw_margin": 2},
            {"item_id": "sg2", "semantic_margin": 1, "raw_margin": 0},
            {"item_id": "sd1", "semantic_margin": -1, "raw_margin": 0},
            {"item_id": "sd2", "semantic_margin": -2, "raw_margin": -1},
            {"item_id": "ro1", "semantic_margin": 0, "raw_margin": 1},
            {"item_id": "ro2", "semantic_margin": -1, "raw_margin": 2},
            {"item_id": "tc1", "semantic_margin": 0, "raw_margin": 0},
            {"item_id": "tc2", "semantic_margin": 0, "raw_margin": 0},
        ]

        selected = select_stratified_items(rows, max_per_stratum=1, min_size=1, seed=7)
        repeated = select_stratified_items(rows, max_per_stratum=1, min_size=1, seed=7)

        self.assertEqual(selected, repeated)
        self.assertEqual(set(selected), {
            "semantic_gold_closer",
            "semantic_distractor_closer",
            "raw_only_gold_closer",
            "tie_control",
        })
        flat = [item_id for item_ids in selected.values() for item_id in item_ids]
        self.assertEqual(len(flat), len(set(flat)))

    def test_summarize_predictions_and_report_join_cache_strata(self) -> None:
        cache_rows = [
            {"item_id": "i1", "stratum": "semantic_gold_closer", "gold_closest_option": "A"},
            {"item_id": "i2", "stratum": "semantic_gold_closer", "gold_closest_option": "A"},
            {"item_id": "i3", "stratum": "semantic_distractor_closer", "gold_closest_option": "B"},
        ]
        prediction_rows = [
            {"item_id": "i1", "condition": "q_to_a_semantic_shortest_path_evidence", "model": "Qwen-9B", "prediction": "A", "is_correct": True, "malformed": False},
            {"item_id": "i2", "condition": "q_to_a_semantic_shortest_path_evidence", "model": "Qwen-9B", "prediction": "B", "is_correct": False, "malformed": False},
            {"item_id": "i3", "condition": "q_to_a_topology_only_path_evidence", "model": "Qwen-9B", "prediction": "B", "is_correct": False, "malformed": False},
        ]

        summary = summarize_prediction_rows(prediction_rows, cache_rows)
        report = build_markdown_report(summary, audit_rows=cache_rows)

        semantic_row = next(row for row in summary if row["condition"] == "q_to_a_semantic_shortest_path_evidence")
        topology_row = next(row for row in summary if row["condition"] == "q_to_a_topology_only_path_evidence")
        self.assertEqual(semantic_row["stratum"], "semantic_gold_closer")
        self.assertAlmostEqual(semantic_row["accuracy"], 0.5)
        self.assertAlmostEqual(topology_row["closest_option_attraction_rate"], 1.0)
        self.assertIn("graph distance acts as a salience prior", report)
        self.assertIn("| Qwen-9B | semantic_gold_closer | q_to_a_semantic_shortest_path_evidence | 2 | 0.500", report)


if __name__ == "__main__":
    unittest.main()
