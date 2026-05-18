import math
import random
import tempfile
import unittest
from pathlib import Path

from src.experiments.schemas import QAItem
from src.model_sweeps.local_logits_sweep import (
    aggregate_trials_by_item,
    build_scoring_prompt,
    get_label_token_ids,
    label_scores_from_logits,
    load_mock_model,
    run_logits_sweep,
    shuffled_option_view,
    trial_key,
    write_summary_csv,
)
from src.experiments.diagnostic_uncertainty import summarize_diagnostics
from src.model_sweeps.conditioned_metrics import conditioned_summary_rows, normalize_prediction_rows
from src.model_sweeps.uncertainty_metrics import (
    bootstrap_ci,
    expected_calibration_error,
    majority_vote,
    prediction_metrics,
    selective_accuracy,
    softmax_scores,
    summarize_prediction_rows,
    vote_entropy,
    wilson_interval,
)


class UncertaintyMetricTests(unittest.TestCase):
    def test_softmax_and_prediction_metrics(self) -> None:
        probs = softmax_scores({"A": 2.0, "B": 1.0, "C": float("-inf")})

        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertGreater(probs["A"], probs["B"])
        self.assertEqual(probs["C"], 0.0)

        metrics = prediction_metrics(probs, gold="A")

        self.assertEqual(metrics["prediction"], "A")
        self.assertAlmostEqual(metrics["confidence"], probs["A"])
        self.assertAlmostEqual(metrics["margin"], probs["A"] - probs["B"])
        self.assertGreater(metrics["entropy"], 0.0)
        self.assertGreaterEqual(metrics["normalized_entropy"], 0.0)
        self.assertLessEqual(metrics["normalized_entropy"], 1.0)
        self.assertAlmostEqual(metrics["nll"], -math.log(probs["A"]))
        self.assertGreaterEqual(metrics["brier_score"], 0.0)

    def test_wilson_bootstrap_and_calibration_metrics(self) -> None:
        low, high = wilson_interval(8, 10)
        self.assertLess(low, 0.8)
        self.assertGreater(high, 0.8)

        ci_low, ci_high = bootstrap_ci([1, 1, 0, 1], iterations=100, seed=7)
        self.assertLessEqual(ci_low, ci_high)
        self.assertGreaterEqual(ci_low, 0.0)
        self.assertLessEqual(ci_high, 1.0)

        rows = [
            {"confidence": 0.9, "is_correct": True},
            {"confidence": 0.8, "is_correct": True},
            {"confidence": 0.7, "is_correct": False},
        ]
        self.assertGreaterEqual(expected_calibration_error(rows, bins=5), 0.0)
        self.assertEqual(selective_accuracy(rows, 0.5), 1.0)

    def test_vote_helpers(self) -> None:
        label, tied, agreement = majority_vote(["B", "A", "B"])

        self.assertEqual(label, "B")
        self.assertFalse(tied)
        self.assertAlmostEqual(agreement, 2 / 3)
        self.assertEqual(vote_entropy([], all_labels=["A", "B"]), 0.0)
        self.assertGreater(vote_entropy(["A", "B"], all_labels=["A", "B"]), 0.0)

    def test_prediction_summary_accepts_legacy_accuracy_only_rows(self) -> None:
        summary = summarize_prediction_rows([
            {"prediction": "A", "is_correct": True, "error": ""},
            {"prediction": "B", "is_correct": False, "error": ""},
        ])

        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["mean_confidence"], 0.0)
        self.assertEqual(summary["mean_nll"], 0.0)


class LocalLogitsSweepUtilityTests(unittest.TestCase):
    def make_item(self) -> QAItem:
        return QAItem(
            item_id="medQA:test:0",
            dataset="medQA",
            split="test",
            linker="metamap",
            question="Which drug is monitored with aPTT?",
            options={"A": "Warfarin", "B": "Heparin", "C": "Aspirin", "D": "Insulin"},
            gold="B",
            answer_text="Heparin",
        )

    def test_scoring_prompt_is_one_letter_prompt(self) -> None:
        prompt = build_scoring_prompt(self.make_item())

        self.assertIn("Which drug is monitored with aPTT?", prompt)
        self.assertIn("B. Heparin", prompt)
        self.assertTrue(prompt.endswith("Answer:"))
        self.assertIn("Answer with one letter only.", prompt)

    def test_shuffled_option_view_remaps_gold(self) -> None:
        item = self.make_item()
        options, effective_gold, source_labels = shuffled_option_view(
            item.options,
            item.gold,
            rng=random.Random(3),
            shuffle=True,
        )

        self.assertEqual(set(options), {"A", "B", "C", "D"})
        self.assertEqual(set(source_labels.values()), {"A", "B", "C", "D"})
        self.assertEqual(source_labels[effective_gold], "B")

    def test_label_token_ids_follow_dcq_notebook_pattern(self) -> None:
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                mapping = {"A": [1], " A": [2], "B": [3], " B": [4], "CD": [5, 6], " CD": [7, 8]}
                return mapping[text]

        ids = get_label_token_ids(FakeTokenizer(), ["A", "B", "CD"])

        self.assertEqual(ids["A"], [1, 2])
        self.assertEqual(ids["B"], [3, 4])
        self.assertEqual(ids["CD"], [6])

    def test_label_scores_use_best_token_variant(self) -> None:
        class Cell:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        class FakeLogits:
            shape = (1, 6)

            def __init__(self):
                self.values = [0.0, 1.0, 3.0, 2.0, 4.0, 5.0]

            def __getitem__(self, key):
                _, token_id = key
                return Cell(self.values[token_id])

        scores = label_scores_from_logits(FakeLogits(), {"A": [1, 2], "B": [3], "C": [9]})

        self.assertEqual(scores["A"], 3.0)
        self.assertEqual(scores["B"], 2.0)
        self.assertEqual(scores["C"], float("-inf"))

    def test_trial_aggregation_uses_source_labels(self) -> None:
        rows = [
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "MedGemma-1.5-4B-it",
                "model": "google/medgemma-1.5-4b-it",
                "prompt_condition": "no_umls",
                "run_index": 0,
                "shuffle_index": 0,
                "gold": "B",
                "prediction": "A",
                "prediction_source_label": "B",
                "is_correct": True,
                "confidence": 0.7,
                "margin": 0.3,
                "entropy": 0.6,
                "normalized_entropy": 0.4,
                "nll": 0.2,
                "brier_score": 0.1,
                "elapsed_s": 1.0,
                "error": "",
            },
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "MedGemma-1.5-4B-it",
                "model": "google/medgemma-1.5-4b-it",
                "prompt_condition": "no_umls",
                "run_index": 0,
                "shuffle_index": 1,
                "gold": "B",
                "prediction": "C",
                "prediction_source_label": "B",
                "is_correct": True,
                "confidence": 0.8,
                "margin": 0.4,
                "entropy": 0.5,
                "normalized_entropy": 0.3,
                "nll": 0.1,
                "brier_score": 0.05,
                "elapsed_s": 1.5,
                "error": "",
            },
        ]

        aggregates = aggregate_trials_by_item(rows)

        self.assertEqual(len(aggregates), 1)
        self.assertEqual(aggregates[0]["prediction"], "B")
        self.assertTrue(aggregates[0]["is_correct"])
        self.assertEqual(aggregates[0]["num_successful_trials"], 2)
        self.assertEqual(trial_key(rows[0]), "medQA:test:0::no_umls::0::0")

    def test_mock_logits_backend_produces_trial_rows(self) -> None:
        rows = run_logits_sweep(
            [self.make_item()],
            model_bundle=load_mock_model(),
            num_samples=2,
            num_shuffles=2,
            seed=11,
        )

        self.assertEqual(len(rows), 4)
        self.assertEqual({row["model_loader"] for row in rows}, {"mock"})
        self.assertTrue(all(row["prediction"] in {"A", "B", "C", "D"} for row in rows))
        self.assertTrue(all(isinstance(row["option_probabilities"], dict) for row in rows))

    def test_summary_csv_writes_uncertainty_columns(self) -> None:
        rows = [
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "MedGemma-1.5-4B-it",
                "model": "google/medgemma-1.5-4b-it",
                "prompt_condition": "no_umls",
                "run_index": 0,
                "shuffle_index": 0,
                "gold": "A",
                "prediction": "A",
                "prediction_source_label": "A",
                "is_correct": True,
                "confidence": 0.9,
                "margin": 0.8,
                "entropy": 0.1,
                "normalized_entropy": 0.05,
                "nll": 0.1,
                "brier_score": 0.02,
                "elapsed_s": 1.0,
                "error": "",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            write_summary_csv(path, rows)
            text = path.read_text()

        self.assertIn("accuracy_ci_low", text)
        self.assertIn("mean_confidence", text)
        self.assertIn("MedGemma-1.5-4B-it", text)


class DiagnosticUncertaintyUtilityTests(unittest.TestCase):
    def test_diagnostic_uncertainty_summarizes_binary_and_numeric_fields(self) -> None:
        rows = [
            {
                "experiment": "exp2_answer_grounding",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "correct_answer_grounded": True,
                "question_cui_count": 4,
            },
            {
                "experiment": "exp2_answer_grounding",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "correct_answer_grounded": False,
                "question_cui_count": 2,
            },
        ]

        group_keys, summary = summarize_diagnostics(rows, iterations=50, seed=5)
        by_metric = {row["metric"]: row for row in summary}

        self.assertEqual(group_keys, ["experiment", "dataset", "split", "linker"])
        self.assertEqual(by_metric["correct_answer_grounded"]["metric_type"], "binary")
        self.assertEqual(by_metric["correct_answer_grounded"]["mean"], 0.5)
        self.assertEqual(by_metric["question_cui_count"]["metric_type"], "numeric")
        self.assertEqual(by_metric["question_cui_count"]["mean"], 3.0)

    def test_conditioned_metrics_join_predictions_to_diagnostics(self) -> None:
        trial_rows = [
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "MedGemma-1.5-4B-it",
                "model": "google/medgemma-1.5-4b-it",
                "prompt_condition": "no_umls",
                "run_index": 0,
                "shuffle_index": 0,
                "gold": "A",
                "prediction": "A",
                "prediction_source_label": "A",
                "is_correct": True,
                "confidence": 0.9,
                "margin": 0.7,
                "entropy": 0.2,
                "normalized_entropy": 0.1,
                "nll": 0.1,
                "brier_score": 0.02,
                "elapsed_s": 1.0,
                "error": "",
            },
            {
                "item_id": "medQA:test:1",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "MedGemma-1.5-4B-it",
                "model": "google/medgemma-1.5-4b-it",
                "prompt_condition": "no_umls",
                "run_index": 0,
                "shuffle_index": 0,
                "gold": "B",
                "prediction": "A",
                "prediction_source_label": "A",
                "is_correct": False,
                "confidence": 0.8,
                "margin": 0.5,
                "entropy": 0.3,
                "normalized_entropy": 0.2,
                "nll": 2.0,
                "brier_score": 1.2,
                "elapsed_s": 1.0,
                "error": "",
            },
        ]
        predictions = normalize_prediction_rows(trial_rows)
        diagnostics = [
            {"item_id": "medQA:test:0", "correct_answer_grounded": True},
            {"item_id": "medQA:test:1", "correct_answer_grounded": False},
        ]

        summary = conditioned_summary_rows(
            predictions,
            diagnostics,
            conditions=["correct_answer_grounded"],
        )
        by_value = {row["condition_value"]: row for row in summary}

        self.assertEqual(by_value[True]["accuracy"], 1.0)
        self.assertEqual(by_value[False]["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
