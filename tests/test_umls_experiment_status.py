import tempfile
import unittest
from pathlib import Path

from src.experiments import umls_experiment_status as status


class UmlsExperimentStatusTests(unittest.TestCase):
    def test_model_contract_separates_fixed_step1_from_downstream_sweep(self) -> None:
        self.assertEqual(
            status.STEP1_MODELS,
            (
                "Qwen/Qwen3.5-0.8B",
                "Qwen/Qwen3.5-2B",
                "Qwen/Qwen3.5-4B",
                "Qwen/Qwen3.5-9B",
            ),
        )
        self.assertEqual(
            status.SCALE_MODELS,
            (
                "Qwen/Qwen3.5-0.8B",
                "Qwen/Qwen3.5-2B",
                "Qwen/Qwen3.5-4B",
                "Qwen/Qwen3.5-9B",
                "Qwen/Qwen3.5-27B",
                "Qwen/Qwen3.5-35B",
            ),
        )
        self.assertEqual(status.STEP1_SUBSET_FALLBACK_MODEL, "Qwen/Qwen3.5-9B")

    def test_step23_status_contract_uses_spacy_only_by_default(self) -> None:
        self.assertEqual(status.LINKERS, ("metamap", "spacy"))
        self.assertEqual(status.STEP23_LINKERS, ("spacy",))
        self.assertEqual(len(status.expected_cells()), 24)
        self.assertEqual(len(status.expected_cells(models=status.STEP1_MODELS)), 16)
        self.assertEqual(len(status.expected_cells(status.STEP23_LINKERS, status.SCALE_MODELS)), 12)

    def test_expected_step3_conditions_are_exact_q_to_a_depths(self) -> None:
        self.assertEqual(
            status.STEP3_CONDITIONS,
            (
                "q_to_a_exact_depth_1_evidence",
                "q_to_a_exact_depth_2_evidence",
                "q_to_a_exact_depth_3_evidence",
                "q_to_a_exact_depth_4_evidence",
                "q_to_a_exact_depth_5_evidence",
            ),
        )
        self.assertNotIn("chain_depth_1_width_tight_evidence", status.STEP3_CONDITIONS)

    def test_recovery_commands_start_with_step1_when_predictions_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_root = status.OUTPUT_ROOT
            old_step1 = status.STEP1_PREDICTIONS_DIR
            old_step2 = status.STEP2_PREDICTIONS_DIR
            old_step3 = status.STEP3_PREDICTIONS_DIR
            old_subsets = status.SUBSETS_DIR
            old_tables = status.TABLES_DIR
            root = Path(tmp)
            status.OUTPUT_ROOT = root
            status.STEP1_PREDICTIONS_DIR = root / "step1_predictions" / "qwen"
            status.STEP2_PREDICTIONS_DIR = root / "step2_predictions" / "qwen"
            status.STEP3_PREDICTIONS_DIR = root / "step3_predictions" / "qwen"
            status.SUBSETS_DIR = root / "step1_subsets"
            status.TABLES_DIR = root / "tables"
            try:
                built = status.build_status()
            finally:
                status.OUTPUT_ROOT = old_root
                status.STEP1_PREDICTIONS_DIR = old_step1
                status.STEP2_PREDICTIONS_DIR = old_step2
                status.STEP3_PREDICTIONS_DIR = old_step3
                status.SUBSETS_DIR = old_subsets
                status.TABLES_DIR = old_tables

        self.assertEqual(built["overall"], "incomplete")
        self.assertEqual(
            built["commands"][0],
            "STEP=1 LIMIT=0 bash scripts/run_operational_umls_experiments.sh",
        )

    def test_subset_audit_resolves_large_models_to_9b_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_subsets = status.SUBSETS_DIR
            status.SUBSETS_DIR = Path(tmp)
            manifest = status.SUBSETS_DIR / (
                "medQA_test_spacy_Qwen__Qwen3_5-9B_non_thinking_baseline_wrong.jsonl"
            )
            manifest.write_text('{"subset_label":"still_wrong"}\n', encoding="utf-8")
            try:
                entries, missing = status.audit_subsets(linkers=("spacy",))
            finally:
                status.SUBSETS_DIR = old_subsets

        by_model = {
            entry["model"]: entry
            for entry in entries
            if entry["dataset"] == "medQA" and entry["linker"] == "spacy"
        }
        self.assertTrue(by_model["Qwen/Qwen3.5-9B"]["nonempty"])
        self.assertIsNone(by_model["Qwen/Qwen3.5-9B"].get("fallback_model"))
        self.assertTrue(by_model["Qwen/Qwen3.5-27B"]["nonempty"])
        self.assertEqual(by_model["Qwen/Qwen3.5-27B"]["fallback_model"], "Qwen/Qwen3.5-9B")
        self.assertTrue(by_model["Qwen/Qwen3.5-35B"]["nonempty"])
        self.assertEqual(by_model["Qwen/Qwen3.5-35B"]["fallback_model"], "Qwen/Qwen3.5-9B")
        self.assertNotIn("medQA_test_spacy_Qwen__Qwen3_5-27B", "\n".join(missing))


if __name__ == "__main__":
    unittest.main()
