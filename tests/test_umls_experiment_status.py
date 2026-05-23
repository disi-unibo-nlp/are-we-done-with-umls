import tempfile
import unittest
from pathlib import Path

from src.experiments import umls_experiment_status as status


class UmlsExperimentStatusTests(unittest.TestCase):
    def test_scale_model_contract_is_qwen35_only(self) -> None:
        self.assertEqual(
            status.SCALE_MODELS,
            (
                "Qwen/Qwen3.5-0.8B",
                "Qwen/Qwen3.5-2B",
                "Qwen/Qwen3.5-4B",
                "Qwen/Qwen3.5-9B",
            ),
        )

    def test_expected_step3_conditions_are_depth_width_matrix(self) -> None:
        self.assertEqual(len(status.STEP3_CONDITIONS), 16)
        self.assertIn("chain_depth_1_width_tight_evidence", status.STEP3_CONDITIONS)
        self.assertIn("chain_depth_4_width_full_evidence", status.STEP3_CONDITIONS)
        self.assertNotIn("q_to_a_depth_2_obscured", status.STEP3_CONDITIONS)

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


if __name__ == "__main__":
    unittest.main()
