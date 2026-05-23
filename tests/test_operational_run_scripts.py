import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class OperationalRunScriptTests(unittest.TestCase):
    def script_text(self, relpath: str) -> str:
        return (REPO_ROOT / relpath).read_text(encoding="utf-8")

    def test_kept_shell_scripts_parse(self) -> None:
        scripts = [
            "scripts/run_downstream_umls_probe.sh",
            "scripts/run_operational_umls_experiments.sh",
            "scripts/umls_experiment_monitor.sh",
            "scripts/openclaw.sh",
        ]
        result = subprocess.run(
            ["bash", "-n", *scripts],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_operational_runner_dry_run_reports_step1_conditions(self) -> None:
        env = os.environ.copy()
        env.update({
            "DRY_RUN": "1",
            "STEP": "1",
            "LIMIT": "1",
            "DATASETS": "medQA",
            "LINKERS": "metamap",
            "QWEN_MODEL_LIST": "Qwen/Qwen3.5-4B",
        })
        result = subprocess.run(
            ["bash", "scripts/run_operational_umls_experiments.sh"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no_umls", result.stdout)
        self.assertIn("metadata_rich_evidence", result.stdout)
        self.assertIn("DRY_RUN=1", result.stdout)

    def test_operational_runner_defines_chain_matrix_without_obscured_conditions(self) -> None:
        text = self.script_text("scripts/run_operational_umls_experiments.sh")
        self.assertIn("chain_depth_${depth}_width_${width}_evidence", text)
        self.assertNotIn("q_to_a_depth_2_obscured", text)
        self.assertNotIn("pooled_width_tight_obscured", text)
        self.assertNotIn("prior_work_obscured_evidence", text)

    def test_downstream_wrapper_accepts_subset_and_relation_scope(self) -> None:
        text = self.script_text("scripts/run_downstream_umls_probe.sh")
        self.assertIn("ITEM_ID_FILE", text)
        self.assertIn("--item-id-file", text)
        self.assertIn("RELATION_SCOPE", text)
        self.assertIn("--relation-scope", text)


if __name__ == "__main__":
    unittest.main()
