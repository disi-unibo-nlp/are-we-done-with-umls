import os
import subprocess
import tempfile
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
            "scripts/run_overnight_umls_all_models.sh",
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

    def test_operational_runner_defines_exact_q_to_a_step3_without_width_matrix(self) -> None:
        text = self.script_text("scripts/run_operational_umls_experiments.sh")
        self.assertIn("q_to_a_exact_depth_${depth}_evidence", text)
        self.assertIn("for depth in 1 2 3 4 5", text)
        self.assertIn("Q2A_DEPTH_CONDITIONS", text)
        self.assertNotIn("CHAIN_CONDITIONS", text)
        self.assertNotIn("chain_depth_${depth}_width_${width}_evidence", text)
        self.assertNotIn("q_to_a_depth_2_obscured", text)
        self.assertNotIn("pooled_width_tight_obscured", text)
        self.assertNotIn("prior_work_obscured_evidence", text)

    def test_operational_runner_defaults_step23_to_spacy_only(self) -> None:
        text = self.script_text("scripts/run_operational_umls_experiments.sh")
        self.assertIn('STEP23_LINKERS="${STEP23_LINKERS:-spacy}"', text)
        self.assertIn('run_filtered_step "step2" "${STEP2_CONDITIONS}" "item_cuis" "${TRIPLETS_ZIP}" "${STEP23_LINKERS}" "0"', text)
        self.assertIn('run_filtered_step "step3" "${Q2A_DEPTH_CONDITIONS}" "full" "${graph_zip}" "${STEP23_LINKERS}" "${STEP3_REQUIRE_Q_TO_A_PATH}"', text)

    def test_operational_runner_dry_run_step2_uses_step23_linkers(self) -> None:
        env = os.environ.copy()
        env.update({
            "DRY_RUN": "1",
            "STEP": "2",
            "LIMIT": "1",
            "DATASETS": "medQA",
            "LINKERS": "metamap spacy",
            "STEP23_LINKERS": "spacy",
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
        self.assertIn("medQA_test_spacy_Qwen__Qwen3_5-4B_non_thinking_baseline_wrong.jsonl", result.stdout)
        self.assertNotIn("medQA_test_metamap_Qwen__Qwen3_5-4B_non_thinking_baseline_wrong.jsonl", result.stdout)

    def test_operational_runner_defaults_step2_to_three_ordered_umls_conditions(self) -> None:
        text = self.script_text("scripts/run_operational_umls_experiments.sh")
        self.assertIn(
            'STEP2_CONDITIONS="${STEP2_CONDITIONS:-relation_only_evidence q_to_a_path_evidence metadata_rich_evidence}"',
            text,
        )
        self.assertNotIn(
            'STEP2_CONDITIONS="${STEP2_CONDITIONS:-relation_only_evidence metadata_rich_evidence gold_only_evidence shuffled_answer_evidence q_to_a_path_evidence}"',
            text,
        )

    def test_operational_runner_default_sweep_keeps_large_models_with_step1_fallback(self) -> None:
        operational = self.script_text("scripts/run_operational_umls_experiments.sh")
        overnight = self.script_text("scripts/run_overnight_umls_all_models.sh")
        for text in (operational, overnight):
            self.assertIn("Qwen/Qwen3.5-0.8B", text)
            self.assertIn("Qwen/Qwen3.5-9B", text)
            self.assertIn("Qwen/Qwen3.5-27B", text)
            self.assertIn("Qwen/Qwen3.5-35B", text)
            self.assertIn('STEP1_SUBSET_FALLBACK_MODEL="${STEP1_SUBSET_FALLBACK_MODEL:-Qwen/Qwen3.5-9B}"', text)

    def test_operational_runner_uses_9b_step1_manifest_for_large_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subsets = root / "step1_subsets"
            subsets.mkdir(parents=True)
            fallback_manifest = subsets / (
                "medQA_test_spacy_Qwen__Qwen3_5-9B_non_thinking_baseline_wrong.jsonl"
            )
            fallback_manifest.write_text('{"item_id":"x"}\n', encoding="utf-8")

            env = os.environ.copy()
            env.update({
                "DRY_RUN": "1",
                "STEP": "2",
                "LIMIT": "1",
                "DATASETS": "medQA",
                "STEP23_LINKERS": "spacy",
                "QWEN_MODEL_LIST": "Qwen/Qwen3.5-27B",
                "OUTPUT_ROOT": str(root),
                "UV_BIN": "/usr/bin/true",
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
        self.assertIn("Using STEP 1 fallback manifest for model=Qwen/Qwen3.5-27B", result.stdout)
        self.assertIn(str(fallback_manifest), result.stdout)
        self.assertIn("Qwen3.5-27B", result.stdout)

    def test_downstream_wrapper_accepts_subset_and_relation_scope(self) -> None:
        text = self.script_text("scripts/run_downstream_umls_probe.sh")
        self.assertIn("ITEM_ID_FILE", text)
        self.assertIn("--item-id-file", text)
        self.assertIn("RELATION_SCOPE", text)
        self.assertIn("--relation-scope", text)
        self.assertIn("MAX_ITEMS_PER_CONDITION", text)
        self.assertIn("--max-items-per-condition", text)
        self.assertIn("REQUIRE_Q_TO_A_PATH", text)
        self.assertIn("--require-q-to-a-path", text)
        self.assertIn("EVIDENCE_CACHE_JSONL", text)
        self.assertIn("--evidence-cache-jsonl", text)
        self.assertIn("REQUIRE_EVIDENCE_CACHE", text)
        self.assertIn("--require-evidence-cache", text)

    def test_operational_step3_defaults_to_path_present_sampling(self) -> None:
        operational = self.script_text("scripts/run_operational_umls_experiments.sh")
        overnight = self.script_text("scripts/run_overnight_umls_all_models.sh")
        for text in (operational, overnight):
            self.assertIn('MAX_ITEMS_PER_CONDITION="${MAX_ITEMS_PER_CONDITION:-${MAX_EXPERIMENT_INSTANCES:-100}}"', text)
            self.assertIn('STEP3_REQUIRE_Q_TO_A_PATH="${STEP3_REQUIRE_Q_TO_A_PATH:-1}"', text)

    def test_operational_runner_can_reuse_fixed_nonthinking_subset_for_reasoning_outputs(self) -> None:
        text = self.script_text("scripts/run_operational_umls_experiments.sh")
        self.assertIn("STEP1_SUBSET_ROOT", text)
        self.assertIn('STEP1_SUBSET_MODE="${STEP1_SUBSET_MODE:-${QWEN_MODE}}"', text)
        self.assertIn('"${STEP1_SUBSET_ROOT}"', text)
        self.assertIn('"${STEP1_SUBSET_MODE}"', text)

    def test_overnight_runner_allows_subset_only_step_selection(self) -> None:
        text = self.script_text("scripts/run_overnight_umls_all_models.sh")
        self.assertIn('RUN_STEPS="${RUN_STEPS:-2 3}"', text)
        self.assertIn('STEP23_LINKERS="${STEP23_LINKERS:-spacy}"', text)
        self.assertIn('run_steps "${model}" "${RUN_STEPS}"', text)

    def test_reasoning_mode_plumbs_vllm_budget_and_parser_settings(self) -> None:
        downstream = self.script_text("scripts/run_downstream_umls_probe.sh")
        operational = self.script_text("scripts/run_operational_umls_experiments.sh")
        overnight = self.script_text("scripts/run_overnight_umls_all_models.sh")

        self.assertIn("QWEN_THINKING_TOKEN_BUDGET", downstream)
        self.assertIn("--thinking-token-budget", downstream)
        self.assertIn("QWEN_ANSWER_TOKEN_BUDGET", downstream)
        self.assertIn("QWEN_THINKING_TOKEN_BUDGET", operational)
        self.assertIn("QWEN_ANSWER_TOKEN_BUDGET", operational)
        self.assertIn("QWEN_THINKING_TOKEN_BUDGET + QWEN_ANSWER_TOKEN_BUDGET", operational)
        self.assertIn("VLLM_REASONING_PARSER", overnight)
        self.assertIn("--reasoning-parser", overnight)
        self.assertIn("VLLM_REASONING_CONFIG", overnight)
        self.assertIn("--reasoning-config", overnight)

    def test_operational_tables_use_q_to_a_depth_analysis_not_depth_width(self) -> None:
        text = self.script_text("scripts/run_operational_umls_experiments.sh")
        self.assertIn("q_to_a_depth_analysis.py", text)
        self.assertNotIn("depth_width_analysis.py", text)


if __name__ == "__main__":
    unittest.main()
