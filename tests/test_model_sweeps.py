import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.experiments.loaders import load_metamap_records
from src.model_sweeps.qwen_vllm_sweep import (
    NON_REASONING_PROMPT,
    REASONING_PROMPT,
    boxed_answer_regex,
    build_prompt,
    build_request_payload,
    decoding_params,
    extract_answer,
    normalize_limit,
    response_text,
    run_sweep,
    summarize_predictions,
)
from src.model_sweeps.conditioned_metrics import (
    conditioned_delta_rows,
    downstream_condition_summary_rows,
    intervention_by_diagnostic_rows,
    main_claim_summary_rows,
    paired_intervention_effect_rows,
    write_downstream_condition_summary_outputs,
    write_intervention_by_diagnostic_outputs,
    write_main_claim_summary_outputs,
    write_paired_intervention_effect_outputs,
)
from src.experiments.evidence_cards import condition_type_for_condition
from src.model_sweeps.downstream_umls_probe import run_medgemma_probe, run_qwen_probe
from src.model_sweeps.local_logits_sweep import load_mock_model
from src.experiments.schemas import QAItem
from src.experiments.umls_index import CuiMetadataIndex, RelationIndex

import zipfile


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def write_zip_text(path: Path, member: str, text: str) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, text)


class QwenVLLMSweepTests(unittest.TestCase):
    def make_item(self, item_id: str, gold: str = "A") -> QAItem:
        return QAItem(
            item_id=item_id,
            dataset="medQA",
            split="test",
            linker="metamap",
            question="Which option is correct?",
            options={"A": "Correct", "B": "Incorrect"},
            gold=gold,
            answer_text="Correct",
        )

    def test_decoding_params_match_qwen35_text_modes(self) -> None:
        non_thinking = decoding_params("non_thinking")
        thinking = decoding_params("thinking")

        self.assertEqual(non_thinking["temperature"], 1.0)
        self.assertEqual(non_thinking["top_p"], 1.0)
        self.assertEqual(non_thinking["top_k"], 20)
        self.assertEqual(non_thinking["min_p"], 0.0)
        self.assertEqual(non_thinking["presence_penalty"], 2.0)
        self.assertEqual(non_thinking["repetition_penalty"], 1.0)
        self.assertFalse(non_thinking["enable_thinking"])

        self.assertEqual(thinking["temperature"], 1.0)
        self.assertEqual(thinking["top_p"], 0.95)
        self.assertEqual(thinking["top_k"], 20)
        self.assertEqual(thinking["min_p"], 0.0)
        self.assertEqual(thinking["presence_penalty"], 1.5)
        self.assertEqual(thinking["repetition_penalty"], 1.0)
        self.assertTrue(thinking["enable_thinking"])

    def test_mode_prompts_are_module_level_templates(self) -> None:
        self.assertIn("medical expert tasked with answering multiple-choice medical questions", NON_REASONING_PROMPT)
        self.assertIn("{question}", NON_REASONING_PROMPT)
        self.assertIn("{options}", NON_REASONING_PROMPT)
        self.assertIn("{umls_subgraph}", NON_REASONING_PROMPT)
        self.assertIn("medical expert tasked with answering multiple-choice medical questions", REASONING_PROMPT)
        self.assertIn("Briefly reason over the clinically relevant facts and UMLS relations", REASONING_PROMPT)
        self.assertIn("When private reasoning ends", REASONING_PROMPT)
        self.assertIn("{question}", REASONING_PROMPT)
        self.assertIn("{options}", REASONING_PROMPT)
        self.assertIn("{umls_subgraph}", REASONING_PROMPT)

    def test_payload_contains_vllm_extra_params_and_thinking_toggle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.jsonl"
            write_jsonl(path, [{
                "question": "Which therapy is monitored with aPTT?",
                "options": {"A": "Warfarin", "B": "Heparin", "C": "Aspirin", "D": "Insulin"},
                "answer_idx": "B",
                "answer": "Heparin",
                "concept": [],
                "option0": [],
                "option1": [],
                "option2": [],
                "option3": [],
            }])
            item = load_metamap_records(path, dataset="medQA", split="test")[0]

        payload = build_request_payload(
            item,
            model="Qwen/Qwen3.5-0.8B",
            mode="non_thinking",
            max_tokens=128,
        )

        self.assertEqual(payload["model"], "Qwen/Qwen3.5-0.8B")
        self.assertEqual(payload["max_tokens"], 128)
        self.assertEqual(payload["top_k"], 20)
        self.assertEqual(payload["min_p"], 0.0)
        self.assertEqual(payload["repetition_penalty"], 1.0)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertIn("medical expert tasked with answering multiple-choice medical questions", payload["messages"][0]["content"])
        self.assertIn(r"\boxed{}", payload["messages"][-1]["content"])
        self.assertIn("Do not include explanation", payload["messages"][-1]["content"])
        self.assertIn("A. Warfarin", payload["messages"][-1]["content"])

    def test_payload_accepts_deterministic_intervention_decoding_overrides(self) -> None:
        item = self.make_item("first")

        payload = build_request_payload(
            item,
            model="Qwen/Qwen3.5-4B",
            mode="non_thinking",
            max_tokens=64,
            decoding_overrides={
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 20,
                "presence_penalty": 0.0,
                "seed": 42,
            },
        )

        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["top_p"], 1.0)
        self.assertEqual(payload["top_k"], 20)
        self.assertEqual(payload["presence_penalty"], 0.0)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_thinking_payload_sets_budget_and_final_answer_prompt(self) -> None:
        item = self.make_item("first")

        payload = build_request_payload(
            item,
            model="Qwen/Qwen3.5-0.8B",
            mode="thinking",
            max_tokens=1536,
            evidence="C_Q -> associated_with -> C_A",
            thinking_token_budget=1024,
        )

        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(payload["thinking_token_budget"], 1024)
        self.assertEqual(payload["structured_outputs"], {"regex": r"\\boxed\{(?:A|B)\}"})
        user_prompt = payload["messages"][-1]["content"]
        self.assertIn("Briefly reason over the clinically relevant facts and UMLS relations", user_prompt)
        self.assertIn("C_Q -> associated_with -> C_A", user_prompt)
        self.assertIn(r"\boxed{}", user_prompt)
        self.assertNotIn("{umls_subgraph}", user_prompt)
        self.assertNotIn("Do not explain your reasoning", user_prompt)

    def test_response_text_accepts_vllm_reasoning_field_names(self) -> None:
        response = {
            "choices": [{
                "message": {
                    "content": r"\boxed{B}",
                    "reasoning": "private trace",
                }
            }]
        }

        self.assertEqual(response_text(response), (r"\boxed{B}", "private trace"))

        legacy_response = {
            "choices": [{
                "message": {
                    "content": r"\boxed{C}",
                    "reasoning_content": "legacy private trace",
                }
            }]
        }

        self.assertEqual(response_text(legacy_response), (r"\boxed{C}", "legacy private trace"))

    def test_non_thinking_payload_omits_thinking_budget(self) -> None:
        item = self.make_item("first")

        payload = build_request_payload(
            item,
            model="Qwen/Qwen3.5-0.8B",
            mode="non_thinking",
            max_tokens=32,
            thinking_token_budget=1024,
        )

        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("thinking_token_budget", payload)
        self.assertNotIn("structured_outputs", payload)

    def test_boxed_answer_regex_escapes_non_letter_options(self) -> None:
        self.assertEqual(
            boxed_answer_regex(["yes", "no", "maybe"]),
            r"\\boxed\{(?:yes|no|maybe)\}",
        )
        self.assertEqual(
            boxed_answer_regex(["A", "B.1"]),
            r"\\boxed\{(?:A|B\.1)\}",
        )

    def test_prompt_can_include_umls_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.jsonl"
            write_jsonl(path, [{
                "question": "Which therapy is monitored with aPTT?",
                "options": {"A": "Warfarin", "B": "Heparin", "C": "Aspirin", "D": "Insulin"},
                "answer_idx": "B",
                "answer": "Heparin",
                "concept": [],
                "option0": [],
                "option1": [],
                "option2": [],
                "option3": [],
            }])
            item = load_metamap_records(path, dataset="medQA", split="test")[0]

        prompt = build_prompt(item, evidence="heparin -> monitored_by -> activated partial thromboplastin time")

        self.assertIn("UMLS evidence:", prompt)
        self.assertIn("heparin -> monitored_by", prompt)
        self.assertNotIn("{umls_subgraph}", prompt)

    def test_extract_answer_accepts_json_and_fallback_text(self) -> None:
        self.assertEqual(extract_answer('{"answer": "B", "rationale": "text"}', ["A", "B", "C", "D"]), "B")
        self.assertEqual(extract_answer('{"answer": "b", "rationale": "text"}', ["A", "B", "C", "D"]), "B")
        self.assertEqual(extract_answer(r"\boxed{ C }", ["A", "B", "C", "D"]), "C")
        self.assertEqual(extract_answer("Final answer: C", ["A", "B", "C", "D"]), "C")
        self.assertIsNone(extract_answer("The best answer is C. because...", ["A", "B", "C", "D"]))
        self.assertIsNone(extract_answer("Option C is tempting but wrong.", ["A", "B", "C", "D"]))
        self.assertIsNone(extract_answer("I cannot determine this.", ["A", "B", "C", "D"]))

    def test_extract_answer_preserves_option_case_for_pubmedqa(self) -> None:
        # pubmedQA uses lowercase option keys; extract_answer must return the canonical key
        self.assertEqual(extract_answer(r"\boxed{yes}", ["yes", "no", "maybe"]), "yes")
        self.assertEqual(extract_answer(r"\boxed{YES}", ["yes", "no", "maybe"]), "yes")
        self.assertEqual(extract_answer(r"\boxed{No}", ["yes", "no", "maybe"]), "no")
        self.assertEqual(extract_answer("Final answer: maybe", ["yes", "no", "maybe"]), "maybe")
        self.assertEqual(extract_answer('{"answer": "yes"}', ["yes", "no", "maybe"]), "yes")

    def test_summary_counts_accuracy_by_mode(self) -> None:
        rows = [
            {"mode": "non_thinking", "is_correct": True, "prediction": "B"},
            {"mode": "non_thinking", "is_correct": False, "prediction": "A"},
            {"mode": "thinking", "is_correct": True, "prediction": "C"},
        ]

        summary = summarize_predictions(rows)

        self.assertEqual(summary[("non_thinking",)]["n"], 2)
        self.assertEqual(summary[("non_thinking",)]["accuracy"], 0.5)
        self.assertEqual(summary[("non_thinking",)]["malformed"], 0)
        self.assertEqual(summary[("thinking",)]["n"], 1)
        self.assertEqual(summary[("thinking",)]["accuracy"], 1.0)
        self.assertEqual(summary[("thinking",)]["malformed"], 0)

    def test_summary_counts_missing_predictions_as_malformed(self) -> None:
        rows = [
            {"mode": "thinking", "is_correct": False, "prediction": None, "elapsed_s": 2.0},
            {"mode": "thinking", "is_correct": True, "prediction": "B", "elapsed_s": 4.0},
        ]

        summary = summarize_predictions(rows)

        self.assertEqual(summary[("thinking",)]["n"], 2)
        self.assertEqual(summary[("thinking",)]["malformed"], 1)
        self.assertEqual(summary[("thinking",)]["mean_elapsed_s"], 3.0)

    def test_zero_limit_means_whole_dataset(self) -> None:
        self.assertIsNone(normalize_limit(0))
        self.assertIsNone(normalize_limit(-1))
        self.assertEqual(normalize_limit(7), 7)

    def test_run_sweep_resumes_existing_jsonl_without_duplicates(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.seen_item_ids: list[str] = []

            def chat_completion(self, payload: dict) -> dict:
                question = payload["messages"][-1]["content"]
                item_id = "second" if "Second" in question else "unknown"
                self.seen_item_ids.append(item_id)
                return {"choices": [{"message": {"content": '{"answer": "A", "rationale": "ok"}'}}]}

        items = [
            self.make_item("first"),
            QAItem(
                item_id="second",
                dataset="medQA",
                split="test",
                linker="metamap",
                question="Second question?",
                options={"A": "Correct", "B": "Incorrect"},
                gold="A",
                answer_text="Correct",
            ),
        ]

        existing = [{
            "item_id": "first",
            "dataset": "medQA",
            "split": "test",
            "model": "Qwen/Qwen3.5-0.8B",
            "mode": "non_thinking",
            "gold": "A",
            "prediction": "A",
            "is_correct": True,
            "content": '{"answer": "A", "rationale": "cached"}',
            "reasoning_content": "",
            "elapsed_s": 1.0,
        }]

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "predictions.jsonl"
            write_jsonl(output_path, existing)

            client = FakeClient()
            rows = run_sweep(
                items,
                client,
                model="Qwen/Qwen3.5-0.8B",
                mode="non_thinking",
                max_tokens=16,
                output_jsonl=output_path,
                existing_rows=existing,
            )

            persisted = [json.loads(line) for line in output_path.read_text().splitlines()]

        self.assertEqual(client.seen_item_ids, ["second"])
        self.assertEqual([row["item_id"] for row in rows], ["first", "second"])
        self.assertEqual([row["item_id"] for row in persisted], ["first", "second"])

    def test_run_sweep_can_submit_requests_concurrently_for_vllm_batching(self) -> None:
        class SlowFakeClient:
            def __init__(self) -> None:
                self.active = 0
                self.max_active = 0
                self.lock = threading.Lock()

            def chat_completion(self, payload: dict) -> dict:
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.05)
                with self.lock:
                    self.active -= 1
                return {"choices": [{"message": {"content": '{"answer": "A", "rationale": "ok"}'}}]}

        client = SlowFakeClient()
        rows = run_sweep(
            [self.make_item("first"), self.make_item("second")],
            client,
            model="Qwen/Qwen3.5-0.8B",
            mode="non_thinking",
            max_tokens=16,
            concurrency=2,
        )

        self.assertEqual(client.max_active, 2)
        self.assertEqual({row["item_id"] for row in rows}, {"first", "second"})

    def test_run_sweep_records_exhausted_request_errors(self) -> None:
        class FailingClient:
            def chat_completion(self, payload: dict) -> dict:
                raise RuntimeError("request failed")

        rows = run_sweep(
            [self.make_item("first")],
            FailingClient(),
            model="Qwen/Qwen3.5-0.8B",
            mode="non_thinking",
            max_tokens=16,
            max_retries=1,
        )

        self.assertEqual(rows[0]["item_id"], "first")
        self.assertIsNone(rows[0]["prediction"])
        self.assertFalse(rows[0]["is_correct"])
        self.assertIn("request failed", rows[0]["error"])

    def test_downstream_medgemma_probe_emits_one_row_per_evidence_condition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta_path = root / "cui_detail.tsv.zip"
            rel_path = root / "triplets.csv.zip"
            write_zip_text(
                meta_path,
                "cui_detail.tsv",
                "CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF\n"
                "C_Q\tQuestion concept\tT033\tFinding\t\t\n"
                "C_A\tCorrect\tT047\tDisease or Syndrome\tMSH\tCorrect definition.\n",
            )
            write_zip_text(
                rel_path,
                "triplets.csv",
                "cui_1,cui_2,re,rel_type\nC_Q,C_A,RO,associated_with\n",
            )
            item = self.make_item("first", gold="A")
            item = QAItem(
                item_id=item.item_id,
                dataset=item.dataset,
                split=item.split,
                linker="metamap",
                question=item.question,
                options=item.options,
                gold=item.gold,
                answer_text=item.answer_text,
                question_cuis=["C_Q"],
                option_cuis={"A": ["C_A"], "B": []},
            )

            rows = run_medgemma_probe(
                [item],
                load_mock_model(),
                conditions=("no_umls", "metadata_rich_evidence"),
                metadata=CuiMetadataIndex.from_zip(meta_path),
                relations=RelationIndex.from_zip(rel_path, keep_cuis={"C_Q", "C_A"}),
            )

        self.assertEqual([row["condition"] for row in rows], ["no_umls", "metadata_rich_evidence"])
        self.assertEqual(rows[1]["prompt_condition"], "metadata_rich_evidence")
        self.assertIn("confidence", rows[1])
        self.assertFalse(rows[1]["malformed"])
        self.assertEqual(rows[1]["evidence_view"], "full_committed_card")
        self.assertEqual(rows[1]["condition_type"], "realistic")

    def test_downstream_probe_marks_oracle_gold_evidence_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta_path = root / "cui_detail.tsv.zip"
            rel_path = root / "triplets.csv.zip"
            write_zip_text(
                meta_path,
                "cui_detail.tsv",
                "CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF\n"
                "C_Q\tQuestion concept\tT033\tFinding\t\t\n"
                "C_GOLD\tCorrect\tT047\tDisease or Syndrome\tMSH\tCorrect definition.\n",
            )
            write_zip_text(
                rel_path,
                "triplets.csv",
                "cui_1,cui_2,re,rel_type\nC_Q,C_GOLD,RO,associated_with\n",
            )
            item = QAItem(
                item_id="first",
                dataset="medQA",
                split="test",
                linker="metamap",
                question="Which option is correct?",
                options={"A": "Correct", "B": "Incorrect"},
                gold="A",
                answer_text="Correct",
                question_cuis=["C_Q"],
                option_cuis={"A": [], "B": []},
            )

            rows = run_medgemma_probe(
                [item],
                load_mock_model(),
                conditions=("oracle_gold_evidence",),
                metadata=CuiMetadataIndex.from_zip(meta_path),
                relations=RelationIndex.from_zip(rel_path, keep_cuis={"C_Q", "C_GOLD"}),
            )

        self.assertEqual(rows[0]["condition"], "oracle_gold_evidence")
        self.assertEqual(rows[0]["condition_type"], "oracle")
        self.assertTrue(rows[0]["oracle"])
        self.assertTrue(rows[0]["oracle_lookup_success"])
        self.assertEqual(rows[0]["oracle_num_candidate_cuis"], 1)
        self.assertFalse(rows[0]["oracle_ambiguous"])
        self.assertEqual(rows[0]["oracle_selected_cui_source"], "exact_answer_text_lookup")
        self.assertEqual(rows[0]["oracle_gold_cui_source"], "exact_answer_text_lookup")
        self.assertEqual(rows[0]["evidence_view"], "full_committed_card")

    def test_downstream_condition_type_mapping_marks_realistic_controls_and_oracles(self) -> None:
        self.assertEqual(condition_type_for_condition("no_umls"), "baseline")
        self.assertEqual(condition_type_for_condition("relation_only_evidence"), "realistic")
        self.assertEqual(condition_type_for_condition("metadata_rich_evidence"), "realistic")
        self.assertEqual(condition_type_for_condition("q_to_a_path_evidence"), "realistic")
        self.assertEqual(condition_type_for_condition("distractor_contrast_evidence"), "realistic_or_contrastive")
        self.assertEqual(condition_type_for_condition("random_same_size_evidence"), "negative_control")
        self.assertEqual(condition_type_for_condition("shuffled_answer_evidence"), "negative_control")
        self.assertEqual(condition_type_for_condition("gold_only_evidence"), "diagnostic")
        self.assertEqual(condition_type_for_condition("oracle_gold_evidence"), "oracle")
        self.assertEqual(condition_type_for_condition("q_to_a_d4_shortcut_1_evidence"), "synthetic_control")

    def test_downstream_qwen_probe_passes_evidence_to_generation_client(self) -> None:
        class CapturingClient:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def chat_completion(self, payload: dict) -> dict:
                self.prompts.append(payload["messages"][-1]["content"])
                return {"choices": [{"message": {"content": '{"answer": "A", "rationale": "ok"}'}}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta_path = root / "cui_detail.tsv.zip"
            rel_path = root / "triplets.csv.zip"
            write_zip_text(
                meta_path,
                "cui_detail.tsv",
                "CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF\n"
                "C_Q\tQuestion concept\tT033\tFinding\t\t\n"
                "C_A\tCorrect\tT047\tDisease or Syndrome\tMSH\tCorrect definition.\n",
            )
            write_zip_text(
                rel_path,
                "triplets.csv",
                "cui_1,cui_2,re,rel_type\nC_Q,C_A,RO,associated_with\n",
            )
            item = QAItem(
                item_id="first",
                dataset="medQA",
                split="test",
                linker="metamap",
                question="Which option is correct?",
                options={"A": "Correct", "B": "Incorrect"},
                gold="A",
                answer_text="Correct",
                question_cuis=["C_Q"],
                option_cuis={"A": ["C_A"], "B": []},
            )
            client = CapturingClient()

            rows = run_qwen_probe(
                [item],
                client,
                model="Qwen/Qwen3.5-4B",
                mode="non_thinking",
                max_tokens=16,
                conditions=("no_umls", "metadata_rich_evidence"),
                metadata=CuiMetadataIndex.from_zip(meta_path),
                relations=RelationIndex.from_zip(rel_path, keep_cuis={"C_Q", "C_A"}),
            )

        self.assertEqual([row["condition"] for row in rows], ["no_umls", "metadata_rich_evidence"])
        self.assertNotIn("UMLS evidence:", client.prompts[0])
        self.assertIn("UMLS evidence:", client.prompts[1])
        self.assertIn("Correct definition.", client.prompts[1])

    def test_downstream_qwen_probe_annotates_exact_q_to_a_depth_rows(self) -> None:
        class CapturingClient:
            def chat_completion(self, payload: dict) -> dict:
                return {"choices": [{"message": {"content": r"\boxed{A}"}}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta_path = root / "cui_detail.tsv.zip"
            rel_path = root / "triplets.csv.zip"
            write_zip_text(
                meta_path,
                "cui_detail.tsv",
                "CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF\n"
                "C_Q\tQuestion concept\tT033\tFinding\t\t\n"
                "C_A\tCorrect\tT047\tDisease or Syndrome\tMSH\tCorrect definition.\n"
                "C_B\tIncorrect\tT047\tDisease or Syndrome\tMSH\tIncorrect definition.\n",
            )
            write_zip_text(
                rel_path,
                "triplets.csv",
                "cui_1,cui_2,re,rel_type\n"
                "C_Q,C_A,RO,associated_with\n"
                "C_Q,C_B,RO,associated_with\n",
            )
            item = QAItem(
                item_id="first",
                dataset="medQA",
                split="test",
                linker="metamap",
                question="Which option is correct?",
                options={"A": "Correct", "B": "Incorrect"},
                gold="A",
                answer_text="Correct",
                question_cuis=["C_Q"],
                option_cuis={"A": ["C_A"], "B": ["C_B"]},
            )

            rows = run_qwen_probe(
                [item],
                CapturingClient(),
                model="Qwen/Qwen3.5-4B",
                mode="non_thinking",
                max_tokens=16,
                conditions=("q_to_a_exact_depth_1_evidence",),
                metadata=CuiMetadataIndex.from_zip(meta_path),
                relations=RelationIndex.from_zip(rel_path, keep_cuis={"C_Q", "C_A", "C_B"}),
            )

        self.assertEqual(rows[0]["condition"], "q_to_a_exact_depth_1_evidence")
        self.assertEqual(rows[0]["condition_type"], "realistic")
        self.assertEqual(rows[0]["retrieval_method"], "q_to_a_exact_depth_1")
        self.assertEqual(rows[0]["q_to_a_depth_semantics"], "exact_hops")
        self.assertEqual(rows[0]["q_to_a_depth"], 1)
        self.assertTrue(rows[0]["q_to_a_path_found"])
        self.assertEqual(rows[0]["q_to_a_path_edge_count"], 2)
        self.assertEqual(rows[0]["q_to_a_path_node_count"], 3)

    def test_downstream_qwen_probe_can_skip_missing_exact_q_to_a_paths(self) -> None:
        class CapturingClient:
            def __init__(self) -> None:
                self.calls = 0

            def chat_completion(self, payload: dict) -> dict:
                self.calls += 1
                return {"choices": [{"message": {"content": r"\boxed{A}"}}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta_path = root / "cui_detail.tsv.zip"
            rel_path = root / "triplets.csv.zip"
            write_zip_text(
                meta_path,
                "cui_detail.tsv",
                "CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF\n"
                "C_Q1\tQuestion one\tT033\tFinding\t\t\n"
                "C_A1\tAnswer one\tT047\tDisease or Syndrome\t\t\n"
                "C_Q2\tQuestion two\tT033\tFinding\t\t\n"
                "C_A2\tAnswer two\tT047\tDisease or Syndrome\t\t\n",
            )
            write_zip_text(
                rel_path,
                "triplets.csv",
                "cui_1,cui_2,re,rel_type\nC_Q1,C_A1,RO,associated_with\n",
            )
            items = [
                QAItem(
                    item_id="has_path",
                    dataset="medQA",
                    split="test",
                    linker="spacy",
                    question="Which option is correct?",
                    options={"A": "Answer one"},
                    gold="A",
                    answer_text="Answer one",
                    question_cuis=["C_Q1"],
                    option_cuis={"A": ["C_A1"]},
                ),
                QAItem(
                    item_id="missing_path",
                    dataset="medQA",
                    split="test",
                    linker="spacy",
                    question="Which option is correct?",
                    options={"A": "Answer two"},
                    gold="A",
                    answer_text="Answer two",
                    question_cuis=["C_Q2"],
                    option_cuis={"A": ["C_A2"]},
                ),
            ]
            client = CapturingClient()

            rows = run_qwen_probe(
                items,
                client,
                model="Qwen/Qwen3.5-4B",
                mode="non_thinking",
                max_tokens=16,
                conditions=("q_to_a_exact_depth_1_evidence",),
                metadata=CuiMetadataIndex.from_zip(meta_path),
                relations=RelationIndex.from_zip(rel_path, keep_cuis={"C_Q1", "C_A1", "C_Q2", "C_A2"}),
                require_q_to_a_path=True,
            )

        self.assertEqual(client.calls, 1)
        self.assertEqual([row["item_id"] for row in rows], ["has_path"])
        self.assertTrue(rows[0]["q_to_a_path_found"])

    def test_downstream_qwen_probe_can_require_precomputed_evidence_cache(self) -> None:
        class CapturingClient:
            def __init__(self) -> None:
                self.prompts = []

            def chat_completion(self, payload: dict) -> dict:
                self.prompts.append("\n".join(message["content"] for message in payload["messages"]))
                return {"choices": [{"message": {"content": r"\boxed{A}"}}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta_path = root / "cui_detail.tsv.zip"
            rel_path = root / "triplets.csv.zip"
            write_zip_text(
                meta_path,
                "cui_detail.tsv",
                "CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF\n",
            )
            write_zip_text(
                rel_path,
                "triplets.csv",
                "cui_1,cui_2,re,rel_type\n",
            )
            items = [
                QAItem(
                    item_id="cached",
                    dataset="medQA",
                    split="test",
                    linker="spacy",
                    question="Which option is correct?",
                    options={"A": "Answer one"},
                    gold="A",
                    answer_text="Answer one",
                ),
                QAItem(
                    item_id="uncached",
                    dataset="medQA",
                    split="test",
                    linker="spacy",
                    question="Which option is correct?",
                    options={"A": "Answer two"},
                    gold="A",
                    answer_text="Answer two",
                ),
            ]
            condition = "q_to_a_exact_depth_1_evidence"
            cache = {
                ("cached", "medQA", "test", "spacy", condition): SimpleNamespace(
                    text="UMLS evidence:\nQuestion concept -- associated with --> Answer one",
                    q_to_a_depth_semantics="exact_hops",
                    q_to_a_depth=1,
                    q_to_a_path_found=True,
                    q_to_a_path_node_count=2,
                    q_to_a_path_edge_count=1,
                    q_to_a_answer_cui_count=1,
                    q_to_a_answer_cui_covered_count=1,
                    q_to_a_path_selection_policy="cached",
                )
            }
            client = CapturingClient()

            rows = run_qwen_probe(
                items,
                client,
                model="Qwen/Qwen3.5-4B",
                mode="non_thinking",
                max_tokens=16,
                conditions=(condition,),
                metadata=CuiMetadataIndex.from_zip(meta_path),
                relations=RelationIndex.from_zip(rel_path, keep_cuis=set()),
                evidence_cache=cache,
                require_evidence_cache=True,
            )

        self.assertEqual([row["item_id"] for row in rows], ["cached"])
        self.assertEqual(len(client.prompts), 1)
        self.assertIn("Question concept -- associated with --> Answer one", client.prompts[0])
        self.assertEqual(rows[0]["q_to_a_path_selection_policy"], "cached")

    def test_downstream_qwen_probe_caps_items_per_condition_after_filtering(self) -> None:
        class CapturingClient:
            def __init__(self) -> None:
                self.calls = 0

            def chat_completion(self, payload: dict) -> dict:
                self.calls += 1
                return {"choices": [{"message": {"content": r"\boxed{A}"}}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta_path = root / "cui_detail.tsv.zip"
            rel_path = root / "triplets.csv.zip"
            metadata_lines = ["CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF"]
            relation_lines = ["cui_1,cui_2,re,rel_type"]
            items = []
            keep_cuis = set()
            for index in range(4):
                q_cui = f"C_Q{index}"
                a_cui = f"C_A{index}"
                metadata_lines.extend([
                    f"{q_cui}\tQuestion {index}\tT033\tFinding\t\t",
                    f"{a_cui}\tAnswer {index}\tT047\tDisease or Syndrome\t\t",
                ])
                relation_lines.append(f"{q_cui},{a_cui},RO,associated_with")
                keep_cuis.update({q_cui, a_cui})
                items.append(QAItem(
                    item_id=f"item_{index}",
                    dataset="medQA",
                    split="test",
                    linker="spacy",
                    question="Which option is correct?",
                    options={"A": f"Answer {index}"},
                    gold="A",
                    answer_text=f"Answer {index}",
                    question_cuis=[q_cui],
                    option_cuis={"A": [a_cui]},
                ))
            write_zip_text(meta_path, "cui_detail.tsv", "\n".join(metadata_lines) + "\n")
            write_zip_text(rel_path, "triplets.csv", "\n".join(relation_lines) + "\n")
            client = CapturingClient()

            rows = run_qwen_probe(
                items,
                client,
                model="Qwen/Qwen3.5-4B",
                mode="non_thinking",
                max_tokens=16,
                conditions=("q_to_a_exact_depth_1_evidence",),
                metadata=CuiMetadataIndex.from_zip(meta_path),
                relations=RelationIndex.from_zip(rel_path, keep_cuis=keep_cuis),
                require_q_to_a_path=True,
                max_items_per_condition=2,
                selection_seed=42,
            )

        self.assertEqual(client.calls, 2)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["q_to_a_path_found"] for row in rows))

    def test_downstream_qwen_probe_resumes_without_duplicate_item_condition_rows(self) -> None:
        class CapturingClient:
            def __init__(self) -> None:
                self.calls = 0

            def chat_completion(self, payload: dict) -> dict:
                self.calls += 1
                return {"choices": [{"message": {"content": '{"answer": "A", "rationale": "ok"}'}}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta_path = root / "cui_detail.tsv.zip"
            rel_path = root / "triplets.csv.zip"
            output_path = root / "probe.jsonl"
            write_zip_text(
                meta_path,
                "cui_detail.tsv",
                "CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF\n"
                "C_Q\tQuestion concept\tT033\tFinding\t\t\n"
                "C_A\tCorrect\tT047\tDisease or Syndrome\tMSH\tCorrect definition.\n",
            )
            write_zip_text(
                rel_path,
                "triplets.csv",
                "cui_1,cui_2,re,rel_type\nC_Q,C_A,RO,associated_with\n",
            )
            item = QAItem(
                item_id="first",
                dataset="medQA",
                split="test",
                linker="metamap",
                question="Which option is correct?",
                options={"A": "Correct", "B": "Incorrect"},
                gold="A",
                answer_text="Correct",
                question_cuis=["C_Q"],
                option_cuis={"A": ["C_A"], "B": []},
            )
            existing = [{
                "item_id": "first",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model": "Qwen/Qwen3.5-4B",
                "mode": "non_thinking",
                "condition": "no_umls",
                "prediction": "A",
                "is_correct": True,
                "error": "",
            }]
            write_jsonl(output_path, existing)
            client = CapturingClient()

            rows = run_qwen_probe(
                [item],
                client,
                model="Qwen/Qwen3.5-4B",
                mode="non_thinking",
                max_tokens=16,
                conditions=("no_umls", "metadata_rich_evidence"),
                metadata=CuiMetadataIndex.from_zip(meta_path),
                relations=RelationIndex.from_zip(rel_path, keep_cuis={"C_Q", "C_A"}),
                output_jsonl=output_path,
                existing_rows=existing,
            )
            persisted = [json.loads(line) for line in output_path.read_text().splitlines()]

        self.assertEqual(client.calls, 1)
        self.assertEqual([row["condition"] for row in rows], ["no_umls", "metadata_rich_evidence"])
        self.assertEqual([row["condition"] for row in persisted], ["no_umls", "metadata_rich_evidence"])

    def test_conditioned_delta_rows_compare_no_umls_to_umls_condition(self) -> None:
        predictions = [
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "condition": "no_umls",
                "prediction": "A",
                "is_correct": False,
                "confidence": 0.40,
                "entropy": 1.20,
                "error": "",
            },
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "condition": "q_to_a_path_evidence",
                "prediction": "B",
                "is_correct": True,
                "confidence": 0.80,
                "entropy": 0.50,
                "error": "",
            },
        ]
        diagnostics = [
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "experiment": "retrieval_faithfulness",
                "condition": "q_to_a_2hop",
                "metrics": {"gold_path_available": True},
            }
        ]

        rows = conditioned_delta_rows(
            predictions,
            diagnostics,
            conditions=["gold_path_available"],
            umls_condition="q_to_a_path_evidence",
        )

        self.assertEqual(rows[0]["condition"], "gold_path_available")
        self.assertTrue(rows[0]["condition_value"])
        self.assertEqual(rows[0]["n"], 1)
        self.assertEqual(rows[0]["no_umls_accuracy"], 0.0)
        self.assertEqual(rows[0]["umls_accuracy"], 1.0)
        self.assertEqual(rows[0]["delta"], 1.0)
        self.assertAlmostEqual(rows[0]["confidence_delta"], 0.40)
        self.assertAlmostEqual(rows[0]["entropy_delta"], -0.70)

    def test_intervention_by_diagnostic_rows_compare_each_condition_to_no_umls(self) -> None:
        predictions = [
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "MockLogits",
                "model": "mock-local-logits",
                "mode": "non_thinking",
                "condition": "no_umls",
                "prediction": "A",
                "is_correct": False,
                "confidence": 0.40,
                "margin": 0.10,
                "entropy": 1.20,
                "error": "",
            },
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "MockLogits",
                "model": "mock-local-logits",
                "mode": "non_thinking",
                "condition": "distractor_contrast_evidence",
                "prediction": "B",
                "is_correct": True,
                "confidence": 0.80,
                "margin": 0.50,
                "entropy": 0.60,
                "error": "",
            },
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "MockLogits",
                "model": "mock-local-logits",
                "mode": "non_thinking",
                "condition": "random_same_size_evidence",
                "prediction": "A",
                "is_correct": False,
                "confidence": 0.50,
                "margin": 0.20,
                "entropy": 1.00,
                "error": "",
            },
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "MockLogits",
                "model": "mock-local-logits",
                "mode": "non_thinking",
                "condition": "shuffled_answer_evidence",
                "prediction": "A",
                "is_correct": False,
                "confidence": 0.45,
                "margin": 0.10,
                "entropy": 1.10,
                "error": "",
            },
        ]
        diagnostics = [{
            "item_id": "medQA:test:0",
            "dataset": "medQA",
            "split": "test",
            "linker": "metamap",
            "metrics": {"distractor_contrast_available": True},
        }]

        rows = intervention_by_diagnostic_rows(
            predictions,
            diagnostics,
            diagnostics_to_include=["distractor_contrast_available"],
            evidence_conditions=["distractor_contrast_evidence", "random_same_size_evidence", "shuffled_answer_evidence"],
        )
        contrast = next(row for row in rows if row["evidence_condition"] == "distractor_contrast_evidence")

        self.assertEqual(contrast["diagnostic_name"], "distractor_contrast_available")
        self.assertTrue(contrast["diagnostic_value"])
        self.assertEqual(contrast["condition"], "distractor_contrast_evidence")
        self.assertEqual(contrast["condition_type"], "realistic_or_contrastive")
        self.assertEqual(contrast["n"], 1)
        self.assertEqual(contrast["accuracy"], 1.0)
        self.assertEqual(contrast["delta_vs_no_umls"], 1.0)
        self.assertEqual(contrast["wrong_to_right"], 1)
        self.assertEqual(contrast["right_to_wrong"], 0)
        self.assertEqual(contrast["net_correction"], 1)
        self.assertAlmostEqual(contrast["mean_confidence"], 0.80)
        self.assertAlmostEqual(contrast["mean_confidence_delta"], 0.40)
        self.assertAlmostEqual(contrast["mean_margin_delta"], 0.40)
        self.assertAlmostEqual(contrast["mean_entropy_delta"], -0.60)
        self.assertEqual(contrast["accuracy_gap_vs_random"], 1.0)
        self.assertEqual(contrast["accuracy_gap_vs_shuffled"], 1.0)
        self.assertEqual(contrast["net_correction_gap_vs_random"], 1)
        self.assertEqual(contrast["net_correction_gap_vs_shuffled"], 1)

    def test_paired_intervention_effect_rows_count_corrections_and_control_gaps(self) -> None:
        predictions = [
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "MockLogits",
                "model": "mock-local-logits",
                "mode": "non_thinking",
                "condition": "no_umls",
                "prediction": "A",
                "is_correct": False,
                "confidence": 0.40,
                "margin": 0.10,
                "entropy": 1.20,
                "error": "",
            },
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "MockLogits",
                "model": "mock-local-logits",
                "mode": "non_thinking",
                "condition": "q_to_a_path_evidence",
                "prediction": "B",
                "is_correct": True,
                "confidence": 0.80,
                "margin": 0.50,
                "entropy": 0.60,
                "error": "",
            },
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "MockLogits",
                "model": "mock-local-logits",
                "mode": "non_thinking",
                "condition": "random_same_size_evidence",
                "prediction": "A",
                "is_correct": False,
                "confidence": 0.45,
                "margin": 0.15,
                "entropy": 1.10,
                "error": "",
            },
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "MockLogits",
                "model": "mock-local-logits",
                "mode": "non_thinking",
                "condition": "shuffled_answer_evidence",
                "prediction": "A",
                "is_correct": False,
                "confidence": 0.42,
                "margin": 0.11,
                "entropy": 1.00,
                "error": "",
            },
        ]
        diagnostics = [{
            "item_id": "medQA:test:0",
            "dataset": "medQA",
            "split": "test",
            "linker": "metamap",
            "metrics": {"gold_path_available": True},
        }]

        rows = paired_intervention_effect_rows(
            predictions,
            diagnostics,
            diagnostics_to_include=["gold_path_available"],
            evidence_conditions=["q_to_a_path_evidence", "random_same_size_evidence", "shuffled_answer_evidence"],
        )
        q_to_a = next(
            row
            for row in rows
            if row["condition"] == "q_to_a_path_evidence"
            and row["diagnostic_name"] == "gold_path_available"
        )
        all_q_to_a = next(
            row
            for row in rows
            if row["condition"] == "q_to_a_path_evidence"
            and row["diagnostic_name"] == "all"
        )

        self.assertEqual(q_to_a["condition_type"], "realistic")
        self.assertEqual(q_to_a["no_umls_correct"], 0)
        self.assertEqual(q_to_a["condition_correct"], 1)
        self.assertEqual(q_to_a["wrong_to_right"], 1)
        self.assertEqual(q_to_a["right_to_wrong"], 0)
        self.assertEqual(q_to_a["same_correct"], 0)
        self.assertEqual(q_to_a["same_wrong"], 0)
        self.assertEqual(q_to_a["net_correction"], 1)
        self.assertAlmostEqual(q_to_a["confidence_delta"], 0.40)
        self.assertAlmostEqual(q_to_a["margin_delta"], 0.40)
        self.assertAlmostEqual(q_to_a["entropy_delta"], -0.60)
        self.assertEqual(q_to_a["net_correction_gap_vs_random"], 1)
        self.assertEqual(q_to_a["net_correction_gap_vs_shuffled"], 1)
        self.assertEqual(all_q_to_a["diagnostic_value"], "all")

    def test_downstream_split_tables_keep_missing_uncertainty_as_na_and_add_control_gaps(self) -> None:
        predictions = [
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "Qwen",
                "model": "Qwen/Qwen3.5-4B",
                "condition": "no_umls",
                "condition_type": "baseline",
                "prediction": "A",
                "is_correct": False,
                "error": "",
            },
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "Qwen",
                "model": "Qwen/Qwen3.5-4B",
                "condition": "q_to_a_path_evidence",
                "condition_type": "realistic",
                "prediction": "B",
                "is_correct": True,
                "error": "",
            },
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "Qwen",
                "model": "Qwen/Qwen3.5-4B",
                "condition": "random_same_size_evidence",
                "condition_type": "negative_control",
                "prediction": "A",
                "is_correct": False,
                "error": "",
            },
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "Qwen",
                "model": "Qwen/Qwen3.5-4B",
                "condition": "shuffled_answer_evidence",
                "condition_type": "negative_control",
                "prediction": "A",
                "is_correct": False,
                "error": "",
            },
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "Qwen",
                "model": "Qwen/Qwen3.5-4B",
                "condition": "oracle_gold_evidence",
                "condition_type": "oracle",
                "prediction": "B",
                "is_correct": True,
                "error": "",
            },
        ]

        realistic = downstream_condition_summary_rows(predictions, table_kind="realistic")
        controls = downstream_condition_summary_rows(predictions, table_kind="controls")
        diagnostic_oracle = downstream_condition_summary_rows(predictions, table_kind="diagnostic_oracle")
        q_to_a = next(row for row in realistic if row["condition"] == "q_to_a_path_evidence")

        self.assertEqual(q_to_a["condition_type"], "realistic")
        self.assertEqual(q_to_a["accuracy"], 1.0)
        self.assertIsNone(q_to_a["mean_confidence"])
        self.assertEqual(q_to_a["accuracy_gap_vs_random"], 1.0)
        self.assertEqual(q_to_a["accuracy_gap_vs_shuffled"], 1.0)
        self.assertEqual([row["condition_type"] for row in controls], ["negative_control", "negative_control"])
        self.assertEqual(diagnostic_oracle[0]["condition_type"], "oracle")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "realistic.csv"
            md_path = root / "realistic.md"
            write_downstream_condition_summary_outputs(csv_path, md_path, realistic)
            self.assertIn("NA", md_path.read_text(encoding="utf-8"))

    def test_downstream_strict_accuracy_counts_malformed_as_incorrect(self) -> None:
        predictions = [
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "Qwen",
                "model": "Qwen/Qwen3.5-4B",
                "condition": "no_umls",
                "prediction": "A",
                "is_correct": True,
                "error": "",
            },
            {
                "item_id": "medQA:test:1",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "Qwen",
                "model": "Qwen/Qwen3.5-4B",
                "condition": "no_umls",
                "prediction": None,
                "is_correct": False,
                "error": "missing_prediction",
            },
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "Qwen",
                "model": "Qwen/Qwen3.5-4B",
                "condition": "q_to_a_path_evidence",
                "prediction": "B",
                "is_correct": True,
                "error": "",
            },
            {
                "item_id": "medQA:test:1",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "Qwen",
                "model": "Qwen/Qwen3.5-4B",
                "condition": "q_to_a_path_evidence",
                "prediction": None,
                "is_correct": False,
                "error": "missing_prediction",
            },
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "Qwen",
                "model": "Qwen/Qwen3.5-4B",
                "condition": "random_same_size_evidence",
                "prediction": "A",
                "is_correct": False,
                "error": "",
            },
            {
                "item_id": "medQA:test:1",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "Qwen",
                "model": "Qwen/Qwen3.5-4B",
                "condition": "random_same_size_evidence",
                "prediction": None,
                "is_correct": False,
                "error": "missing_prediction",
            },
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "Qwen",
                "model": "Qwen/Qwen3.5-4B",
                "condition": "shuffled_answer_evidence",
                "prediction": "A",
                "is_correct": False,
                "error": "",
            },
            {
                "item_id": "medQA:test:1",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "model_key": "Qwen",
                "model": "Qwen/Qwen3.5-4B",
                "condition": "shuffled_answer_evidence",
                "prediction": None,
                "is_correct": False,
                "error": "missing_prediction",
            },
        ]

        realistic = downstream_condition_summary_rows(predictions, table_kind="realistic")
        q_to_a = next(row for row in realistic if row["condition"] == "q_to_a_path_evidence")

        self.assertEqual(q_to_a["accuracy"], 1.0)
        self.assertEqual(q_to_a["strict_accuracy"], 0.5)
        self.assertEqual(q_to_a["strict_delta_vs_no_umls"], 0.0)
        self.assertEqual(q_to_a["strict_accuracy_gap_vs_random"], 0.5)
        self.assertEqual(q_to_a["strict_accuracy_gap_vs_shuffled"], 0.5)

    def test_intervention_output_writers_create_csv_and_markdown_tables(self) -> None:
        intervention_rows = [{
            "dataset": "medQA",
            "split": "test",
            "linker": "metamap",
            "model_key": "MockLogits",
            "model": "mock-local-logits",
            "mode": "",
            "diagnostic_name": "gold_path_available",
            "diagnostic_value": True,
            "condition": "q_to_a_path_evidence",
            "condition_type": "realistic",
            "evidence_condition": "q_to_a_path_evidence",
            "n": 1,
            "accuracy": 1.0,
            "delta_vs_no_umls": 1.0,
            "malformed_rate": 0.0,
            "wrong_to_right": 1,
            "right_to_wrong": 0,
            "net_correction": 1,
            "mean_confidence": 0.8,
            "mean_margin": 0.5,
            "mean_entropy": 0.6,
            "mean_confidence_delta": 0.4,
            "mean_margin_delta": 0.4,
            "mean_entropy_delta": -0.6,
            "accuracy_gap_vs_random": 1.0,
            "accuracy_gap_vs_shuffled": 1.0,
            "net_correction_gap_vs_random": 1,
            "net_correction_gap_vs_shuffled": 1,
        }]
        paired_rows = [{
            "dataset": "medQA",
            "split": "test",
            "linker": "metamap",
            "model_key": "MockLogits",
            "model": "mock-local-logits",
            "diagnostic_name": "gold_path_available",
            "diagnostic_value": True,
            "condition": "q_to_a_path_evidence",
            "condition_type": "realistic",
            "evidence_condition": "q_to_a_path_evidence",
            "n": 1,
            "no_umls_correct": 0,
            "condition_correct": 1,
            "wrong_to_right": 1,
            "right_to_wrong": 0,
            "same_correct": 0,
            "same_wrong": 0,
            "net_correction": 1,
            "confidence_delta": 0.4,
            "margin_delta": 0.4,
            "entropy_delta": -0.6,
            "net_correction_gap_vs_random": 1,
            "net_correction_gap_vs_shuffled": 1,
        }]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intervention_csv = root / "intervention.csv"
            intervention_md = root / "intervention.md"
            paired_csv = root / "paired.csv"
            paired_md = root / "paired.md"
            summary_csv = root / "main_claims.csv"
            summary_md = root / "main_claims.md"

            write_intervention_by_diagnostic_outputs(intervention_csv, intervention_md, intervention_rows)
            write_paired_intervention_effect_outputs(paired_csv, paired_md, paired_rows)
            write_main_claim_summary_outputs(summary_csv, summary_md, main_claim_summary_rows())

            intervention_text = intervention_csv.read_text(encoding="utf-8")
            paired_text = paired_md.read_text(encoding="utf-8")
            summary_text = summary_csv.read_text(encoding="utf-8")

        self.assertIn("accuracy_gap_vs_random", intervention_text)
        self.assertIn("condition_type", intervention_text)
        self.assertIn("q_to_a_path_evidence", paired_text)
        self.assertIn("Controlled UMLS interventions", summary_text)


if __name__ == "__main__":
    unittest.main()
