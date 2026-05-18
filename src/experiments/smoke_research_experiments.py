from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments import exp1_evidence_availability, exp2_answer_grounding, exp3_retrieval_faithfulness
from src.experiments.diagnostic_uncertainty import summarize_diagnostics, write_csv as write_uncertainty_csv
from src.experiments.loaders import iter_jsonl
from src.model_sweeps.conditioned_metrics import conditioned_summary_rows, write_csv as write_conditioned_csv
from src.model_sweeps.local_logits_sweep import (
    aggregate_trials_by_item,
    default_output_paths as logits_default_output_paths,
    load_mock_model,
    run_logits_sweep,
    write_jsonl,
    write_summary_csv,
)
from src.model_sweeps.qwen_vllm_sweep import build_request_payload
from src.experiments.cli_utils import default_input_path, default_reference_path, load_items


DEFAULT_OUTPUT_DIR = Path("outputs") / "diagnostics" / "smoke_research_questions"


class SmokeFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def read_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fh:
        return fh.readline().strip().split(",")


def validate_jsonl_rows(path: Path, expected_rows: int, required_columns: set[str]) -> None:
    rows = read_jsonl(path)
    require(len(rows) == expected_rows, f"{path} expected {expected_rows} rows, found {len(rows)}")
    missing = [required_columns - set(row) for row in rows]
    missing = [item for item in missing if item]
    require(not missing, f"{path} rows missing required columns: {missing[:3]}")


def validate_summary(path: Path, required_columns: set[str]) -> None:
    require(path.exists(), f"Missing summary {path}")
    header = set(read_csv_header(path))
    require(required_columns <= header, f"{path} missing summary columns: {sorted(required_columns - header)}")


def diagnostic_args(
    dataset: str,
    split: str,
    linker: str,
    limit: int,
    output_jsonl: Path,
    summary_csv: Path,
) -> Namespace:
    return Namespace(
        dataset=dataset,
        split=split,
        linker=linker,
        input=None,
        reference=None,
        output_jsonl=output_jsonl,
        summary_csv=summary_csv,
        limit=limit,
        dry_run=False,
        metadata_zip=Path("data/UMLS/cui_detail.tsv.zip"),
        triplets_zip=Path("data/UMLS/triplets.csv.zip"),
    )


def run_diagnostic_case(
    name: str,
    runner: Callable[[Namespace], list[dict]],
    dataset: str,
    split: str,
    linker: str,
    expected_rows: int,
    required_columns: set[str],
    summary_columns: set[str],
    output_dir: Path,
    limit: int = 2,
) -> dict:
    output_jsonl = output_dir / f"{name}.jsonl"
    summary_csv = output_dir / f"{name}.summary.csv"
    rows = runner(diagnostic_args(dataset, split, linker, limit, output_jsonl, summary_csv))
    require(len(rows) == expected_rows, f"{name} runner returned {len(rows)}, expected {expected_rows}")
    validate_jsonl_rows(output_jsonl, expected_rows, required_columns)
    validate_summary(summary_csv, summary_columns)
    return {
        "name": name,
        "kind": "diagnostic",
        "dataset": dataset,
        "split": split,
        "linker": linker,
        "rows": len(rows),
        "jsonl": str(output_jsonl),
        "summary_csv": str(summary_csv),
        "status": "ok",
    }


def run_diagnostic_smokes(output_dir: Path, limit: int) -> tuple[list[dict], list[dict]]:
    manifest: list[dict] = []
    all_rows: list[dict] = []
    cases = [
        (
            "exp1_medQA_test_metamap",
            exp1_evidence_availability.run,
            "medQA",
            "test",
            "metamap",
            limit * 5,
            {"item_id", "view", "gold_answer_grounded", "evidence_token_estimate", "experiment"},
            {"experiment", "condition", "mean_gold_answer_grounded", "mean_evidence_token_estimate"},
        ),
        (
            "exp2_medQA_test_metamap",
            exp2_answer_grounding.run,
            "medQA",
            "test",
            "metamap",
            limit,
            {"item_id", "question_grounded", "correct_answer_grounded", "option_cui_counts", "experiment"},
            {"experiment", "mean_correct_answer_grounded", "mean_question_grounded"},
        ),
        (
            "exp3_medQA_test_metamap",
            exp3_retrieval_faithfulness.run,
            "medQA",
            "test",
            "metamap",
            limit * 4,
            {"item_id", "method", "gold_path_available", "num_retrieved_nodes", "experiment"},
            {"experiment", "condition", "mean_gold_path_available", "mean_num_retrieved_nodes"},
        ),
        (
            "exp1_medMCQA_test_spacy",
            exp1_evidence_availability.run,
            "medMCQA",
            "test",
            "spacy",
            limit * 5,
            {"item_id", "view", "gold_answer_grounded", "evidence_token_estimate", "experiment"},
            {"experiment", "condition", "mean_gold_answer_grounded", "mean_evidence_token_estimate"},
        ),
        (
            "exp2_medMCQA_test_spacy",
            exp2_answer_grounding.run,
            "medMCQA",
            "test",
            "spacy",
            limit,
            {"item_id", "question_grounded", "correct_answer_grounded", "option_cui_counts", "experiment"},
            {"experiment", "mean_correct_answer_grounded", "mean_question_grounded"},
        ),
        (
            "exp3_medMCQA_test_spacy",
            exp3_retrieval_faithfulness.run,
            "medMCQA",
            "test",
            "spacy",
            limit * 4,
            {"item_id", "method", "gold_path_available", "num_retrieved_nodes", "experiment"},
            {"experiment", "condition", "mean_gold_path_available", "mean_num_retrieved_nodes"},
        ),
        (
            "exp1_pubmedQA_test_metamap",
            exp1_evidence_availability.run,
            "pubmedQA",
            "test",
            "metamap",
            limit * 5,
            {"item_id", "view", "gold_answer_grounded", "evidence_token_estimate", "experiment"},
            {"experiment", "condition", "mean_gold_answer_grounded", "mean_evidence_token_estimate"},
        ),
        (
            "exp2_pubmedQA_test_spacy",
            exp2_answer_grounding.run,
            "pubmedQA",
            "test",
            "spacy",
            limit,
            {"item_id", "question_grounded", "answer_option_metrics_applicable", "experiment"},
            {"experiment", "mean_answer_option_metrics_applicable", "mean_question_grounded"},
        ),
    ]
    for case in cases:
        record = run_diagnostic_case(*case, output_dir=output_dir, limit=limit)
        manifest.append(record)
        all_rows.extend(read_jsonl(Path(record["jsonl"])))
    return manifest, all_rows


def run_uncertainty_smoke(output_dir: Path, diagnostic_rows: list[dict]) -> dict:
    group_keys, summary_rows = summarize_diagnostics(diagnostic_rows, iterations=50, seed=17)
    output_csv = output_dir / "diagnostic_uncertainty.summary.csv"
    write_uncertainty_csv(output_csv, group_keys, summary_rows)
    validate_summary(output_csv, {"metric", "metric_type", "mean", "ci_low", "ci_high"})
    require(len(summary_rows) > 0, "diagnostic uncertainty summary produced no rows")
    return {
        "name": "diagnostic_uncertainty",
        "kind": "uncertainty",
        "rows": len(summary_rows),
        "summary_csv": str(output_csv),
        "status": "ok",
    }


def run_qwen_dry_run_smoke(limit: int) -> dict:
    items = load_items(
        "medQA",
        "test",
        "metamap",
        input_path=default_input_path("medQA", "test", "metamap"),
        reference_path=default_reference_path("medQA", "test"),
        limit=limit,
    )
    payload = build_request_payload(items[0], "Qwen/Qwen3.5-0.8B", "non_thinking", max_tokens=256)
    require(payload["chat_template_kwargs"] == {"enable_thinking": False}, "Qwen payload missing thinking toggle")
    require(payload["max_tokens"] == 256, "Qwen payload max_tokens mismatch")
    require("Return JSON" in payload["messages"][0]["content"], "Qwen prompt missing JSON instruction")
    return {
        "name": "qwen_vllm_payload",
        "kind": "model_payload",
        "model": "Qwen/Qwen3.5-0.8B",
        "rows": limit,
        "status": "ok",
    }


def run_mock_logits_smoke(output_dir: Path, limit: int) -> tuple[dict, list[dict]]:
    items = load_items(
        "medQA",
        "test",
        "metamap",
        input_path=default_input_path("medQA", "test", "metamap"),
        reference_path=default_reference_path("medQA", "test"),
        limit=limit,
    )
    model_bundle = load_mock_model()
    logits_dir = output_dir / "mock_logits"
    output_jsonl, summary_csv = logits_default_output_paths("MockLogits", "medQA", "test", logits_dir)
    rows = run_logits_sweep(
        items,
        model_bundle=model_bundle,
        num_samples=2,
        num_shuffles=3,
        seed=42,
        output_jsonl=None,
    )
    expected_rows = limit * 2 * 3
    require(len(rows) == expected_rows, f"mock logits expected {expected_rows} rows, found {len(rows)}")
    write_jsonl(output_jsonl, rows)
    write_summary_csv(summary_csv, rows)
    validate_jsonl_rows(
        output_jsonl,
        expected_rows,
        {"item_id", "prediction", "confidence", "option_probabilities", "option_source_labels"},
    )
    validate_summary(
        summary_csv,
        {"accuracy", "accuracy_ci_low", "ece", "mean_confidence", "mean_vote_entropy"},
    )
    return {
        "name": "mock_logits_uncertainty",
        "kind": "model_logits",
        "dataset": "medQA",
        "split": "test",
        "rows": len(rows),
        "jsonl": str(output_jsonl),
        "summary_csv": str(summary_csv),
        "status": "ok",
    }, aggregate_trials_by_item(rows)


def run_conditioned_smoke(output_dir: Path, prediction_rows: list[dict], diagnostic_rows: list[dict]) -> dict:
    grounding_rows = [
        row
        for row in diagnostic_rows
        if row.get("experiment") in {"answer_grounding", "exp2_answer_grounding"}
        and row.get("dataset") == "medQA"
        and row.get("linker") == "metamap"
    ]
    rows = conditioned_summary_rows(
        prediction_rows,
        grounding_rows,
        conditions=["correct_answer_grounded", "all_options_grounded"],
    )
    require(len(rows) > 0, "conditioned summary produced no rows")
    output_csv = output_dir / "conditioned_mock_logits.summary.csv"
    write_conditioned_csv(output_csv, rows)
    validate_summary(output_csv, {"condition", "condition_value", "accuracy", "mean_confidence", "mean_nll"})
    return {
        "name": "conditioned_mock_logits",
        "kind": "conditioned_model_metrics",
        "rows": len(rows),
        "summary_csv": str(output_csv),
        "status": "ok",
    }


def write_manifest(output_dir: Path, records: list[dict], limit: int) -> Path:
    manifest = {
        "status": "ok",
        "limit": limit,
        "records": records,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run small-sample smoke checks for research-question experiments.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require(args.limit > 0, "--limit must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, diagnostic_rows = run_diagnostic_smokes(args.output_dir, args.limit)
    manifest.append(run_uncertainty_smoke(args.output_dir, diagnostic_rows))
    manifest.append(run_qwen_dry_run_smoke(limit=1))
    logits_record, prediction_rows = run_mock_logits_smoke(args.output_dir, args.limit)
    manifest.append(logits_record)
    manifest.append(run_conditioned_smoke(args.output_dir, prediction_rows, diagnostic_rows))
    manifest_path = write_manifest(args.output_dir, manifest, args.limit)
    print(json.dumps({
        "status": "ok",
        "manifest": str(manifest_path),
        "checks": len(manifest),
    }, indent=2))


if __name__ == "__main__":
    main()
