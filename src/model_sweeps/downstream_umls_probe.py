from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.cli_utils import collect_cuis, default_input_path, default_reference_path, load_items
from src.experiments.evidence_cards import build_evidence_card_details, condition_metadata, condition_type_for_condition
from src.experiments.paper_config import DEFAULT_CONFIG_PATH, load_paper_experiment_config
from src.experiments.schemas import QAItem
from src.experiments.umls_index import CuiMetadataIndex, RelationIndex
from src.model_sweeps.local_logits_sweep import (
    aggregate_trials_by_item,
    load_local_model,
    load_mock_model,
    score_item_once,
    summarize_prediction_rows,
    unload_model,
)
from src.model_sweeps.qwen_vllm_sweep import DEFAULT_BASE_URL, VLLMClient, load_existing_rows, predict_item


DEFAULT_QWEN_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_MEDGEMMA_MODEL_KEY = "MedGemma-1.5-4B-it"
DEFAULT_MEDGEMMA_MODEL_ID = "google/medgemma-1.5-4b-it"


def normalize_limit(limit: int | None) -> int | None:
    if limit is None or limit <= 0:
        return None
    return limit


def _annotate_probe_row(row: dict, condition: str, card_details=None) -> dict:
    spec = condition_metadata(condition)
    out = dict(row)
    out["condition"] = condition
    out["condition_type"] = condition_type_for_condition(condition)
    out["evidence_condition"] = condition
    out["prompt_condition"] = condition
    out.setdefault("confidence", None)
    out.setdefault("margin", None)
    out.setdefault("entropy", None)
    out["malformed"] = out.get("prediction") is None or bool(out.get("error"))
    out["evidence_view"] = spec.evidence_view if spec.evidence_view != "none" else ""
    out["retrieval_method"] = spec.retrieval_method or ""
    out["oracle"] = bool(getattr(card_details, "oracle", False))
    out["oracle_gold_cui_source"] = getattr(card_details, "oracle_gold_cui_source", "")
    out["oracle_lookup_success"] = bool(getattr(card_details, "oracle_lookup_success", False))
    out["oracle_num_candidate_cuis"] = int(getattr(card_details, "oracle_num_candidate_cuis", 0))
    out["oracle_ambiguous"] = bool(getattr(card_details, "oracle_ambiguous", False))
    out["oracle_selected_cui_source"] = getattr(card_details, "oracle_selected_cui_source", "")
    out["option_evidence_symmetric"] = bool(getattr(card_details, "option_evidence_symmetric", False))
    out["uses_gold_label_for_evidence"] = bool(getattr(card_details, "uses_gold_label_for_evidence", False))
    out["q_to_a_depth_semantics"] = getattr(card_details, "q_to_a_depth_semantics", "")
    out["q_to_a_depth"] = int(getattr(card_details, "q_to_a_depth", 0))
    out["q_to_a_path_found"] = bool(getattr(card_details, "q_to_a_path_found", False))
    out["q_to_a_path_node_count"] = int(getattr(card_details, "q_to_a_path_node_count", 0))
    out["q_to_a_path_edge_count"] = int(getattr(card_details, "q_to_a_path_edge_count", 0))
    out["q_to_a_answer_cui_count"] = int(getattr(card_details, "q_to_a_answer_cui_count", 0))
    out["q_to_a_answer_cui_covered_count"] = int(getattr(card_details, "q_to_a_answer_cui_covered_count", 0))
    out["q_to_a_path_selection_policy"] = getattr(card_details, "q_to_a_path_selection_policy", "")
    return out


def _stable_condition_seed(seed: int, condition: str) -> int:
    digest = hashlib.sha256(condition.encode("utf-8")).hexdigest()
    return seed + int(digest[:8], 16)


def _capped_tasks(
    tasks: list[tuple[str, QAItem, Any]],
    max_items_per_condition: int | None,
    seed: int,
    condition: str,
) -> list[tuple[str, QAItem, Any]]:
    if max_items_per_condition is None or max_items_per_condition <= 0:
        return tasks
    if len(tasks) <= max_items_per_condition:
        return tasks
    indexed = list(enumerate(tasks))
    rng = random.Random(_stable_condition_seed(seed, condition))
    rng.shuffle(indexed)
    selected_indexes = {index for index, _ in indexed[:max_items_per_condition]}
    return [task for index, task in enumerate(tasks) if index in selected_indexes]


def _capped_items(items: list[QAItem], max_items: int | None, seed: int) -> list[QAItem]:
    if max_items is None or max_items <= 0:
        return items
    if len(items) <= max_items:
        return items
    indexed = list(enumerate(items))
    rng = random.Random(_stable_condition_seed(seed, "global_items"))
    rng.shuffle(indexed)
    selected_indexes = {index for index, _ in indexed[:max_items]}
    return [item for index, item in enumerate(items) if index in selected_indexes]


def run_medgemma_probe(
    items: list[QAItem],
    model_bundle: dict[str, Any],
    conditions: tuple[str, ...],
    metadata: CuiMetadataIndex,
    relations: RelationIndex,
    num_samples: int = 1,
    num_shuffles: int = 1,
    seed: int = 42,
) -> list[dict]:
    rows: list[dict] = []
    for condition in conditions:
        for item in items:
            card = build_evidence_card_details(item, condition, metadata, relations)
            for run_index in range(max(1, num_samples)):
                for shuffle_index in range(max(1, num_shuffles)):
                    row_seed = seed + run_index * 1000 + shuffle_index
                    row = score_item_once(
                        item,
                        model_bundle,
                        run_index=run_index,
                        shuffle_index=shuffle_index,
                        seed=row_seed,
                        shuffle_options=num_shuffles > 1,
                        prompt_condition=condition,
                        evidence=card.text,
                    )
                    rows.append(_annotate_probe_row(row, condition, card))
    return rows


def run_qwen_probe(
    items: list[QAItem],
    client: VLLMClient,
    model: str,
    mode: str,
    max_tokens: int,
    conditions: tuple[str, ...],
    metadata: CuiMetadataIndex,
    relations: RelationIndex,
    max_retries: int = 2,
    retry_sleep_s: float = 2.0,
    decoding_overrides: dict | None = None,
    thinking_token_budget: int | None = None,
    output_jsonl: Path | None = None,
    existing_rows: list[dict] | None = None,
    concurrency: int = 1,
    require_q_to_a_path: bool = False,
    max_items_per_condition: int | None = None,
    selection_seed: int = 42,
) -> list[dict]:
    rows: list[dict] = list(existing_rows or [])
    completed = {
        (
            str(row.get("item_id", "")),
            str(row.get("dataset", "")),
            str(row.get("split", "")),
            str(row.get("linker", "")),
            str(row.get("model", "")),
            str(row.get("mode", "")),
            str(row.get("condition", row.get("prompt_condition", ""))),
        )
        for row in rows
    }

    source_items = (
        items
        if require_q_to_a_path
        else _capped_items(items, max_items_per_condition, selection_seed)
    )

    pending: list[tuple[str, QAItem, Any]] = []
    for condition in conditions:
        condition_tasks: list[tuple[str, QAItem, Any]] = []
        for item in source_items:
            completed_key = (item.item_id, item.dataset, item.split, item.linker, model, mode, condition)
            if completed_key in completed and not require_q_to_a_path:
                condition_tasks.append((condition, item, None))
                continue
            card = build_evidence_card_details(item, condition, metadata, relations)
            if (
                require_q_to_a_path
                and card.q_to_a_depth_semantics
                and not card.q_to_a_path_found
            ):
                continue
            condition_tasks.append((condition, item, card))
        if require_q_to_a_path:
            condition_tasks = _capped_tasks(
                condition_tasks,
                max_items_per_condition=max_items_per_condition,
                seed=selection_seed,
                condition=condition,
            )
        pending.extend(
            (condition, item, card)
            for condition, item, card in condition_tasks
            if (item.item_id, item.dataset, item.split, item.linker, model, mode, condition) not in completed
        )

    output_fh = None
    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_fh = output_jsonl.open("a" if existing_rows else "w", encoding="utf-8")

    write_lock = threading.Lock()

    def _process(condition: str, item: QAItem, card) -> dict:
        row = predict_item(
            item,
            client,
            model=model,
            mode=mode,
            max_tokens=max_tokens,
            evidence=card.text,
            condition=condition,
            max_retries=max_retries,
            retry_sleep_s=retry_sleep_s,
            decoding_overrides=decoding_overrides,
            thinking_token_budget=thinking_token_budget,
        )
        return _annotate_probe_row(row, condition, card)

    def _flush(annotated: dict) -> None:
        rows.append(annotated)
        completed.add((
            str(annotated.get("item_id", "")),
            str(annotated.get("dataset", "")),
            str(annotated.get("split", "")),
            str(annotated.get("linker", "")),
            str(annotated.get("model", "")),
            str(annotated.get("mode", "")),
            str(annotated.get("condition", annotated.get("prompt_condition", ""))),
        ))
        if output_fh is not None:
            output_fh.write(json.dumps(annotated, ensure_ascii=False) + "\n")
            output_fh.flush()

    try:
        if concurrency <= 1:
            for condition, item, card in pending:
                _flush(_process(condition, item, card))
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {executor.submit(_process, c, i, card): (c, i) for c, i, card in pending}
                for future in as_completed(futures):
                    annotated = future.result()
                    with write_lock:
                        _flush(annotated)
    finally:
        if output_fh is not None:
            output_fh.close()
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mean_available(rows: list[dict], metric: str) -> float | None:
    values = [
        float(row[metric])
        for row in rows
        if isinstance(row.get(metric), (int, float))
    ]
    if not values:
        return None
    return sum(values) / len(values)


def write_probe_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple, list[dict]] = {}
    aggregate_source = aggregate_trials_by_item(rows) if any("run_index" in row for row in rows) else rows
    for row in aggregate_source:
        key = (
            row.get("model_key", ""),
            row.get("model", ""),
            row.get("dataset", ""),
            row.get("split", ""),
            row.get("linker", ""),
            row.get("condition", row.get("prompt_condition", "")),
            row.get("condition_type", condition_type_for_condition(row.get("condition", row.get("prompt_condition", "")))),
        )
        groups.setdefault(key, []).append(row)
    fieldnames = [
        "model_key",
        "model",
        "dataset",
        "split",
        "linker",
        "condition",
        "condition_type",
        "n",
        "valid",
        "malformed",
        "malformed_rate",
        "accuracy",
        "mean_confidence",
        "mean_margin",
        "mean_entropy",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for key, group in sorted(groups.items()):
            model_key, model, dataset, split, linker, condition, condition_type = key
            summary = summarize_prediction_rows(group)
            summary["mean_confidence"] = _mean_available(group, "confidence")
            summary["mean_margin"] = _mean_available(group, "margin")
            summary["mean_entropy"] = _mean_available(group, "entropy")
            row = {
                "model_key": model_key,
                "model": model,
                "dataset": dataset,
                "split": split,
                "linker": linker,
                "condition": condition,
                "condition_type": condition_type,
                **summary,
            }
            writer.writerow({
                name: f"{value:.6f}" if isinstance(value, float) else value
                for name, value in row.items()
                if name in fieldnames
            })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run controlled downstream probes over UMLS evidence conditions.")
    parser.add_argument("--dataset", default="medQA")
    parser.add_argument("--split", default="test")
    parser.add_argument("--linker", choices=["metamap", "spacy"], default="metamap")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--metadata-zip", type=Path, default=Path("data/UMLS/cui_detail.tsv.zip"))
    parser.add_argument("--triplets-zip", type=Path, default=Path("data/UMLS/triplets.csv.zip"))
    parser.add_argument("--relation-scope", choices=["item_cuis", "full"], default="item_cuis",
                        help="Load only QA-linked relations or the full relation index.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--condition", action="append", default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--item-id-file", type=Path, default=None,
                        help="Optional JSONL/CSV/text file of item_id values to keep before limit is applied.")
    parser.add_argument("--backend", choices=["medgemma", "qwen"], required=True)
    parser.add_argument("--mock-model", action="store_true")
    parser.add_argument("--model-key", default=DEFAULT_MEDGEMMA_MODEL_KEY)
    parser.add_argument("--model-id", default=DEFAULT_MEDGEMMA_MODEL_ID)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32", "auto"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-shuffles", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-mode", choices=["non_thinking", "thinking"], default="non_thinking")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--thinking-token-budget", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent vLLM requests.")
    parser.add_argument("--require-q-to-a-path", action="store_true",
                        help="For Q-to-answer evidence conditions, skip model generation when no path is present.")
    parser.add_argument("--max-items-per-condition", type=int, default=0,
                        help="Deterministically cap generated items per evidence condition after filtering.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_paper_experiment_config(args.config)
    conditions = tuple(args.condition or config.downstream_probe_conditions)
    input_path = args.input or default_input_path(args.dataset, args.split, args.linker)
    reference = args.reference or default_reference_path(args.dataset, args.split)
    items = load_items(
        args.dataset,
        args.split,
        args.linker,
        input_path=input_path,
        reference_path=reference if reference.exists() else None,
        limit=normalize_limit(args.limit),
        item_id_file=args.item_id_file,
    )
    if args.dry_run:
        print(json.dumps({
            "backend": args.backend,
            "dataset": args.dataset,
            "split": args.split,
            "linker": args.linker,
            "items": len(items),
            "item_id_file": str(args.item_id_file) if args.item_id_file else "",
            "conditions": conditions,
            "triplets_zip": str(args.triplets_zip),
            "relation_scope": args.relation_scope,
            "qwen_model": args.qwen_model,
            "qwen_mode": args.qwen_mode,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "presence_penalty": args.presence_penalty,
            "thinking_token_budget": args.thinking_token_budget,
            "seed": args.seed,
            "require_q_to_a_path": args.require_q_to_a_path,
            "max_items_per_condition": args.max_items_per_condition,
            "enable_thinking": args.qwen_mode == "thinking",
            "output_jsonl": str(args.output_jsonl),
            "summary_csv": str(args.summary_csv),
        }, indent=2))
        return

    metadata = CuiMetadataIndex.from_zip(args.metadata_zip)
    keep_cuis = None if args.relation_scope == "full" else collect_cuis(items)
    relations = RelationIndex.from_zip(args.triplets_zip, keep_cuis=keep_cuis)
    if args.backend == "qwen":
        client = VLLMClient(args.base_url, api_key=args.api_key, timeout_s=args.timeout_s)
        decoding_overrides = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "presence_penalty": args.presence_penalty,
            "seed": args.seed,
        }
        rows = run_qwen_probe(
            items,
            client,
            model=args.qwen_model,
            mode=args.qwen_mode,
            max_tokens=args.max_tokens,
            conditions=conditions,
            metadata=metadata,
            relations=relations,
            decoding_overrides=decoding_overrides,
            thinking_token_budget=args.thinking_token_budget,
            output_jsonl=args.output_jsonl,
            existing_rows=load_existing_rows(args.output_jsonl) if args.resume else [],
            concurrency=args.concurrency,
            require_q_to_a_path=args.require_q_to_a_path,
            max_items_per_condition=normalize_limit(args.max_items_per_condition),
            selection_seed=args.seed,
        )
    elif args.mock_model:
        rows = run_medgemma_probe(
            items,
            load_mock_model(),
            conditions=conditions,
            metadata=metadata,
            relations=relations,
            num_samples=args.num_samples,
            num_shuffles=args.num_shuffles,
            seed=args.seed,
        )
    else:
        model_bundle = load_local_model(
            args.model_key,
            model_id=args.model_id,
            device_map=args.device_map,
            dtype=args.dtype,
        )
        try:
            rows = run_medgemma_probe(
                items,
                model_bundle,
                conditions=conditions,
                metadata=metadata,
                relations=relations,
                num_samples=args.num_samples,
                num_shuffles=args.num_shuffles,
                seed=args.seed,
            )
        finally:
            unload_model(model_bundle)
    write_jsonl(args.output_jsonl, rows)
    write_probe_summary(args.summary_csv, rows)


if __name__ == "__main__":
    main()
