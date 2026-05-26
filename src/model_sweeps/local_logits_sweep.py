from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.cli_utils import default_input_path, default_reference_path, load_items
from src.experiments.schemas import QAItem
from src.model_sweeps.uncertainty_metrics import (
    group_rows,
    majority_vote,
    prediction_metrics,
    softmax_scores,
    summarize_prediction_rows,
    vote_entropy,
)


DEFAULT_MODEL_KEY = "MedGemma-1.5-4B-it"
MODEL_CONFIGS = {
    "MedGemma-1.5-4B-it": {
        "model_id": "google/medgemma-1.5-4b-it",
        "requires_processor": True,
    },
    "Qwen3-VL-4B-Instruct": {
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "requires_processor": True,
    },
}


def model_slug(value: str) -> str:
    return value.replace("/", "__").replace(".", "_").replace(" ", "_")


def normalize_limit(limit: int | None) -> int | None:
    if limit is None or limit <= 0:
        return None
    return limit


def build_scoring_prompt(
    item: QAItem,
    options: dict[str, str] | None = None,
    evidence: str | None = None,
) -> str:
    resolved_options = options or item.options
    lines = [
        "You are answering a medical multiple-choice question.",
        "Choose exactly one option.",
        "",
        "Question:",
        item.question,
        "",
        "Options:",
    ]
    for key, value in resolved_options.items():
        lines.append(f"{key}. {value}")
    if evidence:
        lines.extend(["", "UMLS evidence:", evidence])
    lines.extend(["", "Answer with one letter only.", "", "Answer:"])
    return "\n".join(lines)


def shuffled_option_view(
    options: dict[str, str],
    gold: str | None,
    rng: random.Random,
    shuffle: bool,
) -> tuple[dict[str, str], str | None, dict[str, str]]:
    labels = list(options)
    entries = [(label, options[label]) for label in labels]
    if shuffle:
        rng.shuffle(entries)

    shuffled_options: dict[str, str] = {}
    option_source_labels: dict[str, str] = {}
    effective_gold: str | None = None
    for new_label, (source_label, text) in zip(labels, entries):
        shuffled_options[new_label] = text
        option_source_labels[new_label] = source_label
        if gold is not None and source_label == gold:
            effective_gold = new_label
    return shuffled_options, effective_gold, option_source_labels


def get_label_token_ids(tokenizer: Any, labels: list[str]) -> dict[str, list[int]]:
    label_token_ids: dict[str, list[int]] = {}
    for label in labels:
        token_ids: set[int] = set()
        for variant in (label, f" {label}"):
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if len(encoded) == 1:
                token_ids.add(int(encoded[0]))
        if not token_ids:
            encoded = tokenizer.encode(label, add_special_tokens=False)
            if encoded:
                token_ids.add(int(encoded[-1]))
        label_token_ids[label] = sorted(token_ids)
    return label_token_ids


def _torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float32":
        return torch_module.float32
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    return torch_module.float16


def load_local_model(
    model_key: str,
    model_id: str | None = None,
    device_map: str = "auto",
    dtype: str = "bfloat16",
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Local logits sweeps require torch and transformers. Install the project model extras first."
        ) from exc

    resolved_model_id = model_id or MODEL_CONFIGS.get(model_key, {}).get("model_id") or model_key
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(resolved_model_id, trust_remote_code=True)
    except Exception:
        processor = None

    tokenizer = getattr(processor, "tokenizer", None) if processor is not None else None
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(resolved_model_id, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    common_kwargs = {
        "device_map": device_map,
        "torch_dtype": _torch_dtype(torch, dtype),
        "trust_remote_code": True,
    }
    try:
        model = AutoModelForImageTextToText.from_pretrained(resolved_model_id, **common_kwargs)
        model_loader = "AutoModelForImageTextToText"
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(resolved_model_id, **common_kwargs)
        model_loader = "AutoModelForCausalLM"
    model.eval()

    return {
        "model_key": model_key,
        "model_id": resolved_model_id,
        "model": model,
        "tokenizer": tokenizer,
        "processor": processor,
        "model_loader": model_loader,
    }


def load_mock_model(model_key: str = "MockLogits", model_id: str = "mock-local-logits") -> dict[str, Any]:
    return {
        "model_key": model_key,
        "model_id": model_id,
        "model": None,
        "tokenizer": None,
        "processor": None,
        "model_loader": "mock",
        "mock_scores": True,
    }


def unload_model(model_bundle: dict[str, Any] | None) -> None:
    if model_bundle is None:
        return
    try:
        import torch
    except ImportError:
        torch = None
    model_bundle.clear()
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_model_inputs(model_bundle: dict[str, Any], prompt: str, image: Any = None) -> Any:
    processor = model_bundle.get("processor")
    tokenizer = model_bundle["tokenizer"]
    if image is not None and processor is not None:
        try:
            return processor(text=prompt, images=image, return_tensors="pt")
        except Exception:
            pass
    if processor is not None:
        try:
            return processor(text=prompt, return_tensors="pt")
        except Exception:
            pass
    return tokenizer(prompt, return_tensors="pt")


def move_inputs_to_device(inputs: Any, model: Any) -> dict[str, Any]:
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def label_scores_from_logits(logits: Any, label_token_ids: dict[str, list[int]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    vocab_size = logits.shape[-1]
    for label, token_ids in label_token_ids.items():
        token_scores = [
            float(logits[0, token_id].item())
            for token_id in token_ids
            if token_id < vocab_size
        ]
        scores[label] = max(token_scores) if token_scores else float("-inf")
    return scores


def score_answer_labels(
    model_bundle: dict[str, Any],
    prompt: str,
    labels: list[str],
    image: Any = None,
) -> dict:
    if model_bundle.get("mock_scores"):
        label_scores = {
            label: float(len(labels) - index)
            for index, label in enumerate(labels)
        }
        probabilities = softmax_scores(label_scores)
        metrics = prediction_metrics(probabilities)
        return {
            "label_token_ids": {label: [] for label in labels},
            "label_scores": label_scores,
            "label_probabilities": probabilities,
            "prediction": metrics["prediction"],
        }

    import torch

    model = model_bundle["model"]
    inputs = prepare_model_inputs(model_bundle, prompt=prompt, image=image)
    inputs = move_inputs_to_device(inputs, model)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    label_token_ids = get_label_token_ids(model_bundle["tokenizer"], labels)
    label_scores = label_scores_from_logits(logits, label_token_ids)
    probabilities = softmax_scores(label_scores)
    metrics = prediction_metrics(probabilities)
    return {
        "label_token_ids": label_token_ids,
        "label_scores": label_scores,
        "label_probabilities": probabilities,
        "prediction": metrics["prediction"],
    }


def score_item_once(
    item: QAItem,
    model_bundle: dict[str, Any],
    run_index: int,
    shuffle_index: int,
    seed: int,
    shuffle_options: bool,
    prompt_condition: str = "no_umls",
    evidence: str | None = None,
) -> dict:
    rng = random.Random(seed)
    options, effective_gold, option_source_labels = shuffled_option_view(
        item.options,
        item.gold,
        rng=rng,
        shuffle=shuffle_options,
    )
    labels = list(options)
    prompt = build_scoring_prompt(item, options=options, evidence=evidence)
    started = time.time()
    try:
        score = score_answer_labels(model_bundle, prompt=prompt, labels=labels)
        elapsed = time.time() - started
        probabilities = score["label_probabilities"]
        metrics = prediction_metrics(probabilities, effective_gold)
        prediction = metrics["prediction"]
        prediction_source_label = option_source_labels.get(prediction) if prediction is not None else None
        return {
            "item_id": item.item_id,
            "dataset": item.dataset,
            "split": item.split,
            "linker": item.linker,
            "model_key": model_bundle["model_key"],
            "model": model_bundle["model_id"],
            "model_loader": model_bundle.get("model_loader"),
            "prompt_condition": prompt_condition,
            "run_index": run_index,
            "shuffle_index": shuffle_index,
            "seed": seed,
            "gold": item.gold,
            "effective_gold": effective_gold,
            "prediction": prediction,
            "prediction_source_label": prediction_source_label,
            "is_correct": prediction_source_label == item.gold if item.gold is not None else None,
            "confidence": metrics["confidence"],
            "margin": metrics["margin"],
            "entropy": metrics["entropy"],
            "normalized_entropy": metrics["normalized_entropy"],
            "nll": metrics["nll"],
            "brier_score": metrics["brier_score"],
            "option_logit_scores": score["label_scores"],
            "option_probabilities": probabilities,
            "option_source_labels": option_source_labels,
            "elapsed_s": elapsed,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - preserve per-item failures in sweep output.
        return {
            "item_id": item.item_id,
            "dataset": item.dataset,
            "split": item.split,
            "linker": item.linker,
            "model_key": model_bundle["model_key"],
            "model": model_bundle["model_id"],
            "prompt_condition": prompt_condition,
            "run_index": run_index,
            "shuffle_index": shuffle_index,
            "seed": seed,
            "gold": item.gold,
            "effective_gold": effective_gold,
            "prediction": None,
            "prediction_source_label": None,
            "is_correct": False if item.gold is not None else None,
            "confidence": None,
            "margin": None,
            "entropy": None,
            "normalized_entropy": None,
            "nll": None,
            "brier_score": None,
            "option_logit_scores": {},
            "option_probabilities": {},
            "option_source_labels": option_source_labels,
            "elapsed_s": time.time() - started,
            "error": str(exc),
        }


def trial_key(row: dict) -> str:
    return "::".join(
        str(row.get(key, ""))
        for key in ("item_id", "prompt_condition", "run_index", "shuffle_index")
    )


def load_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_logits_sweep(
    items: list[QAItem],
    model_bundle: dict[str, Any],
    num_samples: int = 1,
    num_shuffles: int = 1,
    seed: int = 42,
    output_jsonl: Path | None = None,
    existing_rows: list[dict] | None = None,
    prompt_condition: str = "no_umls",
) -> list[dict]:
    rows = list(existing_rows or [])
    completed = {trial_key(row) for row in rows}
    output_fh = None
    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_fh = output_jsonl.open("a", encoding="utf-8")
    try:
        for item in items:
            for run_index in range(num_samples):
                for shuffle_index in range(num_shuffles):
                    row_seed = seed + run_index * 1000 + shuffle_index
                    planned = {
                        "item_id": item.item_id,
                        "prompt_condition": prompt_condition,
                        "run_index": run_index,
                        "shuffle_index": shuffle_index,
                    }
                    if trial_key(planned) in completed:
                        continue
                    row = score_item_once(
                        item,
                        model_bundle,
                        run_index=run_index,
                        shuffle_index=shuffle_index,
                        seed=row_seed,
                        shuffle_options=num_shuffles > 1,
                        prompt_condition=prompt_condition,
                    )
                    rows.append(row)
                    completed.add(trial_key(row))
                    if output_fh is not None:
                        output_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        output_fh.flush()
    finally:
        if output_fh is not None:
            output_fh.close()
    return rows


def aggregate_trials_by_item(rows: list[dict]) -> list[dict]:
    groups = group_rows(rows, ("item_id", "model_key", "model", "dataset", "split", "linker", "prompt_condition"))
    aggregates = []
    for key, group in sorted(groups.items()):
        item_id, model_key, model, dataset, split, linker, prompt_condition = key
        valid = [row for row in group if row.get("prediction_source_label") is not None and not row.get("error")]
        labels = [row.get("prediction_source_label") for row in valid]
        final_prediction, vote_tie, vote_agreement = majority_vote(labels)
        gold = group[0].get("gold")
        aggregate = {
            "item_id": item_id,
            "dataset": dataset,
            "split": split,
            "linker": linker,
            "model_key": model_key,
            "model": model,
            "prompt_condition": prompt_condition,
            "gold": gold,
            "prediction": final_prediction,
            "is_correct": final_prediction == gold if gold is not None and final_prediction is not None else False,
            "num_trials": len(group),
            "num_successful_trials": len(valid),
            "vote_tie": vote_tie,
            "vote_agreement": vote_agreement,
            "vote_entropy": vote_entropy(labels),
            "error": "" if valid else "; ".join(sorted({str(row.get("error")) for row in group if row.get("error")})),
        }
        for metric_key in (
            "confidence",
            "margin",
            "entropy",
            "normalized_entropy",
            "nll",
            "brier_score",
            "elapsed_s",
        ):
            values = [
                float(row[metric_key])
                for row in valid
                if isinstance(row.get(metric_key), (int, float))
            ]
            aggregate[metric_key] = fmean(values) if values else None
        aggregates.append(aggregate)
    return aggregates


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_rows = aggregate_trials_by_item(rows)
    groups = group_rows(aggregate_rows, ("model_key", "model", "dataset", "split", "linker", "prompt_condition"))
    fieldnames = [
        "model_key",
        "model",
        "dataset",
        "split",
        "linker",
        "prompt_condition",
        "n",
        "valid",
        "malformed",
        "malformed_rate",
        "malformed_ci_low",
        "malformed_ci_high",
        "accuracy",
        "accuracy_ci_low",
        "accuracy_ci_high",
        "ece",
        "mean_confidence",
        "mean_margin",
        "mean_entropy",
        "mean_normalized_entropy",
        "mean_nll",
        "mean_brier_score",
        "selective_accuracy_50",
        "selective_accuracy_70",
        "selective_accuracy_90",
        "mean_vote_agreement",
        "mean_vote_entropy",
        "mean_successful_trials",
        "mean_elapsed_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for key, group in sorted(groups.items()):
            model_key, model, dataset, split, linker, prompt_condition = key
            summary = summarize_prediction_rows(group)
            row = {
                "model_key": model_key,
                "model": model,
                "dataset": dataset,
                "split": split,
                "linker": linker,
                "prompt_condition": prompt_condition,
                **summary,
                "mean_vote_agreement": fmean(
                    float(item["vote_agreement"])
                    for item in group
                    if isinstance(item.get("vote_agreement"), (int, float))
                ),
                "mean_vote_entropy": fmean(
                    float(item["vote_entropy"])
                    for item in group
                    if isinstance(item.get("vote_entropy"), (int, float))
                ),
                "mean_successful_trials": fmean(
                    float(item["num_successful_trials"])
                    for item in group
                    if isinstance(item.get("num_successful_trials"), (int, float))
                ),
            }
            writer.writerow({
                name: f"{value:.6f}" if isinstance(value, float) else value
                for name, value in row.items()
                if name in fieldnames
            })


def default_output_paths(
    model_key: str,
    dataset: str,
    split: str,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    root = output_dir or Path("outputs") / "model_sweeps" / "medgemma_logits"
    stem = f"{model_slug(model_key)}_{dataset}_{split}_logits"
    return root / f"{stem}.jsonl", root / f"{stem}.summary.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local logit-scored medical QA sweeps.")
    parser.add_argument("--dataset", default="medQA")
    parser.add_argument("--split", default="test")
    parser.add_argument("--linker", choices=["metamap", "spacy"], default="metamap")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--limit", type=int, default=10, help="Number of examples to run. Use 0 for all examples.")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-shuffles", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32", "auto"], default="bfloat16")
    parser.add_argument("--mock-model", action="store_true", help="Use deterministic mock logits for smoke tests.")
    parser.add_argument("--prompt-condition", default="no_umls")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input or default_input_path(args.dataset, args.split, args.linker)
    reference = args.reference or default_reference_path(args.dataset, args.split)
    items = load_items(
        args.dataset,
        args.split,
        args.linker,
        input_path=input_path,
        reference_path=reference if reference.exists() else None,
        limit=normalize_limit(args.limit),
    )
    default_jsonl, default_summary = default_output_paths(args.model_key, args.dataset, args.split, args.output_dir)
    output_jsonl = args.output_jsonl or default_jsonl
    summary_csv = args.summary_csv or default_summary

    if args.dry_run:
        sample_prompt = build_scoring_prompt(items[0]) if items else ""
        print(json.dumps({
            "model_key": args.model_key,
            "model_id": args.model_id or MODEL_CONFIGS.get(args.model_key, {}).get("model_id") or args.model_key,
            "dataset": args.dataset,
            "split": args.split,
            "linker": args.linker,
            "items": len(items),
            "num_samples": args.num_samples,
            "num_shuffles": args.num_shuffles,
            "output_jsonl": str(output_jsonl),
            "summary_csv": str(summary_csv),
            "sample_prompt": sample_prompt,
        }, indent=2))
        return

    if args.mock_model:
        model_bundle = load_mock_model(
            model_key=args.model_key,
            model_id=args.model_id or "mock-local-logits",
        )
    else:
        model_bundle = load_local_model(
            args.model_key,
            model_id=args.model_id,
            device_map=args.device_map,
            dtype=args.dtype,
        )
    try:
        existing_rows = load_existing_rows(output_jsonl) if args.resume else []
        rows = run_logits_sweep(
            items,
            model_bundle=model_bundle,
            num_samples=max(1, args.num_samples),
            num_shuffles=max(1, args.num_shuffles),
            seed=args.seed,
            output_jsonl=output_jsonl if args.resume else None,
            existing_rows=existing_rows,
            prompt_condition=args.prompt_condition,
        )
        write_jsonl(output_jsonl, rows)
        write_summary_csv(summary_csv, rows)
    finally:
        unload_model(model_bundle)


if __name__ == "__main__":
    main()
