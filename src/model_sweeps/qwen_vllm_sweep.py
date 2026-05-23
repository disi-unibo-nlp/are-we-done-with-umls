from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.cli_utils import default_input_path, default_reference_path, load_items
from src.experiments.schemas import QAItem


DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_BASE_URL = "http://localhost:8000/v1"


def decoding_params(mode: str) -> dict:
    if mode == "non_thinking":
        return {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 2.0,
            "repetition_penalty": 1.0,
            "enable_thinking": False,
        }
    if mode == "thinking":
        return {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "enable_thinking": True,
        }
    raise ValueError(f"Unsupported mode: {mode}")


def apply_decoding_overrides(params: dict, overrides: dict | None = None) -> dict:
    resolved = dict(params)
    for key, value in (overrides or {}).items():
        if value is not None:
            resolved[key] = value
    return resolved


def build_prompt(item: QAItem, evidence: str | None = None) -> str:
    lines = [
        "You are answering a medical multiple-choice question.",
        "Choose exactly one answer option.",
        r"Output exactly one token in this format: \boxed{A}",
        "Do not explain your reasoning. Do not restate the question. Do not add any other text.",
    ]
    if evidence:
        lines.append("Use the provided UMLS evidence as the only external evidence source.")
    lines.extend([
        "",
        "Question:",
        item.question,
        "",
        "Options:",
    ])
    for key, value in item.options.items():
        lines.append(f"{key}. {value}")
    if evidence:
        lines.extend(["", "UMLS evidence:", evidence])
    lines.extend([
        "",
        r"Final answer only: \boxed{X}",
    ])
    return "\n".join(lines)


def build_request_payload(
    item: QAItem,
    model: str,
    mode: str,
    max_tokens: int,
    evidence: str | None = None,
    decoding_overrides: dict | None = None,
) -> dict:
    params = apply_decoding_overrides(decoding_params(mode), decoding_overrides)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": r"You are a strict multiple-choice classifier. Reply with exactly one token: \boxed{A}, \boxed{B}, etc. No explanation.",
            },
            {
                "role": "user",
                "content": build_prompt(item, evidence=evidence),
            }
        ],
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "top_k": params["top_k"],
        "min_p": params["min_p"],
        "presence_penalty": params["presence_penalty"],
        "repetition_penalty": params["repetition_penalty"],
        "chat_template_kwargs": {"enable_thinking": params["enable_thinking"]},
        "max_tokens": max_tokens,
    }
    if params.get("seed") is not None:
        payload["seed"] = params["seed"]
    return payload


class VLLMClient:
    def __init__(self, base_url: str, api_key: str = "EMPTY", timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    def chat_completion(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM request failed with HTTP {exc.code}: {detail}") from exc


def _message_from_response(response: dict) -> dict:
    choices = response.get("choices") or []
    if not choices:
        return {}
    return choices[0].get("message") or {}


def response_text(response: dict) -> tuple[str, str]:
    message = _message_from_response(response)
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    return str(content), str(reasoning)


def extract_answer(text: str, allowed_options: Iterable[str]) -> str | None:
    allowed = list(allowed_options)
    option_pattern = "|".join(re.escape(option) for option in allowed)

    # Primary: \boxed{X} format
    boxed_match = re.search(rf"\\boxed\s*\{{\s*({option_pattern})\s*\}}", text, flags=re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1).upper()

    # Fallback: JSON {"answer": "X"}
    try:
        parsed = json.loads(text)
        answer = str(parsed.get("answer", "")).strip()
        if answer.upper() in allowed:
            return answer.upper()
    except json.JSONDecodeError:
        pass

    # Fallback: answer-only or explicit final-answer patterns.  Avoid broad
    # "option X" matching inside rationales because that can convert a long,
    # truncated explanation into a false parse.
    patterns = [
        rf'"answer"\s*:\s*"({option_pattern})"',
        rf"\bfinal\s+answer\s*(?:is|:)\s*({option_pattern})\b",
        rf"^\s*({option_pattern})\s*$",
        rf"^\s*({option_pattern})[\).:]\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def summarize_predictions(rows: list[dict], group_keys: tuple[str, ...] = ("mode",)) -> dict[tuple, dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in group_keys)].append(row)

    summary: dict[tuple, dict] = {}
    for key, grouped in groups.items():
        correctness = [1.0 if row.get("is_correct") else 0.0 for row in grouped]
        elapsed = [
            float(row["elapsed_s"])
            for row in grouped
            if isinstance(row.get("elapsed_s"), (int, float))
        ]
        summary[key] = {
            "n": len(grouped),
            "accuracy": fmean(correctness) if correctness else 0.0,
            "malformed": sum(1 for row in grouped if row.get("prediction") is None),
            "mean_elapsed_s": fmean(elapsed) if elapsed else 0.0,
        }
    return summary


def default_output_paths(model: str, dataset: str, split: str, mode: str) -> tuple[Path, Path]:
    model_slug = model.replace("/", "__").replace(".", "_")
    stem = f"{model_slug}_{dataset}_{split}_{mode}"
    root = Path("outputs") / "model_sweeps"
    return root / f"{stem}.jsonl", root / f"{stem}.summary.csv"


def normalize_limit(limit: int | None) -> int | None:
    if limit is None or limit <= 0:
        return None
    return limit


def load_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
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


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_predictions(rows, group_keys=("mode",))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["mode", "n", "accuracy", "malformed", "mean_elapsed_s"])
        writer.writeheader()
        for key, values in sorted(summary.items()):
            writer.writerow({
                "mode": key[0],
                "n": values["n"],
                "accuracy": f"{values['accuracy']:.6f}",
                "malformed": values["malformed"],
                "mean_elapsed_s": f"{values['mean_elapsed_s']:.6f}",
            })


def predict_item(
    item: QAItem,
    client: VLLMClient,
    model: str,
    mode: str,
    max_tokens: int,
    evidence: str | None = None,
    condition: str = "no_umls",
    max_retries: int = 2,
    retry_sleep_s: float = 2.0,
    decoding_overrides: dict | None = None,
) -> dict:
    payload = build_request_payload(
        item,
        model=model,
        mode=mode,
        max_tokens=max_tokens,
        evidence=evidence,
        decoding_overrides=decoding_overrides,
    )
    started = time.time()
    last_error: Exception | None = None
    response: dict | None = None
    attempts = max(1, max_retries + 1)
    for attempt in range(attempts):
        try:
            response = client.chat_completion(payload)
            break
        except Exception as exc:  # noqa: BLE001 - benchmark rows should preserve exhausted request errors.
            last_error = exc
            if attempt + 1 < attempts and retry_sleep_s > 0:
                time.sleep(retry_sleep_s)
    elapsed = time.time() - started
    if response is None:
        return {
            "item_id": item.item_id,
            "dataset": item.dataset,
            "split": item.split,
            "linker": item.linker,
            "model": model,
            "mode": mode,
            "condition": condition,
            "prompt_condition": condition,
            "gold": item.gold,
            "prediction": None,
            "is_correct": False if item.gold is not None else None,
            "malformed": True,
            "content": "",
            "reasoning_content": "",
            "elapsed_s": elapsed,
            "error": str(last_error) if last_error is not None else "request failed",
        }
    content, reasoning = response_text(response)
    prediction = extract_answer(content, item.options.keys())
    if prediction is None and reasoning:
        prediction = extract_answer(reasoning, item.options.keys())
    return {
        "item_id": item.item_id,
        "dataset": item.dataset,
        "split": item.split,
        "linker": item.linker,
        "model": model,
        "mode": mode,
        "condition": condition,
        "prompt_condition": condition,
        "gold": item.gold,
        "prediction": prediction,
        "is_correct": prediction == item.gold if item.gold is not None else None,
        "malformed": prediction is None,
        "content": content,
        "reasoning_content": reasoning,
        "elapsed_s": elapsed,
    }


def run_sweep(
    items: list[QAItem],
    client: VLLMClient,
    model: str,
    mode: str,
    max_tokens: int,
    sleep_s: float = 0.0,
    output_jsonl: Path | None = None,
    existing_rows: list[dict] | None = None,
    concurrency: int = 1,
    max_retries: int = 2,
    retry_sleep_s: float = 2.0,
    decoding_overrides: dict | None = None,
) -> list[dict]:
    rows = list(existing_rows or [])
    completed_item_ids = {str(row.get("item_id")) for row in rows}
    pending_items = [item for item in items if item.item_id not in completed_item_ids]
    output_fh = None
    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_fh = output_jsonl.open("a", encoding="utf-8")

    try:
        if concurrency <= 1:
            row_iter = (
                predict_item(
                    item,
                    client,
                    model=model,
                    mode=mode,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    retry_sleep_s=retry_sleep_s,
                    decoding_overrides=decoding_overrides,
                )
                for item in pending_items
            )
        else:
            executor = ThreadPoolExecutor(max_workers=concurrency)
            futures = [
                executor.submit(
                    predict_item,
                    item,
                    client,
                    model,
                    mode,
                    max_tokens,
                    None,
                    "no_umls",
                    max_retries,
                    retry_sleep_s,
                    decoding_overrides,
                )
                for item in pending_items
            ]
            row_iter = (future.result() for future in as_completed(futures))

        for row in row_iter:
            rows.append(row)
            completed_item_ids.add(str(row["item_id"]))
            if output_fh is not None:
                output_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_fh.flush()
            if sleep_s:
                time.sleep(sleep_s)
    finally:
        if "executor" in locals():
            executor.shutdown(wait=True, cancel_futures=False)
        if output_fh is not None:
            output_fh.close()
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen sweeps through a vLLM OpenAI-compatible server.")
    parser.add_argument("--dataset", default="medQA")
    parser.add_argument("--split", default="test")
    parser.add_argument("--linker", choices=["metamap", "spacy"], default="metamap")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=["non_thinking", "thinking"], default="non_thinking")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--limit", type=int, default=10, help="Number of examples to run. Use 0 for all examples.")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests to send to vLLM.")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep-s", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Append to an existing JSONL and skip completed item IDs.")
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
    max_tokens = args.max_tokens
    if max_tokens is None:
        max_tokens = 768 if args.mode == "thinking" else 256
    decoding_overrides = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "seed": args.seed,
    }

    if args.dry_run:
        payload = build_request_payload(
            items[0],
            args.model,
            args.mode,
            max_tokens,
            decoding_overrides=decoding_overrides,
        ) if items else {}
        print(json.dumps({
            "base_url": args.base_url,
            "model": args.model,
            "mode": args.mode,
            "items": len(items),
            "sample_payload": payload,
        }, indent=2))
        return

    default_jsonl, default_summary = default_output_paths(args.model, args.dataset, args.split, args.mode)
    output_jsonl = args.output_jsonl or default_jsonl
    summary_csv = args.summary_csv or default_summary
    existing_rows = load_existing_rows(output_jsonl) if args.resume else []

    client = VLLMClient(args.base_url, api_key=args.api_key, timeout_s=args.timeout_s)
    rows = run_sweep(
        items,
        client,
        args.model,
        args.mode,
        max_tokens,
        sleep_s=args.sleep_s,
        output_jsonl=output_jsonl if args.resume else None,
        existing_rows=existing_rows,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        retry_sleep_s=args.retry_sleep_s,
        decoding_overrides=decoding_overrides,
    )
    write_jsonl(output_jsonl, rows)
    write_summary_csv(summary_csv, rows)


if __name__ == "__main__":
    main()
