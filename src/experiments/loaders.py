from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .schemas import QAItem


OPTION_LETTERS = ("A", "B", "C", "D")
PUBMEDQA_OPTIONS = ("yes", "no", "maybe")


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {exc}") from exc


def _normalize_options(raw: dict, dataset: str) -> dict[str, str]:
    opts = raw.get("options")
    if isinstance(opts, dict):
        return {letter: str(opts.get(letter, "") or "") for letter in OPTION_LETTERS}
    if isinstance(opts, list):
        if dataset.lower() == "pubmedqa" and len(opts) == 3:
            return {key: str(value) for key, value in zip(PUBMEDQA_OPTIONS, opts)}
        return {letter: str(value) for letter, value in zip(OPTION_LETTERS, opts)}
    if all(key in raw for key in ("opa", "opb", "opc", "opd")):
        return {
            "A": str(raw.get("opa", "") or ""),
            "B": str(raw.get("opb", "") or ""),
            "C": str(raw.get("opc", "") or ""),
            "D": str(raw.get("opd", "") or ""),
        }
    return {}


def _normalize_gold(raw: dict, options: dict[str, str]) -> tuple[str | None, str | None]:
    gold = raw.get("answer_idx")
    if isinstance(gold, int):
        keys = list(options)
        gold = keys[gold] if 0 <= gold < len(keys) else None
    elif gold is not None:
        gold = str(gold)
    answer = raw.get("answer")
    if answer is None and gold in options:
        answer = options[gold]
    # For formats where the answer field itself is an option key (e.g. pubmedQA)
    if gold is None and answer is not None and str(answer) in options:
        gold = str(answer)
    return gold, str(answer) if answer is not None else None


def load_metamap_records(
    path: Path,
    dataset: str,
    split: str,
    linker: str = "metamap",
) -> list[QAItem]:
    rows: list[QAItem] = []
    for idx, raw in enumerate(iter_jsonl(path)):
        options = _normalize_options(raw, dataset)
        gold, answer = _normalize_gold(raw, options)
        option_cuis = {
            key: list(raw.get(f"option{i}", []) or [])
            for i, key in enumerate(options)
        }
        rows.append(QAItem(
            item_id=f"{dataset}:{split}:{idx}",
            dataset=dataset,
            split=split,
            linker=linker,
            question=str(raw.get("question", "") or ""),
            options=options,
            gold=gold,
            answer_text=answer,
            question_cuis=list(raw.get("concept", []) or []),
            option_cuis=option_cuis,
            metadata={"source_path": str(path)},
        ))
    return rows


def load_spacy_records(
    path: Path,
    dataset: str,
    split: str,
    options_per_item: int,
    reference_path: Path | None = None,
    linker: str = "spacy",
) -> list[QAItem]:
    raw_rows = list(iter_jsonl(path))
    if len(raw_rows) % options_per_item != 0:
        raise ValueError(
            f"{path} has {len(raw_rows)} rows, not a multiple of {options_per_item}."
        )

    references = (
        load_metamap_records(reference_path, dataset, split, linker="reference")
        if reference_path is not None
        else []
    )
    items: list[QAItem] = []
    default_keys = list(PUBMEDQA_OPTIONS if dataset.lower() == "pubmedqa" else OPTION_LETTERS)
    option_keys = default_keys[:options_per_item]

    for item_idx in range(0, len(raw_rows), options_per_item):
        group = raw_rows[item_idx:item_idx + options_per_item]
        idx = item_idx // options_per_item
        reference = references[idx] if idx < len(references) else None
        options = (
            dict(reference.options)
            if reference is not None and reference.options
            else {key: str(group[i].get("ans", "") or "") for i, key in enumerate(option_keys)}
        )
        keys = list(options)
        option_cuis = {
            key: list(group[i].get("ac", []) or [])
            for i, key in enumerate(keys)
            if i < len(group)
        }
        question = reference.question if reference is not None else str(group[0].get("sent", "") or "")
        gold = reference.gold if reference is not None else None
        answer_text = reference.answer_text if reference is not None else None
        items.append(QAItem(
            item_id=f"{dataset}:{split}:{idx}",
            dataset=dataset,
            split=split,
            linker=linker,
            question=question,
            options=options,
            gold=gold,
            answer_text=answer_text,
            question_cuis=list(group[0].get("qc", []) or []),
            option_cuis=option_cuis,
            metadata={"source_path": str(path)},
        ))
    return items

