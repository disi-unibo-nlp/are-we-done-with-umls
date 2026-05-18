from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean

from src.experiments.loaders import load_metamap_records, load_spacy_records
from src.experiments.schemas import QAItem


def dataset_options_per_item(dataset: str) -> int:
    return 3 if dataset.lower() == "pubmedqa" else 4


def default_input_path(dataset: str, split: str, linker: str) -> Path:
    base = Path("data") / dataset
    if linker == "metamap":
        return base / "metamap_mapped" / f"{split}.jsonl"
    if linker == "spacy":
        return base / "spacy_grounded" / f"{split}.grounded.jsonl"
    raise ValueError(f"Unsupported linker: {linker}")


def default_reference_path(dataset: str, split: str) -> Path:
    return Path("data") / dataset / "metamap_mapped" / f"{split}.jsonl"


def load_items(
    dataset: str,
    split: str,
    linker: str,
    input_path: Path | None = None,
    reference_path: Path | None = None,
    limit: int | None = None,
) -> list[QAItem]:
    resolved_input = input_path or default_input_path(dataset, split, linker)
    if linker == "metamap":
        items = load_metamap_records(resolved_input, dataset=dataset, split=split)
    elif linker == "spacy":
        resolved_reference = reference_path or default_reference_path(dataset, split)
        items = load_spacy_records(
            resolved_input,
            dataset=dataset,
            split=split,
            options_per_item=dataset_options_per_item(dataset),
            reference_path=resolved_reference if resolved_reference.exists() else None,
        )
    else:
        raise ValueError(f"Unsupported linker: {linker}")
    return items[:limit] if limit is not None else items


def collect_cuis(items: list[QAItem]) -> set[str]:
    cuis: set[str] = set()
    for item in items:
        cuis.update(item.question_cuis)
        for option in item.option_cuis.values():
            cuis.update(option)
    return cuis


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summary_key(row: dict) -> tuple:
    return (
        row.get("experiment", ""),
        row.get("dataset", ""),
        row.get("split", ""),
        row.get("linker", ""),
        row.get("condition", row.get("view", row.get("method", ""))),
    )


def _metric_items(row: dict) -> dict:
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        out = dict(metrics)
        for key, value in row.items():
            if key not in out:
                out[key] = value
        return out
    return row


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(_summary_key(row), []).append(row)

    numeric_keys = sorted({
        key
        for row in rows
        for key, value in _metric_items(row).items()
        if isinstance(value, (int, float, bool))
    })
    fieldnames = ["experiment", "dataset", "split", "linker", "condition", "n"]
    fieldnames.extend(f"mean_{key}" for key in numeric_keys)

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for key, group_rows in sorted(groups.items()):
            experiment, dataset, split, linker, condition = key
            out = {
                "experiment": experiment,
                "dataset": dataset,
                "split": split,
                "linker": linker,
                "condition": condition,
                "n": len(group_rows),
            }
            for numeric_key in numeric_keys:
                values = [
                    float(_metric_items(row)[numeric_key])
                    for row in group_rows
                    if isinstance(_metric_items(row).get(numeric_key), (int, float, bool))
                ]
                out[f"mean_{numeric_key}"] = f"{fmean(values):.6f}" if values else ""
            writer.writerow(out)


def default_output_paths(experiment: str, dataset: str, split: str, linker: str) -> tuple[Path, Path]:
    stem = f"{experiment}_{dataset}_{split}_{linker}"
    root = Path("outputs") / "diagnostics"
    return root / f"{stem}.jsonl", root / f"{stem}.summary.csv"
