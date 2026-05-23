from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


BASELINE_CONDITION = "no_umls"
STANDARD_UMLS_CONDITION = "metadata_rich_evidence"


def qwen_slug(model: str) -> str:
    return model.replace("/", "__").replace(".", "_")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def group_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("split", "")),
        str(row.get("linker", "")),
        str(row.get("model", "")),
        str(row.get("mode", "")),
    )


def condition_name(row: dict[str, Any]) -> str:
    return str(row.get("condition", row.get("prompt_condition", "")))


def is_valid_correct(row: dict[str, Any]) -> bool:
    if row.get("prediction") is None:
        return False
    if row.get("error") not in {None, ""}:
        return False
    return bool(row.get("is_correct", False))


def build_group_manifests(
    rows: list[dict[str, Any]],
    *,
    standard_condition: str = STANDARD_UMLS_CONDITION,
) -> tuple[dict[tuple[str, str, str, str, str], list[dict[str, Any]]], list[dict[str, Any]]]:
    by_group_item_condition: dict[
        tuple[str, str, str, str, str],
        dict[str, dict[str, dict[str, Any]]],
    ] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        condition = condition_name(row)
        if condition not in {BASELINE_CONDITION, standard_condition}:
            continue
        by_group_item_condition[group_key(row)][str(row.get("item_id", ""))][condition] = row

    manifests: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []
    for key, by_item in sorted(by_group_item_condition.items()):
        dataset, split, linker, model, mode = key
        manifest_rows: list[dict[str, Any]] = []
        total = baseline_correct = baseline_wrong = helped = still_wrong = missing_standard = 0
        for item_id, condition_rows in sorted(by_item.items()):
            baseline = condition_rows.get(BASELINE_CONDITION)
            if baseline is None:
                continue
            total += 1
            if is_valid_correct(baseline):
                baseline_correct += 1
                continue
            baseline_wrong += 1
            standard = condition_rows.get(standard_condition)
            if standard is None:
                missing_standard += 1
                label = "missing_standard_umls"
                standard_correct = False
            else:
                standard_correct = is_valid_correct(standard)
                label = "umls_helped" if standard_correct else "still_wrong"
            if label == "umls_helped":
                helped += 1
            elif label == "still_wrong":
                still_wrong += 1
            manifest_rows.append({
                "item_id": item_id,
                "dataset": dataset,
                "split": split,
                "linker": linker,
                "model": model,
                "mode": mode,
                "selection": "baseline_wrong",
                "subset_label": label,
                "baseline_condition": BASELINE_CONDITION,
                "standard_umls_condition": standard_condition,
                "baseline_is_correct": False,
                "standard_umls_is_correct": standard_correct,
            })
        manifests[key] = manifest_rows
        summary_rows.append({
            "dataset": dataset,
            "split": split,
            "linker": linker,
            "model": model,
            "mode": mode,
            "total_baseline_rows": total,
            "baseline_correct": baseline_correct,
            "baseline_wrong": baseline_wrong,
            "umls_helped": helped,
            "still_wrong": still_wrong,
            "missing_standard_umls": missing_standard,
        })
    return manifests, summary_rows


def manifest_path(output_dir: Path, key: tuple[str, str, str, str, str]) -> Path:
    dataset, split, linker, model, mode = key
    stem = "_".join([dataset, split, linker, qwen_slug(model), mode, "baseline_wrong"])
    return output_dir / f"{stem}.jsonl"


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "split",
        "linker",
        "model",
        "mode",
        "total_baseline_rows",
        "baseline_correct",
        "baseline_wrong",
        "umls_helped",
        "still_wrong",
        "missing_standard_umls",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# STEP 1 Baseline-Wrong Subsets",
        "",
        "| dataset | linker | model | mode | baseline_wrong | umls_helped | still_wrong | missing_standard |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['linker']} | `{row['model']}` | {row['mode']} | "
            f"{row['baseline_wrong']} | {row['umls_helped']} | {row['still_wrong']} | "
            f"{row['missing_standard_umls']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build STEP 1 baseline-wrong item manifests.")
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--standard-condition", default=STANDARD_UMLS_CONDITION)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.predictions:
        rows.extend(read_jsonl(path))

    manifests, summary_rows = build_group_manifests(rows, standard_condition=args.standard_condition)
    for key, manifest_rows in manifests.items():
        write_jsonl(manifest_path(args.output_dir, key), manifest_rows)
    write_summary_csv(args.output_dir / "step1_subset_summary.csv", summary_rows)
    write_summary_md(args.output_dir / "step1_subset_summary.md", summary_rows)
    print(f"Wrote {len(manifests)} manifests under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
