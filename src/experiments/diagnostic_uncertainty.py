from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import fmean

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model_sweeps.uncertainty_metrics import bootstrap_ci, group_rows, wilson_interval


DEFAULT_GROUP_KEYS = ("experiment", "dataset", "split", "linker", "condition", "view", "method")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def is_numeric(value: object) -> bool:
    return isinstance(value, (int, float, bool)) and math.isfinite(float(value))


def metric_items(row: dict) -> dict:
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        out = dict(metrics)
        for key, value in row.items():
            out.setdefault(key, value)
        return out
    return row


def infer_group_keys(rows: list[dict], requested: list[str] | None = None) -> list[str]:
    if requested:
        return requested
    present = set().union(*(row.keys() for row in rows)) if rows else set()
    return [key for key in DEFAULT_GROUP_KEYS if key in present]


def metric_rows_for_group(group: list[dict], iterations: int, seed: int) -> list[dict]:
    metric_names = sorted({
        key
        for row in group
        for key, value in metric_items(row).items()
        if is_numeric(value)
    })
    rows = []
    for metric in metric_names:
        values = [
            metric_items(row)[metric]
            for row in group
            if is_numeric(metric_items(row).get(metric))
        ]
        if not values:
            continue
        if all(isinstance(value, bool) for value in values):
            successes = sum(1 for value in values if value)
            ci_low, ci_high = wilson_interval(successes, len(values))
            rows.append({
                "metric": metric,
                "metric_type": "binary",
                "n": len(values),
                "mean": successes / len(values),
                "ci_low": ci_low,
                "ci_high": ci_high,
            })
        else:
            numeric = [float(value) for value in values]
            ci_low, ci_high = bootstrap_ci(numeric, iterations=iterations, seed=seed)
            rows.append({
                "metric": metric,
                "metric_type": "numeric",
                "n": len(numeric),
                "mean": fmean(numeric),
                "ci_low": ci_low,
                "ci_high": ci_high,
            })
    return rows


def summarize_diagnostics(
    rows: list[dict],
    group_keys: list[str] | None = None,
    iterations: int = 1000,
    seed: int = 13,
) -> tuple[list[str], list[dict]]:
    resolved_group_keys = infer_group_keys(rows, group_keys)
    groups = group_rows(rows, resolved_group_keys)
    out = []
    for key, group in sorted(groups.items()):
        group_values = dict(zip(resolved_group_keys, key))
        for metric_row in metric_rows_for_group(group, iterations=iterations, seed=seed):
            out.append({**group_values, **metric_row})
    return resolved_group_keys, out


def write_csv(path: Path, group_keys: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [*group_keys, "metric", "metric_type", "n", "mean", "ci_low", "ci_high"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.6f}" if isinstance(value, float) else value
                for key, value in row.items()
                if key in fieldnames
            })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add uncertainty intervals to diagnostic JSONL outputs.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--group-key", action="append", default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = []
    for path in args.inputs:
        rows.extend(read_jsonl(path))
    group_keys, summary_rows = summarize_diagnostics(
        rows,
        group_keys=args.group_key,
        iterations=max(1, args.bootstrap_iterations),
        seed=args.seed,
    )
    write_csv(args.output_csv, group_keys, summary_rows)


if __name__ == "__main__":
    main()
