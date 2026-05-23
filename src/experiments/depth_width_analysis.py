"""Depth x width accuracy analysis for operational STEP 3.

Joins Exp3 retrieval faithfulness JSONL (per-item structural metrics) with
Qwen prediction JSONL (per-item correctness) to produce 2D accuracy tables
stratified by gold_min_path_length (depth) and num_retrieved_nodes (width).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


_DEPTH_BINS = [1, 2, 3, 4]
_WIDTH_BINS = [
    ("tight", 0, 20),
    ("medium", 21, 50),
    ("wide", 51, 100),
    ("full", 101, 99999),
]


def _depth_bin(path_length: int | None) -> str:
    if path_length is None:
        return "no_path"
    return str(min(path_length, 4)) if path_length <= 4 else "4+"


def _width_bin(num_nodes: int) -> str:
    for label, lo, hi in _WIDTH_BINS:
        if lo <= num_nodes <= hi:
            return label
    return "full"


def _row_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("split", "")),
        str(row.get("linker", "")),
        str(row.get("item_id", "")),
    )


def build_structural_index(exp3_rows: list[dict], condition: str = "q_to_a_2hop") -> dict:
    """(dataset, split, linker, item_id) -> structural metrics from Exp3."""
    index: dict[tuple[str, str, str, str], dict] = {}
    for row in exp3_rows:
        if row.get("condition") != condition:
            continue
        index[_row_key(row)] = {
            "gold_path_available": row.get("gold_path_available"),
            "gold_min_path_length": row.get("gold_min_path_length"),
            "num_retrieved_nodes": row.get("num_retrieved_nodes", 0),
            "distractor_path_rate": row.get("distractor_path_rate", 0),
            "decision_relevance_ratio_proxy": row.get("decision_relevance_ratio_proxy", 0),
        }
    return index


def analyse(
    pred_rows: list[dict],
    structural_index: dict,
    target_condition: str,
) -> dict:
    """Return depth-table and width-table for one prediction condition."""
    # depth_bin -> [is_correct]
    depth_correct: dict[str, list[float]] = defaultdict(list)
    # width_bin -> [is_correct]
    width_correct: dict[str, list[float]] = defaultdict(list)
    # (depth_bin, width_bin) -> [is_correct]
    cross_correct: dict[tuple, list[float]] = defaultdict(list)

    for row in pred_rows:
        if row.get("condition") != target_condition:
            continue
        is_correct = float(row.get("is_correct", 0))
        struct = structural_index.get(_row_key(row), {})
        d = _depth_bin(struct.get("gold_min_path_length"))
        w = _width_bin(int(struct.get("num_retrieved_nodes") or 0))
        depth_correct[d].append(is_correct)
        width_correct[w].append(is_correct)
        cross_correct[(d, w)].append(is_correct)

    def _acc(vals: list[float]) -> str:
        return f"{mean(vals):.3f} (n={len(vals)})" if vals else "n/a"

    depth_table = {
        d: _acc(depth_correct[d])
        for d in ["1", "2", "3", "4", "no_path"]
    }
    width_table = {
        w: _acc(width_correct[w])
        for w, _, _ in _WIDTH_BINS
    }
    cross_table = {
        (d, w): _acc(cross_correct[(d, w)])
        for d in ["1", "2", "3", "4", "no_path"]
        for w, _, _ in _WIDTH_BINS
    }
    return {"depth": depth_table, "width": width_table, "cross": cross_table}


def main() -> None:
    parser = argparse.ArgumentParser(description="Depth × width accuracy analysis.")
    parser.add_argument("--exp3-jsonl", required=True, nargs="+", type=Path,
                        help="Exp3 retrieval faithfulness JSONL file(s)")
    parser.add_argument("--pred-jsonl", required=True, nargs="+", type=Path,
                        help="Qwen prediction JSONL files")
    parser.add_argument("--conditions", nargs="+",
                        default=[
                            f"chain_depth_{depth}_width_{width}_evidence"
                            for depth in (1, 2, 3, 4)
                            for width in ("tight", "medium", "wide", "full")
                        ])
    parser.add_argument("--exp3-condition", default="q_to_a_2hop",
                        help="Exp3 condition to use for structural metrics")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    exp3_rows: list[dict] = []
    for path in args.exp3_jsonl:
        exp3_rows.extend(_load_jsonl(path))
    structural_index = build_structural_index(exp3_rows, args.exp3_condition)
    print(f"Structural index: {len(structural_index)} items from Exp3.")

    all_pred_rows: list[dict] = []
    for path in args.pred_jsonl:
        all_pred_rows.extend(_load_jsonl(path))
    print(f"Prediction rows loaded: {len(all_pred_rows)}")

    # Group predictions by (model, dataset, linker)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in all_pred_rows:
        key = (row.get("model", ""), row.get("dataset", ""), row.get("linker", ""))
        groups[key].append(row)

    summary_rows = []
    for (model, dataset, linker), pred_rows in sorted(groups.items()):
        for cond in args.conditions:
            result = analyse(pred_rows, structural_index, cond)
            for depth_bin, acc_str in result["depth"].items():
                summary_rows.append({
                    "model": model, "dataset": dataset, "linker": linker,
                    "condition": cond, "axis": "depth", "bin": depth_bin,
                    "accuracy": acc_str,
                })
            for width_bin, acc_str in result["width"].items():
                summary_rows.append({
                    "model": model, "dataset": dataset, "linker": linker,
                    "condition": cond, "axis": "width", "bin": width_bin,
                    "accuracy": acc_str,
                })

    out_csv = args.output_dir / "depth_width_accuracy.csv"
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["model", "dataset", "linker", "condition", "axis", "bin", "accuracy"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {out_csv} ({len(summary_rows)} rows)")

    # Also write a markdown summary
    out_md = args.output_dir / "depth_width_accuracy.md"
    lines = ["# Depth x Width Accuracy Analysis\n"]
    for (model, dataset, linker), pred_rows in sorted(groups.items()):
        lines.append(f"## {model} / {dataset} / {linker}\n")
        for cond in args.conditions:
            result = analyse(pred_rows, structural_index, cond)
            lines.append(f"### {cond}")
            lines.append("**Depth axis:**")
            for d in ["1", "2", "3", "4", "no_path"]:
                lines.append(f"- depth={d}: {result['depth'][d]}")
            lines.append("**Width axis:**")
            for w, _, _ in _WIDTH_BINS:
                lines.append(f"- width={w}: {result['width'][w]}")
            lines.append("")
    out_md.write_text("\n".join(lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
