"""Q-to-answer exact-depth accuracy analysis for operational STEP 3."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


EXACT_DEPTH_PREFIX = "q_to_a_exact_depth_"
EXACT_DEPTH_SUFFIX = "_evidence"
DEFAULT_CONDITIONS = tuple(
    f"{EXACT_DEPTH_PREFIX}{depth}{EXACT_DEPTH_SUFFIX}"
    for depth in (1, 2, 3, 4)
)


def parse_exact_depth(condition: str) -> int | None:
    if not condition.startswith(EXACT_DEPTH_PREFIX) or not condition.endswith(EXACT_DEPTH_SUFFIX):
        return None
    depth_text = condition[len(EXACT_DEPTH_PREFIX):-len(EXACT_DEPTH_SUFFIX)]
    try:
        return int(depth_text)
    except ValueError:
        return None


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _strict_correct(row: dict) -> bool:
    return bool(row.get("is_correct")) and not bool(row.get("malformed")) and row.get("prediction") is not None


def _row_condition(row: dict) -> str:
    return str(row.get("condition", row.get("prompt_condition", "")))


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def analyse(rows: list[dict], conditions: tuple[str, ...] = DEFAULT_CONDITIONS) -> list[dict]:
    groups: dict[tuple[str, str, str, str, str, str, int], list[dict]] = defaultdict(list)
    allowed = set(conditions)
    for row in rows:
        condition = _row_condition(row)
        if condition not in allowed:
            continue
        depth = parse_exact_depth(condition)
        if depth is None:
            continue
        key = (
            str(row.get("model_key", row.get("model", ""))),
            str(row.get("model", "")),
            str(row.get("dataset", "")),
            str(row.get("split", "")),
            str(row.get("linker", "")),
            str(row.get("mode", "")),
            depth,
        )
        groups[key].append(row)

    summary_rows = []
    for key, group in sorted(groups.items()):
        model_key, model, dataset, split, linker, mode, depth = key
        condition = f"{EXACT_DEPTH_PREFIX}{depth}{EXACT_DEPTH_SUFFIX}"
        n = len(group)
        malformed = [1.0 if row.get("malformed") or row.get("prediction") is None else 0.0 for row in group]
        path_found = [1.0 if row.get("q_to_a_path_found") else 0.0 for row in group]
        edge_counts = [float(row.get("q_to_a_path_edge_count") or 0) for row in group]
        node_counts = [float(row.get("q_to_a_path_node_count") or 0) for row in group]
        summary_rows.append({
            "model_key": model_key,
            "model": model,
            "dataset": dataset,
            "split": split,
            "linker": linker,
            "mode": mode,
            "condition": condition,
            "depth": depth,
            "n": n,
            "accuracy": _mean([1.0 if row.get("is_correct") else 0.0 for row in group]),
            "strict_accuracy": _mean([1.0 if _strict_correct(row) else 0.0 for row in group]),
            "malformed_rate": _mean(malformed),
            "path_found_rate": _mean(path_found),
            "mean_path_edge_count": _mean(edge_counts),
            "mean_path_node_count": _mean(node_counts),
        })
    return summary_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_key",
        "model",
        "dataset",
        "split",
        "linker",
        "mode",
        "condition",
        "depth",
        "n",
        "accuracy",
        "strict_accuracy",
        "malformed_rate",
        "path_found_rate",
        "mean_path_edge_count",
        "mean_path_node_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.6f}" if isinstance(value, float) else value
                for key, value in row.items()
            })


def write_markdown(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Q-to-Answer Exact-Depth Analysis", ""]
    current_group: tuple[str, str, str, str] | None = None
    for row in rows:
        group = (row["model"], row["dataset"], row["linker"], row["mode"])
        if group != current_group:
            model, dataset, linker, mode = group
            lines.extend(["", f"## {model} / {dataset} / {linker} / {mode}", ""])
            lines.append("| Depth | N | Accuracy | Strict accuracy | Path found | Malformed |")
            lines.append("|---:|---:|---:|---:|---:|---:|")
            current_group = group
        lines.append(
            f"| {row['depth']} | {row['n']} | {row['accuracy']:.3f} | "
            f"{row['strict_accuracy']:.3f} | {row['path_found_rate']:.3f} | "
            f"{row['malformed_rate']:.3f} |"
        )
    if not rows:
        lines.append("- No exact-depth Q-to-answer prediction rows found.")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build STEP 3 exact Q-to-answer depth tables.")
    parser.add_argument("--pred-jsonl", required=True, nargs="+", type=Path)
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict] = []
    for path in args.pred_jsonl:
        rows.extend(_load_jsonl(path))
    summary_rows = analyse(rows, tuple(args.conditions))

    out_csv = args.output_dir / "q_to_a_depth_accuracy.csv"
    out_md = args.output_dir / "q_to_a_depth_accuracy.md"
    write_csv(out_csv, summary_rows)
    write_markdown(out_md, summary_rows)
    print(f"Wrote {out_csv} ({len(summary_rows)} rows)")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
