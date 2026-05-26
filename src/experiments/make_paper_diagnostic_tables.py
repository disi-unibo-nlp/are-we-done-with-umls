from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model_sweeps.uncertainty_metrics import bootstrap_ci, group_rows, wilson_interval


@dataclass(frozen=True)
class PaperTableOutputs:
    evidence_views: Path
    answer_grounding: Path
    retrieval_faithfulness: Path
    diagnostic_summary: Path


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_row(row: dict) -> dict:
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        out = dict(row)
        out["metrics"] = dict(metrics)
        return out
    identity = {
        key: row.get(key)
        for key in ("item_id", "dataset", "split", "linker", "experiment", "condition")
        if key in row
    }
    condition = identity.get("condition") or row.get("view") or row.get("method") or "item"
    identity["condition"] = condition
    metrics = {
        key: value
        for key, value in row.items()
        if key not in identity and key not in {"view", "method"}
    }
    identity["metrics"] = metrics
    return identity


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        rows.extend(normalize_row(row) for row in read_jsonl(path))
    return rows


def _bool_summary(group: list[dict], metric: str) -> dict[str, object]:
    values = [
        row["metrics"][metric]
        for row in group
        if isinstance(row.get("metrics"), dict) and isinstance(row["metrics"].get(metric), bool)
    ]
    if not values:
        return {f"{metric}_rate": "", f"{metric}_ci_low": "", f"{metric}_ci_high": ""}
    successes = sum(1 for value in values if value)
    ci_low, ci_high = wilson_interval(successes, len(values))
    return {
        f"{metric}_rate": successes / len(values),
        f"{metric}_ci_low": ci_low,
        f"{metric}_ci_high": ci_high,
    }


def _mean_summary(
    group: list[dict],
    metric: str,
    bootstrap_iterations: int,
    seed: int = 13,
) -> dict[str, object]:
    values = [
        float(row["metrics"][metric])
        for row in group
        if isinstance(row.get("metrics"), dict)
        and isinstance(row["metrics"].get(metric), (int, float))
        and not isinstance(row["metrics"].get(metric), bool)
    ]
    if not values:
        return {f"mean_{metric}": "", f"{metric}_ci_low": "", f"{metric}_ci_high": ""}
    ci_low, ci_high = bootstrap_ci(values, iterations=max(1, bootstrap_iterations), seed=seed)
    return {
        f"mean_{metric}": fmean(values),
        f"{metric}_ci_low": ci_low,
        f"{metric}_ci_high": ci_high,
    }


def _format_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key, "")) for key in fieldnames})


def evidence_view_rows(rows: list[dict], bootstrap_iterations: int) -> list[dict]:
    selected = [row for row in rows if row.get("experiment") == "evidence_availability"]
    groups = group_rows(selected, ("dataset", "split", "linker", "condition"))
    out: list[dict] = []
    for key, group in sorted(groups.items()):
        dataset, split, linker, condition = key
        row = {"dataset": dataset, "split": split, "linker": linker, "condition": condition, "n": len(group)}
        for metric in (
            "gold_answer_grounded",
            "gold_answer_evidence_available",
            "distractor_contrast_available",
            "evidence_view_adds_new_signal_over_relation_only",
        ):
            row.update(_bool_summary(group, metric))
        row.update(_mean_summary(group, "evidence_token_estimate", bootstrap_iterations))
        out.append(row)
    return out


def answer_grounding_rows(rows: list[dict], bootstrap_iterations: int) -> list[dict]:
    selected = [row for row in rows if row.get("experiment") == "answer_grounding"]
    groups = group_rows(selected, ("dataset", "split", "linker"))
    out: list[dict] = []
    for key, group in sorted(groups.items()):
        dataset, split, linker = key
        row = {"dataset": dataset, "split": split, "linker": linker, "n": len(group)}
        for metric in (
            "question_grounded",
            "correct_answer_grounded",
            "all_options_grounded",
            "any_option_grounded",
        ):
            row.update(_bool_summary(group, metric))
        row.update(_mean_summary(group, "distractor_grounding_rate", bootstrap_iterations))
        out.append(row)
    return out


def retrieval_faithfulness_rows(rows: list[dict], bootstrap_iterations: int) -> list[dict]:
    selected = [row for row in rows if row.get("experiment") == "retrieval_faithfulness"]
    groups = group_rows(selected, ("dataset", "split", "linker", "condition"))
    decision_relevance: dict[tuple[str, str, str, str], float] = {}
    out: list[dict] = []
    for key, group in sorted(groups.items()):
        dataset, split, linker, condition = key
        row = {"dataset": dataset, "split": split, "linker": linker, "condition": condition, "n": len(group)}
        row.update(_bool_summary(group, "gold_path_available"))
        for metric in (
            "gold_min_path_length",
            "distractor_path_rate",
            "num_retrieved_nodes",
            "num_retrieved_edges",
            "decision_relevance_ratio_proxy",
        ):
            row.update(_mean_summary(group, metric, bootstrap_iterations))
        relevance = row.get("mean_decision_relevance_ratio_proxy")
        if isinstance(relevance, float):
            decision_relevance[key] = relevance
        out.append(row)

    random_relevance = {
        key[:3]: value
        for key, value in decision_relevance.items()
        if key[3] == "random_same_size"
    }
    for row in out:
        relevance = row.get("mean_decision_relevance_ratio_proxy")
        random_value = random_relevance.get((row["dataset"], row["split"], row["linker"]))
        row["random_control_gap"] = (
            relevance - random_value
            if isinstance(relevance, float) and isinstance(random_value, float)
            else ""
        )
    return out


def write_summary_md(path: Path, evidence: list[dict], answer: list[dict], retrieval: list[dict]) -> None:
    lines = [
        "# Diagnostic Summary",
        "",
        "## Experiment 1: Evidence availability and enrichment",
        f"Rows: {sum(int(row['n']) for row in evidence) if evidence else 0}",
        "",
        "## Experiment 2: Answer-option grounding",
        f"Rows: {sum(int(row['n']) for row in answer) if answer else 0}",
        "",
        "## Experiment 3: Retrieval faithfulness",
        f"Rows: {sum(int(row['n']) for row in retrieval) if retrieval else 0}",
        "",
        "Intervals use Wilson intervals for binary rates and bootstrap intervals for means.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_paper_tables(
    inputs: list[Path],
    output_dir: Path,
    bootstrap_iterations: int = 1000,
) -> PaperTableOutputs:
    rows = load_rows(inputs)
    evidence = evidence_view_rows(rows, bootstrap_iterations)
    answer = answer_grounding_rows(rows, bootstrap_iterations)
    retrieval = retrieval_faithfulness_rows(rows, bootstrap_iterations)

    outputs = PaperTableOutputs(
        evidence_views=output_dir / "table_evidence_views.csv",
        answer_grounding=output_dir / "table_answer_grounding.csv",
        retrieval_faithfulness=output_dir / "table_retrieval_faithfulness.csv",
        diagnostic_summary=output_dir / "table_diagnostic_summary.md",
    )
    _write_csv(outputs.evidence_views, evidence, sorted({key for row in evidence for key in row}) or ["n"])
    _write_csv(outputs.answer_grounding, answer, sorted({key for row in answer for key in row}) or ["n"])
    _write_csv(outputs.retrieval_faithfulness, retrieval, sorted({key for row in retrieval for key in row}) or ["n"])
    write_summary_md(outputs.diagnostic_summary, evidence, answer, retrieval)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build paper diagnostic tables around the three UMLS experiments.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "paper_tables")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = build_paper_tables(args.inputs, args.output_dir, args.bootstrap_iterations)
    print(json.dumps({key: str(value) for key, value in outputs.__dict__.items()}, indent=2))


if __name__ == "__main__":
    main()
