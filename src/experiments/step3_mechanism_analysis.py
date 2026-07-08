from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


RELATION_RE = re.compile(r"--\s*(.*?)\s*-->")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_many_jsonl(paths: Sequence[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denominator = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def condition_depth(condition: str, prefix: str) -> int | None:
    suffix = "_evidence"
    if not condition.startswith(prefix) or not condition.endswith(suffix):
        return None
    raw_depth = condition[len(prefix):-len(suffix)]
    try:
        return int(raw_depth)
    except ValueError:
        return None


def _row_condition(row: dict) -> str:
    return str(row.get("condition") or row.get("evidence_condition") or row.get("prompt_condition") or "")


def _row_model(row: dict) -> str:
    return str(row.get("model") or row.get("model_id") or "unknown")


def _row_item_id(row: dict) -> str:
    return str(row.get("item_id") or row.get("id") or "")


def _row_malformed(row: dict) -> bool:
    return bool(row.get("malformed") or row.get("is_malformed") or row.get("parse_failed"))


def _row_correct(row: dict) -> bool:
    if "is_correct" in row:
        return bool(row["is_correct"])
    prediction = row.get("prediction")
    gold = row.get("gold")
    return bool(prediction and gold and prediction == gold)


def summarize_prediction_rows(rows: Iterable[dict], condition_prefix: str) -> list[dict]:
    grouped: dict[tuple[str, int, str], dict[str, int | str]] = {}
    for row in rows:
        condition = _row_condition(row)
        depth = condition_depth(condition, condition_prefix)
        if depth is None:
            continue
        model = _row_model(row)
        key = (model, depth, condition)
        bucket = grouped.setdefault(
            key,
            {
                "model": model,
                "condition": condition,
                "depth": depth,
                "n": 0,
                "valid": 0,
                "malformed": 0,
                "correct": 0,
            },
        )
        bucket["n"] = int(bucket["n"]) + 1
        if _row_malformed(row):
            bucket["malformed"] = int(bucket["malformed"]) + 1
            continue
        bucket["valid"] = int(bucket["valid"]) + 1
        if _row_correct(row):
            bucket["correct"] = int(bucket["correct"]) + 1

    out: list[dict] = []
    for bucket in grouped.values():
        valid = int(bucket["valid"])
        correct = int(bucket["correct"])
        accuracy = correct / valid if valid else 0.0
        ci_low, ci_high = wilson_interval(correct, valid)
        out.append(
            {
                **bucket,
                "accuracy": accuracy,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return sorted(out, key=lambda item: (str(item["model"]), int(item["depth"])))


def _exact_sign_p_value(first_wins: int, reference_wins: int) -> float:
    discordant = first_wins + reference_wins
    if discordant == 0:
        return 1.0
    lower_tail = sum(math.comb(discordant, k) for k in range(min(first_wins, reference_wins) + 1))
    return min(1.0, 2 * lower_tail / (2 ** discordant))


def paired_depth_comparisons(
    rows: Iterable[dict],
    condition_prefix: str,
    depths: Iterable[int],
    reference_depth: int,
) -> list[dict]:
    by_model_item_depth: dict[tuple[str, str, int], bool] = {}
    for row in rows:
        if _row_malformed(row):
            continue
        condition = _row_condition(row)
        depth = condition_depth(condition, condition_prefix)
        if depth is None:
            continue
        item_id = _row_item_id(row)
        if not item_id:
            continue
        by_model_item_depth[(_row_model(row), item_id, depth)] = _row_correct(row)

    models = sorted({model for model, _, _ in by_model_item_depth})
    out: list[dict] = []
    for model in models:
        reference_items = {
            item_id
            for row_model, item_id, depth in by_model_item_depth
            if row_model == model and depth == reference_depth
        }
        for depth in depths:
            comparison_items = {
                item_id
                for row_model, item_id, row_depth in by_model_item_depth
                if row_model == model and row_depth == depth
            }
            common = sorted(reference_items & comparison_items)
            if not common:
                continue
            first_correct = [by_model_item_depth[(model, item_id, depth)] for item_id in common]
            ref_correct = [by_model_item_depth[(model, item_id, reference_depth)] for item_id in common]
            first_wins = sum(1 for first, ref in zip(first_correct, ref_correct) if first and not ref)
            reference_wins = sum(1 for first, ref in zip(first_correct, ref_correct) if ref and not first)
            accuracy = sum(first_correct) / len(common)
            reference_accuracy = sum(ref_correct) / len(common)
            out.append(
                {
                    "model": model,
                    "depth": depth,
                    "reference_depth": reference_depth,
                    "common_n": len(common),
                    "accuracy": accuracy,
                    "reference_accuracy": reference_accuracy,
                    "delta_vs_reference": accuracy - reference_accuracy,
                    "discordant_b": first_wins,
                    "discordant_c": reference_wins,
                    "mcnemar_p": _exact_sign_p_value(first_wins, reference_wins),
                }
            )
    return sorted(out, key=lambda item: (str(item["model"]), int(item["depth"])))


def _evidence_text(row: dict) -> str:
    for key in ("evidence_text", "text", "evidence", "umls_evidence", "prompt", "input"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def relation_label_counts(rows: Iterable[dict], condition_prefix: str) -> dict[int, Counter[str]]:
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        condition = _row_condition(row)
        depth = condition_depth(condition, condition_prefix)
        if depth is None:
            continue
        for label in RELATION_RE.findall(_evidence_text(row)):
            cleaned = " ".join(label.strip().split())
            if cleaned:
                counts[depth][cleaned] += 1
    return dict(sorted(counts.items()))


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def coverage_rows_from_paths(named_paths: Sequence[tuple[str, Path]]) -> list[dict]:
    return [
        {"source": label, "count": count_jsonl_rows(path)}
        for label, path in named_paths
        if path.exists()
    ]


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _summary_table(rows: Sequence[dict]) -> list[str]:
    if not rows:
        return ["No rows available."]
    lines = [
        "| Model | Depth | N | Accuracy | 95% CI | Valid | Malformed | Correct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {depth} | {n} | {acc} | {lo}-{hi} | {valid} | {malformed} | {correct} |".format(
                model=row["model"],
                depth=row["depth"],
                n=row["n"],
                acc=_fmt(float(row["accuracy"])),
                lo=_fmt(float(row["ci_low"])),
                hi=_fmt(float(row["ci_high"])),
                valid=row["valid"],
                malformed=row["malformed"],
                correct=row["correct"],
            )
        )
    return lines


def _pairwise_table(rows: Sequence[dict]) -> list[str]:
    if not rows:
        return ["No paired comparisons available."]
    lines = [
        "| Model | Depth | Reference | Common N | Accuracy | Reference Accuracy | Delta | b | c | p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {depth} | {ref} | {n} | {acc} | {ref_acc} | {delta} | {b} | {c} | {p} |".format(
                model=row["model"],
                depth=row["depth"],
                ref=row["reference_depth"],
                n=row["common_n"],
                acc=_fmt(float(row["accuracy"])),
                ref_acc=_fmt(float(row["reference_accuracy"])),
                delta=_fmt(float(row["delta_vs_reference"])),
                b=row["discordant_b"],
                c=row["discordant_c"],
                p=_fmt(float(row["mcnemar_p"])),
            )
        )
    return lines


def _relation_table(relation_counts: dict[int, Counter[str]], limit: int = 12) -> list[str]:
    if not relation_counts:
        return ["No relation-label evidence text was provided."]
    lines = [
        "| Depth | Relation label | Count |",
        "|---:|---|---:|",
    ]
    for depth, counts in sorted(relation_counts.items()):
        counter = counts if isinstance(counts, Counter) else Counter(counts)
        for label, count in counter.most_common(limit):
            lines.append(f"| {depth} | {label} | {count} |")
    return lines


def _coverage_table(rows: Sequence[dict]) -> list[str]:
    if not rows:
        return ["No coverage inputs were provided."]
    lines = [
        "| Source | Count |",
        "|---|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['source']} | {row['count']} |")
    return lines


def build_markdown_report(
    faithful_summary: Sequence[dict] | None = None,
    shortcut_summary: Sequence[dict] | None = None,
    shortcut_pairwise: Sequence[dict] | None = None,
    faithful_pairwise: Sequence[dict] | None = None,
    relation_counts: dict[int, Counter[str]] | None = None,
    coverage_rows: Sequence[dict] | None = None,
) -> str:
    faithful_summary = faithful_summary or []
    shortcut_summary = shortcut_summary or []
    shortcut_pairwise = shortcut_pairwise or []
    faithful_pairwise = faithful_pairwise or []
    relation_counts = relation_counts or {}
    coverage_rows = coverage_rows or []

    lines = [
        "# STEP 3 Mechanism Analysis",
        "",
        "Central interpretation: UMLS graph distance is not semantic reasoning depth. The depth-D prompt interventions test a visible graph-topology perturbation, but UMLS paths mix biomedical relations with ontology-maintenance relations such as qualifiers, subsets, mappings, and broad/narrow hierarchy bookkeeping. A monotonic depth story is therefore not the right mechanism claim.",
        "",
        "The stronger claim to carry into the paper is that path availability is sparse under strict balanced controls, and once balanced prompts are constructed, shortcut depth changes do not produce a reliable causal ordering. This points to relation semantics and ontology topology as the source of variance, not simply the number of displayed edges.",
        "",
        "## Faithful Exact-Depth Runs",
        "",
        *_summary_table(faithful_summary),
        "",
        "## Synthetic Shortcut Runs",
        "",
        *_summary_table(shortcut_summary),
        "",
        "## Shortcut Pairwise Deltas Against D4",
        "",
        *_pairwise_table(shortcut_pairwise),
        "",
        "## Faithful Common-Item Deltas Against D4",
        "",
        *_pairwise_table(faithful_pairwise),
        "",
        "## Relation Labels Observed In Faithful Evidence",
        "",
        *_relation_table(relation_counts),
        "",
        "## Coverage Funnel",
        "",
        *_coverage_table(coverage_rows),
        "",
        "## Paper-Facing Use",
        "",
        "Use STEP3 as an identifiability result: UMLS evidence is not automatically a chain-of-reasoning substrate. The operational signal depends on which relation families survive construction, whether every answer option can be controlled at the same path depth, and whether the displayed graph contains medically discriminative relations rather than ontology scaffolding.",
        "",
    ]
    return "\n".join(lines)


def _merge_relation_counts(*counts_by_depth: dict[int, Counter[str]]) -> dict[int, Counter[str]]:
    merged: dict[int, Counter[str]] = defaultdict(Counter)
    for counts in counts_by_depth:
        for depth, counter in counts.items():
            merged[depth].update(counter)
    return dict(sorted(merged.items()))


def build_report_from_rows(
    faithful_rows: Sequence[dict],
    shortcut_rows: Sequence[dict],
    faithful_cache_rows: Sequence[dict] | None = None,
    shortcut_cache_rows: Sequence[dict] | None = None,
    coverage_rows: Sequence[dict] | None = None,
) -> str:
    faithful_summary = summarize_prediction_rows(faithful_rows, "q_to_a_exact_depth_")
    shortcut_summary = summarize_prediction_rows(shortcut_rows, "q_to_a_d4_shortcut_")
    faithful_pairwise = paired_depth_comparisons(
        faithful_rows,
        "q_to_a_exact_depth_",
        depths=(1, 2, 3),
        reference_depth=4,
    )
    shortcut_pairwise = paired_depth_comparisons(
        shortcut_rows,
        "q_to_a_d4_shortcut_",
        depths=(1, 2, 3),
        reference_depth=4,
    )
    relation_counts = relation_label_counts(faithful_cache_rows or [], "q_to_a_exact_depth_")
    if not relation_counts:
        relation_counts = relation_label_counts(shortcut_cache_rows or [], "q_to_a_d4_shortcut_")
    return build_markdown_report(
        faithful_summary=faithful_summary,
        shortcut_summary=shortcut_summary,
        shortcut_pairwise=shortcut_pairwise,
        faithful_pairwise=faithful_pairwise,
        relation_counts=relation_counts,
        coverage_rows=coverage_rows or [],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the STEP3 UMLS-depth mechanism analysis report.")
    parser.add_argument("--faithful-pred-jsonl", nargs="*", type=Path, default=[])
    parser.add_argument("--shortcut-pred-jsonl", nargs="*", type=Path, default=[])
    parser.add_argument("--faithful-cache-jsonl", nargs="*", type=Path, default=[])
    parser.add_argument("--shortcut-cache-jsonl", nargs="*", type=Path, default=[])
    parser.add_argument("--coverage-jsonl", action="append", nargs=2, metavar=("LABEL", "PATH"), default=[])
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args(argv)

    faithful_rows = read_many_jsonl(args.faithful_pred_jsonl)
    shortcut_rows = read_many_jsonl(args.shortcut_pred_jsonl)
    faithful_cache_rows = read_many_jsonl(args.faithful_cache_jsonl)
    shortcut_cache_rows = read_many_jsonl(args.shortcut_cache_jsonl)
    coverage_rows = coverage_rows_from_paths([(label, Path(path)) for label, path in args.coverage_jsonl])
    report = build_report_from_rows(
        faithful_rows,
        shortcut_rows,
        faithful_cache_rows=faithful_cache_rows,
        shortcut_cache_rows=shortcut_cache_rows,
        coverage_rows=coverage_rows,
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
