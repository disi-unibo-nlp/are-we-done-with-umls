"""Graph-distance mechanism cache, summaries, and reports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from src.experiments.cli_utils import collect_cuis, default_input_path, default_reference_path, load_items
from src.experiments.evidence_cards import (
    EvidenceCardDetails,
    _is_administrative_relation_label,
    _q_to_a_shortest_path_lines,
)
from src.experiments.schemas import QAItem
from src.experiments.umls_index import CuiMetadataIndex, RelationIndex
from src.model_sweeps.downstream_umls_probe import evidence_card_to_cache_row


GRAPH_DISTANCE_CONDITIONS = (
    "q_to_a_raw_shortest_path_evidence",
    "q_to_a_semantic_shortest_path_evidence",
    "q_to_a_topology_only_path_evidence",
)
STRATA = (
    "semantic_gold_closer",
    "semantic_distractor_closer",
    "raw_only_gold_closer",
    "tie_control",
)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def option_distances(item: QAItem, relations: RelationIndex, max_hops: int = 4) -> dict[str, int | None]:
    distances: dict[str, int | None] = {}
    for label, option_text in item.options.items():
        if not option_text:
            continue
        cuis = item.option_cuis.get(label, [])
        distances[label] = relations.shortest_path_length(item.question_cuis, cuis, max_hops=max_hops) if cuis else None
    return distances


def distance_margin(distances: dict[str, int | None], gold: str | None) -> int | None:
    if gold is None or gold not in distances or distances[gold] is None:
        return None
    if any(value is None for value in distances.values()):
        return None
    distractor_distances = [
        value
        for label, value in distances.items()
        if label != gold and value is not None
    ]
    if not distractor_distances:
        return None
    return min(distractor_distances) - int(distances[gold])


def closest_options(distances: dict[str, int | None]) -> list[str]:
    finite = {label: value for label, value in distances.items() if value is not None}
    if not finite:
        return []
    best = min(finite.values())
    return sorted(label for label, value in finite.items() if value == best)


def unique_closest_option(distances: dict[str, int | None]) -> str:
    closest = closest_options(distances)
    return closest[0] if len(closest) == 1 else ""


def classify_distance_signal(distances: dict[str, int | None], gold: str | None) -> str:
    if gold is None or gold not in distances:
        return "no_gold"
    if distances[gold] is None:
        return "gold_missing"
    if any(value is None for value in distances.values()):
        return "partial"
    closest = closest_options(distances)
    if closest == [gold]:
        return "gold_unique_closest"
    if gold in closest:
        return "gold_tied_closest"
    return "gold_not_closest"


def assign_stratum(row: dict) -> str | None:
    semantic_margin = row.get("semantic_margin")
    raw_margin = row.get("raw_margin")
    if semantic_margin is None or raw_margin is None:
        return None
    semantic_margin = int(semantic_margin)
    raw_margin = int(raw_margin)
    if semantic_margin >= 1:
        return "semantic_gold_closer"
    if semantic_margin <= -1 and raw_margin <= 0:
        return "semantic_distractor_closer"
    if raw_margin >= 1 and semantic_margin <= 0:
        return "raw_only_gold_closer"
    if raw_margin == 0 and semantic_margin == 0:
        return "tie_control"
    return None


def option_distance_row(
    item: QAItem,
    raw_relations: RelationIndex,
    semantic_relations: RelationIndex,
    max_hops: int = 4,
) -> dict:
    raw_distances = option_distances(item, raw_relations, max_hops=max_hops)
    semantic_distances = option_distances(item, semantic_relations, max_hops=max_hops)
    row = {
        "item_id": item.item_id,
        "dataset": item.dataset,
        "split": item.split,
        "linker": item.linker,
        "gold": item.gold or "",
        "raw_distances": raw_distances,
        "semantic_distances": semantic_distances,
        "raw_margin": distance_margin(raw_distances, item.gold),
        "semantic_margin": distance_margin(semantic_distances, item.gold),
        "raw_signal": classify_distance_signal(raw_distances, item.gold),
        "semantic_signal": classify_distance_signal(semantic_distances, item.gold),
        "raw_closest_option": unique_closest_option(raw_distances),
        "semantic_closest_option": unique_closest_option(semantic_distances),
    }
    row["stratum"] = assign_stratum(row)
    return row


def _stable_rng(seed: int, stratum: str) -> random.Random:
    digest = hashlib.sha256(stratum.encode("utf-8")).hexdigest()
    return random.Random(seed + int(digest[:8], 16))


def select_stratified_items(
    rows: Iterable[dict],
    max_per_stratum: int = 100,
    min_size: int = 30,
    seed: int = 42,
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        item_id = str(row.get("item_id", ""))
        stratum = row.get("stratum") or assign_stratum(row)
        if not item_id or stratum not in STRATA or item_id in seen:
            continue
        grouped[str(stratum)].append(item_id)
        seen.add(item_id)

    selected: dict[str, list[str]] = {}
    for stratum in STRATA:
        item_ids = list(grouped.get(stratum, []))
        if len(item_ids) < min_size:
            continue
        rng = _stable_rng(seed, stratum)
        rng.shuffle(item_ids)
        selected[stratum] = sorted(item_ids[:max_per_stratum])
    return selected


def _graph_distance_card_details(
    item: QAItem,
    condition: str,
    metadata: CuiMetadataIndex,
    raw_relations: RelationIndex,
    semantic_relations: RelationIndex,
) -> EvidenceCardDetails:
    if condition == "q_to_a_raw_shortest_path_evidence":
        relations = raw_relations
        semantics = "shortest_paths"
        topology_only = False
    elif condition == "q_to_a_semantic_shortest_path_evidence":
        relations = semantic_relations
        semantics = "semantic_shortest_paths"
        topology_only = False
    elif condition == "q_to_a_topology_only_path_evidence":
        relations = raw_relations
        semantics = "topology_only_shortest_paths"
        topology_only = True
    else:
        raise ValueError(f"Unsupported graph-distance condition: {condition}")

    (
        graph_lines,
        node_count,
        edge_count,
        answer_count,
        covered_count,
        selection_policy,
        linked_options_complete,
        max_depth,
    ) = _q_to_a_shortest_path_lines(item, relations, metadata, topology_only=topology_only)
    return EvidenceCardDetails(
        "\n".join(graph_lines),
        q_to_a_depth_semantics=semantics,
        q_to_a_depth=max_depth,
        q_to_a_path_found=linked_options_complete and answer_count > 0 and covered_count == answer_count,
        q_to_a_path_node_count=node_count,
        q_to_a_path_edge_count=edge_count,
        q_to_a_answer_cui_count=answer_count,
        q_to_a_answer_cui_covered_count=covered_count,
        q_to_a_path_selection_policy=selection_policy,
    )


def build_cache_rows(
    items: Sequence[QAItem],
    metadata: CuiMetadataIndex,
    raw_relations: RelationIndex,
    semantic_relations: RelationIndex,
    *,
    max_hops: int = 4,
    max_per_stratum: int = 100,
    min_stratum_size: int = 30,
    seed: int = 42,
    conditions: Sequence[str] = GRAPH_DISTANCE_CONDITIONS,
) -> tuple[list[dict], list[dict], list[dict]]:
    audit_rows = [
        option_distance_row(item, raw_relations, semantic_relations, max_hops=max_hops)
        for item in items
    ]
    selected = select_stratified_items(
        audit_rows,
        max_per_stratum=max_per_stratum,
        min_size=min_stratum_size,
        seed=seed,
    )
    stratum_by_item = {
        item_id: stratum
        for stratum, item_ids in selected.items()
        for item_id in item_ids
    }
    audit_by_item = {row["item_id"]: row for row in audit_rows}
    item_by_id = {item.item_id: item for item in items}
    manifest_rows: list[dict] = []
    cache_rows: list[dict] = []
    for item_id in sorted(stratum_by_item):
        item = item_by_id[item_id]
        distance_row = audit_by_item[item_id]
        cards: list[tuple[str, EvidenceCardDetails]] = []
        failed = False
        for condition in conditions:
            card = _graph_distance_card_details(item, condition, metadata, raw_relations, semantic_relations)
            if not card.q_to_a_path_found:
                failed = True
                break
            cards.append((condition, card))
        if failed:
            continue
        manifest_rows.append({"item_id": item_id, "stratum": stratum_by_item[item_id]})
        for condition, card in cards:
            row = evidence_card_to_cache_row(item, condition, card)
            row.update({
                "stratum": stratum_by_item[item_id],
                "raw_margin": distance_row["raw_margin"],
                "semantic_margin": distance_row["semantic_margin"],
                "raw_signal": distance_row["raw_signal"],
                "semantic_signal": distance_row["semantic_signal"],
                "raw_closest_option": distance_row["raw_closest_option"],
                "semantic_closest_option": distance_row["semantic_closest_option"],
            })
            if condition == "q_to_a_semantic_shortest_path_evidence":
                closest = distance_row["semantic_closest_option"]
            else:
                closest = distance_row["raw_closest_option"]
            row["closest_option_for_condition"] = closest
            row["gold_closest_option"] = closest
            cache_rows.append(row)
    return audit_rows, manifest_rows, cache_rows


def _row_condition(row: dict) -> str:
    return str(row.get("condition") or row.get("evidence_condition") or row.get("prompt_condition") or "")


def _row_model(row: dict) -> str:
    return str(row.get("model") or row.get("model_id") or "unknown")


def _row_malformed(row: dict) -> bool:
    return bool(row.get("malformed") or row.get("is_malformed") or row.get("parse_failed") or row.get("prediction") is None)


def _row_correct(row: dict) -> bool:
    if "is_correct" in row:
        return bool(row["is_correct"])
    return bool(row.get("prediction") and row.get("gold") and row.get("prediction") == row.get("gold"))


def _condition_closest_option(condition: str, cache_row: dict) -> str:
    if cache_row.get("closest_option_for_condition"):
        return str(cache_row["closest_option_for_condition"])
    if condition == "q_to_a_semantic_shortest_path_evidence" and cache_row.get("semantic_closest_option"):
        return str(cache_row["semantic_closest_option"])
    if condition in {"q_to_a_raw_shortest_path_evidence", "q_to_a_topology_only_path_evidence"} and cache_row.get("raw_closest_option"):
        return str(cache_row["raw_closest_option"])
    return str(cache_row.get("gold_closest_option") or "")


def summarize_prediction_rows(prediction_rows: Iterable[dict], cache_rows: Iterable[dict]) -> list[dict]:
    cache_by_item_condition: dict[tuple[str, str], dict] = {}
    cache_by_item: dict[str, dict] = {}
    for row in cache_rows:
        item_id = str(row.get("item_id", ""))
        condition = _row_condition(row)
        if item_id:
            cache_by_item.setdefault(item_id, row)
            if condition:
                cache_by_item_condition[(item_id, condition)] = row

    groups: dict[tuple[str, str, str], dict] = {}
    for row in prediction_rows:
        item_id = str(row.get("item_id", ""))
        condition = _row_condition(row)
        cache_row = cache_by_item_condition.get((item_id, condition), cache_by_item.get(item_id))
        if not cache_row:
            continue
        stratum = str(cache_row.get("stratum") or "")
        if not stratum:
            continue
        key = (_row_model(row), stratum, condition)
        bucket = groups.setdefault(
            key,
            {
                "model": _row_model(row),
                "stratum": stratum,
                "condition": condition,
                "n": 0,
                "valid": 0,
                "malformed": 0,
                "correct": 0,
                "closest_option_attractions": 0,
                "closest_option_attraction_n": 0,
            },
        )
        bucket["n"] += 1
        if _row_malformed(row):
            bucket["malformed"] += 1
            continue
        bucket["valid"] += 1
        if _row_correct(row):
            bucket["correct"] += 1
        closest = _condition_closest_option(condition, cache_row)
        if closest:
            bucket["closest_option_attraction_n"] += 1
            if str(row.get("prediction", "")) == closest:
                bucket["closest_option_attractions"] += 1

    out: list[dict] = []
    for bucket in groups.values():
        valid = int(bucket["valid"])
        attraction_n = int(bucket["closest_option_attraction_n"])
        out.append({
            **bucket,
            "accuracy": int(bucket["correct"]) / valid if valid else 0.0,
            "closest_option_attraction_rate": int(bucket["closest_option_attractions"]) / attraction_n if attraction_n else 0.0,
        })
    return sorted(out, key=lambda row: (row["model"], row["stratum"], row["condition"]))


def _audit_table(audit_rows: Sequence[dict]) -> list[str]:
    counts = Counter(row.get("stratum") or "unassigned" for row in audit_rows)
    lines = ["| Stratum | N |", "|---|---:|"]
    for key, count in sorted(counts.items()):
        lines.append(f"| {key} | {count} |")
    return lines


def _summary_table(summary_rows: Sequence[dict]) -> list[str]:
    if not summary_rows:
        return ["No prediction rows available."]
    lines = [
        "| Model | Stratum | Condition | N | Accuracy | Closest-option attraction | Malformed |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {model} | {stratum} | {condition} | {n} | {acc:.3f} | {attr:.3f} | {malformed} |".format(
                model=row["model"],
                stratum=row["stratum"],
                condition=row["condition"],
                n=row["n"],
                acc=float(row["accuracy"]),
                attr=float(row["closest_option_attraction_rate"]),
                malformed=row["malformed"],
            )
        )
    return lines


def build_markdown_report(summary_rows: Sequence[dict], audit_rows: Sequence[dict]) -> str:
    lines = [
        "# Graph-Distance Mechanism Report",
        "",
        "Claim: graph distance acts as a salience prior, not a correctness guarantee. The key test is whether semantic filtering makes the gold answer uniquely closer than distractors, and whether raw/topology-only prompts over-attract the model toward graph-nearest options.",
        "",
        "## Distance Audit",
        "",
        *_audit_table(audit_rows),
        "",
        "## LLM Results By Distance Stratum",
        "",
        *_summary_table(summary_rows),
        "",
    ]
    return "\n".join(lines)


def write_summary_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "stratum",
        "condition",
        "n",
        "valid",
        "malformed",
        "correct",
        "accuracy",
        "closest_option_attractions",
        "closest_option_attraction_n",
        "closest_option_attraction_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_many_jsonl(paths: Sequence[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def _build_cache_command(args: argparse.Namespace) -> int:
    items = load_items(
        args.dataset,
        args.split,
        args.linker,
        input_path=args.input,
        reference_path=args.reference,
        limit=args.limit if args.limit > 0 else None,
        item_id_file=args.item_id_file,
    )
    metadata = CuiMetadataIndex.from_zip(args.metadata_zip)
    keep_cuis = None if args.relation_scope == "full" else collect_cuis(items)
    raw_relations = RelationIndex.from_zip(args.triplets_zip, keep_cuis=keep_cuis)
    semantic_relations = raw_relations.filter_edges(_is_administrative_relation_label)
    audit_rows, manifest_rows, cache_rows = build_cache_rows(
        items,
        metadata,
        raw_relations,
        semantic_relations,
        max_hops=args.max_hops,
        max_per_stratum=args.max_per_stratum,
        min_stratum_size=args.min_stratum_size,
        seed=args.seed,
        conditions=args.condition or GRAPH_DISTANCE_CONDITIONS,
    )
    write_jsonl(args.audit_jsonl, audit_rows)
    write_jsonl(args.manifest_jsonl, manifest_rows)
    write_jsonl(args.evidence_cache_jsonl, cache_rows)
    print(json.dumps({
        "event": "graph_distance_cache_done",
        "items": len(items),
        "audit_rows": len(audit_rows),
        "manifest_rows": len(manifest_rows),
        "cache_rows": len(cache_rows),
        "audit_jsonl": str(args.audit_jsonl),
        "manifest_jsonl": str(args.manifest_jsonl),
        "evidence_cache_jsonl": str(args.evidence_cache_jsonl),
    }, indent=2))
    return 0


def _summarize_command(args: argparse.Namespace) -> int:
    cache_rows = read_jsonl(args.evidence_cache_jsonl)
    audit_rows = read_jsonl(args.audit_jsonl)
    prediction_rows = _load_many_jsonl(args.pred_jsonl)
    summary_rows = summarize_prediction_rows(prediction_rows, cache_rows)
    if args.summary_csv:
        write_summary_csv(args.summary_csv, summary_rows)
    if args.summary_jsonl:
        write_jsonl(args.summary_jsonl, summary_rows)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(build_markdown_report(summary_rows, audit_rows), encoding="utf-8")
    return 0


def _write_report_command(args: argparse.Namespace) -> int:
    summary_rows = read_jsonl(args.summary_jsonl)
    audit_rows = read_jsonl(args.audit_jsonl)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown_report(summary_rows, audit_rows), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and analyze graph-distance mechanism experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_cache = subparsers.add_parser("build-cache")
    build_cache.add_argument("--dataset", default="medQA")
    build_cache.add_argument("--split", default="test")
    build_cache.add_argument("--linker", default="spacy")
    build_cache.add_argument("--input", type=Path, default=None)
    build_cache.add_argument("--reference", type=Path, default=None)
    build_cache.add_argument("--item-id-file", type=Path, default=None)
    build_cache.add_argument("--limit", type=int, default=0)
    build_cache.add_argument("--metadata-zip", type=Path, default=Path("data/UMLS/cui_detail.tsv.zip"))
    build_cache.add_argument("--triplets-zip", type=Path, default=Path("data/UMLS/triplets.csv.zip"))
    build_cache.add_argument("--relation-scope", choices=("full", "item_cuis"), default="full")
    build_cache.add_argument("--max-hops", type=int, default=4)
    build_cache.add_argument("--max-per-stratum", type=int, default=100)
    build_cache.add_argument("--min-stratum-size", type=int, default=30)
    build_cache.add_argument("--seed", type=int, default=42)
    build_cache.add_argument("--condition", action="append", choices=GRAPH_DISTANCE_CONDITIONS)
    build_cache.add_argument("--audit-jsonl", type=Path, required=True)
    build_cache.add_argument("--manifest-jsonl", type=Path, required=True)
    build_cache.add_argument("--evidence-cache-jsonl", type=Path, required=True)
    build_cache.set_defaults(func=_build_cache_command)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--pred-jsonl", nargs="+", type=Path, required=True)
    summarize.add_argument("--evidence-cache-jsonl", type=Path, required=True)
    summarize.add_argument("--audit-jsonl", type=Path, required=True)
    summarize.add_argument("--summary-csv", type=Path, default=None)
    summarize.add_argument("--summary-jsonl", type=Path, default=None)
    summarize.add_argument("--output-md", type=Path, default=None)
    summarize.set_defaults(func=_summarize_command)

    write_report = subparsers.add_parser("write-report")
    write_report.add_argument("--summary-jsonl", type=Path, required=True)
    write_report.add_argument("--audit-jsonl", type=Path, required=True)
    write_report.add_argument("--output-md", type=Path, required=True)
    write_report.set_defaults(func=_write_report_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-cache":
        args.input = args.input or default_input_path(args.dataset, args.split, args.linker)
        args.reference = args.reference or default_reference_path(args.dataset, args.split)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
