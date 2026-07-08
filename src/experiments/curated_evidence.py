from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.cli_utils import collect_cuis, default_input_path, default_reference_path, load_items
from src.experiments.evidence_cards import (
    Q_TO_A_EXACT_PATH_EXPANSION_BUDGET,
    Q_TO_A_EXACT_PATHS_PER_ANSWER_CUI,
    _plain_relation_label,
)
from src.experiments.schemas import QAItem
from src.experiments.umls_index import CuiMetadataIndex, RelationIndex


ALLOWED_DECISIONS = {
    "keep_clinical",
    "keep_generic_valid",
    "drop_administrative",
    "drop_spurious",
    "drop_ambiguous",
}
KEEP_DECISIONS = {"keep_clinical", "keep_generic_valid"}
ADMINISTRATIVE_RELATION_TERMS = {
    "allowed qualifier",
    "answer to",
    "can be qualified",
    "concept in subset",
    "entry term",
    "has answer",
    "mapping qualifier",
    "mapped from",
    "mapped to",
    "qualifier applies",
    "subset includes",
    "used for",
}
CLINICAL_RELATION_TERMS = {
    "active ingredient",
    "associated finding",
    "associated morphology",
    "causative agent",
    "causes",
    "clinically associated",
    "complication",
    "contraindicated",
    "diagnoses",
    "disease has",
    "finding site",
    "has manifestation",
    "induces",
    "interprets",
    "may be treated",
    "may treat",
    "measures",
    "pathological process",
    "primary anatomic site",
    "sign or symptom",
    "therapeutic",
    "ingredient",
}
GENERIC_VALID_RELATION_TERMS = {
    "associated with",
    "broader relation",
    "has child",
    "has broader relation",
    "has narrower relation",
    "has parent",
    "has subtype",
    "inverse is a",
    "is a",
    "narrower relation",
    "possibly equivalent",
    "related to",
    "same as",
}
BAD_CONCEPT_NAME_TERMS = {
    "^",
    ":",
    "{",
    "}",
    " - action",
    "action",
    "age unit",
    "administration method",
    "anatomy qualifier",
    "assessment answer",
    "beginning",
    "behavior descriptors",
    "case - situation",
    "clinical nec",
    "conceptual entity",
    "concept attribute",
    "conversions (qualifier value)",
    "degree or extent",
    "department - no suggested values",
    "descriptors",
    "diseases component",
    "due to",
    "error",
    "facility (object)",
    "finding aspects",
    "findings aspects",
    "general adjectival modifier",
    "general qualifier",
    "hl7 vocabulary",
    "instruction imperative",
    "linkage concept",
    "loinc",
    "living arrangements",
    "mdfattributetype",
    "note:",
    "numdiff:",
    "organization unit",
    "pctdiff:",
    "performed by",
    "presence (property)",
    "present ++++",
    "property (qualifier value)",
    "property or attribute",
    "qualifier value",
    "relationship -",
    "relationships",
    "report:find",
    "severity modifier",
    "sensation quality",
    "special information qualifier",
    "specimen source codes",
    "specimen type -",
    "specimenentitytype",
    "terminology entity",
    "thesaurus of psychological index terms",
    "treated with",
    "allergy severity",
    "true case status",
    "unapproved attribute",
    "unnamed umls concept",
    "vocabulary version",
}
HARD_FALLBACK_BAD_CONCEPT_NAME_TERMS = {
    "assessment answer",
    "clinical nec",
    "concept attribute",
    "conceptual entity",
    "department - no suggested values",
    "error",
    "facility (object)",
    "finding aspects",
    "findings aspects",
    "general adjectival modifier",
    "general qualifier",
    "hl7 vocabulary",
    "linkage concept",
    "living arrangements",
    "mdfattributetype",
    "note:",
    "organization unit",
    "performed by",
    "property (qualifier value)",
    "property or attribute",
    "relationship -",
    "relationships",
    "report:find",
    "special information qualifier",
    "specimen source codes",
    "specimenentitytype",
    "terminology entity",
    "thesaurus of psychological index terms",
    "true case status",
    "unapproved attribute",
    "unnamed umls concept",
    "vocabulary version",
}
FORBIDDEN_ANNOTATION_FIELDS = {
    "gold",
    "answer",
    "answer_idx",
    "answer_text",
    "model_prediction",
    "prediction",
    "is_correct",
}
REQUIRED_ANNOTATION_FIELDS = {
    "item_id",
    "depth",
    "option_label",
    "option_text",
    "target_cui",
    "path_id",
    "path_cuis",
    "path_edges",
    "decision",
    "reviewer",
    "notes",
}
CURATED_SELECTION_POLICY = (
    "silver_curated_umls_only_option_balanced_"
    f"max_{Q_TO_A_EXACT_PATHS_PER_ANSWER_CUI}_paths_per_option"
)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _concept_name(cui: str, metadata: CuiMetadataIndex) -> str:
    return metadata.get(cui).name.strip() or "unnamed UMLS concept"


def _path_edges(path: Sequence[str], relations: RelationIndex, metadata: CuiMetadataIndex) -> list[dict]:
    edges: list[dict] = []
    for left, right in zip(path, path[1:]):
        edges.append({
            "left_cui": left,
            "left_name": _concept_name(left, metadata),
            "relation": _plain_relation_label(relations.relation_label(left, right)),
            "right_cui": right,
            "right_name": _concept_name(right, metadata),
        })
    return edges


def _stable_path_id(
    item_id: str,
    depth: int,
    option_label: str,
    target_cui: str,
    path_cuis: Sequence[str],
) -> str:
    payload = {
        "item_id": item_id,
        "depth": depth,
        "option_label": option_label,
        "target_cui": target_cui,
        "path_cuis": list(path_cuis),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"curated-{digest[:16]}"


def _all_answer_cuis(item: QAItem) -> list[str]:
    out: list[str] = []
    for cuis in item.option_cuis.values():
        out.extend(cuis)
    return list(dict.fromkeys(out))


def candidate_rows_for_items(
    items: Sequence[QAItem],
    metadata: CuiMetadataIndex,
    relations: RelationIndex,
    *,
    depths: Sequence[int] = (1, 2, 3, 4),
    max_paths_per_target: int = Q_TO_A_EXACT_PATHS_PER_ANSWER_CUI,
    max_expansions_per_target: int = Q_TO_A_EXACT_PATH_EXPANSION_BUDGET,
) -> list[dict]:
    rows: list[dict] = []
    for item in items:
        answer_cuis = set(_all_answer_cuis(item))
        for depth in depths:
            for option_label, option_text in item.options.items():
                option_cuis = list(dict.fromkeys(item.option_cuis.get(option_label, [])))
                if not option_text or not option_cuis:
                    continue
                for target_cui in option_cuis:
                    paths = relations.clean_exact_paths(
                        item.question_cuis,
                        target_cui,
                        depth,
                        forbidden_internal=answer_cuis,
                        max_paths=max_paths_per_target,
                        max_expansions=max_expansions_per_target,
                    )
                    for path_rank, path in enumerate(paths):
                        path_id = _stable_path_id(item.item_id, depth, option_label, target_cui, path)
                        rows.append({
                            "item_id": item.item_id,
                            "dataset": item.dataset,
                            "split": item.split,
                            "linker": item.linker,
                            "question": item.question,
                            "options": item.options,
                            "depth": depth,
                            "option_label": option_label,
                            "option_text": option_text,
                            "target_cui": target_cui,
                            "path_id": path_id,
                            "path_rank": path_rank,
                            "path_cuis": path,
                            "path_edges": _path_edges(path, relations, metadata),
                            "decision": "",
                            "reviewer": "",
                            "notes": "",
                        })
    return sorted(rows, key=lambda row: (row["item_id"], row["depth"], row["option_label"], row["target_cui"], row["path_id"]))


def _candidate_identity(row: dict) -> tuple:
    return (
        row.get("item_id"),
        int(row.get("depth") or 0),
        row.get("option_label"),
        row.get("target_cui"),
        tuple(row.get("path_cuis") or []),
        json.dumps(row.get("path_edges") or [], sort_keys=True),
    )


def validate_annotation_rows(annotation_rows: Sequence[dict], candidate_rows: Sequence[dict] | None = None) -> None:
    candidates = {str(row.get("path_id", "")): row for row in candidate_rows or []}
    for index, row in enumerate(annotation_rows, 1):
        forbidden = sorted(FORBIDDEN_ANNOTATION_FIELDS & set(row))
        if forbidden:
            raise ValueError(f"Annotation row {index} contains forbidden fields: {', '.join(forbidden)}")
        missing = sorted(field for field in REQUIRED_ANNOTATION_FIELDS if field not in row)
        if missing:
            raise ValueError(f"Annotation row {index} is missing fields: {', '.join(missing)}")
        decision = str(row.get("decision") or "")
        if decision not in ALLOWED_DECISIONS:
            raise ValueError(f"Annotation row {index} has invalid decision: {decision!r}")
        if not str(row.get("reviewer") or "").strip():
            raise ValueError(f"Annotation row {index} has empty reviewer")
        path_id = str(row.get("path_id") or "")
        if candidates:
            candidate = candidates.get(path_id)
            if candidate is None:
                raise ValueError(f"Annotation row {index} references unknown path_id: {path_id}")
            if _candidate_identity(row) != _candidate_identity(candidate):
                raise ValueError(f"Annotation row {index} path does not match candidate: {path_id}")


def _kept_annotation_rows(annotation_rows: Sequence[dict]) -> list[dict]:
    validate_annotation_rows(annotation_rows)
    return [row for row in annotation_rows if row.get("decision") in KEEP_DECISIONS]


def _normalized_relation_labels(row: dict) -> list[str]:
    labels = []
    for edge in row.get("path_edges") or []:
        label = " ".join(str(edge.get("relation") or "").casefold().replace("_", " ").split())
        if label:
            labels.append(label)
    return labels


def _relation_contains(labels: Sequence[str], terms: set[str]) -> bool:
    return any(term in label for label in labels for term in terms)


def _has_named_simple_path(row: dict) -> bool:
    path_cuis = [str(cui) for cui in row.get("path_cuis") or [] if str(cui)]
    edges = row.get("path_edges") or []
    if len(path_cuis) < 2 or len(path_cuis) != len(set(path_cuis)) or not edges:
        return False
    for edge in edges:
        if not str(edge.get("left_name") or "").strip():
            return False
        if not str(edge.get("right_name") or "").strip():
            return False
    return True


def _has_bad_concept_name(row: dict) -> bool:
    for edge in row.get("path_edges") or []:
        for field in ("left_name", "right_name"):
            name = " ".join(str(edge.get(field) or "").casefold().split())
            if any(term in name for term in BAD_CONCEPT_NAME_TERMS):
                return True
    return False


def _has_hard_fallback_bad_concept_name(row: dict) -> bool:
    for edge in row.get("path_edges") or []:
        for field in ("left_name", "right_name"):
            name = " ".join(str(edge.get(field) or "").casefold().split())
            if any(term in name for term in HARD_FALLBACK_BAD_CONCEPT_NAME_TERMS):
                return True
    return False


def _bad_concept_name_count(row: dict) -> int:
    count = 0
    seen: set[tuple[str, str]] = set()
    for edge in row.get("path_edges") or []:
        for field in ("left_name", "right_name"):
            name = " ".join(str(edge.get(field) or "").casefold().split())
            for term in BAD_CONCEPT_NAME_TERMS:
                key = (name, term)
                if key in seen:
                    continue
                if term in name:
                    seen.add(key)
                    count += 1
    return count


def _candidate_quality_score(row: dict) -> int | None:
    labels = _normalized_relation_labels(row)
    if not _has_named_simple_path(row):
        return None
    if _relation_contains(labels, ADMINISTRATIVE_RELATION_TERMS):
        return None
    if _has_hard_fallback_bad_concept_name(row):
        return None
    if not _relation_contains(labels, CLINICAL_RELATION_TERMS | GENERIC_VALID_RELATION_TERMS):
        return None
    score = 0
    for label in labels:
        if any(term in label for term in CLINICAL_RELATION_TERMS):
            score += 4
        elif any(term in label for term in GENERIC_VALID_RELATION_TERMS):
            score += 2
        else:
            score -= 1
    score -= _bad_concept_name_count(row)
    return score if score > 0 else None


def auto_annotate_rows(
    candidate_rows: Sequence[dict],
    *,
    reviewer: str = "silver_rule_v1",
    coverage_fallback: bool = False,
) -> list[dict]:
    """Create deterministic silver labels without using answer keys or model predictions."""
    rows: list[dict] = []
    for candidate in candidate_rows:
        row = dict(candidate)
        labels = _normalized_relation_labels(row)
        if not _has_named_simple_path(row):
            decision = "drop_spurious"
            notes = "auto: malformed or unnamed path"
        elif _has_bad_concept_name(row):
            decision = "drop_spurious"
            notes = "auto: low-information ontology scaffold concept present"
        elif _relation_contains(labels, ADMINISTRATIVE_RELATION_TERMS):
            decision = "drop_administrative"
            notes = "auto: administrative relation present"
        elif _relation_contains(labels, CLINICAL_RELATION_TERMS):
            decision = "keep_clinical"
            notes = "auto: clinical relation present; no administrative relation"
        elif labels and all(
            any(term in label for term in GENERIC_VALID_RELATION_TERMS)
            for label in labels
        ):
            decision = "keep_generic_valid"
            notes = "auto: generic ontology relation path; no administrative relation"
        else:
            decision = "drop_ambiguous"
            notes = "auto: no clinical or accepted generic relation pattern"
        row["decision"] = decision
        row["reviewer"] = reviewer
        row["notes"] = notes
        score = _candidate_quality_score(row)
        if score is not None:
            row["curation_score"] = score
        rows.append(row)
    if coverage_fallback:
        by_group: dict[tuple[str, int, str], list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            by_group[
                (str(row.get("item_id") or ""), int(row.get("depth") or 0), str(row.get("option_label") or ""))
            ].append(index)
        for indexes in by_group.values():
            if any(rows[index].get("decision") in KEEP_DECISIONS for index in indexes):
                continue
            scored: list[tuple[int, int, str, int]] = []
            for index in indexes:
                score = _candidate_quality_score(rows[index])
                if score is None:
                    continue
                scored.append((
                    score,
                    -int(rows[index].get("path_rank") or 0),
                    str(rows[index].get("path_id") or ""),
                    index,
                ))
            if not scored:
                continue
            score, _negative_rank, _path_id, index = max(scored)
            labels = _normalized_relation_labels(rows[index])
            rows[index]["decision"] = (
                "keep_clinical"
                if _relation_contains(labels, CLINICAL_RELATION_TERMS)
                else "keep_generic_valid"
            )
            rows[index]["reviewer"] = reviewer
            rows[index]["notes"] = f"auto: coverage fallback best non-administrative path; score={score}"
            rows[index]["curation_score"] = score
    return rows


def _option_labels_requiring_evidence(item: QAItem) -> list[str]:
    return [
        label
        for label, option_text in item.options.items()
        if option_text and item.option_cuis.get(label)
    ]


def _render_curated_rows(rows: Sequence[dict], max_paths_per_option: int) -> tuple[str, int, int, int]:
    lines: list[str] = []
    nodes: set[str] = set()
    edge_keys: set[tuple[str, str, str]] = set()
    target_cuis: set[str] = set()
    by_option: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_option[str(row["option_label"])].append(row)

    for option_label in sorted(by_option):
        option_rows = sorted(
            by_option[option_label],
            key=lambda row: (-int(row.get("curation_score") or 0), int(row.get("path_rank", 1_000_000)), str(row["path_id"])),
        )[:max_paths_per_option]
        for row in option_rows:
            target_cuis.add(str(row["target_cui"]))
            nodes.update(str(cui) for cui in row.get("path_cuis") or [])
            for edge in row.get("path_edges") or []:
                key = (
                    str(edge.get("left_cui") or ""),
                    str(edge.get("right_cui") or ""),
                    str(edge.get("relation") or ""),
                )
                if key in edge_keys:
                    continue
                edge_keys.add(key)
                lines.append(
                    "- {left_name} -- {relation} --> {right_name}".format(
                        left_name=edge.get("left_name") or edge.get("left_cui") or "unnamed UMLS concept",
                        relation=edge.get("relation") or "related to",
                        right_name=edge.get("right_name") or edge.get("right_cui") or "unnamed UMLS concept",
                    )
                )
    return "\n".join(lines), len(nodes), len(edge_keys), len(target_cuis)


def build_curated_evidence_cache_rows(
    items: Sequence[QAItem],
    annotation_rows: Sequence[dict],
    *,
    depths: Sequence[int] = (1, 2, 3, 4),
    max_paths_per_option: int = Q_TO_A_EXACT_PATHS_PER_ANSWER_CUI,
) -> list[dict]:
    kept_rows = _kept_annotation_rows(annotation_rows)
    kept_by_item_depth_option: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in kept_rows:
        kept_by_item_depth_option[
            (str(row["item_id"]), int(row["depth"]), str(row["option_label"]))
        ].append(row)

    out: list[dict] = []
    for item in items:
        required_option_labels = _option_labels_requiring_evidence(item)
        if not required_option_labels:
            continue
        for depth in depths:
            selected: list[dict] = []
            missing = False
            for option_label in required_option_labels:
                option_rows = kept_by_item_depth_option.get((item.item_id, depth, option_label), [])
                if not option_rows:
                    missing = True
                    break
                selected.extend(option_rows)
            if missing:
                continue
            text, node_count, edge_count, target_count = _render_curated_rows(selected, max_paths_per_option)
            if not text:
                continue
            condition = f"q_to_a_curated_depth_{depth}_evidence"
            out.append({
                "item_id": item.item_id,
                "dataset": item.dataset,
                "split": item.split,
                "linker": item.linker,
                "condition": condition,
                "text": text,
                "oracle": False,
                "oracle_gold_cui_source": "",
                "oracle_lookup_success": False,
                "oracle_num_candidate_cuis": 0,
                "oracle_ambiguous": False,
                "oracle_selected_cui_source": "",
                "option_evidence_symmetric": True,
                "uses_gold_label_for_evidence": False,
                "q_to_a_depth_semantics": "silver_curated_exact_hops",
                "q_to_a_depth": depth,
                "q_to_a_path_found": True,
                "q_to_a_path_node_count": node_count,
                "q_to_a_path_edge_count": edge_count,
                "q_to_a_answer_cui_count": len(required_option_labels),
                "q_to_a_answer_cui_covered_count": len(required_option_labels),
                "q_to_a_path_selection_policy": CURATED_SELECTION_POLICY,
                "curated_target_cui_count": target_count,
                "curated_kept_path_count": len(selected),
            })
    return sorted(out, key=lambda row: (row["item_id"], row["condition"]))


def _parse_depths(values: Sequence[str] | None) -> tuple[int, ...]:
    if not values:
        return (1, 2, 3, 4)
    return tuple(int(value) for value in values)


def _load_items_from_args(args: argparse.Namespace) -> list[QAItem]:
    input_path = args.input or default_input_path(args.dataset, args.split, args.linker)
    reference = args.reference or default_reference_path(args.dataset, args.split)
    limit = None if args.limit is None or args.limit <= 0 else args.limit
    return load_items(
        args.dataset,
        args.split,
        args.linker,
        input_path=input_path,
        reference_path=reference if reference.exists() else None,
        limit=limit,
        item_id_file=args.item_id_file,
    )


def _cmd_export_candidates(args: argparse.Namespace) -> int:
    depths = _parse_depths(args.depth)
    items = _load_items_from_args(args)
    metadata = CuiMetadataIndex.from_zip(args.metadata_zip)
    keep_cuis = None if args.relation_scope == "full" else collect_cuis(items)
    relations = RelationIndex.from_zip(args.triplets_zip, keep_cuis=keep_cuis)
    rows = candidate_rows_for_items(
        items,
        metadata,
        relations,
        depths=depths,
        max_paths_per_target=args.max_paths_per_target,
        max_expansions_per_target=args.max_expansions_per_target,
    )
    write_jsonl(args.output_jsonl, rows)
    print(json.dumps({"event": "curated_candidates_exported", "rows": len(rows), "output_jsonl": str(args.output_jsonl)}))
    return 0


def _cmd_auto_annotate(args: argparse.Namespace) -> int:
    candidates = read_jsonl(args.candidate_jsonl)
    rows = auto_annotate_rows(candidates, reviewer=args.reviewer, coverage_fallback=args.coverage_fallback)
    validate_annotation_rows(rows, candidate_rows=candidates)
    write_jsonl(args.output_jsonl, rows)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["decision"])] += 1
    print(json.dumps({
        "event": "curated_annotations_auto_created",
        "rows": len(rows),
        "decision_counts": dict(sorted(counts.items())),
        "reviewer": args.reviewer,
        "output_jsonl": str(args.output_jsonl),
    }))
    return 0


def _cmd_validate_annotations(args: argparse.Namespace) -> int:
    annotations = read_jsonl(args.annotations_jsonl)
    candidates = read_jsonl(args.candidate_jsonl) if args.candidate_jsonl else None
    validate_annotation_rows(annotations, candidate_rows=candidates)
    print(json.dumps({"event": "curated_annotations_valid", "rows": len(annotations)}))
    return 0


def _cmd_build_cache(args: argparse.Namespace) -> int:
    depths = _parse_depths(args.depth)
    items = _load_items_from_args(args)
    annotations = read_jsonl(args.annotations_jsonl)
    candidates = read_jsonl(args.candidate_jsonl) if args.candidate_jsonl else None
    validate_annotation_rows(annotations, candidate_rows=candidates)
    rows = build_curated_evidence_cache_rows(items, annotations, depths=depths)
    write_jsonl(args.output_jsonl, rows)
    print(json.dumps({
        "event": "curated_evidence_cache_built",
        "items": len(items),
        "annotation_rows": len(annotations),
        "cache_rows": len(rows),
        "output_jsonl": str(args.output_jsonl),
    }))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export, validate, and render human-curated UMLS path evidence.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_data_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--dataset", default="medQA")
        subparser.add_argument("--split", default="test")
        subparser.add_argument("--linker", choices=["metamap", "spacy"], default="spacy")
        subparser.add_argument("--input", type=Path, default=None)
        subparser.add_argument("--reference", type=Path, default=None)
        subparser.add_argument("--item-id-file", type=Path, default=None)
        subparser.add_argument("--limit", type=int, default=0)
        subparser.add_argument("--depth", action="append", default=None)

    export_candidates = subparsers.add_parser("export-candidates")
    add_data_args(export_candidates)
    export_candidates.add_argument("--metadata-zip", type=Path, default=Path("data/UMLS/cui_detail.tsv.zip"))
    export_candidates.add_argument("--triplets-zip", type=Path, default=Path("data/UMLS/triplets.csv.zip"))
    export_candidates.add_argument("--relation-scope", choices=["item_cuis", "full"], default="full")
    export_candidates.add_argument("--max-paths-per-target", type=int, default=Q_TO_A_EXACT_PATHS_PER_ANSWER_CUI)
    export_candidates.add_argument("--max-expansions-per-target", type=int, default=Q_TO_A_EXACT_PATH_EXPANSION_BUDGET)
    export_candidates.add_argument("--output-jsonl", type=Path, required=True)
    export_candidates.set_defaults(func=_cmd_export_candidates)

    auto_annotate = subparsers.add_parser("auto-annotate")
    auto_annotate.add_argument("--candidate-jsonl", type=Path, required=True)
    auto_annotate.add_argument("--reviewer", default="silver_rule_v1")
    auto_annotate.add_argument("--coverage-fallback", action="store_true")
    auto_annotate.add_argument("--output-jsonl", type=Path, required=True)
    auto_annotate.set_defaults(func=_cmd_auto_annotate)

    validate = subparsers.add_parser("validate-annotations")
    validate.add_argument("--annotations-jsonl", type=Path, required=True)
    validate.add_argument("--candidate-jsonl", type=Path, default=None)
    validate.set_defaults(func=_cmd_validate_annotations)

    build_cache = subparsers.add_parser("build-cache")
    add_data_args(build_cache)
    build_cache.add_argument("--annotations-jsonl", type=Path, required=True)
    build_cache.add_argument("--candidate-jsonl", type=Path, default=None)
    build_cache.add_argument("--output-jsonl", type=Path, required=True)
    build_cache.set_defaults(func=_cmd_build_cache)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
