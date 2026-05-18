from __future__ import annotations

from dataclasses import dataclass

from .schemas import QAItem
from .umls_index import CuiMetadataIndex, RelationIndex


@dataclass(frozen=True)
class EvidenceConditionSpec:
    evidence_view: str
    retrieval_method: str | None
    oracle: bool = False


@dataclass(frozen=True)
class EvidenceCardDetails:
    text: str | None
    oracle: bool = False
    oracle_gold_cui_source: str = ""
    oracle_lookup_success: bool = False
    oracle_num_candidate_cuis: int = 0
    oracle_ambiguous: bool = False
    oracle_selected_cui_source: str = ""
    option_evidence_symmetric: bool = False
    uses_gold_label_for_evidence: bool = False


CONDITION_SPECS = {
    "relation_only_evidence": EvidenceConditionSpec("MRCONSO_MRREL", "pooled_1hop"),
    "metadata_rich_evidence": EvidenceConditionSpec("full_committed_card", None),
    "q_to_a_path_evidence": EvidenceConditionSpec("full_committed_card", "q_to_a_2hop"),
    "random_same_size_evidence": EvidenceConditionSpec("MRCONSO_MRREL", "random_same_size"),
    "shuffled_answer_evidence": EvidenceConditionSpec("full_committed_card", None),
    "gold_only_evidence": EvidenceConditionSpec("full_committed_card", None),
    "distractor_contrast_evidence": EvidenceConditionSpec("full_committed_card", None),
    "oracle_gold_evidence": EvidenceConditionSpec("full_committed_card", None, oracle=True),
}


CONDITION_TYPES = {
    "no_umls": "baseline",
    "relation_only_evidence": "realistic",
    "metadata_rich_evidence": "realistic",
    "q_to_a_path_evidence": "realistic",
    "distractor_contrast_evidence": "realistic_or_contrastive",
    "random_same_size_evidence": "negative_control",
    "shuffled_answer_evidence": "negative_control",
    "gold_only_evidence": "diagnostic",
    "oracle_gold_evidence": "oracle",
}


def condition_metadata(condition: str) -> EvidenceConditionSpec:
    return CONDITION_SPECS.get(condition, EvidenceConditionSpec("none", None))


def condition_type_for_condition(condition: str) -> str:
    return CONDITION_TYPES.get(condition, "unknown")


def _format_concept(cui: str, metadata: CuiMetadataIndex, view: str, label: str = "") -> list[str]:
    detail = metadata.get(cui)
    name = detail.name or cui
    prefix = f"{label} " if label else ""
    lines = [f"- {prefix}{cui}: {name}"]
    if view in {"MRCONSO_MRREL_MRDEF", "full_committed_card"} and detail.definition:
        lines.append(f"  Definition: {detail.definition}")
    if view in {"MRCONSO_MRREL_MRSTY", "full_committed_card"} and detail.semantic_types:
        lines.append(f"  Semantic types: {', '.join(detail.semantic_types)}")
    return lines


def _all_answer_cuis(item: QAItem) -> list[str]:
    out: list[str] = []
    for cuis in item.option_cuis.values():
        out.extend(cuis)
    return list(dict.fromkeys(out))


def _metadata_lines(item: QAItem, metadata: CuiMetadataIndex, view: str, shuffled: bool = False) -> list[str]:
    lines: list[str] = []
    if item.question_cuis:
        lines.append("Question concepts:")
        for cui in list(dict.fromkeys(item.question_cuis))[:8]:
            lines.extend(_format_concept(cui, metadata, view))

    option_items = list(item.option_cuis.items())
    if shuffled and option_items:
        cuis_by_option = [cuis for _, cuis in option_items]
        shifted = cuis_by_option[1:] + cuis_by_option[:1]
        option_items = [
            (label, shifted[index])
            for index, (label, _) in enumerate(option_items)
        ]

    if option_items:
        lines.append("Answer-option concepts:")
        for label, cuis in option_items:
            if not cuis:
                lines.append(f"- {label}: no linked answer CUI")
                continue
            for cui in list(dict.fromkeys(cuis))[:4]:
                lines.extend(_format_concept(cui, metadata, view, label=label))
    return lines


def _option_concept_lines(
    label: str,
    cuis: list[str],
    metadata: CuiMetadataIndex,
    view: str,
) -> list[str]:
    if not cuis:
        return [f"- {label}: no linked answer CUI"]
    lines: list[str] = []
    for cui in list(dict.fromkeys(cuis))[:4]:
        lines.extend(_format_concept(cui, metadata, view, label=label))
    return lines


def _gold_cuis_for_oracle(item: QAItem, metadata: CuiMetadataIndex) -> tuple[list[str], str, int]:
    linked = item.gold_cuis()
    if linked:
        unique_linked = list(dict.fromkeys(linked))
        return unique_linked, "linked_gold_cui", len(unique_linked)
    candidates = []
    if item.answer_text:
        candidates.append(item.answer_text)
    if item.gold is not None and item.gold in item.options:
        candidates.append(item.options[item.gold])
    for candidate in candidates:
        cuis = metadata.find_exact_by_name(candidate)
        if cuis:
            return list(dict.fromkeys(cuis)), "exact_answer_text_lookup", len(set(cuis))
    for candidate in candidates:
        cuis = metadata.find_normalized_by_name(candidate)
        if cuis:
            return list(dict.fromkeys(cuis)), "normalized_answer_text_lookup", len(set(cuis))
    return [], "unavailable", 0


def _gold_only_lines(
    item: QAItem,
    metadata: CuiMetadataIndex,
    view: str,
    gold_cuis: list[str] | None = None,
) -> list[str]:
    label = item.gold or "gold"
    resolved_cuis = list(dict.fromkeys(gold_cuis if gold_cuis is not None else item.gold_cuis()))
    lines = ["Gold answer evidence:"]
    lines.extend(_option_concept_lines(label, resolved_cuis, metadata, view))
    return lines


def _contrast_lines(item: QAItem, metadata: CuiMetadataIndex, view: str) -> list[str]:
    lines = ["Distractor contrast evidence:"]
    for label, cuis in item.option_cuis.items():
        lines.append(f"Option {label} evidence:")
        lines.extend(_option_concept_lines(label, cuis, metadata, view))
    return lines


def _validate_symmetric_option_evidence(item: QAItem, lines: list[str]) -> None:
    text = "\n".join(lines)
    forbidden = ("Gold option", "Distractor option", "Gold answer evidence:")
    if any(marker in text for marker in forbidden):
        raise ValueError("distractor_contrast_evidence must not expose gold/distractor labels")
    missing = [
        label
        for label in item.options
        if f"Option {label} evidence:" not in text
    ]
    if missing:
        raise ValueError(
            "distractor_contrast_evidence must include symmetric evidence sections for every option: "
            + ", ".join(missing)
        )


def _relation_lines(nodes: set[str], relations: RelationIndex, limit: int = 24) -> list[str]:
    edges = relations.edges(nodes, limit=limit)
    if not edges:
        return []
    lines = ["Relations:"]
    for left, right, label in edges:
        lines.append(f"- {left} --{label}-- {right}")
    return lines


def _path_lines(item: QAItem, relations: RelationIndex) -> list[str]:
    answer_cuis = _all_answer_cuis(item)
    edges = relations.path_edges_between(item.question_cuis, answer_cuis, max_hops=2)
    if not edges:
        return ["Q-to-answer paths: no two-hop or shorter UMLS path found."]
    lines = ["Q-to-answer paths:"]
    for left, right, label in edges:
        lines.append(f"- {left} --{label}-- {right}")
    return lines


def _random_nodes(item: QAItem, relations: RelationIndex) -> set[str]:
    answer_cuis = _all_answer_cuis(item)
    path_nodes = relations.path_nodes_between(item.question_cuis, answer_cuis, max_hops=2)
    return relations.deterministic_sample(max(1, len(path_nodes)), key=item.item_id)


def build_evidence_card_details(
    item: QAItem,
    condition: str,
    metadata: CuiMetadataIndex,
    relations: RelationIndex,
) -> EvidenceCardDetails:
    if condition == "no_umls":
        return EvidenceCardDetails(None)

    spec = condition_metadata(condition)
    if spec.evidence_view == "none":
        raise ValueError(f"Unsupported downstream probe condition: {condition}")

    lines = [f"Evidence condition: {condition}"]
    oracle_source = ""

    if condition == "oracle_gold_evidence":
        gold_cuis, oracle_source, candidate_count = _gold_cuis_for_oracle(item, metadata)
        lines.append("Oracle condition: uses gold answer evidence as an upper bound, not a realistic pipeline.")
        lines.extend(_gold_only_lines(item, metadata, spec.evidence_view, gold_cuis=gold_cuis))
        return EvidenceCardDetails(
            "\n".join(lines),
            oracle=True,
            oracle_gold_cui_source=oracle_source,
            oracle_lookup_success=bool(gold_cuis),
            oracle_num_candidate_cuis=candidate_count,
            oracle_ambiguous=candidate_count > 1,
            oracle_selected_cui_source=oracle_source,
        )

    if condition == "gold_only_evidence":
        lines.extend(_gold_only_lines(item, metadata, spec.evidence_view))
        return EvidenceCardDetails("\n".join(lines))

    if condition == "distractor_contrast_evidence":
        contrast_lines = _contrast_lines(item, metadata, spec.evidence_view)
        _validate_symmetric_option_evidence(item, contrast_lines)
        lines.extend(contrast_lines)
        return EvidenceCardDetails(
            "\n".join(lines),
            option_evidence_symmetric=True,
            uses_gold_label_for_evidence=False,
        )

    if condition == "shuffled_answer_evidence":
        lines.extend(_metadata_lines(item, metadata, spec.evidence_view, shuffled=True))
        return EvidenceCardDetails("\n".join(lines))

    if condition == "q_to_a_path_evidence":
        path_nodes = relations.path_nodes_between(item.question_cuis, _all_answer_cuis(item), max_hops=2)
        lines.extend(_metadata_lines(item, metadata, spec.evidence_view))
        lines.extend(_path_lines(item, relations))
        if path_nodes:
            lines.extend(_relation_lines(path_nodes, relations))
        return EvidenceCardDetails("\n".join(lines))

    if condition == "random_same_size_evidence":
        nodes = _random_nodes(item, relations)
        lines.extend(_relation_lines(nodes, relations))
        if len(lines) == 1:
            lines.append("Relations: no random same-size relation edges available.")
        return EvidenceCardDetails("\n".join(lines))

    lines.extend(_metadata_lines(item, metadata, spec.evidence_view))
    if condition == "relation_only_evidence":
        seeds = set(item.question_cuis + _all_answer_cuis(item))
        lines.extend(_relation_lines(relations.pooled_neighborhood(seeds, hops=1, cap=80), relations))
    return EvidenceCardDetails("\n".join(lines))


def build_evidence_card(
    item: QAItem,
    condition: str,
    metadata: CuiMetadataIndex,
    relations: RelationIndex,
) -> str | None:
    return build_evidence_card_details(item, condition, metadata, relations).text
