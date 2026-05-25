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
    q_to_a_depth_semantics: str = ""
    q_to_a_depth: int = 0
    q_to_a_path_found: bool = False
    q_to_a_path_node_count: int = 0
    q_to_a_path_edge_count: int = 0
    q_to_a_answer_cui_count: int = 0
    q_to_a_answer_cui_covered_count: int = 0
    q_to_a_path_selection_policy: str = ""


Q_TO_A_DEPTHS = (1, 2, 3, 4, 5)
Q_TO_A_EXACT_PATHS_PER_ANSWER_CUI = 2
Q_TO_A_EXACT_PATH_SELECTION_POLICY = "clean_exact_hops_all_answer_cuis_max_2_paths_per_cui"
Q_TO_A_EXACT_DEPTH_CONDITION_SPECS = {
    f"q_to_a_exact_depth_{depth}_evidence": EvidenceConditionSpec(
        "full_committed_card",
        f"q_to_a_exact_depth_{depth}",
    )
    for depth in Q_TO_A_DEPTHS
}

RELATION_LABELS = {
    "AQ": "has allowed qualifier",
    "CHD": "has child",
    "PAR": "has parent",
    "QB": "can be qualified by",
    "RB": "has broader relation to",
    "RN": "has narrower relation to",
    "RO": "related to",
    "RQ": "related and possibly synonymous with",
    "SY": "synonymous with",
    "isa": "is a",
    "inverse_isa": "has subtype",
}


CONDITION_SPECS = {
    "relation_only_evidence": EvidenceConditionSpec("MRCONSO_MRREL", "pooled_1hop"),
    "metadata_rich_evidence": EvidenceConditionSpec("full_committed_card", None),
    "q_to_a_path_evidence": EvidenceConditionSpec("full_committed_card", "q_to_a_2hop"),
    "random_same_size_evidence": EvidenceConditionSpec("MRCONSO_MRREL", "random_same_size"),
    "shuffled_answer_evidence": EvidenceConditionSpec("full_committed_card", None),
    "gold_only_evidence": EvidenceConditionSpec("full_committed_card", None),
    "distractor_contrast_evidence": EvidenceConditionSpec("full_committed_card", None),
    "oracle_gold_evidence": EvidenceConditionSpec("full_committed_card", None, oracle=True),
    **Q_TO_A_EXACT_DEPTH_CONDITION_SPECS,
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
    **{condition: "realistic" for condition in Q_TO_A_EXACT_DEPTH_CONDITION_SPECS},
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


def _concept_name(cui: str, metadata: CuiMetadataIndex) -> str | None:
    name = metadata.get(cui).name.strip()
    return name or "unnamed UMLS concept"


def _plain_relation_label(label: str) -> str:
    cleaned = label.strip()
    if not cleaned:
        return "related to"
    if cleaned in RELATION_LABELS:
        return RELATION_LABELS[cleaned]
    return cleaned.replace("_", " ").replace("-", " ")


def _format_plain_concept(cui: str, metadata: CuiMetadataIndex, view: str) -> list[str]:
    detail = metadata.get(cui)
    name = detail.name.strip()
    if not name:
        return []
    lines = [f"- {name} (CUI: {cui})"]
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


def _source_agnostic_metadata_lines(item: QAItem, metadata: CuiMetadataIndex, view: str) -> list[str]:
    lines = ["Relevant concepts:"]
    cuis: list[str] = []
    cuis.extend(list(dict.fromkeys(item.question_cuis))[:8])
    for option_cuis in item.option_cuis.values():
        cuis.extend(list(dict.fromkeys(option_cuis))[:4])
    seen: set[str] = set()
    for cui in cuis:
        if cui in seen:
            continue
        seen.add(cui)
        lines.extend(_format_plain_concept(cui, metadata, view))
    if len(lines) == 1:
        lines.append("- no linked UMLS concepts with readable names")
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


def _node_priority(item: QAItem) -> dict[str, int]:
    priority: dict[str, int] = {}
    index = 0
    for cui in item.question_cuis:
        if cui and cui not in priority:
            priority[cui] = index
            index += 1
    for option_cuis in item.option_cuis.values():
        for cui in option_cuis:
            if cui and cui not in priority:
                priority[cui] = index
                index += 1
    return priority


def _orient_edge(left: str, right: str, node_priority: dict[str, int] | None = None) -> tuple[str, str]:
    if not node_priority:
        return left, right
    left_priority = node_priority.get(left)
    right_priority = node_priority.get(right)
    if right_priority is not None and (left_priority is None or right_priority < left_priority):
        return right, left
    return left, right


def _relation_edge_lines(
    edges: list[tuple[str, str, str]],
    metadata: CuiMetadataIndex,
    header: str,
    limit: int | None = 24,
    node_priority: dict[str, int] | None = None,
) -> list[str]:
    if not edges:
        return []
    lines = [header]
    emitted = 0
    for left, right, label in edges:
        oriented_left, oriented_right = _orient_edge(left, right, node_priority)
        left_name = _concept_name(oriented_left, metadata)
        right_name = _concept_name(oriented_right, metadata)
        if not left_name or not right_name:
            continue
        lines.append(f"- {left_name} -- {_plain_relation_label(label)} --> {right_name}")
        emitted += 1
        if limit is not None and emitted >= limit:
            break
    if emitted == 0:
        return []
    return lines


def _relation_lines(
    nodes: set[str],
    relations: RelationIndex,
    metadata: CuiMetadataIndex,
    limit: int = 24,
    node_priority: dict[str, int] | None = None,
) -> list[str]:
    return _relation_edge_lines(
        relations.edges(nodes),
        metadata,
        "UMLS graph:",
        limit=limit,
        node_priority=node_priority,
    )


def _path_lines(item: QAItem, relations: RelationIndex, metadata: CuiMetadataIndex) -> list[str]:
    answer_cuis = _all_answer_cuis(item)
    edges = relations.path_edges_between(item.question_cuis, answer_cuis, max_hops=2)
    if not edges:
        return ["Q-to-answer paths: no two-hop or shorter UMLS path found."]
    return _relation_edge_lines(
        edges,
        metadata,
        "Q-to-answer paths:",
        limit=None,
    ) or ["Q-to-answer paths: no readable two-hop or shorter UMLS path found."]


def _q_to_a_graph_lines(
    item: QAItem,
    relations: RelationIndex,
    metadata: CuiMetadataIndex,
) -> tuple[list[str], int, int]:
    answer_cuis = _all_answer_cuis(item)
    path_nodes = relations.path_nodes_between(item.question_cuis, answer_cuis, max_hops=2)
    lines = _path_lines(item, relations, metadata)
    if path_nodes:
        lines.extend(
            _relation_lines(
                path_nodes,
                relations,
                metadata,
                node_priority=_node_priority(item),
            )
        )
    edge_count = len(relations.path_edges_between(item.question_cuis, answer_cuis, max_hops=2))
    return lines, len(path_nodes), edge_count


def _q_to_a_exact_depth_lines(
    item: QAItem,
    relations: RelationIndex,
    metadata: CuiMetadataIndex,
    depth: int,
) -> tuple[list[str], int, int, int, int, str]:
    answer_cuis = _all_answer_cuis(item)
    answer_count = len(answer_cuis)
    if not answer_cuis:
        return [
            "Q-to-answer paths: no linked answer-option CUI available."
        ], 0, 0, 0, 0, Q_TO_A_EXACT_PATH_SELECTION_POLICY

    paths_by_target = relations.clean_exact_paths_between(
        item.question_cuis,
        answer_cuis,
        hops=depth,
        forbidden_internal=set(answer_cuis),
        max_paths_per_target=Q_TO_A_EXACT_PATHS_PER_ANSWER_CUI,
    )
    covered_count = len(paths_by_target)
    if covered_count != answer_count:
        missing = [
            f"{metadata.get(cui).name or cui} (CUI: {cui})"
            for cui in answer_cuis
            if cui not in paths_by_target
        ]
        return [
            "Q-to-answer paths: no complete UMLS path set found.",
            "Missing answer CUIs: " + ", ".join(missing),
        ], 0, 0, answer_count, covered_count, Q_TO_A_EXACT_PATH_SELECTION_POLICY

    lines = ["Q-to-answer paths:"]
    nodes: set[str] = set()
    edge_keys: set[tuple[str, str]] = set()
    edge_count = 0
    for target in answer_cuis:
        target_name = metadata.get(target).name or target
        lines.append(f"Paths to {target_name} (CUI: {target}):")
        for index, path in enumerate(paths_by_target[target], start=1):
            lines.append(f"- Path {index}:")
            nodes.update(path)
            for left, right in zip(path, path[1:]):
                left_name = _concept_name(left, metadata)
                right_name = _concept_name(right, metadata)
                if not left_name or not right_name:
                    continue
                label = relations.relation_label(left, right)
                lines.append(f"  - {left_name} -- {_plain_relation_label(label)} --> {right_name}")
                key = tuple(sorted((left, right)))
                if key not in edge_keys:
                    edge_keys.add(key)
                    edge_count += 1
    return lines, len(nodes), edge_count, answer_count, covered_count, Q_TO_A_EXACT_PATH_SELECTION_POLICY


def _random_nodes(item: QAItem, relations: RelationIndex) -> set[str]:
    answer_cuis = _all_answer_cuis(item)
    path_nodes = relations.path_nodes_between(item.question_cuis, answer_cuis, max_hops=2)
    return relations.deterministic_sample(max(1, len(path_nodes)), key=item.item_id)


def _parse_q_to_a_exact_depth_condition(condition: str) -> int | None:
    prefix = "q_to_a_exact_depth_"
    suffix = "_evidence"
    if not condition.startswith(prefix) or not condition.endswith(suffix):
        return None
    depth_text = condition[len(prefix):-len(suffix)]
    try:
        depth = int(depth_text)
    except ValueError:
        return None
    if depth not in Q_TO_A_DEPTHS:
        return None
    return depth


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

    lines = []
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

    if condition == "metadata_rich_evidence":
        lines.extend(_source_agnostic_metadata_lines(item, metadata, spec.evidence_view))
        graph_lines, node_count, edge_count = _q_to_a_graph_lines(item, relations, metadata)
        lines.extend(graph_lines)
        return EvidenceCardDetails(
            "\n".join(lines),
            q_to_a_depth_semantics="max_hops",
            q_to_a_depth=2,
            q_to_a_path_found=edge_count > 0,
            q_to_a_path_node_count=node_count,
            q_to_a_path_edge_count=edge_count,
        )

    if condition == "q_to_a_path_evidence":
        graph_lines, node_count, edge_count = _q_to_a_graph_lines(item, relations, metadata)
        lines.extend(graph_lines)
        return EvidenceCardDetails(
            "\n".join(lines),
            q_to_a_depth_semantics="max_hops",
            q_to_a_depth=2,
            q_to_a_path_found=edge_count > 0,
            q_to_a_path_node_count=node_count,
            q_to_a_path_edge_count=edge_count,
        )

    exact_depth = _parse_q_to_a_exact_depth_condition(condition)
    if exact_depth is not None:
        (
            graph_lines,
            node_count,
            edge_count,
            answer_count,
            covered_count,
            selection_policy,
        ) = _q_to_a_exact_depth_lines(
            item,
            relations,
            metadata,
            exact_depth,
        )
        lines.extend(graph_lines)
        return EvidenceCardDetails(
            "\n".join(lines),
            q_to_a_depth_semantics="exact_hops",
            q_to_a_depth=exact_depth,
            q_to_a_path_found=answer_count > 0 and covered_count == answer_count,
            q_to_a_path_node_count=node_count,
            q_to_a_path_edge_count=edge_count,
            q_to_a_answer_cui_count=answer_count,
            q_to_a_answer_cui_covered_count=covered_count,
            q_to_a_path_selection_policy=selection_policy,
        )

    if condition == "random_same_size_evidence":
        nodes = _random_nodes(item, relations)
        lines.extend(_relation_lines(nodes, relations, metadata, node_priority=_node_priority(item)))
        if not lines:
            lines.append("UMLS graph: no readable random same-size relation edges available.")
        return EvidenceCardDetails("\n".join(lines))

    if condition == "relation_only_evidence":
        seeds = set(item.question_cuis + _all_answer_cuis(item))
        graph_lines = _relation_lines(
            relations.pooled_neighborhood(seeds, hops=1, cap=80),
            relations,
            metadata,
            node_priority=_node_priority(item),
        )
        lines.extend(graph_lines or ["UMLS graph: no readable relation edges available."])
        return EvidenceCardDetails("\n".join(lines))

    lines.extend(_metadata_lines(item, metadata, spec.evidence_view))
    return EvidenceCardDetails("\n".join(lines))


def build_evidence_card(
    item: QAItem,
    condition: str,
    metadata: CuiMetadataIndex,
    relations: RelationIndex,
) -> str | None:
    return build_evidence_card_details(item, condition, metadata, relations).text
