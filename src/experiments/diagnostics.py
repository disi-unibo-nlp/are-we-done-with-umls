from __future__ import annotations

from .schemas import QAItem
from .umls_index import CuiMetadataIndex, RelationIndex


EVIDENCE_VIEWS = (
    "MRCONSO_only",
    "MRCONSO_MRREL",
    "MRCONSO_MRREL_MRDEF",
    "MRCONSO_MRREL_MRSTY",
    "full_committed_card",
)


def _ratio(count: int, total: int) -> float:
    return count / total if total else 0.0


def _standard_row(
    item: QAItem,
    experiment: str,
    condition: str,
    metrics: dict,
    aliases: dict | None = None,
) -> dict:
    row = {
        "item_id": item.item_id,
        "dataset": item.dataset,
        "split": item.split,
        "linker": item.linker,
        "experiment": experiment,
        "condition": condition,
        "metrics": dict(metrics),
    }
    row.update(metrics)
    if aliases:
        row.update(aliases)
    return row


def answer_grounding_metrics(item: QAItem) -> dict:
    base_metrics = {
        "question_grounded": bool(item.question_cuis),
        "num_question_cuis": len(set(item.question_cuis)),
        "question_cui_count": len(set(item.question_cuis)),
    }
    if item.dataset.lower() == "pubmedqa":
        base_metrics["answer_option_metrics_applicable"] = False
        return _standard_row(item, "answer_grounding", "item", base_metrics)

    option_counts = {key: len(set(value)) for key, value in item.option_cuis.items()}
    distractors = item.distractor_cuis()
    grounded_distractors = sum(1 for cuis in distractors.values() if cuis)
    num_total_answer_cuis = len({
        cui
        for cuis in item.option_cuis.values()
        for cui in cuis
    })
    metrics = {
        **base_metrics,
        "answer_option_metrics_applicable": True,
        "question_grounded": bool(item.question_cuis),
        "correct_answer_grounded": bool(item.gold_cuis()),
        "all_options_grounded": bool(option_counts) and all(count > 0 for count in option_counts.values()),
        "any_option_grounded": any(count > 0 for count in option_counts.values()),
        "distractor_grounding_rate": _ratio(grounded_distractors, len(distractors)),
        "num_gold_answer_cuis": len(set(item.gold_cuis())),
        "num_total_answer_cuis": num_total_answer_cuis,
        "num_options_with_cui": sum(1 for count in option_counts.values() if count > 0),
        "option_cui_counts": option_counts,
    }
    return _standard_row(item, "answer_grounding", "item", metrics)


def _any_gold(gold_cuis: list[str], predicate) -> bool:
    return any(predicate(cui) for cui in gold_cuis)


def _distractor_metadata_coverage(item: QAItem, metadata: CuiMetadataIndex) -> float:
    distractors = item.distractor_cuis()
    covered = 0
    for cuis in distractors.values():
        if any(metadata.has_name(cui) or metadata.has_definition(cui) or metadata.has_semantic_type(cui)
               for cui in cuis):
            covered += 1
    return _ratio(covered, len(distractors))


def _all_option_cuis(item: QAItem) -> list[str]:
    return [
        cui
        for cuis in item.option_cuis.values()
        for cui in cuis
    ]


def _view_has_name(view: str) -> bool:
    return view in EVIDENCE_VIEWS


def _view_has_relations(view: str) -> bool:
    return view in {
        "MRCONSO_MRREL",
        "MRCONSO_MRREL_MRDEF",
        "MRCONSO_MRREL_MRSTY",
        "full_committed_card",
    }


def _view_has_definitions(view: str) -> bool:
    return view in {"MRCONSO_MRREL_MRDEF", "full_committed_card"}


def _view_has_semantic_types(view: str) -> bool:
    return view in {"MRCONSO_MRREL_MRSTY", "full_committed_card"}


def _cuis_have_view_evidence(
    cuis: list[str],
    view: str,
    metadata: CuiMetadataIndex,
    relations: RelationIndex,
) -> bool:
    if not cuis:
        return False
    if _view_has_name(view) and any(metadata.has_name(cui) for cui in cuis):
        return True
    if _view_has_relations(view) and any(relations.degree(cui) > 0 for cui in cuis):
        return True
    if _view_has_definitions(view) and any(metadata.has_definition(cui) for cui in cuis):
        return True
    if _view_has_semantic_types(view) and any(metadata.has_semantic_type(cui) for cui in cuis):
        return True
    return False


def evidence_availability_rows(
    item: QAItem,
    metadata: CuiMetadataIndex,
    relations: RelationIndex,
) -> list[dict]:
    gold_cuis = item.gold_cuis()
    option_cuis = _all_option_cuis(item)
    relation_degree = sum(relations.degree(cui) for cui in set(gold_cuis))
    base = {
        "gold_answer_grounded": bool(gold_cuis),
        "gold_has_name": _any_gold(gold_cuis, metadata.has_name),
        "gold_has_definition": _any_gold(gold_cuis, metadata.has_definition),
        "gold_has_semantic_type": _any_gold(gold_cuis, metadata.has_semantic_type),
        "gold_relation_degree": relation_degree,
        "distractor_metadata_coverage": _distractor_metadata_coverage(item, metadata),
    }
    rows = []
    for view in EVIDENCE_VIEWS:
        token_estimate = 0
        if _view_has_name(view):
            token_estimate += sum(len(metadata.get(cui).name.split()) for cui in gold_cuis)
        if _view_has_relations(view):
            token_estimate += relation_degree * 3
        if _view_has_definitions(view):
            token_estimate += sum(len(metadata.get(cui).definition.split()) for cui in gold_cuis)
        if _view_has_semantic_types(view):
            token_estimate += sum(len(metadata.get(cui).semantic_types) for cui in gold_cuis)
        gold_evidence = _cuis_have_view_evidence(gold_cuis, view, metadata, relations)
        distractor_evidence = any(
            _cuis_have_view_evidence(cuis, view, metadata, relations)
            for cuis in item.distractor_cuis().values()
        )
        adds_definition = _view_has_definitions(view) and any(
            metadata.has_definition(cui) for cui in option_cuis
        )
        adds_semantic_type = _view_has_semantic_types(view) and any(
            metadata.has_semantic_type(cui) for cui in option_cuis
        )
        metrics = {
            **base,
            "gold_answer_evidence_available": bool(gold_cuis) and gold_evidence,
            "distractor_contrast_available": gold_evidence and distractor_evidence,
            "evidence_view_adds_definition": adds_definition,
            "evidence_view_adds_semantic_type": adds_semantic_type,
            "evidence_view_adds_new_signal_over_relation_only": adds_definition or adds_semantic_type,
            "evidence_token_estimate": token_estimate,
        }
        rows.append(_standard_row(
            item,
            "evidence_availability",
            view,
            metrics,
            aliases={"view": view},
        ))
    return rows


def retrieval_faithfulness_rows(item: QAItem, relations: RelationIndex) -> list[dict]:
    q_cuis = list(dict.fromkeys(item.question_cuis))
    gold_cuis = item.gold_cuis()
    all_option_cuis = [
        cui
        for cuis in item.option_cuis.values()
        for cui in cuis
    ]
    all_option_cuis = list(dict.fromkeys(all_option_cuis))
    distractors = item.distractor_cuis()
    gold_len = relations.shortest_path_length(q_cuis, gold_cuis, max_hops=2)
    distractor_hits = sum(
        relations.shortest_path_length(q_cuis, cuis, max_hops=2) is not None
        for cuis in distractors.values()
    )
    answer_path_nodes = relations.path_nodes_between(q_cuis, all_option_cuis, max_hops=2)
    q_to_a_nodes = answer_path_nodes or set(q_cuis + all_option_cuis)
    random_size = max(1, len(q_to_a_nodes))

    rows: list[dict] = []
    methods = {
        "pooled_1hop": relations.pooled_neighborhood(q_cuis + all_option_cuis, hops=1, cap=500),
        "pooled_2hop_capped": relations.pooled_neighborhood(q_cuis + all_option_cuis, hops=2, cap=500),
        "q_to_a_2hop": q_to_a_nodes,
        "random_same_size": relations.deterministic_sample(random_size, key=item.item_id),
    }
    for method, nodes in methods.items():
        nodes = set(nodes)
        answer_connected = len(nodes & answer_path_nodes)
        metrics = {
            "gold_path_available": gold_len is not None,
            "gold_min_path_length": gold_len,
            "distractor_path_rate": _ratio(distractor_hits, len(distractors)),
            "num_seed_nodes": len(set(q_cuis + all_option_cuis)),
            "num_question_seed_nodes": len(set(q_cuis)),
            "num_answer_seed_nodes": len(set(all_option_cuis)),
            "num_gold_answer_seed_nodes": len(set(gold_cuis)),
            "num_retrieved_nodes": len(nodes),
            "num_retrieved_edges": relations.edge_count(nodes),
            "answer_connected_node_count": answer_connected,
            "decision_relevance_ratio_proxy": answer_connected / max(1, len(nodes)),
            "evidence_token_estimate": len(nodes) * 2 + relations.edge_count(nodes) * 3,
        }
        rows.append(_standard_row(
            item,
            "retrieval_faithfulness",
            method,
            metrics,
            aliases={"method": method},
        ))
    return rows
