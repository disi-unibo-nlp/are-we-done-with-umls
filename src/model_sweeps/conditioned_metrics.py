from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.evidence_cards import condition_type_for_condition
from src.model_sweeps.local_logits_sweep import aggregate_trials_by_item
from src.model_sweeps.uncertainty_metrics import summarize_prediction_rows, wilson_interval


DEFAULT_DIAGNOSTICS = (
    "correct_answer_grounded",
    "all_options_grounded",
    "gold_path_available",
    "gold_answer_evidence_available",
    "distractor_contrast_available",
)

DEFAULT_EVIDENCE_CONDITIONS = (
    "relation_only_evidence",
    "metadata_rich_evidence",
    "q_to_a_path_evidence",
    "gold_only_evidence",
    "distractor_contrast_evidence",
    "oracle_gold_evidence",
    "random_same_size_evidence",
    "shuffled_answer_evidence",
)

REALISTIC_CONDITIONS_FOR_CONTROL_GAPS = {
    "relation_only_evidence",
    "metadata_rich_evidence",
    "q_to_a_path_evidence",
    "distractor_contrast_evidence",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_prediction_rows(rows: list[dict]) -> list[dict]:
    if any("run_index" in row or "shuffle_index" in row for row in rows):
        rows = aggregate_trials_by_item(rows)
    normalized = []
    for row in rows:
        out = dict(row)
        out.setdefault("model_key", out.get("model", ""))
        out.setdefault("linker", "metamap")
        out.setdefault("condition", out.get("prompt_condition", out.get("mode", "")))
        out.setdefault("prompt_condition", out.get("condition", out.get("mode", "")))
        out.setdefault("condition_type", condition_type_for_condition(str(out.get("condition", ""))))
        out.setdefault("error", "" if out.get("prediction") is not None else "missing_prediction")
        normalized.append(out)
    return normalized


def parse_filters(filters: list[str] | None) -> dict[str, str]:
    parsed = {}
    for item in filters or []:
        if "=" not in item:
            raise ValueError(f"Diagnostic filter must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def value_matches(raw: object, expected: str) -> bool:
    if isinstance(raw, bool):
        return expected.lower() in {"true", "1", "yes"} if raw else expected.lower() in {"false", "0", "no"}
    return str(raw) == expected


def filter_diagnostics(rows: list[dict], filters: dict[str, str]) -> list[dict]:
    if not filters:
        return rows
    return [
        row
        for row in rows
        if all(value_matches(row.get(key), expected) for key, expected in filters.items())
    ]


def infer_condition_columns(rows: list[dict]) -> list[str]:
    return sorted({
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, bool)
    })


def diagnostic_metrics(row: dict) -> dict:
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        out = dict(metrics)
        for key, value in row.items():
            out.setdefault(key, value)
        return out
    return row


def diagnostic_identity(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row.get("item_id", "")),
        str(row.get("dataset", "")),
        str(row.get("split", "")),
        str(row.get("linker", "")),
    )


def prediction_identity(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row.get("item_id", "")),
        str(row.get("dataset", "")),
        str(row.get("split", "")),
        str(row.get("linker", "metamap")),
    )


def condition_index(rows: list[dict], condition: str) -> dict[str, bool]:
    out = {}
    for row in rows:
        item_id = row.get("item_id")
        metrics = diagnostic_metrics(row)
        if item_id is None or not isinstance(metrics.get(condition), bool):
            continue
        value = bool(metrics[condition])
        out[str(item_id)] = value
        out["::".join(diagnostic_identity(row))] = value
    return out


def conditioned_summary_rows(
    prediction_rows: list[dict],
    diagnostic_rows: list[dict],
    conditions: list[str] | None = None,
) -> list[dict]:
    resolved_conditions = conditions or infer_condition_columns(diagnostic_rows)
    out = []
    for condition in resolved_conditions:
        by_item = condition_index(diagnostic_rows, condition)
        for condition_value in (False, True):
            subset = [
                row
                for row in prediction_rows
                if by_item.get("::".join(prediction_identity(row)), by_item.get(str(row.get("item_id")))) is condition_value
            ]
            if not subset:
                continue
            summary = summarize_prediction_rows(subset)
            summary["mean_confidence"] = _mean_metric(subset, "confidence")
            summary["mean_margin"] = _mean_metric(subset, "margin")
            summary["mean_entropy"] = _mean_metric(subset, "entropy")
            summary["mean_normalized_entropy"] = _mean_metric(subset, "normalized_entropy")
            summary["mean_nll"] = _mean_metric(subset, "nll")
            summary["mean_brier_score"] = _mean_metric(subset, "brier_score")
            successes = sum(1 for row in subset if row.get("is_correct"))
            ci_low, ci_high = wilson_interval(successes, len(subset))
            out.append({
                "condition": condition,
                "condition_value": condition_value,
                **summary,
                "trial_accuracy_ci_low": ci_low,
                "trial_accuracy_ci_high": ci_high,
            })
    return out


def _row_condition(row: dict) -> str:
    return str(row.get("condition", row.get("prompt_condition", row.get("mode", ""))))


def _row_condition_type(row: dict) -> str:
    value = row.get("condition_type")
    if value:
        return str(value)
    return condition_type_for_condition(_row_condition(row))


def _model_group(row: dict) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("split", "")),
        str(row.get("linker", "metamap")),
        str(row.get("model_key", row.get("model", ""))),
        str(row.get("model", "")),
        str(row.get("mode", "")),
    )


def _item_model_key(row: dict) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("item_id", "")),
        str(row.get("dataset", "")),
        str(row.get("split", "")),
        str(row.get("linker", "metamap")),
        str(row.get("model_key", row.get("model", ""))),
        str(row.get("model", "")),
    )


def _prediction_has_diagnostic(row: dict, by_item: dict[str, bool], value: bool) -> bool:
    return by_item.get("::".join(prediction_identity(row)), by_item.get(str(row.get("item_id")))) is value


def _mean_metric(rows: list[dict], metric: str) -> float | None:
    values = [
        float(row[metric])
        for row in rows
        if isinstance(row.get(metric), (int, float))
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _valid_prediction_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("prediction") is not None and row.get("error") in {None, ""}]


def _accuracy(rows: list[dict]) -> float:
    valid = _valid_prediction_rows(rows)
    if not valid:
        return 0.0
    return sum(1 for row in valid if row.get("is_correct")) / len(valid)


def _malformed_rate(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    malformed = sum(1 for row in rows if row.get("prediction") is None or row.get("error") not in {None, ""})
    return malformed / len(rows)


def _summary_for_condition(rows: list[dict], condition: str) -> dict:
    subset = [row for row in rows if _row_condition(row) == condition]
    summary = summarize_prediction_rows(subset)
    return {**summary, "rows": subset}


def _summary_subset(rows: list[dict], condition: str) -> dict:
    subset = [row for row in rows if _row_condition(row) == condition]
    summary = summarize_prediction_rows(subset)
    return {**summary, "rows": subset}


def _metric_or_none(summary: dict, key: str) -> float | None:
    value = summary.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _add_control_accuracy_gaps(rows: list[dict]) -> None:
    by_group = {
        (
            row["dataset"],
            row["split"],
            row["linker"],
            row["model_key"],
            row["model"],
            row["mode"],
            row["diagnostic_name"],
            row["diagnostic_value"],
            row.get("condition", row.get("evidence_condition")),
        ): row
        for row in rows
    }
    for row in rows:
        row["accuracy_gap_vs_random"] = None
        row["accuracy_gap_vs_shuffled"] = None
        row["net_correction_gap_vs_random"] = None
        row["net_correction_gap_vs_shuffled"] = None
        if row.get("condition", row.get("evidence_condition")) not in REALISTIC_CONDITIONS_FOR_CONTROL_GAPS:
            continue
        prefix = (
            row["dataset"],
            row["split"],
            row["linker"],
            row["model_key"],
            row["model"],
            row["mode"],
            row["diagnostic_name"],
            row["diagnostic_value"],
        )
        random_row = by_group.get((*prefix, "random_same_size_evidence"))
        shuffled_row = by_group.get((*prefix, "shuffled_answer_evidence"))
        if random_row is not None:
            row["accuracy_gap_vs_random"] = row["accuracy"] - random_row["accuracy"]
            if isinstance(row.get("net_correction"), (int, float)) and isinstance(random_row.get("net_correction"), (int, float)):
                row["net_correction_gap_vs_random"] = row["net_correction"] - random_row["net_correction"]
        if shuffled_row is not None:
            row["accuracy_gap_vs_shuffled"] = row["accuracy"] - shuffled_row["accuracy"]
            if isinstance(row.get("net_correction"), (int, float)) and isinstance(shuffled_row.get("net_correction"), (int, float)):
                row["net_correction_gap_vs_shuffled"] = row["net_correction"] - shuffled_row["net_correction"]


def intervention_by_diagnostic_rows(
    prediction_rows: list[dict],
    diagnostic_rows: list[dict],
    diagnostics_to_include: list[str] | None = None,
    evidence_conditions: list[str] | None = None,
    baseline_condition: str = "no_umls",
) -> list[dict]:
    predictions = normalize_prediction_rows(prediction_rows)
    resolved_diagnostics = diagnostics_to_include or list(DEFAULT_DIAGNOSTICS)
    resolved_conditions = evidence_conditions or list(DEFAULT_EVIDENCE_CONDITIONS)
    out: list[dict] = []
    for diagnostic_name in resolved_diagnostics:
        by_item = condition_index(diagnostic_rows, diagnostic_name)
        for diagnostic_value in (True, False):
            diagnostic_subset = [
                row
                for row in predictions
                if _prediction_has_diagnostic(row, by_item, diagnostic_value)
            ]
            if not diagnostic_subset:
                continue
            groups: dict[tuple[str, str, str, str, str, str], list[dict]] = {}
            for row in diagnostic_subset:
                groups.setdefault(_model_group(row), []).append(row)
            for group_key, group in sorted(groups.items()):
                dataset, split, linker, model_key, model, mode = group_key
                baseline_rows = [row for row in group if _row_condition(row) == baseline_condition]
                baseline_accuracy = _accuracy(baseline_rows)
                baseline_by_key = {
                    _item_model_key(row): row
                    for row in baseline_rows
                }
                for evidence_condition in resolved_conditions:
                    condition_rows = [row for row in group if _row_condition(row) == evidence_condition]
                    if not condition_rows:
                        continue
                    pairs = [
                        (baseline_by_key[_item_model_key(row)], row)
                        for row in condition_rows
                        if _item_model_key(row) in baseline_by_key
                    ]
                    paired_summary = _paired_effect_summary(pairs)
                    accuracy = _accuracy(condition_rows)
                    out.append({
                        "dataset": dataset,
                        "split": split,
                        "linker": linker,
                        "model_key": model_key,
                        "model": model,
                        "mode": mode,
                        "diagnostic_name": diagnostic_name,
                        "diagnostic_value": diagnostic_value,
                        "condition": evidence_condition,
                        "condition_type": condition_type_for_condition(evidence_condition),
                        "evidence_condition": evidence_condition,
                        "n": len(condition_rows),
                        "accuracy": accuracy,
                        "delta_vs_no_umls": accuracy - baseline_accuracy,
                        "malformed_rate": _malformed_rate(condition_rows),
                        "wrong_to_right": paired_summary["wrong_to_right"],
                        "right_to_wrong": paired_summary["right_to_wrong"],
                        "net_correction": paired_summary["net_correction"],
                        "mean_confidence": _mean_metric(condition_rows, "confidence"),
                        "mean_margin": _mean_metric(condition_rows, "margin"),
                        "mean_entropy": _mean_metric(condition_rows, "entropy"),
                        "mean_confidence_delta": paired_summary["mean_confidence_delta"],
                        "mean_margin_delta": paired_summary["mean_margin_delta"],
                        "mean_entropy_delta": paired_summary["mean_entropy_delta"],
                    })
    _add_control_accuracy_gaps(out)
    return out


def _condition_index_by_key(predictions: list[dict]) -> dict[tuple, dict[str, dict]]:
    out: dict[tuple, dict[str, dict]] = {}
    for row in predictions:
        out.setdefault(_item_model_key(row), {})[_row_condition(row)] = row
    return out


def _paired_metric_delta(condition_row: dict, baseline_row: dict, metric: str) -> float | None:
    condition_value = condition_row.get(metric)
    baseline_value = baseline_row.get(metric)
    if isinstance(condition_value, (int, float)) and isinstance(baseline_value, (int, float)):
        return float(condition_value) - float(baseline_value)
    return None


def _mean_optional(values: list[float | None]) -> float | None:
    usable = [value for value in values if isinstance(value, (int, float))]
    return sum(usable) / len(usable) if usable else None


def _paired_effect_summary(pairs: list[tuple[dict, dict]]) -> dict:
    wrong_to_right = 0
    right_to_wrong = 0
    same_correct = 0
    same_wrong = 0
    no_umls_correct = 0
    condition_correct = 0
    confidence_deltas: list[float | None] = []
    margin_deltas: list[float | None] = []
    entropy_deltas: list[float | None] = []
    for baseline, condition_row in pairs:
        baseline_correct = bool(baseline.get("is_correct"))
        evidence_correct = bool(condition_row.get("is_correct"))
        no_umls_correct += 1 if baseline_correct else 0
        condition_correct += 1 if evidence_correct else 0
        if not baseline_correct and evidence_correct:
            wrong_to_right += 1
        elif baseline_correct and not evidence_correct:
            right_to_wrong += 1
        elif baseline_correct and evidence_correct:
            same_correct += 1
        else:
            same_wrong += 1
        confidence_deltas.append(_paired_metric_delta(condition_row, baseline, "confidence"))
        margin_deltas.append(_paired_metric_delta(condition_row, baseline, "margin"))
        entropy_deltas.append(_paired_metric_delta(condition_row, baseline, "entropy"))
    return {
        "no_umls_correct": no_umls_correct,
        "condition_correct": condition_correct,
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "same_correct": same_correct,
        "same_wrong": same_wrong,
        "net_correction": wrong_to_right - right_to_wrong,
        "confidence_delta": _mean_optional(confidence_deltas),
        "margin_delta": _mean_optional(margin_deltas),
        "entropy_delta": _mean_optional(entropy_deltas),
        "mean_confidence_delta": _mean_optional(confidence_deltas),
        "mean_margin_delta": _mean_optional(margin_deltas),
        "mean_entropy_delta": _mean_optional(entropy_deltas),
    }


def _add_paired_control_gaps(rows: list[dict]) -> None:
    by_group = {
        (
            row["dataset"],
            row["split"],
            row["linker"],
            row["model_key"],
            row["model"],
            row["diagnostic_name"],
            row["diagnostic_value"],
            row["evidence_condition"],
        ): row
        for row in rows
    }
    for row in rows:
        row["accuracy_gap_vs_random"] = None
        row["accuracy_gap_vs_shuffled"] = None
        row["net_correction_gap_vs_random"] = None
        row["net_correction_gap_vs_shuffled"] = None
        if row["evidence_condition"] not in REALISTIC_CONDITIONS_FOR_CONTROL_GAPS:
            continue
        prefix = (
            row["dataset"],
            row["split"],
            row["linker"],
            row["model_key"],
            row["model"],
            row["diagnostic_name"],
            row["diagnostic_value"],
        )
        random_row = by_group.get((*prefix, "random_same_size_evidence"))
        shuffled_row = by_group.get((*prefix, "shuffled_answer_evidence"))
        if random_row is not None:
            row["accuracy_gap_vs_random"] = row["condition_accuracy"] - random_row["condition_accuracy"]
            row["net_correction_gap_vs_random"] = row["net_correction"] - random_row["net_correction"]
        if shuffled_row is not None:
            row["accuracy_gap_vs_shuffled"] = row["condition_accuracy"] - shuffled_row["condition_accuracy"]
            row["net_correction_gap_vs_shuffled"] = row["net_correction"] - shuffled_row["net_correction"]


def paired_intervention_effect_rows(
    prediction_rows: list[dict],
    diagnostic_rows: list[dict],
    diagnostics_to_include: list[str] | None = None,
    evidence_conditions: list[str] | None = None,
    baseline_condition: str = "no_umls",
) -> list[dict]:
    predictions = normalize_prediction_rows(prediction_rows)
    by_key = _condition_index_by_key(predictions)
    resolved_diagnostics = diagnostics_to_include or list(DEFAULT_DIAGNOSTICS)
    resolved_conditions = evidence_conditions or list(DEFAULT_EVIDENCE_CONDITIONS)
    out: list[dict] = []
    diagnostic_specs: list[tuple[str, object, dict[str, bool] | None]] = [("all", "all", None)]
    for diagnostic_name in resolved_diagnostics:
        diagnostic_by_item = condition_index(diagnostic_rows, diagnostic_name)
        diagnostic_specs.extend((diagnostic_name, diagnostic_value, diagnostic_by_item) for diagnostic_value in (True, False))

    for diagnostic_name, diagnostic_value, diagnostic_by_item in diagnostic_specs:
        condition_groups: dict[tuple[str, str, str, str, str, str], list[tuple[dict, dict]]] = {}
        for key, condition_rows in by_key.items():
            baseline = condition_rows.get(baseline_condition)
            if baseline is None:
                continue
            if diagnostic_by_item is not None and not _prediction_has_diagnostic(baseline, diagnostic_by_item, bool(diagnostic_value)):
                continue
            for evidence_condition in resolved_conditions:
                if evidence_condition == baseline_condition or evidence_condition not in condition_rows:
                    continue
                condition_row = condition_rows[evidence_condition]
                group_key = (
                    str(condition_row.get("dataset", "")),
                    str(condition_row.get("split", "")),
                    str(condition_row.get("linker", "metamap")),
                    str(condition_row.get("model_key", condition_row.get("model", ""))),
                    str(condition_row.get("model", "")),
                    evidence_condition,
                )
                condition_groups.setdefault(group_key, []).append((baseline, condition_row))
        for group_key, pairs in sorted(condition_groups.items()):
            dataset, split, linker, model_key, model, evidence_condition = group_key
            paired_summary = _paired_effect_summary(pairs)
            n = len(pairs)
            out.append({
                "dataset": dataset,
                "split": split,
                "linker": linker,
                "model_key": model_key,
                "model": model,
                "diagnostic_name": diagnostic_name,
                "diagnostic_value": diagnostic_value,
                "condition": evidence_condition,
                "condition_type": condition_type_for_condition(evidence_condition),
                "evidence_condition": evidence_condition,
                "n": n,
                "no_umls_accuracy": paired_summary["no_umls_correct"] / n if n else 0.0,
                "condition_accuracy": paired_summary["condition_correct"] / n if n else 0.0,
                **paired_summary,
            })
    _add_paired_control_gaps(out)
    return out


def conditioned_delta_rows(
    prediction_rows: list[dict],
    diagnostic_rows: list[dict],
    conditions: list[str],
    baseline_condition: str = "no_umls",
    umls_condition: str = "q_to_a_path_evidence",
) -> list[dict]:
    predictions = normalize_prediction_rows(prediction_rows)
    out: list[dict] = []
    for condition in conditions:
        by_item = condition_index(diagnostic_rows, condition)
        for condition_value in (True, False):
            subset = [
                row
                for row in predictions
                if by_item.get("::".join(prediction_identity(row)), by_item.get(str(row.get("item_id")))) is condition_value
            ]
            baseline = _summary_for_condition(subset, baseline_condition)
            umls = _summary_for_condition(subset, umls_condition)
            if not baseline["rows"] and not umls["rows"]:
                continue
            baseline_conf = _mean_metric(baseline["rows"], "confidence")
            umls_conf = _mean_metric(umls["rows"], "confidence")
            baseline_entropy = _mean_metric(baseline["rows"], "entropy")
            umls_entropy = _mean_metric(umls["rows"], "entropy")
            out.append({
                "condition": condition,
                "condition_value": condition_value,
                "n": max(baseline["n"], umls["n"]),
                "no_umls_acc": baseline["accuracy"],
                "umls_acc": umls["accuracy"],
                "no_umls_accuracy": baseline["accuracy"],
                "umls_accuracy": umls["accuracy"],
                "delta": umls["accuracy"] - baseline["accuracy"],
                "confidence_delta": (
                    umls_conf - baseline_conf
                    if umls_conf is not None and baseline_conf is not None
                    else None
                ),
                "entropy_delta": (
                    umls_entropy - baseline_entropy
                    if umls_entropy is not None and baseline_entropy is not None
                    else None
                ),
            })
    return out


DOWNSTREAM_TABLE_TYPES = {
    "realistic": {"baseline", "realistic", "realistic_or_contrastive"},
    "controls": {"negative_control"},
    "diagnostic_oracle": {"diagnostic", "oracle"},
}


def _add_downstream_control_gaps(rows: list[dict]) -> None:
    by_group = {
        (
            row["dataset"],
            row["split"],
            row["linker"],
            row["model_key"],
            row["model"],
            row["mode"],
            row["condition"],
        ): row
        for row in rows
    }
    for row in rows:
        row["accuracy_gap_vs_random"] = None
        row["accuracy_gap_vs_shuffled"] = None
        row["net_correction_gap_vs_random"] = None
        row["net_correction_gap_vs_shuffled"] = None
        if row["condition"] not in REALISTIC_CONDITIONS_FOR_CONTROL_GAPS:
            continue
        prefix = (
            row["dataset"],
            row["split"],
            row["linker"],
            row["model_key"],
            row["model"],
            row["mode"],
        )
        random_row = by_group.get((*prefix, "random_same_size_evidence"))
        shuffled_row = by_group.get((*prefix, "shuffled_answer_evidence"))
        if random_row is not None:
            row["accuracy_gap_vs_random"] = row["accuracy"] - random_row["accuracy"]
            row["net_correction_gap_vs_random"] = row["net_correction"] - random_row["net_correction"]
        if shuffled_row is not None:
            row["accuracy_gap_vs_shuffled"] = row["accuracy"] - shuffled_row["accuracy"]
            row["net_correction_gap_vs_shuffled"] = row["net_correction"] - shuffled_row["net_correction"]


def downstream_condition_summary_rows(
    prediction_rows: list[dict],
    table_kind: str,
    baseline_condition: str = "no_umls",
) -> list[dict]:
    if table_kind not in DOWNSTREAM_TABLE_TYPES:
        raise ValueError(f"Unknown downstream table kind: {table_kind}")
    predictions = normalize_prediction_rows(prediction_rows)
    groups: dict[tuple[str, str, str, str, str, str], list[dict]] = {}
    for row in predictions:
        groups.setdefault(_model_group(row), []).append(row)

    all_rows: list[dict] = []
    for group_key, group in sorted(groups.items()):
        dataset, split, linker, model_key, model, mode = group_key
        baseline_rows = [row for row in group if _row_condition(row) == baseline_condition]
        baseline_accuracy = _accuracy(baseline_rows)
        baseline_by_key = {_item_model_key(row): row for row in baseline_rows}
        conditions = sorted({_row_condition(row) for row in group})
        for condition in conditions:
            condition_rows = [row for row in group if _row_condition(row) == condition]
            condition_type = condition_type_for_condition(condition)
            pairs = [
                (baseline_by_key[_item_model_key(row)], row)
                for row in condition_rows
                if condition != baseline_condition and _item_model_key(row) in baseline_by_key
            ]
            paired_summary = _paired_effect_summary(pairs)
            accuracy = _accuracy(condition_rows)
            all_rows.append({
                "dataset": dataset,
                "split": split,
                "linker": linker,
                "model_key": model_key,
                "model": model,
                "mode": mode,
                "condition": condition,
                "condition_type": condition_type,
                "n": len(condition_rows),
                "accuracy": accuracy,
                "delta_vs_no_umls": accuracy - baseline_accuracy,
                "malformed_rate": _malformed_rate(condition_rows),
                "wrong_to_right": paired_summary["wrong_to_right"],
                "right_to_wrong": paired_summary["right_to_wrong"],
                "net_correction": paired_summary["net_correction"],
                "mean_confidence": _mean_metric(condition_rows, "confidence"),
                "mean_margin": _mean_metric(condition_rows, "margin"),
                "mean_entropy": _mean_metric(condition_rows, "entropy"),
            })
    _add_downstream_control_gaps(all_rows)
    allowed_types = DOWNSTREAM_TABLE_TYPES[table_kind]
    return [row for row in all_rows if row["condition_type"] in allowed_types]


def main_claim_summary_rows() -> list[dict]:
    return [
        {
            "claim_name": "Evidence view matters.",
            "diagnostic_metric": "gold_answer_evidence_available, evidence_view_adds_new_signal_over_relation_only",
            "diagnostic_result_summary": "Compare relation-only evidence with metadata-rich committed UMLS cards.",
            "downstream_metric": "delta_vs_no_umls for relation_only_evidence versus metadata_rich_evidence",
            "downstream_result_summary": "Fill after downstream probe runs; report paired effects only for realistic conditions.",
            "control_check": "metadata_rich_evidence should be interpreted against random_same_size_evidence and shuffled_answer_evidence where available.",
            "safe_interpretation": "Richer committed UMLS metadata exposes evidence not visible in relation-only views; metadata availability is not clinical sufficiency.",
        },
        {
            "claim_name": "Answer grounding matters.",
            "diagnostic_metric": "correct_answer_grounded, all_options_grounded, distractor_grounding_rate",
            "diagnostic_result_summary": "Report answer-option grounding separately from question grounding for MedQA and MedMCQA.",
            "downstream_metric": "paired net_correction conditioned on correct_answer_grounded and all_options_grounded",
            "downstream_result_summary": "Fill after downstream probe runs; improvements are meaningful only when answer CUIs enter UMLS space.",
            "control_check": "Realistic evidence should exceed random and shuffled controls within grounded subsets.",
            "safe_interpretation": "UMLS cannot support multiple-choice decisions when answer options are not grounded.",
        },
        {
            "claim_name": "Retrieval faithfulness matters.",
            "diagnostic_metric": "gold_path_available, decision_relevance_ratio_proxy",
            "diagnostic_result_summary": "Compare q_to_a_2hop retrieval with pooled and random retrieval; graph size alone is not enough.",
            "downstream_metric": "delta_vs_no_umls and net_correction for q_to_a_path_evidence conditioned on gold_path_available",
            "downstream_result_summary": "Fill after downstream probe runs; report gaps versus random_same_size_evidence and shuffled_answer_evidence.",
            "control_check": "q_to_a_path_evidence effect should exceed random_same_size_evidence and shuffled_answer_evidence.",
            "safe_interpretation": "UMLS retrieval should be evaluated by decision relevance, not graph size alone; this does not prove model path reasoning.",
        },
        {
            "claim_name": "Controlled UMLS interventions affect downstream behavior when diagnostics succeed.",
            "diagnostic_metric": "correct_answer_grounded, gold_path_available, gold_answer_evidence_available, distractor_contrast_available",
            "diagnostic_result_summary": "Use diagnostic-success strata before interpreting LLM probe behavior.",
            "downstream_metric": "paired wrong_to_right, right_to_wrong, net_correction, confidence_delta, entropy_delta",
            "downstream_result_summary": "Fill after downstream probe runs; diagnostic and oracle conditions are upper-bound or sanity-check analyses.",
            "control_check": "Every realistic intervention is paired against no_umls and compared with random/shuffled negative controls.",
            "safe_interpretation": "LLMs are downstream probes of constructed evidence conditions, not the main object of evaluation and not proof that UMLS universally improves QA.",
        },
    ]


def write_delta_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "condition_value",
        "n",
        "no_umls_acc",
        "umls_acc",
        "delta",
        "confidence_delta",
        "entropy_delta",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.6f}" if isinstance(value, float) else value
                for key, value in row.items()
                if key in fieldnames
            })


def write_delta_markdown(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Condition | Value | n | no-UMLS acc | UMLS acc | delta | confidence delta | entropy delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        def fmt(value: object) -> str:
            if isinstance(value, float):
                return f"{value:.3f}"
            if value is None:
                return "NA"
            return str(value)

        lines.append(
            "| "
            + " | ".join([
                str(row["condition"]),
                str(row["condition_value"]),
                str(row["n"]),
                fmt(row["no_umls_acc"]),
                fmt(row["umls_acc"]),
                fmt(row["delta"]),
                fmt(row["confidence_delta"]),
                fmt(row["entropy_delta"]),
            ])
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_table_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return ""
    return str(value)


def _format_markdown_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return "NA"
    return str(value)


def _write_rows_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_table_value(row.get(key)) for key in fieldnames})


def _write_rows_markdown(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_markdown_value(row.get(key)) for key in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


INTERVENTION_FIELDNAMES = [
    "dataset",
    "split",
    "linker",
    "model_key",
    "model",
    "mode",
    "diagnostic_name",
    "diagnostic_value",
    "condition",
    "condition_type",
    "evidence_condition",
    "n",
    "accuracy",
    "delta_vs_no_umls",
    "malformed_rate",
    "wrong_to_right",
    "right_to_wrong",
    "net_correction",
    "mean_confidence",
    "mean_margin",
    "mean_entropy",
    "mean_confidence_delta",
    "mean_margin_delta",
    "mean_entropy_delta",
    "accuracy_gap_vs_random",
    "accuracy_gap_vs_shuffled",
    "net_correction_gap_vs_random",
    "net_correction_gap_vs_shuffled",
]


PAIRED_FIELDNAMES = [
    "dataset",
    "split",
    "linker",
    "model_key",
    "model",
    "diagnostic_name",
    "diagnostic_value",
    "condition",
    "condition_type",
    "evidence_condition",
    "n",
    "no_umls_correct",
    "condition_correct",
    "no_umls_accuracy",
    "condition_accuracy",
    "wrong_to_right",
    "right_to_wrong",
    "same_correct",
    "same_wrong",
    "net_correction",
    "confidence_delta",
    "margin_delta",
    "entropy_delta",
    "accuracy_gap_vs_random",
    "accuracy_gap_vs_shuffled",
    "net_correction_gap_vs_random",
    "net_correction_gap_vs_shuffled",
]


DOWNSTREAM_SUMMARY_FIELDNAMES = [
    "dataset",
    "split",
    "linker",
    "model_key",
    "model",
    "mode",
    "condition",
    "condition_type",
    "n",
    "accuracy",
    "delta_vs_no_umls",
    "malformed_rate",
    "wrong_to_right",
    "right_to_wrong",
    "net_correction",
    "mean_confidence",
    "mean_margin",
    "mean_entropy",
    "accuracy_gap_vs_random",
    "accuracy_gap_vs_shuffled",
    "net_correction_gap_vs_random",
    "net_correction_gap_vs_shuffled",
]


MAIN_CLAIM_FIELDNAMES = [
    "claim_name",
    "diagnostic_metric",
    "diagnostic_result_summary",
    "downstream_metric",
    "downstream_result_summary",
    "control_check",
    "safe_interpretation",
]


def write_intervention_by_diagnostic_outputs(csv_path: Path, md_path: Path | None, rows: list[dict]) -> None:
    _write_rows_csv(csv_path, rows, INTERVENTION_FIELDNAMES)
    if md_path is not None:
        _write_rows_markdown(md_path, rows, INTERVENTION_FIELDNAMES)


def write_paired_intervention_effect_outputs(csv_path: Path, md_path: Path | None, rows: list[dict]) -> None:
    _write_rows_csv(csv_path, rows, PAIRED_FIELDNAMES)
    if md_path is not None:
        _write_rows_markdown(md_path, rows, PAIRED_FIELDNAMES)


def write_downstream_condition_summary_outputs(csv_path: Path, md_path: Path | None, rows: list[dict]) -> None:
    _write_rows_csv(csv_path, rows, DOWNSTREAM_SUMMARY_FIELDNAMES)
    if md_path is not None:
        _write_rows_markdown(md_path, rows, DOWNSTREAM_SUMMARY_FIELDNAMES)


def write_main_claim_summary_outputs(csv_path: Path, md_path: Path | None, rows: list[dict]) -> None:
    _write_rows_csv(csv_path, rows, MAIN_CLAIM_FIELDNAMES)
    if md_path is not None:
        _write_rows_markdown(md_path, rows, MAIN_CLAIM_FIELDNAMES)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "condition_value",
        "n",
        "valid",
        "malformed",
        "malformed_rate",
        "accuracy",
        "accuracy_ci_low",
        "accuracy_ci_high",
        "ece",
        "mean_confidence",
        "mean_margin",
        "mean_entropy",
        "mean_normalized_entropy",
        "mean_nll",
        "mean_brier_score",
        "selective_accuracy_50",
        "selective_accuracy_70",
        "selective_accuracy_90",
    ]
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
    parser = argparse.ArgumentParser(description="Summarize model predictions under diagnostic conditions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--diagnostics", nargs="+", type=Path, required=True)
    parser.add_argument("--diagnostic-filter", action="append", default=None)
    parser.add_argument("--condition", action="append", default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--delta-table", action="store_true")
    parser.add_argument("--intervention-table", action="store_true")
    parser.add_argument("--paired-table", action="store_true")
    parser.add_argument("--downstream-table-kind", choices=sorted(DOWNSTREAM_TABLE_TYPES), default=None)
    parser.add_argument("--main-claim-summary", action="store_true")
    parser.add_argument("--baseline-condition", default="no_umls")
    parser.add_argument("--umls-condition", default="q_to_a_path_evidence")
    parser.add_argument("--evidence-condition", action="append", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictions = normalize_prediction_rows(read_jsonl(args.predictions))
    diagnostics = []
    for path in args.diagnostics:
        diagnostics.extend(read_jsonl(path))
    diagnostics = filter_diagnostics(diagnostics, parse_filters(args.diagnostic_filter))
    if args.main_claim_summary:
        rows = main_claim_summary_rows()
        write_main_claim_summary_outputs(args.output_csv, args.output_md, rows)
    elif args.downstream_table_kind:
        rows = downstream_condition_summary_rows(
            predictions,
            table_kind=args.downstream_table_kind,
            baseline_condition=args.baseline_condition,
        )
        write_downstream_condition_summary_outputs(args.output_csv, args.output_md, rows)
    elif args.intervention_table:
        rows = intervention_by_diagnostic_rows(
            predictions,
            diagnostics,
            diagnostics_to_include=args.condition or list(DEFAULT_DIAGNOSTICS),
            evidence_conditions=args.evidence_condition or list(DEFAULT_EVIDENCE_CONDITIONS),
            baseline_condition=args.baseline_condition,
        )
        write_intervention_by_diagnostic_outputs(args.output_csv, args.output_md, rows)
    elif args.paired_table:
        rows = paired_intervention_effect_rows(
            predictions,
            diagnostics,
            diagnostics_to_include=args.condition or list(DEFAULT_DIAGNOSTICS),
            evidence_conditions=args.evidence_condition or list(DEFAULT_EVIDENCE_CONDITIONS),
            baseline_condition=args.baseline_condition,
        )
        write_paired_intervention_effect_outputs(args.output_csv, args.output_md, rows)
    elif args.delta_table:
        rows = conditioned_delta_rows(
            predictions,
            diagnostics,
            conditions=args.condition or infer_condition_columns(diagnostics),
            baseline_condition=args.baseline_condition,
            umls_condition=args.umls_condition,
        )
        write_delta_csv(args.output_csv, rows)
        if args.output_md is not None:
            write_delta_markdown(args.output_md, rows)
    else:
        rows = conditioned_summary_rows(predictions, diagnostics, conditions=args.condition)
        write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
