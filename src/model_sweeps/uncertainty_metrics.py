from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from statistics import fmean
from typing import Callable, Iterable, Sequence


def _finite_values(values: Iterable[float | int | None]) -> list[float]:
    return [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]


def _finite_mean(values: Iterable[float | int | None], default: float = 0.0) -> float:
    finite = _finite_values(values)
    return fmean(finite) if finite else default


def softmax_scores(scores: dict[str, float]) -> dict[str, float]:
    finite = {key: float(value) for key, value in scores.items() if math.isfinite(float(value))}
    if not finite:
        return {key: 0.0 for key in scores}
    max_score = max(finite.values())
    exp_values = {key: math.exp(value - max_score) for key, value in finite.items()}
    total = sum(exp_values.values())
    if total <= 0:
        return {key: 0.0 for key in scores}
    return {key: exp_values.get(key, 0.0) / total for key in scores}


def entropy(probabilities: dict[str, float], normalized: bool = False) -> float:
    probs = [float(value) for value in probabilities.values() if value > 0]
    total = -sum(prob * math.log(prob) for prob in probs)
    if not normalized:
        return total
    labels = len(probabilities)
    if labels <= 1:
        return 0.0
    return total / math.log(labels)


def prediction_metrics(probabilities: dict[str, float], gold: str | None = None) -> dict:
    if not probabilities:
        return {
            "prediction": None,
            "confidence": None,
            "margin": None,
            "entropy": None,
            "normalized_entropy": None,
            "nll": None,
            "brier_score": None,
        }

    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    prediction, confidence = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
    out = {
        "prediction": prediction,
        "confidence": confidence,
        "margin": confidence - runner_up,
        "entropy": entropy(probabilities),
        "normalized_entropy": entropy(probabilities, normalized=True),
        "nll": None,
        "brier_score": None,
    }
    if gold is not None and gold in probabilities:
        gold_prob = max(probabilities[gold], 1e-12)
        out["nll"] = -math.log(gold_prob)
        out["brier_score"] = sum(
            (prob - (1.0 if label == gold else 0.0)) ** 2
            for label, prob in probabilities.items()
        )
    return out


def majority_vote(labels: Iterable[str | None]) -> tuple[str | None, bool, float]:
    usable = [label for label in labels if label is not None]
    if not usable:
        return None, False, 0.0
    counts = Counter(usable)
    top_count = max(counts.values())
    tied = sorted(label for label, count in counts.items() if count == top_count)
    return tied[0], len(tied) > 1, top_count / len(usable)


def vote_entropy(labels: Iterable[str | None], all_labels: Sequence[str] | None = None) -> float:
    usable = [label for label in labels if label is not None]
    if not usable:
        return 0.0
    labels_for_denominator = list(all_labels or sorted(set(usable)))
    counts = Counter(usable)
    probabilities = {
        label: counts.get(label, 0) / len(usable)
        for label in labels_for_denominator
    }
    return entropy(probabilities, normalized=True)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    spread = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def bootstrap_ci(
    values: Sequence[float | int | bool],
    reducer: Callable[[list[float]], float] = fmean,
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 13,
) -> tuple[float, float]:
    numeric = [float(value) for value in values]
    if not numeric:
        return 0.0, 0.0
    if len(numeric) == 1:
        value = reducer(numeric)
        return value, value
    rng = random.Random(seed)
    estimates = []
    n = len(numeric)
    for _ in range(iterations):
        sample = [numeric[rng.randrange(n)] for _ in range(n)]
        estimates.append(reducer(sample))
    estimates.sort()
    alpha = 1 - confidence
    low_index = max(0, int(math.floor((alpha / 2) * iterations)))
    high_index = min(iterations - 1, int(math.ceil((1 - alpha / 2) * iterations)) - 1)
    return estimates[low_index], estimates[high_index]


def expected_calibration_error(rows: Iterable[dict], bins: int = 10) -> float:
    usable = [
        (float(row["confidence"]), 1.0 if row.get("is_correct") else 0.0)
        for row in rows
        if isinstance(row.get("confidence"), (int, float))
        and math.isfinite(float(row["confidence"]))
        and isinstance(row.get("is_correct"), bool)
    ]
    if not usable:
        return 0.0
    total = len(usable)
    error = 0.0
    for bin_idx in range(bins):
        low = bin_idx / bins
        high = (bin_idx + 1) / bins
        if bin_idx == bins - 1:
            members = [(conf, corr) for conf, corr in usable if low <= conf <= high]
        else:
            members = [(conf, corr) for conf, corr in usable if low <= conf < high]
        if not members:
            continue
        avg_conf = fmean(conf for conf, _ in members)
        avg_acc = fmean(corr for _, corr in members)
        error += (len(members) / total) * abs(avg_acc - avg_conf)
    return error


def selective_accuracy(rows: Iterable[dict], coverage: float) -> float | None:
    usable = [
        row
        for row in rows
        if isinstance(row.get("confidence"), (int, float))
        and math.isfinite(float(row["confidence"]))
        and isinstance(row.get("is_correct"), bool)
    ]
    if not usable:
        return None
    keep = max(1, int(math.ceil(len(usable) * coverage)))
    selected = sorted(usable, key=lambda row: float(row["confidence"]), reverse=True)[:keep]
    return fmean(1.0 if row["is_correct"] else 0.0 for row in selected)


def summarize_prediction_rows(rows: list[dict]) -> dict:
    attempted = len(rows)
    valid_rows = [row for row in rows if row.get("prediction") is not None and row.get("error") in {None, ""}]
    malformed = attempted - len(valid_rows)
    correctness = [1.0 if row.get("is_correct") else 0.0 for row in valid_rows]
    successes = int(sum(correctness))
    accuracy = fmean(correctness) if correctness else 0.0
    acc_low, acc_high = wilson_interval(successes, len(valid_rows))
    malformed_low, malformed_high = wilson_interval(malformed, attempted)

    return {
        "n": attempted,
        "valid": len(valid_rows),
        "malformed": malformed,
        "malformed_rate": malformed / attempted if attempted else 0.0,
        "malformed_ci_low": malformed_low,
        "malformed_ci_high": malformed_high,
        "accuracy": accuracy,
        "accuracy_ci_low": acc_low,
        "accuracy_ci_high": acc_high,
        "ece": expected_calibration_error(valid_rows),
        "mean_confidence": _finite_mean(row.get("confidence") for row in valid_rows),
        "mean_margin": _finite_mean(row.get("margin") for row in valid_rows),
        "mean_entropy": _finite_mean(row.get("entropy") for row in valid_rows),
        "mean_normalized_entropy": _finite_mean(row.get("normalized_entropy") for row in valid_rows),
        "mean_nll": _finite_mean(row.get("nll") for row in valid_rows),
        "mean_brier_score": _finite_mean(row.get("brier_score") for row in valid_rows),
        "selective_accuracy_50": selective_accuracy(valid_rows, 0.50),
        "selective_accuracy_70": selective_accuracy(valid_rows, 0.70),
        "selective_accuracy_90": selective_accuracy(valid_rows, 0.90),
        "mean_elapsed_s": _finite_mean(row.get("elapsed_s") for row in rows),
    }


def group_rows(rows: Iterable[dict], keys: Sequence[str]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    return dict(groups)
