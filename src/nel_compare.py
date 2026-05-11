"""
NEL tool — compare entity extraction between MetaMap and scispaCy.

Input files (one per line, JSONL):

  dev.metamap.jsonl  — one record per QUESTION
      {
        "concept": [...],         # entities in the question stem
        "option0": [...],         # entities in option A
        "option1": [...],         # entities in option B
        "option2": [...],         # entities in option C
        "option3": [...]          # entities in option D
      }

  dev.spacy.jsonl    — one record per (question, option) PAIR
                       i.e. four consecutive lines per question
      {
        "sent": "<question stem> <option text>",
        "ans":  "<option text>",
        "qc":   [...],            # entities in the question stem
        "ac":   [...]             # entities in THIS one option only
      }

For each tool we compute per-question entity counts for:
  * Question stem (Q)
  * Options combined  — union of the four option lists (O_union)
  * Options combined  — sum of |option_i| for i in 0..3   (O_sum)

We then report Max / Mean / Std / Var across the dataset for each metric,
and the per-question DIFFERENCE (MetaMap − scispaCy).

Run:
    python nel_tool.py \
        --spacy   dev.spacy.jsonl \
        --metamap dev.metamap.jsonl \
        [--csv    per_question.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {exc}") from exc
    return records


def load_metamap(path: Path) -> list[dict]:
    """One row per question.

    Returns a list of dicts with keys:
        q_concepts:      list[str]
        option_concepts: list[list[str]]   (length 4)
    """
    rows: list[dict] = []
    for rec in _read_jsonl(path):
        options = [rec.get(f"option{i}", []) or [] for i in range(4)]
        rows.append({
            "q_concepts": rec.get("concept", []) or [],
            "option_concepts": options,
        })
    return rows


def load_spacy(path: Path) -> list[dict]:
    """One row per question.

    Returns a list of dicts with keys:
        q_concepts:      list[str]
        option_concepts: list[list[str]]   (length 4)
    """
    rows: list[dict] = []
    for rec in _read_jsonl(path):
        options = [rec.get(f"option{i}", []) or [] for i in range(4)]
        rows.append({
            "q_concepts": rec.get("concept", []) or [],
            "option_concepts": options,
        })
    return rows


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_question_counts(rows: list[dict]) -> dict[str, list[int]]:
    """Produce parallel lists of counts, one entry per question.

    Returned keys:
        q_count    — |entities in question stem|
        o_union    — |unique entities across the 4 options|
        o_sum      — sum of per-option entity counts (with duplicates kept
                     across options, but each option list is taken as-is)
    """
    q_counts: list[int] = []
    o_union_counts: list[int] = []
    o_sum_counts: list[int] = []

    for row in rows:
        q_counts.append(len(list(set(row["q_concepts"]))))

        union: set[str] = set()
        total = 0
        for opt in row["option_concepts"]:
            union.update(opt)
            total += len(opt)
        o_union_counts.append(len(union))
        o_sum_counts.append(total)

    return {
        "q_count": q_counts,
        "o_union": o_union_counts,
        "o_sum": o_sum_counts,
    }


def _stats(values: Iterable[float]) -> dict[str, float]:
    vals = list(values)
    if not vals:
        return {"n": 0, "max": float("nan"), "min": float("nan"),
                "mean": float("nan"), "std": float("nan"), "var": float("nan")}
    mean = statistics.fmean(vals)
    # Sample std / var (ddof=1) when n > 1, else 0.
    if len(vals) > 1:
        var = statistics.variance(vals)
        std = math.sqrt(var)
    else:
        var = 0.0
        std = 0.0
    return {
        "n": len(vals),
        "max": max(vals),
        "min": min(vals),
        "mean": mean,
        "std": std,
        "var": var,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_METRIC_LABELS = {
    "q_count": "Question entities",
    "o_union": "Option entities (union of 4)",
    "o_sum":   "Option entities (sum of 4)",
}


def _fmt(stats: dict[str, float]) -> str:
    return (f"n={stats['n']:>5d}  "
            f"max={stats['max']:>7.2f}  "
            f"min={stats['min']:>7.2f}  "
            f"mean={stats['mean']:>7.3f}  "
            f"std={stats['std']:>7.3f}  "
            f"var={stats['var']:>9.3f}")


def print_report(metamap_counts: dict[str, list[int]],
                 spacy_counts: dict[str, list[int]]) -> None:
    print("=" * 88)
    print("Per-question entity-count statistics")
    print("=" * 88)

    for key, label in _METRIC_LABELS.items():
        mm = metamap_counts[key]
        sp = spacy_counts[key]
        n = min(len(mm), len(sp))
        diffs = [mm[i] - sp[i] for i in range(n)]

        print(f"\n[{label}]")
        print(f"  MetaMap   : {_fmt(_stats(mm))}")
        print(f"  scispaCy  : {_fmt(_stats(sp))}")
        print(f"  Diff(M-S) : {_fmt(_stats(diffs))}")

    print()


def write_csv(out_path: Path,
              metamap_counts: dict[str, list[int]],
              spacy_counts: dict[str, list[int]]) -> None:
    n = min(len(metamap_counts["q_count"]), len(spacy_counts["q_count"]))
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "question_idx",
            "mm_q", "sp_q", "diff_q",
            "mm_o_union", "sp_o_union", "diff_o_union",
            "mm_o_sum", "sp_o_sum", "diff_o_sum",
        ])
        for i in range(n):
            row = [
                i,
                metamap_counts["q_count"][i], spacy_counts["q_count"][i],
                metamap_counts["q_count"][i] - spacy_counts["q_count"][i],
                metamap_counts["o_union"][i], spacy_counts["o_union"][i],
                metamap_counts["o_union"][i] - spacy_counts["o_union"][i],
                metamap_counts["o_sum"][i], spacy_counts["o_sum"][i],
                metamap_counts["o_sum"][i] - spacy_counts["o_sum"][i],
            ]
            writer.writerow(row)
    print(f"Wrote per-question counts to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spacy",   type=Path, required=True,
                        help="Path to scispaCy JSONL (4 rows per question).")
    parser.add_argument("--metamap", type=Path, required=True,
                        help="Path to MetaMap JSONL (1 row per question).")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Optional output CSV with per-question counts.")
    args = parser.parse_args()

    metamap_rows = load_metamap(args.metamap)
    spacy_rows = load_spacy(args.spacy)

    if len(metamap_rows) != len(spacy_rows):
        print(f"WARNING: question count mismatch — "
              f"MetaMap={len(metamap_rows)}  scispaCy={len(spacy_rows)}. "
              f"Stats will be computed on the first "
              f"{min(len(metamap_rows), len(spacy_rows))} questions of each.")

    mm_counts = per_question_counts(metamap_rows)
    sp_counts = per_question_counts(spacy_rows)

    print_report(mm_counts, sp_counts)

    if args.csv:
        write_csv(args.csv, mm_counts, sp_counts)


if __name__ == "__main__":
    main()
