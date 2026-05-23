"""EL coverage statistics: MetaMap vs scispaCy comparison table.

Reads Exp2 (answer grounding) and Exp3 (retrieval faithfulness) JSONL files
and produces a two-column (MetaMap | scispaCy) summary per dataset.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean_or_na(values: list[float]) -> str:
    return f"{mean(values):.3f}" if values else "n/a"


def compute_el_coverage(
    diag_dir: Path,
    datasets: list[str],
    split: str,
    linkers: list[str],
) -> list[dict]:
    """Return one row per (dataset, metric) with a column per linker."""
    # metric_name -> linker -> list[float]
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for dataset in datasets:
        for linker in linkers:
            # Exp2: answer grounding
            exp2_path = diag_dir / f"exp2_answer_grounding_{dataset}_{split}_{linker}.jsonl"
            if exp2_path.exists():
                for row in _load_jsonl(exp2_path):
                    key = (dataset, linker)
                    data[(dataset, "question_grounded")][linker].append(
                        float(row.get("question_grounded", 0))
                    )
                    data[(dataset, "correct_answer_grounded")][linker].append(
                        float(row.get("correct_answer_grounded", 0))
                    )
                    data[(dataset, "all_options_grounded")][linker].append(
                        float(row.get("all_options_grounded", 0))
                    )
                    data[(dataset, "distractor_grounding_rate")][linker].append(
                        float(row.get("distractor_grounding_rate", 0))
                    )
                    data[(dataset, "num_question_cuis")][linker].append(
                        float(row.get("num_question_cuis", 0))
                    )
                    # avg gold-answer CUIs from num_gold_answer_cuis if present
                    if "num_gold_answer_cuis" in row:
                        data[(dataset, "num_gold_answer_cuis")][linker].append(
                            float(row["num_gold_answer_cuis"])
                        )

            # Exp3: retrieval faithfulness — gold path length and availability
            exp3_path = diag_dir / f"exp3_retrieval_faithfulness_{dataset}_{split}_{linker}.jsonl"
            if exp3_path.exists():
                for row in _load_jsonl(exp3_path):
                    if row.get("condition") != "q_to_a_2hop":
                        continue
                    if row.get("gold_path_available") is not None:
                        data[(dataset, "gold_path_available")][linker].append(
                            float(row["gold_path_available"])
                        )
                    if row.get("gold_min_path_length") is not None:
                        data[(dataset, "gold_min_path_length_mean")][linker].append(
                            float(row["gold_min_path_length"])
                        )
                    data[(dataset, "distractor_path_rate")][linker].append(
                        float(row.get("distractor_path_rate", 0))
                    )

    metric_order = [
        "question_grounded",
        "correct_answer_grounded",
        "all_options_grounded",
        "distractor_grounding_rate",
        "num_question_cuis",
        "num_gold_answer_cuis",
        "gold_path_available",
        "gold_min_path_length_mean",
        "distractor_path_rate",
    ]

    rows_out = []
    for dataset in datasets:
        for metric in metric_order:
            row: dict = {"dataset": dataset, "metric": metric}
            for linker in linkers:
                vals = data.get((dataset, metric), {}).get(linker, [])
                row[linker] = _mean_or_na(vals)
            rows_out.append(row)
    return rows_out


def main() -> None:
    parser = argparse.ArgumentParser(description="EL coverage statistics table.")
    parser.add_argument("--diag-dir", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", default=["medQA", "medMCQA"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--linkers", nargs="+", default=["metamap", "spacy"])
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    rows = compute_el_coverage(args.diag_dir, args.datasets, args.split, args.linkers)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as fh:
        fieldnames = ["dataset", "metric"] + args.linkers
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output_csv}")

    if args.output_md:
        lines = ["| dataset | metric | " + " | ".join(args.linkers) + " |"]
        lines.append("|---|---| " + " | ".join(["---"] * len(args.linkers)) + " |")
        for row in rows:
            vals = " | ".join(row[l] for l in args.linkers)
            lines.append(f"| {row['dataset']} | {row['metric']} | {vals} |")
        args.output_md.write_text("\n".join(lines) + "\n")
        print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
