from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.cli_utils import (
    collect_cuis,
    default_output_paths,
    load_items,
    write_jsonl,
    write_summary_csv,
)
from src.experiments.diagnostics import evidence_availability_rows
from src.experiments.umls_index import CuiMetadataIndex, RelationIndex


EXPERIMENT = "evidence_availability"
OUTPUT_STEM = "exp1_evidence_availability"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit committed UMLS metadata availability.")
    parser.add_argument("--dataset", required=True, help="Dataset directory name, e.g. medQA.")
    parser.add_argument("--split", required=True, help="Split name, e.g. test.")
    parser.add_argument("--linker", choices=["metamap", "spacy"], required=True)
    parser.add_argument("--input", type=Path, default=None, help="Override mapped input JSONL.")
    parser.add_argument("--reference", type=Path, default=None, help="Reference question-level JSONL for scispaCy.")
    parser.add_argument("--metadata-zip", type=Path, default=Path("data/UMLS/cui_detail.tsv.zip"))
    parser.add_argument("--triplets-zip", type=Path, default=Path("data/UMLS/triplets.csv.zip"))
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit items for smoke runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned inputs without writing outputs.")
    return parser


def run(args: argparse.Namespace) -> list[dict]:
    items = load_items(args.dataset, args.split, args.linker, args.input, args.reference, args.limit)
    if args.dry_run:
        print(f"{EXPERIMENT}: {len(items)} items from {args.dataset}/{args.split}/{args.linker}")
        return []
    metadata = CuiMetadataIndex.from_zip(args.metadata_zip)
    relations = RelationIndex.from_zip(args.triplets_zip, keep_cuis=collect_cuis(items))
    rows: list[dict] = []
    for item in items:
        for row in evidence_availability_rows(item, metadata, relations):
            row.setdefault("experiment", EXPERIMENT)
            rows.append(row)
    output_jsonl, summary_csv = default_output_paths(OUTPUT_STEM, args.dataset, args.split, args.linker)
    write_jsonl(args.output_jsonl or output_jsonl, rows)
    write_summary_csv(args.summary_csv or summary_csv, rows)
    return rows


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
