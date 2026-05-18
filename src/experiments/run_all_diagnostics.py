from __future__ import annotations

import argparse
import subprocess
import sys


MATRIX = (
    ("medQA", ("dev", "test"), ("metamap", "spacy"), ("exp1", "exp2", "exp3")),
    ("medMCQA", ("dev", "test"), ("metamap", "spacy"), ("exp1", "exp2", "exp3")),
    ("pubmedQA", ("dev", "test", "train"), ("metamap", "spacy"), ("exp1", "exp2")),
)

SCRIPT_BY_EXPERIMENT = {
    "exp1": "src/experiments/exp1_evidence_availability.py",
    "exp2": "src/experiments/exp2_answer_grounding.py",
    "exp3": "src/experiments/exp3_retrieval_faithfulness.py",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the grounded v1 UMLS diagnostics matrix.")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without running them.")
    parser.add_argument("--limit", type=int, default=None, help="Limit items per job for smoke runs.")
    return parser


def iter_jobs() -> list[list[str]]:
    jobs: list[list[str]] = []
    for dataset, splits, linkers, experiments in MATRIX:
        for split in splits:
            for linker in linkers:
                for experiment in experiments:
                    jobs.append([
                        sys.executable,
                        SCRIPT_BY_EXPERIMENT[experiment],
                        "--dataset",
                        dataset,
                        "--split",
                        split,
                        "--linker",
                        linker,
                    ])
    return jobs


def main() -> None:
    args = build_parser().parse_args()
    jobs = iter_jobs()
    for job in jobs:
        if args.limit is not None:
            job.extend(["--limit", str(args.limit)])
        if args.dry_run:
            print(" ".join(job))
        else:
            subprocess.run(job, check=True)


if __name__ == "__main__":
    main()

