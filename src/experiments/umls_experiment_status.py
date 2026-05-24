from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "outputs/operational_umls"))
STEP1_PREDICTIONS_DIR = OUTPUT_ROOT / "step1_predictions" / "qwen"
STEP2_PREDICTIONS_DIR = OUTPUT_ROOT / "step2_predictions" / "qwen"
STEP3_PREDICTIONS_DIR = OUTPUT_ROOT / "step3_predictions" / "qwen"
SUBSETS_DIR = OUTPUT_ROOT / "step1_subsets"
TABLES_DIR = OUTPUT_ROOT / "tables"

SCALE_MODELS = (
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
)
DATASETS = ("medQA", "medMCQA")
LINKERS = tuple(os.environ.get("LINKERS", "metamap spacy").split())
STEP23_LINKERS = tuple(os.environ.get("STEP23_LINKERS", "spacy").split())
SPLIT = "test"
MODE = "non_thinking"

STEP1_CONDITIONS = ("no_umls", "metadata_rich_evidence")
STEP2_CONDITIONS = (
    "relation_only_evidence",
    "q_to_a_path_evidence",
    "metadata_rich_evidence",
)
STEP3_CONDITIONS = tuple(
    f"q_to_a_exact_depth_{depth}_evidence"
    for depth in (1, 2, 3, 4)
)


def qwen_slug(model: str) -> str:
    return model.replace("/", "__").replace(".", "_")


def expected_cells(linkers: tuple[str, ...] = LINKERS) -> list[dict[str, str]]:
    return [
        {
            "dataset": dataset,
            "split": SPLIT,
            "linker": linker,
            "model": model,
            "mode": MODE,
        }
        for dataset in DATASETS
        for linker in linkers
        for model in SCALE_MODELS
    ]


def prediction_path(root: Path, cell: dict[str, str]) -> Path:
    stem = "_".join([
        cell["dataset"],
        cell["split"],
        cell["linker"],
        qwen_slug(cell["model"]),
        cell["mode"],
        "qwen_probe",
    ])
    return root / f"{stem}.jsonl"


def manifest_path(cell: dict[str, str]) -> Path:
    stem = "_".join([
        cell["dataset"],
        cell["split"],
        cell["linker"],
        qwen_slug(cell["model"]),
        cell["mode"],
        "baseline_wrong",
    ])
    return SUBSETS_DIR / f"{stem}.jsonl"


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            if limit is not None and index >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def audit_prediction_dir(
    root: Path,
    expected_conditions: tuple[str, ...],
    linkers: tuple[str, ...] = LINKERS,
) -> tuple[list[dict[str, Any]], list[str]]:
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    for cell in expected_cells(linkers):
        path = prediction_path(root, cell)
        entry: dict[str, Any] = {**cell, "path": str(path), "exists": path.is_file(), "nonempty": False}
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
            entries.append(entry)
            continue
        rows = read_jsonl(path)
        present_conditions = sorted({str(row.get("condition", row.get("prompt_condition", ""))) for row in rows})
        missing_conditions = [condition for condition in expected_conditions if condition not in present_conditions]
        malformed = sum(1 for row in rows if row.get("prediction") is None or row.get("error") not in {None, ""})
        entry.update({
            "nonempty": True,
            "rows": len(rows),
            "conditions": present_conditions,
            "missing_conditions": missing_conditions,
            "malformed_rows": malformed,
            "malformed_rate": malformed / len(rows) if rows else 0.0,
        })
        if missing_conditions:
            missing.append(f"{path} missing conditions: {', '.join(missing_conditions)}")
        entries.append(entry)
    return entries, missing


def audit_subsets(linkers: tuple[str, ...] = LINKERS) -> tuple[list[dict[str, Any]], list[str]]:
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    for cell in expected_cells(linkers):
        path = manifest_path(cell)
        entry: dict[str, Any] = {**cell, "path": str(path), "exists": path.is_file(), "nonempty": False}
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
            entries.append(entry)
            continue
        rows = read_jsonl(path)
        labels = {str(row.get("subset_label", "")) for row in rows}
        entry.update({
            "nonempty": True,
            "rows": len(rows),
            "labels": sorted(labels),
            "umls_helped": sum(1 for row in rows if row.get("subset_label") == "umls_helped"),
            "still_wrong": sum(1 for row in rows if row.get("subset_label") == "still_wrong"),
        })
        entries.append(entry)
    return entries, missing


def audit_summary_tables() -> tuple[list[dict[str, Any]], list[str]]:
    expected = [
        SUBSETS_DIR / "step1_subset_summary.csv",
        SUBSETS_DIR / "step1_subset_summary.md",
    ]
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    for path in expected:
        entry = {"path": str(path), "exists": path.is_file(), "nonempty": path.is_file() and path.stat().st_size > 0}
        if not entry["nonempty"]:
            missing.append(str(path))
        entries.append(entry)
    return entries, missing


def recovery_commands(status: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    if status["step1_predictions"]["missing"]:
        commands.append("STEP=1 LIMIT=0 bash scripts/run_operational_umls_experiments.sh")
    elif status["step1_subsets"]["missing"]:
        commands.append("STEP=1 LIMIT=0 bash scripts/run_operational_umls_experiments.sh")
    elif status["step2_predictions"]["missing"]:
        commands.append("STEP=2 LIMIT=0 bash scripts/run_operational_umls_experiments.sh")
    elif status["step3_predictions"]["missing"]:
        commands.append("STEP=3 LIMIT=0 bash scripts/run_operational_umls_experiments.sh")
    elif status["summary_tables"]["missing"]:
        commands.append("STEP=tables LIMIT=0 bash scripts/run_operational_umls_experiments.sh")
    return commands


def read_step1_summary() -> list[dict[str, str]]:
    path = SUBSETS_DIR / "step1_subset_summary.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def build_status() -> dict[str, Any]:
    step1_predictions, missing_step1_predictions = audit_prediction_dir(STEP1_PREDICTIONS_DIR, STEP1_CONDITIONS)
    subsets, missing_subsets = audit_subsets()
    step2_predictions, missing_step2_predictions = audit_prediction_dir(
        STEP2_PREDICTIONS_DIR,
        STEP2_CONDITIONS,
        STEP23_LINKERS,
    )
    step3_predictions, missing_step3_predictions = audit_prediction_dir(
        STEP3_PREDICTIONS_DIR,
        STEP3_CONDITIONS,
        STEP23_LINKERS,
    )
    summary_tables, missing_summary_tables = audit_summary_tables()
    status: dict[str, Any] = {
        "workflow": "operational_umls",
        "root": str(OUTPUT_ROOT),
        "expected_cells": len(expected_cells()),
        "expected_step23_cells": len(expected_cells(STEP23_LINKERS)),
        "step1_linkers": list(LINKERS),
        "step23_linkers": list(STEP23_LINKERS),
        "step1_predictions": {
            "complete": not missing_step1_predictions,
            "missing": missing_step1_predictions,
            "files": step1_predictions,
        },
        "step1_subsets": {
            "complete": not missing_subsets,
            "missing": missing_subsets,
            "files": subsets,
            "summary": read_step1_summary(),
        },
        "step2_predictions": {
            "complete": not missing_step2_predictions,
            "missing": missing_step2_predictions,
            "files": step2_predictions,
        },
        "step3_predictions": {
            "complete": not missing_step3_predictions,
            "missing": missing_step3_predictions,
            "files": step3_predictions,
        },
        "summary_tables": {
            "complete": not missing_summary_tables,
            "missing": missing_summary_tables,
            "files": summary_tables,
        },
    }
    status["commands"] = recovery_commands(status)
    status["overall"] = (
        "complete"
        if status["step1_predictions"]["complete"]
        and status["step1_subsets"]["complete"]
        and status["step2_predictions"]["complete"]
        and status["step3_predictions"]["complete"]
        and status["summary_tables"]["complete"]
        else "incomplete"
    )
    return status


def write_markdown(path: Path, status: dict[str, Any]) -> None:
    lines = [
        "# Operational UMLS Experiment Status",
        "",
        f"- Overall: `{status['overall']}`",
        f"- Expected STEP 1 cells: `{status['expected_cells']}`",
        f"- Expected STEP 2/3 cells: `{status['expected_step23_cells']}`",
        f"- STEP 1 linkers: `{', '.join(status['step1_linkers'])}`",
        f"- STEP 2/3 linkers: `{', '.join(status['step23_linkers'])}`",
        f"- STEP 1 predictions complete: `{status['step1_predictions']['complete']}`",
        f"- STEP 1 subsets complete: `{status['step1_subsets']['complete']}`",
        f"- STEP 2 predictions complete: `{status['step2_predictions']['complete']}`",
        f"- STEP 3 predictions complete: `{status['step3_predictions']['complete']}`",
        "",
        "## Missing",
    ]
    missing: list[str] = []
    for section in ("step1_predictions", "step1_subsets", "step2_predictions", "step3_predictions", "summary_tables"):
        missing.extend(status[section]["missing"])
    lines.extend(f"- {item}" for item in missing) if missing else lines.append("- none")
    lines.extend(["", "## Recovery Commands"])
    lines.extend(f"- `{command}`" for command in status["commands"]) if status["commands"] else lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit operational UMLS experiment artifacts.")
    parser.add_argument("--output-json", type=Path, default=OUTPUT_ROOT / "status.json")
    parser.add_argument("--output-md", type=Path, default=OUTPUT_ROOT / "missing_runs.md")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--commands", action="store_true")
    args = parser.parse_args()

    status = build_status()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output_md, status)
    if args.commands:
        for command in status["commands"]:
            print(command)
    elif args.print_json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print(f"overall={status['overall']}")
        print(f"step1_predictions_complete={status['step1_predictions']['complete']}")
        print(f"step1_subsets_complete={status['step1_subsets']['complete']}")
        print(f"step2_predictions_complete={status['step2_predictions']['complete']}")
        print(f"step3_predictions_complete={status['step3_predictions']['complete']}")
        if status["commands"]:
            print("next_commands:")
            for command in status["commands"]:
                print(f"- {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
