"""
Build per-option JSONL files for MedQA from a raw split + a grounded (MetaMap)
split. Four rows per question (one per option), matching the dev.spacy.jsonl
shape.

Expected directory layout (paths can be overridden by flags):

    <root>/
        dev.jsonl         train.jsonl         test.jsonl          (raw MedQA)
        grounded/
            dev.grounded.jsonl
            train.grounded.jsonl
            test.grounded.jsonl

Raw record (one per question):
    {
      "question": "...",
      "answer":   "Ceftriaxone",
      "options":  {"A": "...", "B": "...", "C": "...", "D": "..."},
      ...
    }

Grounded record (one per question, same row order as the raw file):
    {
      "concept":  [CUIs in question stem],
      "option0":  [CUIs in option A],
      "option1":  [CUIs in option B],
      "option2":  [CUIs in option C],
      "option3":  [CUIs in option D]
    }

Output (4 rows per question):
    {
      "sent": "<question stem> <option text>",
      "ans":  "<option text>",
      "qc":   [CUIs in question stem],
      "ac":   [CUIs for this option]
    }

Usage:
    python expand_to_per_option.py --root /path/to/medQA
    python expand_to_per_option.py --root /path/to/medQA --splits dev
    python expand_to_per_option.py --root /path/to/medQA \
        --grounded-dir grounded --out-dir grounded
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator


OPTION_LETTERS = ("A", "B", "C", "D")
METAMAP_OPTION_KEYS = ("option0", "option1", "option2", "option3")
MEDMCQA_OPTION_KEYS = ("opa", "opb", "opc", "opd")  # MedMCQA top-level keys


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {exc}") from exc


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


# ---------------------------------------------------------------------------
# Field extraction (tolerant to a few variants)
# ---------------------------------------------------------------------------

def _question_stem(raw: dict) -> str:
    return (raw.get("question") or raw.get("sent") or "").rstrip()


def _options_in_order(raw: dict) -> list[str]:
    """Return the option texts in fixed order.

    Length depends on the dataset:
      - MedQA / MedMCQA: 4 options
      - PubMedQA:        3 options (yes / no / maybe)

    Supported schemas:
      - MedQA:    options = {"A": "...", "B": "...", "C": "...", "D": "..."}
      - PubMedQA: options as a list of strings (variable length, typically 3)
      - MedMCQA:  top-level opa / opb / opc / opd  (no nested "options")
      - generic:  options as a list of N strings
    """
    opts = raw.get("options")
    if isinstance(opts, dict):
        return [opts.get(letter, "") for letter in OPTION_LETTERS]
    if isinstance(opts, list):
        # Honour whatever length is given (3 for PubMedQA, 4 for MedQA, ...).
        return [str(o) for o in opts]
    # MedMCQA: opa/opb/opc/opd at the top level
    if any(k in raw for k in MEDMCQA_OPTION_KEYS):
        return [raw.get(k, "") or "" for k in MEDMCQA_OPTION_KEYS]
    return ["", "", "", ""]


def _q_concepts(grounded: dict) -> list[str]:
    # "concept" is the original key; accept "q_concepts" as a fallback.
    return list(grounded.get("concept") or grounded.get("q_concepts") or [])


def _option_concepts(grounded: dict, n_options: int) -> list[list[str]]:
    """Return per-option CUI lists, of length ``n_options``.

    If the grounded record has no option-level CUIs (e.g. PubMedQA), every
    list is empty.
    """
    if isinstance(grounded.get("option_concepts"), list):
        oc = grounded["option_concepts"]
        return [list(oc[i] or []) if i < len(oc) else [] for i in range(n_options)]
    # MedQA-style option0..option3 (works for n_options <= 4).
    if any(k in grounded for k in METAMAP_OPTION_KEYS):
        return [list(grounded.get(METAMAP_OPTION_KEYS[i], []) or [])
                if i < len(METAMAP_OPTION_KEYS) else []
                for i in range(n_options)]
    # No option-level grounding (e.g. PubMedQA).
    return [[] for _ in range(n_options)]


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------

def expand_split(raw_path: Path,
                 grounded_path: Path,
                 out_path: Path) -> tuple[int, int]:
    """Stream raw + grounded in lock-step; write N rows per question.

    N = number of options in the raw record (3 for PubMedQA, 4 for MedQA /
    MedMCQA).

    Returns (n_questions, n_rows_written).
    """
    n_raw = _count_lines(raw_path)
    n_gr = _count_lines(grounded_path)
    if n_raw != n_gr:
        raise ValueError(
            f"Line-count mismatch:\n"
            f"  {raw_path}  has {n_raw} rows\n"
            f"  {grounded_path}  has {n_gr} rows\n"
            f"They must align by row order."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_q = 0
    n_rows = 0
    with out_path.open("w", encoding="utf-8") as out_fh:
        for raw, gr in zip(_iter_jsonl(raw_path), _iter_jsonl(grounded_path)):
            stem = _question_stem(raw)
            options = _options_in_order(raw)
            qc = _q_concepts(gr)
            ac_per_option = _option_concepts(gr, len(options))

            for opt_text, ac in zip(options, ac_per_option):
                sent = f"{stem} {opt_text}".strip() if opt_text else stem
                row = {
                    "sent": sent,
                    "ans":  opt_text,
                    "qc":   qc,
                    "ac":   list(ac),
                }
                out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
            n_q += 1
    return n_q, n_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root", type=Path, required=True,
        help="Root folder, e.g. /path/to/medQA. Raw <split>.jsonl files "
             "live directly under here.",
    )
    parser.add_argument(
        "--grounded-dir", default="grounded",
        help="Subdirectory under --root that holds <split>.grounded.jsonl. "
             "Default: grounded",
    )
    parser.add_argument(
        "--grounded-suffix", default=".jsonl",
        help="Filename suffix for grounded files. Default: .grounded.jsonl",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["test"],
        help="Splits to process. Default: dev train test.",
    )
    parser.add_argument(
        "--out-dir", default="spacy_grounded",
        help="Subdirectory under --root to write outputs into. "
             "Default: grounded  (so files land next to the grounded files).",
    )
    parser.add_argument(
        "--out-suffix", default=".grounded.jsonl",
        help="Filename suffix for outputs. Default: .per_option.jsonl",
    )
    args = parser.parse_args()

    root = args.root
    grounded_dir = root / args.grounded_dir
    out_dir = root / args.out_dir

    print(f"Root         : {root}")
    print(f"Grounded dir : {grounded_dir}")
    print(f"Output dir   : {out_dir}")
    print(f"Splits       : {args.splits}\n")

    grand_total = 0
    for split in args.splits:
        # Try a few common extensions for the raw file (.jsonl preferred).
        raw_candidates = [root / f"{split}.jsonl", root / f"{split}.json"]
        raw_path = next((p for p in raw_candidates if p.exists()), None)

        grounded_candidates = [
            grounded_dir / f"{split}{args.grounded_suffix}",
            grounded_dir / f"{split}.grounded.json",
        ]
        grounded_path = next((p for p in grounded_candidates if p.exists()), None)

        out_path = out_dir / f"{split}{args.out_suffix}"

        if raw_path is None:
            tried = ", ".join(str(p) for p in raw_candidates)
            print(f"[skip] {split}: missing raw file (tried {tried})")
            continue
        if grounded_path is None:
            tried = ", ".join(str(p) for p in grounded_candidates)
            print(f"[skip] {split}: missing grounded file (tried {tried})")
            continue

        print(f"[{split}]")
        print(f"  raw      : {raw_path}")
        print(f"  grounded : {grounded_path}")
        n_q, n_rows = expand_split(raw_path, grounded_path, out_path)
        print(f"  wrote    : {out_path}  ({n_q} questions -> {n_rows} rows)\n")
        grand_total += n_rows

    print(f"Done. Total rows written across splits: {grand_total}")


if __name__ == "__main__":
    main()