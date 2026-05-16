"""
Merge raw MedQA-family JSONL with MetaMap-grounded JSONL.

Pairs files by row order:
    <root>/<split>.jsonl              <-- raw question (MedQA / MedMCQA /
    <root>/<split>.json                   PubMedQA shape, auto-detected)
    <root>/metamap_mapped/<split>.jsonl  <-- {"concept": [...],
                                              "option0": [...], ...}

Writes:
    <root>/metamap_mapped/<split>.text.jsonl

Each output row = the raw record (kept verbatim, all its fields) plus the
MetaMap CUI fields merged in:

    {
      ... all original raw fields ...
      "concept":  [CUIs in question stem],
      "option0":  [CUIs in option A / opa / options[0]],
      "option1":  [...],
      "option2":  [...],
      "option3":  [...]                <-- only if present in the grounded file
    }

Datasets handled:
  * MedQA      "options" = {"A": "...", "B": "...", "C": "...", "D": "..."}
  * MedMCQA    top-level opa / opb / opc / opd  (+ cop)
  * PubMedQA   "options" = ["yes", "no", "maybe"]
               grounded may only have "concept" (no option0..3)

Usage:
    python expand_to_text.py --root data/medQA
    python expand_to_text.py --root data/pubmedQA --splits dev test
    python expand_to_text.py --root data/medMCQA --splits dev test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator


METAMAP_DIR_DEFAULT = "metamap_mapped"
OUT_SUFFIX_DEFAULT = ".text.jsonl"

# Keys we copy across from the grounded record. We do NOT rename them — the
# user asked for the merged file to keep the metamap field names as-is so it
# stays compatible with whatever consumes them downstream.
GROUNDED_KEYS = ("concept", "option0", "option1", "option2", "option3")

# Output column orders per dataset. Detected from the raw record's fields.
ORDER_MEDQA = (
    "question", "options", "answer_idx", "answer",
    "concept", "option0", "option1", "option2", "option3",
)
ORDER_MEDMCQA = (
    "id", "question", "opa", "opb", "opc", "opd", "cop",
    "concept", "option0", "option1", "option2", "option3",
)
# After MedMCQA -> MedQA-style harmonisation, fields are renamed:
ORDER_MEDMCQA_UNIFIED = (
    "id", "question", "options", "answer_idx", "answer",
    "concept", "option0", "option1", "option2", "option3",
)
ORDER_PUBMEDQA = (
    "pmids", "question", "options", "answer",
    "concept",
)

LETTERS = ("A", "B", "C", "D")
MEDMCQA_OPTION_KEYS = ("opa", "opb", "opc", "opd")


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


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _detect_schema(raw: dict) -> str:
    """Return one of 'medqa', 'medmcqa', 'pubmedqa'."""
    if any(k in raw for k in MEDMCQA_OPTION_KEYS):
        return "medmcqa"
    if "pmids" in raw or (
        isinstance(raw.get("options"), list)
        and "answer" in raw
        and "answer_idx" not in raw
    ):
        return "pubmedqa"
    return "medqa"


def _medmcqa_to_medqa_style(raw: dict) -> dict:
    """Turn MedMCQA-style fields into MedQA-style.

    - opa/opb/opc/opd  ->  options = {"A":..., "B":..., "C":..., "D":...}
    - cop (0..3 int)   ->  answer_idx = "A".."D"  and answer = options[idx]
    All other raw fields are carried over unchanged.
    """
    out = dict(raw)

    options = {LETTERS[i]: raw.get(MEDMCQA_OPTION_KEYS[i], "") or ""
               for i in range(4)}
    out["options"] = options
    # Drop the original opa..opd keys so the record stops carrying duplicates.
    for k in MEDMCQA_OPTION_KEYS:
        out.pop(k, None)

    cop = raw.get("cop")
    if isinstance(cop, int) and 0 <= cop < 4:
        letter = LETTERS[cop]
        out["answer_idx"] = letter
        out["answer"] = options[letter]
    out.pop("cop", None)

    return out


def _reorder(d: dict, order: tuple[str, ...]) -> dict:
    """Return a new dict with keys in ``order`` first (when present), then
    any remaining keys in their original insertion order."""
    seen = set()
    out: dict = {}
    for k in order:
        if k in d:
            out[k] = d[k]
            seen.add(k)
    for k, v in d.items():
        if k not in seen:
            out[k] = v
    return out


def merge_record(raw: dict, grounded: dict) -> dict:
    """Return a new dict that is raw + grounded's CUI fields, in a stable
    per-dataset column order. For MedMCQA also harmonises the option/answer
    fields to MedQA style.
    """
    schema = _detect_schema(raw)

    if schema == "medmcqa":
        base = _medmcqa_to_medqa_style(raw)
        order = ORDER_MEDMCQA_UNIFIED
    elif schema == "pubmedqa":
        base = dict(raw)
        order = ORDER_PUBMEDQA
    else:  # medqa
        base = dict(raw)
        order = ORDER_MEDQA

    # Merge the grounded CUI fields. Raw fields win on name collision (they
    # shouldn't collide in practice).
    for key in GROUNDED_KEYS:
        if key in grounded and key not in base:
            base[key] = grounded[key]

    return _reorder(base, order)


def merge_split(raw_path: Path,
                grounded_path: Path,
                out_path: Path) -> tuple[int, int]:
    """Stream raw + grounded in lock-step; write one merged row per question.

    Returns (n_input_questions, n_rows_written).
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
            merged = merge_record(raw, gr)
            out_fh.write(json.dumps(merged, ensure_ascii=False) + "\n")
            n_q += 1
            n_rows += 1
    return n_q, n_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=Path, required=True,
                        help="Dataset root, e.g. data/medQA")
    parser.add_argument("--splits", nargs="+", default=["dev", "train", "test"],
                        help="Splits to process. Default: dev train test.")
    parser.add_argument("--metamap-dir", default=METAMAP_DIR_DEFAULT,
                        help=f"Subdir under --root with metamap files. "
                             f"Default: {METAMAP_DIR_DEFAULT}")
    parser.add_argument("--metamap-suffix", default=".jsonl",
                        help="Filename suffix for metamap files. Default: .jsonl")
    parser.add_argument("--out-dir", default=None,
                        help="Subdir under --root to write outputs. "
                             "Default: same as --metamap-dir")
    parser.add_argument("--out-suffix", default=OUT_SUFFIX_DEFAULT,
                        help=f"Filename suffix for outputs. "
                             f"Default: {OUT_SUFFIX_DEFAULT}")
    args = parser.parse_args()

    root: Path = args.root
    metamap_dir = root / args.metamap_dir
    out_dir = root / (args.out_dir or args.metamap_dir)

    print(f"Root         : {root}")
    print(f"Metamap dir  : {metamap_dir}")
    print(f"Output dir   : {out_dir}")
    print(f"Splits       : {args.splits}\n")

    grand_total = 0
    for split in args.splits:
        raw_path = _first_existing([
            root / f"{split}.jsonl",
            root / f"{split}.json",
        ])
        grounded_path = _first_existing([
            metamap_dir / f"{split}{args.metamap_suffix}",
            metamap_dir / f"{split}.json",
        ])
        out_path = out_dir / f"{split}{args.out_suffix}"

        if raw_path is None:
            print(f"[skip] {split}: missing raw file under {root}")
            continue
        if grounded_path is None:
            print(f"[skip] {split}: missing metamap file under {metamap_dir}")
            continue

        print(f"[{split}]")
        print(f"  raw     : {raw_path}")
        print(f"  metamap : {grounded_path}")
        n_q, n_rows = merge_split(raw_path, grounded_path, out_path)
        print(f"  wrote   : {out_path}  ({n_rows} rows)\n")
        grand_total += n_rows

    print(f"Done. Total rows written across splits: {grand_total}")


if __name__ == "__main__":
    main()
