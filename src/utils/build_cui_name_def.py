"""
Build a CUI | String Name | Semantic Types | Definition table from
MRCONSO.RRF + MRSTY.RRF + MRDEF.RRF.

For each CUI:
  - String Name (STR) = the canonical English preferred name from MRCONSO
                  (ISPREF=Y, TS=P, STT=PF; fallback: any English atom).
  - Semantic Types (STY) = all semantic types from MRSTY, joined by '|'.
                  Also output TUIs (the T-codes) separately.
  - Definition (DEF)  = ONE definition picked from MRDEF using a source-
                  vocabulary preference list. MRDEF has no built-in
                  "primary" flag, so we pick by (a) drop suppressed,
                  (b) prefer high-quality sources (NCI > MSH > MEDLINEPLUS
                  > SNOMEDCT_US > others), (c) longest definition as a
                  tiebreaker.

Usage:
    # Whole UMLS:
    python build_cui_name_def.py \
        --umls-dir ~/Desktop \
        --out cui_name_def.tsv

    # Only the CUIs you care about (much smaller / faster):
    python build_cui_name_def.py \
        --umls-dir ~/Desktop \
        --cuis-file medqa_cuis.txt \
        --out medqa_cui_name_def.tsv

    # Skip CUIs with no definition:
    python build_cui_name_def.py \
        --umls-dir ~/Desktop \
        --out cui_name_def.tsv \
        --require-def

Output is a TSV with columns:
    CUI \\t STR \\t TUI \\t STY \\t DEF_SAB \\t DEF
(use --jsonl to write JSONL instead).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator


# ---------- RRF schema ------------------------------------------------------

MRCONSO_COLS = ("CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI",
                "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR",
                "SRL", "SUPPRESS", "CVF")
MRSTY_COLS = ("CUI", "TUI", "STN", "STY", "ATUI", "CVF")
MRDEF_COLS = ("CUI", "AUI", "ATUI", "SATUI", "SAB", "DEF", "SUPPRESS", "CVF")


# ---------- Source-vocabulary preference for definitions --------------------
#
# Order matters. Lower index = higher priority. Tweak to taste.
DEF_SAB_PRIORITY = [
    "NCI",          # NCI Thesaurus — broad, high quality, especially for
                    # diseases / drugs / anatomy.
    "MSH",          # MeSH — authoritative, concise.
    "MEDLINEPLUS",  # Plain-language, patient-facing.
    "SNOMEDCT_US",  # Clinical SNOMED definitions.
    "HPO",          # Human Phenotype Ontology — good for phenotype CUIs.
    "CSP",          # NLM's Computer Retrieval of Information on Sci. Projects
                    # thesaurus — short but often present.
    "GO",           # Gene Ontology.
    "MTHMST",       # Older Metathesaurus source.
]


def _sab_rank(sab: str) -> int:
    try:
        return DEF_SAB_PRIORITY.index(sab)
    except ValueError:
        return len(DEF_SAB_PRIORITY)  # everything else, lower priority


# ---------- IO --------------------------------------------------------------

def iter_rrf(path: Path, cols: tuple[str, ...]) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").rstrip("|").split("|")
            if len(parts) < len(cols):
                parts += [""] * (len(cols) - len(parts))
            yield dict(zip(cols, parts))


def read_cui_list(path: Path) -> set[str]:
    cuis: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            c = line.strip()
            if c:
                cuis.add(c)
    return cuis


# ---------- MRCONSO: pick one preferred English name per CUI ---------------

def build_name_index(mrconso_path: Path,
                     wanted: set[str] | None = None,
                     keep_suppressed: bool = False) -> dict[str, str]:
    """Return {cui: preferred_name}.

    Strategy:
      1. English atoms only (LAT == 'ENG').
      2. Drop suppressed unless keep_suppressed.
      3. Prefer the canonical preferred atom: ISPREF='Y' AND TS='P' AND STT='PF'.
      4. Fall back to the first English atom we see.
    """
    preferred: dict[str, str] = {}
    fallback: dict[str, str] = {}

    for r in iter_rrf(mrconso_path, MRCONSO_COLS):
        cui = r["CUI"]
        if wanted is not None and cui not in wanted:
            continue
        if r["LAT"] != "ENG":
            continue
        if not keep_suppressed and r["SUPPRESS"] not in ("N", ""):
            continue

        if (r["ISPREF"] == "Y" and r["TS"] == "P" and r["STT"] == "PF"
                and cui not in preferred):
            preferred[cui] = r["STR"]
        elif cui not in fallback:
            fallback[cui] = r["STR"]

    # Merge: preferred wins, fallback fills the gaps.
    for cui, s in fallback.items():
        preferred.setdefault(cui, s)
    return preferred


# ---------- MRSTY: collect ALL semantic types per CUI ----------------------

def build_sty_index(mrsty_path: Path,
                    wanted: set[str] | None = None,
                    ) -> dict[str, list[tuple[str, str]]]:
    """Return {cui: [(tui, sty), ...]} — every semantic type the CUI has.

    MRSTY has no SUPPRESS column. A CUI typically has 1-3 STYs; we keep
    them in MRSTY row order (which is also the order UMLS asserts them).
    """
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    seen_per_cui: dict[str, set[str]] = defaultdict(set)

    for r in iter_rrf(mrsty_path, MRSTY_COLS):
        cui = r["CUI"]
        if wanted is not None and cui not in wanted:
            continue
        tui = r["TUI"]
        # Defensive de-dupe (shouldn't fire in well-formed MRSTY).
        if tui in seen_per_cui[cui]:
            continue
        seen_per_cui[cui].add(tui)
        out[cui].append((tui, r["STY"]))
    return dict(out)


# ---------- MRDEF: pick one definition per CUI ------------------------------

def build_def_index(mrdef_path: Path,
                    wanted: set[str] | None = None,
                    keep_suppressed: bool = False
                    ) -> dict[str, tuple[str, str]]:
    """Return {cui: (sab, def_text)}.

    For each CUI we keep the single best definition based on
    (sab_priority, -len(DEF)) — lower is better, so higher-priority SAB and
    longer definitions win.
    """
    # We need to consider all candidates per CUI, then pick the best.
    candidates: dict[str, list[tuple[int, int, str, str]]] = defaultdict(list)

    for r in iter_rrf(mrdef_path, MRDEF_COLS):
        cui = r["CUI"]
        if wanted is not None and cui not in wanted:
            continue
        if not keep_suppressed and r["SUPPRESS"] not in ("N", ""):
            continue
        sab = r["SAB"]
        text = r["DEF"]
        if not text:
            continue
        # Sort key: (sab_rank, -length) — smaller = better.
        candidates[cui].append((_sab_rank(sab), -len(text), sab, text))

    best: dict[str, tuple[str, str]] = {}
    for cui, rows in candidates.items():
        rows.sort()  # ascending by (rank, -length)
        _, _, sab, text = rows[0]
        best[cui] = (sab, text)
    return best


# ---------- Output ----------------------------------------------------------

def _format_sty(sty_rows: list[tuple[str, str]]) -> tuple[str, str]:
    """Return (tui_joined, sty_joined) using '|' as separator."""
    if not sty_rows:
        return "", ""
    tuis = "|".join(t for t, _ in sty_rows)
    stys = "|".join(s for _, s in sty_rows)
    return tuis, stys


def write_tsv(out_path: Path,
              names: dict[str, str],
              stys:  dict[str, list[tuple[str, str]]],
              defs:  dict[str, tuple[str, str]],
              require_def: bool) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["CUI", "STR", "TUI", "STY", "DEF_SAB", "DEF"])
        for cui in sorted(names):
            sab, dtext = defs.get(cui, ("", ""))
            if require_def and not dtext:
                continue
            tui_s, sty_s = _format_sty(stys.get(cui, []))
            w.writerow([cui, names[cui], tui_s, sty_s, sab, dtext])
            n += 1
    return n


def write_jsonl(out_path: Path,
                names: dict[str, str],
                stys:  dict[str, list[tuple[str, str]]],
                defs:  dict[str, tuple[str, str]],
                require_def: bool) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for cui in sorted(names):
            sab, dtext = defs.get(cui, ("", ""))
            if require_def and not dtext:
                continue
            sty_rows = stys.get(cui, [])
            fh.write(json.dumps({
                "cui": cui,
                "str": names[cui],
                "tui": [t for t, _ in sty_rows],
                "sty": [s for _, s in sty_rows],
                "def_sab": sab,
                "def": dtext,
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------- CLI -------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--umls-dir", type=Path, required=True,
                        help="Folder containing MRCONSO.RRF and MRDEF.RRF.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output file. Format inferred from extension "
                             "(.tsv default, .jsonl if --jsonl).")
    parser.add_argument("--jsonl", action="store_true",
                        help="Write JSONL instead of TSV.")
    parser.add_argument("--cuis-file", type=Path, default=None,
                        help="Optional text file with one CUI per line. "
                             "If given, only these CUIs are processed.")
    parser.add_argument("--require_def", action="store_true",
                        help="Drop CUIs with no definition (smaller output).")
    parser.add_argument("--keep_suppressed", action="store_true",
                        help="Don't filter SUPPRESS != 'N' rows.")
    args = parser.parse_args()

    mrconso = args.umls_dir / "MRCONSO.RRF"
    mrdef = args.umls_dir / "MRDEF.RRF"
    mrsty = args.umls_dir / "MRSTY.RRF"
    for p in (mrconso, mrsty, mrdef):
        if not p.exists():
            raise FileNotFoundError(p)

    wanted = read_cui_list(args.cuis_file) if args.cuis_file else None
    if wanted is not None:
        print(f"Restricting to {len(wanted):,} CUIs from {args.cuis_file}")

    print(f"Scanning {mrconso} for names ...")
    names = build_name_index(mrconso, wanted=wanted,
                             keep_suppressed=args.keep_suppressed)
    print(f"  -> {len(names):,} CUIs with English names")
    
    print(f"Scanning {mrsty} for semantic types ...")
    stys = build_sty_index(mrsty, wanted=wanted)
    print(f"  -> {len(stys):,} CUIs with Semantic Types")

    print(f"Scanning {mrdef} for definitions ...")
    defs = build_def_index(mrdef, wanted=wanted,
                           keep_suppressed=args.keep_suppressed)
    print(f"  -> {len(defs):,} CUIs with at least one definition")

    if args.jsonl:
        n = write_jsonl(args.out, names, stys, defs, args.require_def)
    else:
        n = write_tsv(args.out, names, stys, defs, args.require_def)
    print(f"Wrote {n:,} rows -> {args.out}")


if __name__ == "__main__":
    main()
