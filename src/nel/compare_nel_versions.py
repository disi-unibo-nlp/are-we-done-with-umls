#!/usr/bin/env python3
"""
Compare NEL results (question CUIs vs option CUIs) across UMLS versions.

Reads the mapped files produced by build_kb_and_nel_medqa.py:
    <stem>.mapped.<VERSION>.jsonl
each line:
    {"_idx", "concept":[q CUIs minus option CUIs], "option0..3":[...],
     "_mentions": {"question":[{text,cuis}], "options":[[...]x4]}}

For each version it reports, over the whole file:
  - QUESTION CUIs  : mean per item, total, distinct-across-corpus
  - OPTION   CUIs  : mean per item, total, distinct-across-corpus
  - CONCEPT field  : (question minus options -> what goes into your KG)

Plus corpus-level drift vs the earliest version (Jaccard / added / lost),
and item-level counts of how many items' question-CUI set changed.

Usage
-----
  python compare_nel_versions.py out/dev.mapped.*.jsonl -o compare_dev
  # or point at a dir + stem:
  python compare_nel_versions.py --dir out --stem dev -o compare_dev
"""

import argparse
import csv
import glob
import json
import os
import re
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*", help="mapped jsonl files (e.g. out/dev.mapped.*.jsonl)")
    p.add_argument("--dir", default=".", help="dir to search if no files given")
    p.add_argument("--stem", default="dev", help="input stem, matches <stem>.mapped.*.jsonl")
    p.add_argument("-o", "--outprefix", default="compare_nel")
    return p.parse_args()


VER_RE = re.compile(r"\.mapped\.([^.]+)\.jsonl$")


def question_cuis(rec):
    """True question CUI set: prefer _mentions.question; fall back to 'concept'."""
    ms = rec.get("_mentions", {}).get("question")
    if ms is not None:
        return {cui for m in ms for cui, _ in m.get("cuis", [])}
    return set(rec.get("concept", []))


def option_cuis(rec):
    s = set()
    for j in range(4):
        s.update(rec.get(f"option{j}", []))
    return s


def load(path):
    """_idx -> record (falls back to line number if _idx missing)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r.get("_idx", i)] = r
    return out


def main():
    args = parse_args()
    files = args.files or sorted(glob.glob(os.path.join(args.dir, f"{args.stem}.mapped.*.jsonl")))
    if not files:
        sys.exit("no mapped files found")

    versions = {}  # version -> {idx: record}
    for path in files:
        m = VER_RE.search(os.path.basename(path))
        if not m:
            print(f"[skip] can't parse version from {path}", file=sys.stderr)
            continue
        versions[m.group(1)] = load(path)
    vers = sorted(versions)  # chronological: 2020AA < 2021AA < 2022AB < 2023AA
    print("versions:", ", ".join(vers))

    # precompute per-version, per-item sets
    q_sets = {v: {i: question_cuis(r) for i, r in versions[v].items()} for v in vers}
    o_sets = {v: {i: option_cuis(r) for i, r in versions[v].items()} for v in vers}
    c_sets = {v: {i: set(r.get("concept", [])) for i, r in versions[v].items()} for v in vers}

    # ---------- Table 1: per-version aggregate ----------
    def agg(setmap):
        idxs = list(setmap)
        n = len(idxs)
        total = sum(len(setmap[i]) for i in idxs)
        distinct = len(set().union(*setmap.values())) if idxs else 0
        return n, total, (total / n if n else 0), distinct

    t1_path = f"{args.outprefix}_by_version.csv"
    with open(t1_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["version", "items",
                    "Q_mean", "Q_total", "Q_distinct",
                    "OPT_mean", "OPT_total", "OPT_distinct",
                    "CONCEPT_mean", "CONCEPT_distinct"])
        print("\n=== per-version CUI counts ===")
        print(f"{'ver':8} {'items':>6} {'Q_mean':>8} {'Q_dist':>8} {'OPT_mean':>9} {'OPT_dist':>9} {'CONC_mean':>10} {'CONC_dist':>10}")
        for v in vers:
            n, qt, qm, qd = agg(q_sets[v])
            _, ot, om, od = agg(o_sets[v])
            _, ct, cm, cd = agg(c_sets[v])
            w.writerow([v, n, f"{qm:.2f}", qt, qd, f"{om:.2f}", ot, od, f"{cm:.2f}", cd])
            print(f"{v:8} {n:>6} {qm:>8.2f} {qd:>8} {om:>9.2f} {od:>9} {cm:>10.2f} {cd:>10}")

    # ---------- Table 2: corpus distinct-CUI drift vs baseline ----------
    base = vers[0]
    base_q = set().union(*q_sets[base].values()) if q_sets[base] else set()
    base_o = set().union(*o_sets[base].values()) if o_sets[base] else set()
    t2_path = f"{args.outprefix}_drift_vs_{base}.csv"
    with open(t2_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["version", "scope", "distinct", f"added_vs_{base}", f"lost_vs_{base}", f"jaccard_vs_{base}"])
        print(f"\n=== distinct-CUI drift vs {base} ===")
        print(f"{'ver':8} {'scope':9} {'distinct':>9} {'added':>7} {'lost':>7} {'jaccard':>8}")
        for scope, cur_all, base_all in [("question", q_sets, base_q), ("options", o_sets, base_o)]:
            for v in vers:
                cur = set().union(*cur_all[v].values()) if cur_all[v] else set()
                added = len(cur - base_all)
                lost = len(base_all - cur)
                union = len(cur | base_all)
                jac = (len(cur & base_all) / union) if union else 1.0
                w.writerow([v, scope, len(cur), added, lost, f"{jac:.4f}"])
                print(f"{v:8} {scope:9} {len(cur):>9} {added:>7} {lost:>7} {jac:>8.4f}")

    # ---------- Table 3: item-level changes vs baseline ----------
    t3_path = f"{args.outprefix}_item_changes_vs_{base}.csv"
    with open(t3_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["version", "items_compared",
                    "items_Qset_changed", "pct_Qset_changed",
                    "mean_Q_added", "mean_Q_removed"])
        print(f"\n=== item-level question-CUI change vs {base} ===")
        print(f"{'ver':8} {'n':>6} {'changed':>8} {'pct':>7} {'add/it':>8} {'rem/it':>8}")
        for v in vers:
            common = set(q_sets[base]) & set(q_sets[v])
            changed = added_sum = removed_sum = 0
            for i in common:
                a = q_sets[v][i] - q_sets[base][i]
                r = q_sets[base][i] - q_sets[v][i]
                if a or r:
                    changed += 1
                added_sum += len(a)
                removed_sum += len(r)
            n = len(common)
            pct = 100 * changed / n if n else 0
            w.writerow([v, n, changed, f"{pct:.2f}",
                        f"{added_sum/n:.3f}" if n else 0, f"{removed_sum/n:.3f}" if n else 0])
            print(f"{v:8} {n:>6} {changed:>8} {pct:>6.2f}% {added_sum/n if n else 0:>8.3f} {removed_sum/n if n else 0:>8.3f}")

    print(f"\nwrote:\n  {t1_path}\n  {t2_path}\n  {t3_path}")


if __name__ == "__main__":
    main()
