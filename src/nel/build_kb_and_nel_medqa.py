#!/usr/bin/env python3
"""
Build a UMLS KB (any release) + run scispaCy NEL on a MedQA file.
Version-agnostic, uniform CandidateGenerator mechanism for all versions.

  2020AA           -> uses official scispaCy index (auto-download, no build)
  2021AA/2022AB/.. -> builds KB from local UMLS META, then links

Examples
--------
  # self-built version
  python build_kb_and_nel_medqa.py --version 2021AA \
      --meta ~/nel/2021AA/META --input ~/nel/dev.jsonl --outdir ~/nel/out

  # official 2020AA baseline (same mechanism, for fair comparison)
  python build_kb_and_nel_medqa.py --version 2020AA \
      --input ~/nel/dev.jsonl --outdir ~/nel/out

Outputs in --outdir:
  umls_<VER>.jsonl            concept KB (self-built versions only)
  linker_<VER>/               tfidf + nmslib index (self-built versions only)
  <inputstem>.mapped.<VER>.jsonl   NEL result (concept = question minus options, option0..3, _mentions)
"""

import argparse
import json
import os
import sys


def build_kb_jsonl(meta_path, kb_jsonl):
    """MRCONSO/MRSTY/MRDEF -> concept jsonl (all sources, ENG, non-suppressed)."""
    from scispacy import umls_utils
    if os.path.exists(kb_jsonl):
        print(f"[build] {kb_jsonl} exists, skipping concept export", flush=True)
        return
    assert os.path.exists(os.path.join(meta_path, "MRCONSO.RRF")), f"MRCONSO.RRF not in {meta_path}"
    print("[build] reading MRCONSO ...", flush=True)
    cd = {}
    umls_utils.read_umls_concepts(meta_path, cd)      # source=None (all), lang=ENG, non_suppressed=True
    print("[build] reading MRSTY ...", flush=True)
    umls_utils.read_umls_types(meta_path, cd)
    print("[build] reading MRDEF ...", flush=True)
    umls_utils.read_umls_definitions(meta_path, cd)
    for c in cd.values():
        c["aliases"] = list(set(c["aliases"]))
        if "canonical_name" not in c:
            al = c["aliases"]
            c["canonical_name"] = al[0]
            del al[0]
        c.pop("is_from_preferred_source", None)
    with open(kb_jsonl, "w", encoding="utf-8") as fo:
        for v in cd.values():
            fo.write(json.dumps(v) + "\n")
    print(f"[build] wrote {kb_jsonl}: {len(cd)} concepts", flush=True)


def _paths(linker_dir):
    """The 4 index files, named as scispaCy expects (version-independent)."""
    return (
        os.path.join(linker_dir, "nmslib_index.bin"),        # 0: ann index
        os.path.join(linker_dir, "tfidf_vectorizer.joblib"), # 1: vectorizer
        os.path.join(linker_dir, "tfidf_vectors_sparse.npz"),# 2: tfidf vectors
        os.path.join(linker_dir, "concept_aliases.json"),    # 3: aliases
    )


def build_index(kb_jsonl, linker_dir, num_threads, m, ef_construction):
    """concept jsonl -> tfidf + nmslib index (inline; no dependence on new scispaCy API)."""
    import numpy, scipy, joblib, nmslib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scispacy.linking_utils import KnowledgeBase

    ann_p, vec_p, vecs_p, alias_p = _paths(linker_dir)
    if all(os.path.exists(p) for p in (ann_p, vec_p, vecs_p, alias_p)):
        print(f"[index] {linker_dir} already built, skipping", flush=True)
        return
    os.makedirs(linker_dir, exist_ok=True)

    M = m or 100                 # defaults == scispaCy official (matches 2020AA)
    EFC = ef_construction or 2000
    NT = num_threads or 60
    kb = KnowledgeBase(kb_jsonl)
    aliases = list(kb.alias_to_cuis.keys())
    print(f"[index] tfidf on {len(aliases)} aliases (M={M} efC={EFC} threads={NT}) ...", flush=True)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=10, dtype=numpy.float32)
    X = vec.fit_transform(aliases)
    joblib.dump(vec, vec_p)
    nz = numpy.array(X.sum(axis=1) != 0).reshape(-1)
    aliases = [a for a, f in zip(aliases, nz) if f]
    X = X[nz]
    json.dump(aliases, open(alias_p, "w"))
    scipy.sparse.save_npz(vecs_p, X.astype(numpy.float32))   # float32: no precision mismatch, disk is fine
    print("[index] building nmslib hnsw index (the long step) ...", flush=True)
    idx = nmslib.init(method="hnsw", space="cosinesimil_sparse", data_type=nmslib.DataType.SPARSE_VECTOR)
    idx.addDataPointBatch(X)
    idx.createIndex({"M": M, "indexThreadQty": NT, "efConstruction": EFC, "post": 0}, print_progress=True)
    idx.saveIndex(ann_p)
    print("[index] done", flush=True)


def load_cg(version, kb_jsonl, linker_dir, k):
    from scispacy.candidate_generation import CandidateGenerator
    from scispacy.linking_utils import KnowledgeBase
    if version == "2020AA":
        return CandidateGenerator()          # default = official umls (2020AA)

    import numpy, scipy, joblib, nmslib
    ann_p, vec_p, vecs_p, alias_p = _paths(linker_dir)
    aliases = json.load(open(alias_p))
    vec = joblib.load(vec_p)
    X = scipy.sparse.load_npz(vecs_p).astype(numpy.float32)
    ann = nmslib.init(method="hnsw", space="cosinesimil_sparse", data_type=nmslib.DataType.SPARSE_VECTOR)
    ann.addDataPointBatch(X)
    ann.loadIndex(ann_p)
    ann.setQueryTimeParams({"efSearch": 200})
    kb = KnowledgeBase(kb_jsonl)
    # the KB kwarg was renamed across scispaCy versions ('kb' vs 'umls'): try both
    try:
        return CandidateGenerator(ann_index=ann, tfidf_vectorizer=vec,
                                  ann_concept_aliases_list=aliases, kb=kb)
    except TypeError:
        return CandidateGenerator(ann_index=ann, tfidf_vectorizer=vec,
                                  ann_concept_aliases_list=aliases, umls=kb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="2020AA | 2021AA | 2022AB | 2023AA")
    ap.add_argument("--input", required=True, help="MedQA jsonl (question + options{A..D})")
    ap.add_argument("--meta", default=None, help="UMLS META dir (required for self-built versions)")
    ap.add_argument("--outdir", default=".", help="output directory")
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--max_per_mention", type=int, default=5)
    # optional low-memory index params (use for ALL versions if used for any)
    ap.add_argument("--num_threads", type=int, default=None)
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--efc", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ver = args.version
    kb_jsonl = os.path.join(args.outdir, f"umls_{ver}.jsonl")
    linker_dir = os.path.join(args.outdir, f"linker_{ver}")
    stem = os.path.splitext(os.path.basename(args.input))[0]
    out_path = os.path.join(args.outdir, f"{stem}.mapped.{ver}.jsonl")

    # ---- build (skip for official 2020AA) ----
    if ver != "2020AA":
        if not args.meta:
            sys.exit("--meta is required for self-built versions")
        build_kb_jsonl(args.meta, kb_jsonl)
        build_index(kb_jsonl, linker_dir, args.num_threads, args.M, args.efc)

    # ---- candidate generator ----
    print(f"[nel] loading candidate generator ({ver}) ...", flush=True)
    cg = load_cg(ver, kb_jsonl, linker_dir, args.k)

    def link_mentions(texts):
        if not texts:
            return []
        batch = cg(texts, k=args.k)
        out = []
        for cands in batch:
            scored = [(c.concept_id, round(float(max(c.similarities)), 4))
                      for c in cands if c.similarities and max(c.similarities) >= args.threshold]
            scored.sort(key=lambda x: x[1], reverse=True)
            out.append(scored[:args.max_per_mention])
        return out

    # ---- load MedQA ----
    import spacy
    from tqdm import tqdm
    nlp = spacy.load("en_core_sci_sm")   # NER only

    data = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"[nel] {len(data)} MedQA items", flush=True)

    OPT = ["A", "B", "C", "D"]
    texts = []
    for d in data:
        texts.append(d["question"])
        for k in OPT:
            texts.append(d["options"][k])

    text_mentions, uniq = [], set()
    for doc in tqdm(nlp.pipe(texts, batch_size=32), total=len(texts), desc="NER"):
        ms = [e.text.strip() for e in doc.ents if e.text.strip()]
        text_mentions.append(ms)
        uniq.update(ms)

    uniq = list(uniq)
    print(f"[nel] linking {len(uniq)} unique mentions ...", flush=True)
    m2c = dict(zip(uniq, link_mentions(uniq)))

    def concepts_of(ms):
        s, detail = set(), []
        for m in ms:
            cuis = m2c.get(m, [])
            if cuis:
                s.update(c for c, _ in cuis)
                detail.append({"text": m, "cuis": cuis})
        return s, detail

    results = []
    for i, d in enumerate(data):
        base = i * 5
        q_set, q_det = concepts_of(text_mentions[base])
        opt_sets, opt_dets = [], []
        for j in range(4):
            cs, dt = concepts_of(text_mentions[base + 1 + j])
            opt_sets.append(cs); opt_dets.append(dt)
        all_opt = set().union(*opt_sets) if opt_sets else set()
        rec = {"_idx": i, "concept": sorted(q_set - all_opt)}
        for j in range(4):
            rec[f"option{j}"] = sorted(opt_sets[j])
        rec["_mentions"] = {"question": q_det, "options": opt_dets}
        results.append(rec)

    with open(out_path, "w", encoding="utf-8") as fo:
        for r in results:
            fo.write(json.dumps(r) + "\n")

    distinct = len({c for r in results for c in
                    (set(r["concept"]) | {x for j in range(4) for x in r[f"option{j}"]})})
    print(f"[done] wrote {out_path}", flush=True)
    print(f"[done] {ver}: items={len(results)} distinct_CUIs={distinct} "
          f"mean_q_concepts={sum(len(r['concept']) for r in results)/len(results):.2f}", flush=True)


if __name__ == "__main__":
    main()
