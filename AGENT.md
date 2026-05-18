# Agent Guide

This repository accompanies the paper "Are We Done with UMLS Injection into
Language Models for Medical Question Answering?" It contains review artifacts,
analysis notebooks, derived data, and utility scripts for studying how UMLS is
used in medical question answering and related biomedical reasoning systems.

Use this file as the operational guide for future coding agents. The
repository is research-oriented and data-sensitive: preserve provenance,
schemas, row order, and artifact intent unless the user explicitly asks for a
change.

## Project Overview

The paper analyzes UMLS use across a three-stage pipeline:

1. Entity linking and coverage.
2. Question-specific subgraph retrieval.
3. Knowledge injection into language models.

The code and data support linker comparison, K-hop subgraph scale analysis,
QA-GNN/GREASELM/DRAGON-style preprocessing audits, and review screening or
annotation workflows.

Most work in this repo is notebook or script driven. There is currently no
project package manifest, pinned environment file, or automated test suite.
Before adding environment assumptions, inspect the relevant script or notebook
and document any new dependency explicitly.

## Repository Layout

- `README.md`: Primary project overview, paper summary, inclusion and exclusion
  criteria, retained-paper table, and citation text.
- `data/`: Committed derived/sample data used by the analyses. This includes
  MedQA, MedMCQA, and PubMedQA JSONL artifacts, linker outputs, and compressed
  UMLS-derived files.
- `data/readme.md`: Schema notes for MetaMap-mapped data, scispaCy-grounded
  data, and UMLS-derived CUI detail or triplet files.
- `outputs/`: Generated analysis outputs that are part of the repository, such
  as `per_question_counts.csv`.
- `src/nel/`: Entity-linking utilities, including scispaCy and MetaMap-related
  mapping or comparison code.
- `src/subgraph/`: Subgraph construction and audit utilities for UMLS,
  QA-GNN-style, and DRAGON-style graph preprocessing.
- `src/utils/`: General helpers for dataset expansion, UMLS loading, CUI
  metadata extraction, and inclusion filtering.
- `src/*.ipynb`: Exploratory or reproduction notebooks for metadata lookup,
  MedQA/USMLE preprocessing, and K-hop/subgraph analyses.
- `SCOPUS_Query` and `UMLS.drawio`: Review/search and conceptual diagram
  artifacts.

## Data And Artifact Notes

- Preserve dataset path casing. Existing directories include `data/medQA`,
  `data/medMCQA`, and `data/pubmedQA`.
- Many JSONL files rely on row-order alignment between raw records and grounded
  records. Do not sort, shuffle, filter, or reindex them unless the task is
  specifically about changing that alignment.
- MetaMap-mapped data uses fields such as `concept`, `option0`, `option1`,
  `option2`, and `option3`.
- scispaCy-grounded per-option data uses fields such as `sent`, `ans`, `qc`,
  and `ac`.
- UMLS CUI identifiers should remain strings, for example `C0004238`. Do not
  coerce them into numeric types.
- Full UMLS release files, such as `MRCONSO.RRF`, `MRREL.RRF`, `MRSTY.RRF`,
  and `MRDEF.RRF`, may be local licensed inputs. Do not assume they are
  committed, and do not download, publish, or vendor them into this repository.
- Avoid committing private PDFs, local UMLS downloads, raw external datasets,
  generated caches, `.DS_Store`, or large regenerated artifacts unless the user
  explicitly requests it.

## Development Guidelines

- Keep edits small and close to the requested workflow. This is a research
  artifact repository, so unrelated cleanup can easily disturb reproducibility.
- Prefer existing script patterns and file schemas over introducing new
  conventions.
- Use structured readers and writers for JSONL, CSV, TSV, pickle, and notebook
  data. Avoid ad hoc parsing when the standard library or existing helpers are
  sufficient.
- Preserve output schema compatibility for downstream notebooks and scripts.
  When changing a CLI, keep existing defaults stable unless the task requires a
  breaking change.
- Keep notebook edits minimal. Do not rewrite whole notebooks, clear outputs,
  or rerun expensive cells unless the user asks for that specific notebook work.
- When adding dependencies, document them in the changed script, a README
  update, or a new environment file if the task asks for reproducible setup.
- Use ASCII in code and docs unless editing an existing section that already
  uses non-ASCII notation for paper titles, names, or mathematical symbols.

## Python And Notebook Conventions

- Python utilities are currently plain scripts, not an installed package.
  Run them from the repository root unless their docstring says otherwise.
- Keep command-line interfaces based on `argparse` for consistency with the
  existing utility scripts.
- Prefer `pathlib.Path` for filesystem paths in new or substantially edited
  Python code.
- Stream JSONL and large UMLS files where practical. Several UMLS resources can
  be too large to load naively.
- Validate paired input files by non-empty line count before merging records
  by row order.
- Keep CUI lists and semantic type fields as strings or lists of strings.
- For notebooks, record assumptions in markdown cells near the code they
  affect, especially local paths such as UMLS release directories.

## Running And Verifying Work

There is no single repo-wide test command. Choose the narrowest verification
that matches the change.

For syntax checks on maintained utility scripts:

```bash
python3 -m py_compile \
  src/utils/expand_to_text.py \
  src/utils/expand_to_per_option.py \
  src/utils/build_cui_name_def.py
```

Before changing or relying on a script CLI, inspect its help output:

```bash
python3 src/utils/expand_to_text.py --help
python3 src/utils/expand_to_per_option.py --help
python3 src/utils/build_cui_name_def.py --help
```

For data transforms:

- Count non-empty JSONL input and output rows.
- Confirm paired raw and grounded files have matching row counts before merging.
- Inspect a small sample of output records for expected keys and CUI list
  shapes.
- Keep generated files out of version control unless they are intended research
  artifacts for the current task.

For notebooks:

- Prefer restarting the kernel and running only the affected section when local
  data and dependencies are available.
- If full notebook execution depends on unavailable licensed UMLS files or
  external raw datasets, state that limitation instead of fabricating results.

## Agent Safety Rules

- Do not overwrite committed data or outputs without explicit user intent.
- Do not remove research artifacts because they look unused. Many files support
  paper claims or reproducibility checks outside the immediate code path.
- Do not normalize dataset names, casing, or field names unless the task is
  specifically about schema migration.
- Do not publish, vendor, or regenerate licensed UMLS resources.
- Do not use destructive Git commands such as `git reset --hard` or
  `git checkout --` unless the user explicitly requests that operation.
- If the worktree is dirty, identify whether changes are yours before editing
  overlapping files. Preserve unrelated user changes.
- If a verification step cannot run because dependencies or licensed data are
  missing, report the concrete blocker and the command that was attempted.

## Common Tasks

### Merge Raw Records With MetaMap CUIs

Use `src/utils/expand_to_text.py` when raw dataset records and MetaMap-mapped
CUI records should be merged one row per question.

Example:

```bash
python3 src/utils/expand_to_text.py --root data/medQA --splits dev test
```

The output keeps the raw fields and appends grounded CUI fields such as
`concept` and `option0` through `option3`.

### Expand Questions To Per-Option Rows

Use `src/utils/expand_to_per_option.py` when a question-level file should be
expanded so each answer option becomes one row with `sent`, `ans`, `qc`, and
`ac`.

Example:

```bash
python3 src/utils/expand_to_per_option.py \
  --root data/medQA \
  --grounded-dir metamap_mapped \
  --splits test
```

Verify expected row multiplication: MedQA and MedMCQA usually produce four rows
per question, while PubMedQA usually produces three.

### Build CUI Name, Type, And Definition Tables

Use `src/utils/build_cui_name_def.py` with a local UMLS release directory. The
script expects files such as `MRCONSO.RRF`, `MRSTY.RRF`, and `MRDEF.RRF`.

Example for a restricted CUI list:

```bash
python3 src/utils/build_cui_name_def.py \
  --umls-dir /path/to/UMLS-release \
  --cuis-file data/UMLS/QA_cuis.txt \
  --out outputs/qa_cui_name_def.tsv
```

Do not commit full UMLS source tables or generated large metadata files unless
the user explicitly asks for that artifact.

### Work With Linker Or Subgraph Analyses

- Start from `src/nel/` for scispaCy, MetaMap, and linker comparison workflows.
- Start from `src/subgraph/` for QA-GNN, DRAGON, and K-hop graph construction
  or pruning analysis.
- Cross-check notebook assumptions against `README.md` and `data/readme.md`
  before changing claims or outputs tied to the paper.
