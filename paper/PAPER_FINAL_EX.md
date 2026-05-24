# FINAL UMLS EXPERIMENTS

This file contains the final operational experiment plan for the paper.
It intentionally uses a simple format so the paper-writing agent can transpose
it into manuscript prose without mixing it with older abandoned experiments.

The static analytical diagnostics remain in the project, but the steps below
are the operational experiments that involve LLM usage.

General scope:

- datasets: `medQA`, `medMCQA`
- split: `test`
- linkers: `metamap`, `spacy`
- models: Qwen3.5 `0.8B`, `2B`, `4B`, `9B`, `27B`, `35B`
- decoding: non-thinking mode, deterministic settings
- output root: `outputs/operational_umls/`

Important decision:

We do not mask answer options anymore. If the answer option CUI is part of the
retrieved UMLS graph, the model can see it. The old answer-obscured experiments
are deprecated and should not be used for the final paper claims.

# STEP 1

purpose:

Identify the questions where the model cannot already answer correctly without
UMLS. This reduces contamination from previous LLM knowledge and gives us the
smaller dataset for the next operational experiments.

claim:

UMLS should be evaluated on questions where the model actually needs external
evidence. If the model already answers without UMLS, that item cannot tell us
whether UMLS was useful.

data used:

- original QA items from `medQA` and `medMCQA`
- linked CUIs from MetaMap and scispaCy
- UMLS metadata from `data/UMLS/cui_detail.tsv.zip`
- Qwen3.5 predictions

models/linkers:

- Qwen3.5 `0.8B`, `2B`, `4B`, `9B`, `27B`, `35B`
- MetaMap and scispaCy

conditions:

- `no_umls`: question and answer options only
- `metadata_rich_evidence`: question/options plus standard UMLS metadata
  names, definitions, semantic types, and available linked concepts

output:

- prediction files under `outputs/operational_umls/step1_predictions/`
- filtered item manifests under `outputs/operational_umls/step1_subsets/`
- each selected item is labeled:
  - `umls_helped`: wrong with `no_umls`, correct with `metadata_rich_evidence`
  - `still_wrong`: wrong with both conditions

what we learn:

We learn which questions are useful for testing UMLS as external evidence. The
following steps run only on these baseline-wrong questions.

script:

```bash
STEP=1 LIMIT=0 bash scripts/run_operational_umls_experiments.sh
```

# STEP 2

purpose:

Test how different UMLS evidence forms change model behavior on the filtered
questions from STEP 1.

claim:

Different ways of serializing UMLS metadata and answer evidence can shape model
performance. UMLS is not one fixed intervention; the evidence card matters.

data used:

- filtered manifests from STEP 1
- original QA items
- linked CUIs from MetaMap and scispaCy
- UMLS metadata from `data/UMLS/cui_detail.tsv.zip`
- UMLS relations from `data/UMLS/triplets.csv.zip`

conditions:

- `relation_only_evidence`: readable UMLS relation edges from the local
  question/answer neighborhood, without definitions or CUIs
- `q_to_a_path_evidence`: readable paths from question CUIs to answer-option
  CUIs, without concept metadata
- `metadata_rich_evidence`: the same question-to-answer path graph, augmented
  with concept names, CUIs, definitions, and semantic types

output:

- prediction files under `outputs/operational_umls/step2_predictions/`
- summary tables under `outputs/operational_umls/tables/`

what we learn:

We learn whether performance changes when the model first sees broad UMLS graph
relations, then focused question-to-answer paths, and finally the same focused
paths enriched with UMLS metadata. This keeps the main paper evidence realistic:
no gold-only or oracle diagnostic condition is used as main-claim evidence.

script:

```bash
STEP=2 LIMIT=0 bash scripts/run_operational_umls_experiments.sh
```

# STEP 3

purpose:

Test what happens when the STEP 2 Q-to-answer path method is constrained to
shorter or longer exact paths from question CUIs to answer-option CUIs.

claim:

The usefulness of focused UMLS path evidence depends on path length. Direct and
short paths should be easier to use than longer exact paths, which are more
likely to pass through generic or weakly relevant UMLS nodes.

data used:

- filtered manifests from STEP 1
- original QA items
- linked CUIs from MetaMap and scispaCy
- UMLS metadata from `data/UMLS/cui_detail.tsv.zip`
- UMLS relations from `data/UMLS/triplets.csv.zip`
- optional expanded relations from `data/UMLS/triplets_extended.csv.zip` when
  available

conditions:

- exact Q-to-answer path depth 1
- exact Q-to-answer path depth 2
- exact Q-to-answer path depth 3
- exact Q-to-answer path depth 4

The condition names are:

```text
q_to_a_exact_depth_1_evidence
q_to_a_exact_depth_2_evidence
q_to_a_exact_depth_3_evidence
q_to_a_exact_depth_4_evidence
```

output:

- prediction files under `outputs/operational_umls/step3_predictions/`
- Q-to-answer depth tables under `outputs/operational_umls/tables/`

what we learn:

We learn whether the model benefits from focused Q-to-answer paths only at
short depths or whether longer exact paths still provide useful decision
evidence. This removes the old width axis and isolates path length.

script:

```bash
STEP=3 LIMIT=0 bash scripts/run_operational_umls_experiments.sh
```

# STATIC ANALYTICAL STATS THAT REMAIN

These experiments do not use an LLM, but they remain important because they
explain whether UMLS evidence can reach the model.

## E1 Evidence Availability

purpose:

Check whether linked CUIs have names, definitions, semantic types, and useful
answer/distractor evidence.

data used:

- QA linker artifacts
- `data/UMLS/cui_detail.tsv.zip`
- `data/UMLS/triplets.csv.zip`

output:

- `exp1_evidence_availability_*.jsonl`
- evidence availability tables

what we learn:

Whether the committed UMLS view contains evidence that could be serialized for
the model.

## E2 Answer Grounding

purpose:

Check whether question stems and answer options enter UMLS space.

data used:

- MetaMap artifacts
- scispaCy artifacts

output:

- `exp2_answer_grounding_*.jsonl`
- grounding tables

what we learn:

Whether UMLS can support answer discrimination at all. If answer options are
not grounded, UMLS cannot help those options.

## E3 Retrieval Faithfulness

purpose:

Check whether retrieved UMLS relations connect question CUIs to answer-option
CUIs.

data used:

- linked QA items
- `data/UMLS/triplets.csv.zip`

output:

- `exp3_retrieval_faithfulness_*.jsonl`
- retrieval faithfulness tables

what we learn:

Whether the retrieved graph is structurally relevant to the QA decision.

# DEPRECATED OLD OPERATIONAL EXPERIMENTS

The following old conditions/scripts should not be used for the final paper
claims:

- `prior_work_obscured_evidence`
- `q_to_a_depth_*_obscured`
- `pooled_width_*_obscured`
- depth-width graph-context sweeps
- `distractor_only_path_evidence`
- `scripts/run_new_conditions_sweep.sh`
- `scripts/run_scale_sweep.sh`
- old RQ2/RQ3/RQ4 operational wrappers

Reason:

Those experiments were built around answer-name masking and a prior leakage
audit. The current final plan explicitly removes that preprocessing step and
focuses on real UMLS evidence visible to the model.

# MAIN COMMANDS

Run all operational experiments:

```bash
LIMIT=0 STEP=all bash scripts/run_operational_umls_experiments.sh
```

Dry-run the orchestration:

```bash
DRY_RUN=1 STEP=all bash scripts/run_operational_umls_experiments.sh
```

Run static diagnostics:

```bash
LIMIT=0 bash scripts/run_main_diagnostics.sh
```
