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
- models: Qwen3.5 `0.8B`, `2B`, `4B`, `9B`
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

- Qwen3.5 `0.8B`, `2B`, `4B`, `9B`
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

- `relation_only_evidence`: UMLS concept names plus relation edges
- `metadata_rich_evidence`: concept names, definitions, semantic types, and
  option concept metadata
- `gold_only_evidence`: evidence for the gold answer only; diagnostic upper
  bound, not realistic at inference time
- `shuffled_answer_evidence`: answer evidence assigned to different option
  labels; control for whether the model follows the injected evidence
- `q_to_a_path_evidence`: metadata plus paths from question CUIs to answer
  option CUIs

output:

- prediction files under `outputs/operational_umls/step2_predictions/`
- summary tables under `outputs/operational_umls/tables/`

what we learn:

We learn which UMLS evidence forms help, hurt, or control the model on items
where the baseline model was wrong. `gold_only_evidence` is interpreted only as
an upper bound. It is not main-claim evidence.

script:

```bash
STEP=2 LIMIT=0 bash scripts/run_operational_umls_experiments.sh
```

# STEP 3

purpose:

Test whether the model can use chains in the UMLS graph, and how performance
changes when the graph path becomes longer or the injected graph becomes wider.

claim:

The usefulness of UMLS graph evidence depends on both chain depth and context
width. More UMLS graph context is not automatically better.

data used:

- filtered manifests from STEP 1
- original QA items
- linked CUIs from MetaMap and scispaCy
- UMLS metadata from `data/UMLS/cui_detail.tsv.zip`
- UMLS relations from `data/UMLS/triplets.csv.zip`
- optional expanded relations from `data/UMLS/triplets_extended.csv.zip` when
  available

conditions:

- rows: path depth from question CUI to correct answer CUI
  - depth 1
  - depth 2
  - depth 3
  - depth 4
- columns: total UMLS graph context injected around the path
  - `tight`: up to 20 nodes
  - `medium`: up to 50 nodes
  - `wide`: up to 100 nodes
  - `full`: up to 200 nodes

The condition names are:

```text
chain_depth_1_width_tight_evidence
chain_depth_1_width_medium_evidence
chain_depth_1_width_wide_evidence
chain_depth_1_width_full_evidence
...
chain_depth_4_width_full_evidence
```

output:

- prediction files under `outputs/operational_umls/step3_predictions/`
- depth x width matrix tables under `outputs/operational_umls/tables/`

what we learn:

We learn whether performance changes as the graph chain gets longer and as more
connected UMLS context nodes are injected. This tests graph use rather than
only metadata use.

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
