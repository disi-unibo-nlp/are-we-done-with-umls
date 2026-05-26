# FINAL OPERATIONAL UMLS EXPERIMENTS

This file contains the operational LLM experiments in addition to the static
analytical stats over datasets and UMLS entities.

From now on, assume operational experiments run over the Qwen3.5 parameter
sweep and the two linking algorithms: MetaMap and scispaCy.

Important change:

We should not omit the answer option if it is in the graph by masking it.
The old answer-obscured preprocessing step is removed from active experiments.

# STEP 1

Identify the instances to work on.

purpose:

Avoid possible knowledge contamination and previous LLM knowledge.

run:

- LLM baseline: QA with only question and answer options
- LLM QA + UMLS: QA with standard UMLS metadata

conditions:

- `no_umls`
- `metadata_rich_evidence`

selection:

For each model, dataset, and linker, select the questions where the model was
not able to answer correctly without UMLS. Then label whether the same item
becomes correct after showing UMLS or still remains wrong.

output:

A smaller dataset for each model/dataset/linker slice. This is the dataset used
by the next steps.

# STEP 2

Test performance by changing UMLS metadata and evidence form.

This is done only on the filtered dataset instances from STEP 1.

conditions:

- `relation_only_evidence`
- `metadata_rich_evidence`
- `gold_only_evidence`
- `shuffled_answer_evidence`
- `q_to_a_path_evidence`

claim:

Different usage of UMLS metadata and richness can shape model behavior and
performance.

# STEP 3

Test whether the model can exploit chains in the UMLS graph.

We perturb the graph to define different chain lengths and graph widths.

We expand the UMLS graph when possible so that the path between question nodes
and the correct answer node can be studied at different lengths.

Example:

```text
question_node -> n_1 -> n_2 -> answer_node
```

We analyze how the model uses this knowledge and whether additional connected
context nodes help.

matrix:

- rows: number of steps from question node to the correct answer option
- columns: number of total UMLS nodes injected around the chain

conditions:

```text
chain_depth_1_width_tight_evidence
chain_depth_1_width_medium_evidence
chain_depth_1_width_wide_evidence
chain_depth_1_width_full_evidence
...
chain_depth_4_width_full_evidence
```

This step also runs only on the filtered questions from STEP 1.

# STATIC ANALYTICAL STATS THAT REMAIN

These remain active but are not the operational LLM experiments:

- E1 evidence availability
- E2 answer grounding
- E3 retrieval faithfulness

They explain UMLS coverage and retrieval bottlenecks.

# DEPRECATED

Do not use these old operational settings for final claims:

- answer-obscured preprocessing
- `prior_work_obscured_evidence`
- `q_to_a_depth_*_obscured`
- `pooled_width_*_obscured`
- `distractor_only_path_evidence`
- old `run_new_conditions_sweep.sh`
- old `run_scale_sweep.sh`
