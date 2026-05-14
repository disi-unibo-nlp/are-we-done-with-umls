# Are We Done with UMLS Injection into Language Models for Medical Question Answering?

This repository accompanies the paper **"Are We Done with UMLS Injection into Language Models for Medical Question Answering?"** It contains the code, data, and per-paper annotations supporting our systematic review of how the Unified Medical Language System (UMLS) is operationalized in medical question answering (MQA) and related reasoning tasks.

---

## TL;DR

We systematically review 25 papers that integrate UMLS into language models for MQA and related reasoning settings. We find that the pipeline used to consume UMLS — *linking → subgraph retrieval → knowledge injection* — varies substantially across systems in ways that are seldom reported, making published benchmark comparisons difficult to interpret. We propose a reporting checklist and design recommendations to improve reproducibility.

---

## Repository Contents

- **`src/`** — Scripts for reproducing the linker comparison (scispaCy vs. MetaMap), $K$-hop subgraph scale measurements, and pruning audit on DRAGON preprocessing artifacts.
- **`outputs/`** — Linker coverage statistics on MedQA, $K$-hop scale measurements on PubMedQA, and pre/post-pruning statistics on MedQA.

---

## Review Methodology

### Search

We followed a PRISMA-style systematic review based on Scopus, targeting open-access English-language conference and journal papers published between 2020 and 2025 that use knowledge graphs and structured representations for NLP-based question answering and reasoning.

### Inclusion Criteria

A paper was retained if:

1. **UMLS plays a substantive methodological role** — not merely cited as background or used only for vocabulary normalization.
2. **The task is MQA or a closely related reasoning setting** — including biomedical textual entailment, knowledge-augmented clinical NER (when used to test UMLS-injected knowledge), and retrieval-augmented medical QA.
3. **The pipeline is described in enough detail for stage-level comparison** — or, where the paper is structurally important to a specific argument in the review, sufficient public information allows us to characterize its UMLS use.

### Adjacent Settings (Cited but Not Reviewed in Depth)

Several lines of UMLS-related work fall outside the analytical frame of this review but inform its motivation:

- **UMLS-based benchmark generation** (e.g., Adjei-Frempah et al., 2024)
- **Hierarchy-oriented knowledge fusion** (e.g., Lu et al., 2023)
- **Dataset construction via CUI normalization** (e.g., Wu et al., 2021)
  
We cite individual works from these settings in the paper where their evidence directly informs our analysis.

### Annotation Process

Two medical NLP researchers among the authors independently labeled the screened papers, reaching **90.2% agreement**; remaining disagreements were resolved through discussion.

---

## Three-Stage Analytical Scheme

The paper decomposes UMLS use in MQA into three pipeline stages:

| Stage | What it does | Where variation enters |
|---|---|---|
| **1. Entity Linking & Coverage** | Grounds input spans to UMLS concepts | Linker choice, span coverage on question stems vs. answer options |
| **2. Subgraph Retrieval** | Selects a question-specific neighborhood of the graph | Expansion depth ($K$), pruning budget, connectivity criterion, fallback behavior |
| **3. Knowledge Injection** | Determines how the selected knowledge is exposed to the model | Linearization, structural injection, or parameter encoding |

Each stage attenuates evidence asymmetrically across the input, and the cumulative effect is rarely reported.

---

## Key Findings

- **Only 6 of 25 reviewed papers** supply enough pipeline detail to enable partial reproduction.
- **Linker choice** is the most consistently reported pipeline detail; scispaCy dominates ($n=9$).
- **Answer-option spans** fail to link to any UMLS concept in roughly **60%** of MedQA instances under the QA-GNN/GREASELM preprocessing pipeline, against ~4% for question stems.
- **The $K=2$ hop default** is empirically workable but only because of aggressive downstream pruning to a 200-node budget; deeper expansion grows subgraphs by multiple orders of magnitude per hop.
- After pruning, **only ~1.5 answer-side concepts on average** survive per question–option pair in DRAGON's released MedQA preprocessing, despite the multiple-choice format requiring discrimination among four options.
- **QA-GNN, GREASELM, and DRAGON** share the same pruning structure but use different scoring functions (RoBERTa-large MLM loss in QA-GNN/GREASELM; GloVe-50 dot product in DRAGON), a divergence not acknowledged in cross-system comparisons.

---

## Repository Link

> Anonymized review repository: <https://anonymous.4open.science/r/are-we-done-with-umls>

---

# Included Papers

The 25 papers retained for full analysis are listed below.

| # | Filename | UMLS Role | MQA/Reasoning | NEL Used | Subgraph Built | Key Points |
|---|---|---|---|---|---|---|
| 1 | Yasunaga et al. (2022) — DRAGON: Deep Bidirectional Language-Knowledge Graph Pretraining | Biomedical KG during pretraining with PubMed text; retrieves local UMLS subgraphs aligned to text segments. | MedQA-USMLE, PubMedQA, BioASQ | Yes | Yes | Deeply bidirectional text-KG pretraining; 2-hop expansion + entity linking; masked LM + KG link prediction joint objectives. Strong fit. |
| 2 | Choudhury et al. (2023) — MediSage | UMLS indirectly involved through vocabulary mapping; system is mainly OMOP-KB/SNOMED-based. | Clinical reasoning, not MQA benchmark | Yes (fallback NEL) | No | Conversational clinical decision-support over EHR + clinical KG; BioLinkBERT NEL fallback. Not centered on UMLS as core source. |
| 3 | Ding et al. (2023) — KnowledgeDA | UMLS as English healthcare KG for domain knowledge localization and augmentation around medical QA pairs. | PubMedQA, CMedQA | Yes | Yes | Mention-to-entity similarity matching; 2-hop relation-consistent replacements; data augmentation around UMLS-grounded entities. |
| 4 | Katta et al. (2024) — KG-Infused LLM for Virtual Health Assistant | Builds large Neo4j KG from UMLS subsets (MSH + NCI); retrieves definitions and relations as LLM context. | BioASQ, MedicationQA, ExpertQA medical subsets | NER + CUI/string lookup | No | RAG-style pipeline with semantic matching + learning-to-rank; not question-specific subgraph construction. |
| 5 | Laube & Eliasmith (2024) — QAVSA | UMLS as external biomedical KG for MedQA-USMLE, with vector symbolic algebra instead of GNN. | MedQA-USMLE | Yes (inherited preprocessing) | Yes | Follows QA-GNN-style graph setup; replaces GNN with VSA + MLP reasoning. |
| 6 | Park et al. (2023) — Relation-Aware Language-Graph Transformer | Inherits UMLS-based MedQA KG; reasons via relation-centric meta-path tokens. | MedQA-USMLE + commonsense QA | Yes | Yes | Per-QA subgraph retrieval; meta-path relation encoding. Q→A connectivity. |
| 7 | Yasunaga et al. (2021) — QA-GNN | Self-constructed biomedical KG: Disease Database portion of UMLS + DrugBank. | MedQA-USMLE | Yes | Yes | Question–choice-specific subgraph + LM-based relevance scoring + joint LM+KG reasoning. Canonical benchmark paper. |
| 8 | Sengupta et al. (2024) — Towards Efficient Methods using KG Embeddings | UMLS definitions + pre-trained UMLS KGE fused into BERT/RoBERTa. | COVID-QA, PubMedQA | Yes (MetaMap) | No | Embedding/definition infusion; not subgraph-construction. |
| 9 | Tian et al. (2024) — Graph Neural Prompting (GNP) | UMLS as biomedical-domain KG encoded into graph neural prompts for LLM prompting. | PubMedQA, BioASQ | Yes | Yes | NEL → 2-hop subgraph → GNN + cross-modality pooling + domain projector. |
| 10 | Vladika et al. (2024) — Adapter Modules for Knowledge Enhancement | UMLS as biomedical KG for adapter-based knowledge injection; UMLS20 subset with partition-specific adapters. | BioASQ-7b, PubMedQA, MedNLI, HoC | No | No (METIS partitions) | Top-20 UMLS relations + METIS partitioning; entity-prediction adapter training. |
| 11 | Yadav et al. (2020) — Medical Knowledge-Enriched Textual Entailment | UMLS + SNOMED-CT + ICD10 used to enrich vocabulary graph for entailment. | MEDIQA-RQE entailment | Not explicit | Yes | Knowledge-enriched graph + BERT via GCN + attention fusion. |
| 12 | Yang et al. (2024) — KG-Rank | UMLS as primary medical KG; one-hop triples retrieved and re-ranked before LLM generation. | LiveQA, ExpertQA-Med/Bio, MedicationQA | Yes (Medical NER prompt) | No (one-hop only) | Similarity ranking + MMR + cross-encoder re-ranking. |
| 13 | Zhang et al. (2022) — DRLK | Pre-computed medical KG: DDB (UMLS-derived) + DrugBank. | MedQA-USMLE, MedMCQA | Yes (inherited) | Yes | QA-GNN/GREASELM-style preprocessing; dynamic hierarchical LM-KG interaction. |
| 14 | Zhang et al. (2022) — GREASELM | Self-constructed medical KG: DDB portion of UMLS + DrugBank. | MedQA-USMLE | Yes | Yes | NEL + 2-hop bridge entities + top-200 node retrieval; LM-KG cross-modal fusion across layers. |
| 15 | Zhao et al. (2024) — Heterogeneous Knowledge Grounding | UMLS mentioned only as future direction; method uses unspecified structured medical KB + textbook corpus. | MedQA-USMLE | Not explicit | No | Medical RAG; UMLS not operational in method as reported. |
| 16 | Tian et al. (2024) — InfuserKI | UMLS triplets as domain KG for infuser-guided knowledge integration via adapters. | PubMedQA | No | No | Parameter-level knowledge integration; question-style supervision from UMLS triplets. |
| 17 | Lu et al. (2023) — HiPrompt | Biomedical KG–to-hierarchy alignment via hierarchy-oriented few-shot prompting. | Knowledge fusion (BKF), not MQA | No | No | Retrieval + re-ranking for entity–term alignment. *Borderline case retained for §5 discussion of adjacent settings.* |
| 18 | Mannion et al. (2023) — UMLS-KGI-BERT | UMLS central to pretraining via entity prediction, link prediction, triple classification. | Clinical NER (related reasoning) | No | No | Strong UMLS integration for pretraining; downstream NER. Cited in §5.1 as the sole reviewed system that evaluates linker performance directly. |
| 19 | Sukhwal et al. (2022) — Healthcare QA Assistant | UMLS to enrich disease-specific KG built from PubMed text; CUIs + synonym merging. | Skin eczema QA assistant | Yes (SciSpaCy + UMLS linking) | Yes | Disease-specific KG from PubMed abstracts + OpenIE; KG-only vs. LM-only vs. joint reasoning comparison. |
| 20 | Wu et al. (2021) — Chest ImaGenome | UMLS used as indexing/normalization layer (scene-graph nodes → CUIs). | Chest X-ray scene graph dataset | No | No | Anatomy-centered scene graphs from radiology reports; UMLS as normalization layer only. *Retained for adjacent-settings discussion.* |
| 21 | Dernbach et al. (2024) — GLaM | UMLS as training graph for graph-aligned LLM fine-tuning; subgraph-encoded QA generation. | Fact recall, multi-hop reasoning over UMLS | No | Yes | QA pairs from UMLS graph neighborhoods + subgraph encodings. |
| 22 | Lu et al. (2023) — Selective UMLS Knowledge Infusion (s41598) | UMLS as injected biomedical KG with semantic-group partitioning for adapter pretraining. | BioASQ7b, PubMedQA, MedQA | No | No | Extends MoP by comparing METIS partitions vs. semantic groupings. |
| 23 | Meng et al. (2021) — Mixture-of-Partitions (MoP) | UMLS partitioned via METIS into smaller subgraphs; adapter pretraining per partition. | PubMedQA, BioASQ7b, BioASQ8b, MedQA, MedNLI | No | No | Global KG partitions, not per-question retrieval. |
| 24 | Michalopoulos et al. (2021) — UmlsBERT | Pretraining with UMLS Metathesaurus: CUI-shared term linking + semantic type embeddings. | MedNLI, clinical NER tasks | Yes | No | UMLS concept + semantic-type knowledge in BERT pretraining. |
| 25 | Lu et al. (2021) - DAKI | Uses the UMLS Metathesaurus graph as one of three knowledge sources, encoded through a dedicated knowledge adapter and combined with other adapters via an attention-based controller. | Medical QA / related reasoning: evaluates on MEDIQA-2019, TRECQA-2017, and MedNLI, plus NER tasks. | No | No | The UMLS adapter is pretrained on graph triples. |

---

# Excluded Papers

Papers screened during manual review but not included in the analytical corpus. Exclusion rationales are organized by the type of mismatch with our inclusion criteria.

## UMLS Not Methodologically Used

| Filename | Exclusion Rationale |
|---|---|
| Gupta et al. (2024) — Cancer CPG QA using LLMs | Method centers on NCCN guideline KG, not UMLS. UMLS appears only as background. |
| Aasman (2024) — Neuro-Symbolic Document Matching | UMLS mentioned as example healthcare taxonomy; task is document matching. |
| Chen et al. (2023) — RoBERT-Agr | No UMLS involvement; agricultural domain. |
| Crum et al. (2024) — Enriching Ontologies with Disjointness Axioms | Ontology enrichment on DBpedia; no UMLS-based method. |
| Liu et al. (2024) — KNOWFORMER | Generic KG reasoning; no UMLS. |
| Patel (2024) — Optimizing KGE for QA Performance | Generic KGQA; not UMLS-based, not medical. |
| Schulz — Ontology-Based Annotation of Clinical Narratives | Annotation principles; not a QA/reasoning system. |
| Tian et al. (2022) — Evaluation System Framework for Medical AI | Evaluation framework; UMLS not methodologically involved. |
| Wang et al. (2024) — Understanding Reasoning Ability of LMs | Analyzes LM reasoning over generic KGs; not UMLS. |
| Youn & Tagkopoulos (2023) — KGLM | Generic KG link prediction; no medical/UMLS scope. |
| Zhang et al. (2021) — GMH | KG completion; no UMLS, not QA. |
| Zhang et al. (2024) — Making LLMs Perform Better in KG Completion | Generic KGC; no UMLS, not medical QA. |

## Task Outside Scope (UMLS Used, but Not for MQA/Reasoning)

| Filename | Exclusion Rationale |
|---|---|
| Adjei-Frempah et al. (2024) — KB2Bench | UMLS used to generate benchmark QA pairs, not to answer questions. Benchmark construction is outside our scope. |
| Luo et al. (2023) — Systematic Assessment of Factual Knowledge in LLMs | Generates factual questions from KGs to test LLMs; benchmark-style evaluation, not UMLS-based MQA methodology. |
| Sakhovskiy et al. (2023) — Graph-Enriched Biomedical LMs (Research Proposal) | Research proposal without concrete implementation. |
| Wallace et al. (2024) — Pruning Cycles in UMLS Metathesaurus | Task is graph cleanup / cycle pruning; not MQA. |
| Jiang et al. (2023) — Text-Augmented Open KGC | Uses UMLS-PubMed as KGC benchmark; task is KG completion, not QA. |
| He et al. (2020) — BERT-MK | UMLS-central but downstream tasks are entity typing and relation classification, not MQA/reasoning. |

## Survey / Review Articles

| Filename | Exclusion Rationale |
|---|---|
| Agrawal et al. (2024) — Can KGs Reduce Hallucinations in LLMs (Survey) | Survey of KG-augmented LLM methods; not a primary methodology paper. |
| Cai et al. (2024) — Bringing Two Worlds Together (Survey) | Broad survey of LLM-KG integration. |

---

## Citation

If you use this review, the per-paper annotations, or the linker/subgraph audit data, please cite:

```bibtex
@inproceedings{anonymous2026umls,
  title     = {Are We Done with UMLS Injection into Language Models
               for Medical Question Answering?},
  author    = {Anonymous},
  booktitle = {Proceedings of EMNLP},
  year      = {2026}
}
```

---

## Contact

For questions, corrections, or to flag a paper we may have missed, please open an Issue on this repository.
