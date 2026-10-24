<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>🧬 Are We Done with UMLS Injection into Language Models for Medical Question Answering? 🧬</h1><br>
      A systematic review of how the <i>Unified Medical Language System</i> is operationalized in medical QA
    </summary>
  </ul>
</div>

[![Awesome](https://awesome.re/badge.svg)](#)
<!--[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) -->

This repository accompanies the paper **"Are We Done with UMLS Injection into Language Models for Medical Question Answering?"** It contains the code, data, and per-paper annotations supporting our systematic review of how the Unified Medical Language System (UMLS) is operationalized in medical question answering (MQA) and related reasoning tasks.

## 🔔 News

- **[2026-XX-XX]** Repository released with per-paper annotations, linker comparison, and subgraph audit scripts.

## 🔍 Contents

- [🩺 TL;DR](#tldr)
- [📦 Repository Contents](#repo)
- [🧪 Review Methodology](#method)
- [🧭 Three-Stage Analytical Scheme](#scheme)
- [💡 Key Findings](#findings)
- [📜 Papers](#papers)
    - [🏷️ Tagset](#tagset)
    - [🎯 The List](#list)
    - [🚫 Excluded Papers](#excluded)
- [🚩 Citation](#citation)
- [📬 Contact](#contact)

<a id="tldr"></a>
## 🩺 TL;DR

We systematically review 25 papers that integrate UMLS into language models for MQA and related reasoning settings. We find that the pipeline used to consume UMLS — *linking → subgraph retrieval → knowledge injection* — varies substantially across systems in ways that are seldom reported, making published benchmark comparisons difficult to interpret. We propose a reporting checklist and design recommendations to improve reproducibility.

> Anonymized review repository: <https://anonymous.4open.science/r/are-we-done-with-umls>

<a id="repo"></a>
## 📦 Repository Contents

- **`src/`** — Scripts for reproducing the linker comparison (scispaCy vs. MetaMap), $K$-hop subgraph scale measurements, and pruning audit on DRAGON preprocessing artifacts.
- **`outputs/`** — Linker coverage statistics on MedQA, $K$-hop scale measurements on PubMedQA, and pre/post-pruning statistics on MedQA.
- **`README.md`** — The per-paper annotation of all 25 reviewed systems ([The List](#list)), recording the UMLS tables, subset filter, entity linker, $K$-hop budget, and injection mechanism of each.

### Data

No UMLS data, subsets, or derived artifacts are redistributed here: the NLM licence does not permit it. To reproduce our experiments, obtain the releases you need from the [UTS](https://uts.nlm.nih.gov/uts/) under your own licence, and download the QA benchmarks from their original sources ([MedQA](https://github.com/jind11/MedQA), [MedMCQA](https://medmcqa.github.io/), [PubMedQA](https://pubmedqa.github.io/)). The scripts in `src/` then regenerate the linked and grounded files locally; `data/readme.md` documents their schema.

<a id="method"></a>
## 🧪 Review Methodology

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

<a id="scheme"></a>
## 🧭 Three-Stage Analytical Scheme

The paper decomposes UMLS use in MQA into three pipeline stages:

| Stage | What it does | Where variation enters |
|---|---|---|
| **1. Entity Linking & Coverage** | Grounds input spans to UMLS concepts | Linker choice, span coverage on question stems vs. answer options |
| **2. Subgraph Retrieval** | Selects a question-specific neighborhood of the graph | Expansion depth ($K$), pruning budget, connectivity criterion, fallback behavior |
| **3. Knowledge Injection** | Determines how the selected knowledge is exposed to the model | Linearization, structural injection, or parameter encoding |

Each stage attenuates evidence asymmetrically across the input, and the cumulative effect is rarely reported.

<a id="findings"></a>
## 💡 Key Findings

- **Only 6 of 25 reviewed papers** supply enough pipeline detail to enable partial reproduction.
- **Linker choice** is the most consistently reported pipeline detail; scispaCy dominates ($n=9$).
- **Answer-option spans** fail to link to any UMLS concept in roughly **60%** of MedQA instances under the QA-GNN/GREASELM preprocessing pipeline, against ~4% for question stems.
- **The $K=2$ hop default** is empirically workable but only because of aggressive downstream pruning to a 200-node budget; deeper expansion grows subgraphs by multiple orders of magnitude per hop.
- After pruning, **only ~1.5 answer-side concepts on average** survive per question–option pair in DRAGON's released MedQA preprocessing, despite the multiple-choice format requiring discrimination among four options.
- **QA-GNN, GREASELM, and DRAGON** share the same pruning structure but use different scoring functions (RoBERTa-large MLM loss in QA-GNN/GREASELM; GloVe-50 dot product in DRAGON), a divergence not acknowledged in cross-system comparisons.

<a id="papers"></a>
## 📜 Papers

<a id="tagset"></a>
### 🏷️ Tagset

We tag each paper along the three pipeline stages plus the UMLS subset it draws from. These tags surface the dimensions along which reviewed systems most often diverge.

#### UMLS source (which Metathesaurus tables are touched)

| Tag | Explanation |
|---|---|
| ![](https://img.shields.io/badge/MRCONSO-2f5d6e) | Uses `MRCONSO` (concept names / strings) |
| ![](https://img.shields.io/badge/MRREL-3a7d8c) | Uses `MRREL` (concept relations) |
| ![](https://img.shields.io/badge/MRSTY-4e9aa6) | Uses `MRSTY` (semantic types) |
| ![](https://img.shields.io/badge/MRDEF-6bb0b8) | Uses `MRDEF` (definitions) |
| ![](https://img.shields.io/badge/SRDEF-8ec6c0) | Uses `SRDEF` (Semantic Network, not Metathesaurus) |
| ![](https://img.shields.io/badge/UMLS%20subset-b08a5a) | Restricts UMLS via vocabulary, semantic type, or relation filter (e.g., DDB, SNOMED CT, MSH/NCI) |

#### Stage 1 — Entity linking

| Tag | Explanation |
|---|---|
| ![](https://img.shields.io/badge/NEL%3A%20scispaCy-7c7f55) | Entity linking via scispaCy |
| ![](https://img.shields.io/badge/NEL%3A%20MetaMap-94a35e) | Entity linking via MetaMap |
| ![](https://img.shields.io/badge/NEL%3A%20cTAKES-aab66a) | Entity linking via cTAKES |
| ![](https://img.shields.io/badge/NEL%3A%20string%20match-c0c97a) | Surface-form / string-matching linker |
| ![](https://img.shields.io/badge/NEL%3A%20neural%20embedding-d6dc88) | Neural-embedding-based linker |
| ![](https://img.shields.io/badge/NEL%3A%20LLM%20prompt-e3e89a) | LLM-prompted entity extraction |
| ![](https://img.shields.io/badge/NEL%3A%20not%20specified-bfbfbf) | Linker not reported |

#### Stage 2 — Subgraph retrieval ($K$-hop and budget)

| Tag | Explanation |
|---|---|
| ![](https://img.shields.io/badge/K%3D1-d98d49) | One-hop neighborhood |
| ![](https://img.shields.io/badge/K%3D2-e16d50) | Two-hop expansion |
| ![](https://img.shields.io/badge/K%3D3-c25a47) | Three-hop expansion |
| ![](https://img.shields.io/badge/Budget%3A%20200%20nodes-7d4f30) | Hard pruning budget to 200 nodes |
| ![](https://img.shields.io/badge/Budget%3A%2020%20rel.-7d4f30) | Capped at top-20 relations |
| ![](https://img.shields.io/badge/Budget%3A%20unspecified-bfbfbf) | Pruning budget not reported |
| ![](https://img.shields.io/badge/No%20per--question%20retrieval-9e9e9e) | No question-specific subgraph (global / partitioned KG) |

#### Stage 3 — Knowledge injection

| Tag | Explanation |
|---|---|
| ![](https://img.shields.io/badge/Structural%20injection-84648a) | KG fed to the model as structured graph (GNN, cross-modal layers) |
| ![](https://img.shields.io/badge/Textual%20linearization-c381a7) | KG flattened to text and prepended/appended to the prompt |
| ![](https://img.shields.io/badge/Parameter%20encoding-6d738d) | KG knowledge stored in adapters / pretraining objectives |
| ![](https://img.shields.io/badge/Injection%3A%20not%20specified-bfbfbf) | Injection mechanism not reported |

<a id="list"></a>
### 🎯 The List

> [!Note]
> Ordered as in the paper's per-system annotation table. Tags below each entry summarize the **UMLS tables used**, an optional **UMLS subset filter**, the **entity linker**, the **$K$-hop / budget** of subgraph retrieval, and the **injection mechanism**. The *Key Points* line preserves our qualitative annotation of what the paper actually does.

1. **DRAGON: Deep Bidirectional Language-Knowledge Graph Pretraining** — Yasunaga et al. (NeurIPS 2022) [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/f43c9e7e3a5e9d9ddb1c1c0b7a3f0e2e-Abstract-Conference.html)] [[code](https://github.com/michiyasunaga/dragon)]<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e) ![](https://img.shields.io/badge/MRREL-3a7d8c)<br>![](https://img.shields.io/badge/NEL%3A%20scispaCy-7c7f55)<br>![](https://img.shields.io/badge/K%3D2-e16d50) ![](https://img.shields.io/badge/Budget%3A%20200%20nodes-7d4f30)<br>![](https://img.shields.io/badge/Structural%20injection-84648a)<br>**Key Points:** Deeply bidirectional text-KG pretraining; 2-hop expansion + entity linking; masked LM + KG link prediction joint objectives. Strong fit.

2. **MediSage** — Choudhury et al. (WWW 2023)<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e) ![](https://img.shields.io/badge/MRREL-3a7d8c) ![](https://img.shields.io/badge/MRSTY-4e9aa6)<br>![](https://img.shields.io/badge/NEL%3A%20string%20match-c0c97a)<br>![](https://img.shields.io/badge/K%3D3-c25a47) ![](https://img.shields.io/badge/Budget%3A%20unspecified-bfbfbf)<br>![](https://img.shields.io/badge/Injection%3A%20not%20specified-bfbfbf)<br>**Key Points:** Conversational clinical decision-support over EHR + clinical KG; BioLinkBERT NEL fallback. Not centered on UMLS as core source.

3. **KnowledgeDA** — Ding et al. (Findings of ACL 2023) [[paper](https://aclanthology.org/2023.findings-acl.477/)] [[code](https://github.com/RuiqingDing/KnowledgeDA)]<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e) ![](https://img.shields.io/badge/MRREL-3a7d8c) ![](https://img.shields.io/badge/MRSTY-4e9aa6) ![](https://img.shields.io/badge/UMLS%20subset-b08a5a)<br>![](https://img.shields.io/badge/NEL%3A%20neural%20embedding-d6dc88)<br>![](https://img.shields.io/badge/K%3D2-e16d50) ![](https://img.shields.io/badge/Budget%3A%20unspecified-bfbfbf)<br>![](https://img.shields.io/badge/Textual%20linearization-c381a7)<br>**Key Points:** Mention-to-entity similarity matching; 2-hop relation-consistent replacements; data augmentation around UMLS-grounded entities. UMLS 2021AA; 1.15M CUIs, 4.88M triples after filtering to medicine-related semantic types and 502 relation types.

4. **KG-Infused LLM for Virtual Health Assistant** — Katta et al. (ICMLA 2024) [[code](https://github.com/SivaKatta1010/NeoKG-LLM)]<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e) ![](https://img.shields.io/badge/MRREL-3a7d8c) ![](https://img.shields.io/badge/MRDEF-6bb0b8) ![](https://img.shields.io/badge/UMLS%20subset-b08a5a)<br>![](https://img.shields.io/badge/NEL%3A%20scispaCy-7c7f55)<br>![](https://img.shields.io/badge/K%3D1-d98d49) ![](https://img.shields.io/badge/Budget%3A%2020%20rel.-7d4f30)<br>![](https://img.shields.io/badge/Textual%20linearization-c381a7)<br>**Key Points:** Builds large Neo4j KG from UMLS subsets (MSH + NCI), 622K nodes / 4.94M edges; RAG-style pipeline with semantic matching + learning-to-rank; not question-specific subgraph construction.

5. **QAVSA** — Laube & Eliasmith (RepL4NLP 2024)<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e) ![](https://img.shields.io/badge/MRREL-3a7d8c)<br>![](https://img.shields.io/badge/NEL%3A%20scispaCy-7c7f55)<br>![](https://img.shields.io/badge/K%3D2-e16d50) ![](https://img.shields.io/badge/Budget%3A%20unspecified-bfbfbf)<br>![](https://img.shields.io/badge/Structural%20injection-84648a)<br>**Key Points:** Follows QA-GNN-style graph setup (300K nodes / 1M edges); replaces GNN with vector symbolic algebra + MLP reasoning on MedQA-USMLE.

6. **QAT: Relation-Aware Language-Graph Transformer** — Park et al. (AAAI 2023)<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e) ![](https://img.shields.io/badge/MRREL-3a7d8c)<br>![](https://img.shields.io/badge/NEL%3A%20not%20specified-bfbfbf)<br>![](https://img.shields.io/badge/K%3D2-e16d50) ![](https://img.shields.io/badge/Budget%3A%20200%20nodes-7d4f30)<br>![](https://img.shields.io/badge/Structural%20injection-84648a)<br>**Key Points:** Per-QA subgraph retrieval (9K nodes / 44K edges); meta-path relation encoding. Q→A connectivity.

7. **QA-GNN: Reasoning with Language Models and Knowledge Graphs for Question Answering** — Yasunaga et al. (NAACL 2021) [[paper](https://aclanthology.org/2021.naacl-main.45/)] [[code](https://github.com/michiyasunaga/qagnn)]<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e) ![](https://img.shields.io/badge/MRREL-3a7d8c) ![](https://img.shields.io/badge/UMLS%20subset-b08a5a)<br>![](https://img.shields.io/badge/NEL%3A%20scispaCy-7c7f55)<br>![](https://img.shields.io/badge/K%3D2-e16d50) ![](https://img.shields.io/badge/Budget%3A%20200%20nodes-7d4f30)<br>![](https://img.shields.io/badge/Structural%20injection-84648a)<br>**Key Points:** Self-constructed biomedical KG (DDB portion of UMLS + DrugBank; 9K nodes / 44K edges). Question–choice-specific subgraph + LM-based relevance scoring + joint LM+KG reasoning. Canonical benchmark paper.

8. **Towards Efficient Methods using KG Embeddings** — Sengupta et al. (BIBM 2024) [[code](https://github.com/saptarshi059/CDQA-project)]<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e) ![](https://img.shields.io/badge/MRREL-3a7d8c) ![](https://img.shields.io/badge/MRDEF-6bb0b8)<br>![](https://img.shields.io/badge/NEL%3A%20MetaMap-94a35e)<br>![](https://img.shields.io/badge/K%3D1-d98d49) ![](https://img.shields.io/badge/Budget%3A%20unspecified-bfbfbf)<br>![](https://img.shields.io/badge/Structural%20injection-84648a)<br>**Key Points:** UMLS definitions + pre-trained UMLS KGE fused into BERT/RoBERTa for COVID-QA / PubMedQA. Embedding/definition infusion; not subgraph construction.

9. **Graph Neural Prompting (GNP)** — Tian et al. (AAAI 2024)<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e) ![](https://img.shields.io/badge/MRREL-3a7d8c)<br>![](https://img.shields.io/badge/NEL%3A%20scispaCy-7c7f55)<br>![](https://img.shields.io/badge/K%3D2-e16d50) ![](https://img.shields.io/badge/Budget%3A%20unspecified-bfbfbf)<br>![](https://img.shields.io/badge/Structural%20injection-84648a)<br>**Key Points:** NEL → 2-hop subgraph → GNN + cross-modality pooling + domain projector. UMLS as biomedical-domain KG encoded into graph neural prompts for LLM prompting.

10. **Adapter Modules for Knowledge Enhancement** — Vladika et al. (ICAART 2024)<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e) ![](https://img.shields.io/badge/MRREL-3a7d8c) ![](https://img.shields.io/badge/UMLS%20subset-b08a5a)<br>![](https://img.shields.io/badge/NEL%3A%20not%20specified-bfbfbf)<br>![](https://img.shields.io/badge/No%20per--question%20retrieval-9e9e9e)<br>![](https://img.shields.io/badge/Parameter%20encoding-6d738d)<br>**Key Points:** UMLS 2023AA; SNOMED CT US subset (1.75M triplets). Top-20 UMLS relations + METIS partitioning; entity-prediction adapter training; evaluated on BioASQ-7b / PubMedQA / MedNLI / HoC.

11. **Medical Knowledge-Enriched Textual Entailment** — Yadav et al. (COLING 2020) [[paper](https://aclanthology.org/2020.coling-main.288/)] [[code](https://github.com/VishalPallagani/Medical-Knowledge-enriched-Textual-Entailment)]<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e) ![](https://img.shields.io/badge/MRREL-3a7d8c)<br>![](https://img.shields.io/badge/NEL%3A%20scispaCy-7c7f55)<br>![](https://img.shields.io/badge/K%3D2-e16d50) ![](https://img.shields.io/badge/Budget%3A%20unspecified-bfbfbf)<br>![](https://img.shields.io/badge/Structural%20injection-84648a)<br>**Key Points:** UMLS + SNOMED-CT + ICD10 to enrich vocabulary graph for MEDIQA-RQE entailment. Knowledge-enriched graph + BERT via GCN + attention fusion.

12. **KG-Rank** — Yang et al. (BioNLP 2024) [[code](https://github.com/YangRui525/KG-Rank)]<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e) ![](https://img.shields.io/badge/MRREL-3a7d8c)<br>![](https://img.shields.io/badge/NEL%3A%20LLM%20prompt-e3e89a)<br>![](https://img.shields.io/badge/K%3D1-d98d49) ![](https://img.shields.io/badge/Budget%3A%20unspecified-bfbfbf)<br>![](https://img.shields.io/badge/Textual%20linearization-c381a7)<br>**Key Points:** UMLS as primary medical KG; one-hop triples retrieved and re-ranked before LLM generation. Similarity ranking + MMR + cross-encoder re-ranking. Evaluated on LiveQA, ExpertQA-Med/Bio, MedicationQA.

13. **DRLK** — Zhang et al. (EMNLP 2022) [[code](https://github.com/MZ-MiaoZhang/DRLK)]<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e) ![](https://img.shields.io/badge/MRREL-3a7d8c)<br>![](https://img.shields.io/badge/NEL%3A%20scispaCy-7c7f55)<br>![](https://img.shields.io/badge/K%3D2-e16d50) ![](https://img.shields.io/badge/Budget%3A%20200%20nodes-7d4f30)<br>![](https://img.shields.io/badge/Structural%20injection-84648a)<br>**Key Points:** Pre-computed medical KG (DDB ⊂ UMLS + DrugBank; 9K nodes / 44K edges). QA-GNN/GREASELM-style preprocessing; dynamic hierarchical LM-KG interaction.

14. **GREASELM** — Zhang et al. (ICLR 2022) [[code](https://github.com/snap-stanford/GreaseLM)]<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e) ![](https://img.shields.io/badge/MRREL-3a7d8c) ![](https://img.shields.io/badge/UMLS%20subset-b08a5a)<br>![](https://img.shields.io/badge/NEL%3A%20scispaCy-7c7f55)<br>![](https://img.shields.io/badge/K%3D2-e16d50) ![](https://img.shields.io/badge/Budget%3A%20200%20nodes-7d4f30)<br>![](https://img.shields.io/badge/Structural%20injection-84648a)<br>**Key Points:** Self-constructed medical KG (DDB ⊂ UMLS + DrugBank; 9K nodes / 44K edges). NEL + 2-hop bridge entities + top-200 node retrieval; LM-KG cross-modal fusion across layers.

15. **Heterogeneous Knowledge Grounding** — Zhao et al. (WWW 2024)<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e) ![](https://img.shields.io/badge/MRREL-3a7d8c) ![](https://img.shields.io/badge/UMLS%20subset-b08a5a)<br>![](https://img.shields.io/badge/NEL%3A%20neural%20embedding-d6dc88)<br>![](https://img.shields.io/badge/K%3D1-d98d49) ![](https://img.shields.io/badge/Budget%3A%20unspecified-bfbfbf)<br>![](https://img.shields.io/badge/Textual%20linearization-c381a7)<br>**Key Points:** DDB subset (50K triples); UMLS mentioned mainly as future direction — method uses unspecified structured medical KB + textbook corpus. Medical RAG; UMLS not operational in method as reported.

16. **InfuserKI** — Tian et al. (EMNLP 2024)<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e) ![](https://img.shields.io/badge/MRREL-3a7d8c)<br>![](https://img.shields.io/badge/NEL%3A%20not%20specified-bfbfbf)<br>![](https://img.shields.io/badge/No%20per--question%20retrieval-9e9e9e)<br>![](https://img.shields.io/badge/Parameter%20encoding-6d738d)<br>**Key Points:** UMLS triplets (2.5K & 25K) as domain KG for infuser-guided knowledge integration via adapters. Parameter-level integration; question-style supervision from UMLS triplets.

17. **HiPrompt** — Lu et al. (SIGIR 2023)<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e)<br>![](https://img.shields.io/badge/NEL%3A%20not%20specified-bfbfbf)<br>![](https://img.shields.io/badge/No%20per--question%20retrieval-9e9e9e)<br>![](https://img.shields.io/badge/Injection%3A%20not%20specified-bfbfbf)<br>**Key Points:** Biomedical KG–to-hierarchy alignment via hierarchy-oriented few-shot prompting; retrieval + re-ranking for entity–term alignment. *Borderline case retained for §5 discussion of adjacent settings.*

18. **UMLS-KGI-BERT** — Mannion et al. (ClinicalNLP 2023)<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e) ![](https://img.shields.io/badge/MRREL-3a7d8c) ![](https://img.shields.io/badge/MRSTY-4e9aa6) ![](https://img.shields.io/badge/UMLS%20subset-b08a5a)<br>![](https://img.shields.io/badge/NEL%3A%20string%20match-c0c97a)<br>![](https://img.shields.io/badge/K%3D1-d98d49) ![](https://img.shields.io/badge/Budget%3A%20unspecified-bfbfbf)<br>![](https://img.shields.io/badge/Parameter%20encoding-6d738d)<br>**Key Points:** UMLS 2022AB; restricted to human-biology/medicine semantic types (2.53M CUIs). Pretraining via entity prediction, link prediction, triple classification; downstream clinical NER. Cited in §5.1 as the sole reviewed system that evaluates linker performance directly.

19. **Healthcare QA Assistant** — Sukhwal et al. (ICIS 2022)<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e)<br>![](https://img.shields.io/badge/NEL%3A%20scispaCy-7c7f55)<br>![](https://img.shields.io/badge/K%3D1-d98d49) ![](https://img.shields.io/badge/Budget%3A%20unspecified-bfbfbf)<br>![](https://img.shields.io/badge/Injection%3A%20not%20specified-bfbfbf)<br>**Key Points:** Disease-specific KG (7.17K CUIs) from PubMed abstracts + OpenIE; CUIs + synonym merging; KG-only vs. LM-only vs. joint reasoning comparison on a skin-eczema QA assistant.

20. **Chest ImaGenome** — Wu et al. (NeurIPS 2021)<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e)<br>![](https://img.shields.io/badge/NEL%3A%20scispaCy-7c7f55)<br>![](https://img.shields.io/badge/No%20per--question%20retrieval-9e9e9e)<br>![](https://img.shields.io/badge/Injection%3A%20not%20specified-bfbfbf)<br>**Key Points:** Anatomy-centered scene graphs from radiology reports; UMLS as indexing/normalization layer only (scene-graph nodes → CUIs). *Retained for adjacent-settings discussion.*

21. **GLaM: Graph-Aligned Language Models** — Dernbach et al. (AAAI-SSS 2024)<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e) ![](https://img.shields.io/badge/MRREL-3a7d8c) ![](https://img.shields.io/badge/UMLS%20subset-b08a5a)<br>![](https://img.shields.io/badge/NEL%3A%20not%20specified-bfbfbf)<br>![](https://img.shields.io/badge/No%20per--question%20retrieval-9e9e9e)<br>![](https://img.shields.io/badge/Textual%20linearization-c381a7)<br>**Key Points:** UMLS as training graph for graph-aligned LLM fine-tuning; 4 relation types, 126K triples. QA pairs synthesized from UMLS graph neighborhoods + subgraph encodings; evaluated on fact recall and multi-hop reasoning over UMLS.

22. **Selective UMLS Knowledge Infusion** — Park et al. (Scientific Reports, 2024)<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e) ![](https://img.shields.io/badge/MRREL-3a7d8c) ![](https://img.shields.io/badge/MRSTY-4e9aa6) ![](https://img.shields.io/badge/SRDEF-8ec6c0)<br>![](https://img.shields.io/badge/NEL%3A%20not%20specified-bfbfbf)<br>![](https://img.shields.io/badge/No%20per--question%20retrieval-9e9e9e)<br>![](https://img.shields.io/badge/Parameter%20encoding-6d738d)<br>**Key Points:** Extends MoP by comparing METIS partitions vs. semantic-group partitioning of UMLS for adapter pretraining; evaluated on BioASQ7b / PubMedQA / MedQA. (SRDEF is part of the UMLS Semantic Network, not the Metathesaurus.)

23. **Mixture-of-Partitions (MoP)** — Meng et al. (EMNLP 2021) [[paper](https://aclanthology.org/2021.emnlp-main.327/)] [[code](https://github.com/cambridgeltl/mop)]<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e) ![](https://img.shields.io/badge/MRREL-3a7d8c) ![](https://img.shields.io/badge/UMLS%20subset-b08a5a)<br>![](https://img.shields.io/badge/NEL%3A%20not%20specified-bfbfbf)<br>![](https://img.shields.io/badge/No%20per--question%20retrieval-9e9e9e)<br>![](https://img.shields.io/badge/Parameter%20encoding-6d738d)<br>**Key Points:** SNOMED CT US subset. UMLS partitioned via METIS into smaller subgraphs; adapter pretraining per partition. Global KG partitions, not per-question retrieval. Evaluated on PubMedQA / BioASQ7b / BioASQ8b / MedQA / MedNLI.

24. **UmlsBERT** — Michalopoulos et al. (NAACL 2021) [[paper](https://aclanthology.org/2021.naacl-main.139/)] [[code](https://github.com/gmichalo/UmlsBERT)]<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e) ![](https://img.shields.io/badge/MRSTY-4e9aa6)<br>![](https://img.shields.io/badge/NEL%3A%20cTAKES-aab66a)<br>![](https://img.shields.io/badge/No%20per--question%20retrieval-9e9e9e)<br>![](https://img.shields.io/badge/Parameter%20encoding-6d738d)<br>**Key Points:** UMLS 2020AA. Pretraining with UMLS Metathesaurus: CUI-shared term linking + semantic type embeddings. Evaluated on MedNLI + clinical NER.

25. **DAKI** — Lu et al. (Findings of EMNLP 2021) [[paper](https://aclanthology.org/2021.findings-emnlp.20/)]<br>![](https://img.shields.io/badge/MRCONSO-2f5d6e) ![](https://img.shields.io/badge/MRREL-3a7d8c)<br>![](https://img.shields.io/badge/NEL%3A%20not%20specified-bfbfbf)<br>![](https://img.shields.io/badge/K%3D1-d98d49) ![](https://img.shields.io/badge/Budget%3A%20unspecified-bfbfbf)<br>![](https://img.shields.io/badge/Parameter%20encoding-6d738d)<br>**Key Points:** UMLS Metathesaurus graph (1.77M triples) as one of three knowledge sources, encoded through a dedicated knowledge adapter and combined via an attention-based controller. UMLS adapter pretrained on graph triples; evaluated on MEDIQA-2019, TRECQA-2017, MedNLI + NER.

<a id="excluded"></a>
### 🚫 Excluded Papers

Papers screened during manual review but not included in the analytical corpus. Exclusion rationales are organized by the type of mismatch with our inclusion criteria.

#### UMLS Not Methodologically Used

| Paper | Exclusion Rationale |
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
| Agrawal et al. (2024) — Can KGs Reduce Hallucinations in LLMs (Survey) | Survey of KG-augmented LLM methods; not a primary methodology paper. |
| Cai et al. (2024) — Bringing Two Worlds Together (Survey) | Broad survey of LLM-KG integration. |

#### Task Outside Scope (UMLS Used, but Not for MQA/Reasoning)

| Paper | Exclusion Rationale |
|---|---|
| Adjei-Frempah et al. (2024) — KB2Bench | UMLS used to generate benchmark QA pairs, not to answer questions. Benchmark construction is outside our scope. |
| Luo et al. (2023) — Systematic Assessment of Factual Knowledge in LLMs | Generates factual questions from KGs to test LLMs; benchmark-style evaluation, not UMLS-based MQA methodology. |
| Sakhovskiy et al. (2023) — Graph-Enriched Biomedical LMs (Research Proposal) | Research proposal without concrete implementation. |
| Wallace et al. (2024) — Pruning Cycles in UMLS Metathesaurus | Task is graph cleanup / cycle pruning; not MQA. |
| Jiang et al. (2023) — Text-Augmented Open KGC | Uses UMLS-PubMed as KGC benchmark; task is KG completion, not QA. |
| He et al. (2020) — BERT-MK | UMLS-central but downstream tasks are entity typing and relation classification, not MQA/reasoning. |

<a id="citation"></a>
## 🚩 Citation

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

<a id="contact"></a>
## 📬 Contact

For questions, corrections, or to flag a paper we may have missed, please open an Issue on this repository.

<p align="right"><a href="#user-content-toc">🔝Back to top</a></p>