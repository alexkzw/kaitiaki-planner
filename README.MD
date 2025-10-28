# Kaitiaki-Planner: Budget-Aware RAG for Multilingual Fairness

> **A capstone research project exploring fairness in bilingual Retrieval-Augmented Generation (RAG) systems for English and Te Reo Māori**

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.11+-blue.svg"></a>
  <a href="https://nodejs.org/"><img alt="Node" src="https://img.shields.io/badge/node-18+-green.svg"></a>
  <a href="https://www.anthropic.com/"><img alt="Anthropic" src="https://img.shields.io/badge/LLM-Claude%203.5%20Sonnet-purple.svg"></a>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the System](#running-the-system)
- [Project Structure](#project-structure)
- [System Architecture](#system-architecture)
  - [Three Budget Allocation Strategies](#three-budget-allocation-strategies)
  - [Retrieval Methods](#retrieval-methods)
- [Evaluation](#evaluation)
- [Key Results](#key-results)
- [Ethical & Cultural Context](#ethical--cultural-context)
- [Documentation](#documentation)
- [Technologies](#technologies)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

**Kaitiaki-Planner** investigates whether **dynamic budget allocation** in RAG systems can reduce performance disparities between high-resource (English) and low-resource (Te Reo Māori) languages. Grounded in the Māori concept of **kaitiakitanga** (guardianship), the system allocates computational resources to protect the integrity of indigenous language interactions with AI.

### Key Findings

| Metric | BM25 Baseline | Embeddings | Improvement |
|---|---:|---:|---:|
| **Te Reo Māori Accuracy** | 60.0% (9/15) | 66.67% (10/15) | **+11.1%** |
| **English Accuracy** | 100% (15/15) | 100% (15/15) | 0% (ceiling) |
| **Fairness Gap (EN–MI)** | 40.0pp | 33.3pp | **−16.7%** |
| **Budget Allocation Effect** | — | **No effect** (p = 1.0) | Null result |

**Critical Insight:** Semantic embeddings provide modest overall improvement (+11.1%), but this masks a **complexity-dependent trade-off**: simple Māori queries improve substantially (**+27.27pp**, p=0.048, +50% relative gain) while complex queries regress severely (**-50pp**, p=0.007, -66.67% relative loss). Budget allocation provided **zero benefit** across all conditions (ANOVA p=1.0).

**Conclusion:** **Retrieval quality > Budget allocation**, but complexity stratification reveals critical performance gaps requiring targeted solutions beyond current approaches.

---

## Quick Start

### Prerequisites

- Python **3.11+**
- Node.js **18+**
- Anthropic API key

### Installation

```bash
# Clone repository
git clone https://github.com/alexkzw/kaitiaki-planner.git
cd kaitiaki-planner

# Install Python dependencies
cd notebook
pip install -r requirements.txt

# Build embeddings (precomputed vectors)
python build_embeddings.py

# Install Node.js dependencies
cd ../services/orchestrator-ts
npm install

# Set up environment
echo "ANTHROPIC_API_KEY=your_key_here" > ../../notebook/.env
```

### Running the System

<details>
<summary><b>Phase 2A: BM25 Baseline Evaluation</b></summary>

```bash
# Terminal 1: Start orchestrator (http://localhost:8000)
cd services/orchestrator-ts
npm run dev
```

```bash
# Terminal 2: Start BM25 retriever
cd services/retriever-py
python -m uvicorn app:app --port 8001
```

```bash
# Terminal 3: Verify setup
cd notebook
python 01_setup_and_testing.py

# Run BM25 evaluation (saves to outputs/baseline_bm25_results.csv)
python 02a_baseline_bm25_evaluation.py
```
</details>

<details>
<summary><b>Phase 2B: Embeddings Evaluation (Service Mode)</b></summary>

```bash
# Stop BM25 retriever in Terminal 2 (services/retriever-py)
# Keep orchestrator running in Terminal 1
```

```bash
# Terminal 2: Start Embeddings retriever (loads data/corpus_embeddings.pkl)
cd notebook
python retriever_embeddings.py
```

```bash
# Terminal 3: Sanity-check the embeddings retriever
cd notebook
python test_embedding_retriever.py
```
</details>

<details>
<summary><b>Phase 3: Full Embeddings Evaluation</b></summary>

```bash
# Evaluates 90 queries with embeddings retriever
# Saves to outputs/full_evaluation_results.csv
cd notebook
python 02_full_evaluation.py
```
</details>

<details>
<summary><b>Phase 4: Compare & Analyse</b></summary>

```bash
# Compare BM25 vs Embeddings
cd notebook
python 03_baseline_comparison.py

# Generate figures
python 04_analysis_and_visualisation.py

# Run statistical tests
python 05_statistical_test.py
```
</details>

---

## Project Structure

```text
kaitiaki-planner/
├─ data/                          # Knowledge base
│  ├─ corpus.json                 # 30 Wikipedia docs (15 EN, 15 MI)
│  ├─ corpus_embeddings.pkl       # Precomputed embeddings
│  └─ ATTRIBUTION.md              # CC-BY-SA license
│
├─ eval/
│  └─ tasks_labeled.yaml          # 30 evaluation queries
│
├─ notebook/                      # Experiments & analysis
│  ├─ build_embeddings.py         # Generate embeddings
│  ├─ retriever_embeddings.py     # Semantic retriever service
│  ├─ 01_setup_and_testing.py     # System validation
│  ├─ 02_full_evaluation.py       # Embeddings evaluation (90 queries)
│  ├─ 02a_baseline_bm25_evaluation.py  # BM25 baseline evaluation
│  ├─ 03_baseline_comparison.py   # Compare BM25 vs embeddings
│  ├─ 04_analysis_and_visualisation.py  # Generate figures
│  ├─ 05_statistical_test.py      # Statistical analysis
│  ├─ claude_client.py            # LLM wrapper with budget tracking
│  └─ eval_utils.py               # Evaluation utilities
│
├─ services/
│  └─ orchestrator-ts/            # Query orchestration
│     └─ src/server.ts            # Budget allocation logic
│
├─ outputs/                       # Results & figures
│  ├─ full_evaluation_results.csv      # Embeddings evaluation (90 rows)
│  ├─ baseline_bm25_results.csv        # BM25 baseline (90 rows)
│  ├─ baseline_vs_embeddings_comparison.csv
│  ├─ query_level_improvements.csv
│  ├─ statistical_tests.csv
│  ├─ complexity_analysis.csv
│  ├─ figures/                         # 6 visualization files
│  │  ├─ before_after_comparison.png
│  │  ├─ gc_by_complexity.png
│  │  ├─ null_result_identical_conditions.png
│  │  └─ (3 more figures)
│  └─ capstone_report.docx      # 4-page report
│
└─ README.md
```

---

## System Architecture

```mermaid
flowchart LR
    A["User Query"] --> B["Orchestrator - Budget Planning"]
    B --> C["Retriever - Semantic Search"]
    C --> D["Claude 3.5 Sonnet - Answer Generation"]
    D --> E["Grounded Answer"]
```

</details>

### Three Budget Allocation Strategies

| Strategy | Description | English Budget | Māori Budget |
|---|---|---:|---:|
| **Uniform** | Equal treatment (baseline) | 5 docs | 5 docs |
| **Language-Aware** | More context for low-resource language | 5 docs | 8 docs |
| **Fairness-Aware** | Prioritise Māori + complex queries | 5 docs (simple)<br>8 docs (complex) | 8 docs |

### Retrieval Methods

**Baseline: BM25** (keyword-based)
- Fast exact matching
- Poor semantic understanding
- Results: 60.0% Māori accuracy overall
- Complexity pattern: Simple 54.5% (6/11) vs Complex 75% (3/4)

**Improved: Semantic Embeddings** (multilingual)
- Model: `paraphrase-multilingual-mpnet-base-v2` (768-dim)
- Cosine similarity + 30% keyword boost
- Results: 66.67% Māori accuracy overall (**+11.1% improvement**)
- Complexity pattern: Simple 81.82% (9/11) vs Complex 25% (1/4)
- **Critical finding:** Embeddings help simple queries but hurt complex queries

---

## Evaluation

### Dataset

- **30 queries** (15 English, 15 Te Reo Māori)  
- **Topics:** Māori culture, NZ geography, wildlife, history  
- **Complexity:** 20 simple (factual), 10 complex (inference)

### Metric: Grounded Correctness (GC)

Binary evaluation:

```text
GC = 1  if (gold_doc_id in retrieved_passages) AND (answer cites gold passage)
GC = 0  otherwise
```

### Statistical Tests

- **Independent t-test** (t=2.646, p=0.013): Language gap is statistically significant
- **Cohen's d** (0.966): Large effect size for EN-MI disparity
- **Complexity-stratified tests**: Simple improvement p=0.048 (sig.), Complex regression p=0.007 (sig.)
- **Confidence intervals**: Māori simple [0.546, 1.090], Māori complex [-0.546, 1.046] (extreme uncertainty, n=4)

---

## Key Results

### Impact of Retrieval Quality

Switching from BM25 to semantic embeddings:
- **+11.1% relative improvement** for Te Reo Māori overall (9/15 -> 10/15)
- **−16.7% reduction** in fairness gap (40pp -> 33.3pp)
- **Language gap statistically significant** (t=2.646, p=0.013, Cohen's d=0.966)

### Complexity-Dependent Trade-Off (Critical Finding)

**Simple Māori queries** (11 queries):
- BM25: 54.55% (6/11) -> Embeddings: 81.82% (9/11)
- **+27.27pp improvement** (p=0.048, +50% relative gain)
- Fairness gap: 18.2pp (EN 100% vs MI 81.82%)

**Complex Māori queries** (4 queries):
- BM25: 75% (3/4) -> Embeddings: 25% (1/4)
- **-50pp regression** (p=0.007, -66.67% relative loss)
- Fairness gap: 75pp (EN 100% vs MI 25%)

**Interpretation:** Embeddings create a two-tier fairness outcome—equitable on simple queries but severely inequitable on complex reasoning tasks. Current approaches (BM25 or embeddings) cannot solve the complexity gap.

### Null Result: Budget Allocation

Allocating 60% more budget (5 -> 8 documents):
- **0% improvement** across all conditions
- **p = 1.0** (perfect equivalence, ANOVA F≈0)
- **Null finding:** Budget allocation provides no benefit when retrieval quality determines which passages are ranked in top 5

---

## Ethical & Cultural Context

This project centres **Māori data sovereignty** principles from [Te Mana Raraunga](https://www.temanararaunga.maori.nz/):

- **Rangatiratanga** (Authority): Māori govern how Te Reo Māori appears in AI  
- **Whakapapa** (Relationality): Knowledge exists in cultural context  
- **Kaitiakitanga** (Guardianship): Computational stewardship responsibility

See `outputs/capstone_report.pdf` for full ethical analysis.

---

## Documentation

- **[Capstone Report](outputs/capstone_report.docx)** — 4-page technical report with results and ethical analysis
- **[Data Attribution](data/ATTRIBUTION.md)** — CC-BY-SA license compliance
- **[Statistical Tests](outputs/statistical_tests.csv)** — Complete statistical analysis results
- **[Complexity Analysis](outputs/complexity_analysis.csv)** — Stratified performance by query complexity

---

## Technologies

| Component | Technology | Purpose |
|---|---|---|
| **Orchestrator** | TypeScript + Fastify | Budget allocation logic |
| **Retriever** | Python + FastAPI | Semantic search (embeddings) |
| **LLM** | Claude 3.5 Sonnet | Answer generation (~$0.0048/query) |
| **Embeddings** | sentence-transformers | Multilingual semantic vectors |
| **Analysis** | SciPy + Matplotlib | Statistics & visualisation |

---

## License

MIT License — see [LICENSE](LICENSE).

**Educational Use:** This project is for academic research. Māori text used with cultural sensitivity and respect. Wikipedia content used under CC-BY-SA (see `data/ATTRIBUTION.md`).

---

## Acknowledgments

- Te Mana Raraunga for Māori data sovereignty principles  
- Anthropic for Claude API access  
- Wikipedia contributors for Te Reo Māori content (CC-BY-SA)