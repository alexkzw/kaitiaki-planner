# Kaitiaki-Planner: Budget-Aware RAG for Multilingual Fairness

> **A capstone research project exploring fairness in bilingual Retrieval-Augmented Generation (RAG) systems for English and Te Reo Māori**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Node](https://img.shields.io/badge/node-18+-green.svg)](https://nodejs.org/)
[![Anthropic](https://img.shields.io/badge/LLM-Claude%203.5%20Sonnet-purple.svg)](https://www.anthropic.com/)

---

## Overview

**Kaitiaki-Planner** investigates whether **dynamic budget allocation** in RAG systems can reduce performance disparities between high-resource (English) and low-resource (Te Reo Māori) languages. Grounded in the Māori concept of **kaitiakitanga** (guardianship), the system allocates computational resources to protect the integrity of indigenous language interactions with AI.

### Key Findings

| Metric | BM25 Baseline | Embeddings | Improvement |
|--------|---------------|------------|-------------|
| **Te Reo Māori Accuracy** | 53.3% (8/15) | 80.0% (12/15) | **+50%** |
| **English Accuracy** | 100% (15/15) | 100% (15/15) | 0% (ceiling) |
| **Fairness Gap (EN-MI)** | 46.7% | 20.0% | **-57.1%** |
| **Budget Allocation Effect** | — | **No effect** (p = 1.0) | Null result |

**Critical Insight**: Improving retrieval architecture (changing from BM25 to semantic embeddings) yielded **50% relative improvement** for Te Reo Māori, while sophisticated budget allocation provided **zero benefit** when retrieval quality was already high.

**Conclusion**: **Retrieval quality > Budget allocation**

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
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

```bash

# Phase 2A: Running BM25 Baseline Evaluation

# Terminal 1: Start orchestrator
cd services/orchestrator-ts
npm run dev
# Runs on http://localhost:8000

# Terminal 2: Start baseline BM25 Retriever 
cd services/retriever-py
python -m uvicorn app:app --port 8001

# Terminal 3: Run BM25 evaluation
# Evaluates 30 queries using BM25 retriever
# Saves results to outputs/baseline_bm25_results.csv
cd notebook
python 02a_baseline_bm25_evaluation.py

# Phase 2B: Running BM25 Baseline Evaluation
# STOP BM25 Retriever (Terminal 2 - retriver-py)
# Keep Terminal 1 orchestrator running 

# Terminal 2: Start Embeddings Retriever 
# Loads data/corpus_embeddings.pkl
# Runs semantic search using multilingual enmbeddings
cd notebook
python retriever_embeddings.py


# Terminal 3: Run evaluation
cd notebook
python 02_full_evaluation.py
# Cost: ~$0.40
```

---

## Project Structure

```
kaitiaki-planner/
├── data/                          # Knowledge base
│   ├── corpus.json                # 30 Wikipedia docs (15 EN, 15 MI)
│   ├── corpus_embeddings.pkl      # Precomputed embeddings
│   └── ATTRIBUTION.md             # CC-BY-SA license
│
├── eval/
│   └── tasks_labeled.yaml         # 30 evaluation queries
│
├── notebook/                      # Experiments & analysis
│   ├── build_embeddings.py        # Generate embeddings
│   ├── retriever_embeddings.py    # Semantic retriever service
│   ├── 02_full_evaluation.py      # Main evaluation script
│   ├── 03_analysis_and_visualisation.py  # Generate figures
│   ├── 04_statistical_test.py     # Statistical analysis
│   ├── claude_client.py           # LLM wrapper with budget tracking
│   └── eval_utils.py              # Evaluation utilities
│
├── services/
│   └── orchestrator-ts/           # Query orchestration
│       └── src/server.ts          # Budget allocation logic
│
├── outputs/                       # Results & figures
│   ├── full_evaluation_results.csv
│   ├── figures/                   # 6 visualization files
│   └── capstone_report.pdf        # 4-page technical report
│
└── README.md
```

---

## System Architecture

The system implements a three-stage RAG pipeline:

```
User Query -> Orchestrator (Budget Planning) -> Retriever (Semantic Search)
    -> Claude 3.5 Sonnet (Answer Generation) -> Grounded Answer
```

### Three Budget Allocation Strategies

| Strategy | Description | English Budget | Māori Budget |
|----------|-------------|----------------|--------------|
| **Uniform** | Equal treatment (baseline) | 5 docs | 5 docs |
| **Language-Aware** | More context for low-resource language | 5 docs | 8 docs |
| **Fairness-Aware** | Prioritize Māori + complex queries | 5 docs (simple)<br>8 docs (complex) | 8 docs |

### Retrieval Methods

**Baseline: BM25** (keyword-based)
- Fast exact matching
- Poor semantic understanding
- Results: 53.3% Māori accuracy

**Improved: Semantic Embeddings** (multilingual)
- Model: `paraphrase-multilingual-mpnet-base-v2` (768-dim)
- Cosine similarity + keyword boost (1.2×)
- Results: 80.0% Māori accuracy (+50% improvement)

---

## Evaluation

### Dataset

- **30 queries** (15 English, 15 Te Reo Māori)
- **Topics**: Māori culture, NZ geography, wildlife, history
- **Complexity**: 20 simple (factual), 10 complex (inference)

### Metric: Grounded Correctness (GC)

Binary evaluation:
```
GC = 1  if (gold_doc_id in retrieved_passages) AND (answer cites gold passage)
GC = 0  otherwise
```

### Statistical Tests

- **ANOVA** (F=0.000, p=1.0): No difference between budget conditions
- **t-test** (p=0.0719): English-Māori gap marginally non-significant
- **Cohen's d** (0.683): Medium effect size

---

## Key Results

### Impact of Retrieval Quality

Switching from BM25 to semantic embeddings:
- **+71% relative improvement** for Te Reo Māori (8/15 → 12/15 queries)
- **-57.1% reduction** in fairness gap (46.7% → 20.0%)
- **Statistically significant** (Chi-square p < 0.05)

### Null Result: Budget Allocation

Allocating 60% more budget (5 to 8 documents):
- **0% improvement** across all conditions
- **p = 1.0** (perfect equivalence)
- **Null finding**: Budget allocation provides no benefit when retrieval quality is high

---

## Ethical & Cultural Context

This project centers **Māori data sovereignty** principles from [Te Mana Raraunga](https://www.temanararaunga.maori.nz/):

- **Rangatiratanga** (Authority): Māori govern how Te Reo Māori appears in AI
- **Whakapapa** (Relationality): Knowledge exists in cultural context
- **Kaitiakitanga** (Guardianship): Computational stewardship responsibility

See `outputs/capstone_report.pdf` for full ethical analysis.

---

## Documentation

- **[Capstone Report](outputs/capstone_report.pdf)** - 4-page technical report with ethical analysis
- **[Data Attribution](data/ATTRIBUTION.md)** - CC-BY-SA license compliance
---

## Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestrator** | TypeScript + Fastify | Budget allocation logic |
| **Retriever** | Python + FastAPI | Semantic search (embeddings) |
| **LLM** | Claude 3.5 Sonnet | Answer generation (~$0.0048/query) |
| **Embeddings** | sentence-transformers | Multilingual semantic vectors |
| **Analysis** | SciPy + Matplotlib | Statistics & visualization |

---

## License

MIT License - See [LICENSE](LICENSE) file.

**Educational Use**: This project is for academic research. Māori text used with cultural sensitivity and respect. Wikipedia content used under CC-BY-SA license (see `data/ATTRIBUTION.md`).

---

## Acknowledgments

- Te Mana Raraunga for Māori data sovereignty principles
- Anthropic for Claude API access
- Wikipedia contributors for Te Reo Māori content (CC-BY-SA)

---
