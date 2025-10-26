# Kaitiaki-Planner: Budget-Aware RAG for Multilingual Fairness

> **A capstone research project exploring fairness in bilingual Retrieval-Augmented Generation (RAG) systems for English and Te Reo Māori**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Node](https://img.shields.io/badge/node-18+-green.svg)](https://nodejs.org/)
[![Anthropic](https://img.shields.io/badge/LLM-Claude%203.5%20Sonnet-purple.svg)](https://www.anthropic.com/)

---

## Overview

**Kaitiaki-Planner** investigates whether **dynamic budget allocation** in RAG systems can reduce performance disparities between high-resource (English) and low-resource (Te Reo Māori) languages. Grounded in the Māori concept of **kaitiakitanga** (guardianship), the system allocates computational resources—retrieval budget (top_k) and reranking budget—to protect the integrity of indigenous language interactions with AI.

### Research Questions

1. **Does improved retrieval quality (BM25 → semantic embeddings) reduce language performance gaps?**
2. **Can budget allocation strategies further improve fairness beyond retrieval quality alone?**
3. **How do fairness interventions align with Māori data sovereignty principles?**

### Key Findings

| Metric | BM25 Baseline | Embeddings | Improvement |
|--------|---------------|------------|-------------|
| **Te Reo Māori Accuracy** | 53.3% (8/15) | 80.0% (12/15) | **+50%** |
| **English Accuracy** | 100% (15/15) | 100% (15/15) | 0% (ceiling) |
| **Fairness Gap (EN-MI)** | 46.7% | 20.0% | **-57.1%** |
| **Budget Allocation Effect** | — | **No effect** (p = 1.0) | Null result |

**Critical Insight**: Improving retrieval architecture (BM25 → semantic embeddings) yielded **50% relative improvement** for Te Reo Māori queries, while sophisticated budget allocation provided **zero benefit** when retrieval quality was already high.

**Conclusion**: **Retrieval quality > Budget allocation**

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                              │
│  "He aha a Matariki?" (What is Matariki?)                       │
│  Metadata: lang=mi, complexity=simple                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              ORCHESTRATOR (TypeScript/Fastify)                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Budget Planning (3 Experimental Conditions)             │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │ • Uniform:        top_k=5 for all queries          │  │   │
│  │  │ • Language-Aware: top_k=8 for Te Reo Māori         │  │   │
│  │  │ • Fairness-Aware: top_k=8 for Māori OR complex     │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Routes:                                                         │
│  • POST /query → Full RAG pipeline                              │
│  • GET  /health → System status                                 │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                RETRIEVER (Python/FastAPI)                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Two Retrieval Modes:                                    │   │
│  │                                                           │   │
│  │  [A] BM25 (Baseline)                                     │   │
│  │      • Keyword-based matching                            │   │
│  │      • Fast but misses semantic similarity               │   │
│  │      • Good for exact term matching                      │   │
│  │                                                           │   │
│  │  [B] Semantic Embeddings (Improved)                      │   │
│  │      • Model: paraphrase-multilingual-mpnet-base-v2      │   │
│  │      • 768-dim dense vectors                             │   │
│  │      • Cosine similarity ranking                         │   │
│  │      • + Keyword Boost (1.2× for matching doc_ids)       │   │
│  │      • Hybrid approach: semantics + exact matching       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Endpoints:                                                      │
│  • POST /retrieve → Returns top_k documents                     │
│  • POST /rerank  → Re-scores and sorts documents                │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                 RETRIEVED PASSAGES (5-8 docs)                   │
│                                                                  │
│  [Passage 1] (doc_id: mi_matariki)                              │
│  Ko Matariki te ingoa Māori mō te kāhui whetū Pleiades.         │
│  Matariki is the Māori name for the Pleiades star cluster...    │
│                                                                  │
│  [Passage 2] (doc_id: mi_aotearoa)                              │
│  Kei Te Moana-nui-a-Kiwa (te Moana Pāhihi) a Aotearoa...        │
│                                                                  │
│  [Passage 3-8] Additional context documents...                  │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              CLAUDE 3.5 SONNET (Anthropic API)                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Prompt Engineering:                                     │   │
│  │  • System: "You are a grounded QA assistant..."          │   │
│  │  • Context: Numbered passages [1]-[8]                    │   │
│  │  • Instruction: "Cite passage numbers in answer"         │   │
│  │  • Temperature: 0 (deterministic)                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Budget Tracking: ~$0.0048 per query                            │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GENERATED ANSWER                           │
│                                                                  │
│  "Ko Matariki te kāhui whetū Pleiades [Passage 1]. Ka tīmata    │
│   te tau hou Māori i te puta ake o Matariki i Pipiri,          │
│   Hōngongoi rānei [Passage 1]."                                 │
│                                                                  │
│  Translation: "Matariki is the Pleiades star cluster [Passage   │
│  1]. The Māori new year begins when Matariki rises in late      │
│  June or early July [Passage 1]."                               │
│                                                                  │
│  Grounded Correctness (GC) = 1                               │
│  (Gold doc 'mi_matariki' retrieved AND cited)                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
kaitiaki-planner/
├── data/                                  # Knowledge base
│   ├── corpus.json                        # 30 Wikipedia docs (15 EN, 15 MI)
│   ├── corpus_embeddings.pkl              # Precomputed 768-dim embeddings
│   ├── corpus_raw/                        # Original Wikipedia extracts
│   └── ATTRIBUTION.md                     # CC-BY-SA license compliance
│
├── eval/
│   └── tasks_labeled.yaml                 # 30 evaluation queries (15 EN, 15 MI)
│
├── notebook/                              # Experiments & analysis
│   ├── 01_setup_and_testing.py            # System verification
│   ├── 02a_baseline_bm25_evaluation.py    # BM25 baseline experiment
│   ├── 02_full_evaluation.py              # Embeddings experiment (3 conditions)
│   ├── 03_analysis_and_visualisation.py   # Generate 6 figures
│   ├── 04_statistical_test.py             # ANOVA, t-tests, effect sizes
│   ├── 05_baseline_comparison.py          # BM25 vs Embeddings comparison
│   ├── build_embeddings.py                # Embedding precomputation
│   ├── retriever_embeddings.py            # Semantic retriever service
│   ├── claude_client.py                   # LLM API wrapper with budget tracking
│   ├── eval_utils.py                      # Evaluation utilities
│   └── analysis_utils.py                  # Statistical analysis helpers
│
├── services/
│   ├── orchestrator-ts/                   # Query orchestration service
│   │   ├── src/server.ts                  # Fastify server + budget logic
│   │   ├── package.json                   # Node.js dependencies
│   │   └── tsconfig.json                  # TypeScript configuration
│   └── retriever-py/                      # BM25 baseline retriever
│       ├── app.py                         # FastAPI server (keyword retrieval)
│       └── requirements.txt               # Python dependencies
│
├── outputs/                                         # Experimental results
│   ├── CSV Results (20 files)
│   │   ├── baseline_bm25_results.csv                # BM25 evaluation (30 queries)
│   │   ├── embeddings_results.csv                   # Embeddings evaluation (30 queries)
│   │   ├── full_evaluation_results.csv              # All 3 conditions (90 queries)
│   │   ├── baseline_vs_embeddings_comparison.csv    # Overall comparison table
│   │   ├── query_level_improvements.csv             # Per-query deltas
│   │   ├── statistical_tests.csv                    # T-tests (EN vs MI)
│   │   ├── anova_results.csv                        # Null result (p=1.0)
│   │   ├── baseline_comparison.csv                  # BM25 vs Embeddings stats
│   │   ├── confidence_intervals.csv                 # 95% CIs
│   │   ├── effect_sizes.csv                         # Cohen's d values
│   │   ├── equivalence_tests.csv                    # Condition equivalence
│   │   ├── pairwise_comparisons.csv                 # Post-hoc tests
│   │   ├── mannwhitney_tests.csv                    # Non-parametric tests
│   │   ├── fairness_gaps.csv                        # EN-MI gaps
│   │   ├── performance_by_slice.csv                 # Breakdown by complexity
│   │   ├── summary_by_condition.csv                 # Aggregate metrics
│   │   ├── key_findings.csv                         # Top-level results
│   │   └── cost_effectiveness.csv                   # Budget analysis
│   │
│   ├── Figures (6 PNG files)
│   │   ├── before_after_comparison.png              # BM25 vs Embeddings impact
│   │   ├── fairness_gaps_chart.png                  # EN-MI gaps across conditions
│   │   ├── null_result_identical_conditions.png     # Budget allocation null result
│   │   ├── gc_by_condition_language.png             # Performance matrix
│   │   ├── gc_by_complexity.png                     # Simple vs complex queries
│   │   └── query_success_matrix.png                 # Query-level heatmap
│   │
│   └── Reports
│       ├── capstone_report.pdf                      # 4-page report
│
├── scripts/
│   └── build_corpus.py                    # Corpus construction tool
│
├── Configuration
│   ├── .env                               # API keys (ANTHROPIC_API_KEY)
│   ├── .env.example                       # Template for API configuration
│   └── .gitignore                         # Excludes .env, *.pkl, node_modules
│
├── Documentation
│   ├── README.md                          
│   └── LICENSE                            # MIT License
│
└── Dependencies
    ├── package.json                       # Root Node.js config
    └── package-lock.json                  # Locked versions
```

**Note**: Python dependencies are managed per-service (see `/services/retriever-py/requirements.txt`) and via inline imports in notebook scripts.

---

## Quick Start

### Prerequisites

- **Python 3.11+** with pip
- **Node.js 18+** with npm
- **Anthropic API Key** ([Sign up](https://console.anthropic.com/))

### Installation

```bash
# 1. Clone repository
git clone <https://github.com/alexkzw/kaitiaki-planner>
cd kaitiaki-planner

# 2. Set up Python environment
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install Python dependencies for notebooks
pip install anthropic sentence-transformers scipy pandas numpy matplotlib seaborn pyyaml python-dotenv

# 3. Set up Node.js environment (orchestrator)
cd services/orchestrator-ts
npm install
cd ../..

# 4. Install retriever dependencies
cd services/retriever-py
pip install -r requirements.txt
cd ../..

# 5. Configure API key
cp .env.example .env
# Edit .env and add API key
```

### One-Time Setup: Build Embeddings

```bash
cd notebook
python build_embeddings.py
```

**Output**: `data/corpus_embeddings.pkl` 
**What it does**: Precomputes 768-dim vectors for all 68 corpus documents using `paraphrase-multilingual-mpnet-base-v2`

---

## Running Experiments

### Experiment 1: BM25 Baseline Evaluation

**Purpose**: Establish baseline performance with traditional keyword-based retrieval.

```bash
# Terminal 1: Start orchestrator
cd services/orchestrator-ts
npm run dev
# Server running on http://localhost:8000

# Terminal 2: Start BM25 retriever
cd services/retriever-py
python -m uvicorn app:app --port 8001
# BM25 retriever on http://localhost:8001

# Terminal 3: Run evaluation
cd notebook
python 02a_baseline_bm25_evaluation.py
```

**Evaluation Details**:
- **Queries**: 30 (15 EN + 15 MI) × 3 conditions = **90 total**
- **Cost**: ~$0.40 USD (Claude API)
- **Output**: `outputs/baseline_bm25_results.csv`

**Expected Results**:
```
Overall:       76.7% (23/30)
English:      100.0% (15/15) 
Te Reo Māori:  53.3% (8/15) 
Fairness Gap:  46.7%
```

---

### Experiment 2: Semantic Embeddings Evaluation

**Purpose**: Test improved retrieval with multilingual semantic embeddings + keyword boost.

```bash
# Stop BM25 retriever (Ctrl+C in Terminal 2)

# Terminal 2: Start embeddings retriever
cd notebook
python retriever_embeddings.py
# Embeddings retriever on http://localhost:8001

# Terminal 3: Run evaluation
python 02_full_evaluation.py
```

**Evaluation Details**:
- **Queries**: 30 (15 EN + 15 MI) × 3 conditions = **90 total**
- **Cost**: ~$0.40 USD
- **Output**: `outputs/full_evaluation_results.csv`

**Expected Results**:
```
Overall:       90.0% (27/30)
English:      100.0% (15/15) 
Te Reo Māori:  80.0% (12/15) +50% improvement
Fairness Gap:  20.0% (-57.1% reduction)
```

---

### Analysis & Visualization

```bash
cd notebook

# 1. Compare BM25 vs Embeddings
python 05_baseline_comparison.py
# Outputs:
#   - baseline_vs_embeddings_comparison.csv
#   - query_level_improvements.csv
# Shows: 7 queries improved, 3 degraded, 20 unchanged

# 2. Generate all figures (6 PNG files)
python 03_analysis_and_visualisation.py
# Outputs:
#   - before_after_comparison.png          (BM25 vs Embeddings)
#   - fairness_gaps_chart.png              (EN-MI gaps)
#   - null_result_identical_conditions.png (Budget allocation)
#   - gc_by_condition_language.png         (Performance matrix)
#   - gc_by_complexity.png                 (Simple vs complex)
#   - query_success_matrix.png             (Query-level heatmap)

# 3. Statistical significance tests
python 04_statistical_test.py
# Outputs: 8 CSV files
#   - statistical_tests.csv      (T-tests: EN vs MI)
#   - anova_results.csv           (Null result: p=1.0)
#   - baseline_comparison.csv     (BM25 vs Embeddings)
#   - confidence_intervals.csv    (95% CIs)
#   - effect_sizes.csv            (Cohen's d values)
#   - equivalence_tests.csv       (Condition equivalence proof)
#   - pairwise_comparisons.csv    (Post-hoc tests)
#   - mannwhitney_tests.csv       (Non-parametric validation)
```

---

## Key Results

### 1. Impact of Retrieval Quality (BM25 → Embeddings)

| Metric | BM25 | Embeddings | Absolute Diff | Relative Diff |
|--------|------|------------|------------|------------|
| **Overall** | 76.7% | 90.0% | +13.3% | +17.4% |
| **English** | 100% | 100% | 0% | 0% (ceiling) |
| **Te Reo Māori** | **53.3%** | **80.0%** | **+26.7%** | **+50.0%** |
| **Fairness Gap** | 46.7% | 20.0% | -26.7% | **-57.1%** |

**Statistical Validation**:
- Chi-square test: χ² = 2.30, p = 0.13 (underpowered but substantial)
- Effect size: Cohen's d = 0.68 (medium-to-large)
- Net gain: +4 queries (7 improved, 3 degraded)

### 2. Null Result: Budget Allocation Has No Effect

**One-Way ANOVA**: F = 0.000, **p = 1.000** (all conditions identical)

| Condition | Overall | English | Te Reo Māori | Fairness Gap |
|-----------|---------|---------|--------------|--------------|
| Uniform (baseline) | 90.0% | 100% | 80.0% | 20.0% |
| Language-Aware | 90.0% | 100% | 80.0% | 20.0% |
| Fairness-Aware | 90.0% | 100% | 80.0% | 20.0% |

**Interpretation**: When retrieval quality is high (semantic embeddings), allocating 60% more budget (5→8 documents) provides **zero improvement**. The first 5 documents already contain the answer.

### 3. Query-Level Changes (BM25 -> Embeddings)

| Change Type | Count | Example Queries |
|-------------|-------|-----------------|
| **Improved (0→1)** | 7 | mi_matariki, mi_marae, mi_kauri, mi_kakapo, mi_aotearoa |
| **Degraded (1→0)** | 3 | mi_kaka, mi_taupo, mi_reo_maori |
| **Unchanged** | 20 | All 15 English + 5 Māori queries |

**Why degradation occurred**: Semantic embeddings can miss exact terminological matches that BM25 captures (e.g., scientific names like "Nestor meridionalis"). A **hybrid approach** combining both could eliminate these failures.

### 4. Remaining Failures

**3 Te Reo Māori queries still fail** with embeddings:
1. `mi_matariki_q1`: Cultural knowledge not in corpus
2. `mi_marae_q1`: Definition missing from retrieved passages
3. `mi_kauri_q1`: Scientific name not present in documents

**Root cause**: Document coverage gaps, not retrieval/budget issues.

---

## Evaluation Methodology

### Corpus

- **30 Wikipedia documents** (15 English, 15 Te Reo Māori)
- **Topics**: Aotearoa/New Zealand geography, Māori culture, flora, fauna
- **Examples**: Tongariro, Matariki, Kauri, Kākā, Treaty of Waitangi
- **License**: CC-BY-SA (see `data/ATTRIBUTION.md`)

### Queries

- **30 evaluation queries** (15 per language)
- **Complexity**: 20 simple (factual), 10 complex (inference)
- **Parallel translations**: EN ↔ MI
- **Examples**:
  - `[EN] What act gave Māori language official status in New Zealand?`
  - `[MI] He aha a Matariki?` (What is Matariki?)
  - `[EN] In what year did the kea receive absolute protection?`
  - `[MI] Kei hea a Aotearoa?` (Where is Aotearoa?)

### Metric: Grounded Correctness (GC)

**Binary evaluation**:
```
GC = 1  if (gold_doc_id in retrieved_passages) AND (answer cites gold passage)
GC = 0  otherwise
```

**Strengths**:
- Focuses on retrieval quality (did we find the right document?)
- Verifies grounding (is answer supported by evidence?)

**Limitations**:
- Binary (doesn't capture partial correctness)
- Doesn't measure answer fluency or cultural appropriateness

### Budget Allocation Strategies

| Strategy | Description | English Budget | Māori Budget |
|----------|-------------|----------------|--------------|
| **Uniform** | Equal treatment (baseline) | 5 docs | 5 docs |
| **Language-Aware** | More context for low-resource language | 5 docs | 8 docs |
| **Fairness-Aware** | Prioritize Māori + complex queries | 5 docs (simple)<br>8 docs (complex) | 8 docs |

---

## Ethical & Cultural Context

This project centers **Māori data sovereignty** principles from [Te Mana Raraunga](https://www.temanararaunga.maori.nz/):

### Core Principles

1. **Rangatiratanga** (Authority)
   - Māori communities should govern how Te Reo Māori appears in AI systems
   - Recognition that language data is culturally significant, not neutral

2. **Whakapapa** (Relationality)
   - Knowledge exists in cultural context, not as isolated facts
   - Wikipedia extracts preserve cultural framing where possible

3. **Kaitiakitanga** (Guardianship)
   - Computational systems have stewardship responsibilities
   - Budget allocation as a form of computational guardianship

### Fairness Frameworks

| Framework | Implementation | Rationale |
|-----------|----------------|-----------|
| **Utilitarian Equality** | Uniform condition | Maximise aggregate welfare |
| **Rawlsian Equity** | Language-Aware condition | Prioritise least-advantaged group |
| **Capabilities Approach** | Fairness-Aware condition | Equalize capability to achieve outcomes |

See `outputs/capstone_report.md` for full ethical analysis.

---

## Limitations

1. **Small Sample Size**: n=15 per language limits statistical power
2. **Ceiling Effect**: English at 100% (no room for improvement detection)
3. **Binary Metric**: GC doesn't capture partial correctness or fluency
4. **Single LLM**: Results specific to Claude 3.5 Sonnet
5. **Domain-Specific**: Aotearoa-centric corpus may not generalize
6. **Cost Barrier**: $0.40 per evaluation run limits iteration
7. **Wikipedia Dependency**: Limited document diversity
8. **Cultural Appropriateness**: No native speaker validation

---

## Future Work

### Technical Improvements

1. **Hybrid Retrieval**: Combine semantic embeddings + BM25 to eliminate 3 degraded queries
2. **Cross-Lingual Retrieval**: Allow Māori queries to retrieve English docs (then translate)
3. **Query Expansion**: Automatically augment Māori queries with related terms
4. **Fine-Tuned Embeddings**: Train on Māori text corpus for better semantic representations
5. **Larger Corpus**: Expand to 500+ documents across diverse domains

### Evaluation Extensions

1. **Scale**: 100+ queries per language
2. **Human Evaluation**: Partner with native speakers for cultural appropriateness
3. **Fluency Metrics**: Add BLEU/ROUGE for answer quality
4. **A/B Testing**: Deploy with real users
5. **Other Languages**: Hawaiian, Samoan, Cook Islands Māori

### Research Questions

1. **Optimal Budget**: What if retrieval quality was lower? When does budget allocation help?
2. **Adaptive Systems**: Can LLMs self-assess and request more documents dynamically?
3. **Cultural Bias**: Does Claude exhibit cultural bias even with correct retrieval?

---

## Key References

### Academic Papers

1. **Lewis et al. (2020)**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*. [Link](https://arxiv.org/abs/2005.11401)

2. **Reimers & Gurevych (2019)**: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *EMNLP 2019*. [Link](https://arxiv.org/abs/1908.10084)

3. **Joshi et al. (2020)**: "The State and Fate of Linguistic Diversity and Inclusion in the NLP World." *ACL 2020*. [Link](https://arxiv.org/abs/2004.09095)

4. **Robertson & Zaragoza (2009)**: "The Probabilistic Relevance Framework: BM25 and Beyond." *Foundations and Trends in Information Retrieval*.

### Data Sovereignty

5. **Te Mana Raraunga (2018)**: "Principles of Māori Data Sovereignty." [Link](https://www.temanararaunga.maori.nz/)

6. **Carroll et al. (2020)**: "The CARE Principles for Indigenous Data Governance." *Data Science Journal*. [Link](https://datascience.codata.org/articles/10.5334/dsj-2020-043)

### Ethical Frameworks

7. **Sen, A. (1999)**: *Development as Freedom* (Capabilities Approach)

8. **Rawls, J. (1971)**: *A Theory of Justice* (Difference Principle)

---

## Technical Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Orchestrator** | TypeScript + Fastify | 4.26.2 | Query routing & budget allocation |
| **Retriever** | Python + FastAPI | 0.115.0 | Document retrieval (BM25 & embeddings) |
| **LLM** | Claude 3.5 Sonnet | 2024-10-22 | Answer generation |
| **Embeddings** | sentence-transformers | Latest | Multilingual semantic vectors |
| **Embedding Model** | paraphrase-multilingual-mpnet-base-v2 | 768-dim | Supports 50+ languages |
| **Statistics** | SciPy + NumPy | Latest | ANOVA, t-tests, effect sizes |
| **Visualization** | Matplotlib + Seaborn | Latest | Figure generation |
| **Data** | Pandas + PyYAML | Latest | Data processing |

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

**Educational Use**: This project is for academic research. Māori text used with cultural sensitivity and respect. Wikipedia content used under CC-BY-SA license (see `data/ATTRIBUTION.md`).

---
