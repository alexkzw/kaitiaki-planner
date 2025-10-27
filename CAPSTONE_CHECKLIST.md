# Capstone Project Checklist - Target: 10/10

## ✅ Core Requirements

### **1. Code & Implementation** ✓
- [x] Budget-aware orchestrator (TypeScript)
- [x] BM25 baseline retriever (Python)
- [x] Embeddings-based retriever (Python)
- [x] Claude integration with budget tracking
- [x] Three experimental conditions implemented
- [ ] **CRITICAL: Baseline BM25 evaluation results**
- [x] Full embeddings evaluation results
- [ ] **CRITICAL: Baseline vs embeddings comparison**

### **2. Evaluation & Analysis** ✓/⚠️
- [x] 30 labeled evaluation queries (English + Māori)
- [x] Grounded Correctness metric implementation
- [x] Statistical tests (t-tests, ANOVA, Mann-Whitney, etc.)
- [x] Visualizations (6 figures generated)
- [ ] **MISSING: Baseline comparison visualization**
- [ ] **MISSING: Query-level improvement analysis**

### **3. Documentation** ⚠️
- [x] Capstone report (4 pages) with ethical analysis
- [ ] **CRITICAL: README with reproduction instructions**
- [ ] **MISSING: Architecture diagram**
- [ ] **MISSING: Data provenance documentation**
- [ ] **MISSING: Limitations and future work section**

---

## 📋 What You MUST Do for 10/10

### **Priority 1: Empirical Evidence (CRITICAL)**

#### A. Capture Baseline BM25 Results
```bash
# 1. Preserve current embeddings results
cd outputs
cp full_evaluation_results.csv embeddings_results.csv

# 2. Start BM25 retriever
cd ../services/retriever-py
python -m uvicorn app:app --port 8001

# 3. In another terminal, run baseline evaluation
cd notebook
python 02a_baseline_bm25_evaluation.py
```

**Expected output**: `outputs/baseline_bm25_results.csv`
- English: ~100% (15/15)
- Māori: ~46.7% (7/15)
- Gap: ~53.3%

#### B. Run Comparison Analysis
```bash
python 05_baseline_comparison.py
```

**Expected outputs**:
- `outputs/baseline_vs_embeddings_comparison.csv`
- `outputs/query_level_improvements.csv`
- Terminal summary showing +71% Māori improvement

---

### **Priority 2: Comprehensive README**

Create `/Users/alex/kaitiaki-planner/README.md` with:

1. **Project Overview**
   - One-paragraph description
   - Research questions
   - Key findings summary

2. **Architecture Diagram**
   - Show: Query → Orchestrator → Retriever → Claude → Answer
   - Label three experimental conditions

3. **Reproduction Instructions**
   - Environment setup (Python 3.11, Node.js)
   - Install dependencies
   - API key configuration
   - Step-by-step to run experiments

4. **File Structure**
   ```
   kaitiaki-planner/
   ├── data/                  # Corpus (44 documents)
   ├── eval/                  # 30 labeled queries
   ├── notebook/              # Evaluation scripts
   ├── services/
   │   ├── orchestrator-ts/   # Budget allocation logic
   │   └── retriever-py/      # BM25 baseline
   ├── outputs/               # Results & figures
   └── README.md
   ```

5. **Key Results**
   - Before/after table
   - Link to figures
   - Statistical significance summary

---

### **Priority 3: Enhanced Report Sections**

#### Add to your `capstone_report.md`:

**Section 6: Limitations and Threats to Validity**
- Small sample size (n=15 per language)
- English queries achieve perfect scores (ceiling effect)
- Corpus limited to 44 documents (2 per topic)
- Binary GC metric (doesn't capture partial correctness)
- Claude as both generator and evaluator (potential bias)
- Aotearoa-centric topics (limited generalizability)

**Section 7: Future Work**
- Expand to 100+ queries across diverse domains
- Test with other low-resource languages (Hawaiian, Samoan, Cook Islands Māori)
- Implement cross-lingual retrieval (query in Māori, retrieve English docs)
- Human evaluation with native Māori speakers
- Production deployment with real users
- Query expansion techniques for Māori queries
- Fine-tune embeddings on Māori text

---

### **Priority 4: Data Provenance**

Create `data/DATA_PROVENANCE.md`:

```markdown
# Data Provenance

## Corpus Sources
All 44 documents derived from Wikipedia (accessed October 2024):
- English: https://en.wikipedia.org/
- Māori: https://mi.wikipedia.org/

## Document Selection Criteria
- Topics relevant to Aotearoa/New Zealand culture
- Parallel English-Māori coverage where available
- Verified factual accuracy (Wikipedia reliability)

## Evaluation Queries
Created by project author (Alex) October 2024.
- 15 English queries (10 simple, 5 complex)
- 15 Māori queries (11 simple, 4 complex)
- Gold standard annotations based on corpus content

## Ethical Considerations
- Wikipedia content under CC-BY-SA license
- Māori text used with cultural sensitivity
- No personally identifiable information
- Educational use only
```

---

### **Priority 5: Code Comments & Documentation**

Add docstrings to ALL Python scripts:

**Example for `retriever_embeddings.py`:**
```python
"""
Embedding-Based Retriever Service
=================================

Replaces BM25 with semantic search using multilingual sentence transformers.

Key Features:
- Model: paraphrase-multilingual-mpnet-base-v2
- Keyword boosting (30%) for doc_id matches
- Compatible with orchestrator /retrieve and /rerank endpoints

Performance:
- Māori queries: 80% (vs 46.7% with BM25)
- English queries: 100%

Usage:
    python retriever_embeddings.py
    # Runs on http://localhost:8001

Author: Alex
Date: October 2025
"""
```

---

### **Priority 6: Reproducibility Package**

Create `REPRODUCIBILITY.md`:

```markdown
# Reproducibility Instructions

## Hardware Requirements
- Processor: Any modern CPU
- RAM: 8GB minimum (16GB recommended for embeddings)
- Disk: 2GB free space

## Software Requirements
- Python 3.11+
- Node.js 18+
- Anthropic API key (set in `.env`)

## Step-by-Step Reproduction

### 1. Environment Setup
\`\`\`bash
git clone <your-repo>
cd kaitiaki-planner

# Python setup
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Node.js setup (orchestrator)
cd services/orchestrator-ts
npm install
cd ../..
\`\`\`

### 2. Configuration
\`\`\`bash
# Create .env file
echo "ANTHROPIC_API_KEY=your-key-here" > .env
\`\`\`

### 3. Run Baseline Evaluation (BM25)
\`\`\`bash
# Terminal 1: Start orchestrator
cd services/orchestrator-ts
npm start

# Terminal 2: Start BM25 retriever
cd services/retriever-py
python -m uvicorn app:app --port 8001

# Terminal 3: Run evaluation (~$0.40, 60-90 min)
cd notebook
python 02a_baseline_bm25_evaluation.py
\`\`\`

**Expected output**: `outputs/baseline_bm25_results.csv`

### 4. Run Embeddings Evaluation
\`\`\`bash
# Stop BM25 retriever (Ctrl+C in Terminal 2)

# Terminal 2: Start embeddings retriever
cd notebook
python retriever_embeddings.py

# Terminal 3: Run evaluation (~$0.40, 60-90 min)
python 02_full_evaluation.py
\`\`\`

**Expected output**: `outputs/full_evaluation_results.csv`

### 5. Analysis & Visualization
\`\`\`bash
python 03_analysis_and_visualisation.py  # Generates 6 figures
python 04_statistical_test.py            # Generates 8 CSVs
python 05_baseline_comparison.py         # Compares BM25 vs embeddings
\`\`\`

### 6. Expected Total Cost
- Baseline evaluation: ~$0.40 USD
- Embeddings evaluation: ~$0.40 USD
- Statistical tests: $0.00 (no API calls)
- **Total: ~$0.80 USD**

### 7. Expected Results
| Metric | BM25 | Embeddings | Improvement |
|--------|------|------------|-------------|
| Māori accuracy | 46.7% | 80.0% | +71% |
| Fairness gap | 53.3% | 20.0% | -62% |

## Troubleshooting
- Port conflicts: Change ports in scripts
- API rate limits: Add delays between requests
- Memory issues: Reduce batch size in embeddings
```

---

### **Priority 7: Visualization Enhancements**

Update `03_analysis_and_visualisation.py` to include:

1. **Baseline Comparison Figure** (add to existing script)
2. **Architecture Diagram** (create separately)
3. **Query Success Heatmap** (already exists ✓)

---

## 📊 Final Deliverables Checklist

### Files to Submit
- [ ] `capstone_report.md` (4 pages, updated with limitations + future work)
- [ ] `README.md` (comprehensive, with architecture diagram)
- [ ] `REPRODUCIBILITY.md` (step-by-step instructions)
- [ ] `data/DATA_PROVENANCE.md` (source attribution)
- [ ] `CAPSTONE_CHECKLIST.md` (this file, checked off)

### Data Files
- [ ] `outputs/baseline_bm25_results.csv` ← **CRITICAL: Run this!**
- [ ] `outputs/embeddings_results.csv` ← **CRITICAL: Rename existing**
- [ ] `outputs/full_evaluation_results.csv` (latest)
- [ ] `outputs/baseline_vs_embeddings_comparison.csv` ← **CRITICAL: Run comparison**
- [ ] `outputs/query_level_improvements.csv` ← **CRITICAL: Run comparison**
- [ ] All 8 statistical test CSVs ✓
- [ ] All 6 figure PNGs ✓

### Code Files (All Present ✓)
- [ ] All scripts documented with docstrings
- [ ] `requirements.txt` up to date
- [ ] `package.json` for orchestrator
- [ ] All test scripts functional

---

## 🎯 Scoring Rubric Alignment

### Technical Excellence (30%)
- [x] Working system with three conditions
- [x] Proper evaluation methodology
- [ ] **NEED: Baseline results for comparison**
- [x] Statistical rigor

### Ethical & Cultural Analysis (25%)
- [x] Tikanga Māori integration (kaitiakitanga)
- [x] Māori data sovereignty (Te Mana Raraunga)
- [x] Multiple ethical frameworks (Rawls, Sen, etc.)
- [x] Machine consciousness discussion
- [x] Cross-cultural implications

### Critical Thinking (20%)
- [x] Null result interpretation (quality > allocation)
- [x] Limitations acknowledged
- [ ] **NEED: Future work section**
- [x] Multiple perspectives synthesized

### Documentation & Reproducibility (15%)
- [ ] **NEED: Comprehensive README**
- [ ] **NEED: Reproducibility guide**
- [ ] **NEED: Data provenance**
- [x] Code comments

### Presentation & Writing (10%)
- [x] Clear, academic writing
- [x] Proper structure
- [x] Citations formatted
- [ ] **NICE TO HAVE: Visual architecture diagram**

---

## ⏱️ Time Estimates

| Task | Time | Cost |
|------|------|------|
| Run baseline BM25 eval | 90 min | $0.40 |
| Run comparison script | 5 min | $0.00 |
| Write README | 60 min | $0.00 |
| Write REPRODUCIBILITY | 30 min | $0.00 |
| Add limitations/future work to report | 45 min | $0.00 |
| Add code docstrings | 60 min | $0.00 |
| Create architecture diagram | 30 min | $0.00 |
| Final review & polish | 60 min | $0.00 |
| **TOTAL** | **6 hours** | **$0.40** |

---

## 🚀 Recommended Order

1. **TODAY**: Run baseline BM25 evaluation (90 min + $0.40) ← **MUST DO**
2. **TODAY**: Run comparison script (5 min)
3. **TOMORROW**: Write README (60 min)
4. **TOMORROW**: Update report with limitations + future work (45 min)
5. **TOMORROW**: Write reproducibility guide (30 min)
6. **BEFORE SUBMISSION**: Add code docstrings (60 min)
7. **BEFORE SUBMISSION**: Final review (60 min)

---

## ✨ Bonus Points (Optional, for 10/10)

- [ ] Create visual architecture diagram (draw.io, Lucidchart, or code)
- [ ] Add unit tests for key functions
- [ ] Create requirements-dev.txt with testing dependencies
- [ ] Add GitHub Actions CI/CD (if submitting via GitHub)
- [ ] Record 5-min demo video walking through results
- [ ] Create poster summarizing findings (A4 one-pager)

---

## 📝 Final Pre-Submission Checklist

**Day Before Submission**:
- [ ] All baseline results captured
- [ ] All comparison analyses run
- [ ] README complete and tested by fresh eyes
- [ ] Report polished (spell-check, grammar, citations)
- [ ] All code runs without errors
- [ ] No hardcoded paths (use relative paths)
- [ ] No API keys committed to git
- [ ] .gitignore updated (.env, .venv, __pycache__, etc.)

**Submission Day**:
- [ ] Create clean archive: `tar -czf kaitiaki-planner.tar.gz kaitiaki-planner/`
- [ ] Test extraction and reproduction on clean machine (if possible)
- [ ] Final git commit with meaningful message
- [ ] Backup all files to external drive
- [ ] Submit via required platform

---

**You're aiming for 10/10. The difference between 8/10 and 10/10 is:**
1. **Empirical rigor** (baseline results)
2. **Reproducibility** (clear documentation)
3. **Limitations** (honest self-critique)
4. **Polish** (no rough edges)

**Current status**: ~7.5/10 (excellent work, but missing baseline comparison)
**With checklist complete**: 10/10 ✓
