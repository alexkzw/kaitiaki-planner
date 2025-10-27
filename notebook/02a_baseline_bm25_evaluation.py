"""
Baseline BM25 Evaluation
========================

This script runs the evaluation using the BM25 retriever to establish
baseline performance. Results are saved separately to preserve comparison data.

IMPORTANT:
- Make sure services/retriever-py/app.py (BM25) is running on port 8001
- Make sure orchestrator is running on port 8000
- Expected cost: ~$0.40 USD
- Results saved to: outputs/baseline_bm25_results.csv

Usage:
    1. Stop any running retriever
    2. Start BM25 retriever: cd services/retriever-py && python -m uvicorn app:app --port 8001
    3. Start orchestrator: cd services/orchestrator-ts && npm start
    4. Run this script: python 02a_baseline_bm25_evaluation.py
"""

import os
import time
import pandas as pd
import requests
from pathlib import Path
from dotenv import load_dotenv

# Import custom modules
from claude_client import ClaudeClient
from eval_utils import load_corpus, load_eval_tasks, grounded_correctness

# ============================================================================
# 1. Setup and Load Data
# ============================================================================

print("\n" + "="*70)
print("BASELINE BM25 EVALUATION")
print("="*70)
print("\nIMPORTANT: Ensure BM25 retriever (services/retriever-py/app.py)")
print("           is running on http://localhost:8001")
print("\n" + "="*70)

input("\nPress ENTER to continue (or Ctrl+C to abort)...")

print("\n1. Loading environment and data...")

# Load environment variables
load_dotenv()
api_key = os.environ.get("ANTHROPIC_API_KEY")

if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file")

# Load data
corpus = load_corpus("../data/corpus.json")
eval_tasks = load_eval_tasks(corpus, "../eval/tasks_labeled.yaml")

print(f"Loaded {len(corpus)} documents")
print(f"Loaded {len(eval_tasks)} evaluation tasks")

# Initialize Claude client
claude = ClaudeClient(api_key=api_key, max_spend_usd=5.0)
ORCH = "http://localhost:8000"

print(f"  Claude client initialized")
print(f"  Current spend: ${claude.total_cost_usd:.4f}")
print(f"  Budget remaining: ${claude.max_spend_usd - claude.total_cost_usd:.2f}")

# ============================================================================
# 2. Verify BM25 Retriever is Running
# ============================================================================

print("\n2. Verifying BM25 retriever connection...")
try:
    r = requests.get("http://localhost:8001/", timeout=5)
    if r.status_code == 200:
        info = r.json()
        print(f"  Connected to: {info.get('service', 'unknown')}")
        print(f"  Version: {info.get('version', 'unknown')}")

        # Verify it's BM25, not embeddings
        if 'embedding' in str(info).lower():
            print("\n ERROR: Embedding retriever is running, not BM25!")
            print("   Stop the embedding retriever and start BM25 retriever:")
            print("   cd services/retriever-py && python -m uvicorn app:app --port 8001")
            exit(1)
    else:
        print(f"  Unexpected status code: {r.status_code}")
        exit(1)
except Exception as e:
    print(f"  Connection failed: {e}")
    print("\nStart BM25 retriever with:")
    print("  cd services/retriever-py && python -m uvicorn app:app --port 8001")
    exit(1)

# ============================================================================
# 3. Run Full Evaluation
# ============================================================================

# Three experimental conditions
modes = ["uniform", "language_aware", "fairness_aware"]

results = []
errors = []

print(f"\n3. Running evaluation across {len(modes)} conditions...")
print(f"   Total queries: {len(eval_tasks) * len(modes)} ({len(eval_tasks)} tasks × {len(modes)} modes)")

for mode_idx, mode_name in enumerate(modes):
    print(f"\n{'='*70}")
    print(f"CONDITION {mode_idx+1}/{len(modes)}: {mode_name.upper()}")
    print(f"{'='*70}")

    for i, task in enumerate(eval_tasks):
        task_id = task['id']
        query = task['query']
        lang = task['lang']
        complexity = task['complexity']
        gold_citation = task['gold_citations'][0]  # Get first (and only) gold citation
        gold_doc_id = gold_citation['doc_id']

        print(f"\n[{i+1}/{len(eval_tasks)}] {task_id} ({lang}, {complexity})")
        print(f"  Query: {query}")
        print(f"  Gold doc: {gold_doc_id}")

        try:
            # Step 1: Call orchestrator for retrieval
            payload = {
                "query": query,
                "lang": lang,
                "complexity": complexity,
                "mode": mode_name,
                "use_rerank": True
            }

            t0 = time.time()
            r = requests.post(f"{ORCH}/query", json=payload, timeout=30)
            orch_ms = (time.time() - t0) * 1000

            if r.status_code != 200:
                error_msg = f"HTTP {r.status_code}"
                errors.append({"task_id": task_id, "error": error_msg})
                print(f"  ERROR - {error_msg}")
                continue

            data = r.json()
            passages = data.get('passages', [])

            if not passages:
                error_msg = "No passages returned"
                errors.append({"task_id": task_id, "error": error_msg})
                print(f"  ERROR - {error_msg}")
                continue

            # Extract plan details
            plan = data.get('plan', {})
            top_k = plan.get('top_k', -1)
            rerank_k = plan.get('rerank_k', -1)

            # Step 2: Generate answer with Claude
            print(f"  Generating answer (budget: ${claude.max_spend_usd - claude.total_cost_usd:.2f} remaining)...")

            response = claude.generate_answer(
                query=query,
                passages=passages[:3],  # Use top 3 passages
                max_tokens=150
            )

            answer = response['answer']
            cost = response['cost_usd']

            print(f"  Answer: {answer[:100]}...")
            print(f"  Cost: ${cost:.4f}")

            # Step 3: Evaluate with Grounded Correctness
            gc = grounded_correctness(
                response['citations'],
                gold_citation
            )

            print(f"  GC Score: {gc} ({'PASS' if gc == 1 else 'FAIL'})")

            # Record result
            results.append({
                "task_id": task_id,
                "query": query,
                "lang": lang,
                "complexity": complexity,
                "mode": mode_name,
                "gold_doc_id": gold_doc_id,
                "top_k": top_k,
                "rerank_k": rerank_k,
                "num_passages": len(passages),
                "retrieved_doc_ids": [p.get('doc_id') for p in passages],
                "answer": answer,
                "gc": gc,
                "refusal": response['refusal'],
                "cost": cost,
                "orch_ms": orch_ms,
                "input_tokens": response['usage']['input_tokens'],
                "output_tokens": response['usage']['output_tokens']
            })

        except Exception as e:
            error_msg = str(e)
            errors.append({"task_id": task_id, "mode": mode_name, "error": error_msg})
            print(f"  ERROR - {error_msg}")
            continue

# ============================================================================
# 4. Save Results
# ============================================================================

print(f"\n{'='*70}")
print("SAVING BASELINE BM25 RESULTS")
print(f"{'='*70}")

# Convert to DataFrame
df_all = pd.DataFrame(results)

# Compute summary stats
total_tasks = len(results)
total_cost = df_all['cost'].sum()

print(f"\nCompleted: {total_tasks}/{len(eval_tasks) * len(modes)} tasks")
print(f"Total cost: ${total_cost:.4f}")

if errors:
    print(f"\n{len(errors)} errors occurred:")
    for err in errors[:5]:
        print(f"  - {err}")

# Save to separate file for baseline
output_dir = Path("../outputs")
output_dir.mkdir(exist_ok=True)

baseline_path = output_dir / "baseline_bm25_results.csv"
df_all.to_csv(baseline_path, index=False)

print(f"\nBaseline BM25 results saved to: {baseline_path}")

# Print summary statistics
print(f"\n{'='*70}")
print("BASELINE BM25 PERFORMANCE SUMMARY")
print(f"{'='*70}")

for mode_name in modes:
    mode_df = df_all[df_all['mode'] == mode_name]

    print(f"\n{mode_name.upper()}:")
    print(f"  Overall: {mode_df['gc'].mean():.1%} ({mode_df['gc'].sum()}/{len(mode_df)})")

    en_df = mode_df[mode_df['lang'] == 'en']
    mi_df = mode_df[mode_df['lang'] == 'mi']

    print(f"  English: {en_df['gc'].mean():.1%} ({en_df['gc'].sum()}/{len(en_df)})")
    print(f"  Māori:   {mi_df['gc'].mean():.1%} ({mi_df['gc'].sum()}/{len(mi_df)})")

    gap = en_df['gc'].mean() - mi_df['gc'].mean()
    print(f"  Gap:     {gap:.1%} (EN - MI)")

print(f"\n{'='*70}")
print("BASELINE EVALUATION COMPLETE")
print(f"{'='*70}")
print(f"\nResults saved to: {baseline_path}")
print(f"Total cost: ${total_cost:.4f}")
print(f"\nNext steps:")
print(f"  1. Review results in {baseline_path}")
print(f"  2. Stop BM25 retriever")
print(f"  3. Start embedding retriever for potential improved results")
print(f"  4. Compare baseline vs embeddings performance")
