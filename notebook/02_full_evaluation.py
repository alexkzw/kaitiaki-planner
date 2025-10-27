"""
Full Evaluation
======================

IMPORTANT: This script performs expensive API calls (~$0.40 USD).
Run ONCE only. Results are saved to CSV for analysis in Days 3-5.

Expected time: 60-90 minutes
Expected cost: ~$0.40 USD
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
# 2. Pre-Flight Checks
# ============================================================================

print("\n2. Running pre-flight checks...")
print("="*70)

# Check 1: Orchestrator running
try:
    r = requests.get(f"{ORCH}/", timeout=5)
    if r.status_code == 200:
        print("Orchestrator running")
    else:
        raise Exception(f"Orchestrator returned {r.status_code}")
except Exception as e:
    print(f" Cannot connect to orchestrator: {e}")
    print("  Start it with: cd services/orchestrator-ts && npm run dev")
    raise

# Check 2: Budget
remaining = claude.max_spend_usd - claude.total_cost_usd
needed = len(eval_tasks) * 3 * 0.005  # Conservative estimate

if remaining >= needed:
    print(f"Budget sufficient (have ${remaining:.2f}, need ~${needed:.2f})")
else:
    print(f"   WARNING: Budget may be insufficient")
    print(f"   Have: ${remaining:.2f}, Need: ~${needed:.2f}")

# Check 3: Data loaded
if len(eval_tasks) == 30:
    print(f"All 30 tasks loaded")
else:
    print(f"WARNING: Expected 30 tasks, got {len(eval_tasks)}")

print("="*70)
print("All checks passed - ready to proceed")

# Confirmation prompt
input("\n⏸ Press Enter to start evaluation (or Ctrl+C to cancel)...")

# ============================================================================
# 3. Evaluation Function
# ============================================================================

def run_full_evaluation(tasks, mode_name, description):
    """
    Run complete evaluation for one condition.
    
    Args:
        tasks: List of all evaluation tasks
        mode_name: "uniform", "language_aware", or "fairness_aware"
        description: Human-readable description
    
    Returns:
        DataFrame with results, or None if failed
    """
    print("\n" + "="*70)
    print(f"CONDITION: {mode_name.upper()}")
    print(f"{description}")
    print("="*70)
    print(f"Queries: {len(tasks)}")
    print(f"Expected cost: ~${len(tasks) * 0.0045:.3f} USD")
    print(f"Expected time: ~{len(tasks) * 2 / 60:.0f} minutes")
    print("="*70)
    
    results = []
    start_time = time.time()
    errors = []
    
    for i, task in enumerate(tasks, 1):
        task_id = task['id']
        query = task['query']
        lang = task['lang']
        complexity = task['complexity']
        
        # Progress update every 5 queries
        if i % 5 == 0:
            elapsed = time.time() - start_time
            queries_done = len(results)
            if queries_done > 0:
                avg_time = elapsed / queries_done
                remaining_queries = len(tasks) - queries_done
                remaining_time = remaining_queries * avg_time
                
                print(f"\n[{i}/{len(tasks)}] Progress: {i/len(tasks)*100:.1f}%")
                print(f"  Completed: {queries_done}")
                print(f"  Current spend: ${claude.total_cost_usd:.4f}")
                print(f"  Est. time remaining: {remaining_time/60:.1f} min")
        
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
                print(f"  [{i}] {task_id}: ERROR - {error_msg}")
                continue
            
            data = r.json()
            passages = data.get('passages', [])
            
            if not passages:
                error_msg = "No passages returned"
                errors.append({"task_id": task_id, "error": error_msg})
                print(f"  [{i}] {task_id}: ERROR - {error_msg}")
                continue
            
            # Step 2: Generate answer with Claude
            t0_llm = time.time()
            response = claude.generate_answer(
                query=query,
                passages=passages[:3],  # Use top 3 passages
                max_tokens=150
            )
            llm_ms = (time.time() - t0_llm) * 1000
            
            # Step 3: Calculate grounded correctness
            gc = grounded_correctness(
                response['citations'],
                task['gold_citations'][0]
            )
            
            # Step 4: Store result
            result = {
                "id": task_id,
                "lang": lang,
                "complexity": complexity,
                "mode": mode_name,
                "top_k": data['plan']['top_k'],
                "gc": gc,
                "cost": response["cost_usd"],
                "refusal": response["refusal"],
                "lat_orch_ms": orch_ms,
                "lat_llm_ms": llm_ms,
                "lat_total_ms": orch_ms + llm_ms,
                "input_tokens": response["usage"]["input_tokens"],
                "output_tokens": response["usage"]["output_tokens"],
                "answer_length": len(response["answer"]),
                "num_citations": len(response["citations"])
            }
            
            results.append(result)
            
        except Exception as e:
            error_msg = str(e)[:60]
            errors.append({"task_id": task_id, "error": error_msg})
            print(f"  [{i}] {task_id}: EXCEPTION - {error_msg}")
            continue
    
    elapsed = time.time() - start_time
    
    # Print summary for this condition
    print("\n" + "="*70)
    print(f"COMPLETED: {mode_name.upper()}")
    print("="*70)
    
    if not results:
        print("No queries completed successfully")
        if errors:
            print(f"\nErrors: {len(errors)}")
            for err in errors[:5]:  # Show first 5 errors
                print(f"  - {err['task_id']}: {err['error']}")
        return None
    
    df = pd.DataFrame(results)
    
    print(f"Successful: {len(df)}/{len(tasks)} ({len(df)/len(tasks)*100:.1f}%)")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Cost: ${df['cost'].sum():.4f}")
    print(f"Mean GC: {df['gc'].mean():.3f}")
    print(f"Refusal rate: {df['refusal'].mean():.1%}")
    
    # By language breakdown
    print("\nBy language:")
    lang_summary = df.groupby('lang').agg({
        'gc': 'mean',
        'cost': 'sum'
    }).round(3)
    print(lang_summary)
    
    if errors:
        print(f"\n{len(errors)} queries failed")
    
    return df

# ============================================================================
# 4. Run All Three Conditions
# ============================================================================

# CONDITION 1: UNIFORM
print("\n" + ""*35)
print("STARTING CONDITION 1 OF 3: UNIFORM")
print(""*35)

df_uniform = run_full_evaluation(
    eval_tasks,
    "uniform",
    "Baseline: Fixed top_k=5 for all queries (no fairness consideration)"
)

# Brief pause between conditions
time.sleep(2)

# CONDITION 2: LANGUAGE-AWARE
print("\n" + ""*35)
print("STARTING CONDITION 2 OF 3: LANGUAGE-AWARE")
print(""*35)

df_language_aware = run_full_evaluation(
    eval_tasks,
    "language_aware",
    "Language-aware: top_k=8 for MI, top_k=5 for EN (language fairness)"
)

# Brief pause between conditions
time.sleep(2)

# CONDITION 3: FAIRNESS-AWARE
print("\n" + ""*35)
print("STARTING CONDITION 3 OF 3: FAIRNESS-AWARE")
print(""*35)

df_fairness_aware = run_full_evaluation(
    eval_tasks,
    "fairness_aware",
    "Fairness-aware: top_k=8 for MI OR complex (full fairness protection)"
)

# ============================================================================
# 5. Combine and Save Results
# ============================================================================

print("\n" + "="*70)
print("COMBINING RESULTS")
print("="*70)

all_dfs = []
if df_uniform is not None:
    all_dfs.append(df_uniform)
if df_language_aware is not None:
    all_dfs.append(df_language_aware)
if df_fairness_aware is not None:
    all_dfs.append(df_fairness_aware)

if not all_dfs:
    print("   No results to combine")
    print("   Evaluation failed - check errors above")
    exit(1)
else:
    df_all = pd.concat(all_dfs, ignore_index=True)
    
    print(f"  Combined results: {len(df_all)} rows")
    print(f"  Conditions: {df_all['mode'].nunique()}")
    print(f"  Unique queries: {df_all['id'].nunique()}")
    
    # Save to CSV
    output_dir = Path("../outputs")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "full_evaluation_results.csv"
    df_all.to_csv(output_path, index=False)
    
    print(f"\n Saved results to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")

# ============================================================================
# 6. Quick Summary Statistics
# ============================================================================

print("\n" + "="*70)
print("QUICK SUMMARY")
print("="*70)

print(f"\nOverall Statistics:")
print(f"  Total queries: {len(df_all)}")
print(f"  Success rate: {len(df_all)/(len(eval_tasks)*3)*100:.1f}%")
print(f"  Total cost: ${df_all['cost'].sum():.4f}")
print(f"  Mean GC: {df_all['gc'].mean():.3f}")
print(f"  Refusal rate: {df_all['refusal'].mean():.1%}")

print(f"\nPerformance by Condition:")
perf = df_all.groupby('mode')['gc'].agg(['mean', 'std', 'count']).round(3)
print(perf)

print(f"\nPerformance by Language:")
lang = df_all.groupby('lang')['gc'].agg(['mean', 'std', 'count']).round(3)
print(lang)

print(f"\nCost by Condition:")
cost = df_all.groupby('mode')['cost'].sum().round(4)
print(cost)

# ============================================================================
# 7. Final Status
# ============================================================================

print(f"\nFull evaluation complete")
print(f"Dataset ready for analysis")

print(f"\nNext steps:")
print(f"   1. Run: python run_day3.py")
print(f"   2. This will analyze full_evaluation_results.csv")
print(f"   3. No additional cost!")

print(f"\nBudget status:")
print(f"   Total spend: ${claude.total_cost_usd:.4f}")
print(f"   Budget remaining: ${claude.max_spend_usd - claude.total_cost_usd:.2f}")

print(f"\n IMPORTANT: Do NOT re-run this script!")
print(f"   Results are saved. Use run_day3.py for analysis.")

print("\n" + "="*70)
claude.print_stats()
print("="*70)
