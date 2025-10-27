#!/usr/bin/env python3
"""
Setup and Testing
========================
"""

import os
import time
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from dotenv import load_dotenv

# ============================================================================
# 1. Setup and Imports
# ============================================================================

print("\n1. Loading environment...")
load_dotenv()
api_key = os.environ.get("ANTHROPIC_API_KEY")

if not api_key:
    raise ValueError(
        "ANTHROPIC_API_KEY not found!\n"
        "   1. Create .env file in project root\n"
        "   2. Add: ANTHROPIC_API_KEY=sk-ant-your-key\n"
        "   3. Restart script"
    )

print("API key loaded successfully")

# Import custom modules
from claude_client import ClaudeClient
from eval_utils import (
    load_corpus,
    load_eval_tasks,
    grounded_correctness,
)

print("Custom modules imported")

# ============================================================================
# 2. Initialize Claude Client
# ============================================================================

print("\n2. Initialising Claude client...")
claude = ClaudeClient(api_key=api_key, max_spend_usd=5.0)

print(f"   Claude client initialised")
print(f"   Model: {claude.model}")
print(f"   Budget: ${claude.max_spend_usd:.2f} USD")
print(f"   Current spend: ${claude.total_cost_usd:.4f} USD")

# ============================================================================
# 3. Load Data
# ============================================================================

print("\n3. Loading data...")
corpus = load_corpus("../data/corpus.json")
print(f"   Loaded corpus: {len(corpus)} documents")
print(f"   English: {sum(1 for d in corpus if d['lang']=='en')}")
print(f"   Māori:   {sum(1 for d in corpus if d['lang']=='mi')}")

eval_tasks = load_eval_tasks(corpus, "../eval/tasks_labeled.yaml")
print(f"   Loaded evaluation tasks: {len(eval_tasks)} tasks")

# Show distribution
df_dist = pd.DataFrame(eval_tasks)
print("\nTask distribution:")
print(df_dist.groupby(['lang', 'complexity']).size().unstack(fill_value=0))

# ============================================================================
# 4. Test Orchestrator Connection
# ============================================================================

print("\n4. Testing orchestrator connection...")
ORCH = "http://localhost:8000"

try:
    r = requests.get(f"{ORCH}/", timeout=5)
    if r.status_code == 200:
        data = r.json()
        print("    Orchestrator is running")
        print(f"   Service: {data['service']}")
        print(f"   Version: {data['version']}")
        print(f"   Available modes: {data['modes']}")
    else:
        print(f"Warning: Orchestrator returned {r.status_code}")
        raise Exception(f"Orchestrator not ready: {r.status_code}")
except Exception as e:
    print(f"   ERROR: Cannot connect to orchestrator at {ORCH}")
    print(f"   Make sure orchestrator is running:")
    print(f"   cd services/orchestrator-ts && npm run dev")
    raise

# ============================================================================
# 5. Test Three Budget Allocation Conditions
# ============================================================================

print("\n5. Testing three budget allocation conditions...")
print("="*70)
print("TESTING THREE ORCHESTRATOR MODES")
print("="*70)

test_cases = [
    {"name": "EN-simple", "lang": "en", "complexity": "simple", 
     "query": "What is kauri?"},
    {"name": "EN-complex", "lang": "en", "complexity": "complex", 
     "query": "How many states make up the United States?"},
    {"name": "MI-simple", "lang": "mi", "complexity": "simple", 
     "query": "He aha te Kea?"},
    {"name": "MI-complex", "lang": "mi", "complexity": "complex", 
     "query": "E hia ngā wehenga o te Hononga-o-Amerika?"},
]

results_summary = []

for test_case in test_cases:
    print(f"\n{test_case['name'].upper()}: lang={test_case['lang']}, complexity={test_case['complexity']}")
    print("-" * 70)
    
    for mode in ["uniform", "language_aware", "fairness_aware"]:
        payload = {
            "query": test_case["query"],
            "lang": test_case["lang"],
            "complexity": test_case["complexity"],
            "mode": mode,
            "use_rerank": True
        }
        
        try:
            r = requests.post(f"{ORCH}/query", json=payload, timeout=10)
            
            if r.status_code == 200:
                data = r.json()
                top_k = data['plan']['top_k']
                print(f"  {mode:20s}: top_k={top_k}")
                
                results_summary.append({
                    "test": test_case['name'],
                    "mode": mode,
                    "top_k": top_k
                })
            else:
                print(f"  {mode:20s}: ERROR HTTP {r.status_code}")
                
        except Exception as e:
            print(f"  {mode:20s}: ERROR {str(e)[:40]}")

# Verification
print("\n" + "="*70)
print("VERIFICATION")
print("="*70)

df_results = pd.DataFrame(results_summary)
if not df_results.empty:
    pivot = df_results.pivot(index='test', columns='mode', values='top_k')
    print("\nTop-k values by test case and mode:")
    print(pivot)
    
    print("\nExpected patterns:")
    print("  Uniform: Always 5")
    print("  Language-aware: 5 for EN, 8 for MI")
    print("  Fairness-aware: 5 for EN-simple, 8 for others")
    
    test_passed = True
else:
    print("No results collected")
    test_passed = False

# ============================================================================
# 6. Run Pilot Evaluation
# ============================================================================

print("\n6. Running pilot evaluation...")
print("="*70)
print("PILOT EVALUATION: 3 queries × 3 conditions")
print("="*70)
print("Expected cost: ~$0.04 USD")
print("="*70)

modes = ["uniform", "language_aware", "fairness_aware"]
pilot_tasks = eval_tasks[:3]  # First 3 queries
results = []

start_time = time.time()

for mode in modes:
    print(f"\n--- Mode: {mode.upper()} ---")
    
    for i, task in enumerate(pilot_tasks, 1):
        task_id = task['id']
        query = task['query']
        lang = task['lang']
        complexity = task['complexity']
        
        print(f"  [{i}/3] {task_id}... ", end="", flush=True)
        
        try:
            # Call orchestrator
            payload = {
                "query": query,
                "lang": lang,
                "complexity": complexity,
                "mode": mode,
                "use_rerank": True
            }
            
            r = requests.post(f"{ORCH}/query", json=payload, timeout=30)
            
            if r.status_code != 200:
                print(f"ERROR (HTTP {r.status_code})")
                continue
            
            data = r.json()
            passages = data.get('passages', [])
            
            if not passages:
                print("ERROR (No passages)")
                continue
            
            # Generate answer with Claude
            response = claude.generate_answer(
                query=query,
                passages=passages[:3],
                max_tokens=150
            )
            
            # Calculate grounded correctness
            gc = grounded_correctness(
                response['citations'],
                task['gold_citations'][0]
            )
            
            results.append({
                "id": task_id,
                "lang": lang,
                "complexity": complexity,
                "mode": mode,
                "top_k": data['plan']['top_k'],
                "gc": gc,
                "cost": response["cost_usd"],
                "refusal": response["refusal"]
            })
            
            print(f"GC={gc:.2f}")
            
        except Exception as e:
            print(f"ERROR ({str(e)[:40]})")
            continue

elapsed = time.time() - start_time

# ============================================================================
# 7. Summary
# ============================================================================

print("\n" + "="*70)
print("PILOT SUMMARY")
print("="*70)

if not results:
    print("No queries completed successfully")
    df_pilot = None
else:
    df_pilot = pd.DataFrame(results)
    
    print(f"\n Completed: {len(df_pilot)}/{len(pilot_tasks)*3} queries ({len(df_pilot)/(len(pilot_tasks)*3)*100:.1f}%)")
    print(f" Time: {elapsed:.1f} seconds")
    print(f" Cost: ${df_pilot['cost'].sum():.4f} USD")
    print(f" Mean GC: {df_pilot['gc'].mean():.3f}")
    
    # By mode
    print("\nBy condition:")
    mode_summary = df_pilot.groupby('mode').agg({
        'gc': 'mean',
        'cost': 'sum',
        'top_k': lambda x: list(x.unique())
    }).round(3)
    print(mode_summary)

# ============================================================================
# 8. Save Results
# ============================================================================

if df_pilot is not None:
    output_dir = Path("../outputs")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "pilot_results_day1.csv"
    df_pilot.to_csv(output_path, index=False)
    print(f"\nSaved pilot results to: {output_path}")

# ============================================================================
# 9. Final Status
# ============================================================================

print("\n" + "="*70)
print(" Setup & Testing COMPLETE!")
print("="*70)

if df_pilot is not None:
    print(f"\nSystem verified and working")
    print(f"Pilot evaluation complete")
    print(f"Three conditions tested")
    print(f"\nBudget status:")
    print(f"  Total spend: ${claude.total_cost_usd:.4f} USD")
    print(f"  Budget remaining: ${claude.max_spend_usd - claude.total_cost_usd:.2f} USD")
    print(f"  Ready for Full Evaluation")
else:
    print("\n Pilot failed - review errors above")
    print("Fix issues before proceeding to 02_full_evaluation.py")

print("\n" + "="*70)

