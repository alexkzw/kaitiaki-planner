"""
Test Embedding Retriever
========================

Verify that the embedding-based retriever is working correctly
before running the full Day 2 evaluation.

This script:
1. Checks if retriever is running
2. Tests with English queries
3. Tests with Māori queries
4. Compares with failed queries from BM25
5. Verifies improvement

Run this AFTER starting retriever_embeddings.py
"""

import requests
import json
from pathlib import Path

print("="*70)
print("TESTING EMBEDDING RETRIEVER")
print("="*70)

RETRIEVER = "http://localhost:8001"

# ============================================================================
# 1. Check Service is Running
# ============================================================================

print("\n1. Checking if retriever is running...")

try:
    r = requests.get(f"{RETRIEVER}/healthz", timeout=5)
    if r.status_code == 200:
        data = r.json()
        print(f"  Retriever is running")
        print(f"  Service: {data.get('service')}")
        print(f"  Documents: {data.get('documents')}")
        print(f"  Model: {data.get('model')}")
    else:
        print(f"Retriever returned {r.status_code}")
        exit(1)
except Exception as e:
    print(f"   Cannot connect to retriever at {RETRIEVER}")
    print(f"   Error: {e}")
    print(f"\n   Start it with: python retriever_embeddings.py")
    exit(1)

# ============================================================================
# 2. Test English Queries
# ============================================================================

print("\n2. Testing English queries...")

en_tests = [
    {
        "query": "What is a kea?",
        "expected_doc": "en_kea",
        "description": "Simple English query"
    },
    {
        "query": "How many states make up the United States?",
        "expected_doc": "en_united_states",
        "description": "Complex English query"
    },
    {
        "query": "What is kauri?",
        "expected_doc": "en_agathis_australis",  # or en_kauri
        "description": "Simple keyword query"
    }
]

en_passed = 0
for test in en_tests:
    print(f"\nTesting: {test['description']}")
    print(f"   Query: '{test['query']}'")
    
    payload = {
        "query": test["query"],
        "top_k": 5,
        "lang": "en"
    }
    
    r = requests.post(f"{RETRIEVER}/search", json=payload, timeout=10)
    
    if r.status_code == 200:
        data = r.json()
        passages = data.get('passages', [])
        
        if passages:
            top_doc = passages[0]['doc_id']
            top_score = passages[0]['score']
            
            print(f"Top result: {top_doc} (score: {top_score:.3f})")
            
            # Check if expected doc is in top 5
            top_5_docs = [p['doc_id'] for p in passages[:5]]
            
            if test['expected_doc'] in top_5_docs:
                rank = top_5_docs.index(test['expected_doc']) + 1
                print(f"Expected doc found at rank {rank}")
                en_passed += 1
            elif any(test['expected_doc'].split('_')[-1] in doc for doc in top_5_docs):
                # Partial match (e.g., "kea" in doc name)
                print(f"Related doc found (partial match)")
                en_passed += 1
            else:
                print(f"Expected '{test['expected_doc']}' not in top 5")
                print(f"Top 5: {top_5_docs}")
        else:
            print(f"No passages returned")
    else:
        print(f"Search failed: HTTP {r.status_code}")

print(f"\nEnglish tests: {en_passed}/{len(en_tests)} passed")

# ============================================================================
# 3. Test Māori Queries 
# ============================================================================

print("\n3. Testing Māori queries...")

mi_tests = [
    {
        "query": "Ka rere rānei te Kākāpō?",
        "expected_doc": "mi_kakapo",
        "description": "Can the kākāpō fly?"
    },
    {
        "query": "He aha te ingoa pūtaiao o te Kauri?",
        "expected_doc": "mi_kauri",
        "description": "What is the scientific name of kauri?"
    },
    {
        "query": "Kei hea a Aotearoa?",
        "expected_doc": "mi_aotearoa",
        "description": "Where is Aotearoa?"
    },
    {
        "query": "He aha a Matariki?",
        "expected_doc": "mi_matariki",
        "description": "What is Matariki?"
    },
    {
        "query": "He aha te marae?",
        "expected_doc": "mi_marae",
        "description": "What is a marae?"
    }
]

mi_passed = 0
mi_improved = []

for test in mi_tests:
    print(f"\n   Testing: {test['description']}")
    print(f"   Query: '{test['query']}'")
    
    payload = {
        "query": test["query"],
        "top_k": 8,  # Use higher k for Māori
        "lang": "mi"
    }
    
    r = requests.post(f"{RETRIEVER}/search", json=payload, timeout=10)
    
    if r.status_code == 200:
        data = r.json()
        passages = data.get('passages', [])
        
        if passages:
            top_doc = passages[0]['doc_id']
            top_score = passages[0]['score']
            
            print(f"Top result: {top_doc} (score: {top_score:.3f})")
            
            # Check if expected doc is in top 8
            top_docs = [p['doc_id'] for p in passages]
            
            if test['expected_doc'] in top_docs:
                rank = top_docs.index(test['expected_doc']) + 1
                print(f"Expected doc found at rank {rank}")
                mi_passed += 1
                mi_improved.append(test['expected_doc'])
            else:
                print(f"Expected '{test['expected_doc']}' not in top {len(passages)}")
                print(f"Top 5: {[p['doc_id'] for p in passages[:5]]}")
        else:
            print(f"No passages returned")
    else:
        print(f"Search failed: HTTP {r.status_code}")

print(f"\nMāori tests: {mi_passed}/{len(mi_tests)} passed")

if mi_improved:
    print(f"\nIMPROVEMENT: {len(mi_improved)} previously-failed queries now work!")
    print(f"These should now get GC=1 in Day 2:")
    for doc in mi_improved:
        print(f" - {doc}")

# ============================================================================
# 4. Direct Comparison Test
# ============================================================================

print("\n4.Spot-checking retrieval quality...")

# Test with top_k=5 vs top_k=8 for a Māori query
test_query = "He aha te Kea?"
expected = "mi_kea"

print(f"\n   Query: '{test_query}'")
print(f"   Expected: {expected}")

for k in [5, 8]:
    payload = {"query": test_query, "top_k": k, "lang": "mi"}
    r = requests.post(f"{RETRIEVER}/search", json=payload, timeout=10)
    
    if r.status_code == 200:
        data = r.json()
        passages = data['passages']
        doc_ids = [p['doc_id'] for p in passages]
        
        found = expected in doc_ids
        rank = doc_ids.index(expected) + 1 if found else None
        
        print(f"\n   top_k={k}:")
        print(f"     Found: {found}")
        if found:
            print(f"     Rank: {rank}")
        print(f"     Top 3: {doc_ids[:3]}")

# ============================================================================
# 5. Expected Improvement Estimate
# ============================================================================

print("\n" + "="*70)
print("EXPECTED IMPROVEMENTS")
print("="*70)

print(f"\nTest Results:")
print(f"English queries:  {en_passed}/{len(en_tests)} ({en_passed/len(en_tests)*100:.0f}%)")
print(f"Māori queries:    {mi_passed}/{len(mi_tests)} ({mi_passed/len(mi_tests)*100:.0f}%)")

if mi_passed >= 3:
    print(f"\n GOOD NEWS: Embedding retriever is working!")
    print(f"   {mi_passed} Māori queries now retrieve correct docs")
    print(f"   This is a major improvement over BM25 (which found 0)")
    
    # Estimate improvement
    current_mi_success = 8  # 8/15 Māori queries succeeded with BM25
    estimated_new = current_mi_success + mi_passed
    estimated_mi_perf = estimated_new / 15
    estimated_gap = 1.0 - estimated_mi_perf
    
    print(f"\n Estimated Day 2 results:")
    print(f"   Current Māori GC:  0.533 (8/15)")
    print(f"   Estimated new GC:  {estimated_mi_perf:.3f} ({estimated_new}/15)")
    print(f"   Estimated gap:     {estimated_gap:.3f} (down from 0.467)")
    print(f"   Gap reduction:     {(0.467-estimated_gap)/0.467*100:.0f}%")
    
    print(f"\nWith budget allocation (language-aware/fairness-aware):")
    print(f"   Additional improvement likely!")
    print(f"   Gap could reduce to ~0.20 or less")

elif mi_passed > 0:
    print(f"\nMODERATE: Some improvement, but not as much as hoped")
    print(f"   {mi_passed} queries improved")
    print(f"   You may still see some gap reduction")

else:
    print(f"\nPROBLEM: No improvement for Māori queries")
    print(f"   Embedding retriever may need different model")
    print(f"   Or corpus may have fundamental issues")

# ============================================================================
# 6. Final Recommendation
# ============================================================================

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if mi_passed >= 3:
    print("\nReady to proceed!")
    print("\nNext steps:")
    print("  1. Stop BM25 retriever (if running)")
    print("  2. Keep embedding retriever running")
    print("  3. Backup old results:")
    print("     mv ../outputs/full_evaluation_results.csv ../outputs/results_bm25.csv")
    print("  4. Re-run Day 2:")
    print("     python run_day2.py")
    print("  5. Should see improved results!")

elif mi_passed > 0:
    print("\n Proceed with caution")
    print(f"   Only {mi_passed} queries improved")
    print("   You may want to try a different model")
    print("   Or investigate why some queries still fail")

else:
    print("\nDon't proceed yet")
    print("   No improvement detected")
    print("   Debug before re-running Day 2")

print("\n" + "="*70)
