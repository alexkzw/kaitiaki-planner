"""
Quick Test: Verify Keyword Boost is Working
============================================

This tests that the keyword boost correctly promotes documents whose
IDs match query keywords.
"""

import requests

RETRIEVER = "http://localhost:8001"

print("="*70)
print("TESTING KEYWORD BOOST")
print("="*70)

# Test case: The problematic query
test_cases = [
    {
        "query": "Kei hea a Aotearoa?",  # Where is Aotearoa?
        "expected_top": "mi_aotearoa",
        "description": "Aotearoa location query"
    },
    {
        "query": "He aha te marae?",  # What is a marae?
        "expected_top": "mi_marae",
        "description": "Marae definition query"
    },
    {
        "query": "He aha a Matariki?",  # What is Matariki?
        "expected_top": "mi_matariki",
        "description": "Matariki definition query"
    }
]

passed = 0
for test in test_cases:
    print(f"\n{test['description']}")
    print(f"Query: '{test['query']}'")
    print(f"Expected top result: {test['expected_top']}")

    payload = {
        "query": test["query"],
        "top_k": 5
    }

    r = requests.post(f"{RETRIEVER}/retrieve", json=payload, timeout=10)

    if r.status_code == 200:
        passages = r.json()
        if passages:
            top_doc = passages[0]['doc_id']
            top_score = passages[0]['score']

            print(f"Actual top result: {top_doc} (score: {top_score:.3f})")

            if top_doc == test['expected_top']:
                print("✅ PASS - Correct document ranked first!")
                passed += 1
            else:
                print("❌ FAIL - Wrong document at top")
                print(f"   Top 5: {[p['doc_id'] for p in passages[:5]]}")
        else:
            print("❌ FAIL - No results returned")
    else:
        print(f"❌ FAIL - HTTP {r.status_code}")

print("\n" + "="*70)
print(f"RESULTS: {passed}/{len(test_cases)} tests passed")
print("="*70)

if passed == len(test_cases):
    print("\n✅ All tests passed! Keyword boost is working.")
    print("   You can now re-run the full evaluation.")
elif passed > 0:
    print(f"\n⚠️  Some tests passed ({passed}/{len(test_cases)})")
    print("   Partial improvement expected in evaluation.")
else:
    print("\n❌ No tests passed - keyword boost may not be working")
    print("   Check that retriever was restarted with updated code.")
