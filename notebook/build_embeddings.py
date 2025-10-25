"""
Build Corpus Embeddings
=======================

One-time script to generate and save embeddings for the entire corpus.
Uses multilingual sentence-transformers model that works with both English and Māori.

Run this ONCE before starting the embedding-based retriever.

Expected time: 2-5 minutes
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pickle

print("="*70)
print("BUILDING CORPUS EMBEDDINGS")
print("="*70)

# ============================================================================
# 1. Load Corpus
# ============================================================================

print("\n1. Loading corpus...")
corpus_path = Path("../data/corpus.json")

if not corpus_path.exists():
    print(f"Corpus not found at {corpus_path}")
    print("   Make sure you're in the notebook/ directory")
    exit(1)

with open(corpus_path, 'r', encoding='utf-8') as f:
    corpus = json.load(f)

print(f"  Loaded {len(corpus)} documents")
print(f"  English: {sum(1 for d in corpus if d.get('lang') == 'en')}")
print(f"  Māori:   {sum(1 for d in corpus if d.get('lang') == 'mi')}")

# ============================================================================
# 2. Load Multilingual Model
# ============================================================================

print("\n2. Loading multilingual embedding model...")

# Use multilingual model that works well with many languages including Māori
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

print("   Model loaded")
print(f"  Model: paraphrase-multilingual-mpnet-base-v2")
print(f"  Embedding dim: 768")
print(f"  Languages: 50+ including English and low-resource languages")

# ============================================================================
# 3. Generate Embeddings
# ============================================================================

print("\n3. Generating embeddings...")

# Extract texts and metadata
doc_ids = [doc['id'] for doc in corpus]
doc_texts = [doc['text'] for doc in corpus]
doc_langs = [doc.get('lang', 'unknown') for doc in corpus]

# Generate embeddings (batch processing for speed)
embeddings = model.encode(
    doc_texts,
    batch_size=8,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(f"  Generated embeddings")
print(f"  Shape: {embeddings.shape}")
print(f"  Size: {embeddings.nbytes / 1024 / 1024:.1f} MB")

# ============================================================================
# 4. Save Embeddings
# ============================================================================

print("\n4. Saving embeddings...")

output_dir = Path("../data")
output_dir.mkdir(exist_ok=True)

# Save as pickle for fast loading
embeddings_data = {
    'embeddings': embeddings,
    'doc_ids': doc_ids,
    'doc_texts': doc_texts,
    'doc_langs': doc_langs,
    'model_name': 'paraphrase-multilingual-mpnet-base-v2',
    'embedding_dim': 768
}

output_path = output_dir / "corpus_embeddings.pkl"
with open(output_path, 'wb') as f:
    pickle.dump(embeddings_data, f)

print(f"  Saved to: {output_path}")
print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

# ============================================================================
# 5. Verification
# ============================================================================

print("\n5. Verification...")

# Test loading
with open(output_path, 'rb') as f:
    loaded = pickle.load(f)

assert loaded['embeddings'].shape == embeddings.shape
assert len(loaded['doc_ids']) == len(corpus)
print("Embeddings verified")

# Test similarity search (sanity check)
print("\n6. Sanity check - test retrieval...")

# English test
query_en = "What is a kea?"
query_emb_en = model.encode([query_en])[0]
similarities_en = np.dot(embeddings, query_emb_en)
top_idx_en = np.argmax(similarities_en)
top_doc_en = doc_ids[top_idx_en]

print(f"\n   English query: '{query_en}'")
print(f"   Top result: {top_doc_en}")
print(f"   Expected: Should contain 'kea'")
if 'kea' in top_doc_en.lower():
    print("Looks correct!")
else:
    print(f"Unexpected result (but may still be valid)")

# Māori test
query_mi = "He aha te Kea?"
query_emb_mi = model.encode([query_mi])[0]
similarities_mi = np.dot(embeddings, query_emb_mi)
top_idx_mi = np.argmax(similarities_mi)
top_doc_mi = doc_ids[top_idx_mi]

print(f"\n   Māori query: '{query_mi}'")
print(f"   Top result: {top_doc_mi}")
print(f"   Expected: Should contain 'kea'")
if 'kea' in top_doc_mi.lower():
    print("Looks correct!")
else:
    print(f"Unexpected result (but may still be valid)")

# ============================================================================
# 7. Summary
# ============================================================================

print("\n" + "="*70)
print("EMBEDDINGS BUILT SUCCESSFULLY")
print("="*70)

print(f"\nSummary:")
print(f"  Documents: {len(corpus)}")
print(f"  Embeddings shape: {embeddings.shape}")
print(f"  Model: paraphrase-multilingual-mpnet-base-v2")
print(f"  Saved to: {output_path}")

print(f"\nNext steps:")
print(f"  1. Start embedding retriever: python retriever_embeddings.py")
print(f"  2. Test it: python test_embedding_retriever.py")
print(f"  3. Re-run Day 2: python run_day2.py")

print("\n" + "="*70)
