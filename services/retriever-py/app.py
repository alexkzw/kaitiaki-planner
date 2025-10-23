"""
Budget-Aware RAG Retriever Service
==================================

Simple BM25-based retriever for English and Māori documents.
Supports retrieval and optional reranking.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from pathlib import Path
from typing import List, Optional
import unicodedata

# Initialize FastAPI
app = FastAPI(title="Budget-Aware Retriever")

# ============================================================================
# Data Models
# ============================================================================

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5

class RerankRequest(BaseModel):
    query: str
    candidates: List[dict]
    k: int = 3

# ============================================================================
# Load Corpus
# ============================================================================

CORPUS_PATH = Path(__file__).parent / "../../data/corpus.json"
corpus = []

def normalize_text(text: str) -> str:
    """Normalize text for better matching."""
    # Normalize unicode (important for Māori text)
    text = unicodedata.normalize('NFC', text)
    # Lowercase
    text = text.lower()
    return text

def load_corpus():
    """Load corpus from JSON file."""
    global corpus
    
    if not CORPUS_PATH.exists():
        print(f"WARNING: Corpus not found at {CORPUS_PATH}")
        corpus = []
        return
    
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    # Normalize text for searching
    for doc in corpus:
        doc['text_normalized'] = normalize_text(doc['text'])
    
    print(f"✓ Loaded {len(corpus)} documents")
    print(f"  English: {sum(1 for d in corpus if d['lang']=='en')}")
    print(f"  Māori: {sum(1 for d in corpus if d['lang']=='mi')}")

# Load corpus on startup
load_corpus()

# ============================================================================
# Simple BM25 Scoring
# ============================================================================

def simple_bm25_score(query: str, doc_text: str) -> float:
    """
    Simplified BM25 scoring.
    Good enough for this evaluation without external dependencies.
    """
    query_normalized = normalize_text(query)
    query_terms = set(query_normalized.split())
    
    if not query_terms:
        return 0.0
    
    doc_normalized = normalize_text(doc_text)
    doc_terms = doc_normalized.split()
    
    if not doc_terms:
        return 0.0
    
    # Count term frequencies
    score = 0.0
    for term in query_terms:
        tf = doc_terms.count(term)
        if tf > 0:
            # Simple TF scoring (can be enhanced with IDF later)
            score += tf / (1.0 + tf)
    
    # Normalize by document length (shorter docs get boost)
    length_norm = 1.0 / (1.0 + len(doc_terms) / 100.0)
    score *= length_norm
    
    return score

# ============================================================================
# Retrieval Endpoint
# ============================================================================

@app.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    """
    Retrieve top_k documents using BM25 scoring.
    
    Returns list of passages with metadata.
    """
    if not corpus:
        raise HTTPException(status_code=503, detail="Corpus not loaded")
    
    query = request.query
    top_k = min(request.top_k, len(corpus))
    
    if top_k <= 0:
        return []
    
    # Score all documents
    scored_docs = []
    for doc in corpus:
        score = simple_bm25_score(query, doc['text'])
        scored_docs.append({
            "doc_id": doc['id'],
            "text": doc['text'][:1000],  # Truncate to 1000 chars
            "char_start": 0,
            "char_end": min(1000, len(doc['text'])),
            "score": score,
            "lang": doc['lang']
        })
    
    # Sort by score (descending)
    scored_docs.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top_k
    return scored_docs[:top_k]

# ============================================================================
# Reranking Endpoint
# ============================================================================

@app.post("/rerank")
async def rerank(request: RerankRequest):
    """
    Rerank candidates and return top k.
    
    For now, just returns top k by existing scores.
    Can be enhanced with cross-encoder later.
    """
    candidates = request.candidates
    k = min(request.k, len(candidates))
    
    if k <= 0:
        return []
    
    # If candidates have scores, use them
    if candidates and 'score' in candidates[0]:
        # Already scored, just slice
        return candidates[:k]
    else:
        # Rescore with query
        query = request.query
        for cand in candidates:
            cand['score'] = simple_bm25_score(query, cand.get('text', ''))
        
        # Sort by score
        candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return candidates[:k]

# ============================================================================
# Health Check
# ============================================================================

@app.get("/healthz")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "corpus_loaded": len(corpus) > 0,
        "num_documents": len(corpus)
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "budget-aware-retriever",
        "version": "1.0.0",
        "corpus_size": len(corpus),
        "status": "ready" if corpus else "no corpus loaded"
    }

# ============================================================================
# Startup Message
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("Budget-Aware Retriever Service")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8001)
