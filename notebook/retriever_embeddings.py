"""
Embedding-Based Retriever Service
=================================

Replaces BM25 retriever with semantic search using multilingual embeddings.

Usage:
    python retriever_embeddings.py
    
    # Service will run on http://localhost:8001
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import uvicorn
import json

# ============================================================================
# Global State
# ============================================================================

# Will be loaded on startup
embeddings_data = None
model = None
embeddings_matrix = None
doc_ids = None
doc_texts = None
corpus = None  # Full corpus with metadata

# ============================================================================
# Lifespan Context Manager
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load resources on startup, cleanup on shutdown."""
    global embeddings_data, model, embeddings_matrix, doc_ids, doc_texts, corpus

    print("="*70)
    print("LOADING EMBEDDING RETRIEVER")
    print("="*70)

    # Load pre-computed embeddings
    print("\n1. Loading corpus embeddings...")
    embeddings_path = Path(__file__).parent / "../data/corpus_embeddings.pkl"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {embeddings_path}\n"
            f"Run: python build_embeddings.py first"
        )

    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)

    embeddings_matrix = embeddings_data['embeddings']
    doc_ids = embeddings_data['doc_ids']
    doc_texts = embeddings_data['doc_texts']

    print(f"  Loaded {len(doc_ids)} document embeddings")
    print(f"  Shape: {embeddings_matrix.shape}")

    # Load model for query encoding
    print("\n2. Loading sentence transformer model...")
    model_name = embeddings_data.get('model_name', 'paraphrase-multilingual-mpnet-base-v2')
    model = SentenceTransformer(model_name)
    print(f"  Loaded model: {model_name}")

    # Load full corpus for metadata
    print("\n3. Loading full corpus...")
    corpus_path = Path(__file__).parent / "../data/corpus.json"
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    print(f"  Loaded corpus with metadata")

    print("\n" + "="*70)
    print("EMBEDDING RETRIEVER READY")
    print("="*70)
    print(f"Listening on: http://localhost:8001")
    print(f"Endpoint: POST /search")
    print("="*70 + "\n")

    yield  

    print("\nShutting down embedding retriever...")

app = FastAPI(title="Embedding Retriever", version="1.0.0", lifespan=lifespan)

# ============================================================================
# Request/Response Models
# ============================================================================

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    lang: Optional[str] = None

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5

class RerankRequest(BaseModel):
    query: str
    candidates: List[dict]
    k: int = 3

class Passage(BaseModel):
    doc_id: str
    text: str
    score: float
    char_start: int = 0
    char_end: int = 0

class SearchResponse(BaseModel):
    query: str
    passages: List[Passage]
    retrieval_method: str = "semantic_embeddings"

# ============================================================================
# Health Check
# ============================================================================

@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    if embeddings_matrix is None or model is None:
        raise HTTPException(status_code=503, detail="Embeddings not loaded")
    return {
        "status": "healthy",
        "service": "embedding-retriever",
        "documents": len(doc_ids),
        "model": embeddings_data.get('model_name', 'unknown')
    }

@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "embedding-retriever",
        "version": "1.0.0",
        "method": "semantic_search",
        "model": embeddings_data.get('model_name', 'unknown') if embeddings_data else "not_loaded",
        "documents": len(doc_ids) if doc_ids else 0
    }

# ============================================================================
# Search Endpoint
# ============================================================================

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Semantic search using embeddings.
    
    Compatible with orchestrator's expectations.
    """
    if embeddings_matrix is None or model is None:
        raise HTTPException(
            status_code=503, 
            detail="Retriever not initialized. Embeddings not loaded."
        )
    
    # Encode query
    query_embedding = model.encode([request.query], convert_to_numpy=True)[0]
    
    # Normalize for cosine similarity
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    doc_norms = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    
    # Compute similarities
    similarities = np.dot(doc_norms, query_norm)
    
    # Get top-k indices
    top_k = min(request.top_k, len(similarities))
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Build response passages
    passages = []
    for idx in top_indices:
        doc_id = doc_ids[idx]
        doc_text = doc_texts[idx]
        score = float(similarities[idx])
        
        passages.append(Passage(
            doc_id=doc_id,
            text=doc_text,
            score=score,
            char_start=0,
            char_end=len(doc_text)
        ))
    
    return SearchResponse(
        query=request.query,
        passages=passages
    )

# ============================================================================
# Helper Functions
# ============================================================================

def keyword_boost(doc_id: str, query: str, base_score: float) -> float:
    """
    Boost score if doc_id keywords appear in query.

    This helps prefer documents whose titles match the query topic over
    documents that merely mention the topic in passing.

    Example:
        Query: "Kei hea a Aotearoa?" (Where is Aotearoa?)
        - mi_aotearoa (score=0.525) -> boosted to 0.683 (matches "aotearoa")
        - mi_kauri (score=0.613) -> stays 0.613 (no match)
    """
    import unicodedata
    import re

    # Normalize text (important for Māori diacritics)
    query_normalized = unicodedata.normalize('NFC', query.lower())

    # Extract keywords from doc_id (e.g., "mi_aotearoa" -> ["aotearoa"])
    # Split by underscore and remove language prefix
    parts = doc_id.lower().split('_')
    doc_keywords = parts[1:] if len(parts) > 1 else parts

    # Check if any keyword appears in query
    for keyword in doc_keywords:
        if len(keyword) >= 3:  # Ignore very short keywords
            # Remove common word separators for matching
            keyword_clean = keyword.replace('-', ' ')
            if keyword_clean in query_normalized or keyword in query_normalized:
                # 30% boost - enough to overcome minor ranking errors
                return base_score * 1.3

    return base_score

# ============================================================================
# Retrieve Endpoint (for orchestrator compatibility)
# ============================================================================

@app.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    """
    Retrieve top-k documents using semantic search.

    Compatible with orchestrator's expectations.
    Returns a plain list of passages (not wrapped in a response object).
    """
    if embeddings_matrix is None or model is None:
        raise HTTPException(
            status_code=503,
            detail="Retriever not initialized. Embeddings not loaded."
        )

    # Encode query
    query_embedding = model.encode([request.query], convert_to_numpy=True)[0]

    # Normalize for cosine similarity
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    doc_norms = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)

    # Compute similarities
    similarities = np.dot(doc_norms, query_norm)

    # Apply keyword boost before ranking
    boosted_similarities = similarities.copy()
    for i, doc_id in enumerate(doc_ids):
        boosted_similarities[i] = keyword_boost(doc_id, request.query, similarities[i])

    # Get top-k indices using boosted scores
    top_k = min(request.top_k, len(boosted_similarities))
    top_indices = np.argsort(boosted_similarities)[-top_k:][::-1]

    # Build response - return plain list of dicts for orchestrator compatibility
    passages = []
    for idx in top_indices:
        doc_id = doc_ids[idx]
        doc_text = doc_texts[idx]
        score = float(boosted_similarities[idx])  # Use boosted score

        # Get lang from corpus if available
        lang = "unknown"
        if corpus:
            doc_entry = next((d for d in corpus if d['id'] == doc_id), None)
            if doc_entry:
                lang = doc_entry.get('lang', 'unknown')

        passages.append({
            "doc_id": doc_id,
            "text": doc_text,
            "score": score,
            "char_start": 0,
            "char_end": len(doc_text),
            "lang": lang
        })

    return passages

# ============================================================================
# Rerank Endpoint (for orchestrator compatibility)
# ============================================================================

@app.post("/rerank")
async def rerank(request: RerankRequest):
    """
    Rerank candidates using semantic similarity.

    For semantic embeddings, we re-score the candidates and return top-k.
    """
    if embeddings_matrix is None or model is None:
        raise HTTPException(
            status_code=503,
            detail="Retriever not initialized. Embeddings not loaded."
        )

    candidates = request.candidates
    k = min(request.k, len(candidates))

    if k <= 0 or not candidates:
        return []

    # Encode query
    query_embedding = model.encode([request.query], convert_to_numpy=True)[0]
    query_norm = query_embedding / np.linalg.norm(query_embedding)

    # Re-score each candidate
    rescored = []
    for cand in candidates:
        # Find the document in our embeddings
        doc_id = cand.get('doc_id', '')
        if doc_id in doc_ids:
            idx = doc_ids.index(doc_id)
            doc_norm = embeddings_matrix[idx] / np.linalg.norm(embeddings_matrix[idx])
            score = float(np.dot(doc_norm, query_norm))
        else:
            # If not found, keep original score or set to 0
            score = cand.get('score', 0.0)

        # Apply keyword boost
        boosted_score = keyword_boost(doc_id, request.query, score)

        # Update the candidate with boosted score
        cand_copy = cand.copy()
        cand_copy['score'] = boosted_score
        rescored.append(cand_copy)

    # Sort by boosted score descending
    rescored.sort(key=lambda x: x.get('score', 0), reverse=True)

    # Return top-k
    return rescored[:k]

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STARTING EMBEDDING-BASED RETRIEVER")
    print("="*70)
    print("\nThis replaces the BM25 retriever.")
    print("Make sure orchestrator is configured to use this service.")
    print("\nStarting server...\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
