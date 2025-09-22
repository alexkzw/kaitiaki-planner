'''
- This is the main file for the retrieval layer for the agent system
- Implements embeddings, FAISS, and re-ranking
- Exposes 3 endpoints for the orchestrator (TS) which the notebook can call
'''

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os, pickle, time
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder

APP_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
CHUNK_PATH = os.path.join(DATA_DIR, "chunks.pkl")

# --- Config (right-sized defaults) ---
EMBED_MODEL = os.environ.get("EMBED_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 50))

app = FastAPI(title="Retriever Service", version="0.1.0")

# Lazy load models once per process
emb_model = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder(RERANK_MODEL)

# In-memory stores (persisted on disk after ingest)
index = None           # FAISS index
chunk_store = []       # list[dict]: {"doc_id","lang","title","text","char_start","char_end"}

# ---------- Data models ----------
class Doc(BaseModel):
    id: str
    lang: str = Field(pattern="^(en|mi)$")
    title: str
    text: str

class IngestRequest(BaseModel):
    docs: List[Doc]

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 8

class RerankCandidate(BaseModel):
    doc_id: str
    text: str
    char_start: int
    char_end: int
    score: Optional[float] = None

class RerankRequest(BaseModel):
    query: str
    candidates: List[RerankCandidate]
    k: int = 4

# ---------- Helpers ----------
def chunk_text(text: str, size: int, overlap: int):
    """Return (start,end,text) windows with char offsets for grounded citations."""
    chunks = []
    n = len(text)
    i = 0
    while i < n:
        start = i
        end = min(i + size, n)
        chunks.append((start, end, text[start:end]))
        if end == n: break
        i = end - overlap
        if i < 0: i = 0
    return chunks

def persist():
    """Persist FAISS index + chunk metadata for reproducibility."""
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNK_PATH, "wb") as f:
        pickle.dump(chunk_store, f)

def load_if_exists():
    global index, chunk_store
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNK_PATH):
        with open(CHUNK_PATH, "rb") as f:
            chunk_store = pickle.load(f)
        # Recreate index with correct dim from stored vector length
        # (We don’t store vectors to keep disk small; rebuild index on ingest only.)
        # This service expects ingest before retrieve if index missing.
        return True
    return False

# ---------- Endpoints ----------

@app.post("/ingest")
def ingest(req: IngestRequest):
    """
    WHY: Establish a fixed, tiny corpus for your PoC.
    WHAT: Chunk docs -> embed chunks -> build FAISS -> persist.
    RESULT: /retrieve can now return semantically relevant excerpts + offsets.
    """
    global index, chunk_store
    t0 = time.time()

    # Clear previous state (right-sized behavior for a PoC)
    chunk_store = []

    # 1) Chunk docs and collect metadata
    all_texts = []
    for d in req.docs:
        for (s, e, t) in chunk_text(d.text, CHUNK_SIZE, CHUNK_OVERLAP):
            chunk_store.append({
                "doc_id": d.id, "lang": d.lang, "title": d.title,
                "text": t, "char_start": int(s), "char_end": int(e)
            })
            all_texts.append(t)

    # 2) Embed all chunks (unit-normalized for cosine/IP search)
    X = emb_model.encode(all_texts, normalize_embeddings=True, show_progress_bar=False)
    X = np.array(X, dtype="float32")

    # 3) Build FAISS IP index
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    persist()
    return {
        "status": "ok",
        "chunks": len(chunk_store),
        "build_ms": int((time.time()-t0)*1000)
    }

@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    """
    WHY: First-stage recall (fast, approximate “what might be relevant?”)
    WHAT: Embed query -> FAISS top-k over chunk vectors -> return candidates.
    RESULT: Gives your TS planner/agent a small set of excerpts to consider.
    """
    if index is None or len(chunk_store) == 0:
        raise HTTPException(400, "Index empty. POST /ingest first.")
    qv = emb_model.encode([req.query], normalize_embeddings=True)
    D, I = index.search(np.array(qv, dtype="float32"), req.top_k)
    out = []
    for rank, idx in enumerate(I[0]):
        c = chunk_store[idx]
        out.append({
            "doc_id": c["doc_id"],
            "text": c["text"],
            "char_start": c["char_start"],
            "char_end": c["char_end"],
            "score": float(D[0][rank])
        })
    return out

@app.post("/rerank")
def rerank(req: RerankRequest):
    """
    WHY: Second-stage precision (re-score cands with a cross-encoder).
    WHAT: Cross-encoder predicts a relevance score for (query, passage).
    RESULT: Top-k highest quality excerpts for grounded prompting.
    """
    if len(req.candidates) == 0:
        return []

    pairs = [(req.query, c.text) for c in req.candidates]
    scores = reranker.predict(pairs)     # vectorized; fast on CPU
    scored = []
    for c, s in zip(req.candidates, scores):
        scored.append({
            "doc_id": c.doc_id,
            "text": c.text,
            "char_start": c.char_start,
            "char_end": c.char_end,
            "score": float(s)
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:req.k]
