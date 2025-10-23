"""
Evaluation Utilities for Kaitiaki Planner
==========================================

Helper functions for:
- Loading corpus and evaluation tasks
- Calculating metrics (grounded correctness, fairness gaps)
- Data processing and analysis
"""

import json
import yaml
import unicodedata as ud
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

# ============================================================================
# CORPUS LOADING
# ============================================================================

def load_corpus(corpus_path: str = "../data/corpus.json") -> List[Dict]:
    """
    Load corpus from JSON file.
    
    Args:
        corpus_path: Path to corpus.json
    
    Returns:
        List of document dictionaries
    """
    path = Path(corpus_path)
    
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {path}. "
            f"Run build_corpus.py first or check path."
        )
    
    with open(path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    return corpus

# ============================================================================
# EVALUATION TASKS
# ============================================================================

def nfc(x: str) -> str:
    """Normalise unicode string (NFC normalisation)."""
    return ud.normalize("NFC", x)


def find_offsets(corpus: List[Dict], doc_id: str, snippet: str) -> Tuple[int, int]:
    """
    Find character offsets of snippet in document.
    
    Args:
        corpus: List of documents
        doc_id: Document ID to search in
        snippet: Text snippet to find
    
    Returns:
        (start_offset, end_offset) tuple
    
    Raises:
        ValueError: If document not found or snippet not found
    """
    # Get document text
    doc = next((d for d in corpus if d["id"] == doc_id), None)
    if not doc:
        raise ValueError(f"Document not found: {doc_id}")
    
    text = doc["text"]
    
    # Normalise and search (case-insensitive)
    T = nfc(text).lower()
    S = nfc(snippet).lower()
    i = T.find(S)
    
    if i == -1:
        preview = nfc(text)[:80].replace("\n", " ")
        raise ValueError(
            f"[Gold snippet not found]\n"
            f"  doc_id: {doc_id}\n"
            f"  snippet: '{snippet}'\n"
            f"  doc starts: '{preview}…'"
        )
    
    return i, i + len(S)