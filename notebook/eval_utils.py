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


def load_eval_tasks(
    corpus: List[Dict],
    tasks_path: str = "../eval/tasks_labeled.yaml"
) -> List[Dict]:
    """
    Load evaluation tasks with gold citation offsets.
    
    Args:
        corpus: Loaded corpus (needed to find offsets)
        tasks_path: Path to tasks YAML file
    
    Returns:
        List of task dictionaries with structure:
        {
            "id": str,
            "query": str,
            "lang": "en" | "mi",
            "complexity": "simple" | "complex",
            "gold_citations": [{"doc_id": str, "start": int, "end": int}]
        }
    """
    path = Path(tasks_path)
    
    # Try labeled version first, fall back to unlabeled
    if not path.exists():
        path = Path(tasks_path.replace("_labeled", ""))
        print(f" Using {path.name} (no complexity labels)")
    
    if not path.exists():
        raise FileNotFoundError(f"Tasks file not found at {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        eval_yaml = yaml.safe_load(f)
    
    eval_tasks = []
    for item in eval_yaml:
        # Find offsets for gold answer
        s, e = find_offsets(corpus, item["gold"]["doc_id"], item["gold"]["text_snippet"])
        
        task = {
            "id": item["id"],
            "query": item["query"],
            "lang": item["lang"],
            "complexity": item.get("complexity", "simple"),  # Default if missing
            "gold_citations": [{
                "doc_id": item["gold"]["doc_id"],
                "start": s,
                "end": e
            }]
        }
        eval_tasks.append(task)
    
    return eval_tasks

# ============================================================================
# METRICS
# ============================================================================

def grounded_correctness(
    pred_cites: List[Dict],
    gold: Dict,
    iou_thresh: float = 0.3
) -> float:
    """
    Calculate grounded correctness using IoU (Intersection over Union).
    
    A prediction is correct if ANY predicted citation overlaps with the gold
    citation by at least `iou_thresh`.
    
    Args:
        pred_cites: List of predicted citations
                   [{"doc_id": str, "char_start": int, "char_end": int}, ...]
        gold: Gold citation {"doc_id": str, "start": int, "end": int}
        iou_thresh: Minimum IoU threshold (default: 0.3)
    
    Returns:
        1.0 if correct (IoU >= threshold), else 0.0
    """
    if not pred_cites:
        return 0.0
    
    gs, ge = gold["start"], gold["end"]
    gdoc = gold["doc_id"]
    
    for c in pred_cites:
        # Check if doc_id matches
        if c.get("doc_id") != gdoc:
            continue
        
        # Get predicted span
        s = int(c.get("char_start", -1))
        e = int(c.get("char_end", -1))
        
        # Calculate IoU
        intersection = max(0, min(e, ge) - max(s, gs))
        union = max(ge, e) - min(gs, s)
        
        if union > 0:
            iou = intersection / union
            if iou >= iou_thresh:
                return 1.0
    
    return 0.0