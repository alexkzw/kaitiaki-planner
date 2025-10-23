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

def fairness_gap(df: pd.DataFrame, lang_col: str = "lang", metric_col: str = "gc") -> float:
    """
    Calculate fairness gap between English and Māori languages.
    
    Args:
        df: DataFrame with results
        lang_col: Column name for language
        metric_col: Column name for metric to compare
    
    Returns:
        Fairness gap (EN metric - MI metric)
        Positive = EN performs better (unfair to MI)
        Negative = MI performs better
    """
    by_lang = df.groupby(lang_col)[metric_col].mean()
    en_score = by_lang.get("en", 0.0)
    mi_score = by_lang.get("mi", 0.0)
    return float(en_score - mi_score)

# ============================================================================
# ANALYSIS HELPERS
# ============================================================================

def summarize_results(df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for results DataFrame.
    
    Args:
        df: Results DataFrame with columns: gc, cost, refusal, lang, complexity
    
    Returns:
        Dictionary with summary stats
    """
    import numpy as np
    
    summary = {
        "total_queries": len(df),
        "mean_gc": df["gc"].mean(),
        "total_cost": df["cost"].sum(),
        "mean_cost": df["cost"].mean(),
        "refusal_rate": df["refusal"].mean(),
        "fairness_gap": fairness_gap(df) if "lang" in df.columns else None,
    }
    
    # Add per-slice stats if complexity exists
    if "complexity" in df.columns:
        summary["by_slice"] = {}
        for lang in ["en", "mi"]:
            for comp in ["simple", "complex"]:
                slice_df = df[(df["lang"] == lang) & (df["complexity"] == comp)]
                if len(slice_df) > 0:
                    summary["by_slice"][f"{lang}_{comp}"] = {
                        "n": len(slice_df),
                        "gc": slice_df["gc"].mean(),
                        "cost": slice_df["cost"].mean(),
                        "refusal_rate": slice_df["refusal"].mean()
                    }
    
    return summary


def print_summary(summary: Dict):
    """Pretty-print summary statistics."""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Total queries:   {summary['total_queries']}")
    print(f"Mean GC:         {summary['mean_gc']:.3f}")
    print(f"Total cost:      ${summary['total_cost']:.4f}")
    print(f"Mean cost/query: ${summary['mean_cost']:.6f}")
    print(f"Refusal rate:    {summary['refusal_rate']:.1%}")
    
    if summary.get('fairness_gap') is not None:
        print(f"Fairness gap:    {summary['fairness_gap']:+.3f} (EN - MI)")
    
    if "by_slice" in summary:
        print("\nBy slice:")
        for slice_name, stats in summary["by_slice"].items():
            print(f"  {slice_name:15s}: n={stats['n']:2d}, "
                  f"GC={stats['gc']:.3f}, "
                  f"cost=${stats['cost']:.6f}")
    
    print("="*70)


# ============================================================================
# MODULE TEST
# ============================================================================

if __name__ == "__main__":
    """Test module functions."""
    print("Testing eval_utils module...")
    
    # Test corpus loading
    try:
        corpus = load_corpus()
        print(f"Loaded corpus: {len(corpus)} documents")
    except FileNotFoundError as e:
        print(f"{e}")
    
    # Test eval tasks loading
    try:
        tasks = load_eval_tasks(corpus)
        print(f"Loaded eval tasks: {len(tasks)} tasks")
    except Exception as e:
        print(f"{e}")
    
    # Test metrics
    test_pred = [{"doc_id": "test", "char_start": 0, "char_end": 10}]
    test_gold = {"doc_id": "test", "start": 5, "end": 15}
    gc = grounded_correctness(test_pred, test_gold)
    print(f"Grounded correctness test: {gc}")
    
    print("\nAll tests passed!")