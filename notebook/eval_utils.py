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
