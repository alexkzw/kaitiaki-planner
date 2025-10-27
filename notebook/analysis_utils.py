"""
Analysis Utilities for Budget-Aware RAG Evaluation
==================================================

Reusable functions for analyzing evaluation results.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple

# ============================================================================
# Fairness Gap Calculations
# ============================================================================

def calculate_fairness_gap(df: pd.DataFrame, mode: str, slice_by: str = 'overall') -> Dict:
    """
    Calculate fairness gap (EN - MI performance) for a given condition.
    
    Args:
        df: Results DataFrame
        mode: Condition name ('uniform', 'language_aware', 'fairness_aware')
        slice_by: 'overall', 'simple', or 'complex'
    
    Returns:
        Dictionary with gap statistics
    """
    mode_df = df[df['mode'] == mode]
    
    if slice_by != 'overall':
        mode_df = mode_df[mode_df['complexity'] == slice_by]
    
    en_perf = mode_df[mode_df['lang'] == 'en']['gc'].mean()
    mi_perf = mode_df[mode_df['lang'] == 'mi']['gc'].mean()
    gap = en_perf - mi_perf
    
    return {
        'mode': mode,
        'slice': slice_by,
        'en_perf': en_perf,
        'mi_perf': mi_perf,
        'gap': gap,
        'gap_pct': (gap / en_perf * 100) if en_perf > 0 else 0
    }

def calculate_all_fairness_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate fairness gaps for all conditions and slices.
    
    Args:
        df: Results DataFrame
    
    Returns:
        DataFrame with gap analysis
    """
    gaps = []
    
    for mode in df['mode'].unique():
        # Overall gap
        gaps.append(calculate_fairness_gap(df, mode, 'overall'))
        
        # By complexity
        for complexity in ['simple', 'complex']:
            gaps.append(calculate_fairness_gap(df, mode, complexity))
    
    return pd.DataFrame(gaps)

# ============================================================================
# Statistical Tests
# ============================================================================

def run_ttest(df: pd.DataFrame, mode: str) -> Dict:
    """
    Run independent t-test comparing EN and MI performance.
    
    Args:
        df: Results DataFrame
        mode: Condition to test
    
    Returns:
        Dictionary with test results
    """
    mode_df = df[df['mode'] == mode]
    en_gc = mode_df[mode_df['lang'] == 'en']['gc']
    mi_gc = mode_df[mode_df['lang'] == 'mi']['gc']
    
    t_stat, p_val = stats.ttest_ind(en_gc, mi_gc)
    
    # Calculate Cohen's d effect size
    cohens_d = (en_gc.mean() - mi_gc.mean()) / \
               np.sqrt((en_gc.std()**2 + mi_gc.std()**2) / 2)
    
    return {
        'mode': mode,
        't_statistic': t_stat,
        'p_value': p_val,
        'cohens_d': cohens_d,
        'significant': p_val < 0.05,
        'en_mean': en_gc.mean(),
        'en_std': en_gc.std(),
        'mi_mean': mi_gc.mean(),
        'mi_std': mi_gc.std()
    }

def run_all_ttests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run t-tests for all conditions.
    
    Args:
        df: Results DataFrame
    
    Returns:
        DataFrame with all test results
    """
    results = []
    for mode in df['mode'].unique():
        results.append(run_ttest(df, mode))
    return pd.DataFrame(results)

def effect_size_interpretation(d: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
    
    Returns:
        Interpretation string
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

# ============================================================================
# Cost-Effectiveness Analysis
# ============================================================================

def calculate_cost_effectiveness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cost per correct answer for each condition and language.
    
    Args:
        df: Results DataFrame
    
    Returns:
        DataFrame with cost effectiveness metrics
    """
    results = []
    
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode]
        
        # Overall
        total_cost = mode_df['cost'].sum()
        correct_answers = (mode_df['gc'] > 0).sum()
        cost_per_correct = total_cost / correct_answers if correct_answers > 0 else np.inf
        
        results.append({
            'mode': mode,
            'slice': 'overall',
            'total_cost': total_cost,
            'correct_answers': correct_answers,
            'cost_per_correct': cost_per_correct
        })
        
        # By language
        for lang in ['en', 'mi']:
            lang_df = mode_df[mode_df['lang'] == lang]
            lang_cost = lang_df['cost'].sum()
            lang_correct = (lang_df['gc'] > 0).sum()
            lang_cost_per_correct = lang_cost / lang_correct if lang_correct > 0 else np.inf
            
            results.append({
                'mode': mode,
                'slice': lang.upper(),
                'total_cost': lang_cost,
                'correct_answers': lang_correct,
                'cost_per_correct': lang_cost_per_correct
            })
    
    return pd.DataFrame(results)

# ============================================================================
# Performance Summary
# ============================================================================

def create_performance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive performance summary table.
    
    Args:
        df: Results DataFrame
    
    Returns:
        DataFrame with summary statistics
    """
    summary = df.groupby('mode').agg({
        'gc': ['mean', 'std', 'min', 'max', 'count'],
        'cost': 'sum',
        'refusal': 'mean',
        'lat_total_ms': 'mean'
    }).round(4)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    return summary

def create_slice_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create performance summary by language and complexity.
    
    Args:
        df: Results DataFrame
    
    Returns:
        DataFrame with slice-level statistics
    """
    df['slice'] = df['lang'].str.upper() + '-' + df['complexity']
    
    summary = df.groupby(['mode', 'slice']).agg({
        'gc': ['mean', 'std', 'count'],
        'cost': 'mean',
        'top_k': 'first'
    }).round(4)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    return summary

# ============================================================================
# Key Findings Generation
# ============================================================================

def generate_key_findings(df: pd.DataFrame) -> Dict:
    """
    Generate key findings from evaluation results.
    
    Args:
        df: Results DataFrame
    
    Returns:
        Dictionary with key findings
    """
    gaps_df = calculate_all_fairness_gaps(df)
    
    # Finding 1: Baseline gap
    uniform_gap = gaps_df[
        (gaps_df['mode'] == 'uniform') & 
        (gaps_df['slice'] == 'overall')
    ]['gap'].values[0]
    
    # Finding 2: Language-aware effectiveness
    lang_gap = gaps_df[
        (gaps_df['mode'] == 'language_aware') & 
        (gaps_df['slice'] == 'overall')
    ]['gap'].values[0]
    lang_reduction = ((uniform_gap - lang_gap) / uniform_gap * 100) if uniform_gap != 0 else 0
    
    # Finding 3: Fairness-aware effectiveness
    fair_gap = gaps_df[
        (gaps_df['mode'] == 'fairness_aware') & 
        (gaps_df['slice'] == 'overall')
    ]['gap'].values[0]
    fair_reduction = ((uniform_gap - fair_gap) / uniform_gap * 100) if uniform_gap != 0 else 0
    
    # Finding 4: Cost trade-off
    uniform_cost = df[df['mode'] == 'uniform']['cost'].sum()
    fair_cost = df[df['mode'] == 'fairness_aware']['cost'].sum()
    cost_increase = ((fair_cost - uniform_cost) / uniform_cost * 100) if uniform_cost > 0 else 0
    
    # Finding 5: Best strategy
    best_mode = gaps_df[gaps_df['slice'] == 'overall'].loc[
        gaps_df['gap'].abs().idxmin(), 'mode'
    ]
    
    return {
        'baseline_gap': uniform_gap,
        'language_aware_gap': lang_gap,
        'language_aware_reduction_pct': lang_reduction,
        'fairness_aware_gap': fair_gap,
        'fairness_aware_reduction_pct': fair_reduction,
        'cost_increase_pct': cost_increase,
        'best_mode': best_mode
    }

# ============================================================================
# Validation Checks
# ============================================================================

def validate_results(df: pd.DataFrame) -> Dict:
    """
    Validate evaluation results for quality checks.
    
    Args:
        df: Results DataFrame
    
    Returns:
        Dictionary with validation results
    """
    checks = {}
    
    # Check 1: Completeness
    expected_rows = 30 * 3  # 30 queries × 3 conditions
    checks['completeness'] = {
        'actual': len(df),
        'expected': expected_rows,
        'pct': len(df) / expected_rows * 100,
        'pass': len(df) >= expected_rows * 0.8  # 80% threshold
    }
    
    # Check 2: Mean GC threshold
    mean_gc = df['gc'].mean()
    checks['mean_gc'] = {
        'value': mean_gc,
        'threshold': 0.3,
        'pass': mean_gc >= 0.3
    }
    
    # Check 3: Refusal rate
    refusal_rate = df['refusal'].mean()
    checks['refusal_rate'] = {
        'value': refusal_rate,
        'threshold': 0.5,
        'pass': refusal_rate < 0.5
    }
    
    # Check 4: Cost reasonableness
    total_cost = df['cost'].sum()
    checks['total_cost'] = {
        'value': total_cost,
        'threshold': 1.0,
        'pass': total_cost < 1.0
    }
    
    # Check 5: Three conditions present
    conditions = set(df['mode'].unique())
    checks['conditions'] = {
        'value': list(conditions),
        'expected': ['uniform', 'language_aware', 'fairness_aware'],
        'pass': len(conditions) == 3
    }
    
    # Overall pass
    checks['overall_pass'] = all(check.get('pass', False) for check in checks.values() if 'pass' in check)
    
    return checks

# ============================================================================
# Helper Functions
# ============================================================================

def format_pvalue(p: float) -> str:
    """Format p-value with significance stars."""
    stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    return f"{p:.4f} {stars}"

def format_gap(gap: float) -> str:
    """Format gap with sign and interpretation."""
    interpretation = "unfair" if abs(gap) > 0.1 else "moderate" if abs(gap) > 0.05 else "fair"
    return f"{gap:+.3f} ({interpretation})"
