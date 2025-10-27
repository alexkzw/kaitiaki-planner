"""
Baseline vs Embeddings Comparison Analysis
==========================================

Compares BM25 baseline results with embeddings-based results to quantify
improvement from architectural changes.

Generates:
- Comparison table (CSV)
- Before/after visualisation
- Statistical significance tests
- Detailed query-level comparison

Input files:
- outputs/baseline_bm25_results.csv
- outputs/embeddings_results.csv (or full_evaluation_results.csv)

Output:
- outputs/baseline_vs_embeddings_comparison.csv
- outputs/query_level_improvements.csv
- Terminal summary
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency

# Try to import the correct binomial test function
try:
    from scipy.stats import binomtest  # New scipy versions
except ImportError:
    from scipy.stats import binom_test as binomtest  # Old scipy versions

# ============================================================================
# Load Results
# ============================================================================

print("="*70)
print("BASELINE VS EMBEDDINGS COMPARISON")
print("="*70)

baseline_path = Path("../outputs/baseline_bm25_results.csv")
embeddings_path = Path("../outputs/embeddings_results.csv")

# Fallback to full_evaluation_results.csv if embeddings_results doesn't exist
if not embeddings_path.exists():
    embeddings_path = Path("../outputs/full_evaluation_results.csv")
    print(f"\nUsing full_evaluation_results.csv as embeddings results")

if not baseline_path.exists():
    print(f"\n ERROR: Baseline results not found at {baseline_path}")
    print(f"   Run 02a_baseline_bm25_evaluation.py first to generate baseline data")
    exit(1)

if not embeddings_path.exists():
    print(f"\n ERROR: Embeddings results not found")
    print(f"   Expected at: {embeddings_path}")
    exit(1)

print(f"\nLoading baseline results from: {baseline_path}")
df_baseline = pd.read_csv(baseline_path)

print(f"Loading embeddings results from: {embeddings_path}")
df_embeddings = pd.read_csv(embeddings_path)

print(f"\n  Baseline: {len(df_baseline)} rows")
print(f"  Embeddings: {len(df_embeddings)} rows")

# ============================================================================
# Overall Comparison
# ============================================================================

print("\n" + "="*70)
print("OVERALL PERFORMANCE COMPARISON")
print("="*70)

comparison_data = []

for mode in ['uniform', 'language_aware', 'fairness_aware']:
    # Baseline stats
    baseline_mode = df_baseline[df_baseline['mode'] == mode]
    baseline_en = baseline_mode[baseline_mode['lang'] == 'en']
    baseline_mi = baseline_mode[baseline_mode['lang'] == 'mi']

    baseline_overall = baseline_mode['gc'].mean()
    baseline_en_perf = baseline_en['gc'].mean()
    baseline_mi_perf = baseline_mi['gc'].mean()
    baseline_gap = baseline_en_perf - baseline_mi_perf

    # Embeddings stats
    embed_mode = df_embeddings[df_embeddings['mode'] == mode]
    embed_en = embed_mode[embed_mode['lang'] == 'en']
    embed_mi = embed_mode[embed_mode['lang'] == 'mi']

    embed_overall = embed_mode['gc'].mean()
    embed_en_perf = embed_en['gc'].mean()
    embed_mi_perf = embed_mi['gc'].mean()
    embed_gap = embed_en_perf - embed_mi_perf

    # Calculate improvements
    overall_improvement = embed_overall - baseline_overall
    mi_improvement = embed_mi_perf - baseline_mi_perf
    gap_reduction = baseline_gap - embed_gap

    comparison_data.append({
        'mode': mode,
        'metric': 'Overall',
        'baseline': baseline_overall,
        'embeddings': embed_overall,
        'absolute_improvement': overall_improvement,
        'relative_improvement_pct': (overall_improvement / baseline_overall * 100) if baseline_overall > 0 else 0
    })

    comparison_data.append({
        'mode': mode,
        'metric': 'English',
        'baseline': baseline_en_perf,
        'embeddings': embed_en_perf,
        'absolute_improvement': embed_en_perf - baseline_en_perf,
        'relative_improvement_pct': ((embed_en_perf - baseline_en_perf) / baseline_en_perf * 100) if baseline_en_perf > 0 else 0
    })

    comparison_data.append({
        'mode': mode,
        'metric': 'Māori',
        'baseline': baseline_mi_perf,
        'embeddings': embed_mi_perf,
        'absolute_improvement': mi_improvement,
        'relative_improvement_pct': (mi_improvement / baseline_mi_perf * 100) if baseline_mi_perf > 0 else 0
    })

    comparison_data.append({
        'mode': mode,
        'metric': 'Fairness Gap (EN-MI)',
        'baseline': baseline_gap,
        'embeddings': embed_gap,
        'absolute_improvement': -gap_reduction,  # Negative gap reduction is good
        'relative_improvement_pct': (gap_reduction / baseline_gap * 100) if baseline_gap > 0 else 0
    })

df_comparison = pd.DataFrame(comparison_data)

# Print summary for uniform mode (representative)
print(f"\nUNIFORM MODE (Representative):")
print(f"{'Metric':<25} {'Baseline':<12} {'Embeddings':<12} {'Improvement':<15}")
print("-"*70)

uniform_comp = df_comparison[df_comparison['mode'] == 'uniform']
for _, row in uniform_comp.iterrows():
    metric = row['metric']
    baseline_val = row['baseline']
    embed_val = row['embeddings']
    improvement = row['absolute_improvement']
    rel_pct = row['relative_improvement_pct']

    print(f"{metric:<25} {baseline_val:>6.1%}       {embed_val:>6.1%}       {improvement:>+6.1%} ({rel_pct:>+5.1f}%)")

# ============================================================================
# Key Findings
# ============================================================================

print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

# Focus on uniform mode
uniform_baseline_mi = df_baseline[(df_baseline['mode'] == 'uniform') & (df_baseline['lang'] == 'mi')]['gc'].mean()
uniform_embed_mi = df_embeddings[(df_embeddings['mode'] == 'uniform') & (df_embeddings['lang'] == 'mi')]['gc'].mean()

uniform_baseline_gap = df_baseline[(df_baseline['mode'] == 'uniform') & (df_baseline['lang'] == 'en')]['gc'].mean() - uniform_baseline_mi
uniform_embed_gap = df_embeddings[(df_embeddings['mode'] == 'uniform') & (df_embeddings['lang'] == 'en')]['gc'].mean() - uniform_embed_mi

mi_improvement = uniform_embed_mi - uniform_baseline_mi
gap_reduction = uniform_baseline_gap - uniform_embed_gap

print(f"\n1. Māori Performance Improvement:")
print(f"   BM25 Baseline:     {uniform_baseline_mi:.1%}")
print(f"   Embeddings:        {uniform_embed_mi:.1%}")
print(f"   Absolute gain:     +{mi_improvement:.1%}")
print(f"   Relative gain:     +{(mi_improvement/uniform_baseline_mi)*100:.1f}%")

print(f"\n2. Fairness Gap Reduction:")
print(f"   BM25 Baseline gap: {uniform_baseline_gap:.1%}")
print(f"   Embeddings gap:    {uniform_embed_gap:.1%}")
print(f"   Reduction:         -{gap_reduction:.1%}")
print(f"   % reduction:       {(gap_reduction/uniform_baseline_gap)*100:.1f}%")

# ============================================================================
# Query-Level Analysis
# ============================================================================

print("\n" + "="*70)
print("QUERY-LEVEL IMPROVEMENTS (Uniform Mode)")
print("="*70)

# Focus on uniform mode for clarity
baseline_uniform = df_baseline[df_baseline['mode'] == 'uniform'].copy()
embed_uniform = df_embeddings[df_embeddings['mode'] == 'uniform'].copy()

# Standardize column names
# Baseline has 'task_id', embeddings might have 'id'
if 'id' in embed_uniform.columns and 'task_id' not in embed_uniform.columns:
    embed_uniform = embed_uniform.rename(columns={'id': 'task_id'})

# Check if both dataframes have task_id column
if 'task_id' not in baseline_uniform.columns:
    print("\nERROR: baseline missing 'task_id' column")
    print(f"   Available columns: {list(baseline_uniform.columns)}")
    exit(1)

if 'task_id' not in embed_uniform.columns:
    print("\nERROR: embeddings missing 'task_id' column")
    print(f"   Available columns: {list(embed_uniform.columns)}")
    exit(1)

# Merge on task_id
merged = baseline_uniform.merge(
    embed_uniform,
    on='task_id',
    suffixes=('_baseline', '_embed')
)

# Determine which column name was used for query
query_col_baseline = 'query' if 'query' in merged.columns else 'query_baseline'
query_col_embed = 'query_embed' if 'query_embed' in merged.columns else query_col_baseline

# Calculate improvement for each query
merged['improved'] = (merged['gc_embed'] > merged['gc_baseline']).astype(int)
merged['degraded'] = (merged['gc_embed'] < merged['gc_baseline']).astype(int)
merged['unchanged'] = (merged['gc_embed'] == merged['gc_baseline']).astype(int)

# Query-level summary
query_improvements = []
for _, row in merged.iterrows():
    # Safely get column values with fallbacks
    task_id = row['task_id']
    query = row.get(query_col_baseline, row.get('query', 'Unknown'))
    lang = row.get('lang_baseline', row.get('lang', 'unknown'))
    complexity = row.get('complexity_baseline', row.get('complexity', 'unknown'))
    baseline_gc = row.get('gc_baseline', row.get('gc', 0))
    embeddings_gc = row.get('gc_embed', row.get('gc', 0))

    query_improvements.append({
        'task_id': task_id,
        'query': query,
        'lang': lang,
        'complexity': complexity,
        'baseline_gc': baseline_gc,
        'embeddings_gc': embeddings_gc,
        'improvement': embeddings_gc - baseline_gc,
        'status': 'improved' if row['improved'] else ('degraded' if row['degraded'] else 'unchanged')
    })

df_query_improvements = pd.DataFrame(query_improvements)

# Summary stats
improved_count = merged['improved'].sum()
degraded_count = merged['degraded'].sum()
unchanged_count = merged['unchanged'].sum()

print(f"\nQuery-level changes:")
print(f"  Improved:  {improved_count}/{len(merged)} queries")
print(f"  Degraded:  {degraded_count}/{len(merged)} queries")
print(f"  Unchanged: {unchanged_count}/{len(merged)} queries")

# Show which queries improved (failures that became successes)
print(f"\nQueries that improved (0→1):")
improved_queries = df_query_improvements[
    (df_query_improvements['baseline_gc'] == 0) &
    (df_query_improvements['embeddings_gc'] == 1)
]

if len(improved_queries) > 0:
    for _, row in improved_queries.iterrows():
        print(f"  - {row['task_id']} ({row['lang']}, {row['complexity']})")
        print(f"    Query: {row['query'][:60]}...")
else:
    print(f"  (None)")

# Show queries that degraded (successes that became failures)
print(f"\nQueries that degraded (1→0):")
degraded_queries = df_query_improvements[
    (df_query_improvements['baseline_gc'] == 1) &
    (df_query_improvements['embeddings_gc'] == 0)
]

if len(degraded_queries) > 0:
    for _, row in degraded_queries.iterrows():
        print(f"  - {row['task_id']} ({row['lang']}, {row['complexity']})")
        print(f"    Query: {row['query'][:60]}...")
else:
    print(f"  (None)")

# Show queries still failing
print(f"\nMāori queries still failing in both systems:")
still_failing = df_query_improvements[
    (df_query_improvements['lang'] == 'mi') &
    (df_query_improvements['baseline_gc'] == 0) &
    (df_query_improvements['embeddings_gc'] == 0)
]

if len(still_failing) > 0:
    for _, row in still_failing.iterrows():
        print(f"  - {row['task_id']}")
        print(f"    Query: {row['query']}")
else:
    print(f"  (None - all failures were fixed!)")

# ============================================================================
# Statistical Tests
# ============================================================================

print("\n" + "="*70)
print("STATISTICAL TESTS")
print("="*70)

# McNemar's test for paired binary data
# Tests if the improvement is statistically significant
contingency_table = np.array([
    [
        ((merged['gc_baseline'] == 0) & (merged['gc_embed'] == 0)).sum(),  # Failed both
        ((merged['gc_baseline'] == 0) & (merged['gc_embed'] == 1)).sum()   # Failed baseline, passed embed
    ],
    [
        ((merged['gc_baseline'] == 1) & (merged['gc_embed'] == 0)).sum(),  # Passed baseline, failed embed
        ((merged['gc_baseline'] == 1) & (merged['gc_embed'] == 1)).sum()   # Passed both
    ]
])

print(f"\nContingency Table (Baseline vs Embeddings):")
print(f"                    Embeddings Fail    Embeddings Pass")
print(f"Baseline Fail       {contingency_table[0,0]:>15}    {contingency_table[0,1]:>15}")
print(f"Baseline Pass       {contingency_table[1,0]:>15}    {contingency_table[1,1]:>15}")

# McNemar test (using binomial test)
b = contingency_table[0,1]  # Fail→Pass
c = contingency_table[1,0]  # Pass→Fail
n = b + c

if n > 0:
    # Use binomial test as approximation to McNemar test
    # Under null hypothesis of no difference, b should follow Binomial(n, 0.5)
    try:
        # New scipy API
        result = binomtest(b, n, 0.5, alternative='two-sided')
        p_value = result.pvalue
    except TypeError:
        # Old scipy API (returns p-value directly)
        p_value = binomtest(b, n, 0.5, alternative='two-sided')

    print(f"\nMcNemar's Test (Binomial approximation):")
    print(f"  Fail→Pass (improvements): {b}")
    print(f"  Pass→Fail (degradations): {c}")
    print(f"  p-value: {p_value:.4f}")

    if p_value < 0.05:
        print(f"  Result: Significant improvement (p < 0.05)")
    else:
        print(f"  Result: Not statistically significant at α=0.05")
else:
    print(f"\nMcNemar's Test: N/A (no discordant pairs)")

# ============================================================================
# Save Results
# ============================================================================

print("\n" + "="*70)
print("SAVING COMPARISON RESULTS")
print("="*70)

output_dir = Path("../outputs")

# Save overall comparison
comparison_path = output_dir / "baseline_vs_embeddings_comparison.csv"
df_comparison.to_csv(comparison_path, index=False)
print(f"\nSaved overall comparison to: {comparison_path}")

# Save query-level improvements
query_improvements_path = output_dir / "query_level_improvements.csv"
df_query_improvements.to_csv(query_improvements_path, index=False)
print(f"Saved query-level analysis to: {query_improvements_path}")

print("\n" + "="*70)
print("COMPARISON COMPLETE")
print("="*70)
print(f"\nKey takeaways:")
print(f"  1. Māori performance: {uniform_baseline_mi:.1%} → {uniform_embed_mi:.1%} (+{(mi_improvement/uniform_baseline_mi)*100:.1f}%)")
print(f"  2. Fairness gap: {uniform_baseline_gap:.1%} → {uniform_embed_gap:.1%} ({(gap_reduction/uniform_baseline_gap)*100:.1f}% reduction)")
print(f"  3. {improved_count} queries improved, {degraded_count} degraded")
print(f"\nFiles generated:")
print(f"  - {comparison_path}")
print(f"  - {query_improvements_path}")
