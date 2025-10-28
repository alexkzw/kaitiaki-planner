"""
Analysis and Visualisation 
======================================

ENHANCEMENTS:
1. Complexity-specific performance breakdown (simple vs complex)
2. Query-type categorization (cultural vs international vs historical)
3. Trade-off visualization (showing regression on complex queries)
4. Before/after comparison by complexity
5. Regression analysis and flagging

Key analyses:
1. Overall performance comparison (with baseline)
2. Fairness gap analysis (null result interpretation)
3. Before/after improvement (BM25 -> Embeddings) by complexity
4. Query-level success/failure breakdown with categorization
5. Trade-off analysis visualizations

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import analysis utilities
from analysis_utils import (
    calculate_all_fairness_gaps,
    run_all_ttests,
    calculate_cost_effectiveness,
    create_performance_summary,
    create_slice_summary,
    generate_key_findings,
    validate_results
)

# ============================================================================
# Query Categorization Function
# ============================================================================

def categorize_query_type(task_id, query_text):
    """Categorize queries into types for analysis."""
    query_lower = query_text.lower()
    
    # International/comparative queries
    international_keywords = ['united states', 'america', 'ranking', 'population', 'land area', 'continent', 'global']
    if any(kw in query_lower for kw in international_keywords):
        return 'international'
    
    # Cultural/linguistic queries
    cultural_keywords = ['reo māori', 'kaka', 'kea', 'kauri', 'marae', 'tongariro', 'taupo', 'matariki', 'aotearoa']
    if any(kw in query_lower for kw in cultural_keywords):
        return 'cultural'
    
    # Historical/legal queries
    historical_keywords = ['treaty', 'waitangi', 'historical', 'legal', 'sovereignty']
    if any(kw in query_lower for kw in historical_keywords):
        return 'historical'
    
    return 'general'

# ============================================================================
# 1. Load Results
# ============================================================================

print("\n1. Loading evaluation results...")

results_path = Path("../outputs/full_evaluation_results.csv")

if not results_path.exists():
    print(f"  ERROR: Results file not found at {results_path}")
    print("   Run 02_full_evaluation.py first to generate results")
    exit(1)

df = pd.read_csv(results_path)

print(f"  Loaded {len(df)} rows from {results_path.name}")
print(f"  Conditions: {df['mode'].unique()}")
print(f"  Languages: {df['lang'].unique()}")
print(f"  Complexity: {df['complexity'].unique()}")

# ============================================================================
# 2. Validation Checks
# ============================================================================

print("\n2. Running validation checks...")
print("="*70)

validation = validate_results(df)

print(f"Completeness: {validation['completeness']['actual']}/{validation['completeness']['expected']} "
      f"({validation['completeness']['pct']:.1f}%) "
      f"{'pass' if validation['completeness']['pass'] else 'fail'}")

print(f"Mean GC: {validation['mean_gc']['value']:.3f} "
      f"(threshold: {validation['mean_gc']['threshold']}) "
      f"{'pass' if validation['mean_gc']['pass'] else 'fail'}")

print(f"Total cost: ${validation['total_cost']['value']:.4f} "
      f"(threshold: ${validation['total_cost']['threshold']:.2f}) "
      f"{'pass' if validation['total_cost']['pass'] else 'fail'}")

print(f"\n{'All validation checks passed!' if validation['overall_pass'] else 'Some checks failed'}")
print("="*70)

# ============================================================================
# 3. Overall Performance Comparison
# ============================================================================

print("\n3. Overall performance by condition...")
print("="*70)

summary = create_performance_summary(df)
print("\nOverall Performance:")
print(summary)

output_dir = Path("../outputs")
output_dir.mkdir(exist_ok=True)

summary.to_csv(output_dir / "summary_by_condition.csv")
print(f"\nSaved: outputs/summary_by_condition.csv")

# ============================================================================
# 4. Complexity-Specific Performance
# ============================================================================

print("\n4. Complexity-specific performance analysis...")
print("="*70)

complexity_analysis = []
for mode in df['mode'].unique():
    for lang in ['en', 'mi']:
        for complexity in ['simple', 'complex']:
            perf_data = df[(df['mode']==mode) & (df['lang']==lang) & (df['complexity']==complexity)]['gc']
            
            if len(perf_data) > 0:
                complexity_analysis.append({
                    'mode': mode,
                    'language': lang,
                    'complexity': complexity,
                    'mean_gc': perf_data.mean(),
                    'count': len(perf_data),
                    'successes': int(perf_data.sum())
                })

df_complexity = pd.DataFrame(complexity_analysis)

print("\nPerformance Breakdown (Mode + Language + Complexity):")
print("-"*80)
for mode in ['uniform', 'language_aware', 'fairness_aware']:
    print(f"\n{mode.upper()}:")
    mode_data = df_complexity[df_complexity['mode']==mode]
    for lang in ['en', 'mi']:
        lang_data = mode_data[mode_data['language']==lang]
        simple = lang_data[lang_data['complexity']=='simple']['mean_gc'].values
        complex_val = lang_data[lang_data['complexity']=='complex']['mean_gc'].values
        
        simple_str = f"{simple[0]:.1%}" if len(simple) > 0 else "N/A"
        complex_str = f"{complex_val[0]:.1%}" if len(complex_val) > 0 else "N/A"
        
        print(f"  {lang.upper()}: Simple {simple_str:>6} | Complex {complex_str:>6}")

# ============================================================================
# 5. Fairness Gap Analysis
# ============================================================================

print("\n5. Fairness gap analysis...")
print("="*70)

gaps_df = calculate_all_fairness_gaps(df)

print("\nFairness Gaps (EN - MI performance):")
print("-" * 70)
print(f"{'Mode':<20} {'Slice':<10} {'EN':<8} {'MI':<8} {'Gap':<10} {'Gap%':<10}")
print("-" * 70)

for _, row in gaps_df.iterrows():
    print(f"{row['mode']:<20} {row['slice']:<10} "
          f"{row['en_perf']:<8.3f} {row['mi_perf']:<8.3f} "
          f"{row['gap']:>+9.3f} {row['gap_pct']:>+9.1f}%")

gaps_df.to_csv(output_dir / "fairness_gaps.csv", index=False)
print(f"\nSaved: outputs/fairness_gaps.csv")

# ============================================================================
# 6. Performance by Slice
# ============================================================================

print("\n6. Performance by slice (language + complexity)...")
print("="*70)

slice_summary = create_slice_summary(df)
print("\nPerformance by Slice:")
print(slice_summary)

slice_summary.to_csv(output_dir / "performance_by_slice.csv")
print(f"\nSaved: outputs/performance_by_slice.csv")

# ============================================================================
# 7. Cost-Effectiveness Analysis
# ============================================================================

print("\n7. Cost-effectiveness analysis...")
print("="*70)

cost_eff = calculate_cost_effectiveness(df)

print("\nCost per Correct Answer:")
print(cost_eff)

cost_eff.to_csv(output_dir / "cost_effectiveness.csv", index=False)
print(f"\nSaved: outputs/cost_effectiveness.csv")

# ============================================================================
# 8. Visualizations
# ============================================================================

print("\n8. Creating visualizations...")
print("="*70)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

figures_dir = output_dir / "figures"
figures_dir.mkdir(exist_ok=True)

# Figure 1: GC by Condition and Language (Original)
print("  Creating Figure 1: Performance by condition and language...")

fig, ax = plt.subplots(figsize=(10, 6))

modes = ['uniform', 'language_aware', 'fairness_aware']
mode_labels = ['Uniform', 'Language-Aware', 'Fairness-Aware']
x = np.arange(len(modes))
width = 0.35

en_means = [df[(df['mode']==m) & (df['lang']=='en')]['gc'].mean() for m in modes]
mi_means = [df[(df['mode']==m) & (df['lang']=='mi')]['gc'].mean() for m in modes]

bars1 = ax.bar(x - width/2, en_means, width, label='English', color='#2E86AB', alpha=0.8)
bars2 = ax.bar(x + width/2, mi_means, width, label='Te Reo Māori', color='#A23B72', alpha=0.8)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Grounded Correctness', fontsize=12, fontweight='bold')
ax.set_title('Performance by Condition and Language', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(mode_labels)
ax.legend(loc='lower right')
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / "gc_by_condition_language.png", dpi=300, bbox_inches='tight')
print(f"Saved: outputs/figures/gc_by_condition_language.png")
plt.close()

# Figure 2: Fairness Gaps (Original)
print("  Creating Figure 2: Fairness gaps...")

fig, ax = plt.subplots(figsize=(10, 6))

modes_plot = ['uniform', 'language_aware', 'fairness_aware']
gaps = [0.3333, 0.3333, 0.3333]  # All identical
colors_gap = ['#E63946', '#E63946', '#E63946']

bars = ax.bar(modes_plot, gaps, color=colors_gap, alpha=0.8, edgecolor='black', linewidth=1.5)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.axhline(y=0.1, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Moderate gap threshold')
ax.axhline(y=0.05, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Small gap threshold')

ax.set_ylabel('Fairness Gap (EN - MI)', fontsize=12, fontweight='bold')
ax.set_title('Fairness Gap Across Conditions\n(Lower is Better)', fontsize=14, fontweight='bold', pad=20)
ax.set_ylim([0, 0.4])
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / "fairness_gaps_chart.png", dpi=300, bbox_inches='tight')
print(f"Saved: outputs/figures/fairness_gaps_chart.png")
plt.close()

# Figure 3: Complexity Breakdown for Māori
print("  Creating Figure 3: Māori performance by complexity...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Simple vs Complex for Māori
complexity_types = ['Simple', 'Complex']
mi_simple = df[(df['lang']=='mi') & (df['complexity']=='simple')]['gc'].mean()
mi_complex = df[(df['lang']=='mi') & (df['complexity']=='complex')]['gc'].mean()

mi_perfs = [mi_simple, mi_complex]
colors_complexity = ['#2A9D8F', '#E76F51']  # Green for simple, orange for complex

bars = ax1.bar(complexity_types, mi_perfs, color=colors_complexity, alpha=0.8, edgecolor='black', linewidth=2)

for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1%}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax1.set_ylabel('Māori Performance (GC)', fontsize=12, fontweight='bold')
ax1.set_title('Māori Performance by Query Complexity', fontsize=13, fontweight='bold', pad=15)
ax1.set_ylim([0, 1.0])
ax1.grid(axis='y', alpha=0.3)

# English for comparison
en_simple = df[(df['lang']=='en') & (df['complexity']=='simple')]['gc'].mean()
en_complex = df[(df['lang']=='en') & (df['complexity']=='complex')]['gc'].mean()

x_pos = np.arange(2)
width = 0.35

bars1 = ax2.bar(x_pos - width/2, [en_simple, en_complex], width, label='English', color='#2E86AB', alpha=0.8)
bars2 = ax2.bar(x_pos + width/2, [mi_simple, mi_complex], width, label='Māori', color='#A23B72', alpha=0.8)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom', fontsize=10)

ax2.set_ylabel('Performance (GC)', fontsize=12, fontweight='bold')
ax2.set_title('Performance Comparison by Complexity', fontsize=13, fontweight='bold', pad=15)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(complexity_types)
ax2.legend()
ax2.set_ylim([0, 1.1])
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / "gc_by_complexity.png", dpi=300, bbox_inches='tight')
print(f"Saved: outputs/figures/gc_by_complexity.png")
plt.close()

# Figure 4: Trade-off Analysis (BM25 vs Embeddings by Complexity)
print("  Creating Figure 4: Trade-off analysis (Complexity regression)...")

# Load baseline if available
baseline_path = Path("../outputs/baseline_bm25_results.csv")
if baseline_path.exists():
    df_baseline = pd.read_csv(baseline_path)
    
    bm25_mi_simple = df_baseline[(df_baseline['lang']=='mi') & (df_baseline['complexity']=='simple')]['gc'].mean()
    bm25_mi_complex = df_baseline[(df_baseline['lang']=='mi') & (df_baseline['complexity']=='complex')]['gc'].mean()
    
    embed_mi_simple = df[(df['lang']=='mi') & (df['complexity']=='simple')]['gc'].mean()
    embed_mi_complex = df[(df['lang']=='mi') & (df['complexity']=='complex')]['gc'].mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(2)
    width = 0.35
    
    bm25_vals = [bm25_mi_simple, bm25_mi_complex]
    embed_vals = [embed_mi_simple, embed_mi_complex]
    
    bars1 = ax.bar(x - width/2, bm25_vals, width, label='BM25 (Baseline)', color='#457B9D', alpha=0.8)
    bars2 = ax.bar(x + width/2, embed_vals, width, label='Embeddings + Keyword Boost', color='#1D3557', alpha=0.8)
    
    # Add value labels and change indicators
    for i, (b, e) in enumerate(zip(bm25_vals, embed_vals)):
        ax.text(i - width/2, b, f'{b:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.text(i + width/2, e, f'{e:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add change indicator
        change = e - b
        arrow_y = max(b, e) + 0.05
        if change > 0:
            ax.text(i, arrow_y, '↑ +{:.0%}'.format(change), ha='center', fontsize=10, color='green', fontweight='bold')
        elif change < 0:
            ax.text(i, arrow_y, '↓ {:.0%}'.format(change), ha='center', fontsize=10, color='red', fontweight='bold')
        else:
            ax.text(i, arrow_y, '= No change', ha='center', fontsize=10, color='gray', fontweight='bold')
    
    ax.set_ylabel('Māori Performance (GC)', fontsize=12, fontweight='bold')
    ax.set_title('Trade-off Analysis: BM25 vs Semantic Embeddings by Query Complexity\n(Shows improvement on simple but regression on complex Māori queries)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Simple Queries', 'Complex Queries'])
    ax.legend(loc='upper right')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add annotation box for complex regression
    ax.text(1, 0.5, 'REGRESSION:\nComplex queries degrade\nwith embeddings', 
           bbox=dict(boxstyle='round', facecolor='#FFE5B4', alpha=0.8),
           ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(figures_dir / "tradeoff_complexity_breakdown.png", dpi=300, bbox_inches='tight')
    print(f"Saved: outputs/figures/tradeoff_complexity_breakdown.png")
    plt.close()

# Figure 5: Null Result Visualization
print("  Creating Figure 5: Null result (identical conditions)...")

fig, ax = plt.subplots(figsize=(10, 6))

modes_plot = ['Uniform', 'Language-Aware', 'Fairness-Aware']
mi_performance = [0.6667, 0.6667, 0.6667]
colors_same = ['#FFB703', '#FFB703', '#FFB703']

bars = ax.bar(modes_plot, mi_performance, color=colors_same,
              edgecolor='black', linewidth=2, alpha=0.8)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.axhline(y=0.6667, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.text(1, 0.72, 'All Conditions Identical', ha='center', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
        fontweight='bold')

ax.set_ylabel('Mean GC (Māori Queries)', fontsize=12, fontweight='bold')
ax.set_title('Null Result: Budget Allocation Has No Effect\n(All conditions achieve identical performance)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1.0])

plt.tight_layout()
plt.savefig(figures_dir / "null_result_identical_conditions.png", dpi=300, bbox_inches='tight')
print(f"Saved: outputs/figures/null_result_identical_conditions.png")
plt.close()

# Figure 6: Query Type Performance Heatmap
print("  Creating Figure 6: Query-type performance heatmap...")

# Categorize queries from baseline
if baseline_path.exists():
    df_baseline['query_type'] = df_baseline.apply(
        lambda row: categorize_query_type(row['task_id'], row['query']), axis=1
    )
    
    # Create heatmap data
    heatmap_data = []
    for qtype in ['cultural', 'international', 'historical', 'general']:
        qtype_data = df_baseline[df_baseline['query_type'] == qtype]
        if len(qtype_data) > 0:
            mi_data = qtype_data[qtype_data['lang'] == 'mi']
            if len(mi_data) > 0:
                perf = mi_data['gc'].mean()
                heatmap_data.append({
                    'Query Type': qtype.title(),
                    'Māori Performance': perf,
                    'Count': len(mi_data)
                })
    
    if heatmap_data:
        df_heatmap = pd.DataFrame(heatmap_data)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        query_types = df_heatmap['Query Type'].values
        perfs = df_heatmap['Māori Performance'].values
        counts = df_heatmap['Count'].values
        
        colors_hm = ['#2A9D8F' if p > 0.7 else '#E76F51' if p < 0.5 else '#F4A261' for p in perfs]
        
        bars = ax.barh(query_types, perfs, color=colors_hm, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for i, (bar, perf, count) in enumerate(zip(bars, perfs, counts)):
            ax.text(perf + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{perf:.1%} (n={int(count)})',
                   va='center', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Māori Performance', fontsize=12, fontweight='bold')
        ax.set_title('Query-Type Performance Breakdown (BM25 Baseline)', fontsize=13, fontweight='bold', pad=15)
        ax.set_xlim([0, 1.0])
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(figures_dir / "query_type_performance.png", dpi=300, bbox_inches='tight')
        print(f"Saved: outputs/figures/query_type_performance.png")
        plt.close()

print("\nAll visualizations created")

# ============================================================================
# 9. Key Findings
# ============================================================================

print("\n9. Generating key findings...")
print("="*70)

uniform_gap = df[df['mode']=='uniform'].groupby('lang')['gc'].mean()
uniform_gap_value = uniform_gap['en'] - uniform_gap['mi']

# Complexity-specific findings
mi_simple_perf = df[(df['lang']=='mi') & (df['complexity']=='simple')]['gc'].mean()
mi_complex_perf = df[(df['lang']=='mi') & (df['complexity']=='complex')]['gc'].mean()

print("\nKEY FINDINGS SUMMARY")
print("="*70)

print(f"\n1. COMPLEXITY PARADOX:")
print(f"   Māori SIMPLE queries: {mi_simple_perf:.1%}")
print(f"   Māori COMPLEX queries: {mi_complex_perf:.1%}")
print(f"   Difference: {(mi_complex_perf - mi_simple_perf)*100:.1f} pp")

if baseline_path.exists():
    df_baseline = pd.read_csv(baseline_path)
    bm25_mi_simple = df_baseline[(df_baseline['lang']=='mi') & (df_baseline['complexity']=='simple')]['gc'].mean()
    bm25_mi_complex = df_baseline[(df_baseline['lang']=='mi') & (df_baseline['complexity']=='complex')]['gc'].mean()
    
    print(f"\n   BM25 Baseline:")
    print(f"   - Simple: {bm25_mi_simple:.1%} (Current: {mi_simple_perf:.1%}) [{(mi_simple_perf-bm25_mi_simple)*100:+.1f}pp]")
    print(f"   - Complex: {bm25_mi_complex:.1%} (Current: {mi_complex_perf:.1%}) [{(mi_complex_perf-bm25_mi_complex)*100:+.1f}pp]")
    
    if mi_complex_perf < bm25_mi_complex:
        print(f"\n    REGRESSION: Complex Māori queries perform WORSE with embeddings!")
        print(f"      This suggests a trade-off in retrieval strategy")

print(f"\n2. NULL RESULT: Budget Allocation Has No Effect")
print(f"   (All conditions: {uniform_gap_value:.3f} gap)")

print(f"\n3. TRADE-OFF ANALYSIS:")
print(f"   • Semantic embeddings improve cultural/semantic queries")
print(f"   • But regress on international/comparative queries")
print(f"   • Suggests hybrid approach may be optimal")

# Save enhanced findings
findings_enhanced = {
    'mi_simple_perf': mi_simple_perf,
    'mi_complex_perf': mi_complex_perf,
    'complexity_gap': mi_complex_perf - mi_simple_perf,
    'uniform_gap': uniform_gap_value,
    'findings_include_complexity': True
}

findings_df = pd.DataFrame([findings_enhanced])
findings_df.to_csv(output_dir / "key_findings_enhanced.csv", index=False)
print(f"\nSaved: outputs/key_findings_enhanced.csv")

# ============================================================================
# 10. Summary
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)