"""
Association Rule Mining on Counterfactual Drivers
===================================================

Pipeline (from feature_importance.py):
  1. BoCSoR.explain()  -> finds one differing feature per (sample, CF_neighbor) pair
                         -> transactions_values.csv
  2. extract_labels()  -> extracts change labels (e.g., "SCHL=changes")
                         -> labels_only_unique.csv (one row per sample-CF pair)
  3. aggregate_drivers_by_sample() -> consolidates all drivers per sample across
                                      all its CF neighbors into one transaction
                                      -> aggregated_labels_by_sample.csv

This module (macroscopic_experiment_association_rules.py):
  - Reads aggregated_labels_by_sample.csv (preferred) or labels_only_unique.csv
  - Runs FP-Growth to discover itemsets and association rules
  - Auto-calibrates support/confidence/lift thresholds based on transaction sparsity
  - Keeps both directions of each rule (A->B and B->A) to support directional analysis
  - Generates heatmaps and cross-k comparison summaries

Example rule discovered:
  When SCHL (education level) changes on the decision boundary,
  OCCP (occupation) also changes 70% of the time (confidence=0.70, lift=2.1)

Input CSV format (aggregated_labels_by_sample.csv):
  Sample_ID | Labels          | Num_Labels | Num_CF_Neighbors
  1284      | ['OCCP']        | 1          | 1
  1442      | ['WKHP']        | 1          | 1
  ...

  The 'Labels' column contains Python list literals -- parsed with ast.literal_eval.
  Multi-item rows (e.g. ['OCCP', 'SCHL']) arise when a sample has multiple CF
  neighbors that each identify a different driver; these enable pairwise rules.
"""

import ast
import datetime
import os
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Parallel, delayed
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# headless backend -- needed when running on a server without a display
matplotlib.use('Agg')

# Total logical cores -- used for logging only.
_CPU_CORES = os.cpu_count() or 1

# PERF OPT 2 -- platform-aware parallelism.
# On macOS (M-series) we cap n_jobs at the P-core count to avoid the
# E-core straggler bottleneck.  On Linux/Windows all cores are equivalent.
# Adjust _PERF_CORES manually: M2 base=4, Pro=6/8, Max=8-12, Ultra=16.
import platform as _platform
if _platform.system() == "Darwin":
    _PERF_CORES = 4   # M2 base default
else:
    _PERF_CORES = _CPU_CORES
del _platform


# ---------------------------------------------------------------------------
# Neutral-window helper  -- SINGLE SOURCE OF TRUTH
# ---------------------------------------------------------------------------

def _neutral_window(lift_neutral_half_window):
    """
    Return (lo, hi) for the neutral lift window.

    Rules with lift in [lo, hi] are excluded everywhere:
      - in FP-Growth filtering (_process_one_support, grid_search_fpgrowth_delta)
      - in lift-axis masking of heatmaps (plot_heatmaps, run_k_comparison)

    Formula:
        lo = round(1.0 - lift_neutral_half_window, 4)
        hi = round(1.0 + lift_neutral_half_window, 4)

    With the default half_window=0.25 this gives [0.75, 1.25].

    WHY A SINGLE HELPER?
    The previous code used two different formulas:
      - explore_association_rules and _process_one_support used:
            lo = round(1.0 - half_window, 4)
      - plot_heatmaps and run_k_comparison used:
            lo = round(1.0 - lift_delta * round(half_window / lift_delta), 4)
    When half_window is not an exact multiple of lift_delta those formulas
    diverge, causing heatmap columns to be hidden even though they contain
    valid rules.  This helper eliminates the divergence by centralising the
    logic.

    NEGATIVE CORRELATIONS:
    Rules with lift < lo (e.g. lift < 0.75) indicate pairs of features that
    tend NOT to co-occur on the decision boundary -- an anti-correlation that
    is analytically meaningful.  Only the [lo, hi] band is removed; rules
    below lo are intentionally preserved.  Set LIFT_MIN=0.0 in the entry
    point to include them in the grid search.
    """
    lo = round(1.0 - lift_neutral_half_window, 4)
    hi = round(1.0 + lift_neutral_half_window, 4)
    return lo, hi


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def extract_labels(labels_only_path):
    """
    Read the labels CSV and one-hot encode each transaction for FP-Growth.

    Supports both file formats:
      - aggregated_labels_by_sample.csv  : one row per sample, 'Labels' column
      - labels_only_unique.csv           : one row per sample-CF pair, 'Labels' column

    Both files use the same 'Labels' column containing Python list literals
    (e.g. "['OCCP', 'SCHL']") -- parsed with ast.literal_eval.
    """
    print("  > Loading and encoding labels...")
    df = pd.read_csv(labels_only_path)

    if 'Labels' not in df.columns:
        raise ValueError(
            f"CSV must have a 'Labels' column. "
            f"Found: {df.columns.tolist()}  |  file: {labels_only_path}"
        )

    print(f"    (file: {Path(labels_only_path).name}, {len(df):,} rows)")
    itemsets = df['Labels'].apply(ast.literal_eval)

    te = TransactionEncoder()
    te_ary = te.fit(itemsets).transform(itemsets)
    encoded_df = pd.DataFrame(te_ary, columns=te.columns_)

    print("  > First few encoded rows:")
    print(encoded_df.head())
    print("-" * 50)

    return encoded_df


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def cleanup_empty_folders(output_dir):
    """
    After filtering, some conf_* folders end up with no rules -- delete them.
    If a sup_* folder loses all its conf_* subfolders, delete that too.
    Returns (n_conf_removed, n_sup_removed).
    """
    output_dir = Path(output_dir)
    removed_conf = 0
    removed_sup = 0

    for sup_dir in sorted(output_dir.glob("sup_*")):
        if not sup_dir.is_dir():
            continue
        for conf_dir in sorted(sup_dir.glob("conf_*")):
            if not conf_dir.is_dir():
                continue
            rules_csv = conf_dir / "rules.csv"
            should_remove = False
            if not rules_csv.exists():
                should_remove = True
            else:
                try:
                    should_remove = pd.read_csv(rules_csv).empty
                except Exception:
                    should_remove = True
            if should_remove:
                shutil.rmtree(conf_dir)
                removed_conf += 1
        if not list(sup_dir.glob("conf_*")):
            shutil.rmtree(sup_dir)
            removed_sup += 1

    return removed_conf, removed_sup


# ---------------------------------------------------------------------------
# Grid search (in-memory, no files written)
# ---------------------------------------------------------------------------

def grid_search_fpgrowth_delta(df, sup_min, sup_max, sup_delta,
                               conf_min, conf_max, conf_delta,
                               lift_min, lift_max, lift_delta,
                               lift_neutral_half_window=0.25):
    """
    Quick in-memory grid search. Use explore_association_rules() for full output.
    """
    print(f"\n{'='*70}")
    print("GRID SEARCH: FP-GROWTH (IN-MEMORY)")
    print(f"{'='*70}")

    support_grid    = np.round(np.arange(sup_min,  sup_max  + sup_delta  / 2, sup_delta),  4)
    confidence_grid = np.round(np.arange(conf_min, conf_max + conf_delta / 2, conf_delta), 4)
    lift_grid       = np.round(np.arange(lift_min, lift_max + lift_delta / 2, lift_delta), 4)

    # Unified neutral window.
    lift_neutral_lo, lift_neutral_hi = _neutral_window(lift_neutral_half_window)

    print(f"  > support grid:    {support_grid}")
    print(f"  > confidence grid: {confidence_grid}")
    print(f"  > lift grid:       {lift_grid}")
    print(f"  > neutral window (excluded): [{lift_neutral_lo}, {lift_neutral_hi}]")
    print("-" * 50)

    results = []

    for min_sup in support_grid:
        frequent_itemsets = fpgrowth(df, min_support=min_sup, use_colnames=True)
        if len(frequent_itemsets) == 0:
            continue

        # PERF OPT 1 -- single association_rules() call, then filter by confidence.
        try:
            all_rules_base = association_rules(
                frequent_itemsets, metric="confidence", min_threshold=float(confidence_grid[0])
            )
        except ValueError:
            continue

        if len(all_rules_base) == 0:
            continue

        # Exclude neutral window. Negative correlations (lift < lo) are kept.
        all_rules_base = all_rules_base[
            (all_rules_base['lift'] < lift_neutral_lo) | (all_rules_base['lift'] > lift_neutral_hi)
        ]
        if len(all_rules_base) == 0:
            continue

        for min_conf in confidence_grid:
            rules = all_rules_base[all_rules_base['confidence'] >= min_conf]
            if len(rules) == 0:
                continue

            for min_lift in lift_grid:
                if lift_neutral_lo <= min_lift <= lift_neutral_hi:
                    continue
                filtered = rules[rules['lift'] >= min_lift]
                n = len(filtered)
                results.append({
                    'Support':         min_sup,
                    'Confidence':      min_conf,
                    'Lift':            min_lift,
                    'Number_of_Rules': n,
                    'Max_Lift':        round(filtered['lift'].max(), 4) if n > 0 else 0,
                    'Mean_Confidence': round(filtered['confidence'].mean(), 4) if n > 0 else 0,
                })

    print("  > Done.")

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values(by=['Number_of_Rules', 'Lift'], ascending=[False, False])

    return df_results


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

def plot_heatmaps(summary_df, output_dir, lift_display_step=0.1,
                  lift_neutral_half_window=0.25, lift_delta=0.05):
    """
    3 heatmaps: support-confidence, support-lift, confidence-lift.
    Each cell = max rules over the third parameter (darker = more rules).

    The neutral-window masking on lift axes uses _neutral_window() --
    the same helper used during FP-Growth filtering -- so the columns hidden
    here are exactly the values excluded during computation.

    Negative-correlation columns (lift < neutral_lo) are kept and appear on
    the left side of the lift axis when LIFT_MIN=0.0.

    Parameters
    ----------
    lift_delta : float
        Kept for API compatibility; window computation now delegates to
        _neutral_window(lift_neutral_half_window).
    """
    if summary_df.empty:
        print("  > Summary is empty, skipping heatmaps.")
        return

    output_dir = Path(output_dir)
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    print("  > Generating heatmaps...")

    df = summary_df.copy()
    df['Lift_display'] = (
        (df['Lift_threshold'] / lift_display_step).round() * lift_display_step
    ).round(4)

    # Unified neutral window -- identical to FP-Growth filter boundaries.
    neutral_lo, neutral_hi = _neutral_window(lift_neutral_half_window)

    configs = [
        ('Confidence',   'Support',    'support_confidence', False),
        ('Lift_display', 'Support',    'support_lift',        True),
        ('Lift_display', 'Confidence', 'confidence_lift',     True),
    ]

    for x_col, y_col, suffix, x_is_lift in configs:
        pivot = (
            df.groupby([y_col, x_col])['Number_of_Rules']
            .max()
            .unstack(level=x_col)
            .sort_index(ascending=False)
            .fillna(0)
            .astype(int)
        )

        if x_is_lift:
            # Mask neutral window. Negative-correlation columns survive.
            pivot = pivot.loc[:, ~pivot.columns.to_series().between(
                neutral_lo, neutral_hi, inclusive='both')]
            if (pivot != 0).any(axis=0).any():
                last_nonzero = int(np.where((pivot != 0).any(axis=0).values)[0].max())
                pivot = pivot.iloc[:, :last_nonzero + 1]

        n_cols = len(pivot.columns)
        n_rows = len(pivot.index)
        fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.75), max(4, n_rows * 0.55)))

        img = ax.imshow(pivot.values, aspect='auto', cmap='YlOrBr', interpolation='nearest')

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], rotation=40, ha='right', fontsize=8)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=8)

        x_label = 'Lift' if x_is_lift else x_col
        ax.set_xlabel(x_label, fontsize=11, labelpad=8)
        ax.set_ylabel(y_col, fontsize=11, labelpad=8)
        ax.set_title(
            f"Max Number of Rules -- {y_col} vs {x_label}\n"
            f"(darker = more rules; max over the third parameter)",
            fontsize=11, pad=14
        )

        max_val = pivot.values.max() if pivot.values.max() > 0 else 1
        for ri in range(n_rows):
            for ci in range(n_cols):
                val = pivot.values[ri, ci]
                if val > 0:
                    txt_color = 'white' if (val / max_val) > 0.55 else 'black'
                    ax.text(ci, ri, str(val), ha='center', va='center', fontsize=7, color=txt_color)

        cbar = plt.colorbar(img, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label('Number of Rules', fontsize=9)
        plt.tight_layout()

        fig.savefig(heatmap_dir / f"heatmap_{suffix}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    > saved heatmaps/heatmap_{suffix}.png")

    print(f"  > heatmaps saved to {heatmap_dir}/")


# ---------------------------------------------------------------------------
# Core worker (parallelised over support thresholds)
# ---------------------------------------------------------------------------

def _process_one_support(min_sup, sup_idx, n_sup, df, output_dir,
                         confidence_grid, lift_grid_used,
                         lift_window_lo, lift_window_hi):
    """
    Process one support threshold for explore_association_rules.

    Called in parallel; writes to its own sup_{min_sup}/ subdirectory.
    Returns a list of summary-row dicts.

    Both A->B and B->A directions are kept (different confidence values).
    Negative correlations (lift < lift_window_lo) are preserved alongside
    positive-correlation rules.
    conviction=inf (confidence=1.0) is replaced with np.nan before saving.
    """
    output_dir = Path(output_dir)
    sup_label = f"{min_sup:.2f}"
    sup_dir = output_dir / f"sup_{sup_label}"
    sup_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  [{sup_idx}/{n_sup}] support = {min_sup}")
    print("    > running FP-Growth...")

    frequent_itemsets = fpgrowth(df, min_support=min_sup, use_colnames=True)
    print(f"    > found {len(frequent_itemsets)} frequent itemsets")

    if len(frequent_itemsets) == 0:
        print("    > no frequent itemsets, skipping.")
        return []

    fi = frequent_itemsets.copy()
    fi['itemset_str']    = fi['itemsets'].apply(lambda x: ', '.join(sorted(x)))
    fi['itemset_length'] = fi['itemsets'].apply(len)
    fi = fi[['itemset_str', 'itemset_length', 'support']]
    fi = fi.sort_values(by=['itemset_length', 'support'], ascending=[True, False])
    fi.to_csv(sup_dir / "frequent_itemsets.csv", index=False)

    itemsets_by_len = fi['itemset_length'].value_counts().sort_index().to_dict()
    print(f"    > breakdown: {', '.join(f'len={k}: {v}' for k, v in itemsets_by_len.items())}")

    with open(sup_dir / "frequent_itemsets_summary.txt", 'w') as f:
        f.write("Frequent Itemsets Summary\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Parameters:\n  Min Support: {min_sup}\n\n")
        f.write(f"Results:\n  Total: {len(frequent_itemsets)}\n")
        for length, count in itemsets_by_len.items():
            f.write(f"  len={length}: {count}\n")
        f.write(f"\nAll Frequent Itemsets (by length, then support desc):\n{'-'*60}\n")
        for _, row in fi.iterrows():
            f.write(f"  [{row['itemset_str']}]  support={row['support']:.4f}  length={row['itemset_length']}\n")

    local_summary_rows = []

    # PERF OPT 1 -- single association_rules() call at conf_min, then filter.
    try:
        all_rules_base = association_rules(
            frequent_itemsets, metric="confidence", min_threshold=float(confidence_grid[0])
        )
    except ValueError:
        return local_summary_rows

    if len(all_rules_base) == 0:
        return local_summary_rows

    # Exclude neutral window. Negative correlations (lift < lo) are kept.
    all_rules_base = all_rules_base[
        (all_rules_base['lift'] < lift_window_lo) | (all_rules_base['lift'] > lift_window_hi)
    ]
    all_rules_base = all_rules_base.sort_values('lift', ascending=False).reset_index(drop=True)

    if len(all_rules_base) == 0:
        return local_summary_rows

    for min_conf in confidence_grid:
        conf_label = f"{min_conf:.2f}"
        conf_dir = sup_dir / f"conf_{conf_label}"

        # Cheap pandas filter -- no recomputation of metrics.
        rules = all_rules_base[all_rules_base['confidence'] >= min_conf].reset_index(drop=True)
        if len(rules) == 0:
            continue

        # conviction=inf (confidence=1.0) -> NaN, computed once, shared by both outputs
        conviction_vals = rules['conviction'].replace([np.inf, -np.inf], np.nan)

        # compact output
        # support_raw   : proportion as decimal (e.g. 0.3790)
        # support_pct   : proportion as percentage (e.g. 37.90)
        # confidence_raw: P(consequent | antecedent) as decimal
        # confidence_pct: P(consequent | antecedent) as percentage
        # lift          : ratio -- kept as-is (1.0=independence, >1 positive, <1 negative)
        # leverage      : support(A∪B) - support(A)*support(B), signed difference
        # conviction    : directional strength (inf -> NaN when confidence=1.0)
        fmt = pd.DataFrame()
        fmt['antecedents']    = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
        fmt['consequents']    = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
        fmt['support_raw']    = rules['support'].round(4)
        fmt['support_pct']    = (rules['support']    * 100).round(2)
        fmt['confidence_raw'] = rules['confidence'].round(4)
        fmt['confidence_pct'] = (rules['confidence'] * 100).round(2)
        fmt['lift']           = rules['lift'].round(4)
        fmt['leverage']       = rules['leverage'].round(6)
        fmt['conviction']     = conviction_vals.round(4)

        # detailed output
        det = pd.DataFrame()
        det['antecedents']            = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
        det['consequents']            = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
        det['antecedent_length']      = rules['antecedents'].apply(len)
        det['consequent_length']      = rules['consequents'].apply(len)
        det['rule_length']            = det['antecedent_length'] + det['consequent_length']
        det['antecedent_support_raw'] = rules['antecedent support'].round(4)
        det['antecedent_support_pct'] = (rules['antecedent support'] * 100).round(2)
        det['consequent_support_raw'] = rules['consequent support'].round(4)
        det['consequent_support_pct'] = (rules['consequent support'] * 100).round(2)
        det['support_raw']            = rules['support'].round(4)
        det['support_pct']            = (rules['support']    * 100).round(2)
        det['confidence_raw']         = rules['confidence'].round(4)
        det['confidence_pct']         = (rules['confidence'] * 100).round(2)
        det['lift']                   = rules['lift'].round(4)
        det['leverage']               = rules['leverage'].round(6)
        det['conviction']             = conviction_vals.round(4)

        conf_dir.mkdir(parents=True, exist_ok=True)
        fmt.to_csv(conf_dir / "rules.csv", index=False)
        det.to_csv(conf_dir / "rules_detailed.csv", index=False)

        with open(conf_dir / "summary.txt", 'w') as f:
            f.write("Association Rules Summary\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Parameters:\n")
            f.write(f"  Min Support:    {min_sup}\n")
            f.write(f"  Min Confidence: {min_conf}\n")
            f.write(f"  Neutral Lift Window (excluded): [{lift_window_lo}, {lift_window_hi}]\n\n")
            f.write(f"Results:\n")
            f.write(f"  Frequent Itemsets: {len(frequent_itemsets)}\n")
            f.write(f"  Association Rules: {len(rules)}\n\n")
            f.write(f"Statistics:\n")
            f.write(f"  Avg Support:     {rules['support'].mean()*100:.2f}%\n")
            f.write(f"  Avg Confidence:  {rules['confidence'].mean()*100:.2f}%\n")
            f.write(f"  Avg Lift:        {rules['lift'].mean():.4f}\n")
            f.write(f"  Lift Range:      {rules['lift'].min():.4f} - {rules['lift'].max():.4f}\n")
            f.write(f"  Avg Leverage:    {rules['leverage'].mean():.6f}\n")
            f.write(f"  Avg Rule Length: {det['rule_length'].mean():.2f}\n\n")
            n_inf = conviction_vals.isna().sum()
            if n_inf > 0:
                f.write(f"  Note: {n_inf} rule(s) have confidence=1.0 "
                        f"(conviction=inf -> saved as NaN)\n\n")
            f.write(f"Top 10 Rules (by Lift):\n{'-'*60}\n")
            for idx, row in fmt.head(10).iterrows():
                f.write(f"{idx+1}. {row['antecedents']} => {row['consequents']}\n")
                f.write(f"   support={row['support_pct']:.2f}% | confidence={row['confidence_pct']:.2f}% | lift={row['lift']:.4f} | leverage={row['leverage']:.6f}\n\n")

        print(f"    > [conf={min_conf}] {len(rules)} rules saved to {conf_dir.relative_to(output_dir)}/")

        for min_lift in lift_grid_used:
            filtered = rules[rules['lift'] >= min_lift]
            n = len(filtered)
            rl = (
                filtered['antecedents'].apply(len) + filtered['consequents'].apply(len)
            ) if n > 0 else pd.Series(dtype=float)

            local_summary_rows.append({
                'Support':               min_sup,
                'Confidence':            min_conf,
                'Lift_threshold':        min_lift,
                'Number_of_Rules':       n,
                'Max_Lift':              round(filtered['lift'].max(), 4) if n > 0 else 0.0,
                'Min_Lift':              round(filtered['lift'].min(), 4) if n > 0 else 0.0,
                'Avg_Lift':              round(filtered['lift'].mean(), 4) if n > 0 else 0.0,
                'Avg_Confidence':        round(filtered['confidence'].mean(), 4) if n > 0 else 0.0,
                'Avg_Support':           round(filtered['support'].mean(), 4) if n > 0 else 0.0,
                'Avg_Rule_Length':       round(rl.mean(), 4) if n > 0 else 0.0,
                'Max_Rule_Length':       int(rl.max()) if n > 0 else 0,
                'Num_Frequent_Itemsets': len(frequent_itemsets),
                'Num_FI_length_1':       itemsets_by_len.get(1, 0),
                'Num_FI_length_2':       itemsets_by_len.get(2, 0),
                'Num_FI_length_3plus':   sum(v for k, v in itemsets_by_len.items() if k >= 3),
            })

    return local_summary_rows


# ---------------------------------------------------------------------------
# Full grid exploration
# ---------------------------------------------------------------------------

def explore_association_rules(df, output_dir,
                              sup_min, sup_max, sup_delta,
                              conf_min, conf_max, conf_delta,
                              lift_min, lift_max, lift_delta,
                              lift_neutral_half_window=0.25):
    """
    Full grid search over support x confidence x lift using FP-Growth.

    The neutral lift window is excluded via _neutral_window().
    Negative correlations (lift < lo) are preserved and analysed.
    Both A->B and B->A rule directions are kept.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    support_grid    = np.round(np.arange(sup_min,  sup_max  + sup_delta  / 2, sup_delta),  4)
    confidence_grid = np.round(np.arange(conf_min, conf_max + conf_delta / 2, conf_delta), 4)
    lift_grid       = np.round(np.arange(lift_min, lift_max + lift_delta / 2, lift_delta), 4)

    # Unified neutral window.
    lift_window_lo, lift_window_hi = _neutral_window(lift_neutral_half_window)

    lift_grid_used = [v for v in lift_grid if not (lift_window_lo <= v <= lift_window_hi)]
    total_combos   = len(support_grid) * len(confidence_grid) * len(lift_grid_used)

    print(f"\n{'='*70}")
    print("FULL EXPLORATION: FP-GROWTH ASSOCIATION RULES")
    print(f"{'='*70}")
    print(f"  > support    : {len(support_grid)} values [{support_grid[0]} ... {support_grid[-1]}, step={sup_delta}]")
    print(f"  > confidence : {len(confidence_grid)} values [{confidence_grid[0]} ... {confidence_grid[-1]}, step={conf_delta}]")
    print(f"  > lift       : {len(lift_grid_used)} values used (of {len(lift_grid)} total, "
          f"{len(lift_grid)-len(lift_grid_used)} skipped -- neutral window [{lift_window_lo}, {lift_window_hi}]), step={lift_delta}")
    print(f"  > total combinations: {total_combos:,}")
    print("-" * 50)

    n_jobs = min(_PERF_CORES, len(support_grid))
    print(f"  > Launching parallel FP-Growth over {len(support_grid)} support "
          f"values (n_jobs={n_jobs} of {_CPU_CORES} logical / {_PERF_CORES} perf cores)")

    parallel_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_process_one_support)(
            min_sup=min_sup,
            sup_idx=sup_idx,
            n_sup=len(support_grid),
            df=df,
            output_dir=output_dir,
            confidence_grid=confidence_grid,
            lift_grid_used=lift_grid_used,
            lift_window_lo=lift_window_lo,
            lift_window_hi=lift_window_hi,
        )
        for sup_idx, min_sup in enumerate(support_grid, start=1)
    )

    summary_rows = [row for worker_rows in parallel_results for row in worker_rows]

    print(f"\n{'='*70}")
    print("  > Exploration done, building summary...")

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=['Number_of_Rules', 'Max_Lift'], ascending=[False, False]
        ).reset_index(drop=True)

    summary_df.to_csv(output_dir / "summary.csv", index=False)
    combos_with_rules = int((summary_df['Number_of_Rules'] > 0).sum()) if not summary_df.empty else 0

    with open(output_dir / "exploration_summary.txt", 'w') as f:
        f.write("FULL EXPLORATION SUMMARY\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Parameter Grids:\n")
        f.write(f"  Support    : {len(support_grid)} values [{support_grid[0]} ... {support_grid[-1]}, step={sup_delta}]\n")
        f.write(f"  Confidence : {len(confidence_grid)} values [{confidence_grid[0]} ... {confidence_grid[-1]}, step={conf_delta}]\n")
        f.write(f"  Lift       : {len(lift_grid_used)} values used (of {len(lift_grid)} total, "
                f"neutral window [{lift_window_lo}, {lift_window_hi}] excluded), step={lift_delta}\n\n")
        f.write("Results:\n")
        f.write(f"  Total combinations : {total_combos:,}\n")
        f.write(f"  With >= 1 rule     : {combos_with_rules:,}\n\n")
        if not summary_df.empty and combos_with_rules > 0:
            best = summary_df.iloc[0]
            f.write("Best combination (most rules, then highest lift):\n")
            f.write(f"{'-'*60}\n")
            f.write(f"  Support:         {best['Support']}\n")
            f.write(f"  Confidence:      {best['Confidence']}\n")
            f.write(f"  Lift threshold:  {best['Lift_threshold']}\n")
            f.write(f"  Number of Rules: {int(best['Number_of_Rules'])}\n")
            f.write(f"  Max Lift:        {best['Max_Lift']}\n")
            f.write(f"  Avg Lift:        {best['Avg_Lift']}\n")

    print("  > Cleaning up empty folders...")
    removed_conf, removed_sup = cleanup_empty_folders(output_dir)
    print(f"  > Removed {removed_conf} conf folder(s) and {removed_sup} sup folder(s)")

    plot_heatmaps(summary_df, output_dir,
                  lift_neutral_half_window=lift_neutral_half_window,
                  lift_delta=lift_delta)

    print(f"  > summary saved to {output_dir / 'summary.csv'}")
    print(f"  > total combinations: {total_combos:,}  |  with rules: {combos_with_rules:,}")

    return summary_df


# ---------------------------------------------------------------------------
# Auto-calibration
# ---------------------------------------------------------------------------

def calibrate_parameters(encoded_df, sup_delta=0.02, lift_delta=0.05,
                          conf_delta=0.05, conf_min_floor=0.05,
                          conf_max=1.00):
    """
    Auto-calibrate sup_min, sup_max, lift_max and conf_min from item frequencies.

    lift_min is always 0.0 so that negative correlations are included in the
    grid from the start.

    Returns None if no 2-itemsets can be formed (transactions too sparse).
    """
    print("  > Calibrating parameters from item frequencies...")

    item_supports = encoded_df.mean().sort_values()
    print("  > Item supports:")
    for item, sup in item_supports.items():
        print(f"    {item}: {sup:.4f}")

    if len(item_supports) < 2:
        print("  > Warning: fewer than 2 items -- cannot form pairwise rules.")
        return None

    rarest = item_supports.iloc[0]
    second = item_supports.iloc[1]
    freq_2 = item_supports.iloc[-2]

    raw_sup_min = rarest * second
    sup_min = max(round(np.floor(raw_sup_min / sup_delta) * sup_delta, 4), sup_delta)

    scan_grid = np.round(np.arange(sup_min, freq_2 + sup_delta / 2, sup_delta), 4)
    sup_max = sup_min
    prev_had_2itemsets = False
    fi_first_with_2itemsets = None

    for t in scan_grid:
        fi = fpgrowth(encoded_df, min_support=t, use_colnames=True)
        if fi.empty:
            break
        has_2itemsets = (fi['itemsets'].apply(len).max() >= 2) if not fi.empty else False
        if has_2itemsets:
            if fi_first_with_2itemsets is None:
                fi_first_with_2itemsets = fi
            sup_max = t
            prev_had_2itemsets = True
        elif prev_had_2itemsets:
            break

    raw_lift_max = 1.0 / rarest
    lift_max = min(round(np.ceil(raw_lift_max * 2) / 2, 1), 10.0)

    if not prev_had_2itemsets:
        print(f"  > Warning: no 2-itemsets found at any support threshold for this k.")
        print(f"    Transactions are too sparse to generate association rules.")
        print("-" * 50)
        return None

    calibrated_conf_min = conf_min_floor
    max_conf = None
    try:
        if fi_first_with_2itemsets is not None and not fi_first_with_2itemsets.empty:
            rules_probe = association_rules(
                fi_first_with_2itemsets, metric="confidence", min_threshold=0.01
            )
            if not rules_probe.empty:
                max_conf = rules_probe['confidence'].max()
                calibrated = round(np.floor(max_conf / conf_delta) * conf_delta, 4)
                calibrated_conf_min = max(calibrated, conf_min_floor)
                print(f"  > conf_min calibrated to {calibrated_conf_min} "
                      f"(max observed confidence={max_conf:.4f}, floor={conf_min_floor})")
            else:
                print(f"  > Note: no rules at conf=0.01 for sup_min={sup_min} "
                      f"-- conf_min stays at floor={conf_min_floor}")
    except Exception:
        pass

    params = {
        # grid parameters
        'sup_min':           sup_min,
        'sup_max':           sup_max,
        'sup_delta':         sup_delta,
        'conf_min':          calibrated_conf_min,
        'conf_max':          conf_max,
        'conf_delta':        conf_delta,
        'lift_min':          0.0,   # always 0.0 -- negative correlations included
        'lift_max':          lift_max,
        'lift_delta':        lift_delta,
        # diagnostic values for calibration_log
        'raw_sup_min':       round(raw_sup_min, 6),
        'raw_lift_max':      round(raw_lift_max, 4),
        'max_conf_observed': round(max_conf, 4) if max_conf is not None else None,
        'item_supports':     item_supports.round(4).to_dict(),
    }

    print(f"  > calibrated: sup_min={sup_min} (raw={raw_sup_min:.4f}), "
          f"sup_max={sup_max}, conf_min={calibrated_conf_min} (calibrated from data), "
          f"lift_max={lift_max} (raw ceiling={raw_lift_max:.2f})")
    print("-" * 50)

    return params


# ---------------------------------------------------------------------------
# Calibration log writer
# ---------------------------------------------------------------------------

def _write_calibration_log(k_dir, k, n_transactions, item_supports,
                            params, auto_calibrate, manual_params=None):
    """
    Write per-k calibration artefacts:
      item_supports.csv     -- one row per feature (sorted ascending)
      calibration_log.txt   -- human-readable parameter summary

    Called for every k, including skipped ones (params=None), so the log
    always documents why a k was included or excluded.
    """
    k_dir = Path(k_dir)
    k_dir.mkdir(parents=True, exist_ok=True)

    sup_df = pd.DataFrame({
        'item':        list(item_supports.index),
        'support_raw': [f"{v:.4f}" for v in item_supports.values],
        'support_pct': [f"{v * 100:.2f}" for v in item_supports.values],
    })
    # Both columns are pre-formatted as strings so each has its own fixed
    # precision: support_raw always has 4 decimal places (e.g. 0.3790),
    # support_pct always has 2 (e.g. 37.90).  No float_format needed.
    sup_df.to_csv(k_dir / "item_supports.csv", index=False)

    with open(k_dir / "calibration_log.txt", 'w') as f:
        f.write(f"CALIBRATION LOG -- k={k}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Transactions : {n_transactions:,}\n")
        f.write(f"Items        : {len(item_supports)}\n\n")
        f.write(f"Item Supports (sorted ascending):\n{'-'*40}\n")
        for item, sup in item_supports.items():
            f.write(f"  {item:<12} {sup:.4f}\n")
        f.write("\n")

        if not auto_calibrate:
            f.write("Mode: MANUAL (auto_calibrate=False)\n")
            if manual_params:
                f.write(f"  sup_min  : {manual_params.get('sup_min')}\n")
                f.write(f"  sup_max  : {manual_params.get('sup_max')}\n")
                f.write(f"  conf_min : {manual_params.get('conf_min')}\n")
                f.write(f"  lift_max : {manual_params.get('lift_max')}\n")
            return

        f.write("Mode: AUTO-CALIBRATED\n\n")

        if params is None:
            f.write("Result: SKIPPED\n")
            f.write("Reason: no 2-itemsets found at any support threshold.\n")
            f.write("  The transactions are too sparse to generate association rules.\n")
            f.write("  This typically happens at low k values where most samples have\n")
            f.write("  only 1 CF neighbor, producing single-item transactions.\n")
            return

        f.write("Calibrated Parameters:\n")
        f.write(f"  sup_min          : {params['sup_min']}  "
                f"(raw product = {params['raw_sup_min']:.6f})\n")
        f.write(f"  sup_max          : {params['sup_max']}\n")
        f.write(f"  sup_delta        : {params['sup_delta']}\n")
        f.write(f"  conf_min         : {params['conf_min']}  "
                f"(max observed = {params['max_conf_observed']})\n")
        f.write(f"  conf_max         : {params['conf_max']}\n")
        f.write(f"  conf_delta       : {params['conf_delta']}\n")
        f.write(f"  lift_min         : {params['lift_min']}  "
                f"(0.0 -- negative correlations included)\n")
        f.write(f"  lift_max         : {params['lift_max']}  "
                f"(raw ceiling = {params['raw_lift_max']:.4f}, capped at 10.0)\n")
        f.write(f"  lift_delta       : {params['lift_delta']}\n")


# ---------------------------------------------------------------------------
# K-comparison experiment
# ---------------------------------------------------------------------------

def run_k_comparison(k_labels_map, output_dir,
                     auto_calibrate=True,
                     sup_min=0.02,  sup_max=0.16,  sup_delta=0.02,
                     conf_min=0.05, conf_max=1.00,  conf_delta=0.05,
                     lift_min=0.0,  lift_max=2.5,   lift_delta=0.05,
                     lift_neutral_half_window=0.25):
    """
    Run explore_association_rules for each k and produce a cross-k comparison.

    lift_min=0.0 (default) ensures negative correlations are included in the
    grid for every k, both in auto and manual mode.
    """
    output_dir = Path(output_dir)
    comp_dir = output_dir / "k_comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"K-VARIATION EXPERIMENT -- {len(k_labels_map)} values of k")
    print(f"{'='*70}")
    print(f"  > k values: {sorted(k_labels_map.keys())}")
    print(f"  > auto_calibrate: {auto_calibrate}")
    print("-" * 50)

    k_summaries = {}
    comparison_rows = []

    for k in sorted(k_labels_map.keys()):
        labels_csv = Path(k_labels_map[k])
        k_dir = output_dir / f"k_{k}"
        k_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  k = {k}  |  {labels_csv.name}")
        print(f"{'='*70}")

        if not labels_csv.exists():
            print(f"  > file not found: {labels_csv}")
            comparison_rows.append({
                'k': k, 'skipped_reason': 'file_not_found',
                'n_transactions': None, 'n_items': None,
                'rarest_item_support': None, 'sup_min_used': None,
                'sup_max_used': None, 'lift_max_used': None,
                'conf_min_used': None, 'summary_rows': 0,
                'combos_with_rules': 0, 'max_rules_any_combo': 0,
                'avg_lift_best_combo': None, 'max_lift_observed': None,
            })
            continue

        df_encoded = extract_labels(labels_csv)
        item_supports = df_encoded.mean().sort_values()

        if auto_calibrate:
            params = calibrate_parameters(
                encoded_df=df_encoded,
                sup_delta=sup_delta,
                lift_delta=lift_delta,
                conf_delta=conf_delta,
                conf_min_floor=conf_min,
                conf_max=conf_max,
            )
            _write_calibration_log(
                k_dir=k_dir, k=k,
                n_transactions=len(df_encoded),
                item_supports=item_supports,
                params=params,
                auto_calibrate=True,
            )

            if params is None:
                print(f"  > Skipping k={k} -- not enough co-occurrences to generate rules.")
                comparison_rows.append({
                    'k': k, 'skipped_reason': 'too_sparse',
                    'n_transactions': len(df_encoded),
                    'n_items': df_encoded.shape[1],
                    'rarest_item_support': round(item_supports.iloc[0], 4),
                    'sup_min_used': None, 'sup_max_used': None,
                    'lift_max_used': None, 'conf_min_used': None,
                    'summary_rows': 0, 'combos_with_rules': 0,
                    'max_rules_any_combo': 0, 'avg_lift_best_combo': None,
                    'max_lift_observed': None,
                })
                continue

            k_sup_min  = params['sup_min']
            k_sup_max  = params['sup_max']
            k_lift_max = params['lift_max']
            k_conf_min = params['conf_min']
            k_lift_min = params['lift_min']   # always 0.0
        else:
            k_sup_min  = sup_min
            k_sup_max  = sup_max
            k_lift_max = lift_max
            k_conf_min = conf_min
            k_lift_min = lift_min

            _write_calibration_log(
                k_dir=k_dir, k=k,
                n_transactions=len(df_encoded),
                item_supports=item_supports,
                params=None,
                auto_calibrate=False,
                manual_params={
                    'sup_min': k_sup_min, 'sup_max': k_sup_max,
                    'conf_min': k_conf_min, 'lift_max': k_lift_max,
                },
            )

        summary_df = explore_association_rules(
            df=df_encoded,
            output_dir=k_dir,
            sup_min=k_sup_min,   sup_max=k_sup_max,   sup_delta=sup_delta,
            conf_min=k_conf_min, conf_max=conf_max,   conf_delta=conf_delta,
            lift_min=k_lift_min, lift_max=k_lift_max, lift_delta=lift_delta,
            lift_neutral_half_window=lift_neutral_half_window,
        )

        k_summaries[k] = summary_df

        has_rules_col = not summary_df.empty and 'Number_of_Rules' in summary_df.columns
        with_rules = summary_df[summary_df['Number_of_Rules'] > 0] if has_rules_col else pd.DataFrame()
        comparison_rows.append({
            'k':                   k,
            'skipped_reason':      '',
            'n_transactions':      len(df_encoded),
            'n_items':             df_encoded.shape[1],
            'rarest_item_support': round(item_supports.iloc[0], 4),
            'sup_min_used':        k_sup_min,
            'sup_max_used':        k_sup_max,
            'lift_max_used':       k_lift_max,
            'conf_min_used':       k_conf_min,
            'summary_rows':        len(summary_df),
            'combos_with_rules':   len(with_rules),
            'max_rules_any_combo': int(summary_df['Number_of_Rules'].max()) if has_rules_col else 0,
            'avg_lift_best_combo': round(with_rules['Avg_Lift'].max(), 4) if not with_rules.empty else 0.0,
            'max_lift_observed':   round(summary_df['Max_Lift'].max(), 4) if has_rules_col else 0.0,
        })

    print(f"\n{'='*70}")
    print("  > Building cross-k comparison...")

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(comp_dir / "k_comparison_summary.csv", index=False)

    with open(comp_dir / "k_comparison_summary.txt", "w") as f:
        f.write("K-VARIATION EXPERIMENT SUMMARY\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"k values tested: {sorted(k_labels_map.keys())}\n")
        f.write(f"auto_calibrate:  {auto_calibrate}\n\n")

        skipped = comp_df[comp_df['skipped_reason'] != ''] if not comp_df.empty else pd.DataFrame()
        ran     = comp_df[comp_df['skipped_reason'] == ''] if not comp_df.empty else pd.DataFrame()

        if not skipped.empty:
            f.write(f"Skipped k values ({len(skipped)}):\n{'-'*40}\n")
            for _, row in skipped.iterrows():
                reason = row['skipped_reason']
                if reason == 'too_sparse':
                    detail = (f"no 2-itemsets found "
                              f"(n_transactions={int(row['n_transactions']):,}, "
                              f"rarest_support={row['rarest_item_support']})")
                elif reason == 'file_not_found':
                    detail = "input CSV not found"
                else:
                    detail = reason
                f.write(f"  k={int(row['k'])}: {detail}\n")
            f.write("\n")

        f.write(f"Results per k (processed only):\n{'-'*60}\n")
        if not ran.empty:
            f.write(ran.drop(columns='skipped_reason').to_string(index=False))
        else:
            f.write("  No k values produced results.\n")
        f.write("\n\n")

        if not ran.empty and ran['max_rules_any_combo'].max() > 0:
            best_k = ran.loc[ran['max_rules_any_combo'].idxmax(), 'k']
            f.write(f"Most rules: k={best_k} ({ran['max_rules_any_combo'].max()} rules at best combo)\n")
        else:
            f.write("No rules found for any k value.\n")

    print("  > Saved k_comparison_summary.csv and .txt")

    # cross-k heatmaps
    all_summaries = []
    for k, sdf in k_summaries.items():
        tmp = sdf.copy()
        tmp['k'] = k
        all_summaries.append(tmp)

    if not all_summaries:
        print('  > No rules found in any k -- skipping cross-k heatmaps.')
        print(f'  > Cross-k comparison saved to {comp_dir}/')
        return k_summaries

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        combined = pd.concat(all_summaries, ignore_index=True)

    if combined.empty or 'Lift_threshold' not in combined.columns:
        print('  > No rules found in any k -- skipping cross-k heatmaps.')
        print(f'  > Cross-k comparison saved to {comp_dir}/')
        return k_summaries

    combined['Lift_display'] = (
        (combined['Lift_threshold'] / lift_delta).round() * lift_delta
    ).round(4)

    # Unified neutral window.
    neutral_lo, neutral_hi = _neutral_window(lift_neutral_half_window)

    heatmap_configs = [
        ('Support',      'k_support'),
        ('Confidence',   'k_confidence'),
        ('Lift_display', 'k_lift'),
    ]

    for x_col, suffix in heatmap_configs:
        x_label = 'Lift' if 'lift' in suffix else x_col

        pivot = (
            combined.groupby(['k', x_col])['Number_of_Rules']
            .max()
            .unstack(level=x_col)
            .sort_index(ascending=False)
            .fillna(0)
            .astype(int)
        )

        if 'lift' in suffix:
            # Mask neutral window. Negative-correlation columns survive.
            pivot = pivot.loc[:, ~pivot.columns.to_series().between(
                neutral_lo, neutral_hi, inclusive='both')]
            if (pivot != 0).any(axis=0).any():
                last_nonzero = int(np.where((pivot != 0).any(axis=0).values)[0].max())
                pivot = pivot.iloc[:, :last_nonzero + 1]

        n_cols = len(pivot.columns)
        n_rows = len(pivot.index)
        fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.75), max(4, n_rows * 0.6)))

        img = ax.imshow(pivot.values, aspect='auto', cmap='YlOrBr', interpolation='nearest')

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(
            [f"{v:.2f}" if isinstance(v, float) else str(v) for v in pivot.columns],
            rotation=40, ha='right', fontsize=8
        )
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f"k={v}" for v in pivot.index], fontsize=9)

        ax.set_xlabel(x_label, fontsize=11, labelpad=8)
        ax.set_ylabel('k', fontsize=11, labelpad=8)
        ax.set_title(
            f"Max Number of Rules -- k vs {x_label}\n"
            f"(darker = more rules; max over the other two parameters)",
            fontsize=11, pad=14
        )

        max_val = pivot.values.max() if pivot.values.max() > 0 else 1
        for ri in range(n_rows):
            for ci in range(n_cols):
                val = pivot.values[ri, ci]
                if val > 0:
                    txt_color = 'white' if (val / max_val) > 0.55 else 'black'
                    ax.text(ci, ri, str(val), ha='center', va='center', fontsize=7, color=txt_color)

        cbar = plt.colorbar(img, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label('Number of Rules', fontsize=9)
        plt.tight_layout()

        fig.savefig(comp_dir / f"heatmap_{suffix}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    > saved k_comparison/heatmap_{suffix}.png")

    print(f"  > Cross-k comparison saved to {comp_dir}/")
    return k_summaries


# ---------------------------------------------------------------------------
# Experiment labelling
# ---------------------------------------------------------------------------

def _experiment_label(auto_calibrate,
                       sup_min, sup_max, sup_delta,
                       conf_min, conf_max, conf_delta,
                       lift_min, lift_max, lift_delta,
                       lift_neutral_half_window):
    """
    Build a human-readable folder name for one experiment configuration.

    conf_min shown here is the floor value, not the per-k calibrated value --
    the label identifies the configuration, not derived per-k parameters.
    """
    def fmt(v):
        return f"{v:.2f}"

    if auto_calibrate:
        prefix    = "auto"
        sup_part  = f"sup=auto_d{fmt(sup_delta)}"
        lift_part = f"lift=auto_d{fmt(lift_delta)}_w{fmt(lift_neutral_half_window)}"
    else:
        prefix    = "manual"
        sup_part  = f"sup={fmt(sup_min)}-{fmt(sup_max)}_d{fmt(sup_delta)}"
        lift_part = f"lift={fmt(lift_min)}-{fmt(lift_max)}_d{fmt(lift_delta)}_w{fmt(lift_neutral_half_window)}"

    conf_part = f"conf={fmt(conf_min)}-{fmt(conf_max)}_d{fmt(conf_delta)}"

    return f"{prefix}_{sup_part}_{conf_part}_{lift_part}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"  > M2 parallel backend -- {_CPU_CORES} logical cores / {_PERF_CORES} perf cores (joblib loky)")

    if Path("/content").exists():
        base_dir = Path("/content")
    else:
        base_dir = Path(__file__).resolve().parent.parent

    results_dir = base_dir / "results"

    regions  = ['northeast', 'south']
    k_values = [1, 3, 5, 7]

    # ------------------------------------------------------------------ #
    # Experiment configuration                                            #
    # ------------------------------------------------------------------ #
    AUTO_CALIBRATE        = True
    SUP_MIN, SUP_MAX      = 0.02, 0.50   # fallback if AUTO_CALIBRATE=False
    SUP_DELTA             = 0.02
    CONF_MIN, CONF_MAX    = 0.05, 1.00   # CONF_MIN is conf_min_floor when calibrating
    CONF_DELTA            = 0.05
    LIFT_MIN, LIFT_MAX    = 0.0,  5.0    # 0.0 -- negative correlations (lift < 0.75) included
    LIFT_DELTA            = 0.05
    LIFT_NEUTRAL_HALF_WIN = 0.25         # excludes [0.75, 1.25] -- no analytical signal
    # ------------------------------------------------------------------ #

    exp_label = _experiment_label(
        auto_calibrate=AUTO_CALIBRATE,
        sup_min=SUP_MIN,   sup_max=SUP_MAX,   sup_delta=SUP_DELTA,
        conf_min=CONF_MIN, conf_max=CONF_MAX, conf_delta=CONF_DELTA,
        lift_min=LIFT_MIN, lift_max=LIFT_MAX, lift_delta=LIFT_DELTA,
        lift_neutral_half_window=LIFT_NEUTRAL_HALF_WIN,
    )

    for region in regions:
        important_features_dir = results_dir / region / "important_features"
        ar_output_dir = results_dir / region / "association_rules" / exp_label
        ar_output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print(f"ASSOCIATION RULES -- {region.upper()}")
        print(f"Experiment: {exp_label}")
        print("="*70 + "\n")

        k_labels_map = {}
        for k in k_values:
            p_agg  = important_features_dir / f"k_{k}" / "aggregated_labels_by_sample.csv"
            p_orig = important_features_dir / f"k_{k}" / "labels_only_unique.csv"
            if p_agg.exists():
                k_labels_map[k] = (p_agg, "aggregated (preferred)")
            elif p_orig.exists():
                k_labels_map[k] = (p_orig, "original (fallback)")

        if not k_labels_map:
            print(f"  > No labels files found under {important_features_dir}")
            print(f"    Run feature_importance.py first.")
            continue

        ks_with_agg  = sum(1 for _, (_, src) in k_labels_map.items() if "aggregated" in src)
        ks_with_orig = len(k_labels_map) - ks_with_agg
        print(f"  > Found labels for k = {sorted(k_labels_map.keys())}")
        print(f"    - {ks_with_agg} k-values using aggregated format (preferred)")
        if ks_with_orig:
            print(f"    - {ks_with_orig} k-values using original format (fallback)")

        k_paths_map = {k: path for k, (path, _) in k_labels_map.items()}

        run_k_comparison(
            k_labels_map=k_paths_map,
            output_dir=ar_output_dir,
            auto_calibrate=AUTO_CALIBRATE,
            sup_min=SUP_MIN,   sup_max=SUP_MAX,   sup_delta=SUP_DELTA,
            conf_min=CONF_MIN, conf_max=CONF_MAX, conf_delta=CONF_DELTA,
            lift_min=LIFT_MIN, lift_max=LIFT_MAX, lift_delta=LIFT_DELTA,
            lift_neutral_half_window=LIFT_NEUTRAL_HALF_WIN,
        )

    print("\n" + "="*70)
    print("Done.")
    print("="*70 + "\n")