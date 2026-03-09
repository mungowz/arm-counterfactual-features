"""
Association Rule Mining on Counterfactual Drivers
===================================================

Pipeline (from feature_importance.py):
  1. BoCSoR.explain()  → finds one differing feature per (sample, CF_neighbor) pair
                        → transactions_values.csv
  2. extract_labels()  → extracts change labels (e.g., "SCHL=changes")
                        → labels_only_unique.csv (one row per sample-CF pair)
  3. aggregate_drivers_by_sample()  → consolidates all drivers per sample across
                                      all its CF neighbors into one transaction
                                      → aggregated_labels_by_sample.csv

This module (macroscopic_experiment_association_rules.py):
  - Reads aggregated_labels_by_sample.csv (preferred) or labels_only_unique.csv
  - Runs FP-Growth to discover itemsets and association rules
  - Auto-calibrates support/confidence/lift thresholds based on transaction sparsity
  - Keeps both directions of each rule (A→B and B→A) to support directional analysis
  - Generates heatmaps and cross-k comparison summaries

Example rule discovered:
  When SCHL (education level) changes on the decision boundary,
  OCCP (occupation) also changes 70% of the time (confidence=0.70, lift=2.1)

Input CSV format (aggregated_labels_by_sample.csv):
  Sample_ID | Labels          | Num_Labels | Num_CF_Neighbors
  1284      | ['OCCP']        | 1          | 1
  1442      | ['WKHP']        | 1          | 1
  ...

  The 'Labels' column contains Python list literals — parsed with ast.literal_eval.
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

# headless backend — needed when running on a server without a display
matplotlib.use('Agg')

# Total logical cores — used for logging only.
_CPU_CORES = os.cpu_count() or 1

# PERF OPT 2 — platform-aware parallelism.
#
# On macOS (M-series), Python uses 'spawn' to start worker processes — each
# loky worker must reimport all modules before doing real work.  The M2 also
# has P-cores (fast) and E-cores (slow for CPU-bound tasks); running FP-Growth
# on E-cores creates a straggler bottleneck.  We therefore cap n_jobs at the
# P-core count so all workers land on fast cores, and also cap it at the number
# of tasks to never spawn idle workers.
#
# On Linux (including Colab) Python uses 'fork' — spawn overhead is zero and
# all cores are equivalent, so we use all of them (n_jobs = _CPU_CORES).
#
# On Windows, spawn is used like macOS but all cores are homogeneous (no E/P
# split on most Intel/AMD chips), so we use all cores there too.
#
# You can override _PERF_CORES manually if needed:
#   M2 base → 4    M2 Pro → 6 or 8    M2 Max → 8–12    M2 Ultra → 16
import platform as _platform
if _platform.system() == "Darwin":          # macOS — M-series has P/E cores
    _PERF_CORES = 4  # M2 base default — adjust for Pro / Max / Ultra
else:                                        # Linux (Colab) or Windows
    _PERF_CORES = _CPU_CORES                 # use all available cores
del _platform


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
    (e.g. "['OCCP', 'SCHL']") — parsed with ast.literal_eval.

    FIX: removed dead-code 'Drivers' branch that was never reachable since
    aggregated_labels_by_sample.csv uses 'Labels', not 'Drivers'.
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
    After filtering, some conf_* folders end up with no rules — just delete them.
    If a sup_* folder loses all its conf_* subfolders, delete that too.
    Returns (n_conf_removed, n_sup_removed).

    FIX: added try/except around pd.read_csv to handle zero-byte or corrupted
    rules.csv files (pandas raises EmptyDataError, not a plain exception).
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
                    # zero-byte file, parse error, EmptyDataError, etc.
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
    Quick in-memory grid search, no files saved. Useful for a fast sanity check
    before running the full exploration. Use explore_association_rules() if you
    want everything written to disk.
    """
    print(f"\n{'='*70}")
    print("GRID SEARCH: FP-GROWTH (IN-MEMORY)")
    print(f"{'='*70}")

    support_grid    = np.round(np.arange(sup_min,  sup_max  + sup_delta  / 2, sup_delta),  4)
    confidence_grid = np.round(np.arange(conf_min, conf_max + conf_delta / 2, conf_delta), 4)
    lift_grid       = np.round(np.arange(lift_min, lift_max + lift_delta / 2, lift_delta), 4)

    lift_neutral_lo = round(1.0 - lift_neutral_half_window, 4)
    lift_neutral_hi = round(1.0 + lift_neutral_half_window, 4)

    print(f"  > support grid:    {support_grid}")
    print(f"  > confidence grid: {confidence_grid}")
    print(f"  > lift grid:       {lift_grid}")
    print(f"  > neutral window:  [{lift_neutral_lo}, {lift_neutral_hi}]")
    print("-" * 50)

    results = []

    for min_sup in support_grid:
        frequent_itemsets = fpgrowth(df, min_support=min_sup, use_colnames=True)
        if len(frequent_itemsets) == 0:
            continue

        # PERF OPT 1 — single association_rules() call, then filter by confidence.
        try:
            all_rules_base = association_rules(
                frequent_itemsets, metric="confidence", min_threshold=float(confidence_grid[0])
            )
        except ValueError:
            continue

        if len(all_rules_base) == 0:
            continue

        all_rules_base = all_rules_base[
            (all_rules_base['lift'] < lift_neutral_lo) | (all_rules_base['lift'] > lift_neutral_hi)
        ]
        if len(all_rules_base) == 0:
            continue

        for min_conf in confidence_grid:
            rules = all_rules_base[all_rules_base['confidence'] >= min_conf]
            if len(rules) == 0:
                continue

            # Both directions of each rule (A→B and B→A) are kept intentionally —
            # they carry different confidence values and are both useful for analysis.
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
    3 heatmaps from the summary: support-confidence, support-lift, confidence-lift.
    Each cell = max rules over the third parameter (darker = more rules).

    Lift is always on the x-axis to keep the plots horizontal.
    Values are binned at lift_display_step resolution for readability.

    FIX: the neutral-window exclusion on the lift axis now uses lift_delta
    (the actual filter step) instead of lift_display_step (a display-only
    parameter).  Previously the two were conflated, making the excluded band
    wider than the real filter and hiding columns that contained valid rules.

    Parameters
    ----------
    lift_delta : float
        The step used when building the lift grid — must match the value
        passed to explore_association_rules / run_k_comparison so that the
        neutral window on the heatmap axis matches the actual filter.
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

    # FIX: neutral window uses lift_delta (the actual filter step), not
    # lift_display_step (a cosmetic binning parameter).
    neutral_lo = round(1.0 - lift_delta * round(lift_neutral_half_window / lift_delta), 4)
    neutral_hi = round(1.0 + lift_delta * round(lift_neutral_half_window / lift_delta), 4)

    # (x_col, y_col, file suffix, lift_on_x?)
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
            pivot = pivot.loc[:, ~pivot.columns.to_series().between(neutral_lo, neutral_hi, inclusive='both')]
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
            f"Max Number of Rules — {y_col} vs {x_label}\n"
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

    Called in parallel by Parallel(n_jobs=min(_PERF_CORES, n_tasks), backend='loky').
    Each invocation writes to its own sup_{min_sup}/ subdirectory so there
    are no filesystem conflicts between workers.

    Returns a list of summary-row dicts (one per (conf, lift) combination)
    that the caller flattens into the global summary DataFrame.

    Both directions of symmetric rules (A→B and B→A) are kept — they have
    different confidence values and are both informative for directional analysis.

    FIX: conviction=inf (produced by mlxtend when confidence=1.0) is replaced
    with np.nan before saving to CSV to avoid silent propagation of inf in
    downstream reads.
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

    # PERF OPT 1 — call association_rules() ONCE at the minimum confidence
    # threshold, then derive every higher-confidence subset by pandas filtering.
    #
    # Previously the code called association_rules() once per confidence step
    # (e.g. 20 calls at conf=0.05, 0.10, …, 1.00).  Each call recomputes all
    # metrics (support, confidence, lift, leverage, conviction) from scratch on
    # the full frequent-itemset table — O(|FI|²) work repeated N_conf times.
    # Since rules at conf=0.10 are a strict superset of rules at conf=0.15,
    # one call at conf_min is sufficient; the rest are free pandas boolean masks.
    try:
        all_rules_base = association_rules(
            frequent_itemsets, metric="confidence", min_threshold=float(confidence_grid[0])
        )
    except ValueError:
        return local_summary_rows

    if len(all_rules_base) == 0:
        return local_summary_rows

    # Apply neutral-window filter and sort once — result is shared across all
    # confidence thresholds because lift is independent of the confidence filter.
    all_rules_base = all_rules_base[
        (all_rules_base['lift'] < lift_window_lo) | (all_rules_base['lift'] > lift_window_hi)
    ]
    all_rules_base = all_rules_base.sort_values('lift', ascending=False).reset_index(drop=True)

    if len(all_rules_base) == 0:
        return local_summary_rows

    for min_conf in confidence_grid:
        conf_label = f"{min_conf:.2f}"
        conf_dir = sup_dir / f"conf_{conf_label}"

        # Cheap pandas filter — no recomputation of metrics
        rules = all_rules_base[all_rules_base['confidence'] >= min_conf].reset_index(drop=True)

        if len(rules) == 0:
            continue

        # Both directions of each rule (A→B and B→A) are kept intentionally —
        # they carry different confidence values and are both useful for analysis.

        # compact output
        fmt = pd.DataFrame()
        fmt['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
        fmt['consequents'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
        fmt['support']     = rules['support'].round(4)
        fmt['confidence']  = rules['confidence'].round(4)
        fmt['lift']        = rules['lift'].round(4)

        # detailed output with all mlxtend metrics and itemset lengths
        # FIX: conviction=inf (mlxtend artefact when confidence=1.0) replaced
        # with NaN so downstream CSV reads don't silently propagate infinity.
        conviction_vals = rules['conviction'].replace([np.inf, -np.inf], np.nan)

        det = pd.DataFrame()
        det['antecedents']        = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
        det['consequents']        = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
        det['antecedent_length']  = rules['antecedents'].apply(len)
        det['consequent_length']  = rules['consequents'].apply(len)
        det['rule_length']        = det['antecedent_length'] + det['consequent_length']
        det['antecedent support'] = rules['antecedent support'].values
        det['consequent support'] = rules['consequent support'].values
        det['support']            = rules['support'].values
        det['confidence']         = rules['confidence'].values
        det['lift']               = rules['lift'].values
        det['leverage']           = rules['leverage'].values
        det['conviction']         = conviction_vals.values

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
            f.write(f"  Avg Support:     {rules['support'].mean():.4f}\n")
            f.write(f"  Avg Confidence:  {rules['confidence'].mean():.4f}\n")
            f.write(f"  Avg Lift:        {rules['lift'].mean():.4f}\n")
            f.write(f"  Lift Range:      {rules['lift'].min():.4f} - {rules['lift'].max():.4f}\n")
            f.write(f"  Avg Rule Length: {det['rule_length'].mean():.2f}\n\n")
            # FIX: show conviction as NaN where infinite (confidence=1.0 rules)
            n_inf_conviction = (conviction_vals.isna()).sum()
            if n_inf_conviction > 0:
                f.write(f"  Note: {n_inf_conviction} rule(s) have confidence=1.0 "
                        f"(conviction=inf → saved as NaN)\n\n")
            f.write(f"Top 10 Rules (by Lift):\n{'-'*60}\n")
            for idx, row in fmt.head(10).iterrows():
                f.write(f"{idx+1}. {row['antecedents']} => {row['consequents']}\n")
                f.write(f"   support={row['support']:.4f} | confidence={row['confidence']:.4f} | lift={row['lift']:.4f}\n\n")

        print(f"    > [conf={min_conf}] {len(rules)} rules saved to {conf_dir.relative_to(output_dir)}/")

        for min_lift in lift_grid_used:
            filtered = rules[rules['lift'] >= min_lift]
            n = len(filtered)
            rl = (filtered['antecedents'].apply(len) + filtered['consequents'].apply(len)) if n > 0 else pd.Series(dtype=float)

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

    For each (sup, conf) pair:
      - run FP-Growth to get frequent itemsets
      - generate rules at conf_min (single call), then filter by confidence
      - drop near-independent rules (lift in neutral window)
      - keep both A→B and B→A directions (different confidence, both informative)
      - save rules.csv, rules_detailed.csv, summary.txt

    Also saves a global summary.csv + exploration_summary.txt, cleans up
    empty folders, and generates the heatmaps.

    Output layout:
        output_dir/
        ├── summary.csv
        ├── exploration_summary.txt
        ├── heatmaps/
        ├── sup_0.02/
        │   ├── frequent_itemsets.csv
        │   ├── frequent_itemsets_summary.txt
        │   ├── conf_{conf_min}/
        │   │   ├── rules.csv
        │   │   ├── rules_detailed.csv
        │   │   └── summary.txt
        │   └── ...
        └── ...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    support_grid    = np.round(np.arange(sup_min,  sup_max  + sup_delta  / 2, sup_delta),  4)
    confidence_grid = np.round(np.arange(conf_min, conf_max + conf_delta / 2, conf_delta), 4)
    lift_grid       = np.round(np.arange(lift_min, lift_max + lift_delta / 2, lift_delta), 4)

    lift_window_lo = round(1.0 - lift_neutral_half_window, 4)
    lift_window_hi = round(1.0 + lift_neutral_half_window, 4)

    lift_grid_used = [v for v in lift_grid if not (lift_window_lo <= v <= lift_window_hi)]
    total_combos   = len(support_grid) * len(confidence_grid) * len(lift_grid_used)

    print(f"\n{'='*70}")
    print("FULL EXPLORATION: FP-GROWTH ASSOCIATION RULES")
    print(f"{'='*70}")
    print(f"  > support    : {len(support_grid)} values [{support_grid[0]} ... {support_grid[-1]}, step={sup_delta}]")
    print(f"  > confidence : {len(confidence_grid)} values [{confidence_grid[0]} ... {confidence_grid[-1]}, step={conf_delta}]")
    print(f"  > lift       : {len(lift_grid_used)} values used (of {len(lift_grid)} total, "
          f"{len(lift_grid)-len(lift_grid_used)} skipped — neutral window [{lift_window_lo}, {lift_window_hi}]), step={lift_delta}")
    print(f"  > total combinations: {total_combos:,}")
    print("-" * 50)

    # PERF OPT 2 — cap n_jobs at _PERF_CORES (avoids E-core straggler on M2)
    # and at the number of tasks (avoids spawning idle workers when the grid
    # is smaller than the core count, which is common with auto-calibration).
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

    # FIX: pass lift_delta so plot_heatmaps uses the correct neutral window step
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
    Auto-calibrate sup_min, sup_max, lift_max and conf_min from the actual
    item frequencies. Call this before explore_association_rules() when k
    changes so the grid always covers the right range without manual tuning.

    sup_min  = expected joint support of the two rarest items (independence),
               rounded down to the nearest sup_delta step
    sup_max  = last threshold where FP-Growth still finds at least one 2-itemset
    lift_max = 1 / support(rarest item), rounded up to nearest 0.5, capped at 10
    conf_min = floor of the max confidence observed at the first threshold with
               2-itemsets, rounded down to nearest conf_delta, floored at
               conf_min_floor

    FIX: the returned dict now includes conf_max and conf_delta so callers can
    use it as a complete, self-contained parameter set without relying on
    external variables.

    Returns None if no 2-itemsets can be formed (transactions too sparse).
    """
    print("  > Calibrating parameters from item frequencies...")

    item_supports = encoded_df.mean().sort_values()
    print("  > Item supports:")
    for item, sup in item_supports.items():
        print(f"    {item}: {sup:.4f}")

    if len(item_supports) < 2:
        print("  > Warning: fewer than 2 items — cannot form pairwise rules.")
        return None

    rarest = item_supports.iloc[0]
    second = item_supports.iloc[1]
    freq_2 = item_supports.iloc[-2]  # second most frequent — upper bound for scan

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
    max_conf = None   # assigned inside the try block; None means not observable
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
                      f"— conf_min stays at floor={conf_min_floor}")
    except Exception:
        pass

    # Include raw pre-rounding values and item supports so the caller can
    # write a complete calibration_log without recomputing anything.
    params = {
        # --- grid parameters (used by explore_association_rules) ---
        'sup_min':           sup_min,
        'sup_max':           sup_max,
        'sup_delta':         sup_delta,
        'conf_min':          calibrated_conf_min,
        'conf_max':          conf_max,
        'conf_delta':        conf_delta,
        'lift_min':          0.0,
        'lift_max':          lift_max,
        'lift_delta':        lift_delta,
        # --- raw / diagnostic values (for calibration_log only) ---
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
    Write per-k calibration artefacts to k_dir/:
      item_supports.csv      — one row per feature: item, support (sorted asc)
      calibration_log.txt    — human-readable summary of item frequencies,
                               calibrated/manual parameters, and raw values

    Called for every k, including skipped ones (params=None), so the log
    always documents why a k was included or excluded.

    Addresses three output gaps identified after first execution:
      Gap 1 — item supports were only printed to terminal, never saved
      Gap 2 — raw pre-rounding calibration values were only printed, never saved
      Gap 3 — skipped k values left no trace in any output file
    """
    k_dir = Path(k_dir)
    k_dir.mkdir(parents=True, exist_ok=True)

    # --- item_supports.csv ---
    sup_df = pd.DataFrame({
        'item':    list(item_supports.index),
        'support': [round(v, 4) for v in item_supports.values],
    })
    sup_df.to_csv(k_dir / "item_supports.csv", index=False)

    # --- calibration_log.txt ---
    with open(k_dir / "calibration_log.txt", 'w') as f:
        f.write(f"CALIBRATION LOG — k={k}\n")
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
        f.write(f"  lift_min         : {params['lift_min']}\n")
        f.write(f"  lift_max         : {params['lift_max']}  "
                f"(raw ceiling = {params['raw_lift_max']:.4f},"
                f" capped at 10.0)\n")
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

    If auto_calibrate=True, sup_min/sup_max/lift_max and conf_min are recomputed
    for each k from the actual item frequencies — recommended since k changes the
    transaction distribution. conf_max/delta and lift_delta are shared across k.

    k_labels_map : dict[int, str | Path]
        Maps each k value to the path of its input CSV file.
        Expected path pattern:
            <results_dir>/<region>/important_features/k_<k>/aggregated_labels_by_sample.csv

    Cross-k outputs saved in output_dir/k_comparison/:
      k_comparison_summary.csv / .txt
      heatmap_k_support.png, heatmap_k_confidence.png, heatmap_k_lift.png
    """
    output_dir = Path(output_dir)
    comp_dir = output_dir / "k_comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"K-VARIATION EXPERIMENT — {len(k_labels_map)} values of k")
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

        # --- Gap 3: file not found → record skipped row and continue ---
        if not labels_csv.exists():
            print(f"  > file not found: {labels_csv}")
            print(f"    skipping k={k}.")
            comparison_rows.append({
                'k':                   k,
                'skipped_reason':      'file_not_found',
                'n_transactions':      None,
                'n_items':             None,
                'rarest_item_support': None,
                'sup_min_used':        None,
                'sup_max_used':        None,
                'lift_max_used':       None,
                'conf_min_used':       None,
                'summary_rows':        0,
                'combos_with_rules':   0,
                'max_rules_any_combo': 0,
                'avg_lift_best_combo': None,
                'max_lift_observed':   None,
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

            # --- Gap 1+2+3: save calibration log for every k (including skipped) ---
            _write_calibration_log(
                k_dir=k_dir,
                k=k,
                n_transactions=len(df_encoded),
                item_supports=item_supports,
                params=params,          # None if too sparse
                auto_calibrate=True,
            )

            if params is None:
                print(f"  > Skipping k={k} — not enough co-occurrences to generate rules.")
                # Gap 3: add skipped row with item supports recorded
                comparison_rows.append({
                    'k':                   k,
                    'skipped_reason':      'too_sparse',
                    'n_transactions':      len(df_encoded),
                    'n_items':             df_encoded.shape[1],
                    'rarest_item_support': round(item_supports.iloc[0], 4),
                    'sup_min_used':        None,
                    'sup_max_used':        None,
                    'lift_max_used':       None,
                    'conf_min_used':       None,
                    'summary_rows':        0,
                    'combos_with_rules':   0,
                    'max_rules_any_combo': 0,
                    'avg_lift_best_combo': None,
                    'max_lift_observed':   None,
                })
                continue

            k_sup_min  = params['sup_min']
            k_sup_max  = params['sup_max']
            k_lift_max = params['lift_max']
            k_conf_min = params['conf_min']
        else:
            k_sup_min  = sup_min
            k_sup_max  = sup_max
            k_lift_max = lift_max
            k_conf_min = conf_min

            # Save calibration log even in manual mode (item supports are always useful)
            _write_calibration_log(
                k_dir=k_dir,
                k=k,
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
            lift_min=lift_min,   lift_max=k_lift_max, lift_delta=lift_delta,
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

    # FIX: guard against empty list (no k produced any rules) before concat
    if not all_summaries:
        print('  > No rules found in any k — skipping cross-k heatmaps.')
        print(f'  > Cross-k comparison saved to {comp_dir}/')
        return k_summaries

    # FIX: use warnings.catch_warnings to suppress pandas DeprecationWarning
    # when concatenating DataFrames that may all be empty.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        combined = pd.concat(all_summaries, ignore_index=True)

    if combined.empty or 'Lift_threshold' not in combined.columns:
        print('  > No rules found in any k — skipping cross-k heatmaps.')
        print(f'  > Cross-k comparison saved to {comp_dir}/')
        return k_summaries

    combined['Lift_display'] = (
        (combined['Lift_threshold'] / lift_delta).round() * lift_delta
    ).round(4)

    # FIX: neutral window for cross-k heatmaps uses lift_delta (consistent
    # with explore_association_rules), not lift_display_step.
    neutral_lo = round(1.0 - lift_delta * round(lift_neutral_half_window / lift_delta), 4)
    neutral_hi = round(1.0 + lift_delta * round(lift_neutral_half_window / lift_delta), 4)

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
            pivot = pivot.loc[:, ~pivot.columns.to_series().between(neutral_lo, neutral_hi, inclusive='both')]
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
            f"Max Number of Rules — k vs {x_label}\n"
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
    Build a human-readable folder name that uniquely identifies one experiment
    configuration, so repeated runs with different parameters never overwrite
    each other.

    Format (auto_calibrate=True):
        auto_sup=auto_d<delta>_conf=<min>-<max>_d<delta>_lift=auto_d<delta>_w<window>

    Format (auto_calibrate=False):
        manual_sup=<min>-<max>_d<delta>_conf=<min>-<max>_d<delta>_lift=<min>-<max>_d<delta>_w<window>
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
    """
    Processes both regions (northeast, south) independently.

    Expected directory layout produced by feature_importance.py:
        results/<region>/important_features/k_<k>/aggregated_labels_by_sample.csv

    Fallback (backward compatibility):
        results/<region>/important_features/k_<k>/labels_only_unique.csv

    Each experiment run is saved in its own labelled subfolder under
        results/<region>/association_rules/<exp_label>/
    so repeated runs with different parameters never overwrite each other.
    """
    print(f"  > M2 parallel backend — {_CPU_CORES} logical cores / {_PERF_CORES} perf cores (joblib loky)")

    if Path("/content").exists():
        base_dir = Path("/content")
    else:
        base_dir = Path(__file__).resolve().parent.parent

    results_dir = base_dir / "results"

    regions  = ['northeast', 'south']
    k_values = [1, 3, 5, 7]

    # ------------------------------------------------------------------ #
    # Experiment configuration — edit these values between runs.          #
    # Each unique combination is saved in its own labelled subfolder so   #
    # results are never overwritten.                                       #
    # ------------------------------------------------------------------ #
    AUTO_CALIBRATE        = True
    SUP_MIN, SUP_MAX      = 0.02, 0.50   # fallback if AUTO_CALIBRATE=False
    SUP_DELTA             = 0.02
    CONF_MIN, CONF_MAX    = 0.05, 1.00   # CONF_MIN is conf_min_floor when calibrating
    CONF_DELTA            = 0.05
    LIFT_MIN, LIFT_MAX    = 0.0,  5.0    # fallback if AUTO_CALIBRATE=False
    LIFT_DELTA            = 0.05
    LIFT_NEUTRAL_HALF_WIN = 0.25
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
        print(f"ASSOCIATION RULES — {region.upper()}")
        print(f"Experiment: {exp_label}")
        print("="*70 + "\n")

        # FIX: file renamed from aggregated_drivers_by_sample.csv to
        # aggregated_labels_by_sample.csv to match actual output of
        # feature_importance.py (the old name caused silent skip of every k).
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