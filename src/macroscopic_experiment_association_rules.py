"""
macroscopic_experiment_association_rules.py
===========================================
Runs FP-Growth association-rule mining on the feature-driver transactions
produced by feature_importance.py, searching a grid of support, confidence,
and lift thresholds for each value of k.

Pipeline position
-----------------
    create_dataset.py  →  feature_importance.py  →  [macroscopic_experiment_association_rules.py]

Input  : results/{region}/important_features/k_{k}/aggregated_labels_by_sample.csv
         (or labels_only_unique.csv as fallback)
Outputs: results/{region}/association_rules/{experiment_label}/k_{k}/
           rules.csv, rules_detailed.csv, summary.csv
           heatmaps/heatmap_support_confidence.png  (and two others)
         results/{region}/association_rules/{experiment_label}/k_comparison/
           k_comparison_summary.csv, heatmap_k_support.png  (and two others)

Neutral lift window
-------------------
The window [1 - half_window, 1 + half_window] (default [0.75, 1.25]) is
excluded from both the FP-Growth filtering and the heatmap masking via a
single helper (_neutral_window).  Rules in this band are close to statistical
independence and carry little actionable signal.  Rules with lift < lo
(negative correlations — features that tend NOT to co-occur on the boundary)
are intentionally preserved; set lift_min=0.0 to include them in the grid.

Auto-calibration
----------------
calibrate_parameters() scans item frequencies and FP-Growth output to set
sup_min, sup_max, lift_max, and conf_min automatically.  lift_min is always
forced to 0.0 regardless of calibration mode so negative correlations are
never accidentally excluded.

Parallelism
-----------
explore_association_rules() parallelises over support thresholds using joblib's
loky backend (process-based, bypasses GIL).  On Apple Silicon the worker count
is capped at the P-core count to avoid the E-core straggler problem.

Public API
----------
run_k_comparison(k_labels_map, output_dir, auto_calibrate, ...)
    Run explore_association_rules for each k and build a cross-k comparison.

explore_association_rules(df, output_dir, ...)
    Full support × confidence × lift grid search; writes rules and heatmaps.

calibrate_parameters(encoded_df, ...)
    Auto-calibrate grid bounds from item-support frequencies.
"""

import ast
import datetime
import os
import platform
import shutil
import subprocess
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

# Headless backend: required on servers without a display.
matplotlib.use('Agg')

# ---------------------------------------------------------------------------
# Hardware-aware parallelism
# ---------------------------------------------------------------------------

# Total logical CPU count (including hyperthreading / SMT siblings).
_CPU_CORES = os.cpu_count() or 1

# _PERF_CORES is initialised lazily on first use via _get_perf_cores() to avoid
# running a subprocess at module import time (which would print to stdout and
# slow down any code that merely imports this module without using parallelism).
_PERF_CORES: int | None = None


def _detect_perf_cores() -> int:
    """
    Detect the number of performance (P) cores available.

    On Apple Silicon (Darwin), E-cores are significantly slower than P-cores.
    When joblib distributes work evenly across all cores, the slowest E-core
    worker becomes the bottleneck (straggler effect).  Capping at the P-core
    count avoids this.

    Strategy
    --------
    - macOS: query ``sysctl hw.perflevel0.logicalcpu`` — returns the exact
      logical P-core count for any Apple Silicon chip without hard-coding
      per-model values.  On Intel Macs this key is absent; the fallback
      uses the full logical CPU count (all cores are equivalent on Intel).
    - Linux / Windows: all cores are roughly equivalent — use all of them.

    Chip reference (logical P-core counts):
        M1 base:  4P   |  M1 Pro: 6/8P  |  M1 Max: 8P  |  M1 Ultra: 16P
        M2 base:  4P   |  M2 Pro: 6/8P  |  M2 Max: 8P  |  M2 Ultra: 16P
        M3 base:  4P   |  M3 Pro: 6P    |  M3 Max: 12P
        M4 base:  4P   |  M4 Pro: 10P   |  M4 Max: 12P
    """
    if platform.system() == 'Darwin':
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'hw.perflevel0.logicalcpu'],
                capture_output=True, text=True, check=True,
            )
            p_cores = int(result.stdout.strip())
            print(
                f'  > Apple Silicon detected: {p_cores} P-cores '
                f'(of {_CPU_CORES} logical total) — '
                f'joblib capped at {p_cores} workers'
            )
            return p_cores
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            # Intel Mac or unexpected sysctl failure — use all cores.
            pass
    return _CPU_CORES


def _get_perf_cores() -> int:
    """
    Return the cached P-core count, detecting it on the first call.

    Lazy initialisation avoids running a subprocess at import time, which
    would print to stdout and slow down any code that imports this module
    without actually using parallelism (e.g. unit tests, dry-run checks).
    """
    global _PERF_CORES
    if _PERF_CORES is None:
        _PERF_CORES = _detect_perf_cores()
    return _PERF_CORES


# ---------------------------------------------------------------------------
# Neutral-window helper — single source of truth
# ---------------------------------------------------------------------------

def _neutral_window(lift_neutral_half_window: float) -> tuple[float, float]:
    """
    Return (lo, hi) for the neutral lift window.

    Rules with lift in [lo, hi] are excluded in both FP-Growth filtering and
    heatmap masking.  Using this single helper guarantees the two stages agree
    regardless of the lift_delta step size.

    Formula
    -------
    lo = round(1.0 - lift_neutral_half_window, 4)
    hi = round(1.0 + lift_neutral_half_window, 4)

    With the default half_window=0.25 this gives [0.75, 1.25].

    Negative correlations
    ---------------------
    Rules with lift < lo indicate features that tend NOT to co-occur on the
    decision boundary — an analytically meaningful anti-correlation.  Only
    the [lo, hi] band is removed; rules below lo are intentionally preserved.
    Set lift_min=0.0 in the entry point to include them in the grid search.
    """
    lo = round(1.0 - lift_neutral_half_window, 4)
    hi = round(1.0 + lift_neutral_half_window, 4)
    return lo, hi


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def extract_labels(labels_only_path: Path) -> pd.DataFrame:
    """
    Read the labels CSV and one-hot-encode each transaction for FP-Growth.

    Supported file formats:
        aggregated_labels_by_sample.csv  — one row per sample
        labels_only_unique.csv           — one row per (sample, CF) pair

    Both share a 'Labels' column containing Python list literals
    (e.g. "['OCCP', 'SCHL']") parsed with ast.literal_eval.

    Returns a Boolean-encoded DataFrame (rows = transactions, columns = items).
    """
    print('  > Loading and encoding labels...')
    df = pd.read_csv(labels_only_path)

    if 'Labels' not in df.columns:
        raise ValueError(
            f"CSV must have a 'Labels' column. "
            f"Found: {df.columns.tolist()}  |  file: {labels_only_path}"
        )

    print(f'    (file: {Path(labels_only_path).name}, {len(df):,} rows)')

    # Each cell in the 'Labels' column is a stringified Python list
    # (e.g. "['SCHL', 'OCCP']").  ast.literal_eval safely parses it back.
    itemsets = df['Labels'].apply(ast.literal_eval)

    # TransactionEncoder converts a list-of-lists into a Boolean matrix
    # suitable for mlxtend's fpgrowth().  Each column is one item (feature name);
    # each row is one transaction (True = item present in that transaction).
    # dtype=bool is required by mlxtend: float/sparse inputs trigger
    # DeprecationWarning ("non-bool types result in worse computational
    # performance") and cause AttributeError on SparseArray.round() and
    # TypeError on numpy.bool_.__round__ in downstream arithmetic.
    te     = TransactionEncoder()
    te_ary = te.fit(itemsets).transform(itemsets)
    enc_df = pd.DataFrame(te_ary, columns=te.columns_)

    print('  > First few encoded rows:')
    print(enc_df.head())
    print('-' * 50)

    return enc_df


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def cleanup_empty_folders(output_dir: Path) -> tuple[int, int]:
    """
    Remove conf_* folders that contain no rules and sup_* folders that have
    no remaining conf_* children.

    Returns (n_conf_removed, n_sup_removed).
    """
    output_dir   = Path(output_dir)
    removed_conf = 0
    removed_sup  = 0

    for sup_dir in sorted(output_dir.glob('sup_*')):
        if not sup_dir.is_dir():
            continue
        for conf_dir in sorted(sup_dir.glob('conf_*')):
            if not conf_dir.is_dir():
                continue
            rules_csv     = conf_dir / 'rules.csv'
            should_remove = not rules_csv.exists()
            if not should_remove:
                # A rules.csv that contains only the header row is effectively
                # empty.  The header alone is < 120 bytes; a file with at least
                # one data row is always larger.  Using st_size avoids the cost
                # of reading and parsing the CSV inside a hot cleanup loop.
                try:
                    should_remove = rules_csv.stat().st_size < 120
                except OSError:
                    should_remove = True
            if should_remove:
                shutil.rmtree(conf_dir)
                removed_conf += 1
        if not list(sup_dir.glob('conf_*')):
            shutil.rmtree(sup_dir)
            removed_sup += 1

    return removed_conf, removed_sup


# ---------------------------------------------------------------------------
# In-memory grid search
# ---------------------------------------------------------------------------

def grid_search_fpgrowth_delta(
    df,
    sup_min, sup_max, sup_delta,
    conf_min, conf_max, conf_delta,
    lift_min, lift_max, lift_delta,
    lift_neutral_half_window: float = 0.25,
) -> pd.DataFrame:
    """
    Quick in-memory grid search over support × confidence × lift.

    No files are written; use explore_association_rules() for full output.
    Returns a summary DataFrame sorted by (Number_of_Rules DESC, Lift DESC).
    """
    print(f'\n{"=" * 70}')
    print('GRID SEARCH: FP-GROWTH (IN-MEMORY)')
    print(f'{"=" * 70}')

    # Build the three parameter grids.
    # Adding half a step before rounding avoids floating-point undercount
    # (e.g. np.arange(0.02, 0.50, 0.02) can miss 0.50 due to FP rounding).
    support_grid    = np.round(np.arange(sup_min,  sup_max  + sup_delta  / 2, sup_delta),  4)
    confidence_grid = np.round(np.arange(conf_min, conf_max + conf_delta / 2, conf_delta), 4)
    lift_grid       = np.round(np.arange(lift_min, lift_max + lift_delta / 2, lift_delta), 4)

    # Compute the neutral window once; reused for every lift threshold comparison.
    lift_neutral_lo, lift_neutral_hi = _neutral_window(lift_neutral_half_window)

    print(f'  > support grid    : {support_grid}')
    print(f'  > confidence grid : {confidence_grid}')
    print(f'  > lift grid       : {lift_grid}')
    print(f'  > neutral window (excluded): [{lift_neutral_lo}, {lift_neutral_hi}]')
    print('-' * 50)

    results = []

    for min_sup in support_grid:
        # FP-Growth: mine all frequent itemsets at this support threshold.
        # max_len=4 prunes the search space: association rules with antecedent +
        # consequent length > 4 are rarely actionable and exponentially costly.
        frequent_itemsets = fpgrowth(df, min_support=min_sup, use_colnames=True, max_len=4)
        if len(frequent_itemsets) == 0:
            continue

        # Generate rules at the *lowest* confidence threshold in one call,
        # then filter in-memory for higher thresholds.  This avoids calling
        # FP-Growth and association_rules() separately for each confidence level
        # (which would be O(|conf_grid|) times more expensive).
        try:
            all_rules = association_rules(
                frequent_itemsets,
                metric='confidence',
                min_threshold=float(confidence_grid[0]),
            )
        except ValueError:
            # mlxtend raises ValueError if no rules can be generated
            # (e.g. single-item transactions only).
            continue

        if len(all_rules) == 0:
            continue

        # Remove rules in the neutral window; keep negative correlations (lift < lo).
        # This is the in-memory equivalent of the _process_one_support filter.
        all_rules = all_rules[
            (all_rules['lift'] < lift_neutral_lo) |
            (all_rules['lift'] > lift_neutral_hi)
        ]
        if len(all_rules) == 0:
            continue

        for min_conf in confidence_grid:
            # In-memory confidence filter — no additional FP-Growth run needed.
            rules = all_rules[all_rules['confidence'] >= min_conf]
            if len(rules) == 0:
                continue

            for min_lift in lift_grid:
                # Skip lift values inside the neutral window.
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

    print('  > Done.')

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values(
            by=['Number_of_Rules', 'Lift'], ascending=[False, False]
        )
    return result_df


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

def _render_heatmap(
    pivot: 'pd.DataFrame',
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
    neutral_lo: float,
    neutral_hi: float,
    x_is_lift: bool,
    row_fmt: str = '{v:.2f}',
    row_height: float = 0.55,
) -> None:
    """
    Render a single pivot-table heatmap and save it to *output_path*.

    This is the single implementation shared by plot_heatmaps() (per-k grids)
    and run_k_comparison() (cross-k grids).  Previously both functions contained
    ~70 lines of near-identical matplotlib code; any visual change needed to be
    applied twice.  Extracting this helper eliminates the duplication.

    Parameters
    ----------
    pivot      : pre-built pivot table (rows = y-axis, columns = x-axis).
    x_label    : axis label for the x dimension.
    y_label    : axis label for the y dimension.
    title      : figure title (two-line string is fine).
    output_path: full path including filename for the saved PNG.
    neutral_lo / neutral_hi : neutral lift window boundaries; used to mask
                 x-axis columns when x_is_lift=True.
    x_is_lift  : if True, mask the neutral window from the x-axis columns
                 and trim trailing all-zero columns.
    row_fmt    : format string for y-axis tick labels (e.g. '{v:.2f}' or 'k={v}').
    row_height : inches per row for figsize scaling (use 0.55 for param grids,
                 0.60 for k-comparison grids).
    """
    if x_is_lift:
        pivot = pivot.loc[
            :, ~pivot.columns.to_series().between(
                neutral_lo, neutral_hi, inclusive='both'
            )
        ]
        if (pivot != 0).any(axis=0).any():
            last_nz = int(np.where((pivot != 0).any(axis=0).values)[0].max())
            pivot   = pivot.iloc[:, :last_nz + 1]

    n_cols = len(pivot.columns)
    n_rows = len(pivot.index)
    fig, ax = plt.subplots(
        figsize=(max(10, n_cols * 0.75), max(4, n_rows * row_height))
    )

    img = ax.imshow(
        pivot.values, aspect='auto', cmap='YlOrBr', interpolation='nearest'
    )

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [f'{v:.2f}' if isinstance(v, float) else str(v) for v in pivot.columns],
        rotation=40, ha='right', fontsize=8,
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [row_fmt.format(v=v) for v in pivot.index], fontsize=8
    )

    ax.set_xlabel(x_label, fontsize=11, labelpad=8)
    ax.set_ylabel(y_label, fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=11, pad=14)

    max_val = pivot.values.max() if pivot.values.max() > 0 else 1
    for ri in range(n_rows):
        for ci in range(n_cols):
            val = pivot.values[ri, ci]
            if val > 0:
                txt_color = 'white' if (val / max_val) > 0.55 else 'black'
                ax.text(
                    ci, ri, str(val),
                    ha='center', va='center', fontsize=7, color=txt_color,
                )

    cbar = plt.colorbar(img, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('Number of Rules', fontsize=9)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_heatmaps(
    summary_df: pd.DataFrame,
    output_dir: Path,
    lift_display_step: float        = 0.1,
    lift_neutral_half_window: float = 0.25,
    lift_delta: float               = 0.05,
) -> None:
    """
    Generate three heatmaps — support-confidence, support-lift, and
    confidence-lift — each cell showing the maximum rule count over the
    third parameter (darker = more rules).

    Lift-axis masking uses _neutral_window() (the same helper used during
    FP-Growth filtering), guaranteeing exact alignment between the two stages.
    Negative-correlation columns (lift < neutral_lo) are preserved and appear
    on the left side of the lift axis when lift_min=0.0.

    Parameters
    ----------
    lift_delta  : kept for API compatibility; window computation delegates
                  to _neutral_window(lift_neutral_half_window).
    """
    if summary_df.empty:
        print('  > Summary is empty, skipping heatmaps.')
        return

    output_dir  = Path(output_dir)
    heatmap_dir = output_dir / 'heatmaps'
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    print('  > Generating heatmaps...')

    df = summary_df.copy()
    df['Lift_display'] = (
        (df['Lift_threshold'] / lift_display_step).round() * lift_display_step
    ).round(4)

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
        x_label = 'Lift' if x_is_lift else x_col
        _render_heatmap(
            pivot       = pivot,
            x_label     = x_label,
            y_label     = y_col,
            title       = (
                f'Max Number of Rules — {y_col} vs {x_label}\n'
                f'(darker = more rules; max over the third parameter)'
            ),
            output_path = heatmap_dir / f'heatmap_{suffix}.png',
            neutral_lo  = neutral_lo,
            neutral_hi  = neutral_hi,
            x_is_lift   = x_is_lift,
            row_fmt     = '{v:.2f}',
            row_height  = 0.55,
        )
        print(f'    > saved heatmaps/heatmap_{suffix}.png')

    print(f'  > heatmaps saved to {heatmap_dir}/')


# ---------------------------------------------------------------------------
# Core worker — parallelised over support thresholds
# ---------------------------------------------------------------------------

def _process_one_support(
    min_sup: float,
    sup_idx: int,
    n_sup: int,
    df,
    output_dir: Path,
    confidence_grid,
    lift_grid_used,
    lift_window_lo: float,
    lift_window_hi: float,
) -> list:
    """
    Process one support threshold for explore_association_rules.

    Called in parallel; writes to its own sup_{min_sup}/ subdirectory.
    Returns a list of summary-row dicts for the calling process to aggregate.

    Both A→B and B→A rule directions are retained (they have different
    confidence values and carry independent information).
    conviction=inf (confidence=1.0) is replaced with np.nan before saving.

    Column semantics
    ----------------
    support_raw       proportion as decimal string (trailing zeros preserved)
    support_pct       proportion as percentage (float, 2 d.p.)
    confidence_raw    P(consequent | antecedent) as decimal string
    confidence_pct    P(consequent | antecedent) as percentage
    lift              ratio (1.0=independence, >1 positive, <1 negative)
    leverage          support(A∪B) − support(A)·support(B)
    conviction        directional strength (inf → NaN when confidence=1.0)
    """
    output_dir = Path(output_dir)
    sup_label  = f'{min_sup:.2f}'
    sup_dir    = output_dir / f'sup_{sup_label}'
    sup_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n  [{sup_idx}/{n_sup}] support = {min_sup}')
    print('    > running FP-Growth...')

    # Run FP-Growth to mine all frequent itemsets at this support level.
    # Each itemset is a frozenset of feature names that co-occur in at least
    # min_sup fraction of transactions.
    # max_len=4 prunes the search space: rules with antecedent + consequent
    # length > 4 are rarely actionable and exponentially expensive to mine.
    frequent_itemsets = fpgrowth(df, min_support=min_sup, use_colnames=True, max_len=4)
    print(f'    > {len(frequent_itemsets)} frequent itemsets found')

    if len(frequent_itemsets) == 0:
        print('    > no frequent itemsets, skipping.')
        return []

    fi = frequent_itemsets.copy()
    fi['itemset_str']    = fi['itemsets'].apply(lambda x: ', '.join(sorted(x)))
    fi['itemset_length'] = fi['itemsets'].apply(len)
    fi = fi[['itemset_str', 'itemset_length', 'support']]
    fi = fi.sort_values(by=['itemset_length', 'support'], ascending=[True, False])
    fi.to_csv(sup_dir / 'frequent_itemsets.csv', index=False)

    itemsets_by_len = fi['itemset_length'].value_counts().sort_index().to_dict()
    print(f'    > breakdown: '
          f'{", ".join(f"len={k}: {v}" for k, v in itemsets_by_len.items())}')

    with open(sup_dir / 'frequent_itemsets_summary.txt', 'w') as f:
        f.write('Frequent Itemsets Summary\n')
        f.write(f'{"=" * 60}\n\n')
        f.write(f'Parameters:\n  Min Support: {min_sup}\n\n')
        f.write(f'Results:\n  Total: {len(frequent_itemsets)}\n')
        for length, count in itemsets_by_len.items():
            f.write(f'  len={length}: {count}\n')
        f.write(f'\nAll Frequent Itemsets (by length, then support desc):\n{"-" * 60}\n')
        for _, row in fi.iterrows():
            f.write(
                f'  [{row["itemset_str"]}]  '
                f'support={row["support"]:.4f}  '
                f'length={row["itemset_length"]}\n'
            )

    local_summary_rows = []

    # Single association_rules() call at the lowest confidence threshold;
    # subsequent confidence levels are obtained by in-memory filtering.
    try:
        all_rules = association_rules(
            frequent_itemsets,
            metric='confidence',
            min_threshold=float(confidence_grid[0]),
        )
    except ValueError:
        return local_summary_rows

    if len(all_rules) == 0:
        return local_summary_rows

    # Exclude neutral window; preserve negative correlations (lift < lo).
    all_rules = all_rules[
        (all_rules['lift'] < lift_window_lo) |
        (all_rules['lift'] > lift_window_hi)
    ]
    all_rules = all_rules.sort_values('lift', ascending=False).reset_index(drop=True)

    if len(all_rules) == 0:
        return local_summary_rows

    for min_conf in confidence_grid:
        conf_label = f'{min_conf:.2f}'
        conf_dir   = sup_dir / f'conf_{conf_label}'

        rules = all_rules[all_rules['confidence'] >= min_conf].reset_index(drop=True)
        if len(rules) == 0:
            continue

        # conviction = P(¬consequent) / P(¬consequent | antecedent).
        # When confidence = 1.0, P(¬consequent | antecedent) = 0, giving
        # conviction = inf.  We replace inf with NaN before writing to CSV
        # to avoid breaking CSV parsers that don't handle inf strings.
        # Computed once and reused for both fmt and det DataFrames.
        conviction_vals = rules['conviction'].replace([np.inf, -np.inf], np.nan)

        # Compact output.
        fmt = pd.DataFrame()
        fmt['antecedents']    = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
        fmt['consequents']    = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
        fmt['support_raw']    = [f'{v:.4f}' for v in rules['support']]
        fmt['support_pct']    = (rules['support']    * 100).round(2)
        fmt['confidence_raw'] = [f'{v:.4f}' for v in rules['confidence']]
        fmt['confidence_pct'] = (rules['confidence'] * 100).round(2)
        fmt['lift']           = rules['lift'].round(4)
        fmt['leverage']       = rules['leverage'].round(6)
        fmt['conviction']     = conviction_vals.round(4)

        # Detailed output.
        det = pd.DataFrame()
        det['antecedents']            = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
        det['consequents']            = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
        det['antecedent_length']      = rules['antecedents'].apply(len)
        det['consequent_length']      = rules['consequents'].apply(len)
        det['rule_length']            = det['antecedent_length'] + det['consequent_length']
        det['antecedent_support_raw'] = [f'{v:.4f}' for v in rules['antecedent support']]
        det['antecedent_support_pct'] = (rules['antecedent support'] * 100).round(2)
        det['consequent_support_raw'] = [f'{v:.4f}' for v in rules['consequent support']]
        det['consequent_support_pct'] = (rules['consequent support'] * 100).round(2)
        det['support_raw']            = [f'{v:.4f}' for v in rules['support']]
        det['support_pct']            = (rules['support']    * 100).round(2)
        det['confidence_raw']         = [f'{v:.4f}' for v in rules['confidence']]
        det['confidence_pct']         = (rules['confidence'] * 100).round(2)
        det['lift']                   = rules['lift'].round(4)
        det['leverage']               = rules['leverage'].round(6)
        det['conviction']             = conviction_vals.round(4)

        conf_dir.mkdir(parents=True, exist_ok=True)
        fmt.to_csv(conf_dir / 'rules.csv',          index=False)
        det.to_csv(conf_dir / 'rules_detailed.csv', index=False)

        with open(conf_dir / 'summary.txt', 'w') as f:
            f.write('Association Rules Summary\n')
            f.write(f'{"=" * 60}\n\n')
            f.write('Parameters:\n')
            f.write(f'  Min Support:    {min_sup}\n')
            f.write(f'  Min Confidence: {min_conf}\n')
            f.write(f'  Neutral Lift Window (excluded): '
                    f'[{lift_window_lo}, {lift_window_hi}]\n\n')
            f.write('Results:\n')
            f.write(f'  Frequent Itemsets: {len(frequent_itemsets)}\n')
            f.write(f'  Association Rules: {len(rules)}\n\n')
            f.write('Statistics:\n')
            f.write(f'  Avg Support:     {rules["support"].mean() * 100:.2f}%\n')
            f.write(f'  Avg Confidence:  {rules["confidence"].mean() * 100:.2f}%\n')
            f.write(f'  Avg Lift:        {rules["lift"].mean():.4f}\n')
            f.write(f'  Lift Range:      {rules["lift"].min():.4f} — '
                    f'{rules["lift"].max():.4f}\n')
            f.write(f'  Avg Leverage:    {rules["leverage"].mean():.6f}\n')
            f.write(f'  Avg Rule Length: {det["rule_length"].mean():.2f}\n\n')
            n_inf = conviction_vals.isna().sum()
            if n_inf > 0:
                f.write(f'  Note: {n_inf} rule(s) have confidence=1.0 '
                        f'(conviction=inf → saved as NaN)\n\n')
            f.write(f'Top 10 Rules (by Lift):\n{"-" * 60}\n')
            for idx, row in fmt.head(10).iterrows():
                f.write(
                    f'{idx + 1}. {row["antecedents"]} => {row["consequents"]}\n'
                    f'   support={row["support_pct"]:.2f}% | '
                    f'confidence={row["confidence_pct"]:.2f}% | '
                    f'lift={row["lift"]:.4f} | '
                    f'leverage={row["leverage"]:.6f}\n\n'
                )

        print(f'    > [conf={min_conf}] {len(rules)} rules saved to '
              f'{conf_dir.relative_to(output_dir)}/')

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

def explore_association_rules(
    df,
    output_dir: Path,
    sup_min, sup_max, sup_delta,
    conf_min, conf_max, conf_delta,
    lift_min, lift_max, lift_delta,
    lift_neutral_half_window: float = 0.25,
    inner_n_jobs: int | None        = None,
) -> pd.DataFrame:
    """
    Full grid search over support × confidence × lift using FP-Growth.

    Workers are dispatched in parallel over support thresholds (loky backend).
    The neutral lift window is excluded via _neutral_window().
    Negative correlations and both A→B / B→A rule directions are retained.

    Parameters
    ----------
    inner_n_jobs : int or None
        Number of loky workers for the support-level parallel loop.
        When None (default) the function derives it as
        max(1, perf_cores // outer_k_jobs) to respect the two-level
        parallelism budget.  Pass an explicit value to override.

    Returns a summary DataFrame sorted by (Number_of_Rules DESC, Max_Lift DESC).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    support_grid    = np.round(np.arange(sup_min,  sup_max  + sup_delta  / 2, sup_delta),  4)
    confidence_grid = np.round(np.arange(conf_min, conf_max + conf_delta / 2, conf_delta), 4)
    lift_grid       = np.round(np.arange(lift_min, lift_max + lift_delta / 2, lift_delta), 4)

    lift_window_lo, lift_window_hi = _neutral_window(lift_neutral_half_window)
    lift_grid_used  = [v for v in lift_grid
                       if not (lift_window_lo <= v <= lift_window_hi)]
    total_combos    = len(support_grid) * len(confidence_grid) * len(lift_grid_used)

    print(f'\n{"=" * 70}')
    print('FULL EXPLORATION: FP-GROWTH ASSOCIATION RULES')
    print(f'{"=" * 70}')
    print(f'  > support    : {len(support_grid)} values '
          f'[{support_grid[0]} ... {support_grid[-1]}, step={sup_delta}]')
    print(f'  > confidence : {len(confidence_grid)} values '
          f'[{confidence_grid[0]} ... {confidence_grid[-1]}, step={conf_delta}]')
    print(f'  > lift       : {len(lift_grid_used)} values used '
          f'(of {len(lift_grid)} total, '
          f'{len(lift_grid) - len(lift_grid_used)} skipped — '
          f'neutral window [{lift_window_lo}, {lift_window_hi}], step={lift_delta})')
    print(f'  > total combinations: {total_combos:,}')
    print('-' * 50)

    # Each support threshold is independent: no shared mutable state.
    # loky backend: true multiprocessing (bypasses GIL); each worker gets
    # its own copy of df and the parameter arrays.
    # Cap n_jobs at the number of support values to avoid spawning idle workers.
    perf_cores = _get_perf_cores()
    # Respect the two-level parallelism budget:
    #   outer level (k values)   : outer_jobs workers, each calling this function
    #   inner level (sup values) : inner_n_jobs workers per outer worker
    #
    # When inner_n_jobs is provided by run_k_comparison, it has already been
    # set to max(1, perf_cores // outer_jobs) so outer × inner ≤ perf_cores.
    # When called standalone (inner_n_jobs=None), we use all P-cores safely.
    if inner_n_jobs is None:
        inner_n_jobs = min(perf_cores, len(support_grid))
    n_jobs = min(inner_n_jobs, len(support_grid))
    print(f'  > Launching parallel FP-Growth over {len(support_grid)} support '
          f'values (n_jobs={n_jobs} of {_CPU_CORES} logical / '
          f'{perf_cores} perf cores)')

    parallel_results = Parallel(
        n_jobs     = n_jobs,
        backend    = 'loky',
        verbose    = 0,
        # pre_dispatch controls how many tasks are queued in the worker pool
        # at once.  Setting it equal to n_jobs (instead of the default 2×n_jobs)
        # limits peak task-queue depth: at most n_jobs tasks are submitted before
        # a completed task is collected.  On Apple Silicon's unified memory bus
        # this reduces the window during which multiple large DataFrames coexist
        # in the task queue simultaneously.  Note: loky serialises the DataFrame
        # once per worker at spawn time, not per task; pre_dispatch therefore
        # does not change the number of in-memory DataFrame copies but does
        # reduce peak memory from queued task arguments.
        pre_dispatch = n_jobs,
    )(
        delayed(_process_one_support)(
            min_sup        = min_sup,
            sup_idx        = sup_idx,
            n_sup          = len(support_grid),
            df             = df,
            output_dir     = output_dir,
            confidence_grid= confidence_grid,
            lift_grid_used = lift_grid_used,
            lift_window_lo = lift_window_lo,
            lift_window_hi = lift_window_hi,
        )
        for sup_idx, min_sup in enumerate(support_grid, start=1)
    )

    summary_rows = [row for worker_rows in parallel_results for row in worker_rows]

    print(f'\n{"=" * 70}')
    print('  > Exploration complete, building summary...')

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=['Number_of_Rules', 'Max_Lift'], ascending=[False, False]
        ).reset_index(drop=True)

    summary_df.to_csv(output_dir / 'summary.csv', index=False)
    combos_with_rules = (
        int((summary_df['Number_of_Rules'] > 0).sum())
        if not summary_df.empty else 0
    )

    with open(output_dir / 'exploration_summary.txt', 'w') as f:
        f.write('FULL EXPLORATION SUMMARY\n')
        f.write(f'{"=" * 70}\n\n')
        f.write(f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write('Parameter Grids:\n')
        f.write(f'  Support    : {len(support_grid)} values '
                f'[{support_grid[0]} ... {support_grid[-1]}, step={sup_delta}]\n')
        f.write(f'  Confidence : {len(confidence_grid)} values '
                f'[{confidence_grid[0]} ... {confidence_grid[-1]}, step={conf_delta}]\n')
        f.write(f'  Lift       : {len(lift_grid_used)} values used '
                f'(of {len(lift_grid)} total, '
                f'neutral window [{lift_window_lo}, {lift_window_hi}] excluded), '
                f'step={lift_delta}\n\n')
        f.write('Results:\n')
        f.write(f'  Total combinations : {total_combos:,}\n')
        f.write(f'  With >= 1 rule     : {combos_with_rules:,}\n\n')
        if not summary_df.empty and combos_with_rules > 0:
            best = summary_df.iloc[0]
            f.write('Best combination (most rules, then highest lift):\n')
            f.write(f'{"-" * 60}\n')
            f.write(f'  Support:         {best["Support"]}\n')
            f.write(f'  Confidence:      {best["Confidence"]}\n')
            f.write(f'  Lift threshold:  {best["Lift_threshold"]}\n')
            f.write(f'  Number of Rules: {int(best["Number_of_Rules"])}\n')
            f.write(f'  Max Lift:        {best["Max_Lift"]}\n')
            f.write(f'  Avg Lift:        {best["Avg_Lift"]}\n')

    print('  > Cleaning up empty folders...')
    removed_conf, removed_sup = cleanup_empty_folders(output_dir)
    print(f'  > Removed {removed_conf} conf folder(s) and {removed_sup} sup folder(s)')

    with open(output_dir / 'exploration_summary.txt', 'a') as f:
        f.write('\nParallelism:\n')
        f.write(f'  n_jobs used      : {n_jobs} '
                f'(of {_CPU_CORES} logical / {perf_cores} perf cores)\n\n')
        f.write('Folder cleanup:\n')
        f.write(f'  conf dirs removed: {removed_conf}\n')
        f.write(f'  sup dirs removed : {removed_sup}\n')

    plot_heatmaps(
        summary_df, output_dir,
        lift_neutral_half_window=lift_neutral_half_window,
        lift_delta=lift_delta,
    )

    print(f'  > summary saved to {output_dir / "summary.csv"}')
    print(f'  > total combinations: {total_combos:,}  |  '
          f'with rules: {combos_with_rules:,}')

    return summary_df


# ---------------------------------------------------------------------------
# Auto-calibration
# ---------------------------------------------------------------------------

def calibrate_parameters(
    encoded_df: pd.DataFrame,
    sup_delta: float       = 0.02,
    lift_delta: float      = 0.05,
    conf_delta: float      = 0.05,
    conf_min_floor: float  = 0.05,
    conf_max: float        = 1.00,
) -> dict | None:
    """
    Auto-calibrate sup_min, sup_max, lift_max, and conf_min from item
    frequencies.  lift_min is always set to 0.0 to include negative
    correlations in the grid from the start.

    conf_min is set to the MINIMUM confidence observed in the probe rules
    (floored to the nearest conf_delta step, clamped to conf_min_floor).
    This ensures the full confidence grid is explored rather than collapsing
    to a single point when rules happen to have uniformly high confidence.

    Returns None if no 2-itemsets can be formed (transactions too sparse).
    """
    print('  > Calibrating parameters from item frequencies...')

    item_supports = encoded_df.mean().astype(float).sort_values()
    print('  > Item supports:')
    for item, sup in item_supports.items():
        print(f'    {item}: {sup:.4f}')

    if len(item_supports) < 2:
        print('  > WARNING: fewer than 2 items — cannot form pairwise rules.')
        return None

    # The theoretical minimum support for a 2-itemset containing items A and B
    # is support(A) × support(B) (if they were independent).  Using the two
    # rarest items gives a lower bound on sup_min that is guaranteed to find
    # at least some 2-itemsets without being so low that the grid is dominated
    # by trivial single-item rules.
    rarest = item_supports.iloc[0]    # rarest item
    second = item_supports.iloc[1]    # second rarest
    freq_2 = item_supports.iloc[-2]   # second most frequent (used for sup_max scan)

    raw_sup_min = rarest * second
    # Floor to the nearest sup_delta step; ensure at least sup_delta (never 0.0).
    sup_min = max(
        round(np.floor(raw_sup_min / sup_delta) * sup_delta, 4), sup_delta
    )

    # ── Stage 1: binary search for the natural 2-itemset ceiling ────────
    # FP-Growth is monotone: raising min_support can only remove itemsets,
    # never add them.  The "natural ceiling" is the highest threshold that
    # still yields at least one 2-itemset.  A linear scan is O(range/step)
    # FP-Growth calls; binary search finds the same boundary in O(log(range/step)).
    #
    # Scan upper bound: support of the second-most-frequent item.
    # A 2-itemset {A, B} cannot have support > min(support(A), support(B)),
    # so scanning beyond the second-most-frequent item is guaranteed to find
    # nothing and wastes FP-Growth calls.
    scan_grid               = np.round(
        np.arange(sup_min, freq_2 + sup_delta / 2, sup_delta), 4
    )
    natural_sup_max         = sup_min   # updated below as binary search progresses
    prev_had_2itemsets      = False
    fi_first_with_2itemsets = None

    # --- Phase A: verify at least one 2-itemset exists at sup_min ----------
    # max_len=2: calibration only needs to detect 2-itemset presence/absence —
    # mining longer itemsets here wastes time without contributing to the result.
    fi_lo = fpgrowth(encoded_df, min_support=float(scan_grid[0]), use_colnames=True, max_len=2)
    if not fi_lo.empty and (fi_lo['itemsets'].apply(len) >= 2).any():
        fi_first_with_2itemsets = fi_lo
        prev_had_2itemsets = True

        if len(scan_grid) == 1:
            natural_sup_max = scan_grid[0]
        else:
            # --- Phase B: binary search for the last grid index with 2-itemsets
            lo_idx, hi_idx = 0, len(scan_grid) - 1
            while lo_idx < hi_idx:
                mid_idx = (lo_idx + hi_idx + 1) // 2
                fi_mid  = fpgrowth(
                    encoded_df,
                    min_support=float(scan_grid[mid_idx]),
                    use_colnames=True,
                    max_len=2,
                )
                has_2 = (
                    not fi_mid.empty
                    and (fi_mid['itemsets'].apply(len) >= 2).any()
                )
                if has_2:
                    lo_idx = mid_idx          # ceiling is at or above mid
                else:
                    hi_idx = mid_idx - 1      # ceiling is below mid

            natural_sup_max = scan_grid[lo_idx]
    else:
        # No 2-itemsets even at the lowest threshold — transactions too sparse.
        prev_had_2itemsets = False

    # ── Stage 2: extend grid N steps beyond the natural ceiling ─────────
    # The natural ceiling is where rules stop existing in these data.
    # Adding _DECAY_STEPS above it lets the heatmap show the rule-count
    # curve decaying to zero — which is the diagnostic value of the grid.
    # Those empty sup_* folders are removed automatically by cleanup_empty_folders.
    # No fixed minimum width: grid breadth is fully determined by the data.
    _DECAY_STEPS    = 4
    sup_max = min(
        round(natural_sup_max + _DECAY_STEPS * sup_delta, 4),
        0.50,   # hard cap: above 50% support is trivially dense
    )

    print(
        f'  > support grid: [{sup_min} … {sup_max}]  '
        f'(natural 2-itemset ceiling = {natural_sup_max}, '
        f'+{_DECAY_STEPS} decay steps)'
    )

    # The theoretical maximum lift for a rule involving the rarest item is
    # approximately 1 / support(rarest).  We round up to the nearest 0.5
    # and cap at 10.0 to avoid unbounded grids when very rare items exist.
    raw_lift_max = 1.0 / rarest
    lift_max     = min(round(np.ceil(raw_lift_max * 2) / 2, 1), 10.0)

    if not prev_had_2itemsets:
        print('    - WARNING: no 2-itemsets found at any support threshold.')
        print('      Transactions are too sparse to generate association rules.')
        print('-' * 50)
        return None

    calibrated_conf_min = conf_min_floor
    min_conf_observed   = None
    max_conf_observed   = None
    try:
        if fi_first_with_2itemsets is not None and not fi_first_with_2itemsets.empty:
            rules_probe = association_rules(
                fi_first_with_2itemsets, metric='confidence', min_threshold=0.01
            )
            if not rules_probe.empty:
                # Use the MINIMUM observed confidence to set conf_min so that
                # the full confidence grid is explored from the lowest meaningful
                # value.  Using max_conf here (the previous behaviour) collapsed
                # the grid to a single point whenever rules had high confidence.
                min_conf_observed   = rules_probe['confidence'].min()
                max_conf_observed   = rules_probe['confidence'].max()
                calibrated          = round(
                    np.floor(min_conf_observed / conf_delta) * conf_delta, 4
                )
                calibrated_conf_min = max(calibrated, conf_min_floor)
                print(
                    f'  > conf_min calibrated to {calibrated_conf_min} '
                    f'(min observed confidence={min_conf_observed:.4f}, '
                    f'max observed confidence={max_conf_observed:.4f}, '
                    f'floor={conf_min_floor})'
                )
            else:
                print(
                    f'  > Note: no rules at conf=0.01 for sup_min={sup_min} '
                    f'— conf_min stays at floor={conf_min_floor}'
                )
    except Exception as exc:
        print(
            f'  > WARNING: conf_min calibration failed ({exc!r}) — '
            f'using floor={conf_min_floor}'
        )

    params = {
        'sup_min':           sup_min,
        'sup_max':           sup_max,
        'natural_sup_max':   natural_sup_max,
        'sup_delta':         sup_delta,
        'conf_min':          calibrated_conf_min,
        'conf_max':          conf_max,
        'conf_delta':        conf_delta,
        'lift_min':          0.0,   # always 0.0 — negative correlations included
        'lift_max':          lift_max,
        'lift_delta':        lift_delta,
        # Diagnostic values for calibration_log.txt.
        'raw_sup_min':       round(raw_sup_min, 6),
        'raw_lift_max':      round(raw_lift_max, 4),
        'min_conf_observed': round(min_conf_observed, 4) if min_conf_observed is not None else None,
        'max_conf_observed': round(max_conf_observed, 4) if max_conf_observed is not None else None,
        'item_supports':     item_supports.astype(float).round(4).to_dict(),
    }

    print(
        f'  > calibrated: sup_min={sup_min} (raw={raw_sup_min:.4f}), '
        f'sup_max={sup_max}, conf_min={calibrated_conf_min} (from min observed in data), '
        f'lift_max={lift_max} (raw ceiling={raw_lift_max:.2f})'
    )
    print('-' * 50)

    return params


# ---------------------------------------------------------------------------
# Calibration log writer
# ---------------------------------------------------------------------------

def _write_calibration_log(
    k_dir: Path,
    k: int,
    n_transactions: int,
    item_supports: pd.Series,
    params: dict | None,
    auto_calibrate: bool,
    manual_params: dict | None = None,
) -> None:
    """
    Write per-k calibration artefacts:

        item_supports.csv    — one row per feature, sorted ascending
        calibration_log.txt  — human-readable parameter summary

    Called for every k, including skipped ones (params=None), so the log
    always documents why a k was included or excluded.
    """
    k_dir = Path(k_dir)
    k_dir.mkdir(parents=True, exist_ok=True)

    # Write item support frequencies to CSV for offline inspection.
    # These values are the per-feature "prevalence" in the transaction set —
    # how often each feature name appears as a driver across all samples.
    sup_df = pd.DataFrame({
        'item':        list(item_supports.index),
        'support_raw': [f'{v:.4f}' for v in item_supports.values],
        'support_pct': [f'{v * 100:.2f}' for v in item_supports.values],
    })
    sup_df.to_csv(k_dir / 'item_supports.csv', index=False)

    with open(k_dir / 'calibration_log.txt', 'w') as f:
        f.write(f'CALIBRATION LOG — k={k}\n')
        f.write(f'{"=" * 60}\n\n')
        f.write(f'Transactions : {n_transactions:,}\n')
        f.write(f'Items        : {len(item_supports)}\n\n')
        f.write(f'Item Supports (sorted ascending):\n{"-" * 40}\n')
        for item, sup in item_supports.items():
            f.write(f'  {item:<12} {sup:.4f}\n')
        f.write('\n')

        if not auto_calibrate:
            f.write('Mode: MANUAL (auto_calibrate=False)\n')
            if manual_params:
                f.write(f'  sup_min  : {manual_params.get("sup_min")}\n')
                f.write(f'  sup_max  : {manual_params.get("sup_max")}\n')
                f.write(f'  conf_min : {manual_params.get("conf_min")}\n')
                f.write(f'  lift_max : {manual_params.get("lift_max")}\n')
            return

        f.write('Mode: AUTO-CALIBRATED\n\n')

        if params is None:
            f.write('Result: SKIPPED\n')
            f.write('Reason: no 2-itemsets found at any support threshold.\n')
            f.write('  Transactions are too sparse to generate association rules.\n')
            f.write('  This typically occurs at low k values where most samples\n')
            f.write('  have only 1 CF neighbour, producing single-item transactions.\n')
            return

        f.write('Calibrated Parameters:\n')
        f.write(f'  sup_min          : {params["sup_min"]}  '
                f'(raw product = {params["raw_sup_min"]:.6f})\n')
        f.write(f'  sup_max          : {params["sup_max"]}  '
                f'(natural 2-itemset ceiling = {params["natural_sup_max"]}, '
                f'+{4} decay steps)\n')
        f.write(f'  sup_delta        : {params["sup_delta"]}\n')
        f.write(f'  conf_min         : {params["conf_min"]}  '
                f'(min observed = {params["min_conf_observed"]}, '
                f'max observed = {params["max_conf_observed"]})\n')
        f.write(f'  conf_max         : {params["conf_max"]}\n')
        f.write(f'  conf_delta       : {params["conf_delta"]}\n')
        f.write(f'  lift_min         : {params["lift_min"]}  '
                f'(0.0 — negative correlations included)\n')
        f.write(f'  lift_max         : {params["lift_max"]}  '
                f'(raw ceiling = {params["raw_lift_max"]:.4f}, capped at 10.0)\n')
        f.write(f'  lift_delta       : {params["lift_delta"]}\n')


# ---------------------------------------------------------------------------
# K-comparison experiment
# ---------------------------------------------------------------------------

def _process_one_k(
    k: int,
    labels_csv: Path,
    output_dir: Path,
    auto_calibrate: bool,
    sup_min: float, sup_max: float, sup_delta: float,
    conf_min: float, conf_max: float, conf_delta: float,
    lift_min: float, lift_max: float, lift_delta: float,
    lift_neutral_half_window: float,
    inner_n_jobs: int = 1,
) -> tuple[int, pd.DataFrame | None, dict]:
    """
    Process a single k value: load data, (optionally) calibrate, run grid search.

    Called in parallel from run_k_comparison via joblib.
    Returns (k, summary_df_or_None, comparison_row_dict).
    Each k writes to its own k_{k}/ subdirectory so there is no shared
    mutable filesystem state between workers.

    Parameters
    ----------
    inner_n_jobs : int
        Number of loky workers for the inner support-level parallel loop
        inside explore_association_rules().  Computed by run_k_comparison()
        as max(1, perf_cores // outer_jobs) so that total worker count
        stays within the P-core budget.
    """
    labels_csv = Path(labels_csv)
    k_dir      = output_dir / f'k_{k}'
    k_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"=" * 70}')
    print(f'  k = {k}  |  {labels_csv.name}')
    print(f'{"=" * 70}')

    if not labels_csv.exists():
        print(f'  > File not found: {labels_csv}')
        row = {
            'k': k, 'skipped_reason': 'file_not_found',
            'n_transactions': None, 'n_items': None,
            'rarest_item_support': None, 'sup_min_used': None,
            'sup_max_used': None, 'lift_max_used': None,
            'conf_min_used': None, 'summary_rows': 0,
            'combos_with_rules': 0, 'max_rules_any_combo': 0,
            'avg_lift_best_combo': None, 'max_lift_observed': None,
        }
        return k, None, row

    df_encoded    = extract_labels(labels_csv)
    item_supports = df_encoded.mean().astype(float).sort_values()

    if auto_calibrate:
        params = calibrate_parameters(
            encoded_df     = df_encoded,
            sup_delta      = sup_delta,
            lift_delta     = lift_delta,
            conf_delta     = conf_delta,
            conf_min_floor = conf_min,
            conf_max       = conf_max,
        )
        _write_calibration_log(
            k_dir=k_dir, k=k,
            n_transactions=len(df_encoded),
            item_supports=item_supports,
            params=params,
            auto_calibrate=True,
        )

        if params is None:
            print(f'  > Skipping k={k} — insufficient co-occurrences for rules.')
            row = {
                'k': k, 'skipped_reason': 'too_sparse',
                'n_transactions': len(df_encoded),
                'n_items': df_encoded.shape[1],
                'rarest_item_support': round(item_supports.iloc[0], 4),
                'sup_min_used': None, 'sup_max_used': None,
                'lift_max_used': None, 'conf_min_used': None,
                'summary_rows': 0, 'combos_with_rules': 0,
                'max_rules_any_combo': 0, 'avg_lift_best_combo': None,
                'max_lift_observed': None,
            }
            return k, None, row

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
        df          = df_encoded,
        output_dir  = k_dir,
        sup_min     = k_sup_min,  sup_max  = k_sup_max,  sup_delta  = sup_delta,
        conf_min    = k_conf_min, conf_max = conf_max,   conf_delta = conf_delta,
        lift_min    = k_lift_min, lift_max = k_lift_max, lift_delta = lift_delta,
        lift_neutral_half_window = lift_neutral_half_window,
        inner_n_jobs             = inner_n_jobs,
    )

    has_rules_col = (
        not summary_df.empty and 'Number_of_Rules' in summary_df.columns
    )
    with_rules = (
        summary_df[summary_df['Number_of_Rules'] > 0]
        if has_rules_col
        else pd.DataFrame()
    )
    row = {
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
    }
    return k, summary_df, row


def run_k_comparison(
    k_labels_map: dict,
    output_dir: Path,
    auto_calibrate: bool     = True,
    sup_min: float           = 0.02,
    sup_max: float           = 0.50,
    sup_delta: float         = 0.02,
    conf_min: float          = 0.05,
    conf_max: float          = 1.00,
    conf_delta: float        = 0.05,
    lift_min: float          = 0.0,
    lift_max: float          = 5.0,
    lift_delta: float        = 0.05,
    lift_neutral_half_window: float = 0.25,
) -> dict:
    """
    Run explore_association_rules for each k and produce a cross-k comparison.

    lift_min=0.0 (default) ensures negative correlations are included in the
    grid for every k in both auto and manual mode.

    Parallelism
    -----------
    Each k value is processed in a separate loky worker so that multiple k
    values run concurrently.  Within each worker, explore_association_rules
    spawns its own inner Parallel over support thresholds.  The two levels
    of parallelism are separated by the loky process boundary: the outer
    level handles the k-loop; the inner level handles the support grid.
    n_jobs for the outer level is capped at min(n_k_values, perf_cores) to
    avoid spawning more workers than there is work.

    Parameters
    ----------
    k_labels_map    : dict mapping k (int) → Path of the labels CSV.
    output_dir      : root output directory; per-k and comparison sub-folders
                      are created automatically.
    auto_calibrate  : if True, calibrate sup_min/sup_max/lift_max/conf_min
                      from item frequencies for each k.
    *               : remaining grid parameters used as fallback (manual mode)
                      or as conf_min_floor / conf_max / deltas (auto mode).

    Returns
    -------
    dict mapping k → summary DataFrame for that k.
    """
    output_dir = Path(output_dir)
    comp_dir   = output_dir / 'k_comparison'
    comp_dir.mkdir(parents=True, exist_ok=True)

    k_sorted   = sorted(k_labels_map.keys())
    perf_cores = _get_perf_cores()
    # Outer parallelism: one worker per k value, capped at P-core count.
    outer_jobs = min(len(k_sorted), perf_cores)

    # RAM-aware cap: each loky worker receives a serialised copy of the encoded
    # DataFrame (one copy per outer worker, held in memory until the worker
    # finishes).  On memory-constrained machines spawning too many outer workers
    # can trigger swapping, which is slower than running serially.
    # We estimate peak RSS as outer_jobs × df_size × 2 (serialisation overhead)
    # and reduce outer_jobs until the estimate fits within available RAM.
    # psutil is a soft dependency; if absent we skip the RAM check gracefully.
    try:
        import psutil as _psutil
        # Load the first available labels CSV to get a proxy for df size.
        # All k DataFrames share the same columns; row counts differ slightly.
        _probe_path = Path(next(iter(k_labels_map.values())))
        if _probe_path.exists():
            _probe_df     = pd.read_csv(_probe_path, nrows=0)  # header only
            _n_cols       = len(_probe_df.columns)
            _n_rows_est   = sum(1 for _ in open(_probe_path)) - 1  # approx
            # Estimate: bool DataFrame ≈ 1 byte/cell; add 4× headroom for
            # pandas overhead and association_rules intermediate structures.
            _df_bytes_est = _n_rows_est * _n_cols * 4
            _avail_bytes  = _psutil.virtual_memory().available
            _max_by_ram   = max(1, int(_avail_bytes // (_df_bytes_est * 2)))
            if _max_by_ram < outer_jobs:
                print(
                    f'  > RAM cap: reducing outer_jobs {outer_jobs} → {_max_by_ram} '
                    f'(est. {_df_bytes_est / 1e6:.1f} MB/worker, '
                    f'{_avail_bytes / 1e9:.1f} GB available)'
                )
                outer_jobs = _max_by_ram
    except Exception:
        pass  # psutil absent or probe failed — proceed with CPU-based cap

    # Two-level parallelism budget for Apple Silicon (and any NUMA system):
    #
    #   outer_jobs  : k-level workers running concurrently.
    #   inner_n_jobs: support-level workers each k-worker spawns.
    #
    # Without budget splitting, outer × inner can reach perf_cores² processes
    # (e.g. 4 × 16 = 64 on M1 Ultra), all competing for 16 P-cores.
    # loky serialises the encoded DataFrame into every worker; over-subscription
    # causes quadratic memory pressure and CPU starvation rather than speedup.
    #
    # Setting inner_n_jobs = max(1, perf_cores // outer_jobs) ensures:
    #   outer × inner ≤ perf_cores   (total workers ≤ available P-cores)
    #
    # Example: M1 Ultra, 16 P-cores, 4 k-values
    #   outer_jobs   = min(4, 16) = 4
    #   inner_n_jobs = max(1, 16 // 4) = 4
    #   total workers = 4 × 4 = 16  (≤ 16 P-cores — no over-subscription)
    inner_n_jobs = max(1, perf_cores // outer_jobs)

    print(f'\n{"=" * 70}')
    print(f'K-VARIATION EXPERIMENT — {len(k_labels_map)} k values')
    print(f'{"=" * 70}')
    print(f'  > k values        : {k_sorted}')
    print(f'  > auto_calibrate  : {auto_calibrate}')
    print(f'  > outer n_jobs    : {outer_jobs}  (k-level workers)')
    print(f'  > inner n_jobs    : {inner_n_jobs}  (support-level workers per k, '
          f'total ≤ {outer_jobs * inner_n_jobs} / {perf_cores} P-cores)')
    print('-' * 50)

    raw_results = Parallel(n_jobs=outer_jobs, backend='loky', verbose=0)(
        delayed(_process_one_k)(
            k                        = k,
            labels_csv               = Path(k_labels_map[k]),
            output_dir               = output_dir,
            auto_calibrate           = auto_calibrate,
            sup_min=sup_min,   sup_max=sup_max,   sup_delta=sup_delta,
            conf_min=conf_min, conf_max=conf_max, conf_delta=conf_delta,
            lift_min=lift_min, lift_max=lift_max, lift_delta=lift_delta,
            lift_neutral_half_window = lift_neutral_half_window,
            inner_n_jobs             = inner_n_jobs,
        )
        for k in k_sorted
    )

    k_summaries     = {}
    comparison_rows = []
    for k, summary_df, row in raw_results:
        comparison_rows.append(row)
        if summary_df is not None:
            k_summaries[k] = summary_df

    print(f'\n{"=" * 70}')
    print('  > Building cross-k comparison...')

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(comp_dir / 'k_comparison_summary.csv', index=False)

    with open(comp_dir / 'k_comparison_summary.txt', 'w') as f:
        f.write('K-VARIATION EXPERIMENT SUMMARY\n')
        f.write(f'{"=" * 70}\n\n')
        f.write(f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write(f'k values tested : {sorted(k_labels_map.keys())}\n')
        f.write(f'auto_calibrate  : {auto_calibrate}\n\n')

        skipped = comp_df[comp_df['skipped_reason'] != ''] if not comp_df.empty else pd.DataFrame()
        ran     = comp_df[comp_df['skipped_reason'] == ''] if not comp_df.empty else pd.DataFrame()

        if not skipped.empty:
            f.write(f'Skipped k values ({len(skipped)}):\n{"-" * 40}\n')
            for _, row in skipped.iterrows():
                reason = row['skipped_reason']
                if reason == 'too_sparse':
                    detail = (
                        f'no 2-itemsets found '
                        f'(n_transactions={int(row["n_transactions"]):,}, '
                        f'rarest_support={row["rarest_item_support"]})'
                    )
                elif reason == 'file_not_found':
                    detail = 'input CSV not found'
                else:
                    detail = reason
                f.write(f'  k={int(row["k"])}: {detail}\n')
            f.write('\n')

        f.write(f'Results per k (processed only):\n{"-" * 60}\n')
        if not ran.empty:
            f.write(ran.drop(columns='skipped_reason').to_string(index=False))
        else:
            f.write('  No k values produced results.\n')
        f.write('\n\n')

        if not ran.empty and ran['max_rules_any_combo'].max() > 0:
            best_k = ran.loc[ran['max_rules_any_combo'].idxmax(), 'k']
            f.write(
                f'Most rules: k={best_k} '
                f'({ran["max_rules_any_combo"].max()} rules at best combination)\n'
            )
        else:
            f.write('No rules found for any k value.\n')

    print('  > Saved k_comparison_summary.csv and .txt')

    # Cross-k heatmaps.
    all_summaries = []
    for k, sdf in k_summaries.items():
        tmp      = sdf.copy()
        tmp['k'] = k
        all_summaries.append(tmp)

    if not all_summaries:
        print('  > No rules found in any k — skipping cross-k heatmaps.')
        print(f'  > Cross-k comparison saved to {comp_dir}/')
        return k_summaries

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FutureWarning)
        combined = pd.concat(all_summaries, ignore_index=True)

    if combined.empty or 'Lift_threshold' not in combined.columns:
        print('  > No rules found in any k — skipping cross-k heatmaps.')
        print(f'  > Cross-k comparison saved to {comp_dir}/')
        return k_summaries

    combined['Lift_display'] = (
        (combined['Lift_threshold'] / 0.1).round() * 0.1
    ).round(4)

    neutral_lo, neutral_hi = _neutral_window(lift_neutral_half_window)

    for x_col, suffix in [
        ('Support',      'k_support'),
        ('Confidence',   'k_confidence'),
        ('Lift_display', 'k_lift'),
    ]:
        x_label = 'Lift' if 'lift' in suffix else x_col
        is_lift = 'lift' in suffix

        pivot = (
            combined.groupby(['k', x_col])['Number_of_Rules']
            .max()
            .unstack(level=x_col)
            .sort_index(ascending=False)
            .fillna(0)
            .astype(int)
        )
        _render_heatmap(
            pivot       = pivot,
            x_label     = x_label,
            y_label     = 'k',
            title       = (
                f'Max Number of Rules — k vs {x_label}\n'
                f'(darker = more rules; max over the other two parameters)'
            ),
            output_path = comp_dir / f'heatmap_{suffix}.png',
            neutral_lo  = neutral_lo,
            neutral_hi  = neutral_hi,
            x_is_lift   = is_lift,
            row_fmt     = 'k={v}',
            row_height  = 0.60,
        )
        print(f'    > saved k_comparison/heatmap_{suffix}.png')

    print(f'  > Cross-k comparison saved to {comp_dir}/')
    return k_summaries


# ---------------------------------------------------------------------------
# Experiment labelling
# ---------------------------------------------------------------------------

def _experiment_label(
    auto_calibrate: bool,
    sup_min, sup_max, sup_delta,
    conf_min, conf_max, conf_delta,
    lift_min, lift_max, lift_delta,
    lift_neutral_half_window: float,
) -> str:
    """
    Build a human-readable folder name for one experiment configuration.

    conf_min shown in the label is the floor value, not the per-k calibrated
    value — it identifies the configuration, not derived per-k parameters.
    """
    def fmt(v):
        return f'{v:.2f}'

    if auto_calibrate:
        prefix    = 'auto'
        sup_part  = f'sup=auto_d{fmt(sup_delta)}'
        lift_part = f'lift=auto_d{fmt(lift_delta)}_w{fmt(lift_neutral_half_window)}'
    else:
        prefix    = 'manual'
        sup_part  = f'sup={fmt(sup_min)}-{fmt(sup_max)}_d{fmt(sup_delta)}'
        lift_part = (
            f'lift={fmt(lift_min)}-{fmt(lift_max)}'
            f'_d{fmt(lift_delta)}_w{fmt(lift_neutral_half_window)}'
        )

    conf_part = f'conf={fmt(conf_min)}-{fmt(conf_max)}_d{fmt(conf_delta)}'
    return f'{prefix}_{sup_part}_{conf_part}_{lift_part}'


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    regions: list                    = None,
    k_values: list[int]              = None,
    auto_calibrate: bool             = True,
    sup_min: float                   = 0.02,
    sup_max: float                   = 0.50,
    sup_delta: float                 = 0.02,
    conf_min: float                  = 0.05,
    conf_max: float                  = 1.00,
    conf_delta: float                = 0.05,
    lift_min: float                  = 0.0,
    lift_max: float                  = 5.0,
    lift_delta: float                = 0.05,
    lift_neutral_half_window: float  = 0.25,
    base_dir: Path                   = None,
) -> None:
    """
    Run association-rule mining for all regions and k values.

    Parameters
    ----------
    regions                  : list of region names to process.
    k_values                 : neighbourhood sizes produced by feature_importance.py.
    auto_calibrate           : if True, calibrate grid bounds from item frequencies.
    sup_min / sup_max        : support grid bounds (fallback when auto_calibrate=False).
    sup_delta                : support grid step size.
    conf_min / conf_max      : confidence grid bounds; conf_min is the floor when
                               auto_calibrate=True.
    conf_delta               : confidence grid step size.
    lift_min / lift_max      : lift grid bounds; lift_min=0.0 includes negative
                               correlations.
    lift_delta               : lift grid step size.
    lift_neutral_half_window : half-width of the neutral lift window to exclude.
    base_dir                 : project root directory.
    """
    print(
        f'  > Parallel backend — {_CPU_CORES} logical cores / '
        f'{_get_perf_cores()} perf cores (joblib loky)'
    )

    if base_dir is None:
        if Path('/kaggle/working').exists():
            base_dir = Path('/kaggle/working')
        elif Path('/content').exists():
            base_dir = Path('/content')
        else:
            base_dir = Path(__file__).resolve().parent.parent
    base_dir = Path(base_dir)

    if regions is None:
        regions = ['northeast', 'south']
    if k_values is None:
        k_values = [1, 3, 5, 7]

    results_dir = base_dir / 'results'

    exp_label = _experiment_label(
        auto_calibrate=auto_calibrate,
        sup_min=sup_min,   sup_max=sup_max,   sup_delta=sup_delta,
        conf_min=conf_min, conf_max=conf_max, conf_delta=conf_delta,
        lift_min=lift_min, lift_max=lift_max, lift_delta=lift_delta,
        lift_neutral_half_window=lift_neutral_half_window,
    )

    for region in regions:
        important_features_dir = results_dir / region / 'important_features'
        ar_output_dir          = results_dir / region / 'association_rules' / exp_label
        ar_output_dir.mkdir(parents=True, exist_ok=True)

        print('\n' + '=' * 70)
        print(f'ASSOCIATION RULES — {region.upper()}')
        print(f'Experiment: {exp_label}')
        print('=' * 70 + '\n')

        # Select the best available input file for each k value.
        # aggregated_labels_by_sample.csv (preferred): one row per sample;
        #   each transaction is the union of drivers across all CF neighbours
        #   for that sample.  This produces cleaner, less noisy transactions.
        # labels_only_unique.csv (fallback): one row per (sample, CF) pair;
        #   noisier but available when aggregation failed.
        k_labels_map = {}
        for k in k_values:
            p_agg  = important_features_dir / f'k_{k}' / 'aggregated_labels_by_sample.csv'
            p_orig = important_features_dir / f'k_{k}' / 'labels_only_unique.csv'
            if p_agg.exists():
                k_labels_map[k] = (p_agg, 'aggregated (preferred)')
            elif p_orig.exists():
                k_labels_map[k] = (p_orig, 'original (fallback)')

        ks_with_agg  = sum(1 for _, (_, src) in k_labels_map.items() if 'aggregated' in src)
        ks_with_orig = len(k_labels_map) - ks_with_agg

        log_path = ar_output_dir / 'experiment_log.txt'
        with open(log_path, 'w') as f:
            f.write('EXPERIMENT LOG\n')
            f.write(f'{"=" * 70}\n\n')
            f.write(f'Generated     : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'Region        : {region.upper()}\n')
            f.write(f'Experiment    : {exp_label}\n')
            f.write(f'Output dir    : {ar_output_dir}\n\n')
            f.write('Configuration:\n')
            f.write(f'  auto_calibrate          : {auto_calibrate}\n')
            f.write(f'  sup_min / sup_max       : {sup_min} / {sup_max}  '
                    f'(fallback when auto_calibrate=False)\n')
            f.write(f'  sup_delta               : {sup_delta}\n')
            f.write(f'  conf_min / conf_max     : {conf_min} / {conf_max}  '
                    f'(conf_min is floor when calibrating)\n')
            f.write(f'  conf_delta              : {conf_delta}\n')
            f.write(f'  lift_min / lift_max     : {lift_min} / {lift_max}  '
                    f'(lift_min=0.0 includes negative correlations)\n')
            f.write(f'  lift_delta              : {lift_delta}\n')
            f.write(
                f'  lift_neutral_half_window: {lift_neutral_half_window}  '
                f'(excludes [{round(1.0 - lift_neutral_half_window, 4)}, '
                f'{round(1.0 + lift_neutral_half_window, 4)}])\n\n'
            )
            f.write(f'Parallelism   : {_CPU_CORES} logical / '
                    f'{_get_perf_cores()} perf cores (joblib loky)\n\n')

            if not k_labels_map:
                f.write('Input files   : NONE FOUND\n')
                f.write(f'  Searched under: {important_features_dir}\n')
                f.write('  Run feature_importance.py first.\n')
            else:
                f.write(f'Input files found ({len(k_labels_map)} k values):\n')
                f.write(f'{"-" * 60}\n')
                for k_val, (path, src) in sorted(k_labels_map.items()):
                    f.write(f'  k={k_val:<3}  [{src}]  {path}\n')
                f.write(f'\n  {ks_with_agg} aggregated (preferred)\n')
                if ks_with_orig:
                    f.write(f'  {ks_with_orig} original (fallback)\n')

        if not k_labels_map:
            print(f'  > No labels files found under {important_features_dir}')
            print('    Run feature_importance.py first.')
            continue

        print(f'  > Labels found for k = {sorted(k_labels_map.keys())}')
        print(f'    - {ks_with_agg} k values using aggregated format (preferred)')
        if ks_with_orig:
            print(f'    - {ks_with_orig} k values using original format (fallback)')

        k_paths_map = {k: path for k, (path, _) in k_labels_map.items()}

        k_summaries = run_k_comparison(
            k_labels_map             = k_paths_map,
            output_dir               = ar_output_dir,
            auto_calibrate           = auto_calibrate,
            sup_min                  = sup_min,  sup_max  = sup_max,  sup_delta  = sup_delta,
            conf_min                 = conf_min, conf_max = conf_max, conf_delta = conf_delta,
            lift_min                 = lift_min, lift_max = lift_max, lift_delta = lift_delta,
            lift_neutral_half_window = lift_neutral_half_window,
        )

        k_max_per_k = {
            k: int(sdf['Number_of_Rules'].max())
            for k, sdf in k_summaries.items()
            if not sdf.empty and 'Number_of_Rules' in sdf.columns
        }
        sum_rules = sum(k_max_per_k.values())
        max_rules = max(k_max_per_k.values()) if k_max_per_k else 0
        best_k    = max(k_max_per_k, key=k_max_per_k.get) if k_max_per_k else None
        k_ran     = sorted(k_summaries.keys())
        k_skipped = sorted(set(k_labels_map.keys()) - set(k_summaries.keys()))
        k_with_rules = sorted(k for k, n in k_max_per_k.items() if n > 0)

        with open(log_path, 'a') as f:
            f.write(f'\n{"=" * 70}\n')
            f.write('Post-run Summary:\n')
            f.write(f'{"-" * 60}\n')
            f.write(f'  k values ran              : {k_ran}\n')
            f.write(f'  k values skipped          : '
                    f'{k_skipped if k_skipped else "none"}\n')
            f.write(f'  k values with rules       : '
                    f'{k_with_rules if k_with_rules else "none"}\n')
            f.write(f'  Sum of max rules across k : {sum_rules}  '
                    f'(total signal across all k)\n')
            f.write(f'  Max rules in best combo   : {max_rules}  '
                    f'(strongest single combination, k={best_k})\n')
            f.write(f'  Completed at              : '
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

    print('\n' + '=' * 70)
    print('Done.')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()