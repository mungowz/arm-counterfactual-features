"""
microscopic_experiment_association_rules_values.py
==================================================
Runs FP-Growth association-rule mining at the **value level**, starting from
the label-level rules already produced by
macroscopic_experiment_association_rules.py.

Pipeline position
-----------------
    macroscopic_experiment_association_rules.py  →  [this script]

Idea
----
Step 3 (ARM on labels) identifies which *features* (OCCP, SCHL, WKHP, …)
tend to co-occur on the decision boundary.  This script asks the follow-up
question: *which specific values* of those features drive the boundary?

For every (region, k, experiment_label) triple that already has rules.csv
files, this script:

1. Collects every label that appears in any rule (antecedent OR consequent).
2. From transactions_values.csv it keeps only items whose label prefix
   (the part before '=') is in that set.
3. Builds two transaction formats (same as Step 2 / Step 3):
     - aggregated_values_by_sample.csv   — one row per sample (union of all
       CF-neighbour values, deduplicated)
     - values_only_unique.csv            — one row per (Sample_ID, CF_Neighbor_ID)
4. Runs the same support × confidence × lift grid search used in Step 3,
   using the same auto-calibration logic.
5. Writes rules.csv, rules_detailed.csv, summary.csv, heatmaps, and all
   companion artefacts under:
       results/{region}/association_rules_values/{experiment_label}/k_{k}/

Parallelism
-----------
Outer level: k values processed in parallel (one loky worker per k, capped
at P-core count) — same design as macroscopic_experiment_association_rules.py.
Inner level: support thresholds parallelised within each worker.

Input files (resolved relative to base_dir/results/)
-----------------------------------------------------
    {region}/important_features/k_{k}/transactions_values.csv
        Columns: Sample_ID, CF_Neighbor_ID, Counterfactual_Values
        Counterfactual_Values: stringified list of "LABEL=value" strings,
        e.g. "['OCCP=Nurse-Practitioners', 'SCHL=Regular-HS-Diploma']"

    {region}/association_rules/{experiment_label}/k_{k}/rules.csv
        Columns: antecedents, consequents, …
        antecedents / consequents are label names (OCCP, SCHL, …).

Output files
------------
    {region}/association_rules_values/{experiment_label}/k_{k}/
        aggregated_values_by_sample.csv
        values_only_unique.csv
        summary.csv
        exploration_summary.txt
        calibration_log.txt
        item_supports.csv
        heatmaps/heatmap_support_confidence.png  (and two others)
        sup_{x}/conf_{y}/rules.csv
        sup_{x}/conf_{y}/rules_detailed.csv
        sup_{x}/conf_{y}/summary.txt
        sup_{x}/frequent_itemsets.csv
        sup_{x}/frequent_itemsets_summary.txt
    {region}/association_rules_values/{experiment_label}/k_comparison/
        k_comparison_summary.csv
        k_comparison_summary.txt
        heatmap_k_support.png  (and two others)

Public API
----------
main(regions, k_values, base_dir, ...)
    Entry point — mirrors the signature of
    macroscopic_experiment_association_rules.main().
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

matplotlib.use('Agg')

# ---------------------------------------------------------------------------
# Hardware-aware parallelism
#
# Apple Silicon (M1/M2/M3/M4) exposes two core types:
#   - Performance cores (P-cores): high-throughput, suited for CPU-bound work.
#   - Efficiency cores (E-cores): low-power, designed for background tasks.
#
# On unified-memory architectures, spawning more loky workers than P-cores
# does not increase throughput and can cause memory pressure because each
# worker receives a full serialised copy of the encoded DataFrame.  We
# therefore query hw.perflevel0.logicalcpu via sysctl on macOS and cap
# n_jobs at that value.  On all other platforms we fall back to os.cpu_count().
#
# _CPU_CORES  : total logical CPUs (used for informational output only).
# _PERF_CORES : cached result of _detect_perf_cores(); None until first call.
# ---------------------------------------------------------------------------

_CPU_CORES: int = os.cpu_count() or 1
_PERF_CORES: int | None = None


def _detect_perf_cores() -> int:
    """
    Detect the number of performance (P) cores available on this machine.

    On Apple Silicon, sysctl hw.perflevel0.logicalcpu returns the count of
    high-performance cores only, excluding efficiency cores.  Capping joblib
    workers at this value avoids over-subscription on unified-memory chips
    where each extra worker consumes an additional serialised DataFrame copy.

    On non-Darwin platforms (Linux, Windows) or when sysctl is unavailable,
    the function falls back to os.cpu_count() so behaviour is unchanged on
    standard x86 clusters.

    Returns
    -------
    int
        Number of P-cores (Apple Silicon) or total logical CPUs (all others).
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
                f'(of {_CPU_CORES} logical total) — joblib capped at {p_cores} workers'
            )
            return p_cores
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            pass
    return _CPU_CORES


def _get_perf_cores() -> int:
    """
    Return the cached P-core count, computing it on the first call.

    The result is stored in the module-level _PERF_CORES variable so that
    the sysctl subprocess is only invoked once per interpreter session,
    regardless of how many times this function is called across parallel
    workers or repeated pipeline runs.
    """
    global _PERF_CORES
    if _PERF_CORES is None:
        _PERF_CORES = _detect_perf_cores()
    return _PERF_CORES


# ---------------------------------------------------------------------------
# Neutral-window helper
#
# Rules whose lift falls in the interval [1 - half_window, 1 + half_window]
# indicate near-independence between antecedent and consequent and are
# therefore excluded from the analysis.  The default half_window of 0.25
# produces the exclusion band [0.75, 1.25].
#
# Excluding near-neutral rules keeps the output focused on items that
# genuinely co-occur (lift > 1.25) or are mutually exclusive (lift < 0.75)
# on the decision boundary, and avoids cluttering the result set with
# coincidental associations.
# ---------------------------------------------------------------------------

def _neutral_window(half_window: float) -> tuple[float, float]:
    """
    Compute the symmetric neutral-lift exclusion window.

    Parameters
    ----------
    half_window : float
        Half-width of the neutral band.  A value of 0.25 produces [0.75, 1.25].

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) of the exclusion window, rounded to 4
        decimal places to avoid floating-point comparison artefacts.
    """
    lo = round(1.0 - half_window, 4)
    hi = round(1.0 + half_window, 4)
    return lo, hi


# ---------------------------------------------------------------------------
# Experiment labelling
#
# _experiment_label() builds a human-readable folder name that encodes the
# grid configuration used for a run.  This mirrors the identical function in
# macroscopic_experiment_association_rules.py and serves the same purpose:
# ensure that re-running the microscopic step with different parameters writes
# to a distinct subdirectory rather than overwriting previous results.
#
# The label is injected into the output path in main() so the full tree is:
#     results/{region}/association_rules_values/{experiment_label}/k_{k}/
#
# which is consistent with the macroscopic layout:
#     results/{region}/association_rules/{experiment_label}/k_{k}/
# ---------------------------------------------------------------------------

def _experiment_label(
    auto_calibrate: bool,
    sup_min: float, sup_max: float, sup_delta: float,
    conf_min: float, conf_max: float, conf_delta: float,
    lift_min: float, lift_max: float, lift_delta: float,
    lift_neutral_half_window: float,
) -> str:
    """
    Build a filesystem-safe experiment label encoding the grid configuration.

    The label is inserted as a path component between association_rules_values/
    and k_{k}/ so that different configurations coexist without overwriting
    each other.  The format mirrors macroscopic_experiment_association_rules
    for consistency across pipeline stages.

    Parameters
    ----------
    auto_calibrate : bool
        When True the prefix is 'auto' and sup/lift bounds are omitted
        (they are derived per-k from the data).  When False the prefix is
        'manual' and all bounds are encoded.
    sup_min, sup_max, sup_delta : float
        Support grid parameters (manual mode only for min/max).
    conf_min, conf_max, conf_delta : float
        Confidence grid parameters.  conf_min in the label is the floor
        value, not the per-k calibrated value.
    lift_min, lift_max, lift_delta : float
        Lift grid parameters (manual mode only for min/max).
    lift_neutral_half_window : float
        Half-width of the neutral lift band; encoded so that runs with
        different window widths produce distinguishable directory names.

    Returns
    -------
    str
        Example (auto):
            'auto_sup=auto_d0.02_conf=0.05-1.00_d0.05_lift=auto_d0.05_w0.25'
        Example (manual):
            'manual_sup=0.02-0.50_d0.02_conf=0.05-1.00_d0.05_lift=0.00-5.00_d0.05_w0.25'
    """
    def fmt(v: float) -> str:
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
# Label extraction from Step 3 rules
#
# Step 3 (macroscopic ARM) produces rules.csv files whose antecedents and
# consequents are ACS feature names (e.g. 'OCCP', 'SCHL', 'WKHP').  This
# section collects the union of all such names across every rules.csv that
# exists for a given (region, k) pair.  That union — the "active labels" set
# — is then used to filter transactions_values.csv so that only value-level
# items belonging to the relevant features are kept.
#
# Design note: we scan *all* experiment-label subdirectories for a given k
# (via rglob) so that results from multiple grid runs are merged.  This is
# intentional: if the user ran Step 3 twice with different parameters, both
# sets of rules contribute to the active labels.
# ---------------------------------------------------------------------------

def collect_active_labels(rules_csv_paths: list[Path]) -> set[str]:
    """
    Return the union of all label names referenced in the provided rules files.

    Each rules.csv produced by Step 3 stores antecedents and consequents as
    plain label names (e.g. 'OCCP', 'SCHL').  Multi-item antecedents and
    consequents are encoded as comma-separated strings within a single cell
    (e.g. 'OCCP, SCHL'); this function splits those strings correctly.

    Files that are missing, unreadable, or empty are silently skipped so that
    a single corrupt file cannot block the entire pipeline.

    Parameters
    ----------
    rules_csv_paths : list[Path]
        Paths to rules.csv files, typically returned by find_rules_csvs().
        May contain paths to non-existent files; those are ignored.

    Returns
    -------
    set[str]
        Union of all label names found across all files.  Returns an empty
        set if no files exist or all files are empty / unreadable.
    """
    active: set[str] = set()
    for p in rules_csv_paths:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        for col in ('antecedents', 'consequents'):
            if col not in df.columns:
                continue
            for cell in df[col].dropna():
                # Multi-item antecedents / consequents are stored as
                # comma-separated label names within a single cell.
                for label in str(cell).split(','):
                    label = label.strip()
                    if label:
                        active.add(label)
    return active


def find_rules_csvs(
    results_dir: Path,
    region: str,
    k: int,
) -> list[Path]:
    """
    Recursively find all rules.csv files produced by Step 3 for a given
    (region, k) pair, across all experiment-label subdirectories.

    The expected directory structure is:
        results/{region}/association_rules/{experiment_label}/k_{k}/sup_{x}/conf_{y}/rules.csv

    The experiment label encodes the grid parameters used in Step 3 and is
    not known at call time, so we use rglob with a fixed suffix pattern.
    Multiple experiment-label directories may exist if Step 3 was re-run with
    different parameters; all are included.

    Parameters
    ----------
    results_dir : Path
        Root results directory (typically base_dir / 'results').
    region : str
        Region name (e.g. 'northeast', 'south').
    k : int
        CF neighbourhood size.

    Returns
    -------
    list[Path]
        Sorted list of all matching rules.csv paths.  Returns an empty list
        if the association_rules directory does not exist for this region.
    """
    base = results_dir / region / 'association_rules'
    if not base.exists():
        return []
    return sorted(base.rglob(f'k_{k}/sup_*/conf_*/rules.csv'))


# ---------------------------------------------------------------------------
# Transaction builders
#
# transactions_values.csv (produced by Step 2) stores one row per
# (Sample_ID, CF_Neighbor_ID) pair.  Each row's Counterfactual_Values column
# contains a stringified Python list of 'LABEL=value' strings describing
# every feature that differs between the original instance and its k-th
# counterfactual neighbour.
#
# This section builds two transaction formats from that raw file:
#
#   1. Pair-level (values_only_unique.csv)
#      One row per (Sample_ID, CF_Neighbor_ID).  Values are filtered to keep
#      only items whose label prefix is in active_labels.  This mirrors the
#      raw granularity of Step 2 output and is saved for reference, but is
#      NOT used as input to FP-Growth (see aggregated format below).
#
#   2. Aggregated (aggregated_values_by_sample.csv)
#      One row per Sample_ID.  Values are the deduplicated union of all
#      filtered items across every CF neighbour of that sample.  This is the
#      format consumed by FP-Growth: each sample is treated as a single
#      "shopping basket" containing the set of value-level changes that appear
#      in any of its counterfactual explanations.
#
# Filtering rationale: restricting items to active_labels ensures that the
# value-level rules are anchored to the label-level patterns already
# identified by Step 3.  Items belonging to non-active labels are discarded
# because their labels never appear in a Step 3 rule, meaning they are not
# informative about the decision boundary co-occurrence structure we want to
# characterise at the value level.
# ---------------------------------------------------------------------------

def build_value_transactions(
    transactions_values_path: Path,
    active_labels: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read transactions_values.csv and build pair-level and aggregated
    transaction DataFrames, retaining only items whose label prefix belongs
    to *active_labels*.

    Parameters
    ----------
    transactions_values_path : Path
        Path to transactions_values.csv.
        Required columns:
            Sample_ID              — integer sample identifier.
            CF_Neighbor_ID         — integer counterfactual neighbour identifier.
            Counterfactual_Values  — stringified Python list of 'LABEL=value'
                                     strings, e.g. "['OCCP=Nurse-Practitioners',
                                     'SCHL=Regular-HS-Diploma']".
    active_labels : set[str]
        Set of label names to retain (collected from Step 3 rules.csv files).
        Items whose label prefix (the portion before '=') is not in this set
        are silently discarded.

    Returns
    -------
    agg_df : pd.DataFrame
        Aggregated transactions — one row per Sample_ID.
        Columns: Sample_ID, Values, Num_Values, Num_CF_Neighbors.
        Values contains the sorted, deduplicated union of all filtered
        'LABEL=value' strings across every CF neighbour of that sample.
        This is the DataFrame passed to encode_transactions() for FP-Growth.
    pair_df : pd.DataFrame
        Pair-level transactions — one row per (Sample_ID, CF_Neighbor_ID).
        Columns: Sample_ID, CF_Neighbor_ID, Values.
        Saved to disk for traceability but not used directly for mining.

    Filtering logic
    ---------------
    For each item in Counterfactual_Values:
      1. Parse the stringified list with ast.literal_eval.
      2. Split on the first '=' to extract the label prefix.
      3. Keep the item only if the prefix is in active_labels.
      4. Items that do not contain '=' (malformed) are silently dropped.
    Rows that produce zero filtered items are excluded from both outputs,
    so samples whose only CF changes involve non-active labels disappear
    from the transaction set entirely.
    """
    print(f'  > Loading {transactions_values_path.name} ...')
    df = pd.read_csv(transactions_values_path)

    required = {'Sample_ID', 'CF_Neighbor_ID', 'Counterfactual_Values'}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f'transactions_values.csv is missing columns: {missing}  '
            f'(found: {df.columns.tolist()})'
        )

    print(f'    {len(df):,} rows, {df["Sample_ID"].nunique():,} unique samples')

    # ── Parse and filter — vectorised ────────────────────────────────
    # Using apply() instead of iterrows() is ~10-50× faster on large DataFrames
    # because it avoids per-row Python overhead and keeps the hot path inside
    # pandas rather than a pure-Python loop.
    #
    # _filter_items: parse the stringified list and retain only items whose
    # label prefix (before '=') is in active_labels.  Returns an empty list
    # for malformed cells or rows with no matching items (dropped downstream).
    def _filter_items(raw_values: str) -> list:
        try:
            items = ast.literal_eval(raw_values)
        except (ValueError, SyntaxError):
            return []
        return [
            item for item in items
            if '=' in str(item) and str(item).split('=', 1)[0] in active_labels
        ]

    df = df.copy()
    df['_filtered'] = df['Counterfactual_Values'].apply(_filter_items)

    # Drop rows that produced zero filtered items.
    df = df[df['_filtered'].map(len) > 0].reset_index(drop=True)

    # ── Pair-level DataFrame ──────────────────────────────────────────
    pair_df = df[['Sample_ID', 'CF_Neighbor_ID', '_filtered']].copy()
    pair_df = pair_df.rename(columns={'_filtered': 'Values'})
    pair_df['Values'] = pair_df['Values'].apply(str)
    pair_df = pair_df.reset_index(drop=True)

    # ── Aggregated DataFrame ──────────────────────────────────────────
    # For each Sample_ID: union all filtered items across its CF neighbours,
    # deduplicate, sort for deterministic output, and count unique CF IDs.
    agg_df = (
        df.groupby('Sample_ID', sort=False)
        .agg(
            _items=('_filtered',
                    lambda x: sorted({item for sublist in x for item in sublist})),
            Num_CF_Neighbors=('CF_Neighbor_ID', 'nunique'),
        )
        .reset_index()
    )
    agg_df['Values']     = agg_df['_items'].apply(str)
    agg_df['Num_Values'] = agg_df['_items'].apply(len)
    agg_df = agg_df[['Sample_ID', 'Values', 'Num_Values', 'Num_CF_Neighbors']]

    n_kept_samples = agg_df['Sample_ID'].nunique()
    n_total        = df['Sample_ID'].nunique()
    print(
        f'  > After filtering to active labels {sorted(active_labels)}: '
        f'{n_kept_samples:,} / {n_total:,} samples retained, '
        f'{len(pair_df):,} (sample, CF) pairs'
    )

    return agg_df, pair_df


# ---------------------------------------------------------------------------
# One-hot encoding
#
# mlxtend's fpgrowth() expects a Boolean DataFrame where each row is a
# transaction and each column is a possible item.  TransactionEncoder handles
# the conversion: it builds the global item vocabulary from all transactions
# and then produces a True/False matrix indicating item presence per row.
#
# At the value level, each item is a 'LABEL=value' string such as
# 'OCCP=Software-Developers' or 'WKHP=Full-Time'.  The number of unique items
# can reach into the hundreds (one per distinct value per active label), but
# remains far smaller than label-level encoding because only active-label
# items are present after the filtering step.
# ---------------------------------------------------------------------------

def encode_transactions(values_col: pd.Series, col_name: str = 'Values') -> pd.DataFrame:
    """
    One-hot-encode a Series of stringified Python lists into a Boolean
    DataFrame suitable for mlxtend's fpgrowth().

    Each element of *values_col* must be a string representation of a Python
    list, e.g. "['OCCP=Nurse-Practitioners', 'WKHP=Full-Time']".  The function
    uses ast.literal_eval to parse each string back into a list, then passes
    the resulting iterable of lists to TransactionEncoder.

    Parameters
    ----------
    values_col : pd.Series
        Series whose elements are stringified lists of item strings.
        Typically the 'Values' column of aggregated_values_by_sample.csv.
    col_name : str, optional
        Label used in the progress print statement for diagnostics.
        Does not affect the output DataFrame.  Default: 'Values'.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame with shape (n_transactions, n_unique_items).
        Column names are the unique item strings sorted alphabetically by
        TransactionEncoder.  True indicates that the item is present in that
        transaction (sample).
    """
    itemsets = values_col.apply(ast.literal_eval)
    te       = TransactionEncoder()
    te_ary   = te.fit(itemsets).transform(itemsets)
    enc_df   = pd.DataFrame(te_ary, columns=te.columns_)
    print(f'  > Encoded {col_name}: {len(enc_df):,} transactions × {enc_df.shape[1]:,} items')
    return enc_df


# ---------------------------------------------------------------------------
# Filesystem helpers
#
# After the parallel grid search, many sup_*/conf_*/ subdirectories may be
# empty (no rules satisfied the threshold combination).  cleanup_empty_folders
# removes those directories to keep the output tree navigable and to avoid
# confusing downstream consumers that might interpret an empty directory as a
# valid result.  It mirrors the same function in Step 3 so the output
# structure is consistent across both ARM stages.
# ---------------------------------------------------------------------------

def cleanup_empty_folders(output_dir: Path) -> tuple[int, int]:
    """
    Remove conf_* subdirectories that contain no valid rules.csv, then remove
    any sup_* directories that are left empty after conf_* cleanup.

    A conf_* directory is considered empty if either its rules.csv does not
    exist or the file exists but contains no data rows (e.g. all rules were
    filtered out by the neutral-lift window or the confidence threshold).

    Parameters
    ----------
    output_dir : Path
        The k_{k}/ directory whose sup_*/conf_*/ tree should be cleaned.

    Returns
    -------
    tuple[int, int]
        (n_conf_removed, n_sup_removed) — counts of removed directories,
        useful for the exploration_summary.txt log.
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
                try:
                    should_remove = pd.read_csv(rules_csv).empty
                except Exception:
                    should_remove = True
            if should_remove:
                shutil.rmtree(conf_dir)
                removed_conf += 1
        # Remove the parent sup_* dir if all its conf_* children were deleted.
        if not list(sup_dir.glob('conf_*')):
            shutil.rmtree(sup_dir)
            removed_sup += 1
    return removed_conf, removed_sup


# ---------------------------------------------------------------------------
# Core worker — parallelised over support thresholds (inner parallel level)
#
# _process_one_support() is the innermost unit of work.  It is dispatched by
# explore_association_rules_values() via joblib.Parallel, one invocation per
# support threshold value.  Each worker:
#
#   1. Runs FP-Growth on the full encoded DataFrame at its assigned support.
#   2. Derives association rules at the lowest confidence in the grid
#      (additional confidence filtering is applied in-process per conf value).
#   3. Excludes near-neutral rules using the pre-computed lift window.
#   4. Iterates over confidence × lift combinations, writing rules.csv,
#      rules_detailed.csv, and summary.txt for each non-empty combination.
#   5. Returns a list of summary-row dicts (one per support × conf × lift
#      triple) for aggregation into the global summary.csv.
#
# rules_detailed.csv differs from rules.csv by including per-item antecedent
# and consequent support, rule length statistics, and two extra columns
# (antecedent_labels, consequent_labels) that map each value-level item back
# to its originating ACS feature name.  This back-mapping makes it easy to
# compare value-level rules with the label-level rules from Step 3.
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
    Run FP-Growth at *min_sup* and generate association rules for every
    (confidence, lift) combination in the pre-computed grids.

    This function is designed to be called as a joblib.Parallel worker.  It
    is stateless with respect to module-level variables (all inputs are passed
    as arguments) so that loky can serialise and dispatch it without issues.

    Parameters
    ----------
    min_sup : float
        Minimum support threshold for FP-Growth.
    sup_idx : int
        1-based index of this support value within the full grid (for logging).
    n_sup : int
        Total number of support values in the grid (for logging).
    df : pd.DataFrame
        Boolean-encoded transaction DataFrame (output of encode_transactions()).
    output_dir : Path
        k_{k}/ directory under which sup_*/conf_*/ subdirectories are created.
    confidence_grid : array-like of float
        Sorted ascending confidence thresholds to iterate over.
    lift_grid_used : list of float
        Lift thresholds after removing the neutral window.
    lift_window_lo : float
        Lower bound of the neutral lift window (rules with lift in
        [lift_window_lo, lift_window_hi] are excluded).
    lift_window_hi : float
        Upper bound of the neutral lift window.

    Returns
    -------
    list[dict]
        One summary-row dict per (support, confidence, lift) triple that
        produced at least some rules before the lift filter.  Returns an
        empty list if FP-Growth found no frequent itemsets at this support.
    """
    output_dir = Path(output_dir)
    sup_label  = f'{min_sup:.2f}'
    sup_dir    = output_dir / f'sup_{sup_label}'
    sup_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n  [{sup_idx}/{n_sup}] support = {min_sup}')
    print('    > running FP-Growth...')

    frequent_itemsets = fpgrowth(df, min_support=min_sup, use_colnames=True)
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
    print(
        f'    > breakdown: '
        f'{", ".join(f"len={k}: {v}" for k, v in itemsets_by_len.items())}'
    )

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

    try:
        # Generate all rules at the lowest confidence threshold in one shot.
        # Rules for higher confidence values are derived by subsetting this
        # master DataFrame, avoiding repeated calls to association_rules().
        all_rules = association_rules(
            frequent_itemsets,
            metric='confidence',
            min_threshold=float(confidence_grid[0]),
        )
    except ValueError:
        return local_summary_rows

    if len(all_rules) == 0:
        return local_summary_rows

    # Exclude rules whose lift falls within the neutral window.  These rules
    # indicate near-independence between antecedent and consequent and carry
    # little interpretive value for decision-boundary analysis.
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

        # conviction is undefined (inf) when confidence == 1.0 because the
        # consequent is never absent given the antecedent.  We replace ±inf
        # with NaN so the CSV remains machine-readable by downstream tools.
        conviction_vals = rules['conviction'].replace([np.inf, -np.inf], np.nan)

        # ── Compact output (rules.csv) ────────────────────────────────
        # Antecedents and consequents are sorted and joined as strings for
        # readability; raw and percentage formats are both included so
        # consumers can choose their preferred representation.
        fmt = pd.DataFrame()
        fmt['antecedents']    = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
        fmt['consequents']    = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
        fmt['support_raw']    = [f'{v:.4f}' for v in rules['support']]
        fmt['support_pct']    = (rules['support'] * 100).round(2)
        fmt['confidence_raw'] = [f'{v:.4f}' for v in rules['confidence']]
        fmt['confidence_pct'] = (rules['confidence'] * 100).round(2)
        fmt['lift']           = rules['lift'].round(4)
        fmt['leverage']       = rules['leverage'].round(6)
        fmt['conviction']     = conviction_vals.round(4)

        # ── Detailed output (rules_detailed.csv) ─────────────────────
        # Extends rules.csv with per-side support, rule length columns, and
        # antecedent_labels / consequent_labels which map each value-level
        # item (e.g. 'OCCP=Software-Developers') back to its ACS feature name
        # ('OCCP').  This allows direct cross-referencing with Step 3 results.
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
        det['support_pct']            = (rules['support'] * 100).round(2)
        det['confidence_raw']         = [f'{v:.4f}' for v in rules['confidence']]
        det['confidence_pct']         = (rules['confidence'] * 100).round(2)
        det['lift']                   = rules['lift'].round(4)
        det['leverage']               = rules['leverage'].round(6)
        det['conviction']             = conviction_vals.round(4)

        # Derive the originating ACS label for each item in the rule sides.
        # The split is on '=' so items without '=' (which should not exist
        # after the filtering step but are guarded here defensively) are
        # excluded from the label set.
        det['antecedent_labels'] = rules['antecedents'].apply(
            lambda x: ', '.join(sorted({i.split('=')[0] for i in x if '=' in i}))
        )
        det['consequent_labels'] = rules['consequents'].apply(
            lambda x: ', '.join(sorted({i.split('=')[0] for i in x if '=' in i}))
        )

        conf_dir.mkdir(parents=True, exist_ok=True)
        fmt.to_csv(conf_dir / 'rules.csv',          index=False)
        det.to_csv(conf_dir / 'rules_detailed.csv', index=False)

        with open(conf_dir / 'summary.txt', 'w') as f:
            f.write('Association Rules Summary (value level)\n')
            f.write(f'{"=" * 60}\n\n')
            f.write('Parameters:\n')
            f.write(f'  Min Support:    {min_sup}\n')
            f.write(f'  Min Confidence: {min_conf}\n')
            f.write(
                f'  Neutral Lift Window (excluded): '
                f'[{lift_window_lo}, {lift_window_hi}]\n\n'
            )
            f.write('Results:\n')
            f.write(f'  Frequent Itemsets: {len(frequent_itemsets)}\n')
            f.write(f'  Association Rules: {len(rules)}\n\n')
            f.write('Statistics:\n')
            f.write(f'  Avg Support:     {rules["support"].mean() * 100:.2f}%\n')
            f.write(f'  Avg Confidence:  {rules["confidence"].mean() * 100:.2f}%\n')
            f.write(f'  Avg Lift:        {rules["lift"].mean():.4f}\n')
            f.write(
                f'  Lift Range:      {rules["lift"].min():.4f} — '
                f'{rules["lift"].max():.4f}\n'
            )
            f.write(f'  Avg Leverage:    {rules["leverage"].mean():.6f}\n')
            f.write(f'  Avg Rule Length: {det["rule_length"].mean():.2f}\n\n')
            n_inf = conviction_vals.isna().sum()
            if n_inf > 0:
                f.write(
                    f'  Note: {n_inf} rule(s) have confidence=1.0 '
                    f'(conviction=inf → saved as NaN)\n\n'
                )
            f.write(f'Top 10 Rules (by Lift):\n{"-" * 60}\n')
            for idx, row in fmt.head(10).iterrows():
                f.write(
                    f'{idx + 1}. {row["antecedents"]} => {row["consequents"]}\n'
                    f'   support={row["support_pct"]:.2f}% | '
                    f'confidence={row["confidence_pct"]:.2f}% | '
                    f'lift={row["lift"]:.4f} | '
                    f'leverage={row["leverage"]:.6f}\n\n'
                )

        print(
            f'    > [conf={min_conf}] {len(rules)} rules saved to '
            f'{conf_dir.relative_to(output_dir)}/'
        )

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
                'Num_FI_length_3plus':   sum(v for k2, v in itemsets_by_len.items() if k2 >= 3),
            })

    return local_summary_rows


# ---------------------------------------------------------------------------
# Auto-calibration
#
# Value-level item frequencies differ substantially from label-level
# frequencies: the same ACS feature (e.g. OCCP) now contributes dozens of
# distinct items (one per occupation code) instead of a single label, so
# individual item supports are much lower.  Applying the same fixed support
# grid used in Step 3 would yield no frequent itemsets at the value level.
#
# calibrate_parameters() solves this by deriving data-driven bounds:
#
#   sup_min  — product of the two rarest item frequencies, snapped up to the
#              nearest sup_delta multiple.  This is the tightest threshold at
#              which at least some 2-itemsets can theoretically exist.
#
#   sup_max  — highest support at which 2-itemsets are still found, plus a
#              small decay margin (_DECAY_STEPS * sup_delta) to include the
#              region where 2-itemsets begin to vanish.  Capped at 0.50.
#
#   lift_max — reciprocal of the rarest item frequency, rounded up to the
#              nearest 0.5 step, capped at 10.0.  This bounds the maximum
#              achievable lift given the marginal probabilities in the data.
#
#   conf_min — floor-snapped minimum observed confidence from a probe run
#              at sup_min, or conf_min_floor if the probe produces no rules.
#
# Returns None if no 2-itemsets are found anywhere in the scan range,
# signalling that the transaction set is too sparse for meaningful ARM.
# ---------------------------------------------------------------------------

def calibrate_parameters(
    encoded_df: pd.DataFrame,
    sup_delta: float      = 0.02,
    lift_delta: float     = 0.05,
    conf_delta: float     = 0.05,
    conf_min_floor: float = 0.05,
    conf_max: float       = 1.00,
) -> dict | None:
    """
    Derive data-driven support, confidence and lift grid bounds from the
    item frequency distribution of the encoded transaction DataFrame.

    The calibration strategy is described in the section header above.
    All derived bounds are snapped to grid-aligned values (multiples of the
    corresponding delta parameter) so they integrate cleanly with the uniform
    grids used in explore_association_rules_values().

    Parameters
    ----------
    encoded_df : pd.DataFrame
        Boolean-encoded transaction DataFrame (output of encode_transactions()).
        Column means give per-item support frequencies.
    sup_delta : float
        Support grid step size, used to snap sup_min and define the scan grid.
    lift_delta : float
        Lift grid step size (stored in the returned dict for reference only).
    conf_delta : float
        Confidence grid step size, used to snap conf_min.
    conf_min_floor : float
        Minimum allowed value for conf_min regardless of calibration result.
        Prevents the grid from starting too close to zero.
    conf_max : float
        Maximum confidence threshold (always 1.00; included for symmetry).

    Returns
    -------
    dict or None
        Calibrated parameter dict with keys:
            sup_min, sup_max, natural_sup_max, sup_delta,
            conf_min, conf_max, conf_delta,
            lift_min, lift_max, lift_delta,
            raw_sup_min, raw_lift_max,
            min_conf_observed, max_conf_observed,
            item_supports (dict mapping item → support).
        Returns None if no 2-itemsets are found across the entire scan range,
        indicating the transaction set is too sparse for mining.
    """
    print('  > Calibrating parameters from item frequencies...')

    item_supports = encoded_df.mean().sort_values()
    if len(item_supports) < 2:
        print('  > WARNING: fewer than 2 items — cannot form pairwise rules.')
        return None

    rarest = item_supports.iloc[0]   # support of the least frequent item
    second = item_supports.iloc[1]   # support of the second least frequent item
    freq_2 = item_supports.iloc[-2]  # support of the second most frequent item

    # Theoretical minimum support for a 2-itemset: product of the two rarest
    # item marginals (assumes independence as a lower bound).  Snapped up to
    # the nearest sup_delta multiple and floored at sup_delta itself.
    raw_sup_min = rarest * second
    sup_min = max(
        round(np.floor(raw_sup_min / sup_delta) * sup_delta, 4), sup_delta
    )

    # Scan upward from sup_min to find the natural support ceiling: the
    # highest threshold at which at least one 2-itemset still exists.
    # We stop scanning as soon as 2-itemsets disappear after having been
    # present (the first drop-off point), to avoid scanning the full range.
    _DECAY_STEPS            = 4   # extra steps above natural ceiling for decay margin
    scan_grid               = np.round(np.arange(sup_min, freq_2 + sup_delta / 2, sup_delta), 4)
    natural_sup_max         = sup_min
    prev_had_2itemsets      = False
    fi_first_with_2itemsets = None   # retained for conf_min probe below

    for t in scan_grid:
        fi = fpgrowth(encoded_df, min_support=t, use_colnames=True)
        if fi.empty:
            break
        has_2 = (fi['itemsets'].apply(len) >= 2).any()
        if has_2:
            if fi_first_with_2itemsets is None:
                fi_first_with_2itemsets = fi   # save the first result for conf calibration
            natural_sup_max    = t
            prev_had_2itemsets = True
        elif prev_had_2itemsets:
            # 2-itemsets have appeared and then vanished — stop scanning.
            break

    # Add decay margin and cap at 0.50 to avoid trivially common itemsets.
    sup_max = min(
        round(natural_sup_max + _DECAY_STEPS * sup_delta, 4), 0.50
    )

    if not prev_had_2itemsets:
        print('    - WARNING: no 2-itemsets found — transactions too sparse.')
        return None

    # lift_max: theoretical maximum lift is 1 / P(rarest_item).  Round up
    # to the nearest 0.5 for a clean grid and cap at 10.0 to avoid enormous
    # grids when rarest items are very infrequent.
    raw_lift_max = 1.0 / rarest
    lift_max     = min(round(np.ceil(raw_lift_max * 2) / 2, 1), 10.0)

    # conf_min: run a probe at the lowest support with 2-itemsets and observe
    # the minimum confidence across all generated rules.  Snap down to the
    # nearest conf_delta multiple, but floor at conf_min_floor.
    calibrated_conf_min = conf_min_floor
    min_conf_observed   = None
    max_conf_observed   = None

    try:
        if fi_first_with_2itemsets is not None and not fi_first_with_2itemsets.empty:
            rules_probe = association_rules(
                fi_first_with_2itemsets, metric='confidence', min_threshold=0.01
            )
            if not rules_probe.empty:
                min_conf_observed   = rules_probe['confidence'].min()
                max_conf_observed   = rules_probe['confidence'].max()
                calibrated          = round(
                    np.floor(min_conf_observed / conf_delta) * conf_delta, 4
                )
                calibrated_conf_min = max(calibrated, conf_min_floor)
    except Exception as exc:
        print(f'  > WARNING: conf_min calibration failed ({exc!r}) — using floor={conf_min_floor}')

    return {
        'sup_min':           sup_min,
        'sup_max':           sup_max,
        'natural_sup_max':   natural_sup_max,
        'sup_delta':         sup_delta,
        'conf_min':          calibrated_conf_min,
        'conf_max':          conf_max,
        'conf_delta':        conf_delta,
        'lift_min':          0.0,
        'lift_max':          lift_max,
        'lift_delta':        lift_delta,
        'raw_sup_min':       round(raw_sup_min, 6),
        'raw_lift_max':      round(raw_lift_max, 4),
        'min_conf_observed': round(min_conf_observed, 4) if min_conf_observed is not None else None,
        'max_conf_observed': round(max_conf_observed, 4) if max_conf_observed is not None else None,
        'item_supports':     item_supports.round(4).to_dict(),
    }


# ---------------------------------------------------------------------------
# Heatmaps
#
# Three heatmaps are produced per k_{k}/ directory, each visualising the
# maximum number of rules found across one pair of parameters while
# marginalising over the third:
#
#   heatmap_support_confidence.png  — rows=Support,    cols=Confidence
#   heatmap_support_lift.png        — rows=Support,    cols=Lift
#   heatmap_confidence_lift.png     — rows=Confidence, cols=Lift
#
# For lift axes, the neutral window [lift_window_lo, lift_window_hi] is
# excluded and columns are trimmed at the last non-zero entry to keep the
# chart compact.  Lift values are binned to lift_display_step (default 0.1)
# to reduce axis clutter when the lift grid step (lift_delta) is finer than
# 0.1.
#
# Cell annotation: the rule count is printed inside each cell.  Text colour
# is white when the cell value exceeds 55 % of the maximum for legibility
# against the dark 'YlOrBr' colourmap background.
# ---------------------------------------------------------------------------

def plot_heatmaps(
    summary_df: pd.DataFrame,
    output_dir: Path,
    lift_display_step: float        = 0.1,
    lift_neutral_half_window: float = 0.25,
    lift_delta: float               = 0.05,
) -> None:
    """
    Generate the three parameter-space heatmaps for a single k_{k}/ directory.

    Each heatmap shows the maximum number of association rules obtained across
    the full grid, marginalising over the parameter not shown on either axis.
    This allows the analyst to identify at a glance which (support, confidence,
    lift) regions produce rich rule sets.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Aggregated summary DataFrame returned by explore_association_rules_values().
        Must contain columns: Support, Confidence, Lift_threshold, Number_of_Rules.
    output_dir : Path
        k_{k}/ directory; heatmaps are written to output_dir/heatmaps/.
    lift_display_step : float
        Binning resolution for the lift axis (default 0.1).  Lift values from
        the grid are rounded to the nearest multiple of this value before
        pivoting, reducing axis tick density when lift_delta < 0.1.
    lift_neutral_half_window : float
        Half-width of the neutral lift exclusion band.  Lift columns within
        [1 - half_window, 1 + half_window] are removed from lift-axis charts.
    lift_delta : float
        Stored for reference in axis labels only; does not affect rendering.
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

        if x_is_lift:
            pivot = pivot.loc[
                :, ~pivot.columns.to_series().between(neutral_lo, neutral_hi, inclusive='both')
            ]
            if (pivot != 0).any(axis=0).any():
                last_nz = int(np.where((pivot != 0).any(axis=0).values)[0].max())
                pivot   = pivot.iloc[:, :last_nz + 1]

        n_cols = len(pivot.columns)
        n_rows = len(pivot.index)
        fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.75), max(4, n_rows * 0.55)))

        img = ax.imshow(pivot.values, aspect='auto', cmap='YlOrBr', interpolation='nearest')

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(
            [f'{v:.2f}' if isinstance(v, float) else str(v) for v in pivot.columns],
            rotation=40, ha='right', fontsize=8,
        )
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f'{v:.2f}' for v in pivot.index], fontsize=8)

        x_label = 'Lift' if x_is_lift else x_col
        ax.set_xlabel(x_label, fontsize=11, labelpad=8)
        ax.set_ylabel(y_col,   fontsize=11, labelpad=8)
        ax.set_title(
            f'Max Number of Rules — {y_col} vs {x_label}\n'
            f'(value level, darker = more rules; max over the third parameter)',
            fontsize=11, pad=14,
        )

        max_val = pivot.values.max() if pivot.values.max() > 0 else 1
        for ri in range(n_rows):
            for ci in range(n_cols):
                val = pivot.values[ri, ci]
                if val > 0:
                    txt_color = 'white' if (val / max_val) > 0.55 else 'black'
                    ax.text(ci, ri, str(val), ha='center', va='center',
                            fontsize=7, color=txt_color)

        cbar = plt.colorbar(img, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label('Number of Rules', fontsize=9)
        plt.tight_layout()
        fig.savefig(heatmap_dir / f'heatmap_{suffix}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'    > saved heatmaps/heatmap_{suffix}.png')

    print(f'  > heatmaps saved to {heatmap_dir}/')


# ---------------------------------------------------------------------------
# Calibration log writer
#
# _write_calibration_log() produces two artefacts per k_{k}/ directory:
#
#   item_supports.csv  — per-item support frequencies (raw and percentage),
#                        sorted ascending.  Useful for inspecting why
#                        calibration chose the bounds it did and for
#                        identifying items that are too rare to form rules.
#
#   calibration_log.txt — human-readable summary of the calibration mode
#                         (auto vs manual) and the resulting parameter values
#                         with their derivation rationale.
#
# Both files are written before the grid search begins so they are available
# for inspection even if the search is interrupted.
# ---------------------------------------------------------------------------

def _write_calibration_log(
    k_dir: Path,
    k: int,
    n_transactions: int,
    item_supports: pd.Series,
    params: dict | None,
    auto_calibrate: bool,
    active_labels: set[str],
    manual_params: dict | None = None,
) -> None:
    """
    Write item_supports.csv and calibration_log.txt to *k_dir*.

    Parameters
    ----------
    k_dir : Path
        Output directory for this k value (k_{k}/).  Created if absent.
    k : int
        CF neighbourhood size (used only in the log header).
    n_transactions : int
        Number of transactions (samples) in the encoded DataFrame.
    item_supports : pd.Series
        Per-item support frequencies (column means of the encoded DataFrame),
        sorted ascending.  Written verbatim to item_supports.csv.
    params : dict or None
        Return value of calibrate_parameters().  None when auto-calibration
        found no 2-itemsets (too_sparse) or when auto_calibrate=False.
    auto_calibrate : bool
        Whether auto-calibration was attempted.  Controls the log mode header.
    active_labels : set[str]
        Labels used to filter transactions (written for traceability).
    manual_params : dict or None
        Manual grid bounds (sup_min, sup_max, conf_min, lift_max) written
        when auto_calibrate=False.  Ignored otherwise.
    """
    k_dir = Path(k_dir)
    k_dir.mkdir(parents=True, exist_ok=True)

    sup_df = pd.DataFrame({
        'item':        list(item_supports.index),
        'support_raw': [f'{v:.4f}' for v in item_supports.values],
        'support_pct': [f'{v * 100:.2f}' for v in item_supports.values],
    })
    sup_df.to_csv(k_dir / 'item_supports.csv', index=False)

    with open(k_dir / 'calibration_log.txt', 'w') as f:
        f.write(f'CALIBRATION LOG (value level) — k={k}\n')
        f.write(f'{"=" * 60}\n\n')
        f.write(f'Transactions : {n_transactions:,}\n')
        f.write(f'Items        : {len(item_supports)}\n')
        f.write(f'Active labels: {sorted(active_labels)}\n\n')
        f.write(f'Item Supports (sorted ascending):\n{"-" * 40}\n')
        for item, sup in item_supports.items():
            f.write(f'  {item:<35} {sup:.4f}\n')
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
            f.write('Reason: no 2-itemsets found — transactions too sparse.\n')
            return

        f.write('Calibrated Parameters:\n')
        f.write(f'  sup_min  : {params["sup_min"]}  (raw={params["raw_sup_min"]:.6f})\n')
        f.write(f'  sup_max  : {params["sup_max"]}  (natural ceiling={params["natural_sup_max"]}, +4 decay steps)\n')
        f.write(f'  conf_min : {params["conf_min"]}  (min observed={params["min_conf_observed"]}, max={params["max_conf_observed"]})\n')
        f.write(f'  conf_max : {params["conf_max"]}\n')
        f.write(f'  lift_min : {params["lift_min"]}  (0.0 — negative correlations included)\n')
        f.write(f'  lift_max : {params["lift_max"]}  (raw ceiling={params["raw_lift_max"]:.4f}, capped at 10.0)\n')


# ---------------------------------------------------------------------------
# Full grid exploration
#
# explore_association_rules_values() orchestrates the inner parallel loop:
# it expands the three calibrated (or manual) grids into a Cartesian product,
# dispatches one joblib worker per support value, then aggregates the results
# into summary.csv and exploration_summary.txt.
#
# The three grids are:
#   support    : [sup_min ... sup_max] with step sup_delta
#   confidence : [conf_min ... conf_max] with step conf_delta
#   lift       : [lift_min ... lift_max] with step lift_delta,
#                minus the neutral window [1-half_window, 1+half_window]
#
# The lift grid is pre-filtered before dispatch so workers do not need to
# re-apply the neutral-window exclusion independently.
#
# After all workers complete, empty sup_*/conf_*/ directories are pruned
# and three heatmap PNGs are generated for visual exploration.
# ---------------------------------------------------------------------------

def explore_association_rules_values(
    df,
    output_dir: Path,
    sup_min, sup_max, sup_delta,
    conf_min, conf_max, conf_delta,
    lift_min, lift_max, lift_delta,
    lift_neutral_half_window: float = 0.25,
    inner_n_jobs: int | None        = None,
) -> pd.DataFrame:
    """
    Run the full support × confidence × lift grid search at the value level.

    Parallelises over support thresholds (inner level); confidence and lift
    are processed serially within each worker to avoid redundant FP-Growth
    calls (the frequent itemsets at a given support are computed once and
    reused for all confidence values).

    Parameters
    ----------
    df : pd.DataFrame
        Boolean-encoded transaction DataFrame (output of encode_transactions()).
    output_dir : Path
        k_{k}/ directory under which all output subdirectories are created.
    sup_min, sup_max, sup_delta : float
        Support grid bounds and step size.
    conf_min, conf_max, conf_delta : float
        Confidence grid bounds and step size.
    lift_min, lift_max, lift_delta : float
        Lift grid bounds and step size.  The neutral window is excluded
        internally; lift_min is typically 0.0 to include negative correlations.
    lift_neutral_half_window : float
        Half-width of the neutral lift exclusion band (default 0.25).
    inner_n_jobs : int or None
        Number of loky workers for the support-level parallel loop.
        When None (default) the function derives it as
        max(1, perf_cores // outer_k_jobs) to respect the two-level
        parallelism budget.  Pass an explicit value to override this.

    Returns
    -------
    pd.DataFrame
        summary.csv content: one row per (support, confidence, lift) triple,
        sorted by Number_of_Rules descending then Max_Lift descending.
        Returns an empty DataFrame if no rules were found across the full grid.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    support_grid    = np.round(np.arange(sup_min, sup_max + sup_delta / 2,  sup_delta),  4)
    confidence_grid = np.round(np.arange(conf_min, conf_max + conf_delta / 2, conf_delta), 4)
    lift_grid       = np.round(np.arange(lift_min, lift_max + lift_delta / 2,  lift_delta), 4)

    lift_window_lo, lift_window_hi = _neutral_window(lift_neutral_half_window)
    lift_grid_used = [v for v in lift_grid if not (lift_window_lo <= v <= lift_window_hi)]
    total_combos   = len(support_grid) * len(confidence_grid) * len(lift_grid_used)

    print(f'\n{"=" * 70}')
    print('FULL EXPLORATION: FP-GROWTH ASSOCIATION RULES (VALUE LEVEL)')
    print(f'{"=" * 70}')
    print(f'  > support    : {len(support_grid)} values [{support_grid[0]} ... {support_grid[-1]}, step={sup_delta}]')
    print(f'  > confidence : {len(confidence_grid)} values [{confidence_grid[0]} ... {confidence_grid[-1]}, step={conf_delta}]')
    print(
        f'  > lift       : {len(lift_grid_used)} values used '
        f'(of {len(lift_grid)} total, neutral window [{lift_window_lo}, {lift_window_hi}], step={lift_delta})'
    )
    print(f'  > total combinations: {total_combos:,}')
    print('-' * 50)

    perf_cores = _get_perf_cores()
    # Respect the two-level parallelism budget:
    #   outer level (k values)   : outer_jobs workers, each calling this function
    #   inner level (sup values) : inner_n_jobs workers per outer worker
    #
    # When inner_n_jobs is provided by the caller (run_k_comparison_values),
    # it has already been set to max(1, perf_cores // outer_jobs) so that
    # outer × inner ≤ perf_cores.  When called standalone (inner_n_jobs=None),
    # we use all available P-cores — safe because there is no outer loop.
    if inner_n_jobs is None:
        inner_n_jobs = min(perf_cores, len(support_grid))
    n_jobs = min(inner_n_jobs, len(support_grid))
    print(
        f'  > Launching parallel FP-Growth over {len(support_grid)} support values '
        f'(n_jobs={n_jobs} of {_CPU_CORES} logical / {perf_cores} perf cores)'
    )

    parallel_results = Parallel(
        n_jobs=n_jobs, backend='loky', verbose=0, pre_dispatch=n_jobs,
    )(
        delayed(_process_one_support)(
            min_sup         = min_sup,
            sup_idx         = sup_idx,
            n_sup           = len(support_grid),
            df              = df,
            output_dir      = output_dir,
            confidence_grid = confidence_grid,
            lift_grid_used  = lift_grid_used,
            lift_window_lo  = lift_window_lo,
            lift_window_hi  = lift_window_hi,
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
    combos_with_rules = int((summary_df['Number_of_Rules'] > 0).sum()) if not summary_df.empty else 0

    with open(output_dir / 'exploration_summary.txt', 'w') as f:
        f.write('FULL EXPLORATION SUMMARY (value level)\n')
        f.write(f'{"=" * 70}\n\n')
        f.write(f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write('Parameter Grids:\n')
        f.write(f'  Support    : {len(support_grid)} values [{support_grid[0]} ... {support_grid[-1]}, step={sup_delta}]\n')
        f.write(f'  Confidence : {len(confidence_grid)} values [{confidence_grid[0]} ... {confidence_grid[-1]}, step={conf_delta}]\n')
        f.write(
            f'  Lift       : {len(lift_grid_used)} values used '
            f'(neutral window [{lift_window_lo}, {lift_window_hi}] excluded), step={lift_delta}\n\n'
        )
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
        f.write(f'  n_jobs used: {n_jobs} (of {_CPU_CORES} logical / {perf_cores} perf cores)\n\n')
        f.write('Folder cleanup:\n')
        f.write(f'  conf dirs removed: {removed_conf}\n')
        f.write(f'  sup dirs removed : {removed_sup}\n')

    plot_heatmaps(
        summary_df, output_dir,
        lift_neutral_half_window=lift_neutral_half_window,
        lift_delta=lift_delta,
    )

    print(f'  > summary saved to {output_dir / "summary.csv"}')
    print(f'  > total combinations: {total_combos:,}  |  with rules: {combos_with_rules:,}')

    return summary_df


# ---------------------------------------------------------------------------
# Per-k worker (outer parallel level)
#
# _process_one_k() is the outer unit of work, dispatched by
# run_k_comparison_values() via joblib.Parallel, one invocation per k value.
# Each worker is fully self-contained: it loads its own data, calibrates its
# own parameters, and writes its own output directory (k_{k}/).  Workers
# operate on different k values and therefore on different input files and
# output directories, so there are no race conditions.
#
# The function follows a six-step internal pipeline:
#   A. Collect active labels from Step 3 rules for this (region, k).
#   B. Verify that transactions_values.csv exists for this (region, k).
#   C. Build value-level transaction DataFrames (pair + aggregated).
#   D. One-hot-encode the aggregated transactions.
#   E. Calibrate grid parameters (auto) or apply manual bounds.
#   F. Run the full grid search via explore_association_rules_values().
#
# Skipping behaviour: if any prerequisite is missing (no rules, no
# transactions file, empty transactions after filtering, or too sparse for
# 2-itemsets), the worker returns (k, None, skip_row) so that the outer
# loop can record the skip reason without crashing the entire pipeline.
# ---------------------------------------------------------------------------

def _process_one_k(
    k: int,
    results_dir: Path,
    region: str,
    output_dir: Path,
    auto_calibrate: bool,
    sup_min: float, sup_max: float, sup_delta: float,
    conf_min: float, conf_max: float, conf_delta: float,
    lift_min: float, lift_max: float, lift_delta: float,
    lift_neutral_half_window: float,
    inner_n_jobs: int = 1,
) -> tuple[int, pd.DataFrame | None, dict]:
    """
    End-to-end value-level ARM pipeline for a single (region, k) pair.

    Designed to be called as a joblib.Parallel worker.  All inputs are
    passed explicitly; no shared mutable state is accessed.

    Parameters
    ----------
    k : int
        CF neighbourhood size being processed.
    results_dir : Path
        Root results directory (base_dir / 'results').
    region : str
        Region name (e.g. 'northeast').
    output_dir : Path
        Region-level output directory (results/{region}/association_rules_values/).
        k_{k}/ is created as a subdirectory of this path.
    auto_calibrate : bool
        Whether to derive grid bounds from data (True) or use the supplied
        manual bounds (False).
    sup_min … lift_neutral_half_window : float
        Grid parameters — used directly when auto_calibrate=False, or as
        floor/ceiling constraints when auto_calibrate=True.
    inner_n_jobs : int
        Number of loky workers for the inner support-level parallel loop
        inside explore_association_rules_values().  Computed by
        run_k_comparison_values() as max(1, perf_cores // outer_jobs) so
        that total worker count stays within the P-core budget.

    Returns
    -------
    tuple[int, pd.DataFrame | None, dict]
        (k, summary_df, comparison_row) where:
        - summary_df is the output of explore_association_rules_values(), or
          None if the worker was skipped.
        - comparison_row is a dict with keys matching k_comparison_summary.csv
          columns, including a 'skipped_reason' field (empty string if the
          worker ran successfully).
    """
    k_dir = output_dir / f'k_{k}'
    k_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"=" * 70}')
    print(f'  k = {k}')
    print(f'{"=" * 70}')

    # ── Step A: collect active labels from all Step 3 rules for this k ──
    # Scan every experiment-label subdirectory so that multiple Step 3 runs
    # with different parameters all contribute to the active label set.
    rules_csvs    = find_rules_csvs(results_dir, region, k)
    active_labels = collect_active_labels(rules_csvs)

    if not active_labels:
        print(
            f'  > No rules.csv found for k={k} under '
            f'{results_dir / region / "association_rules"}\n'
            f'    Run macroscopic_experiment_association_rules.py first.'
        )
        row = {
            'k': k, 'skipped_reason': 'no_label_rules',
            'n_transactions': None, 'n_items': None,
            'active_labels': None,
            'sup_min_used': None, 'sup_max_used': None,
            'lift_max_used': None, 'conf_min_used': None,
            'summary_rows': 0, 'combos_with_rules': 0,
            'max_rules_any_combo': 0, 'avg_lift_best_combo': None,
            'max_lift_observed': None,
        }
        return k, None, row

    print(f'  > Active labels from rules: {sorted(active_labels)}')
    print(f'  > Rules files found: {len(rules_csvs)}')

    # ── Step B: verify transactions_values.csv exists for this (region, k) ──
    # This file is produced by Step 2 (feature_importance.py).  If it is
    # absent the worker cannot proceed and records the skip reason.
    tv_path = results_dir / region / 'important_features' / f'k_{k}' / 'transactions_values.csv'
    if not tv_path.exists():
        print(f'  > transactions_values.csv not found: {tv_path}')
        row = {
            'k': k, 'skipped_reason': 'no_transactions_values',
            'n_transactions': None, 'n_items': None,
            'active_labels': str(sorted(active_labels)),
            'sup_min_used': None, 'sup_max_used': None,
            'lift_max_used': None, 'conf_min_used': None,
            'summary_rows': 0, 'combos_with_rules': 0,
            'max_rules_any_combo': 0, 'avg_lift_best_combo': None,
            'max_lift_observed': None,
        }
        return k, None, row

    # ── Step C: build and persist value-level transaction DataFrames ─────
    # Both formats are written to disk regardless of whether ARM succeeds,
    # so they remain available for manual inspection even on sparse datasets.
    agg_df, pair_df = build_value_transactions(tv_path, active_labels)

    # Persist both transaction formats to the k_{k}/ output directory.
    agg_df.to_csv(k_dir / 'aggregated_values_by_sample.csv',  index=False)
    pair_df.to_csv(k_dir / 'values_only_unique.csv',           index=False)
    print(f'  > Saved aggregated_values_by_sample.csv ({len(agg_df):,} rows)')
    print(f'  > Saved values_only_unique.csv ({len(pair_df):,} rows)')

    if agg_df.empty:
        print(f'  > No value transactions after filtering — skipping k={k}.')
        row = {
            'k': k, 'skipped_reason': 'empty_after_filter',
            'n_transactions': 0, 'n_items': 0,
            'active_labels': str(sorted(active_labels)),
            'sup_min_used': None, 'sup_max_used': None,
            'lift_max_used': None, 'conf_min_used': None,
            'summary_rows': 0, 'combos_with_rules': 0,
            'max_rules_any_combo': 0, 'avg_lift_best_combo': None,
            'max_lift_observed': None,
        }
        return k, None, row

    # ── Step D: one-hot-encode the aggregated transactions ────────────
    # FP-Growth runs exclusively on the aggregated format (one transaction
    # per sample).  The pair-level format is saved for traceability only.
    df_encoded    = encode_transactions(agg_df['Values'], col_name='aggregated_values')
    item_supports = df_encoded.mean().sort_values()

    # ── Step E: calibrate grid parameters or apply manual bounds ─────
    # In auto mode, calibrate_parameters() derives data-driven bounds from
    # the item frequency distribution and writes them to calibration_log.txt.
    # In manual mode, the user-supplied bounds are used directly and logged.
    if auto_calibrate:
        params = calibrate_parameters(
            encoded_df=df_encoded,
            sup_delta=sup_delta, lift_delta=lift_delta,
            conf_delta=conf_delta, conf_min_floor=conf_min, conf_max=conf_max,
        )
        _write_calibration_log(
            k_dir=k_dir, k=k,
            n_transactions=len(df_encoded),
            item_supports=item_supports,
            params=params,
            auto_calibrate=True,
            active_labels=active_labels,
        )

        if params is None:
            print(f'  > Skipping k={k} — insufficient co-occurrences for rules.')
            row = {
                'k': k, 'skipped_reason': 'too_sparse',
                'n_transactions': len(df_encoded),
                'n_items': df_encoded.shape[1],
                'active_labels': str(sorted(active_labels)),
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
        k_lift_min = params['lift_min']   # always 0.0 — negative correlations included

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
            active_labels=active_labels,
            manual_params={
                'sup_min': k_sup_min, 'sup_max': k_sup_max,
                'conf_min': k_conf_min, 'lift_max': k_lift_max,
            },
        )

    # ── Step F: run the full grid search ─────────────────────────────
    # Workers from the inner Parallel loop write to sub-paths of k_dir;
    # the summary DataFrame is returned for aggregation into k_comparison.
    summary_df = explore_association_rules_values(
        df=df_encoded, output_dir=k_dir,
        sup_min=k_sup_min,  sup_max=k_sup_max,  sup_delta=sup_delta,
        conf_min=k_conf_min, conf_max=conf_max, conf_delta=conf_delta,
        lift_min=k_lift_min, lift_max=k_lift_max, lift_delta=lift_delta,
        lift_neutral_half_window=lift_neutral_half_window,
        inner_n_jobs=inner_n_jobs,
    )

    has_rules_col = not summary_df.empty and 'Number_of_Rules' in summary_df.columns
    with_rules    = (
        summary_df[summary_df['Number_of_Rules'] > 0]
        if has_rules_col else pd.DataFrame()
    )
    row = {
        'k':                   k,
        'skipped_reason':      '',
        'n_transactions':      len(df_encoded),
        'n_items':             df_encoded.shape[1],
        'active_labels':       str(sorted(active_labels)),
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


# ---------------------------------------------------------------------------
# K-comparison (outer parallel level)
#
# run_k_comparison_values() is the top-level orchestrator for a single region.
# It dispatches one _process_one_k() worker per k value using joblib loky,
# with n_jobs capped at the P-core count to respect Apple Silicon memory
# constraints.  After all workers complete, it:
#
#   1. Aggregates per-k skip/result metadata into k_comparison_summary.csv
#      and k_comparison_summary.txt.
#   2. Concatenates all per-k summary DataFrames and produces three cross-k
#      heatmaps (k vs Support, k vs Confidence, k vs Lift) that let the
#      analyst compare which CF neighbourhood size yields the richest
#      value-level rule sets.
#
# The function returns a dict {k: summary_df} so that main() can compute
# region-level aggregates (total rules, best k) for the final banner.
# ---------------------------------------------------------------------------

def run_k_comparison_values(
    k_values: list[int],
    results_dir: Path,
    region: str,
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
    Run value-level ARM for every k in *k_values* and produce a cross-k
    comparison report for a single region.

    Parameters
    ----------
    k_values : list[int]
        CF neighbourhood sizes to process.  Sorted internally before dispatch.
    results_dir : Path
        Root results directory (base_dir / 'results').
    region : str
        Region name (e.g. 'northeast').
    output_dir : Path
        Region-level output directory for association_rules_values/.
        k_{k}/ subdirectories are created here by each worker.
        k_comparison/ is created here for cross-k artefacts.
    auto_calibrate : bool
        Passed through to each _process_one_k() worker.
    sup_min … lift_neutral_half_window : float
        Grid parameters passed through to each worker.

    Returns
    -------
    dict[int, pd.DataFrame]
        Maps each k that successfully produced rules to its summary DataFrame.
        k values that were skipped are absent from the dict.
    """
    comp_dir = output_dir / 'k_comparison'
    comp_dir.mkdir(parents=True, exist_ok=True)

    k_sorted   = sorted(k_values)
    perf_cores = _get_perf_cores()
    outer_jobs = min(len(k_sorted), perf_cores)

    # Two-level parallelism budget for Apple Silicon (and any NUMA system):
    #
    #   outer_jobs  : number of k-level workers running concurrently.
    #   inner_n_jobs: number of support-level workers each k-worker spawns.
    #
    # Without budget splitting, outer × inner can reach perf_cores² processes
    # (e.g. 4 × 16 = 64 on M1 Ultra), all competing for the same 16 P-cores.
    # loky serialises each DataFrame into every worker; over-subscription
    # causes quadratic memory pressure and CPU starvation rather than speedup.
    #
    # Setting inner_n_jobs = max(1, perf_cores // outer_jobs) ensures:
    #   outer × inner ≤ perf_cores   (total workers ≤ available P-cores)
    #
    # Example: M1 Ultra, 16 P-cores, 4 k-values
    #   outer_jobs  = min(4, 16) = 4
    #   inner_n_jobs = max(1, 16 // 4) = 4
    #   total workers = 4 × 4 = 16  (≤ 16 P-cores — no over-subscription)
    inner_n_jobs = max(1, perf_cores // outer_jobs)

    print(f'\n{"=" * 70}')
    print(f'K-VARIATION EXPERIMENT (VALUE LEVEL) — {len(k_sorted)} k values')
    print(f'{"=" * 70}')
    print(f'  > k values      : {k_sorted}')
    print(f'  > auto_calibrate: {auto_calibrate}')
    print(f'  > outer n_jobs  : {outer_jobs}  (k-level workers)')
    print(f'  > inner n_jobs  : {inner_n_jobs}  (support-level workers per k, '
          f'total ≤ {outer_jobs * inner_n_jobs} / {perf_cores} P-cores)')
    print('-' * 50)

    raw_results = Parallel(n_jobs=outer_jobs, backend='loky', verbose=0)(
        delayed(_process_one_k)(
            k=k,
            results_dir=results_dir,
            region=region,
            output_dir=output_dir,
            auto_calibrate=auto_calibrate,
            sup_min=sup_min,   sup_max=sup_max,   sup_delta=sup_delta,
            conf_min=conf_min, conf_max=conf_max, conf_delta=conf_delta,
            lift_min=lift_min, lift_max=lift_max, lift_delta=lift_delta,
            lift_neutral_half_window=lift_neutral_half_window,
            inner_n_jobs=inner_n_jobs,
        )
        for k in k_sorted
    )

    k_summaries     = {}
    comparison_rows = []
    for k, summary_df, row in raw_results:
        comparison_rows.append(row)
        if summary_df is not None:
            k_summaries[k] = summary_df

    # ── Cross-k comparison report (CSV + txt) ──────────────────────────
    # Aggregates per-k metadata (skipped/processed, grid params used, rule
    # counts and lift statistics) into a single CSV for downstream analysis.
    print(f'\n{"=" * 70}')
    print('  > Building cross-k comparison...')

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(comp_dir / 'k_comparison_summary.csv', index=False)

    with open(comp_dir / 'k_comparison_summary.txt', 'w') as f:
        f.write('K-VARIATION EXPERIMENT SUMMARY (value level)\n')
        f.write(f'{"=" * 70}\n\n')
        f.write(f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write(f'k values tested : {k_sorted}\n')
        f.write(f'auto_calibrate  : {auto_calibrate}\n\n')

        skipped = comp_df[comp_df['skipped_reason'] != ''] if not comp_df.empty else pd.DataFrame()
        ran     = comp_df[comp_df['skipped_reason'] == ''] if not comp_df.empty else pd.DataFrame()

        if not skipped.empty:
            f.write(f'Skipped k values ({len(skipped)}):\n{"-" * 40}\n')
            for _, row in skipped.iterrows():
                reason = row['skipped_reason']
                details = {
                    'no_label_rules':       'no rules.csv found for this k (run Step 3 first)',
                    'no_transactions_values': 'transactions_values.csv not found (run Step 2 first)',
                    'empty_after_filter':   'no value transactions after label filtering',
                    'too_sparse':           f'no 2-itemsets found (n_transactions={row["n_transactions"]})',
                }
                f.write(f'  k={int(row["k"])}: {details.get(reason, reason)}\n')
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
                f'({int(ran["max_rules_any_combo"].max())} rules at best combination)\n'
            )
        else:
            f.write('No rules found for any k value.\n')

    print('  > Saved k_comparison_summary.csv and .txt')

    # ── Cross-k heatmaps ──────────────────────────────────────────────
    # Concatenate per-k summaries and produce three heatmaps (k vs Support,
    # k vs Confidence, k vs Lift) to visualise how the neighbourhood size k
    # affects the richness of value-level rules across the parameter space.
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

        if is_lift:
            pivot = pivot.loc[
                :, ~pivot.columns.to_series().between(neutral_lo, neutral_hi, inclusive='both')
            ]
            if (pivot != 0).any(axis=0).any():
                last_nz = int(np.where((pivot != 0).any(axis=0).values)[0].max())
                pivot   = pivot.iloc[:, :last_nz + 1]

        n_cols = len(pivot.columns)
        n_rows = len(pivot.index)
        fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.75), max(4, n_rows * 0.6)))

        img = ax.imshow(pivot.values, aspect='auto', cmap='YlOrBr', interpolation='nearest')
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(
            [f'{v:.2f}' if isinstance(v, float) else str(v) for v in pivot.columns],
            rotation=40, ha='right', fontsize=8,
        )
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f'k={v}' for v in pivot.index], fontsize=9)
        ax.set_xlabel(x_label, fontsize=11, labelpad=8)
        ax.set_ylabel('k',     fontsize=11, labelpad=8)
        ax.set_title(
            f'Max Number of Rules (value level) — k vs {x_label}\n'
            f'(darker = more rules; max over the other two parameters)',
            fontsize=11, pad=14,
        )

        max_val = pivot.values.max() if pivot.values.max() > 0 else 1
        for ri in range(n_rows):
            for ci in range(n_cols):
                val = pivot.values[ri, ci]
                if val > 0:
                    txt_color = 'white' if (val / max_val) > 0.55 else 'black'
                    ax.text(ci, ri, str(val), ha='center', va='center',
                            fontsize=7, color=txt_color)

        cbar = plt.colorbar(img, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label('Number of Rules', fontsize=9)
        plt.tight_layout()
        fig.savefig(comp_dir / f'heatmap_{suffix}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'    > saved k_comparison/heatmap_{suffix}.png')

    print(f'  > Cross-k comparison saved to {comp_dir}/')
    return k_summaries


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    regions: list[str]               = None,
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
    Entry point for value-level association-rule mining across all regions
    and k values.

    Signature mirrors macroscopic_experiment_association_rules.main() so that
    main.py can invoke both Step 3 and Step 4 with identical parameter sets
    without any adapter code.

    Parameters
    ----------
    regions : list[str] or None
        Regions to process.  Defaults to ['northeast', 'south'] when None.
    k_values : list[int] or None
        CF neighbourhood sizes.  Defaults to [1, 3, 5, 7] when None.
    auto_calibrate : bool
        Whether to derive grid bounds from item frequencies (True, default)
        or use the supplied manual bounds directly (False).
    sup_min, sup_max, sup_delta : float
        Support grid bounds and step.  Used as manual bounds when
        auto_calibrate=False; as calibration floor/ceiling otherwise.
    conf_min, conf_max, conf_delta : float
        Confidence grid bounds and step.
    lift_min, lift_max, lift_delta : float
        Lift grid bounds and step.  lift_min=0.0 includes negative correlations.
    lift_neutral_half_window : float
        Half-width of the neutral lift exclusion window (default 0.25).
    base_dir : Path or None
        Project root directory.  results/ and data/ are resolved relative to
        this path.  Auto-detected from the execution environment when None:
        Kaggle → /kaggle/working, Colab → /content, otherwise the parent
        directory of this script file.
    """
    print(
        f'  > Parallel backend — {_CPU_CORES} logical cores / '
        f'{_get_perf_cores()} perf cores (joblib loky)'
    )

    # ── Base directory resolution ─────────────────────────────────────
    # Supports three common execution environments without requiring the user
    # to pass base_dir explicitly when running in Kaggle or Colab notebooks.
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

    # Build the experiment label once — the same label is used for every
    # region so that a single pipeline run is always self-consistent.
    # The label encodes the full grid configuration, mirroring the macroscopic
    # script so both pipeline stages use a parallel directory structure:
    #   macroscopic : results/{region}/association_rules/{exp_label}/k_{k}/
    #   microscopic : results/{region}/association_rules_values/{exp_label}/k_{k}/
    exp_label = _experiment_label(
        auto_calibrate=auto_calibrate,
        sup_min=sup_min,   sup_max=sup_max,   sup_delta=sup_delta,
        conf_min=conf_min, conf_max=conf_max, conf_delta=conf_delta,
        lift_min=lift_min, lift_max=lift_max, lift_delta=lift_delta,
        lift_neutral_half_window=lift_neutral_half_window,
    )

    for region in regions:
        ar_output_dir = results_dir / region / 'association_rules_values' / exp_label
        ar_output_dir.mkdir(parents=True, exist_ok=True)

        print('\n' + '=' * 70)
        print(f'ASSOCIATION RULES (VALUE LEVEL) — {region.upper()}')
        print(f'Experiment: {exp_label}')
        print('=' * 70 + '\n')

        k_summaries = run_k_comparison_values(
            k_values=k_values,
            results_dir=results_dir,
            region=region,
            output_dir=ar_output_dir,
            auto_calibrate=auto_calibrate,
            sup_min=sup_min,   sup_max=sup_max,   sup_delta=sup_delta,
            conf_min=conf_min, conf_max=conf_max, conf_delta=conf_delta,
            lift_min=lift_min, lift_max=lift_max, lift_delta=lift_delta,
            lift_neutral_half_window=lift_neutral_half_window,
        )

        k_max_per_k = {
            k: int(sdf['Number_of_Rules'].max())
            for k, sdf in k_summaries.items()
            if not sdf.empty and 'Number_of_Rules' in sdf.columns
        }
        sum_rules = sum(k_max_per_k.values())
        max_rules = max(k_max_per_k.values()) if k_max_per_k else 0
        best_k    = max(k_max_per_k, key=k_max_per_k.get) if k_max_per_k else None

        print(f'\n  > Region {region.upper()} complete.')
        print(f'    Sum of max rules across k : {sum_rules}')
        print(f'    Max rules in best combo   : {max_rules}  (k={best_k})')

    print('\n' + '=' * 70)
    print('Done.')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
