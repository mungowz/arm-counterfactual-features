"""
microscopic_experiment_association_rules_values.py
==================================================
Runs FP-Growth association-rule mining at the **value level**, one
macroscopic rule at a time, starting from the label-level rules produced by
macroscopic_experiment_association_rules.py.

Pipeline position
-----------------
    macroscopic_experiment_association_rules.py  →  [this script]

Idea
----
Step 3 (ARM on labels) identifies which *features* (OCCP, SCHL, WKHP, …)
tend to co-occur on the decision boundary.  This script asks the finer
question: *which specific values* of those features drive each individual
label-level rule?

For every unique macroscopic rule r = (antecedents_r, consequents_r) found
across all sup_*/conf_*/rules.csv files produced by Step 3, this script:

1. Derives active_labels_r = labels(antecedents_r) ∪ labels(consequents_r).
   This is the *minimal* label set for rule r.
2. From transactions_values.csv keeps only items whose label prefix is in
   active_labels_r.  Support is then computed over this filtered set:
       denominator = samples with ≥1 CF-change in an active label
   NOT the total number of samples in the region.  This is intentional —
   we characterise patterns within the population that is sensitive to the
   specific features named in the parent macroscopic rule.
3. Builds two transaction formats:
       aggregated_values_by_sample.csv   — one row per sample (union of all
           CF-neighbour items, deduplicated)
       values_only_unique.csv            — one row per (sample, CF_neighbor)
4. Runs the same support × confidence × lift grid search as Step 3,
   with auto-calibration derived from the filtered item frequencies.
5. Writes all artefacts under a dedicated rule subdirectory:
       results/{region}/association_rules_values/{exp_label}/k_{k}/
           rule_{i:03d}__{ant}___{cons}/
               macro_rule_origin.csv     ← parent macroscopic rule (machine)
               macro_rule_origin.txt     ← parent macroscopic rule (human)
               aggregated_values_by_sample.csv
               values_only_unique.csv
               calibration_log.txt
               item_supports.csv
               exploration_summary.txt
               summary.csv              ← includes macro_rule_* columns
               heatmaps/
               sup_{x}/conf_{y}/rules.csv           ← includes macro_rule_* columns
               sup_{x}/conf_{y}/rules_detailed.csv  ← includes macro_rule_* columns
               sup_{x}/conf_{y}/summary.txt         ← includes parent rule header
               sup_{x}/frequent_itemsets.csv
               sup_{x}/frequent_itemsets_summary.txt
           rule_index.csv               ← per-k table: rule_slug → outcome
       results/{region}/association_rules_values/{exp_label}/k_comparison/
           k_comparison_summary.csv
           k_comparison_summary.txt
           heatmap_k_support.png
           heatmap_k_confidence.png
           heatmap_k_lift.png

Macro-rule deduplication
------------------------
Multiple sup_*/conf_*/ directories may produce the same (antecedents,
consequents) pair.  This script deduplicates by (antecedents, consequents)
string so each unique rule is processed exactly once per k.

Parallelism — three levels
--------------------------
Level 1 (outer)  : k values in parallel.
                   outer_jobs = min(n_k, perf_cores)
Level 2 (middle) : macroscopic rules in parallel within each k-worker.
                   rule_jobs  = min(n_rules, max(1, perf_cores // outer_jobs))
Level 3 (inner)  : support thresholds in parallel within each rule-worker.
                   inner_jobs = max(1, perf_cores // (outer_jobs × rule_jobs))
Guarantee: outer × rule × inner ≤ perf_cores at all times.

Input files (resolved relative to base_dir/results/)
-----------------------------------------------------
    {region}/important_features/k_{k}/transactions_values.csv
        Columns: Sample_ID, CF_Neighbor_ID, Counterfactual_Values

    {region}/association_rules/{exp_label}/k_{k}/sup_*/conf_*/rules.csv
        Produced by Step 3; antecedents/consequents are label names.

Public API
----------
main(regions, k_values, base_dir, ...)
    Entry point — mirrors macroscopic_experiment_association_rules.main().
"""

import ast
import datetime
import os
import platform
import re
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
# ---------------------------------------------------------------------------

_CPU_CORES: int         = os.cpu_count() or 1
_PERF_CORES: int | None = None


def _detect_perf_cores() -> int:
    """
    Return the number of P-cores on Apple Silicon, or total logical CPUs
    on all other platforms.
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
            pass
    return _CPU_CORES


def _get_perf_cores() -> int:
    """Return the cached P-core count, detecting it on the first call."""
    global _PERF_CORES
    if _PERF_CORES is None:
        _PERF_CORES = _detect_perf_cores()
    return _PERF_CORES


# ---------------------------------------------------------------------------
# Neutral-window helper
# ---------------------------------------------------------------------------

def _neutral_window(half_window: float) -> tuple[float, float]:
    """
    Return (lo, hi) for the symmetric neutral-lift exclusion band.

    With the default half_window=0.25 this gives [0.75, 1.25].
    Rules with lift in [lo, hi] are excluded from the grid search.
    Rules with lift < lo (negative correlations) are preserved.
    """
    lo = round(1.0 - half_window, 4)
    hi = round(1.0 + half_window, 4)
    return lo, hi


# ---------------------------------------------------------------------------
# Experiment labelling
# ---------------------------------------------------------------------------

def _experiment_label(
    auto_calibrate: bool,
    sup_min: float, sup_max: float, sup_delta: float,
    conf_min: float, conf_max: float, conf_delta: float,
    lift_min: float, lift_max: float, lift_delta: float,
    lift_neutral_half_window: float,
) -> str:
    """
    Build a filesystem-safe string encoding the grid configuration.

    Inserted as a path component between association_rules_values/ and k_{k}/
    so that different configurations coexist without overwriting each other.
    Mirrors the identical function in macroscopic_experiment_association_rules.
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
# Macroscopic rule utilities
# ---------------------------------------------------------------------------

def _parse_labels_from_cell(cell: str) -> set[str]:
    """
    Parse a comma-separated antecedents/consequents cell into a set of label
    names.  E.g. 'OCCP, SCHL' → {'OCCP', 'SCHL'}.
    """
    return {s.strip() for s in str(cell).split(',') if s.strip()}


def _make_rule_slug(rule_idx: int, ant_str: str, cons_str: str) -> str:
    """
    Build a filesystem-safe folder name for a single macroscopic rule.

    Format: rule_{idx:03d}__{ant_clean}___{cons_clean}

    Examples
    --------
    (0, 'SCHL', 'OCCP')        → 'rule_000__SCHL___OCCP'
    (1, 'OCCP, SCHL', 'WKHP') → 'rule_001__OCCP_SCHL___WKHP'
    """
    def clean(s: str) -> str:
        s = s.replace(', ', '_').replace(',', '_')
        s = re.sub(r'[^\w]', '_', s)
        s = re.sub(r'_+', '_', s).strip('_')
        return s

    return f'rule_{rule_idx:03d}__{clean(ant_str)}___{clean(cons_str)}'


def collect_unique_macro_rules(
    results_dir: Path,
    region: str,
    k: int,
) -> list[dict]:
    """
    Collect all unique macroscopic rules for a given (region, k) by scanning
    every sup_*/conf_*/rules.csv produced by Step 3 across all
    experiment-label subdirectories.

    Deduplication is by (antecedents, consequents) string pair.  When the same
    rule appears in multiple files all source paths are recorded; statistics
    (support, confidence, lift) come from the first occurrence (lexicographic
    path order).

    Returns
    -------
    list[dict]
        Sorted list of unique rule dicts.  Each dict contains:
            rule_index        int   — 0-based sequential index
            antecedents       str   — e.g. 'OCCP, SCHL'
            consequents       str   — e.g. 'WKHP'
            active_labels     set   — union of both sides
            rule_slug         str   — filesystem-safe folder name
            source_paths      list  — all rules.csv paths for this rule
            macro_support_pct float — support_pct from first occurrence
            macro_conf_pct    float — confidence_pct from first occurrence
            macro_lift        float — lift from first occurrence
        Returns [] if no rules.csv exist for this (region, k).
    """
    base = results_dir / region / 'association_rules'
    if not base.exists():
        return []

    all_paths = sorted(base.rglob(f'k_{k}/sup_*/conf_*/rules.csv'))
    if not all_paths:
        return []

    seen: dict[tuple, dict] = {}   # (ant_norm, cons_norm) → rule dict

    for p in all_paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        for _, row in df.iterrows():
            ant  = str(row.get('antecedents', '')).strip()
            cons = str(row.get('consequents', '')).strip()
            if not ant or not cons:
                continue
            key = (ant, cons)
            if key not in seen:
                seen[key] = {
                    'antecedents':       ant,
                    'consequents':       cons,
                    'active_labels':     (
                        _parse_labels_from_cell(ant) |
                        _parse_labels_from_cell(cons)
                    ),
                    'source_paths':      [str(p)],
                    'macro_support_pct': float(row.get('support_pct',   0.0)),
                    'macro_conf_pct':    float(row.get('confidence_pct', 0.0)),
                    'macro_lift':        float(row.get('lift',           0.0)),
                }
            else:
                seen[key]['source_paths'].append(str(p))

    unique_rules = []
    for idx, ((ant, cons), info) in enumerate(seen.items()):
        info['rule_index'] = idx
        info['rule_slug']  = _make_rule_slug(idx, ant, cons)
        unique_rules.append(info)

    return unique_rules


# ---------------------------------------------------------------------------
# Transaction builder
# ---------------------------------------------------------------------------

def build_value_transactions(
    transactions_values_path: Path,
    active_labels: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read transactions_values.csv and build aggregated and pair-level
    transaction DataFrames, keeping only items whose label prefix belongs to
    *active_labels*.

    The support denominator is the number of samples that have at least one
    CF-change in an active label (not the total population).  This is
    intentional: we characterise patterns within samples sensitive to the
    features named in the parent macroscopic rule.

    Returns
    -------
    agg_df : pd.DataFrame
        One row per Sample_ID.
        Columns: Sample_ID, Values (str repr of sorted list),
                 Num_Values, Num_CF_Neighbors.
        Passed to encode_transactions() for FP-Growth.
    pair_df : pd.DataFrame
        One row per (Sample_ID, CF_Neighbor_ID).
        Columns: Sample_ID, CF_Neighbor_ID, Values.
        Saved for traceability.
    """
    df = pd.read_csv(transactions_values_path)

    required = {'Sample_ID', 'CF_Neighbor_ID', 'Counterfactual_Values'}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f'transactions_values.csv missing columns: {missing}  '
            f'(found: {df.columns.tolist()})'
        )

    print(
        f'    Loaded {len(df):,} rows, '
        f'{df["Sample_ID"].nunique():,} unique samples'
    )

    def _filter_items(raw: str) -> list:
        try:
            items = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
        return [
            item for item in items
            if '=' in str(item)
            and str(item).split('=', 1)[0] in active_labels
        ]

    df          = df.copy()
    df['_filt'] = df['Counterfactual_Values'].apply(_filter_items)
    df          = df[df['_filt'].map(len) > 0].reset_index(drop=True)

    # ── Pair-level DataFrame ──────────────────────────────────────────
    pair_df = (
        df[['Sample_ID', 'CF_Neighbor_ID', '_filt']]
        .rename(columns={'_filt': 'Values'})
        .assign(Values=lambda d: d['Values'].apply(str))
        .reset_index(drop=True)
    )

    # ── Aggregated DataFrame ──────────────────────────────────────────
    agg_df = (
        df.groupby('Sample_ID', sort=False)
        .agg(
            _items=('_filt',
                    lambda x: sorted({item for sub in x for item in sub})),
            Num_CF_Neighbors=('CF_Neighbor_ID', 'nunique'),
        )
        .reset_index()
    )
    agg_df['Values']     = agg_df['_items'].apply(str)
    agg_df['Num_Values'] = agg_df['_items'].apply(len)
    agg_df = agg_df[['Sample_ID', 'Values', 'Num_Values', 'Num_CF_Neighbors']]

    print(
        f'    After filtering to {sorted(active_labels)}: '
        f'{len(agg_df):,} samples, {len(pair_df):,} (sample, CF) pairs'
    )
    return agg_df, pair_df


# ---------------------------------------------------------------------------
# One-hot encoding
# ---------------------------------------------------------------------------

def encode_transactions(values_col: pd.Series) -> pd.DataFrame:
    """
    One-hot-encode a Series of stringified Python lists into a Boolean
    DataFrame suitable for mlxtend's fpgrowth().
    """
    itemsets = values_col.apply(ast.literal_eval)
    te       = TransactionEncoder()
    te_ary   = te.fit(itemsets).transform(itemsets)
    enc_df   = pd.DataFrame(te_ary, columns=te.columns_)
    print(
        f'    Encoded: {len(enc_df):,} transactions × '
        f'{enc_df.shape[1]:,} unique items'
    )
    return enc_df


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def cleanup_empty_folders(output_dir: Path) -> tuple[int, int]:
    """
    Remove conf_* subdirs with no valid rules.csv, then remove empty sup_* dirs.

    Returns (n_conf_removed, n_sup_removed).
    """
    removed_conf = 0
    removed_sup  = 0
    for sup_dir in sorted(Path(output_dir).glob('sup_*')):
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
        if not list(sup_dir.glob('conf_*')):
            shutil.rmtree(sup_dir)
            removed_sup += 1
    return removed_conf, removed_sup


# ---------------------------------------------------------------------------
# Macro-rule provenance writers
# ---------------------------------------------------------------------------

def _write_macro_rule_origin(rule_dir: Path, macro_rule: dict) -> None:
    """
    Write macro_rule_origin.csv (one row, machine-readable) and
    macro_rule_origin.txt (human-readable) to *rule_dir*.

    These files record the parent macroscopic rule for every microscopic
    result directory, making rule_dir self-describing when processed in
    isolation.
    """
    origin_row = {
        'rule_index':        macro_rule['rule_index'],
        'rule_slug':         macro_rule['rule_slug'],
        'antecedents':       macro_rule['antecedents'],
        'consequents':       macro_rule['consequents'],
        'active_labels':     ', '.join(sorted(macro_rule['active_labels'])),
        'macro_support_pct': macro_rule.get('macro_support_pct', ''),
        'macro_conf_pct':    macro_rule.get('macro_conf_pct', ''),
        'macro_lift':        macro_rule.get('macro_lift', ''),
        'n_source_files':    len(macro_rule.get('source_paths', [])),
        'source_paths':      ' | '.join(macro_rule.get('source_paths', [])),
    }
    pd.DataFrame([origin_row]).to_csv(
        rule_dir / 'macro_rule_origin.csv', index=False
    )

    with open(rule_dir / 'macro_rule_origin.txt', 'w') as f:
        f.write('PARENT MACROSCOPIC RULE\n')
        f.write(f'{"=" * 60}\n\n')
        f.write(f'Rule Index   : {macro_rule["rule_index"]}\n')
        f.write(f'Rule Slug    : {macro_rule["rule_slug"]}\n')
        f.write(f'Antecedents  : {macro_rule["antecedents"]}\n')
        f.write(f'Consequents  : {macro_rule["consequents"]}\n')
        f.write(f'Active Labels: {sorted(macro_rule["active_labels"])}\n\n')
        f.write('Macroscopic Statistics (from first occurrence):\n')
        f.write(f'  Support    : {macro_rule.get("macro_support_pct", 0.0):.2f}%\n')
        f.write(f'  Confidence : {macro_rule.get("macro_conf_pct", 0.0):.2f}%\n')
        f.write(f'  Lift       : {macro_rule.get("macro_lift", 0.0):.4f}\n\n')
        f.write(
            f'Found in {len(macro_rule.get("source_paths", []))} '
            f'rules.csv file(s):\n'
        )
        for p in macro_rule.get('source_paths', []):
            f.write(f'  {p}\n')


# ---------------------------------------------------------------------------
# Calibration log writer
# ---------------------------------------------------------------------------

def _write_calibration_log(
    rule_dir: Path,
    macro_rule: dict,
    n_transactions: int,
    item_supports: pd.Series,
    params: dict | None,
    auto_calibrate: bool,
    manual_params: dict | None = None,
) -> None:
    """
    Write item_supports.csv and calibration_log.txt to *rule_dir*.

    Both files include a header identifying the parent macroscopic rule.
    """
    sup_df = pd.DataFrame({
        'item':        list(item_supports.index),
        'support_raw': [f'{v:.4f}' for v in item_supports.values],
        'support_pct': [f'{v * 100:.2f}' for v in item_supports.values],
    })
    sup_df.to_csv(rule_dir / 'item_supports.csv', index=False)

    m_idx  = macro_rule['rule_index']
    m_ant  = macro_rule['antecedents']
    m_cons = macro_rule['consequents']

    with open(rule_dir / 'calibration_log.txt', 'w') as f:
        f.write('CALIBRATION LOG (value level)\n')
        f.write(f'{"=" * 60}\n\n')
        f.write('Parent Macroscopic Rule:\n')
        f.write(f'  Rule Index  : {m_idx}\n')
        f.write(f'  Antecedents : {m_ant}\n')
        f.write(f'  Consequents : {m_cons}\n\n')
        f.write(f'Transactions : {n_transactions:,}\n')
        f.write(f'Items        : {len(item_supports)}\n')
        f.write(f'Active Labels: {sorted(macro_rule["active_labels"])}\n\n')
        f.write(f'Item Supports (ascending):\n{"-" * 40}\n')
        for item, sup in item_supports.items():
            f.write(f'  {item:<45} {sup:.4f}\n')
        f.write('\n')

        if not auto_calibrate:
            f.write('Mode: MANUAL (auto_calibrate=False)\n')
            if manual_params:
                for k2, v in manual_params.items():
                    f.write(f'  {k2:<10}: {v}\n')
            return

        f.write('Mode: AUTO-CALIBRATED\n\n')
        if params is None:
            f.write('Result: SKIPPED\n')
            f.write('Reason: no 2-itemsets found — transactions too sparse.\n')
            return

        f.write('Calibrated Parameters:\n')
        f.write(
            f'  sup_min  : {params["sup_min"]}  '
            f'(raw={params["raw_sup_min"]:.6f})\n'
        )
        f.write(
            f'  sup_max  : {params["sup_max"]}  '
            f'(natural_ceiling={params["natural_sup_max"]}, +4 decay steps)\n'
        )
        f.write(
            f'  conf_min : {params["conf_min"]}  '
            f'(min_observed={params["min_conf_observed"]}, '
            f'max={params["max_conf_observed"]})\n'
        )
        f.write(f'  conf_max : {params["conf_max"]}\n')
        f.write(
            f'  lift_min : {params["lift_min"]}  '
            f'(0.0 — negative correlations included)\n'
        )
        f.write(
            f'  lift_max : {params["lift_max"]}  '
            f'(raw ceiling={params["raw_lift_max"]:.4f}, capped at 10.0)\n'
        )


# ---------------------------------------------------------------------------
# Auto-calibration
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
    Derive data-driven grid bounds from item frequencies.

    Strategy
    --------
    sup_min  — product of the two rarest item supports, snapped up to the
               nearest sup_delta; floored at sup_delta.
    sup_max  — support at which 2-itemsets stop appearing, plus 4 decay
               steps; capped at 0.50.
    lift_max — 1 / support(rarest item), rounded up to nearest 0.5; capped
               at 10.0.
    conf_min — minimum confidence in a probe run at sup_min, snapped down
               to the nearest conf_delta; floored at conf_min_floor.

    Returns None if no 2-itemsets can be formed.
    """
    print('    Calibrating parameters from item frequencies...')

    item_supports = encoded_df.mean().sort_values()
    if len(item_supports) < 2:
        print('    WARNING: fewer than 2 items — cannot form pairwise rules.')
        return None

    rarest = item_supports.iloc[0]
    second = item_supports.iloc[1]
    freq_2 = item_supports.iloc[-2]

    raw_sup_min = rarest * second
    sup_min = max(
        round(np.floor(raw_sup_min / sup_delta) * sup_delta, 4), sup_delta
    )

    _DECAY_STEPS            = 4
    scan_grid               = np.round(
        np.arange(sup_min, freq_2 + sup_delta / 2, sup_delta), 4
    )
    natural_sup_max         = sup_min
    prev_had_2itemsets      = False
    fi_first_with_2itemsets = None

    for t in scan_grid:
        fi = fpgrowth(encoded_df, min_support=t, use_colnames=True)
        if fi.empty:
            break
        has_2 = (fi['itemsets'].apply(len) >= 2).any()
        if has_2:
            if fi_first_with_2itemsets is None:
                fi_first_with_2itemsets = fi
            natural_sup_max    = t
            prev_had_2itemsets = True
        elif prev_had_2itemsets:
            break

    sup_max = min(
        round(natural_sup_max + _DECAY_STEPS * sup_delta, 4), 0.50
    )

    if not prev_had_2itemsets:
        print('    WARNING: no 2-itemsets found — transactions too sparse.')
        return None

    raw_lift_max = 1.0 / rarest
    lift_max     = min(round(np.ceil(raw_lift_max * 2) / 2, 1), 10.0)

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
        print(
            f'    WARNING: conf_min calibration failed ({exc!r}) '
            f'— using floor={conf_min_floor}'
        )

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
        'min_conf_observed': (
            round(min_conf_observed, 4) if min_conf_observed is not None else None
        ),
        'max_conf_observed': (
            round(max_conf_observed, 4) if max_conf_observed is not None else None
        ),
        'item_supports': item_supports.round(4).to_dict(),
    }


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

def plot_heatmaps(
    summary_df: pd.DataFrame,
    output_dir: Path,
    macro_rule: dict,
    lift_neutral_half_window: float = 0.25,
    lift_delta: float               = 0.05,
    lift_display_step: float        = 0.1,
) -> None:
    """
    Generate three parameter-space heatmaps for a single rule_dir.

    Each heatmap shows the maximum number of microscopic rules over the full
    support × confidence × lift grid, marginalising over the third parameter.
    The parent macroscopic rule is printed in the title.
    """
    if summary_df.empty:
        print('    > Summary empty — skipping heatmaps.')
        return

    output_dir  = Path(output_dir)
    heatmap_dir = output_dir / 'heatmaps'
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    df = summary_df.copy()
    df['Lift_display'] = (
        (df['Lift_threshold'] / lift_display_step).round() * lift_display_step
    ).round(4)

    neutral_lo, neutral_hi = _neutral_window(lift_neutral_half_window)
    rule_title = (
        f'[{macro_rule["antecedents"]}] => [{macro_rule["consequents"]}]'
    )

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
                :,
                ~pivot.columns.to_series().between(
                    neutral_lo, neutral_hi, inclusive='both'
                ),
            ]
            nz_mask = (pivot != 0).any(axis=0).values
            if nz_mask.any():
                last_nz = int(np.where(nz_mask)[0].max())
                pivot   = pivot.iloc[:, :last_nz + 1]

        n_cols = len(pivot.columns)
        n_rows = len(pivot.index)
        fig, ax = plt.subplots(
            figsize=(max(10, n_cols * 0.75), max(4, n_rows * 0.55))
        )

        img = ax.imshow(
            pivot.values, aspect='auto', cmap='YlOrBr', interpolation='nearest'
        )
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(
            [f'{v:.2f}' if isinstance(v, float) else str(v)
             for v in pivot.columns],
            rotation=40, ha='right', fontsize=8,
        )
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f'{v:.2f}' for v in pivot.index], fontsize=8)

        x_label = 'Lift' if x_is_lift else x_col
        ax.set_xlabel(x_label, fontsize=11, labelpad=8)
        ax.set_ylabel(y_col,   fontsize=11, labelpad=8)
        ax.set_title(
            f'Micro-rules — {y_col} vs {x_label}\n'
            f'Macro: {rule_title}\n'
            f'(darker = more rules; max over the third parameter)',
            fontsize=10, pad=14,
        )

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
        fig.savefig(
            heatmap_dir / f'heatmap_{suffix}.png', dpi=150, bbox_inches='tight'
        )
        plt.close(fig)
        print(f'      > saved heatmaps/heatmap_{suffix}.png')


# ---------------------------------------------------------------------------
# Inner worker — one support threshold
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
    macro_rule: dict,
) -> list:
    """
    Run FP-Growth at *min_sup* and write association rules for every
    (confidence, lift) combination.  Called as a joblib.Parallel worker.

    All output files include three provenance columns that trace each
    microscopic rule back to its parent macroscopic rule:
        macro_rule_index    — sequential index of the macroscopic rule
        macro_antecedents   — antecedent label(s) of the macroscopic rule
        macro_consequents   — consequent label(s) of the macroscopic rule

    Returns
    -------
    list[dict]
        One summary-row dict per (support, confidence, lift) triple that
        produced rules.  Empty list if FP-Growth finds no frequent itemsets.
    """
    output_dir = Path(output_dir)
    sup_label  = f'{min_sup:.2f}'
    sup_dir    = output_dir / f'sup_{sup_label}'
    sup_dir.mkdir(parents=True, exist_ok=True)

    m_idx  = macro_rule['rule_index']
    m_ant  = macro_rule['antecedents']
    m_cons = macro_rule['consequents']

    print(f'\n      [{sup_idx}/{n_sup}] sup={min_sup}')

    frequent_itemsets = fpgrowth(df, min_support=min_sup, use_colnames=True)
    n_fi = len(frequent_itemsets)
    print(f'        > {n_fi} frequent itemsets')

    if n_fi == 0:
        return []

    fi = frequent_itemsets.copy()
    fi['itemset_str']    = fi['itemsets'].apply(lambda x: ', '.join(sorted(x)))
    fi['itemset_length'] = fi['itemsets'].apply(len)
    fi = fi[['itemset_str', 'itemset_length', 'support']]
    fi = fi.sort_values(
        by=['itemset_length', 'support'], ascending=[True, False]
    )
    fi.to_csv(sup_dir / 'frequent_itemsets.csv', index=False)

    itemsets_by_len = fi['itemset_length'].value_counts().sort_index().to_dict()

    with open(sup_dir / 'frequent_itemsets_summary.txt', 'w') as f:
        f.write('Frequent Itemsets Summary (value level)\n')
        f.write(f'{"=" * 60}\n\n')
        f.write('Parent Macroscopic Rule:\n')
        f.write(f'  [{m_ant}] => [{m_cons}]  (rule_index={m_idx})\n\n')
        f.write(f'Parameters:\n  Min Support: {min_sup}\n\n')
        f.write(f'Results:\n  Total: {n_fi}\n')
        for length, count in itemsets_by_len.items():
            f.write(f'  len={length}: {count}\n')
        f.write(f'\nAll Frequent Itemsets:\n{"-" * 60}\n')
        for _, row in fi.iterrows():
            f.write(
                f'  [{row["itemset_str"]}]  '
                f'support={row["support"]:.4f}  '
                f'length={row["itemset_length"]}\n'
            )

    local_summary_rows = []

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

    all_rules = all_rules[
        (all_rules['lift'] < lift_window_lo) |
        (all_rules['lift'] > lift_window_hi)
    ]
    all_rules = all_rules.sort_values(
        'lift', ascending=False
    ).reset_index(drop=True)

    if len(all_rules) == 0:
        return local_summary_rows

    for min_conf in confidence_grid:
        conf_dir = sup_dir / f'conf_{min_conf:.2f}'

        rules = all_rules[
            all_rules['confidence'] >= min_conf
        ].reset_index(drop=True)
        if len(rules) == 0:
            continue

        conviction_vals = rules['conviction'].replace([np.inf, -np.inf], np.nan)

        # ── Compact output (rules.csv) ────────────────────────────────
        fmt = pd.DataFrame()
        # Provenance columns first — always present for easy filtering
        fmt['macro_rule_index']  = m_idx
        fmt['macro_antecedents'] = m_ant
        fmt['macro_consequents'] = m_cons
        fmt['antecedents']       = rules['antecedents'].apply(
            lambda x: ', '.join(sorted(x))
        )
        fmt['consequents']       = rules['consequents'].apply(
            lambda x: ', '.join(sorted(x))
        )
        fmt['support_raw']       = [f'{v:.4f}' for v in rules['support']]
        fmt['support_pct']       = (rules['support']    * 100).round(2)
        fmt['confidence_raw']    = [f'{v:.4f}' for v in rules['confidence']]
        fmt['confidence_pct']    = (rules['confidence'] * 100).round(2)
        fmt['lift']              = rules['lift'].round(4)
        fmt['leverage']          = rules['leverage'].round(6)
        fmt['conviction']        = conviction_vals.round(4)

        # ── Detailed output (rules_detailed.csv) ─────────────────────
        det = pd.DataFrame()
        det['macro_rule_index']       = m_idx
        det['macro_antecedents']      = m_ant
        det['macro_consequents']      = m_cons
        det['antecedents']            = fmt['antecedents']
        det['consequents']            = fmt['consequents']
        det['antecedent_length']      = rules['antecedents'].apply(len)
        det['consequent_length']      = rules['consequents'].apply(len)
        det['rule_length']            = (
            det['antecedent_length'] + det['consequent_length']
        )
        det['antecedent_support_raw'] = [
            f'{v:.4f}' for v in rules['antecedent support']
        ]
        det['antecedent_support_pct'] = (
            rules['antecedent support'] * 100
        ).round(2)
        det['consequent_support_raw'] = [
            f'{v:.4f}' for v in rules['consequent support']
        ]
        det['consequent_support_pct'] = (
            rules['consequent support'] * 100
        ).round(2)
        det['support_raw']            = fmt['support_raw']
        det['support_pct']            = fmt['support_pct']
        det['confidence_raw']         = fmt['confidence_raw']
        det['confidence_pct']         = fmt['confidence_pct']
        det['lift']                   = fmt['lift']
        det['leverage']               = fmt['leverage']
        det['conviction']             = fmt['conviction']
        # Back-mapping: value-level item → originating ACS label
        det['antecedent_labels']      = rules['antecedents'].apply(
            lambda x: ', '.join(
                sorted({i.split('=')[0] for i in x if '=' in i})
            )
        )
        det['consequent_labels']      = rules['consequents'].apply(
            lambda x: ', '.join(
                sorted({i.split('=')[0] for i in x if '=' in i})
            )
        )

        conf_dir.mkdir(parents=True, exist_ok=True)
        fmt.to_csv(conf_dir / 'rules.csv',          index=False)
        det.to_csv(conf_dir / 'rules_detailed.csv', index=False)

        with open(conf_dir / 'summary.txt', 'w') as f:
            f.write('Association Rules Summary (value level)\n')
            f.write(f'{"=" * 60}\n\n')
            f.write('Parent Macroscopic Rule:\n')
            f.write(f'  Rule Index  : {m_idx}\n')
            f.write(f'  Antecedents : {m_ant}\n')
            f.write(f'  Consequents : {m_cons}\n\n')
            f.write('Parameters:\n')
            f.write(f'  Min Support    : {min_sup}\n')
            f.write(f'  Min Confidence : {min_conf}\n')
            f.write(
                f'  Neutral Lift Window (excluded): '
                f'[{lift_window_lo}, {lift_window_hi}]\n\n'
            )
            f.write('Results:\n')
            f.write(f'  Frequent Itemsets : {n_fi}\n')
            f.write(f'  Association Rules : {len(rules)}\n\n')
            f.write('Statistics:\n')
            f.write(
                f'  Avg Support    : {rules["support"].mean() * 100:.2f}%\n'
            )
            f.write(
                f'  Avg Confidence : {rules["confidence"].mean() * 100:.2f}%\n'
            )
            f.write(f'  Avg Lift       : {rules["lift"].mean():.4f}\n')
            f.write(
                f'  Lift Range     : {rules["lift"].min():.4f} — '
                f'{rules["lift"].max():.4f}\n'
            )
            f.write(
                f'  Avg Leverage   : {rules["leverage"].mean():.6f}\n'
            )
            f.write(
                f'  Avg Rule Length: {det["rule_length"].mean():.2f}\n\n'
            )
            n_inf = conviction_vals.isna().sum()
            if n_inf > 0:
                f.write(
                    f'  Note: {n_inf} rule(s) have confidence=1.0 '
                    f'(conviction=inf → saved as NaN)\n\n'
                )
            f.write(f'Top 10 Rules by Lift:\n{"-" * 60}\n')
            for idx2, row2 in fmt.head(10).iterrows():
                f.write(
                    f'{idx2 + 1}. {row2["antecedents"]} => {row2["consequents"]}\n'
                    f'   support={row2["support_pct"]:.2f}% | '
                    f'confidence={row2["confidence_pct"]:.2f}% | '
                    f'lift={row2["lift"]:.4f} | '
                    f'leverage={row2["leverage"]:.6f}\n\n'
                )

        print(f'        > [conf={min_conf:.2f}] {len(rules)} rules → {conf_dir.name}/')

        for min_lift in lift_grid_used:
            filtered = rules[rules['lift'] >= min_lift]
            n  = len(filtered)
            rl = (
                filtered['antecedents'].apply(len) +
                filtered['consequents'].apply(len)
            ) if n > 0 else pd.Series(dtype=float)

            local_summary_rows.append({
                'macro_rule_index':      m_idx,
                'macro_antecedents':     m_ant,
                'macro_consequents':     m_cons,
                'Support':               min_sup,
                'Confidence':            min_conf,
                'Lift_threshold':        min_lift,
                'Number_of_Rules':       n,
                'Max_Lift':   round(filtered['lift'].max(), 4)      if n > 0 else 0.0,
                'Min_Lift':   round(filtered['lift'].min(), 4)      if n > 0 else 0.0,
                'Avg_Lift':   round(filtered['lift'].mean(), 4)     if n > 0 else 0.0,
                'Avg_Confidence': round(filtered['confidence'].mean(), 4) if n > 0 else 0.0,
                'Avg_Support':    round(filtered['support'].mean(), 4)    if n > 0 else 0.0,
                'Avg_Rule_Length':  round(rl.mean(), 4) if n > 0 else 0.0,
                'Max_Rule_Length':  int(rl.max())        if n > 0 else 0,
                'Num_Frequent_Itemsets': n_fi,
                'Num_FI_length_1':       itemsets_by_len.get(1, 0),
                'Num_FI_length_2':       itemsets_by_len.get(2, 0),
                'Num_FI_length_3plus':   sum(
                    v for k2, v in itemsets_by_len.items() if k2 >= 3
                ),
            })

    return local_summary_rows


# ---------------------------------------------------------------------------
# Middle worker — one macroscopic rule
# ---------------------------------------------------------------------------

def _process_one_rule(
    macro_rule: dict,
    tv_path: Path,
    k_dir: Path,
    auto_calibrate: bool,
    sup_min: float, sup_max: float, sup_delta: float,
    conf_min: float, conf_max: float, conf_delta: float,
    lift_min: float, lift_max: float, lift_delta: float,
    lift_neutral_half_window: float,
    inner_n_jobs: int = 1,
) -> dict:
    """
    End-to-end microscopic ARM for one macroscopic rule.

    Called as a joblib.Parallel worker at the rule level.

    Steps
    -----
    A. Create rule_dir, write macro_rule_origin files.
    B. Filter transactions to the rule's active_labels.
    C. Persist aggregated and pair-level CSVs.
    D. One-hot-encode the aggregated transactions.
    E. Calibrate grid parameters (auto) or use manual bounds.
    F. Run full grid search via explore_association_rules_values().

    Returns
    -------
    dict
        Row for rule_index.csv.
    """
    rule_dir = k_dir / macro_rule['rule_slug']
    rule_dir.mkdir(parents=True, exist_ok=True)

    m_idx  = macro_rule['rule_index']
    m_ant  = macro_rule['antecedents']
    m_cons = macro_rule['consequents']

    _base = {
        'rule_index':        m_idx,
        'rule_slug':         macro_rule['rule_slug'],
        'antecedents':       m_ant,
        'consequents':       m_cons,
        'active_labels':     str(sorted(macro_rule['active_labels'])),
        'macro_support_pct': macro_rule.get('macro_support_pct', ''),
        'macro_conf_pct':    macro_rule.get('macro_conf_pct', ''),
        'macro_lift':        macro_rule.get('macro_lift', ''),
        'n_source_files':    len(macro_rule.get('source_paths', [])),
    }

    print(f'\n  Rule {m_idx}: [{m_ant}] => [{m_cons}]')

    # A — provenance artefacts
    _write_macro_rule_origin(rule_dir, macro_rule)

    # B — filter transactions
    try:
        agg_df, pair_df = build_value_transactions(
            tv_path, macro_rule['active_labels']
        )
    except Exception as exc:
        print(f'    ERROR: {exc!r}')
        return {
            **_base,
            'skipped_reason': f'transaction_error: {exc!r}',
            'n_transactions': 0, 'n_items': 0,
            'sup_min_used': None, 'conf_min_used': None, 'lift_max_used': None,
            'summary_rows': 0, 'combos_with_rules': 0,
            'max_rules_any_combo': 0, 'max_lift_observed': 0.0,
        }

    # C — persist transaction CSVs
    agg_df.to_csv(rule_dir / 'aggregated_values_by_sample.csv', index=False)
    pair_df.to_csv(rule_dir / 'values_only_unique.csv',          index=False)

    if agg_df.empty:
        print('    No transactions after filtering — skipping.')
        return {
            **_base,
            'skipped_reason': 'empty_after_filter',
            'n_transactions': 0, 'n_items': 0,
            'sup_min_used': None, 'conf_min_used': None, 'lift_max_used': None,
            'summary_rows': 0, 'combos_with_rules': 0,
            'max_rules_any_combo': 0, 'max_lift_observed': 0.0,
        }

    # D — encode
    df_encoded    = encode_transactions(agg_df['Values'])
    item_supports = df_encoded.mean().sort_values()

    # E — calibrate
    if auto_calibrate:
        params = calibrate_parameters(
            encoded_df=df_encoded,
            sup_delta=sup_delta, lift_delta=lift_delta,
            conf_delta=conf_delta, conf_min_floor=conf_min, conf_max=conf_max,
        )
        _write_calibration_log(
            rule_dir=rule_dir, macro_rule=macro_rule,
            n_transactions=len(df_encoded),
            item_supports=item_supports,
            params=params, auto_calibrate=True,
        )
        if params is None:
            print('    Skipping — too sparse for 2-itemsets.')
            return {
                **_base,
                'skipped_reason': 'too_sparse',
                'n_transactions': len(df_encoded),
                'n_items': df_encoded.shape[1],
                'sup_min_used': None, 'conf_min_used': None,
                'lift_max_used': None,
                'summary_rows': 0, 'combos_with_rules': 0,
                'max_rules_any_combo': 0, 'max_lift_observed': 0.0,
            }
        r_sup_min  = params['sup_min']
        r_sup_max  = params['sup_max']
        r_lift_max = params['lift_max']
        r_conf_min = params['conf_min']
        r_lift_min = params['lift_min']   # always 0.0

    else:
        r_sup_min  = sup_min
        r_sup_max  = sup_max
        r_lift_max = lift_max
        r_conf_min = conf_min
        r_lift_min = lift_min
        _write_calibration_log(
            rule_dir=rule_dir, macro_rule=macro_rule,
            n_transactions=len(df_encoded),
            item_supports=item_supports,
            params=None, auto_calibrate=False,
            manual_params={
                'sup_min':  r_sup_min,  'sup_max':  r_sup_max,
                'conf_min': r_conf_min, 'lift_max': r_lift_max,
            },
        )

    # F — grid search
    summary_df = _explore_rule(
        df=df_encoded, rule_dir=rule_dir, macro_rule=macro_rule,
        sup_min=r_sup_min,   sup_max=r_sup_max,   sup_delta=sup_delta,
        conf_min=r_conf_min, conf_max=conf_max,   conf_delta=conf_delta,
        lift_min=r_lift_min, lift_max=r_lift_max, lift_delta=lift_delta,
        lift_neutral_half_window=lift_neutral_half_window,
        inner_n_jobs=inner_n_jobs,
    )

    has_col           = not summary_df.empty and 'Number_of_Rules' in summary_df.columns
    max_rules         = int(summary_df['Number_of_Rules'].max()) if has_col else 0
    max_lift          = round(summary_df['Max_Lift'].max(), 4)   if has_col else 0.0
    combos_with_rules = int((summary_df['Number_of_Rules'] > 0).sum()) if has_col else 0

    return {
        **_base,
        'skipped_reason':      '',
        'n_transactions':      len(df_encoded),
        'n_items':             df_encoded.shape[1],
        'sup_min_used':        r_sup_min,
        'conf_min_used':       r_conf_min,
        'lift_max_used':       r_lift_max,
        'summary_rows':        len(summary_df),
        'combos_with_rules':   combos_with_rules,
        'max_rules_any_combo': max_rules,
        'max_lift_observed':   max_lift,
    }


# ---------------------------------------------------------------------------
# Grid exploration for one rule
# ---------------------------------------------------------------------------

def _explore_rule(
    df,
    rule_dir: Path,
    macro_rule: dict,
    sup_min, sup_max, sup_delta,
    conf_min, conf_max, conf_delta,
    lift_min, lift_max, lift_delta,
    lift_neutral_half_window: float = 0.25,
    inner_n_jobs: int | None        = None,
) -> pd.DataFrame:
    """
    Run the full support × confidence × lift grid for one rule_dir.

    Parallelises over support thresholds (inner / Level-3 workers).

    Returns a summary DataFrame sorted by (Number_of_Rules DESC,
    Max_Lift DESC), or an empty DataFrame if nothing was found.
    """
    rule_dir = Path(rule_dir)

    support_grid    = np.round(
        np.arange(sup_min, sup_max + sup_delta / 2, sup_delta), 4
    )
    confidence_grid = np.round(
        np.arange(conf_min, conf_max + conf_delta / 2, conf_delta), 4
    )
    lift_grid       = np.round(
        np.arange(lift_min, lift_max + lift_delta / 2, lift_delta), 4
    )

    lift_window_lo, lift_window_hi = _neutral_window(lift_neutral_half_window)
    lift_grid_used = [
        v for v in lift_grid
        if not (lift_window_lo <= v <= lift_window_hi)
    ]
    total_combos = (
        len(support_grid) * len(confidence_grid) * len(lift_grid_used)
    )

    m_ant = macro_rule['antecedents']
    m_cons = macro_rule['consequents']
    m_idx  = macro_rule['rule_index']

    print(f'\n    {"=" * 56}')
    print(
        f'    Grid for rule {m_idx}: [{m_ant}] => [{m_cons}]'
    )
    print(
        f'    sup: {len(support_grid)} values | '
        f'conf: {len(confidence_grid)} values | '
        f'lift: {len(lift_grid_used)} values | '
        f'total: {total_combos:,}'
    )

    perf_cores = _get_perf_cores()
    if inner_n_jobs is None:
        inner_n_jobs = min(perf_cores, len(support_grid))
    n_jobs = min(inner_n_jobs, len(support_grid))
    print(f'    inner n_jobs (support): {n_jobs}')

    parallel_results = Parallel(
        n_jobs=n_jobs, backend='loky', verbose=0, pre_dispatch=n_jobs,
    )(
        delayed(_process_one_support)(
            min_sup         = min_sup,
            sup_idx         = sup_idx,
            n_sup           = len(support_grid),
            df              = df,
            output_dir      = rule_dir,
            confidence_grid = confidence_grid,
            lift_grid_used  = lift_grid_used,
            lift_window_lo  = lift_window_lo,
            lift_window_hi  = lift_window_hi,
            macro_rule      = macro_rule,
        )
        for sup_idx, min_sup in enumerate(support_grid, start=1)
    )

    summary_rows = [
        row for worker_rows in parallel_results for row in worker_rows
    ]
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=['Number_of_Rules', 'Max_Lift'], ascending=[False, False]
        ).reset_index(drop=True)

    summary_df.to_csv(rule_dir / 'summary.csv', index=False)
    combos_with_rules = (
        int((summary_df['Number_of_Rules'] > 0).sum())
        if not summary_df.empty else 0
    )

    with open(rule_dir / 'exploration_summary.txt', 'w') as f:
        f.write('FULL EXPLORATION SUMMARY (value level)\n')
        f.write(f'{"=" * 70}\n\n')
        f.write(
            f'Generated: '
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n'
        )
        f.write('Parent Macroscopic Rule:\n')
        f.write(f'  Rule Index  : {m_idx}\n')
        f.write(f'  Antecedents : {m_ant}\n')
        f.write(f'  Consequents : {m_cons}\n\n')
        f.write('Parameter Grids:\n')
        f.write(
            f'  Support    : {len(support_grid)} values '
            f'[{support_grid[0]} … {support_grid[-1]}, step={sup_delta}]\n'
        )
        f.write(
            f'  Confidence : {len(confidence_grid)} values '
            f'[{confidence_grid[0]} … {confidence_grid[-1]}, step={conf_delta}]\n'
        )
        f.write(
            f'  Lift       : {len(lift_grid_used)} values '
            f'(neutral window [{lift_window_lo}, {lift_window_hi}] excluded, '
            f'step={lift_delta})\n\n'
        )
        f.write('Results:\n')
        f.write(f'  Total combinations : {total_combos:,}\n')
        f.write(f'  With >= 1 rule     : {combos_with_rules:,}\n\n')
        if not summary_df.empty and combos_with_rules > 0:
            best = summary_df.iloc[0]
            f.write('Best combination (most rules, then highest lift):\n')
            f.write(f'{"-" * 60}\n')
            f.write(f'  Support         : {best["Support"]}\n')
            f.write(f'  Confidence      : {best["Confidence"]}\n')
            f.write(f'  Lift threshold  : {best["Lift_threshold"]}\n')
            f.write(f'  Number of Rules : {int(best["Number_of_Rules"])}\n')
            f.write(f'  Max Lift        : {best["Max_Lift"]}\n')

    print('    > Cleaning empty folders...')
    rc, rs = cleanup_empty_folders(rule_dir)
    print(f'    > Removed {rc} conf dir(s) and {rs} sup dir(s)')

    with open(rule_dir / 'exploration_summary.txt', 'a') as f:
        f.write('\nParallelism:\n')
        f.write(f'  n_jobs (support-level): {n_jobs}\n\n')
        f.write('Folder cleanup:\n')
        f.write(f'  conf dirs removed: {rc}\n')
        f.write(f'  sup dirs removed : {rs}\n')

    plot_heatmaps(
        summary_df, rule_dir,
        macro_rule=macro_rule,
        lift_neutral_half_window=lift_neutral_half_window,
        lift_delta=lift_delta,
    )

    return summary_df


# ---------------------------------------------------------------------------
# Outer worker — one k value
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
    outer_jobs: int = 1,
) -> tuple[int, list[dict]]:
    """
    Run microscopic ARM for every unique macroscopic rule found for (region, k).

    Level-1 worker (k-level).  Dispatches Level-2 (_process_one_rule) and
    propagates the three-level parallelism budget downward.

    Returns
    -------
    tuple[int, list[dict]]
        (k, rule_index_rows) where rule_index_rows is the list of per-rule
        outcome dicts written to rule_index.csv.
    """
    k_dir = output_dir / f'k_{k}'
    k_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"=" * 70}')
    print(f'  k = {k}  |  region = {region}')
    print(f'{"=" * 70}')

    macro_rules = collect_unique_macro_rules(results_dir, region, k)
    if not macro_rules:
        print(
            f'  > No macroscopic rules found for k={k}.  '
            f'Run Step 3 first.'
        )
        return k, []

    print(f'  > {len(macro_rules)} unique macro rules found for k={k}')

    tv_path = (
        results_dir / region / 'important_features'
        / f'k_{k}' / 'transactions_values.csv'
    )
    if not tv_path.exists():
        print(f'  > transactions_values.csv not found: {tv_path}')
        skipped = [
            {
                'rule_index': r['rule_index'],
                'rule_slug':  r['rule_slug'],
                'antecedents': r['antecedents'],
                'consequents': r['consequents'],
                'active_labels': str(sorted(r['active_labels'])),
                'macro_support_pct': r.get('macro_support_pct', ''),
                'macro_conf_pct':    r.get('macro_conf_pct', ''),
                'macro_lift':        r.get('macro_lift', ''),
                'n_source_files': len(r.get('source_paths', [])),
                'skipped_reason': 'no_transactions_values',
                'n_transactions': None, 'n_items': None,
                'sup_min_used': None, 'conf_min_used': None,
                'lift_max_used': None,
                'summary_rows': 0, 'combos_with_rules': 0,
                'max_rules_any_combo': 0, 'max_lift_observed': 0.0,
            }
            for r in macro_rules
        ]
        pd.DataFrame(skipped).to_csv(k_dir / 'rule_index.csv', index=False)
        return k, skipped

    # Three-level parallelism budget
    # ────────────────────────────────
    # outer_jobs  (k-level)    : passed in from run_k_comparison_values
    # rule_jobs   (rule-level) : max(1, perf_cores // outer_jobs),
    #                            capped at number of rules
    # inner_jobs  (sup-level)  : max(1, perf_cores // (outer × rule))
    #
    # Guarantee: outer × rule × inner ≤ perf_cores
    #
    # Example: M1 Ultra, 16 P-cores, 2 k-values, 8 macro rules
    #   outer_jobs = 2
    #   rule_jobs  = min(8, max(1, 16//2)) = min(8, 8) = 8
    #   inner_jobs = max(1, 16//(2×8)) = max(1, 1) = 1
    #   total = 2 × 8 × 1 = 16 — perfectly saturates all P-cores
    perf_cores = _get_perf_cores()
    rule_jobs  = min(len(macro_rules), max(1, perf_cores // outer_jobs))
    inner_jobs = max(1, perf_cores // (outer_jobs * rule_jobs))

    print(
        f'  > rule_jobs={rule_jobs}, inner_jobs={inner_jobs} '
        f'(outer={outer_jobs} × rule={rule_jobs} × inner={inner_jobs} '
        f'= {outer_jobs * rule_jobs * inner_jobs} ≤ {perf_cores} P-cores)'
    )

    # Dispatch rules in parallel (middle level)
    rule_results: list[dict] = Parallel(
        n_jobs=rule_jobs, backend='loky', verbose=0,
    )(
        delayed(_process_one_rule)(
            macro_rule               = rule,
            tv_path                  = tv_path,
            k_dir                    = k_dir,
            auto_calibrate           = auto_calibrate,
            sup_min=sup_min,   sup_max=sup_max,   sup_delta=sup_delta,
            conf_min=conf_min, conf_max=conf_max, conf_delta=conf_delta,
            lift_min=lift_min, lift_max=lift_max, lift_delta=lift_delta,
            lift_neutral_half_window = lift_neutral_half_window,
            inner_n_jobs             = inner_jobs,
        )
        for rule in macro_rules
    )

    pd.DataFrame(rule_results).to_csv(k_dir / 'rule_index.csv', index=False)
    print(f'\n  > rule_index.csv saved ({len(rule_results)} rows)')

    return k, rule_results


# ---------------------------------------------------------------------------
# K-comparison
# ---------------------------------------------------------------------------

def run_k_comparison_values(
    k_values: list[int],
    results_dir: Path,
    region: str,
    output_dir: Path,
    auto_calibrate: bool    = True,
    sup_min: float          = 0.02,
    sup_max: float          = 0.50,
    sup_delta: float        = 0.02,
    conf_min: float         = 0.05,
    conf_max: float         = 1.00,
    conf_delta: float       = 0.05,
    lift_min: float         = 0.0,
    lift_max: float         = 5.0,
    lift_delta: float       = 0.05,
    lift_neutral_half_window: float = 0.25,
) -> None:
    """
    Run per-rule microscopic ARM for every k in *k_values* and produce a
    cross-k comparison report.

    Three-level parallelism
    -----------------------
    Level 1 (outer)  : k values.
                       outer_jobs = min(n_k, perf_cores)
    Level 2 (middle) : macroscopic rules within each k-worker.
                       rule_jobs = min(n_rules, max(1, perf_cores//outer))
    Level 3 (inner)  : support thresholds within each rule-worker.
                       inner_jobs = max(1, perf_cores//(outer×rule))
    Guarantee: outer × rule × inner ≤ perf_cores.
    """
    comp_dir = output_dir / 'k_comparison'
    comp_dir.mkdir(parents=True, exist_ok=True)

    k_sorted   = sorted(k_values)
    perf_cores = _get_perf_cores()
    outer_jobs = min(len(k_sorted), perf_cores)

    print(f'\n{"=" * 70}')
    print(f'K-VARIATION EXPERIMENT (VALUE LEVEL, PER-RULE) — region: {region}')
    print(f'{"=" * 70}')
    print(f'  > k values      : {k_sorted}')
    print(f'  > outer n_jobs  : {outer_jobs}  (k-level)')
    print(f'  > auto_calibrate: {auto_calibrate}')
    print('-' * 50)

    raw_results = Parallel(
        n_jobs=outer_jobs, backend='loky', verbose=0,
    )(
        delayed(_process_one_k)(
            k                        = k,
            results_dir              = results_dir,
            region                   = region,
            output_dir               = output_dir,
            auto_calibrate           = auto_calibrate,
            sup_min=sup_min,   sup_max=sup_max,   sup_delta=sup_delta,
            conf_min=conf_min, conf_max=conf_max, conf_delta=conf_delta,
            lift_min=lift_min, lift_max=lift_max, lift_delta=lift_delta,
            lift_neutral_half_window = lift_neutral_half_window,
            outer_jobs               = outer_jobs,
        )
        for k in k_sorted
    )

    # Collect all rule_index rows across k values
    all_rows: list[dict] = []
    for k, rule_rows in raw_results:
        for row in rule_rows:
            row_with_k       = dict(row)
            row_with_k['k']  = k
            all_rows.append(row_with_k)

    comp_df = pd.DataFrame(all_rows)
    comp_df.to_csv(comp_dir / 'k_comparison_summary.csv', index=False)

    print(f'\n{"=" * 70}')
    print('  > Building cross-k comparison report...')

    with open(comp_dir / 'k_comparison_summary.txt', 'w') as f:
        f.write(
            'K-VARIATION EXPERIMENT SUMMARY '
            '(value level, per macroscopic rule)\n'
        )
        f.write(f'{"=" * 70}\n\n')
        f.write(
            f'Generated      : '
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
        )
        f.write(f'Region         : {region}\n')
        f.write(f'k values tested: {k_sorted}\n')
        f.write(f'auto_calibrate : {auto_calibrate}\n\n')

        if comp_df.empty:
            f.write('No results produced.\n')
        else:
            skipped_mask = (
                comp_df['skipped_reason'] != ''
                if 'skipped_reason' in comp_df.columns
                else pd.Series([False] * len(comp_df))
            )
            skipped = comp_df[skipped_mask]
            ran     = comp_df[~skipped_mask]

            if not skipped.empty:
                f.write(
                    f'Skipped ({len(skipped)} rule-k pairs):\n'
                    f'{"-" * 40}\n'
                )
                for _, row in skipped.iterrows():
                    f.write(
                        f'  k={row["k"]}  '
                        f'[{row["antecedents"]}]=>[{row["consequents"]}]: '
                        f'{row["skipped_reason"]}\n'
                    )
                f.write('\n')

            if not ran.empty:
                summary_cols = [
                    'k', 'rule_index', 'antecedents', 'consequents',
                    'n_transactions', 'n_items',
                    'max_rules_any_combo', 'max_lift_observed',
                ]
                summary_cols = [c for c in summary_cols if c in ran.columns]
                f.write(f'Results ({len(ran)} rule-k pairs):\n{"-" * 60}\n')
                f.write(ran[summary_cols].to_string(index=False))
                f.write('\n\n')

                if 'max_rules_any_combo' in ran.columns:
                    nz = ran[ran['max_rules_any_combo'] > 0]
                    if not nz.empty:
                        best = nz.loc[nz['max_rules_any_combo'].idxmax()]
                        f.write(
                            f'Best result: k={best["k"]}  '
                            f'[{best["antecedents"]}]=>[{best["consequents"]}]  '
                            f'{int(best["max_rules_any_combo"])} rules '
                            f'at best combination\n'
                        )

    # Cross-k heatmaps (k vs Support/Confidence/Lift, pooled over all rules)
    all_summaries = []
    for k in k_sorted:
        k_dir = output_dir / f'k_{k}'
        if not k_dir.exists():
            continue
        for rule_dir in sorted(k_dir.glob('rule_*')):
            summary_csv = rule_dir / 'summary.csv'
            if not summary_csv.exists():
                continue
            try:
                sdf    = pd.read_csv(summary_csv)
                sdf['k'] = k
                all_summaries.append(sdf)
            except Exception:
                pass

    if not all_summaries:
        print('  > No rule summaries found — skipping cross-k heatmaps.')
        print(f'  > k_comparison saved to {comp_dir}/')
        return

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FutureWarning)
        combined = pd.concat(all_summaries, ignore_index=True)

    if combined.empty or 'Lift_threshold' not in combined.columns:
        print(f'  > k_comparison saved to {comp_dir}/')
        return

    combined['Lift_display'] = (
        (combined['Lift_threshold'] / 0.1).round() * 0.1
    ).round(4)
    neutral_lo, neutral_hi   = _neutral_window(lift_neutral_half_window)

    for x_col, suffix in [
        ('Support',      'k_support'),
        ('Confidence',   'k_confidence'),
        ('Lift_display', 'k_lift'),
    ]:
        is_lift = 'lift' in suffix
        x_label = 'Lift' if is_lift else x_col

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
                :,
                ~pivot.columns.to_series().between(
                    neutral_lo, neutral_hi, inclusive='both'
                ),
            ]
            nz_mask = (pivot != 0).any(axis=0).values
            if nz_mask.any():
                last_nz = int(np.where(nz_mask)[0].max())
                pivot   = pivot.iloc[:, :last_nz + 1]

        n_cols = len(pivot.columns)
        n_rows = len(pivot.index)
        fig, ax = plt.subplots(
            figsize=(max(10, n_cols * 0.75), max(4, n_rows * 0.6))
        )

        img = ax.imshow(
            pivot.values, aspect='auto', cmap='YlOrBr', interpolation='nearest'
        )
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(
            [f'{v:.2f}' if isinstance(v, float) else str(v)
             for v in pivot.columns],
            rotation=40, ha='right', fontsize=8,
        )
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f'k={v}' for v in pivot.index], fontsize=9)
        ax.set_xlabel(x_label, fontsize=11, labelpad=8)
        ax.set_ylabel('k',     fontsize=11, labelpad=8)
        ax.set_title(
            f'Max Micro-Rules (pooled over all macro rules) — k vs {x_label}\n'
            f'(darker = more rules; max over the other two parameters)',
            fontsize=11, pad=14,
        )

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
        fig.savefig(
            comp_dir / f'heatmap_{suffix}.png', dpi=150, bbox_inches='tight'
        )
        plt.close(fig)
        print(f'    > saved k_comparison/heatmap_{suffix}.png')

    print(f'  > k_comparison saved to {comp_dir}/')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    regions: list[str]              = None,
    k_values: list[int]             = None,
    auto_calibrate: bool            = True,
    sup_min: float                  = 0.02,
    sup_max: float                  = 0.50,
    sup_delta: float                = 0.02,
    conf_min: float                 = 0.05,
    conf_max: float                 = 1.00,
    conf_delta: float               = 0.05,
    lift_min: float                 = 0.0,
    lift_max: float                 = 5.0,
    lift_delta: float               = 0.05,
    lift_neutral_half_window: float = 0.25,
    base_dir: Path                  = None,
) -> None:
    """
    Entry point for per-rule value-level association-rule mining.

    Mirrors macroscopic_experiment_association_rules.main() so that main.py
    can invoke both steps with an identical parameter set.

    Parameters
    ----------
    regions : list[str] or None
        Regions to process.  Defaults to ['northeast', 'south'].
    k_values : list[int] or None
        CF neighbourhood sizes.  Defaults to [1, 3, 5, 7].
    auto_calibrate : bool
        Derive grid bounds from item frequencies (True) or use manual values.
    sup_min … lift_neutral_half_window : float
        Grid parameters.  In auto mode used as floor / ceiling constraints.
    base_dir : Path or None
        Project root.  Auto-detected for Kaggle, Colab, and local envs.
    """
    print(
        f'  > Parallel backend — {_CPU_CORES} logical cores / '
        f'{_get_perf_cores()} perf cores (loky, 3-level)\n'
    )

    if base_dir is None:
        if Path('/kaggle/working').exists():
            base_dir = Path('/kaggle/working')
        elif Path('/content').exists():
            base_dir = Path('/content')
        else:
            base_dir = Path(__file__).resolve().parent.parent
    base_dir = Path(base_dir)

    if regions  is None: regions  = ['northeast', 'south']
    if k_values is None: k_values = [1, 3, 5, 7]

    results_dir = base_dir / 'results'

    exp_label = _experiment_label(
        auto_calibrate=auto_calibrate,
        sup_min=sup_min,   sup_max=sup_max,   sup_delta=sup_delta,
        conf_min=conf_min, conf_max=conf_max, conf_delta=conf_delta,
        lift_min=lift_min, lift_max=lift_max, lift_delta=lift_delta,
        lift_neutral_half_window=lift_neutral_half_window,
    )

    for region in regions:
        out_dir = (
            results_dir / region / 'association_rules_values' / exp_label
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        print('\n' + '=' * 70)
        print(
            f'ASSOCIATION RULES (VALUE LEVEL — PER MACRO RULE) — '
            f'{region.upper()}'
        )
        print(f'Experiment label: {exp_label}')
        print('=' * 70 + '\n')

        run_k_comparison_values(
            k_values=k_values,
            results_dir=results_dir,
            region=region,
            output_dir=out_dir,
            auto_calibrate=auto_calibrate,
            sup_min=sup_min,   sup_max=sup_max,   sup_delta=sup_delta,
            conf_min=conf_min, conf_max=conf_max, conf_delta=conf_delta,
            lift_min=lift_min, lift_max=lift_max, lift_delta=lift_delta,
            lift_neutral_half_window=lift_neutral_half_window,
        )

    print('\n' + '=' * 70)
    print('Done.')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()