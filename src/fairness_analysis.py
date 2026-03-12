"""
fairness_analysis_association_rules.py
======================================
Fairness analysis on value-level association rules produced by the pipeline.

Pipeline position
-----------------
    microscopic_experiment_association_rules_values.py  ->  [this script]

Analysis levels
---------------
The analysis runs at three nested levels, mirroring the directory structure
produced by the microscopic ARM step:

  1. Per-configuration  (one rules.csv per sup/conf pair)
         results/{region}/fairness_analysis/{exp_label}/k_{k}/sup_{x}/conf_{y}/

  2. Per-k  (aggregation across all sup/conf configs for a given k)
         results/{region}/fairness_analysis/{exp_label}/k_{k}/

  3. Global  (aggregation across all k values and all configs)
         results/{region}/fairness_analysis/global/

Population-level metrics (from the original dataset CSV, independent of rules)
are computed once per region and written to:
         results/{region}/fairness_analysis/population_fairness.csv

Metrics computed
----------------
Per-configuration and aggregated:
  1.  Group coverage          -- fraction of rules mentioning each demographic value
  2.  Confidence parity       -- mean/median confidence per group (outcome-targeting rules)
  3.  Support parity          -- same for support
  4.  Lift disparity          -- mean lift per group + Mann-Whitney U vs privileged
  5.  Disparate Impact Ratio  -- mean_conf(unprivileged) / mean_conf(privileged); 4/5 rule
  6.  Statistical Parity Diff -- signed confidence gap vs privileged group

Aggregated additionally provides per group:
  7.  mean_dir / std_dir / min_dir / max_dir  across configs or k values
  8.  n_configs  -- how many configs produced rules for that group
  9.  n_violations  -- how many configs flagged a DIR < 0.80 violation

Population-level (dataset CSV):
  10. Base-rate DIR and SPD
  11. Per-group positive rates and sample counts

Intersectional (SEX x RAC1P):
  12. Cross-tabulation of rule count, mean confidence, support, lift

Output structure
----------------
    results/{region}/fairness_analysis/
        population_fairness.csv
        plots/
            population_fairness_{attr}.png

        {exp_label}/k_{k}/sup_{x}/conf_{y}/     <- per-config
            fairness_report.txt
            fairness_metrics.csv
            disparate_impact.csv
            confidence_parity.csv
            support_parity.csv
            rule_coverage.csv
            lift_disparity.csv
            intersectional_analysis.csv
            plots/

        {exp_label}/k_{k}/                       <- per-k aggregation
            k_fairness_report.txt
            k_dir_summary.csv
            k_dir_all_configs.csv
            k_coverage_summary.csv
            k_intersectional_analysis.csv
            plots/
                k_dir_heatmap_sup_conf_{attr}_{group}.png
                k_dir_barplot_{attr}.png

        global/                                  <- global aggregation
            global_fairness_report.txt
            global_dir_summary.csv
            global_dir_evolution.csv
            plots/
                global_dir_evolution_{attr}.png
                global_k_heatmap_{attr}.png

Pipeline usage
--------------
    main(regions, k_values, base_dir, ...)

    Mirrors the signature of microscopic_experiment_association_rules_values.main()
    so that main.py can invoke this step with the same parameter set without any
    adapter code.

Standalone usage
----------------
    # Analyse a structured directory tree
    python fairness_analysis_association_rules.py \\
        --rules results/northeast/association_rules_values/ \\
        --dataset data/acs_income_northeast_2024.csv

    # Custom attributes
    python fairness_analysis_association_rules.py \\
        --rules path/to/rules.csv --output_dir path/to/out/ \\
        --sensitive_attrs GENERE ETNIA \\
        --privileged GENERE=M ETNIA=Bianco
"""

from __future__ import annotations

import argparse
import ast
import datetime
import re
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use('Agg')
warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SENSITIVE_ATTRS: list[str] = ['SEX', 'RAC1P']

DEFAULT_PRIVILEGED_VALUES: dict[str, str] = {
    'SEX':   'Male',
    'RAC1P': 'White-Alone',
}

DEFAULT_OUTCOME_LABEL: str    = 'INCOME_ABOVE_THRESHOLD'
DEFAULT_POSITIVE_OUTCOME: str = '1'

DIR_THRESHOLD: float = 0.80

_PALETTE = {
    'blue':   '#3A7DC9',
    'orange': '#E07B39',
    'red':    '#D94F3D',
    'green':  '#4CAF50',
    'grey':   '#8B8B8B',
}


# ===========================================================================
# SECTION 1 -- Parsing and loading
# ===========================================================================

def parse_itemset(cell: str) -> list[str]:
    """Parse an itemset cell from frozenset/set/tuple/plain-CSV representations."""
    if not isinstance(cell, str):
        return []
    cell = cell.strip()
    if cell.startswith('frozenset('):
        try:
            return [str(x).strip() for x in ast.literal_eval(cell[len('frozenset('):-1])]
        except Exception:
            pass
    if cell.startswith('{') and cell.endswith('}'):
        try:
            parsed = ast.literal_eval(cell)
            if isinstance(parsed, (set, frozenset)):
                return [str(x).strip() for x in parsed]
        except Exception:
            pass
    if cell.startswith('(') and cell.endswith(')'):
        try:
            parsed = ast.literal_eval(cell)
            if isinstance(parsed, tuple):
                return [str(x).strip() for x in parsed]
        except Exception:
            pass
    return [item.strip() for item in cell.split(',') if item.strip()]


def extract_label_value(item: str) -> tuple[str, str] | None:
    """Split LABEL=value -> (label, value), or None if no '='."""
    if '=' not in item:
        return None
    label, _, value = item.partition('=')
    return label.strip(), value.strip()


def load_rules(path: Path) -> pd.DataFrame:
    """Load rules.csv and attach parsed ant_items / con_items / all_items columns."""
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    for required in ('antecedents', 'consequents'):
        if required not in df.columns:
            raise ValueError(f"Column '{required}' missing from {path}")
    df['ant_items'] = df['antecedents'].apply(parse_itemset)
    df['con_items'] = df['consequents'].apply(parse_itemset)
    df['all_items'] = df.apply(lambda r: r['ant_items'] + r['con_items'], axis=1)
    for col in ('support_raw', 'support_pct', 'confidence_raw', 'confidence_pct',
                'lift', 'leverage', 'conviction'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# ---------------------------------------------------------------------------
# Rules discovery
# ---------------------------------------------------------------------------

def discover_rules_structure(
    ar_values_dir: Path,
    k_values: list[int] | None = None,
) -> dict[tuple[str, int, str, str], Path]:
    """
    Recursively discover all rules.csv files and index them by
    (exp_label, k, sup_thresh, conf_thresh).

    Expected path pattern:
        {ar_values_dir}/{exp_label}/k_{k}/sup_{x}/conf_{y}/rules.csv
    """
    index: dict[tuple[str, int, str, str], Path] = {}
    for p in sorted(ar_values_dir.rglob('rules.csv')):
        parts    = p.parts
        k_part   = next((x for x in parts if re.fullmatch(r'k_\d+', x)),  None)
        sup_part = next((x for x in parts if x.startswith('sup_')),  None)
        con_part = next((x for x in parts if x.startswith('conf_')), None)
        if k_part is None or sup_part is None or con_part is None:
            continue
        k_val = int(k_part.split('_')[1])
        if k_values and k_val not in k_values:
            continue
        try:
            base_idx  = parts.index(ar_values_dir.name)
            exp_label = parts[base_idx + 1]
        except (ValueError, IndexError):
            exp_label = 'unknown'
        key = (exp_label, k_val, sup_part.replace('sup_', ''),
               con_part.replace('conf_', ''))
        index[key] = p
    return index


# ===========================================================================
# SECTION 2 -- Per-configuration metric computation
# ===========================================================================

def _has_positive_outcome(con_items: list[str], outcome_label: str,
                           positive_outcome: str) -> bool:
    return any(extract_label_value(i) == (outcome_label, positive_outcome)
               for i in con_items)


def compute_rule_coverage(rules_df: pd.DataFrame,
                           sensitive_attrs: list[str]) -> pd.DataFrame:
    total   = len(rules_df)
    records = []
    for attr in sensitive_attrs:
        counts: dict[str, int] = {}
        for _, row in rules_df.iterrows():
            seen = {lv[1] for item in row['all_items']
                    if (lv := extract_label_value(item)) and lv[0] == attr}
            for v in seen:
                counts[v] = counts.get(v, 0) + 1
        for val, cnt in sorted(counts.items()):
            records.append({
                'attribute':    attr,
                'value':        val,
                'rule_count':   cnt,
                'coverage_pct': round(100.0 * cnt / total, 4) if total else 0.0,
            })
    return pd.DataFrame(records)


def compute_confidence_parity(rules_df: pd.DataFrame,
                               sensitive_attrs: list[str],
                               privileged_values: dict[str, str],
                               outcome_label: str,
                               positive_outcome: str) -> pd.DataFrame:
    if 'confidence_raw' not in rules_df.columns:
        return pd.DataFrame()
    mask     = rules_df['con_items'].apply(
        lambda c: _has_positive_outcome(c, outcome_label, positive_outcome))
    subset   = rules_df[mask] if mask.any() else rules_df
    fallback = not mask.any()
    records  = []
    for attr in sensitive_attrs:
        grp: dict[str, list[float]] = {}
        for _, row in subset.iterrows():
            c = row['confidence_raw']
            if pd.isna(c):
                continue
            for item in row['ant_items']:
                lv = extract_label_value(item)
                if lv and lv[0] == attr:
                    grp.setdefault(lv[1], []).append(float(c))
        for val, confs in grp.items():
            records.append({
                'attribute':          attr,
                'value':              val,
                'is_privileged':      val == privileged_values.get(attr),
                'n_rules':            len(confs),
                'mean_confidence':    round(np.mean(confs),   6),
                'median_confidence':  round(np.median(confs), 6),
                'std_confidence':     round(np.std(confs),    6),
                'min_confidence':     round(np.min(confs),    6),
                'max_confidence':     round(np.max(confs),    6),
                'fallback_all_rules': fallback,
            })
    return pd.DataFrame(records)


def compute_support_parity(rules_df: pd.DataFrame,
                            sensitive_attrs: list[str],
                            privileged_values: dict[str, str]) -> pd.DataFrame:
    if 'support_raw' not in rules_df.columns:
        return pd.DataFrame()
    records = []
    for attr in sensitive_attrs:
        grp: dict[str, list[float]] = {}
        for _, row in rules_df.iterrows():
            s = row['support_raw']
            if pd.isna(s):
                continue
            for item in row['ant_items']:
                lv = extract_label_value(item)
                if lv and lv[0] == attr:
                    grp.setdefault(lv[1], []).append(float(s))
        for val, sups in grp.items():
            records.append({
                'attribute':      attr,
                'value':          val,
                'is_privileged':  val == privileged_values.get(attr),
                'n_rules':        len(sups),
                'mean_support':   round(np.mean(sups),   6),
                'median_support': round(np.median(sups), 6),
                'std_support':    round(np.std(sups),    6),
            })
    return pd.DataFrame(records)


def compute_lift_disparity(rules_df: pd.DataFrame,
                            sensitive_attrs: list[str],
                            privileged_values: dict[str, str]) -> pd.DataFrame:
    if 'lift' not in rules_df.columns:
        return pd.DataFrame()
    grp_lifts: dict[str, dict[str, list[float]]] = {a: {} for a in sensitive_attrs}
    for _, row in rules_df.iterrows():
        lft = row['lift']
        if pd.isna(lft):
            continue
        for attr in sensitive_attrs:
            for item in row['all_items']:
                lv = extract_label_value(item)
                if lv and lv[0] == attr:
                    grp_lifts[attr].setdefault(lv[1], []).append(float(lft))
    records = []
    for attr in sensitive_attrs:
        priv_val   = privileged_values.get(attr)
        priv_lifts = grp_lifts[attr].get(priv_val, [])
        for val, lifts in sorted(grp_lifts[attr].items()):
            is_priv = val == priv_val
            u_stat  = pval = np.nan
            if not is_priv and len(lifts) >= 2 and len(priv_lifts) >= 2:
                try:
                    u_stat, pval = stats.mannwhitneyu(lifts, priv_lifts,
                                                       alternative='two-sided')
                except Exception:
                    pass
            records.append({
                'attribute':          attr,
                'value':              val,
                'is_privileged':      is_priv,
                'n_rules':            len(lifts),
                'mean_lift':          round(np.mean(lifts),   6) if lifts else np.nan,
                'median_lift':        round(np.median(lifts), 6) if lifts else np.nan,
                'std_lift':           round(np.std(lifts),    6) if lifts else np.nan,
                'mannwhitney_u':      round(u_stat, 4) if pd.notna(u_stat) else np.nan,
                'pval_vs_privileged': round(pval,   6) if pd.notna(pval)   else np.nan,
            })
    return pd.DataFrame(records)


def compute_disparate_impact(conf_parity_df: pd.DataFrame,
                              privileged_values: dict[str, str],
                              use_mean: bool = True) -> pd.DataFrame:
    if conf_parity_df.empty:
        return pd.DataFrame()
    metric  = 'mean_confidence' if use_mean else 'median_confidence'
    records = []
    for attr in conf_parity_df['attribute'].unique():
        sub      = conf_parity_df[conf_parity_df['attribute'] == attr]
        priv_val = privileged_values.get(attr)
        priv_row = sub[sub['value'] == priv_val]
        if priv_row.empty:
            continue
        priv_conf = float(priv_row[metric].values[0])
        for _, row in sub[sub['value'] != priv_val].iterrows():
            uc      = float(row[metric])
            dir_val = uc / priv_conf if priv_conf > 0 else np.nan
            spd     = uc - priv_conf
            records.append({
                'attribute':                     attr,
                'privileged_value':              priv_val,
                'unprivileged_value':            row['value'],
                'n_rules_privileged':            int(priv_row['n_rules'].values[0]),
                'n_rules_unprivileged':          int(row['n_rules']),
                'privileged_confidence':         round(priv_conf, 6),
                'unprivileged_confidence':       round(uc, 6),
                'disparate_impact_ratio':        round(dir_val, 6) if pd.notna(dir_val) else np.nan,
                'statistical_parity_difference': round(spd, 6),
                'dir_threshold':                 DIR_THRESHOLD,
                'fairness_violation':            bool(pd.notna(dir_val) and dir_val < DIR_THRESHOLD),
            })
    return pd.DataFrame(records)


def compute_intersectional_analysis(rules_df: pd.DataFrame, attr1: str,
                                     attr2: str, outcome_label: str,
                                     positive_outcome: str) -> pd.DataFrame:
    records = []
    for _, row in rules_df.iterrows():
        v1 = v2 = None
        for item in row['all_items']:
            lv = extract_label_value(item)
            if lv:
                if lv[0] == attr1 and v1 is None:
                    v1 = lv[1]
                elif lv[0] == attr2 and v2 is None:
                    v2 = lv[1]
        if v1 is None or v2 is None:
            continue
        c   = row.get('confidence_raw', np.nan)
        s   = row.get('support_raw',    np.nan)
        lft = row.get('lift',           np.nan)
        records.append({
            attr1: v1, attr2: v2,
            'targets_positive_outcome': int(_has_positive_outcome(
                row['con_items'], outcome_label, positive_outcome)),
            'confidence': float(c)   if pd.notna(c)   else np.nan,
            'support':    float(s)   if pd.notna(s)   else np.nan,
            'lift':       float(lft) if pd.notna(lft) else np.nan,
        })
    if not records:
        return pd.DataFrame()
    df  = pd.DataFrame(records)
    grp = df.groupby([attr1, attr2]).agg(
        n_rules                  = ('confidence', 'count'),
        mean_confidence          = ('confidence', 'mean'),
        mean_support             = ('support',    'mean'),
        mean_lift                = ('lift',        'mean'),
        n_positive_outcome_rules = ('targets_positive_outcome', 'sum'),
    ).reset_index()
    for col in ('mean_confidence', 'mean_support', 'mean_lift'):
        grp[col] = grp[col].round(6)
    return grp


# ===========================================================================
# SECTION 3 -- Population-level fairness (dataset CSV)
# ===========================================================================

def compute_population_fairness(dataset_path: Path,
                                 sensitive_attrs: list[str],
                                 privileged_values: dict[str, str],
                                 outcome_label: str,
                                 positive_outcome: str) -> pd.DataFrame:
    try:
        ds = pd.read_csv(dataset_path, dtype=str)
    except Exception as exc:
        print(f'  [fairness] WARNING -- could not load dataset {dataset_path}: {exc}')
        return pd.DataFrame()
    ds.columns = [c.strip() for c in ds.columns]
    if outcome_label not in ds.columns:
        print(f'  [fairness] WARNING -- outcome column "{outcome_label}" not in dataset.')
        return pd.DataFrame()
    outcome = ds[outcome_label].astype(str)
    records = []
    for attr in sensitive_attrs:
        if attr not in ds.columns:
            print(f'  [fairness] WARNING -- attribute "{attr}" not in dataset; skipped.')
            continue
        col      = ds[attr].astype(str)
        priv_val = privileged_values.get(attr)
        gs: dict[str, dict] = {}
        for val in sorted(col.unique()):
            mask  = col == val
            n     = int(mask.sum())
            n_pos = int((outcome[mask] == str(positive_outcome)).sum())
            gs[val] = {'n': n, 'n_positive': n_pos,
                       'rate': n_pos / n if n else np.nan}
        priv_rate = gs.get(priv_val, {}).get('rate', np.nan)
        for val, g in sorted(gs.items()):
            is_priv = val == priv_val
            rate    = g['rate']
            if is_priv:
                dir_val, spd = 1.0, 0.0
            elif pd.notna(priv_rate) and priv_rate > 0:
                dir_val = rate / priv_rate
                spd     = rate - priv_rate
            else:
                dir_val = spd = np.nan
            records.append({
                'attribute':                     attr,
                'value':                         val,
                'is_privileged':                 is_priv,
                'n_samples':                     g['n'],
                'n_positive':                    g['n_positive'],
                'positive_rate':                 round(rate,    6) if pd.notna(rate)    else np.nan,
                'disparate_impact_ratio':        round(dir_val, 6) if pd.notna(dir_val) else np.nan,
                'statistical_parity_difference': round(spd,     6) if pd.notna(spd)     else np.nan,
                'dir_threshold':                 DIR_THRESHOLD,
                'fairness_violation':            bool(not is_priv and pd.notna(dir_val)
                                                      and dir_val < DIR_THRESHOLD),
            })
    return pd.DataFrame(records)


# ===========================================================================
# SECTION 4 -- Aggregation across configs / k values
# ===========================================================================

def aggregate_dir_results(
    dir_frames: list[pd.DataFrame],
    level_col: str,
    level_values: list,
) -> dict[str, pd.DataFrame]:
    """
    Aggregate a list of disparate_impact DataFrames.

    Returns a dict with keys:
      'long'  -- one row per (frame, group) with the level_col coordinate
      'agg'   -- one row per group with mean/std/min/max DIR across frames,
                 n_configs, n_violations, ever_violates, always_violates
    """
    if not dir_frames:
        return {'long': pd.DataFrame(), 'agg': pd.DataFrame()}
    long_parts = []
    for frame, lv in zip(dir_frames, level_values):
        if frame.empty:
            continue
        f             = frame.copy()
        f[level_col]  = lv
        long_parts.append(f)
    if not long_parts:
        return {'long': pd.DataFrame(), 'agg': pd.DataFrame()}
    long = pd.concat(long_parts, ignore_index=True)
    agg  = (
        long.groupby(['attribute', 'privileged_value', 'unprivileged_value'])
        .agg(
            n_configs    = ('disparate_impact_ratio', 'count'),
            mean_dir     = ('disparate_impact_ratio', 'mean'),
            std_dir      = ('disparate_impact_ratio', 'std'),
            min_dir      = ('disparate_impact_ratio', 'min'),
            max_dir      = ('disparate_impact_ratio', 'max'),
            mean_spd     = ('statistical_parity_difference', 'mean'),
            std_spd      = ('statistical_parity_difference', 'std'),
            n_violations = ('fairness_violation', 'sum'),
        )
        .reset_index()
    )
    for col in ('mean_dir', 'std_dir', 'min_dir', 'max_dir', 'mean_spd', 'std_spd'):
        agg[col] = agg[col].round(6)
    agg['always_violates'] = agg['n_violations'] == agg['n_configs']
    agg['ever_violates']   = agg['n_violations'] > 0
    return {'long': long, 'agg': agg}


def aggregate_coverage_results(coverage_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate coverage DataFrames: sum rule counts, average coverage_pct."""
    if not coverage_frames:
        return pd.DataFrame()
    combined = pd.concat([f for f in coverage_frames if not f.empty], ignore_index=True)
    if combined.empty:
        return pd.DataFrame()
    return (
        combined.groupby(['attribute', 'value'])
        .agg(total_rule_count  = ('rule_count',   'sum'),
             mean_coverage_pct = ('coverage_pct', 'mean'),
             n_configs         = ('rule_count',   'count'))
        .reset_index()
        .assign(mean_coverage_pct=lambda d: d['mean_coverage_pct'].round(4))
    )


# ===========================================================================
# SECTION 5 -- Plots
# ===========================================================================

def _bar_colors(values: list[str], privileged: str) -> list[str]:
    return [_PALETTE['blue'] if v == privileged else _PALETTE['orange'] for v in values]


def plot_coverage_barplot(coverage_df: pd.DataFrame,
                           privileged_values: dict[str, str],
                           output_dir: Path) -> None:
    if coverage_df.empty:
        return
    val_col = 'value' if 'value' in coverage_df.columns else coverage_df.columns[1]
    pct_col = 'coverage_pct' if 'coverage_pct' in coverage_df.columns else 'mean_coverage_pct'
    cnt_col = 'rule_count'   if 'rule_count'   in coverage_df.columns else 'total_rule_count'
    for attr in coverage_df['attribute'].unique():
        sub  = coverage_df[coverage_df['attribute'] == attr].sort_values(pct_col)
        priv = privileged_values.get(attr, '')
        fig, ax = plt.subplots(figsize=(9, max(3, len(sub) * 0.55)))
        bars = ax.barh(sub[val_col], sub[pct_col],
                       color=_bar_colors(sub[val_col].tolist(), priv),
                       edgecolor='white', linewidth=0.5)
        for bar, (_, row) in zip(bars, sub.iterrows()):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f'{bar.get_width():.1f}%  (n={int(row[cnt_col])})',
                    va='center', ha='left', fontsize=8, color='#444444')
        ax.set_xlabel('Rules containing group value (%)', fontsize=10)
        ax.set_title(f'Rule Coverage by {attr}  (blue = privileged: {priv})', fontsize=11)
        ax.set_xlim(0, sub[pct_col].max() * 1.30 + 1)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        fig.savefig(output_dir / f'coverage_barplot_{attr}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_dir_barplot(dir_df: pd.DataFrame, output_dir: Path,
                     title_suffix: str = '') -> None:
    if dir_df.empty:
        return
    unpriv_col = 'unprivileged_value' if 'unprivileged_value' in dir_df.columns else dir_df.columns[2]
    dir_col    = 'disparate_impact_ratio' if 'disparate_impact_ratio' in dir_df.columns else 'mean_dir'
    viol_col   = 'fairness_violation'     if 'fairness_violation'     in dir_df.columns else 'ever_violates'
    for attr in dir_df['attribute'].unique():
        sub = dir_df[dir_df['attribute'] == attr].sort_values(dir_col)
        fig, ax = plt.subplots(figsize=(9, max(3, len(sub) * 0.6 + 1)))
        colors = [_PALETTE['red'] if row[viol_col] else _PALETTE['green']
                  for _, row in sub.iterrows()]
        bars = ax.barh(sub[unpriv_col], sub[dir_col], color=colors, edgecolor='white')
        ax.axvline(DIR_THRESHOLD, color='black', linestyle='--', linewidth=1.2,
                   label=f'4/5 threshold ({DIR_THRESHOLD})')
        ax.axvline(1.0, color=_PALETTE['blue'], linestyle=':', linewidth=1.0,
                   label='Privileged (1.0)')
        for bar, (_, row) in zip(bars, sub.iterrows()):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{row[dir_col]:.3f}", va='center', ha='left', fontsize=8)
        priv_val = dir_df[dir_df['attribute'] == attr]['privileged_value'].iloc[0]
        ax.set_xlabel('Disparate Impact Ratio', fontsize=10)
        ax.set_title(f'{attr}: DIR vs {priv_val}  {title_suffix}\n'
                     '(red = violation, green = fair)', fontsize=10, pad=8)
        ax.set_xlim(0, max(1.1, sub[dir_col].max() * 1.15 + 0.05))
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        fig.savefig(output_dir / f'disparate_impact_barplot_{attr}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_confidence_boxplot(rules_df: pd.DataFrame,
                             sensitive_attrs: list[str],
                             privileged_values: dict[str, str],
                             output_dir: Path) -> None:
    if 'confidence_raw' not in rules_df.columns:
        return
    for attr in sensitive_attrs:
        grp: dict[str, list[float]] = {}
        for _, row in rules_df.iterrows():
            c = row['confidence_raw']
            if pd.isna(c):
                continue
            for item in row['ant_items']:
                lv = extract_label_value(item)
                if lv and lv[0] == attr:
                    grp.setdefault(lv[1], []).append(float(c))
        if not grp:
            continue
        priv   = privileged_values.get(attr, '')
        labels = sorted(grp.keys(), key=lambda v: (v != priv, v))
        data   = [grp[l] for l in labels]
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color='black', linewidth=1.5))
        for patch, color in zip(bp['boxes'], _bar_colors(labels, priv)):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Rule Confidence', fontsize=10)
        ax.set_title(f'Confidence Distribution by {attr}  (blue = {priv})', fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        fig.savefig(output_dir / f'confidence_boxplot_{attr}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_lift_disparity_barplot(lift_df: pd.DataFrame,
                                 privileged_values: dict[str, str],
                                 output_dir: Path) -> None:
    if lift_df.empty:
        return
    for attr in lift_df['attribute'].unique():
        sub  = lift_df[lift_df['attribute'] == attr].sort_values('mean_lift')
        priv = privileged_values.get(attr, '')
        fig, ax = plt.subplots(figsize=(9, max(3, len(sub) * 0.55)))
        bars = ax.barh(sub['value'], sub['mean_lift'],
                       color=_bar_colors(sub['value'].tolist(), priv),
                       edgecolor='white', linewidth=0.5)
        ax.axvline(1.0, color='black', linestyle='--', linewidth=1,
                   label='Lift = 1.0 (independence)')
        for bar, (_, row) in zip(bars, sub.iterrows()):
            lbl = f"{row['mean_lift']:.3f}"
            p   = row.get('pval_vs_privileged', np.nan)
            if pd.notna(p):
                lbl += '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    lbl, va='center', ha='left', fontsize=8)
        ax.set_xlabel('Mean Rule Lift', fontsize=10)
        ax.set_title(f'Mean Lift by {attr}  (* p<.05 vs {priv})', fontsize=10)
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        fig.savefig(output_dir / f'lift_disparity_barplot_{attr}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_intersectional_heatmap(intersect_df: pd.DataFrame,
                                 attr1: str, attr2: str,
                                 output_dir: Path) -> None:
    if intersect_df.empty or attr1 not in intersect_df.columns:
        return
    pivot = intersect_df.pivot_table(index=attr1, columns=attr2,
                                      values='mean_confidence', aggfunc='mean')
    if pivot.empty:
        return
    vals = pivot.values[~np.isnan(pivot.values)]
    vmax = vals.max() if len(vals) else 1
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.3),
                                    max(4, len(pivot.index) * 0.7)))
    img = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel(attr2, fontsize=10)
    ax.set_ylabel(attr1, fontsize=10)
    ax.set_title(f'Mean Rule Confidence -- {attr1} x {attr2}\n(intersectional)',
                 fontsize=11, pad=10)
    for ri in range(len(pivot.index)):
        for ci in range(len(pivot.columns)):
            v = pivot.values[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f'{v:.3f}', ha='center', va='center', fontsize=7)
    plt.colorbar(img, ax=ax, fraction=0.025, pad=0.02).set_label('Mean Confidence', fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / f'intersectional_heatmap_{attr1}_{attr2}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_population_fairness(pop_df: pd.DataFrame,
                              privileged_values: dict[str, str],
                              output_dir: Path) -> None:
    if pop_df.empty:
        return
    for attr in pop_df['attribute'].unique():
        sub  = pop_df[pop_df['attribute'] == attr].sort_values('positive_rate')
        priv = privileged_values.get(attr, '')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(3, len(sub) * 0.55 + 1)))
        c1 = _bar_colors(sub['value'].tolist(), priv)
        b1 = ax1.barh(sub['value'], sub['positive_rate'] * 100,
                      color=c1, edgecolor='white')
        for bar, (_, row) in zip(b1, sub.iterrows()):
            ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     f"{row['positive_rate']*100:.1f}%  (n={row['n_samples']:,})",
                     va='center', ha='left', fontsize=8)
        ax1.set_xlabel('Positive Outcome Rate (%)', fontsize=10)
        ax1.set_title(f'{attr}: Positive Rate by Group', fontsize=10)
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.set_xlim(0, sub['positive_rate'].max() * 130)
        ds = sub[sub['value'] != priv].sort_values('disparate_impact_ratio')
        if not ds.empty:
            c2  = [_PALETTE['red'] if r['fairness_violation'] else _PALETTE['green']
                   for _, r in ds.iterrows()]
            b2  = ax2.barh(ds['value'], ds['disparate_impact_ratio'],
                           color=c2, edgecolor='white')
            ax2.axvline(DIR_THRESHOLD, color='black', linestyle='--', linewidth=1.2,
                        label=f'4/5 ({DIR_THRESHOLD})')
            ax2.axvline(1.0, color=_PALETTE['blue'], linestyle=':', linewidth=1.0,
                        label='Privileged (1.0)')
            for bar, (_, row) in zip(b2, ds.iterrows()):
                ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                         f"{row['disparate_impact_ratio']:.3f}",
                         va='center', ha='left', fontsize=8)
            ax2.set_xlabel('Disparate Impact Ratio', fontsize=10)
            ax2.set_title(f'{attr}: Dataset-Level DIR (red = violation)', fontsize=10)
            ax2.legend(fontsize=8)
            ax2.set_xlim(0, max(1.15, ds['disparate_impact_ratio'].max() * 1.15))
            ax2.spines[['top', 'right']].set_visible(False)
        fig.suptitle(f'Population-Level Fairness -- {attr}', fontsize=12, y=1.02)
        plt.tight_layout()
        fig.savefig(output_dir / f'population_fairness_{attr}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_dir_heatmap_sup_conf(long_dir_df: pd.DataFrame,
                               attr: str,
                               unprivileged_value: str,
                               output_dir: Path,
                               title_prefix: str = '') -> None:
    """
    Heatmap of DIR values across the sup x conf grid for one (attr, group) pair.
    Rows = confidence threshold, columns = support threshold.
    """
    sub = long_dir_df[
        (long_dir_df['attribute'] == attr) &
        (long_dir_df['unprivileged_value'] == unprivileged_value)
    ].copy()
    if sub.empty:
        return
    if 'config' in sub.columns:
        sub['sup_val']  = sub['config'].str.extract(r'sup_([\d.]+)',  expand=False).astype(float)
        sub['conf_val'] = sub['config'].str.extract(r'conf_([\d.]+)', expand=False).astype(float)
    elif 'support_thresh' in sub.columns:
        sub['sup_val']  = pd.to_numeric(sub['support_thresh'],    errors='coerce')
        sub['conf_val'] = pd.to_numeric(sub['confidence_thresh'], errors='coerce')
    else:
        return
    sub   = sub.dropna(subset=['sup_val', 'conf_val', 'disparate_impact_ratio'])
    if sub.empty:
        return
    pivot = (sub.pivot_table(index='conf_val', columns='sup_val',
                              values='disparate_impact_ratio', aggfunc='mean')
               .sort_index(ascending=False))
    n_rows, n_cols = pivot.shape
    fig, ax = plt.subplots(figsize=(max(6, n_cols * 0.8), max(4, n_rows * 0.6)))
    img = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0.4, vmax=1.2)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f'{v:.2f}' for v in pivot.columns], rotation=40,
                        ha='right', fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f'{v:.2f}' for v in pivot.index], fontsize=8)
    ax.set_xlabel('Support Threshold', fontsize=10)
    ax.set_ylabel('Confidence Threshold', fontsize=10)
    ax.set_title(f'{title_prefix}{attr}: DIR for "{unprivileged_value}"\n'
                 f'sup x conf grid  (green >= {DIR_THRESHOLD}, red = violation)',
                 fontsize=10, pad=10)
    for ri in range(n_rows):
        for ci in range(n_cols):
            v = pivot.values[ri, ci]
            if not np.isnan(v):
                color = 'black' if 0.55 < v < 1.1 else 'white'
                ax.text(ci, ri, f'{v:.3f}', ha='center', va='center',
                        fontsize=7, color=color)
    plt.colorbar(img, ax=ax, fraction=0.025, pad=0.02).set_label('DIR', fontsize=9)
    plt.tight_layout()
    safe = re.sub(r'[^\w\-]', '_', unprivileged_value)
    fig.savefig(output_dir / f'dir_heatmap_sup_conf_{attr}_{safe}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_dir_evolution_k(agg_dir_df: pd.DataFrame,
                          output_dir: Path,
                          title_prefix: str = '') -> None:
    """
    Line + scatter plot of mean DIR across k values for every (attr, group).
    Error bars = std across configs within each k.
    One figure per attribute.
    """
    if agg_dir_df.empty or 'k' not in agg_dir_df.columns:
        return
    for attr in agg_dir_df['attribute'].unique():
        sub    = agg_dir_df[agg_dir_df['attribute'] == attr]
        groups = sorted(sub['unprivileged_value'].unique())
        fig, ax = plt.subplots(figsize=(9, 5))
        cmap = (matplotlib.colormaps.get_cmap('tab10')
                if hasattr(matplotlib, 'colormaps')
                else plt.cm.get_cmap('tab10'))
        for gi, grp_val in enumerate(groups):
            g = sub[sub['unprivileged_value'] == grp_val].sort_values('k')
            ax.errorbar(g['k'], g['mean_dir'],
                        yerr=g['std_dir'].fillna(0),
                        marker='o', linewidth=1.5, capsize=4,
                        color=cmap(gi), label=grp_val)
        ax.axhline(DIR_THRESHOLD, color='black', linestyle='--', linewidth=1.2,
                   label=f'4/5 threshold ({DIR_THRESHOLD})')
        ax.axhline(1.0, color=_PALETTE['blue'], linestyle=':', linewidth=1.0,
                   label='Privileged (1.0)')
        ax.set_xlabel('k (CF neighbourhood size)', fontsize=10)
        ax.set_ylabel('Mean DIR (+/- std across configs)', fontsize=10)
        ax.set_title(f'{title_prefix}{attr}: DIR evolution across k\n'
                     '(error bars = std across sup/conf configs)',
                     fontsize=11, pad=10)
        ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        fig.savefig(output_dir / f'global_dir_evolution_{attr}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_global_k_heatmap(k_agg_frames: dict[int, pd.DataFrame],
                           attr: str,
                           output_dir: Path) -> None:
    """Heatmap: rows = k values, columns = unprivileged groups, cells = mean DIR."""
    records = []
    for k_val, df in sorted(k_agg_frames.items()):
        if df.empty:
            continue
        sub = df[df['attribute'] == attr]
        for _, row in sub.iterrows():
            records.append({'k': k_val,
                             'unprivileged_value': row['unprivileged_value'],
                             'mean_dir': row['mean_dir']})
    if not records:
        return
    pivot = (pd.DataFrame(records)
               .pivot_table(index='k', columns='unprivileged_value',
                             values='mean_dir', aggfunc='mean')
               .sort_index(ascending=False))
    n_rows, n_cols = pivot.shape
    fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.4), max(3, n_rows * 0.7)))
    img = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0.4, vmax=1.2)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pivot.columns, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f'k={v}' for v in pivot.index], fontsize=9)
    ax.set_xlabel('Unprivileged Group', fontsize=10)
    ax.set_title(f'{attr}: Mean DIR across k values\n'
                 f'(green >= {DIR_THRESHOLD}, red = violation)',
                 fontsize=11, pad=10)
    for ri in range(n_rows):
        for ci in range(n_cols):
            v = pivot.values[ri, ci]
            if not np.isnan(v):
                color = 'black' if 0.55 < v < 1.1 else 'white'
                ax.text(ci, ri, f'{v:.3f}', ha='center', va='center',
                        fontsize=8, color=color)
    plt.colorbar(img, ax=ax, fraction=0.025, pad=0.02).set_label('Mean DIR', fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / f'global_k_heatmap_{attr}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# SECTION 6 -- Text reports
# ===========================================================================

def _fmt_dir(v: float) -> str:
    if pd.isna(v):
        return 'N/A'
    flag = '  VIOLATION' if v < DIR_THRESHOLD else ''
    return f'{v:.4f}{flag}'


def _write_report(
    path: Path,
    title: str,
    source_desc: str,
    n_rules: int,
    sensitive_attrs: list[str],
    privileged_values: dict[str, str],
    outcome_label: str,
    positive_outcome: str,
    coverage_df: pd.DataFrame,
    conf_parity_df: pd.DataFrame,
    dir_df: pd.DataFrame,
    lift_df: pd.DataFrame,
    intersect_df: pd.DataFrame,
    pop_df: pd.DataFrame,
    extra_sections: list[str] | None = None,
) -> None:
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    hr = '=' * 72
    lines: list[str] = [
        hr, f'FAIRNESS ANALYSIS REPORT -- {title.upper()}', hr,
        f'Generated  : {ts}',
        f'Source     : {source_desc}',
        f'Rules      : {n_rules:,}',
        f'Attributes : {", ".join(sensitive_attrs)}',
        f'Privileged : {privileged_values}',
        f'Outcome    : {outcome_label}={positive_outcome}',
        f'DIR thresh : {DIR_THRESHOLD}  (4/5 rule)',
        '',
    ]

    # Coverage
    lines += [hr, '1. RULE COVERAGE', hr, '']
    val_col = 'value'
    pct_col = 'coverage_pct' if not coverage_df.empty and 'coverage_pct' in coverage_df.columns else 'mean_coverage_pct'
    cnt_col = 'rule_count'   if not coverage_df.empty and 'rule_count'   in coverage_df.columns else 'total_rule_count'
    if coverage_df.empty:
        lines.append('  No sensitive-attribute values found in rules.')
    else:
        for attr in sensitive_attrs:
            sub = coverage_df[coverage_df['attribute'] == attr]
            if sub.empty:
                continue
            lines.append(f'  {attr}:')
            for _, row in sub.sort_values(pct_col, ascending=False).iterrows():
                tag = ' [privileged]' if row.get(val_col, row.get('value')) == privileged_values.get(attr) else ''
                v   = row.get(val_col, row.get('value', ''))
                lines.append(f'    {str(v):<35}{tag}  '
                              f'{int(row[cnt_col]):>5} rules  ({row[pct_col]:.2f}%)')
            lines.append('')

    # Confidence parity
    lines += [hr, '2. CONFIDENCE PARITY', hr, '']
    if conf_parity_df.empty:
        lines.append('  Insufficient data.')
    else:
        for attr in sensitive_attrs:
            sub = conf_parity_df[conf_parity_df['attribute'] == attr]
            if sub.empty:
                continue
            fb   = sub['fallback_all_rules'].any() if 'fallback_all_rules' in sub.columns else False
            note = ' [fallback: all rules]' if fb else ''
            lines.append(f'  {attr}{note}:')
            for _, row in sub.sort_values('mean_confidence', ascending=False).iterrows():
                tag = ' [privileged]' if row['value'] == privileged_values.get(attr) else ''
                lines.append(f'    {str(row["value"]):<35}{tag}  '
                              f'mean={row["mean_confidence"]:.4f}  '
                              f'med={row["median_confidence"]:.4f}  (n={row["n_rules"]})')
            lines.append('')

    # DIR
    lines += [hr, '3. DISPARATE IMPACT RATIO  (4/5 rule)', hr, '']
    if dir_df.empty:
        lines.append('  No DIR computed (insufficient group coverage).')
    else:
        unpriv_col = 'unprivileged_value' if 'unprivileged_value' in dir_df.columns else dir_df.columns[2]
        dir_col    = 'disparate_impact_ratio' if 'disparate_impact_ratio' in dir_df.columns else 'mean_dir'
        viol_col   = 'fairness_violation'     if 'fairness_violation'     in dir_df.columns else 'ever_violates'
        viol_n     = int(dir_df[viol_col].sum())
        lines.append(f'  Pairs tested : {len(dir_df)}    Violations : {viol_n}')
        lines.append('')
        for attr in sensitive_attrs:
            sub = dir_df[dir_df['attribute'] == attr]
            if sub.empty:
                continue
            priv = sub['privileged_value'].iloc[0]
            lines.append(f'  {attr}  (privileged: {priv}):')
            for _, row in sub.sort_values(dir_col).iterrows():
                spd = row.get('statistical_parity_difference', row.get('mean_spd', np.nan))
                lines.append(f'    vs {str(row[unpriv_col]):<33}  '
                              f'DIR={_fmt_dir(row[dir_col])}  '
                              f'SPD={spd:+.4f}')
            lines.append('')

    # Lift
    lines += [hr, '4. LIFT DISPARITY', hr, '']
    if lift_df.empty:
        lines.append('  No lift data.')
    else:
        for attr in sensitive_attrs:
            sub = lift_df[lift_df['attribute'] == attr]
            if sub.empty:
                continue
            lines.append(f'  {attr}:')
            for _, row in sub.sort_values('mean_lift', ascending=False).iterrows():
                tag  = ' [privileged]' if row['value'] == privileged_values.get(attr) else ''
                pstr = (f'  p={row["pval_vs_privileged"]:.4f}'
                        if pd.notna(row.get('pval_vs_privileged')) else '')
                lines.append(f'    {str(row["value"]):<35}{tag}  '
                              f'mean_lift={row["mean_lift"]:.4f}{pstr}')
            lines.append('')

    # Intersectional
    lines += [hr, '5. INTERSECTIONAL ANALYSIS', hr, '']
    if intersect_df.empty:
        lines.append('  No intersectional co-occurrences found.')
    else:
        c0, c1 = intersect_df.columns[0], intersect_df.columns[1]
        lines.append(f'  {len(intersect_df)} ({c0}, {c1}) group combinations in rules.\n')
        lines.append(f'  {c0:<25}  {c1:<30}  {"n_rules":>7}  {"mean_conf":>9}  {"mean_lift":>9}')
        lines.append('  ' + '-' * 85)
        for _, row in intersect_df.sort_values('mean_confidence', ascending=False).iterrows():
            lines.append(f'  {str(row[c0]):<25}  {str(row[c1]):<30}  '
                         f'{int(row["n_rules"]):>7}  '
                         f'{row["mean_confidence"]:>9.4f}  '
                         f'{row["mean_lift"]:>9.4f}')
        lines.append('')

    # Population
    lines += [hr, '6. POPULATION-LEVEL FAIRNESS', hr, '']
    if pop_df.empty:
        lines.append('  No dataset provided.')
    else:
        pv = int(pop_df['fairness_violation'].sum())
        lines.append(f'  Violations (DIR < {DIR_THRESHOLD}) : {pv}')
        lines.append('')
        for attr in sensitive_attrs:
            sub = pop_df[pop_df['attribute'] == attr]
            if sub.empty:
                continue
            lines.append(f'  {attr}:')
            for _, row in sub.sort_values('positive_rate', ascending=False).iterrows():
                tag     = ' [privileged]' if row['value'] == privileged_values.get(attr) else ''
                dir_str = (f'DIR={_fmt_dir(row["disparate_impact_ratio"])}  '
                           if not row['is_privileged'] else '')
                lines.append(f'    {str(row["value"]):<35}{tag}  '
                              f'rate={row["positive_rate"]:.4f}  {dir_str}'
                              f'(n={row["n_samples"]:,})')
            lines.append('')

    # Extra sections (aggregation summaries)
    if extra_sections:
        for section in extra_sections:
            lines.append(section)

    lines += [hr, 'END OF REPORT', hr]
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'  [fairness] Report written -> {path}')


def _aggregation_section(agg_dir_df: pd.DataFrame,
                          level_label: str,
                          sensitive_attrs: list[str]) -> str:
    hr    = '=' * 72
    lines = [hr, f'7. DIR AGGREGATION  ({level_label})', hr, '']
    if agg_dir_df.empty:
        lines.append('  No aggregated DIR data available.')
        return '\n'.join(lines)
    unpriv_col = 'unprivileged_value' if 'unprivileged_value' in agg_dir_df.columns else agg_dir_df.columns[2]
    for attr in sensitive_attrs:
        sub = agg_dir_df[agg_dir_df['attribute'] == attr]
        if sub.empty:
            continue
        lines.append(f'  {attr}:')
        lines.append(f'  {"Group":<35}  {"mean_DIR":>8}  {"std":>6}  '
                     f'{"min":>6}  {"max":>6}  {"n_configs":>9}  {"violations":>10}')
        lines.append('  ' + '-' * 90)
        for _, row in sub.sort_values('mean_dir').iterrows():
            ever = bool(row.get('ever_violates', row.get('n_violations', 0) > 0))
            flag = '  !!' if ever else ''
            lines.append(f'  {str(row[unpriv_col]):<35}  '
                         f'{row["mean_dir"]:>8.4f}  '
                         f'{row["std_dir"]:>6.4f}  '
                         f'{row["min_dir"]:>6.4f}  '
                         f'{row["max_dir"]:>6.4f}  '
                         f'{int(row.get("n_configs", row.get("n_configs_total", 0))):>9}  '
                         f'{int(row["n_violations"]):>10}{flag}')
        lines.append('')
    return '\n'.join(lines)


# ===========================================================================
# SECTION 7 -- Single-config analysis
# ===========================================================================

def analyse_single_config(
    rules_path: Path,
    output_dir: Path,
    sensitive_attrs: list[str],
    privileged_values: dict[str, str],
    outcome_label: str,
    positive_outcome: str,
    pop_df: pd.DataFrame,
    config_label: str = '',
    rules_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Full fairness analysis for one rules.csv.

    Parameters
    ----------
    rules_path : Path
        Path used as a label in the report.  Also loaded if *rules_df* is None.
    rules_df : pd.DataFrame or None
        Pre-loaded rules DataFrame.  When provided, *rules_path* is not read
        from disk (used only for the report label).  Pass this when the caller
        has already concatenated multiple files.

    Returns a dict of DataFrames keyed by metric name, or {} on error.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'plots').mkdir(exist_ok=True)

    if rules_df is None:
        try:
            rules_df = load_rules(rules_path)
        except Exception as exc:
            print(f'  [fairness] Could not load {rules_path}: {exc}')
            return {}
    if rules_df.empty:
        return {}

    coverage_df       = compute_rule_coverage(rules_df, sensitive_attrs)
    conf_parity_df    = compute_confidence_parity(rules_df, sensitive_attrs,
                                                   privileged_values, outcome_label,
                                                   positive_outcome)
    support_parity_df = compute_support_parity(rules_df, sensitive_attrs, privileged_values)
    lift_df           = compute_lift_disparity(rules_df, sensitive_attrs, privileged_values)
    dir_df            = compute_disparate_impact(conf_parity_df, privileged_values)
    intersect_df      = pd.DataFrame()
    if len(sensitive_attrs) >= 2:
        intersect_df  = compute_intersectional_analysis(
            rules_df, sensitive_attrs[0], sensitive_attrs[1],
            outcome_label, positive_outcome)

    def _save(df: pd.DataFrame, name: str) -> None:
        if not df.empty:
            df.to_csv(output_dir / name, index=False)

    _save(coverage_df,       'rule_coverage.csv')
    _save(conf_parity_df,    'confidence_parity.csv')
    _save(support_parity_df, 'support_parity.csv')
    _save(lift_df,           'lift_disparity.csv')
    _save(dir_df,            'disparate_impact.csv')
    _save(intersect_df,      'intersectional_analysis.csv')

    # Summary CSV
    parts = []
    for df_name, df, grp_col, extra_cols in [
        ('disparate_impact', dir_df, 'unprivileged_value',
         ['disparate_impact_ratio', 'statistical_parity_difference', 'fairness_violation']),
        ('rule_coverage', coverage_df, 'value', ['rule_count', 'coverage_pct']),
        ('lift_disparity', lift_df, 'value', ['mean_lift', 'pval_vs_privileged']),
    ]:
        if not df.empty and grp_col in df.columns:
            d = df[['attribute', grp_col] + [c for c in extra_cols if c in df.columns]].copy()
            d.insert(0, 'metric_type', df_name)
            parts.append(d.rename(columns={grp_col: 'group_value'}))
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(output_dir / 'fairness_metrics.csv',
                                                    index=False)

    plots_dir = output_dir / 'plots'
    plot_coverage_barplot(coverage_df, privileged_values, plots_dir)
    plot_dir_barplot(dir_df, plots_dir, title_suffix=config_label)
    plot_confidence_boxplot(rules_df, sensitive_attrs, privileged_values, plots_dir)
    plot_lift_disparity_barplot(lift_df, privileged_values, plots_dir)
    if not intersect_df.empty and len(sensitive_attrs) >= 2:
        plot_intersectional_heatmap(intersect_df, sensitive_attrs[0],
                                     sensitive_attrs[1], plots_dir)

    _write_report(
        path=output_dir / 'fairness_report.txt',
        title=config_label or str(rules_path),
        source_desc=str(rules_path),
        n_rules=len(rules_df),
        sensitive_attrs=sensitive_attrs,
        privileged_values=privileged_values,
        outcome_label=outcome_label,
        positive_outcome=positive_outcome,
        coverage_df=coverage_df,
        conf_parity_df=conf_parity_df,
        dir_df=dir_df,
        lift_df=lift_df,
        intersect_df=intersect_df,
        pop_df=pop_df,
    )

    return {
        'coverage':          coverage_df,
        'confidence_parity': conf_parity_df,
        'support_parity':    support_parity_df,
        'lift_disparity':    lift_df,
        'disparate_impact':  dir_df,
        'intersectional':    intersect_df,
    }


# ===========================================================================
# SECTION 8 -- Per-k aggregation
# ===========================================================================

def analyse_k_level(
    k_val: int,
    config_results: dict[tuple[str, str], dict[str, pd.DataFrame]],
    output_dir: Path,
    sensitive_attrs: list[str],
    privileged_values: dict[str, str],
    outcome_label: str,
    positive_outcome: str,
    pop_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Aggregate per-config results for one k value and write the k-level report.

    Parameters
    ----------
    config_results : dict mapping (sup_str, conf_str) -> metric dict
    output_dir : k-level directory, e.g. fairness_analysis/{exp}/k_{k}/

    Returns
    -------
    dict with keys 'agg_dir', 'agg_coverage', 'long_dir'.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    dir_frames      : list[pd.DataFrame] = []
    coverage_frames : list[pd.DataFrame] = []
    level_values    : list[str]           = []
    long_dir_parts  : list[pd.DataFrame] = []

    for (sup_str, conf_str), res in sorted(config_results.items()):
        if not res:
            continue
        lbl = f'sup_{sup_str}_conf_{conf_str}'

        df = res.get('disparate_impact', pd.DataFrame())
        if not df.empty:
            d2 = df.copy()
            d2['config']            = lbl
            d2['support_thresh']    = sup_str
            d2['confidence_thresh'] = conf_str
            dir_frames.append(df)
            level_values.append(lbl)   # kept aligned with dir_frames
            long_dir_parts.append(d2)

        cf = res.get('coverage', pd.DataFrame())
        if not cf.empty:
            coverage_frames.append(cf)

    agg_result  = aggregate_dir_results(dir_frames, 'config', level_values)
    agg_dir_df  = agg_result['agg']
    long_dir_df = (pd.concat(long_dir_parts, ignore_index=True)
                   if long_dir_parts else pd.DataFrame())
    agg_cov_df  = aggregate_coverage_results(coverage_frames)

    if not agg_dir_df.empty:
        agg_dir_df.to_csv(output_dir / 'k_dir_summary.csv', index=False)
    if not agg_cov_df.empty:
        agg_cov_df.to_csv(output_dir / 'k_coverage_summary.csv', index=False)
    if not long_dir_df.empty:
        long_dir_df.to_csv(output_dir / 'k_dir_all_configs.csv', index=False)

    # Heatmaps sup x conf -> DIR, one per (attr, group)
    if not long_dir_df.empty:
        for attr in sensitive_attrs:
            sub = long_dir_df[long_dir_df['attribute'] == attr]
            for uv in sub['unprivileged_value'].unique():
                plot_dir_heatmap_sup_conf(long_dir_df, attr, uv, plots_dir,
                                           title_prefix=f'k={k_val}  ')
    if not agg_dir_df.empty:
        plot_dir_barplot(agg_dir_df, plots_dir,
                         title_suffix=f'(k={k_val}, mean across {len(config_results)} configs)')

    # Aggregate intersectional
    int_parts = [r.get('intersectional', pd.DataFrame())
                 for r in config_results.values() if r]
    int_parts = [x for x in int_parts if not x.empty]
    agg_int   = pd.DataFrame()
    if int_parts and len(sensitive_attrs) >= 2:
        a0, a1 = sensitive_attrs[0], sensitive_attrs[1]
        agg_int = (
            pd.concat(int_parts, ignore_index=True)
            .groupby([a0, a1])
            .agg(n_rules=('n_rules', 'sum'),
                 mean_confidence=('mean_confidence', 'mean'),
                 mean_support=('mean_support', 'mean'),
                 mean_lift=('mean_lift', 'mean'),
                 n_positive_outcome_rules=('n_positive_outcome_rules', 'sum'))
            .reset_index()
        )
        for col in ('mean_confidence', 'mean_support', 'mean_lift'):
            agg_int[col] = agg_int[col].round(6)
        agg_int.to_csv(output_dir / 'k_intersectional_analysis.csv', index=False)
        plot_intersectional_heatmap(agg_int, a0, a1, plots_dir)

    # Aggregate confidence parity across configs
    cp_parts = [r.get('confidence_parity', pd.DataFrame())
                for r in config_results.values() if r]
    cp_parts = [x for x in cp_parts if not x.empty]
    agg_cp   = pd.DataFrame()
    if cp_parts:
        agg_cp = (
            pd.concat(cp_parts, ignore_index=True)
            .groupby(['attribute', 'value', 'is_privileged'])
            .agg(n_rules=('n_rules', 'sum'),
                 mean_confidence=('mean_confidence', 'mean'),
                 median_confidence=('median_confidence', 'mean'),
                 std_confidence=('std_confidence', 'mean'))
            .reset_index()
        )

    agg_section = _aggregation_section(agg_dir_df, f'k={k_val}', sensitive_attrs)

    _write_report(
        path=output_dir / 'k_fairness_report.txt',
        title=f'k={k_val} -- aggregated across {len(config_results)} configs',
        source_desc=f'k={k_val}, {len(config_results)} sup/conf configurations',
        n_rules=0,
        sensitive_attrs=sensitive_attrs,
        privileged_values=privileged_values,
        outcome_label=outcome_label,
        positive_outcome=positive_outcome,
        coverage_df=agg_cov_df,
        conf_parity_df=agg_cp,
        dir_df=agg_dir_df,
        lift_df=pd.DataFrame(),
        intersect_df=agg_int,
        pop_df=pop_df,
        extra_sections=[agg_section],
    )

    return {'agg_dir': agg_dir_df, 'long_dir': long_dir_df, 'agg_coverage': agg_cov_df}


# ===========================================================================
# SECTION 9 -- Global aggregation
# ===========================================================================

def analyse_global(
    k_agg_results: dict[int, dict[str, pd.DataFrame]],
    output_dir: Path,
    sensitive_attrs: list[str],
    privileged_values: dict[str, str],
    outcome_label: str,
    positive_outcome: str,
    pop_df: pd.DataFrame,
) -> None:
    """
    Aggregate k-level DIR results into the final global summary.

    Writes:
      global_dir_evolution.csv  -- one row per (k, attr, group) with mean_dir
      global_dir_summary.csv    -- one row per (attr, group) aggregated over all k
      global_fairness_report.txt
      plots/global_dir_evolution_{attr}.png
      plots/global_k_heatmap_{attr}.png
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    global_long_parts: list[pd.DataFrame] = []
    for k_val, kres in sorted(k_agg_results.items()):
        adf = kres.get('agg_dir', pd.DataFrame())
        if adf.empty:
            continue
        f      = adf.copy()
        f['k'] = k_val
        global_long_parts.append(f)

    if not global_long_parts:
        print('  [fairness] No k-level DIR data for global aggregation.')
        return

    global_long = pd.concat(global_long_parts, ignore_index=True)
    global_long.to_csv(output_dir / 'global_dir_evolution.csv', index=False)

    global_agg = (
        global_long.groupby(['attribute', 'privileged_value', 'unprivileged_value'])
        .agg(
            n_k_values      = ('k',        'count'),
            mean_dir        = ('mean_dir',  'mean'),
            std_dir         = ('mean_dir',  'std'),
            min_dir         = ('min_dir',   'min'),
            max_dir         = ('max_dir',   'max'),
            mean_spd        = ('mean_spd',  'mean'),
            n_violations    = ('n_violations', 'sum'),
            n_configs_total = ('n_configs',    'sum'),
        )
        .reset_index()
    )
    for col in ('mean_dir', 'std_dir', 'min_dir', 'max_dir', 'mean_spd'):
        global_agg[col] = global_agg[col].round(6)
    global_agg['always_violates'] = global_agg['n_violations'] == global_agg['n_configs_total']
    global_agg['ever_violates']   = global_agg['n_violations'] > 0
    global_agg.to_csv(output_dir / 'global_dir_summary.csv', index=False)

    plot_dir_evolution_k(global_long, plots_dir, title_prefix='GLOBAL  ')
    k_agg_map = {k: kres.get('agg_dir', pd.DataFrame())
                 for k, kres in k_agg_results.items()}
    for attr in sensitive_attrs:
        plot_global_k_heatmap(k_agg_map, attr, plots_dir)

    agg_section = _aggregation_section(global_agg, 'all k values', sensitive_attrs)

    _write_report(
        path=output_dir / 'global_fairness_report.txt',
        title='GLOBAL -- all k values and all configurations',
        source_desc=f'{len(k_agg_results)} k values: {sorted(k_agg_results.keys())}',
        n_rules=0,
        sensitive_attrs=sensitive_attrs,
        privileged_values=privileged_values,
        outcome_label=outcome_label,
        positive_outcome=positive_outcome,
        coverage_df=pd.DataFrame(),
        conf_parity_df=pd.DataFrame(),
        dir_df=global_agg,
        lift_df=pd.DataFrame(),
        intersect_df=pd.DataFrame(),
        pop_df=pop_df,
        extra_sections=[agg_section],
    )
    print(f'  [fairness] Global report -> {output_dir}/global_fairness_report.txt')


# ===========================================================================
# SECTION 10 -- Region-level orchestrator
# ===========================================================================

def analyse_region(
    region: str,
    ar_values_dir: Path,
    fairness_root: Path,
    dataset_paths: list[Path] | None,
    sensitive_attrs: list[str],
    privileged_values: dict[str, str],
    outcome_label: str,
    positive_outcome: str,
    k_values: list[int] | None = None,
) -> None:
    """
    Run the three-level fairness analysis for one region.

    Level 1 -- per-config  (each sup/conf pair inside each k folder)
    Level 2 -- per-k       (aggregation across configs for each k)
    Level 3 -- global      (aggregation across all k values)

    Population-level metrics are computed once from dataset CSVs and reused.
    """
    fairness_root.mkdir(parents=True, exist_ok=True)

    # Population-level (computed once per region)
    pop_df = pd.DataFrame()
    if dataset_paths:
        parts = []
        for dp in dataset_paths:
            pf = compute_population_fairness(dp, sensitive_attrs, privileged_values,
                                              outcome_label, positive_outcome)
            if not pf.empty:
                pf['dataset'] = dp.name
                parts.append(pf)
        if parts:
            pop_df = pd.concat(parts, ignore_index=True)
            pop_df.to_csv(fairness_root / 'population_fairness.csv', index=False)
            pop_plots = fairness_root / 'plots'
            pop_plots.mkdir(exist_ok=True)
            plot_population_fairness(pop_df, privileged_values, pop_plots)

    # Discover rules
    index = discover_rules_structure(ar_values_dir, k_values=k_values)
    if not index:
        print(f'  [fairness] No rules.csv found under {ar_values_dir}; skipping.')
        return

    n_configs = len(index)
    print(f'  [fairness] {n_configs} (k, sup, conf) configurations discovered.')

    # Group by (exp_label, k)
    by_exp_k: dict[tuple[str, int], dict[tuple[str, str], Path]] = defaultdict(dict)
    for (exp_label, k_val, sup_str, conf_str), path in index.items():
        by_exp_k[(exp_label, k_val)][(sup_str, conf_str)] = path

    k_agg_results: dict[int, dict[str, pd.DataFrame]] = {}

    for (exp_label, k_val), configs in sorted(by_exp_k.items()):
        k_dir = fairness_root / exp_label / f'k_{k_val}'
        print(f'\n  [fairness] {region}  exp={exp_label}  k={k_val}'
              f'  ({len(configs)} configs)')

        config_results: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}

        # Level 1: per-config
        for (sup_str, conf_str), rules_path in sorted(configs.items()):
            lbl        = f'k={k_val} sup={sup_str} conf={conf_str}'
            config_dir = k_dir / f'sup_{sup_str}' / f'conf_{conf_str}'
            print(f'    > {lbl}')
            config_results[(sup_str, conf_str)] = analyse_single_config(
                rules_path=rules_path,
                output_dir=config_dir,
                sensitive_attrs=sensitive_attrs,
                privileged_values=privileged_values,
                outcome_label=outcome_label,
                positive_outcome=positive_outcome,
                pop_df=pop_df,
                config_label=lbl,
            )

        # Level 2: per-k
        k_res = analyse_k_level(
            k_val=k_val,
            config_results=config_results,
            output_dir=k_dir,
            sensitive_attrs=sensitive_attrs,
            privileged_values=privileged_values,
            outcome_label=outcome_label,
            positive_outcome=positive_outcome,
            pop_df=pop_df,
        )

        # Accumulate k results (merge if multiple exp_labels share the same k)
        if k_val not in k_agg_results:
            k_agg_results[k_val] = k_res
        else:
            merged = pd.concat(
                [k_agg_results[k_val].get('agg_dir', pd.DataFrame()),
                 k_res.get('agg_dir', pd.DataFrame())],
                ignore_index=True,
            )
            k_agg_results[k_val]['agg_dir'] = merged

    # Level 3: global
    if k_agg_results:
        print(f'\n  [fairness] Building global report for {region} ...')
        analyse_global(
            k_agg_results=k_agg_results,
            output_dir=fairness_root / 'global',
            sensitive_attrs=sensitive_attrs,
            privileged_values=privileged_values,
            outcome_label=outcome_label,
            positive_outcome=positive_outcome,
            pop_df=pop_df,
        )


# ===========================================================================
# SECTION 11 -- Standalone public API
# ===========================================================================

def analyse_rules(
    rules_source: Path,
    output_dir: Path,
    sensitive_attrs: list[str]        = None,
    privileged_values: dict[str, str] = None,
    outcome_label: str                = DEFAULT_OUTCOME_LABEL,
    positive_outcome: str             = DEFAULT_POSITIVE_OUTCOME,
    dataset_paths: list[Path]         = None,
) -> dict[str, pd.DataFrame]:
    """
    Standalone entry point: run the full three-level analysis on any rules
    source (single file or directory tree).

    If *rules_source* is a directory and follows the pipeline layout
    ({exp_label}/k_{k}/sup_{x}/conf_{y}/rules.csv), the three-level
    orchestration is used automatically.  Otherwise falls back to a
    single-config analysis on the concatenated rules.
    """
    if sensitive_attrs   is None:
        sensitive_attrs   = DEFAULT_SENSITIVE_ATTRS
    if privileged_values is None:
        privileged_values = DEFAULT_PRIVILEGED_VALUES

    output_dir.mkdir(parents=True, exist_ok=True)

    if rules_source.is_dir():
        index = discover_rules_structure(rules_source)
        if index:
            analyse_region(
                region='custom',
                ar_values_dir=rules_source,
                fairness_root=output_dir,
                dataset_paths=dataset_paths,
                sensitive_attrs=sensitive_attrs,
                privileged_values=privileged_values,
                outcome_label=outcome_label,
                positive_outcome=positive_outcome,
            )
            return {}

        paths = sorted(rules_source.rglob('rules.csv'))
        if not paths:
            print(f'  [fairness] No rules.csv found under {rules_source}')
            return {}
        frames = []
        for p in paths:
            try:
                frames.append(load_rules(p))
            except Exception:
                pass
        if not frames:
            return {}
        # Concatenate all found rule files and analyse as a single virtual config.
        # rules_path is used only for the report label; the actual data comes from
        # the pre-loaded `rules_df` kwarg passed to analyse_single_config.
        rules_df_combined = pd.concat(frames, ignore_index=True)
        rules_path        = paths[0]          # label reference only
    else:
        rules_df_combined = None
        rules_path        = rules_source

    pop_df = pd.DataFrame()
    if dataset_paths:
        parts = []
        for dp in dataset_paths:
            pf = compute_population_fairness(dp, sensitive_attrs, privileged_values,
                                              outcome_label, positive_outcome)
            if not pf.empty:
                parts.append(pf)
        if parts:
            pop_df = pd.concat(parts, ignore_index=True)

    return analyse_single_config(
        rules_path=rules_path,
        output_dir=output_dir,
        sensitive_attrs=sensitive_attrs,
        privileged_values=privileged_values,
        outcome_label=outcome_label,
        positive_outcome=positive_outcome,
        pop_df=pop_df,
        config_label=str(rules_source),
        rules_df=rules_df_combined,
    )


# ===========================================================================
# SECTION 12 -- Pipeline entry point
# ===========================================================================

def main(
    regions: list[str]                = None,
    k_values: list[int]               = None,
    base_dir: Path                    = None,
    sensitive_attrs: list[str]        = None,
    privileged_values: dict[str, str] = None,
    outcome_label: str                = DEFAULT_OUTCOME_LABEL,
    positive_outcome: str             = DEFAULT_POSITIVE_OUTCOME,
    # Grid parameters -- accepted for signature parity, not used
    auto_calibrate: bool              = True,
    sup_min: float                    = 0.02,
    sup_max: float                    = 0.50,
    sup_delta: float                  = 0.02,
    conf_min: float                   = 0.05,
    conf_max: float                   = 1.00,
    conf_delta: float                 = 0.05,
    lift_min: float                   = 0.0,
    lift_max: float                   = 5.0,
    lift_delta: float                 = 0.05,
    lift_neutral_half_window: float   = 0.25,
) -> None:
    """
    Pipeline entry point -- mirrors the signature of
    microscopic_experiment_association_rules_values.main().

    Runs the three-level fairness analysis (per-config, per-k, global)
    for every requested region.  Grid parameters are accepted for signature
    compatibility but are intentionally unused.
    """
    if base_dir is None:
        if Path('/kaggle/working').exists():
            base_dir = Path('/kaggle/working')
        elif Path('/content').exists():
            base_dir = Path('/content')
        else:
            base_dir = Path(__file__).resolve().parent.parent
    base_dir = Path(base_dir)

    if regions         is None: regions         = ['northeast', 'south']
    if k_values        is None: k_values        = [1, 3, 5, 7]
    if sensitive_attrs is None: sensitive_attrs = DEFAULT_SENSITIVE_ATTRS
    if privileged_values is None: privileged_values = DEFAULT_PRIVILEGED_VALUES

    results_dir = base_dir / 'results'
    data_dir    = base_dir / 'data'

    for region in regions:
        print('\n' + '=' * 70)
        print(f'FAIRNESS ANALYSIS -- {region.upper()}')
        print('=' * 70)

        ar_values_dir = results_dir / region / 'association_rules_values'
        if not ar_values_dir.exists():
            print(f'  [fairness] association_rules_values/ not found for {region}; skipping.')
            continue

        dataset_paths = sorted(data_dir.glob(f'*{region}*.csv')) if data_dir.exists() else []

        analyse_region(
            region=region,
            ar_values_dir=ar_values_dir,
            fairness_root=results_dir / region / 'fairness_analysis',
            dataset_paths=dataset_paths or None,
            sensitive_attrs=sensitive_attrs,
            privileged_values=privileged_values,
            outcome_label=outcome_label,
            positive_outcome=positive_outcome,
            k_values=k_values,
        )

    print('\n' + '=' * 70)
    print('Fairness analysis done.')
    print('=' * 70 + '\n')


# ===========================================================================
# SECTION 13 -- CLI
# ===========================================================================

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Fairness analysis on association rules CSV files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--rules', required=True, type=Path,
                   help='Single rules.csv or directory scanned recursively.')
    p.add_argument('--dataset', type=Path, action='append', dest='datasets',
                   default=[], metavar='PATH',
                   help='Original dataset CSV. Repeatable.')
    p.add_argument('--output_dir', type=Path, default=None,
                   help='Output directory (default: <rules_parent>/fairness_analysis/).')
    p.add_argument('--sensitive_attrs', nargs='+', default=DEFAULT_SENSITIVE_ATTRS,
                   metavar='ATTR')
    p.add_argument('--privileged', nargs='+', default=[], metavar='ATTR=VALUE',
                   help='Privileged values as ATTR=VALUE pairs.')
    p.add_argument('--outcome_label',    default=DEFAULT_OUTCOME_LABEL)
    p.add_argument('--positive_outcome', default=DEFAULT_POSITIVE_OUTCOME)
    return p


if __name__ == '__main__':
    parser = _build_cli()
    args   = parser.parse_args()

    priv_vals = dict(DEFAULT_PRIVILEGED_VALUES)
    for token in args.privileged:
        if '=' in token:
            k, v = token.split('=', 1)
            priv_vals[k.strip()] = v.strip()

    out = args.output_dir or args.rules.parent / 'fairness_analysis'

    analyse_rules(
        rules_source=args.rules,
        output_dir=out,
        sensitive_attrs=args.sensitive_attrs,
        privileged_values=priv_vals,
        outcome_label=args.outcome_label,
        positive_outcome=args.positive_outcome,
        dataset_paths=args.datasets or None,
    )