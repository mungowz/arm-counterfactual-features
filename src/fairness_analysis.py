"""
fairness_analysis_association_rules.py
======================================
Fairness analysis on value-level association rules produced by the pipeline.

Pipeline position
-----------------
    microscopic_experiment_association_rules_values.py  →  [this script]

What this module does
---------------------
For every rules.csv produced by the microscopic step (and optionally the
original ACS dataset), this script computes a comprehensive suite of fairness
metrics centred on sensitive demographic attributes (SEX, RAC1P by default)
and writes reports, CSVs, and plots to a structured output directory.

Metrics computed
----------------
**Rule-level** (from rules.csv alone):
  1. Group coverage — how many rules contain each demographic value
  2. Confidence parity — mean/median rule confidence per group for rules
     whose consequent targets a positive outcome
  3. Support parity — same, for support
  4. Lift disparity — average lift of rules involving each group; Mann-Whitney
     U test vs the privileged group
  5. Disparate Impact Ratio (DIR) — the classic 4/5 rule applied to mean rule
     confidence: DIR = mean_conf(unprivileged) / mean_conf(privileged)
  6. Statistical Parity Difference (SPD) — difference in mean confidence

**Population-level** (requires original dataset CSV):
  7. Base-rate DIR — P(positive | unprivileged) / P(positive | privileged)
  8. Base-rate SPD
  9. Group sample counts and positive rates

**Intersectional** (from rules.csv):
  10. SEX × RAC1P cross-tabulation of rule count, mean confidence, mean support

Output files
------------
    {output_dir}/
        fairness_report.txt                 — human-readable summary
        fairness_metrics.csv                — all metrics in one table
        rule_coverage.csv                   — coverage per group
        confidence_parity.csv               — confidence stats per group
        lift_disparity.csv                  — lift stats per group
        disparate_impact.csv                — DIR + SPD per (attr, group pair)
        intersectional_analysis.csv         — SEX × RAC1P cross-tab
        population_fairness.csv             — dataset-level metrics (if dataset)
        plots/
            coverage_barplot.png
            disparate_impact_barplot.png
            confidence_boxplot.png
            lift_disparity_barplot.png
            intersectional_heatmap.png
            population_fairness_barplot.png (if dataset provided)

Pipeline usage
--------------
    main(regions, k_values, base_dir, ...)

    Mirrors the signature of microscopic_experiment_association_rules_values.main()
    so that main.py can invoke this step with the same parameter set.

Standalone usage
----------------
    python fairness_analysis_association_rules.py --rules path/to/rules.csv
    python fairness_analysis_association_rules.py --rules path/to/dir/ \\
        --dataset path/to/acs_income.csv --output_dir path/to/out/

Generic CSV support
-------------------
    The script makes no assumptions about which LABEL=value items are present.
    Sensitive attributes and their privileged values are configurable.  The
    outcome label is also configurable (default: INCOME_ABOVE_THRESHOLD=1).
"""

from __future__ import annotations

import argparse
import ast
import datetime
import re
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use('Agg')
warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------
# Defaults — all overridable through main() / CLI arguments
# ---------------------------------------------------------------------------

DEFAULT_SENSITIVE_ATTRS: list[str] = ['SEX', 'RAC1P']

# Privileged group for each sensitive attribute.
# The 4/5 disparate-impact rule computes DIR relative to these groups.
DEFAULT_PRIVILEGED_VALUES: dict[str, str] = {
    'SEX':   'Male',
    'RAC1P': 'White-Alone',
}

DEFAULT_OUTCOME_LABEL: str   = 'INCOME_ABOVE_THRESHOLD'
DEFAULT_POSITIVE_OUTCOME: str = '1'

# 4/5 rule: ratios below this are flagged as potential disparate impact
DIR_THRESHOLD: float = 0.80


# ---------------------------------------------------------------------------
# Itemset parsing
# ---------------------------------------------------------------------------

def parse_itemset(cell: str) -> list[str]:
    """
    Parse an itemset cell from various string representations.

    Handles the following formats produced by mlxtend / pandas .to_csv():
      - Plain single item  : ``OCCP=Chief-Executives``
      - Comma-separated    : ``OCCP=Chief-Executives, SCHL=Masters-Degree``
      - frozenset literal  : ``frozenset({'OCCP=Chief-Executives', ...})``
      - set literal        : ``{'OCCP=Chief-Executives'}``
      - tuple literal      : ``('OCCP=Chief-Executives',)``

    Parameters
    ----------
    cell : str
        Raw string value from the antecedents or consequents column.

    Returns
    -------
    list[str]
        List of individual ``LABEL=value`` item strings, stripped of whitespace.
    """
    if not isinstance(cell, str):
        return []
    cell = cell.strip()

    # frozenset({'A', 'B'}) or frozenset({'A'})
    if cell.startswith('frozenset('):
        inner = cell[len('frozenset('):-1].strip()
        try:
            parsed = ast.literal_eval(inner)
            return [str(x).strip() for x in parsed]
        except Exception:
            pass

    # {'A', 'B'} — set literal
    if cell.startswith('{') and cell.endswith('}'):
        try:
            parsed = ast.literal_eval(cell)
            if isinstance(parsed, (set, frozenset)):
                return [str(x).strip() for x in parsed]
        except Exception:
            pass

    # ('A', 'B') or ('A',) — tuple literal
    if cell.startswith('(') and cell.endswith(')'):
        try:
            parsed = ast.literal_eval(cell)
            if isinstance(parsed, tuple):
                return [str(x).strip() for x in parsed]
        except Exception:
            pass

    # Comma-separated (plain or frozenset-escaped)
    return [item.strip() for item in cell.split(',') if item.strip()]


def extract_label_value(item: str) -> tuple[str, str] | None:
    """
    Split a ``LABEL=value`` string into its components.

    Returns ``None`` for any string that does not contain ``=``.

    Parameters
    ----------
    item : str
        A single item string, e.g. ``'SEX=Male'``.

    Returns
    -------
    tuple[str, str] or None
        ``(label, value)`` pair, both stripped of whitespace, or ``None``.
    """
    if '=' not in item:
        return None
    label, _, value = item.partition('=')
    return label.strip(), value.strip()


def load_rules(path: Path) -> pd.DataFrame:
    """
    Load a rules.csv file and enrich it with parsed itemset columns.

    Required columns: ``antecedents``, ``consequents``.
    Numeric columns (``support_raw``, ``confidence_raw``, ``lift``, etc.)
    are coerced to float where present.

    Parameters
    ----------
    path : Path
        Path to the rules.csv file.

    Returns
    -------
    pd.DataFrame
        Original columns plus:
        ``ant_items``  — list of antecedent item strings
        ``con_items``  — list of consequent item strings
        ``all_items``  — concatenation of both lists
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    for required in ('antecedents', 'consequents'):
        if required not in df.columns:
            raise ValueError(f"Column '{required}' missing from {path}")

    df['ant_items'] = df['antecedents'].apply(parse_itemset)
    df['con_items'] = df['consequents'].apply(parse_itemset)
    df['all_items'] = df.apply(lambda r: r['ant_items'] + r['con_items'], axis=1)

    for col in (
        'support_raw', 'support_pct',
        'confidence_raw', 'confidence_pct',
        'lift', 'leverage', 'conviction',
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def collect_rules_from_dir(rules_dir: Path) -> pd.DataFrame:
    """
    Recursively find all rules.csv files under *rules_dir* and concatenate them.

    Path metadata columns ``source_file``, ``region``, ``k``, ``support_thresh``,
    and ``confidence_thresh`` are extracted from the path where possible.

    Parameters
    ----------
    rules_dir : Path
        Root directory to scan (e.g. ``results/northeast/association_rules_values/``).

    Returns
    -------
    pd.DataFrame
        Concatenation of all loaded rule DataFrames with path-derived metadata.
        Returns an empty DataFrame if no rules.csv files are found.
    """
    paths = sorted(rules_dir.rglob('rules.csv'))
    if not paths:
        print(f'  [fairness] No rules.csv files found under {rules_dir}')
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for p in paths:
        try:
            df = load_rules(p)
        except Exception as exc:
            print(f'  [fairness] WARNING — could not load {p}: {exc}')
            continue

        df['source_file'] = str(p)

        # Extract structured metadata from the path
        parts = p.parts
        k_match   = next((x for x in parts if re.fullmatch(r'k_\d+', x)), None)
        sup_match = next((x for x in parts if x.startswith('sup_')), None)
        con_match = next((x for x in parts if x.startswith('conf_')), None)

        # Region: component just before 'association_rules_values'
        try:
            ar_idx = next(i for i, x in enumerate(parts) if 'association_rules' in x)
            df['region'] = parts[ar_idx - 1] if ar_idx > 0 else 'unknown'
        except StopIteration:
            df['region'] = 'unknown'

        df['k']                  = int(k_match.split('_')[1])   if k_match   else np.nan
        df['support_thresh']     = sup_match.replace('sup_', '') if sup_match else ''
        df['confidence_thresh']  = con_match.replace('conf_', '') if con_match else ''

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Rule-level fairness metrics
# ---------------------------------------------------------------------------

def compute_rule_coverage(
    rules_df: pd.DataFrame,
    sensitive_attrs: list[str],
) -> pd.DataFrame:
    """
    Count how many rules contain each sensitive-attribute value, anywhere
    (antecedent or consequent).

    A rule is counted once per unique group value it contains — if both
    ``SEX=Male`` and ``SEX=Female`` somehow appear in the same rule, that
    rule is counted for both groups.

    Parameters
    ----------
    rules_df : pd.DataFrame
        Loaded rules with ``all_items`` column.
    sensitive_attrs : list[str]
        Attribute names to analyse (e.g. ``['SEX', 'RAC1P']``).

    Returns
    -------
    pd.DataFrame
        Columns: ``attribute``, ``value``, ``rule_count``, ``coverage_pct``.
    """
    total = len(rules_df)
    records: list[dict] = []

    for attr in sensitive_attrs:
        group_counts: dict[str, int] = {}
        for _, row in rules_df.iterrows():
            seen = {
                v for item in row['all_items']
                if (lv := extract_label_value(item)) and lv[0] == attr
                for v in [lv[1]]
            }
            for val in seen:
                group_counts[val] = group_counts.get(val, 0) + 1

        for val, count in sorted(group_counts.items()):
            records.append({
                'attribute':    attr,
                'value':        val,
                'rule_count':   count,
                'coverage_pct': round(100.0 * count / total, 4) if total else 0.0,
            })

    return pd.DataFrame(records)


def _has_positive_outcome(
    con_items: list[str],
    outcome_label: str,
    positive_outcome: str,
) -> bool:
    """Return True if the consequent list contains the positive-outcome item."""
    return any(
        extract_label_value(item) == (outcome_label, positive_outcome)
        for item in con_items
    )


def compute_confidence_parity(
    rules_df: pd.DataFrame,
    sensitive_attrs: list[str],
    privileged_values: dict[str, str],
    outcome_label: str,
    positive_outcome: str,
) -> pd.DataFrame:
    """
    For each sensitive attribute, collect rules whose **consequent** contains
    the positive outcome and whose **antecedent** contains a group value for
    that attribute.  Summarise the confidence distribution per group.

    If no outcome-targeting rules exist (the outcome label is absent from
    consequents), all rules are used as a fallback and a warning is printed.

    Parameters
    ----------
    rules_df : pd.DataFrame
    sensitive_attrs : list[str]
    privileged_values : dict[str, str]
    outcome_label : str
    positive_outcome : str

    Returns
    -------
    pd.DataFrame
        Columns: ``attribute``, ``value``, ``is_privileged``, ``n_rules``,
        ``mean_confidence``, ``median_confidence``, ``std_confidence``,
        ``min_confidence``, ``max_confidence``.
    """
    if 'confidence_raw' not in rules_df.columns:
        return pd.DataFrame()

    outcome_mask = rules_df['con_items'].apply(
        lambda c: _has_positive_outcome(c, outcome_label, positive_outcome)
    )
    outcome_rules = rules_df[outcome_mask]

    fallback_used = False
    if outcome_rules.empty:
        outcome_rules = rules_df  # fallback: all rules
        fallback_used = True

    if fallback_used:
        print(
            f'  [fairness] NOTE — no rules with consequent '
            f'{outcome_label}={positive_outcome} found; '
            f'confidence parity computed over all {len(rules_df)} rules.'
        )

    records: list[dict] = []
    for attr in sensitive_attrs:
        group_confs: dict[str, list[float]] = {}
        for _, row in outcome_rules.iterrows():
            conf = row['confidence_raw']
            if pd.isna(conf):
                continue
            for item in row['ant_items']:
                lv = extract_label_value(item)
                if lv and lv[0] == attr:
                    group_confs.setdefault(lv[1], []).append(float(conf))

        for val, confs in group_confs.items():
            records.append({
                'attribute':        attr,
                'value':            val,
                'is_privileged':    val == privileged_values.get(attr),
                'n_rules':          len(confs),
                'mean_confidence':  round(np.mean(confs), 6),
                'median_confidence': round(np.median(confs), 6),
                'std_confidence':   round(np.std(confs), 6),
                'min_confidence':   round(np.min(confs), 6),
                'max_confidence':   round(np.max(confs), 6),
                'fallback_all_rules': fallback_used,
            })

    return pd.DataFrame(records)


def compute_support_parity(
    rules_df: pd.DataFrame,
    sensitive_attrs: list[str],
    privileged_values: dict[str, str],
) -> pd.DataFrame:
    """
    Summarise ``support_raw`` for rules that contain each sensitive-attribute
    group value (antecedents only, since support is antecedent-driven).

    Parameters
    ----------
    rules_df : pd.DataFrame
    sensitive_attrs : list[str]
    privileged_values : dict[str, str]

    Returns
    -------
    pd.DataFrame
        Columns: ``attribute``, ``value``, ``is_privileged``, ``n_rules``,
        ``mean_support``, ``median_support``, ``std_support``.
    """
    if 'support_raw' not in rules_df.columns:
        return pd.DataFrame()

    records: list[dict] = []
    for attr in sensitive_attrs:
        group_sups: dict[str, list[float]] = {}
        for _, row in rules_df.iterrows():
            sup = row['support_raw']
            if pd.isna(sup):
                continue
            for item in row['ant_items']:
                lv = extract_label_value(item)
                if lv and lv[0] == attr:
                    group_sups.setdefault(lv[1], []).append(float(sup))

        for val, sups in group_sups.items():
            records.append({
                'attribute':      attr,
                'value':          val,
                'is_privileged':  val == privileged_values.get(attr),
                'n_rules':        len(sups),
                'mean_support':   round(np.mean(sups), 6),
                'median_support': round(np.median(sups), 6),
                'std_support':    round(np.std(sups), 6),
            })

    return pd.DataFrame(records)


def compute_lift_disparity(
    rules_df: pd.DataFrame,
    sensitive_attrs: list[str],
    privileged_values: dict[str, str],
) -> pd.DataFrame:
    """
    Compare the lift distribution of rules involving each sensitive-attribute
    group value.

    A Mann-Whitney U test is run between each unprivileged group and the
    privileged group (when both have ≥ 2 observations) to test for a
    statistically significant difference in lift rank.

    Parameters
    ----------
    rules_df : pd.DataFrame
    sensitive_attrs : list[str]
    privileged_values : dict[str, str]

    Returns
    -------
    pd.DataFrame
        Columns: ``attribute``, ``value``, ``is_privileged``, ``n_rules``,
        ``mean_lift``, ``median_lift``, ``std_lift``,
        ``mannwhitney_u``, ``pval_vs_privileged``.
    """
    if 'lift' not in rules_df.columns:
        return pd.DataFrame()

    group_lifts: dict[str, dict[str, list[float]]] = {a: {} for a in sensitive_attrs}

    for _, row in rules_df.iterrows():
        lift = row['lift']
        if pd.isna(lift):
            continue
        for attr in sensitive_attrs:
            for item in row['all_items']:
                lv = extract_label_value(item)
                if lv and lv[0] == attr:
                    group_lifts[attr].setdefault(lv[1], []).append(float(lift))

    records: list[dict] = []
    for attr in sensitive_attrs:
        priv_val   = privileged_values.get(attr)
        priv_lifts = group_lifts[attr].get(priv_val, [])

        for val, lifts in sorted(group_lifts[attr].items()):
            is_privileged = (val == priv_val)
            u_stat = pval = np.nan

            if not is_privileged and len(lifts) >= 2 and len(priv_lifts) >= 2:
                try:
                    u_stat, pval = stats.mannwhitneyu(
                        lifts, priv_lifts, alternative='two-sided'
                    )
                except Exception:
                    pass

            records.append({
                'attribute':          attr,
                'value':              val,
                'is_privileged':      is_privileged,
                'n_rules':            len(lifts),
                'mean_lift':          round(np.mean(lifts), 6)   if lifts else np.nan,
                'median_lift':        round(np.median(lifts), 6) if lifts else np.nan,
                'std_lift':           round(np.std(lifts), 6)    if lifts else np.nan,
                'mannwhitney_u':      round(u_stat, 4) if pd.notna(u_stat) else np.nan,
                'pval_vs_privileged': round(pval, 6)   if pd.notna(pval)   else np.nan,
            })

    return pd.DataFrame(records)


def compute_disparate_impact(
    confidence_parity_df: pd.DataFrame,
    privileged_values: dict[str, str],
    use_mean: bool = True,
) -> pd.DataFrame:
    """
    Compute the Disparate Impact Ratio (DIR) and Statistical Parity Difference
    (SPD) for each (attribute, unprivileged-group) pair.

    DIR  = conf(unprivileged) / conf(privileged)
    SPD  = conf(unprivileged) − conf(privileged)

    The 4/5 rule flags DIR < 0.80 as a potential fairness violation.

    Parameters
    ----------
    confidence_parity_df : pd.DataFrame
        Output of :func:`compute_confidence_parity`.
    privileged_values : dict[str, str]
        Privileged group value per attribute.
    use_mean : bool
        If True, use ``mean_confidence``; otherwise ``median_confidence``.

    Returns
    -------
    pd.DataFrame
        Columns: ``attribute``, ``privileged_value``, ``unprivileged_value``,
        ``privileged_confidence``, ``unprivileged_confidence``,
        ``disparate_impact_ratio``, ``statistical_parity_difference``,
        ``dir_threshold``, ``fairness_violation``.
    """
    if confidence_parity_df.empty:
        return pd.DataFrame()

    metric = 'mean_confidence' if use_mean else 'median_confidence'
    records: list[dict] = []

    for attr in confidence_parity_df['attribute'].unique():
        sub = confidence_parity_df[confidence_parity_df['attribute'] == attr]
        priv_val = privileged_values.get(attr)
        priv_row = sub[sub['value'] == priv_val]

        if priv_row.empty:
            print(
                f'  [fairness] WARNING — privileged value "{priv_val}" not found '
                f'in confidence parity results for attribute {attr}; DIR skipped.'
            )
            continue

        priv_conf = float(priv_row[metric].values[0])

        for _, row in sub[sub['value'] != priv_val].iterrows():
            unpriv_conf = float(row[metric])
            dir_val = unpriv_conf / priv_conf if priv_conf > 0 else np.nan
            spd     = unpriv_conf - priv_conf

            records.append({
                'attribute':                    attr,
                'privileged_value':             priv_val,
                'unprivileged_value':           row['value'],
                'n_rules_privileged':           int(priv_row['n_rules'].values[0]),
                'n_rules_unprivileged':         int(row['n_rules']),
                'privileged_confidence':        round(priv_conf, 6),
                'unprivileged_confidence':      round(unpriv_conf, 6),
                'disparate_impact_ratio':       round(dir_val, 6) if pd.notna(dir_val) else np.nan,
                'statistical_parity_difference': round(spd, 6),
                'dir_threshold':                DIR_THRESHOLD,
                'fairness_violation':           bool(pd.notna(dir_val) and dir_val < DIR_THRESHOLD),
            })

    return pd.DataFrame(records)


def compute_intersectional_analysis(
    rules_df: pd.DataFrame,
    attr1: str,
    attr2: str,
    outcome_label: str,
    positive_outcome: str,
) -> pd.DataFrame:
    """
    Cross-tabulate rules by the values of two sensitive attributes
    that co-occur in the same rule.

    Only rules containing at least one item for **each** attribute are included.
    Rules that contain multiple values for the same attribute (possible due to
    aggregation across CF neighbours) are assigned the first value encountered.

    Parameters
    ----------
    rules_df : pd.DataFrame
    attr1, attr2 : str
        The two sensitive attributes to cross-tabulate.
    outcome_label : str
    positive_outcome : str

    Returns
    -------
    pd.DataFrame
        Grouped by (attr1_value, attr2_value) with aggregated rule statistics.
    """
    records: list[dict] = []

    for _, row in rules_df.iterrows():
        val1 = val2 = None
        for item in row['all_items']:
            lv = extract_label_value(item)
            if lv:
                if lv[0] == attr1 and val1 is None:
                    val1 = lv[1]
                elif lv[0] == attr2 and val2 is None:
                    val2 = lv[1]

        if val1 is None or val2 is None:
            continue

        targets = _has_positive_outcome(row['con_items'], outcome_label, positive_outcome)
        conf    = row.get('confidence_raw', np.nan)
        sup     = row.get('support_raw',    np.nan)
        lft     = row.get('lift',           np.nan)

        records.append({
            attr1:                    val1,
            attr2:                    val2,
            'targets_positive_outcome': int(targets),
            'confidence':             float(conf) if pd.notna(conf) else np.nan,
            'support':                float(sup)  if pd.notna(sup)  else np.nan,
            'lift':                   float(lft)  if pd.notna(lft)  else np.nan,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    grp = df.groupby([attr1, attr2]).agg(
        n_rules                   = ('confidence', 'count'),
        mean_confidence           = ('confidence', 'mean'),
        mean_support              = ('support',    'mean'),
        mean_lift                 = ('lift',       'mean'),
        n_positive_outcome_rules  = ('targets_positive_outcome', 'sum'),
    ).reset_index()

    grp['mean_confidence'] = grp['mean_confidence'].round(6)
    grp['mean_support']    = grp['mean_support'].round(6)
    grp['mean_lift']       = grp['mean_lift'].round(6)

    return grp


# ---------------------------------------------------------------------------
# Population-level fairness (from original dataset)
# ---------------------------------------------------------------------------

def compute_population_fairness(
    dataset_path: Path,
    sensitive_attrs: list[str],
    privileged_values: dict[str, str],
    outcome_label: str,
    positive_outcome: str,
) -> pd.DataFrame:
    """
    Compute base-rate fairness metrics directly from the original dataset CSV.

    This provides the ground-truth disparate impact independent of which rules
    were mined — essential context for interpreting rule-level metrics.

    Parameters
    ----------
    dataset_path : Path
        Path to the original ACS CSV file.
    sensitive_attrs : list[str]
    privileged_values : dict[str, str]
    outcome_label : str
    positive_outcome : str

    Returns
    -------
    pd.DataFrame
        Columns: ``attribute``, ``value``, ``is_privileged``, ``n_samples``,
        ``n_positive``, ``positive_rate``, ``disparate_impact_ratio``,
        ``statistical_parity_difference``, ``dir_threshold``, ``fairness_violation``.
        Empty DataFrame if the file is unreadable or outcome column is absent.
    """
    try:
        ds = pd.read_csv(dataset_path, dtype=str)
    except Exception as exc:
        print(f'  [fairness] WARNING — could not load dataset {dataset_path}: {exc}')
        return pd.DataFrame()

    ds.columns = [c.strip() for c in ds.columns]

    if outcome_label not in ds.columns:
        print(
            f'  [fairness] WARNING — outcome column "{outcome_label}" not found '
            f'in dataset columns: {ds.columns.tolist()}'
        )
        return pd.DataFrame()

    outcome = ds[outcome_label].astype(str)
    records: list[dict] = []

    for attr in sensitive_attrs:
        if attr not in ds.columns:
            print(f'  [fairness] WARNING — attribute "{attr}" not in dataset; skipped.')
            continue

        col      = ds[attr].astype(str)
        priv_val = privileged_values.get(attr)

        # Compute per-group positive rate
        group_stats: dict[str, dict] = {}
        for val in sorted(col.unique()):
            mask = col == val
            n    = int(mask.sum())
            n_pos = int((outcome[mask] == str(positive_outcome)).sum())
            rate = n_pos / n if n > 0 else np.nan
            group_stats[val] = {'n': n, 'n_positive': n_pos, 'rate': rate}

        priv_rate = group_stats.get(priv_val, {}).get('rate', np.nan)

        for val, gs in sorted(group_stats.items(), key=lambda x: x[0]):
            is_priv   = (val == priv_val)
            rate      = gs['rate']

            if is_priv:
                dir_val = 1.0
                spd     = 0.0
            elif pd.notna(priv_rate) and priv_rate > 0:
                dir_val = rate / priv_rate
                spd     = rate - priv_rate
            else:
                dir_val = spd = np.nan

            records.append({
                'attribute':                     attr,
                'value':                         val,
                'is_privileged':                 is_priv,
                'n_samples':                     gs['n'],
                'n_positive':                    gs['n_positive'],
                'positive_rate':                 round(rate, 6) if pd.notna(rate) else np.nan,
                'disparate_impact_ratio':        round(dir_val, 6) if pd.notna(dir_val) else np.nan,
                'statistical_parity_difference': round(spd, 6) if pd.notna(spd) else np.nan,
                'dir_threshold':                 DIR_THRESHOLD,
                'fairness_violation':            bool(
                    not is_priv and pd.notna(dir_val) and dir_val < DIR_THRESHOLD
                ),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Aggregate summary across all rule configs
# ---------------------------------------------------------------------------

def compute_aggregate_summary(
    dir_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    lift_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assemble one consolidated metrics table from the individual results frames,
    suitable for the fairness_metrics.csv master output.
    """
    frames = []

    if not dir_df.empty:
        df = dir_df[['attribute', 'unprivileged_value', 'disparate_impact_ratio',
                     'statistical_parity_difference', 'fairness_violation']].copy()
        df.insert(0, 'metric_type', 'disparate_impact_rule_level')
        frames.append(df.rename(columns={'unprivileged_value': 'group_value'}))

    if not coverage_df.empty:
        df = coverage_df[['attribute', 'value', 'rule_count', 'coverage_pct']].copy()
        df.insert(0, 'metric_type', 'rule_coverage')
        frames.append(df.rename(columns={'value': 'group_value'}))

    if not lift_df.empty:
        df = lift_df[['attribute', 'value', 'mean_lift', 'pval_vs_privileged']].copy()
        df.insert(0, 'metric_type', 'lift_disparity')
        frames.append(df.rename(columns={'value': 'group_value'}))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

_PALETTE = {
    'blue':   '#3A7DC9',
    'orange': '#E07B39',
    'red':    '#D94F3D',
    'green':  '#4CAF50',
    'grey':   '#8B8B8B',
    'yellow': '#F5C518',
}


def _bar_colors(values: list[str], privileged: str) -> list[str]:
    """Assign colours to bars: privileged = blue, others = orange."""
    return [_PALETTE['blue'] if v == privileged else _PALETTE['orange'] for v in values]


def plot_coverage_barplot(
    coverage_df: pd.DataFrame,
    privileged_values: dict[str, str],
    output_dir: Path,
) -> None:
    """Horizontal bar chart of rule coverage percentage per group."""
    if coverage_df.empty:
        return

    for attr in coverage_df['attribute'].unique():
        sub = coverage_df[coverage_df['attribute'] == attr].sort_values('coverage_pct')
        priv = privileged_values.get(attr, '')

        fig, ax = plt.subplots(figsize=(9, max(3, len(sub) * 0.55)))
        colors = _bar_colors(sub['value'].tolist(), priv)
        bars = ax.barh(sub['value'], sub['coverage_pct'], color=colors, edgecolor='white', linewidth=0.5)

        for bar, count in zip(bars, sub['rule_count']):
            ax.text(
                bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{bar.get_width():.1f}%  (n={count})',
                va='center', ha='left', fontsize=8, color='#444444',
            )

        ax.set_xlabel('Rules containing group value (%)', fontsize=10)
        ax.set_title(
            f'Rule Coverage by {attr} Group\n'
            f'(blue = privileged: {priv})',
            fontsize=11, pad=10,
        )
        ax.set_xlim(0, sub['coverage_pct'].max() * 1.30 + 1)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        fig.savefig(output_dir / f'coverage_barplot_{attr}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_dir_barplot(
    dir_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Horizontal bar chart of Disparate Impact Ratio per (attribute, group) pair."""
    if dir_df.empty:
        return

    fig, axes = plt.subplots(
        1, dir_df['attribute'].nunique(),
        figsize=(7 * dir_df['attribute'].nunique(), max(3, len(dir_df) * 0.65 + 1)),
        squeeze=False,
    )

    for ax_col, attr in enumerate(sorted(dir_df['attribute'].unique())):
        sub = dir_df[dir_df['attribute'] == attr].sort_values('disparate_impact_ratio')
        ax = axes[0][ax_col]

        colors = [
            _PALETTE['red'] if row['fairness_violation'] else _PALETTE['green']
            for _, row in sub.iterrows()
        ]
        bars = ax.barh(sub['unprivileged_value'], sub['disparate_impact_ratio'],
                       color=colors, edgecolor='white')

        # 4/5-rule threshold line
        ax.axvline(DIR_THRESHOLD, color='black', linestyle='--', linewidth=1.2,
                   label=f'4/5 threshold ({DIR_THRESHOLD})')

        # Privileged reference at 1.0
        ax.axvline(1.0, color=_PALETTE['blue'], linestyle=':', linewidth=1.0, label='Privileged (1.0)')

        for bar, (_, row) in zip(bars, sub.iterrows()):
            ax.text(
                bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{row['disparate_impact_ratio']:.3f}",
                va='center', ha='left', fontsize=8,
            )

        ax.set_xlabel('Disparate Impact Ratio', fontsize=10)
        ax.set_title(
            f'{attr}\n(vs privileged: {dir_df[dir_df["attribute"] == attr]["privileged_value"].iloc[0]})',
            fontsize=10, pad=8,
        )
        ax.set_xlim(0, max(1.1, sub['disparate_impact_ratio'].max() * 1.15 + 0.05))
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('Rule-Level Disparate Impact Ratio  (red = violation, green = fair)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'disparate_impact_barplot.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_confidence_boxplot(
    rules_df: pd.DataFrame,
    sensitive_attrs: list[str],
    privileged_values: dict[str, str],
    output_dir: Path,
) -> None:
    """Box-plot of confidence values grouped by sensitive-attribute group."""
    if 'confidence_raw' not in rules_df.columns:
        return

    for attr in sensitive_attrs:
        group_data: dict[str, list[float]] = {}
        for _, row in rules_df.iterrows():
            conf = row['confidence_raw']
            if pd.isna(conf):
                continue
            for item in row['ant_items']:
                lv = extract_label_value(item)
                if lv and lv[0] == attr:
                    group_data.setdefault(lv[1], []).append(float(conf))

        if not group_data:
            continue

        priv = privileged_values.get(attr, '')
        labels = sorted(group_data.keys(), key=lambda v: (v != priv, v))
        data   = [group_data[l] for l in labels]
        colors = _bar_colors(labels, priv)

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color='black', linewidth=1.5))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Rule Confidence', fontsize=10)
        ax.set_title(
            f'Confidence Distribution by {attr} Group\n'
            f'(blue = privileged: {priv})',
            fontsize=11,
        )
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        fig.savefig(output_dir / f'confidence_boxplot_{attr}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_lift_disparity_barplot(
    lift_df: pd.DataFrame,
    privileged_values: dict[str, str],
    output_dir: Path,
) -> None:
    """Bar chart of mean lift per sensitive-attribute group."""
    if lift_df.empty:
        return

    for attr in lift_df['attribute'].unique():
        sub  = lift_df[lift_df['attribute'] == attr].sort_values('mean_lift')
        priv = privileged_values.get(attr, '')
        colors = _bar_colors(sub['value'].tolist(), priv)

        fig, ax = plt.subplots(figsize=(9, max(3, len(sub) * 0.55)))
        bars = ax.barh(sub['value'], sub['mean_lift'], color=colors,
                       edgecolor='white', linewidth=0.5)

        # Reference line at lift=1
        ax.axvline(1.0, color='black', linestyle='--', linewidth=1, label='Lift = 1.0 (independence)')

        for bar, (_, row) in zip(bars, sub.iterrows()):
            lbl = f"{row['mean_lift']:.3f}"
            if pd.notna(row.get('pval_vs_privileged')):
                p = row['pval_vs_privileged']
                sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                if sig:
                    lbl += f' {sig}'
            ax.text(
                bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                lbl, va='center', ha='left', fontsize=8,
            )

        ax.set_xlabel('Mean Rule Lift', fontsize=10)
        ax.set_title(
            f'Mean Rule Lift by {attr} Group\n'
            f'(* p<.05, ** p<.01, *** p<.001 vs privileged; blue = {priv})',
            fontsize=10, pad=8,
        )
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        fig.savefig(output_dir / f'lift_disparity_barplot_{attr}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_intersectional_heatmap(
    intersect_df: pd.DataFrame,
    attr1: str,
    attr2: str,
    output_dir: Path,
) -> None:
    """Heatmap of mean confidence for each (attr1, attr2) combination."""
    if intersect_df.empty or attr1 not in intersect_df.columns or attr2 not in intersect_df.columns:
        return

    pivot = intersect_df.pivot_table(
        index=attr1, columns=attr2, values='mean_confidence', aggfunc='mean'
    )

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.3),
                                    max(4, len(pivot.index) * 0.7)))
    img = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                    vmin=0, vmax=pivot.values[~np.isnan(pivot.values)].max() if not np.all(np.isnan(pivot.values)) else 1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel(attr2, fontsize=10)
    ax.set_ylabel(attr1, fontsize=10)
    ax.set_title(f'Mean Rule Confidence — {attr1} × {attr2}\n(intersectional analysis)',
                 fontsize=11, pad=10)

    for ri in range(len(pivot.index)):
        for ci in range(len(pivot.columns)):
            val = pivot.values[ri, ci]
            if not np.isnan(val):
                ax.text(ci, ri, f'{val:.3f}', ha='center', va='center',
                        fontsize=7, color='black')

    plt.colorbar(img, ax=ax, fraction=0.025, pad=0.02).set_label('Mean Confidence', fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / f'intersectional_heatmap_{attr1}_{attr2}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_population_fairness(
    pop_df: pd.DataFrame,
    privileged_values: dict[str, str],
    output_dir: Path,
) -> None:
    """Bar charts of positive rate and DIR from the original dataset."""
    if pop_df.empty:
        return

    for attr in pop_df['attribute'].unique():
        sub  = pop_df[pop_df['attribute'] == attr].sort_values('positive_rate')
        priv = privileged_values.get(attr, '')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(3, len(sub) * 0.55 + 1)))

        # Positive-rate bar
        colors = _bar_colors(sub['value'].tolist(), priv)
        bars1 = ax1.barh(sub['value'], sub['positive_rate'] * 100,
                         color=colors, edgecolor='white')
        for bar, (_, row) in zip(bars1, sub.iterrows()):
            ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     f"{row['positive_rate'] * 100:.1f}%  (n={row['n_samples']:,})",
                     va='center', ha='left', fontsize=8)
        ax1.set_xlabel('Positive Outcome Rate (%)', fontsize=10)
        ax1.set_title(f'{attr}: Positive Rate by Group', fontsize=10)
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.set_xlim(0, sub['positive_rate'].max() * 130)

        # DIR bar (unprivileged groups only)
        dir_sub = sub[sub['value'] != priv].sort_values('disparate_impact_ratio')
        if not dir_sub.empty:
            dir_colors = [
                _PALETTE['red'] if row['fairness_violation'] else _PALETTE['green']
                for _, row in dir_sub.iterrows()
            ]
            bars2 = ax2.barh(dir_sub['value'], dir_sub['disparate_impact_ratio'],
                             color=dir_colors, edgecolor='white')
            ax2.axvline(DIR_THRESHOLD, color='black', linestyle='--', linewidth=1.2,
                        label=f'4/5 threshold ({DIR_THRESHOLD})')
            ax2.axvline(1.0, color=_PALETTE['blue'], linestyle=':', linewidth=1.0,
                        label='Privileged (1.0)')
            for bar, (_, row) in zip(bars2, dir_sub.iterrows()):
                ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                         f"{row['disparate_impact_ratio']:.3f}",
                         va='center', ha='left', fontsize=8)
            ax2.set_xlabel('Disparate Impact Ratio', fontsize=10)
            ax2.set_title(f'{attr}: Dataset-Level DIR\n(red = violation)', fontsize=10)
            ax2.legend(fontsize=8)
            ax2.set_xlim(0, max(1.15, dir_sub['disparate_impact_ratio'].max() * 1.15))
            ax2.spines[['top', 'right']].set_visible(False)
        else:
            ax2.text(0.5, 0.5, 'No unprivileged groups found',
                     ha='center', va='center', transform=ax2.transAxes)

        fig.suptitle(f'Population-Level Fairness Metrics — {attr}', fontsize=12, y=1.02)
        plt.tight_layout()
        fig.savefig(output_dir / f'population_fairness_{attr}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def _fmt_dir(dir_val: float) -> str:
    if pd.isna(dir_val):
        return 'N/A'
    flag = '  ⚠ VIOLATION' if dir_val < DIR_THRESHOLD else ''
    return f'{dir_val:.4f}{flag}'


def write_fairness_report(
    output_dir: Path,
    rules_path_desc: str,
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
) -> None:
    """Write a human-readable fairness report to ``fairness_report.txt``."""
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines: list[str] = []
    hr = '=' * 72

    lines += [
        hr,
        'FAIRNESS ANALYSIS REPORT — ASSOCIATION RULES (VALUE LEVEL)',
        hr,
        f'Generated : {ts}',
        f'Rules source : {rules_path_desc}',
        f'Total rules analysed : {n_rules:,}',
        f'Sensitive attributes : {", ".join(sensitive_attrs)}',
        f'Privileged groups    : {privileged_values}',
        f'Outcome item         : {outcome_label}={positive_outcome}',
        f'DIR threshold (4/5)  : {DIR_THRESHOLD}',
        '',
    ]

    # --- Rule Coverage ---
    lines += [hr, '1. RULE COVERAGE BY DEMOGRAPHIC GROUP', hr, '']
    if coverage_df.empty:
        lines.append('  No sensitive-attribute values found in rules.')
    else:
        for attr in sensitive_attrs:
            sub = coverage_df[coverage_df['attribute'] == attr]
            if sub.empty:
                lines.append(f'  {attr}: no coverage data.')
                continue
            lines.append(f'  {attr}:')
            for _, row in sub.sort_values('coverage_pct', ascending=False).iterrows():
                priv_tag = ' [privileged]' if row['value'] == privileged_values.get(attr) else ''
                lines.append(
                    f'    {row["value"]:<35}{priv_tag}  '
                    f'{row["rule_count"]:>5} rules  '
                    f'({row["coverage_pct"]:.2f}%)'
                )
            lines.append('')

    # --- Confidence Parity ---
    lines += [hr, '2. CONFIDENCE PARITY (antecedent-group → outcome rules)', hr, '']
    if conf_parity_df.empty:
        lines.append('  Insufficient data.')
    else:
        for attr in sensitive_attrs:
            sub = conf_parity_df[conf_parity_df['attribute'] == attr]
            if sub.empty:
                continue
            fallback = sub['fallback_all_rules'].any() if 'fallback_all_rules' in sub.columns else False
            note = ' [all rules used as fallback — no outcome items in consequents]' if fallback else ''
            lines.append(f'  {attr}{note}:')
            for _, row in sub.sort_values('mean_confidence', ascending=False).iterrows():
                priv_tag = ' [privileged]' if row['value'] == privileged_values.get(attr) else ''
                lines.append(
                    f'    {row["value"]:<35}{priv_tag}  '
                    f'mean={row["mean_confidence"]:.4f}  '
                    f'med={row["median_confidence"]:.4f}  '
                    f'(n={row["n_rules"]})'
                )
            lines.append('')

    # --- Disparate Impact ---
    lines += [hr, '3. DISPARATE IMPACT RATIO (rule-level, 4/5 rule)', hr, '']
    if dir_df.empty:
        lines.append('  No DIR computed (insufficient group coverage).')
    else:
        violations = dir_df['fairness_violation'].sum()
        lines.append(f'  Total (attr, group) pairs tested : {len(dir_df)}')
        lines.append(f'  Fairness violations (DIR < {DIR_THRESHOLD}) : {violations}')
        lines.append('')
        for attr in sensitive_attrs:
            sub = dir_df[dir_df['attribute'] == attr]
            if sub.empty:
                continue
            priv = sub['privileged_value'].iloc[0]
            lines.append(f'  {attr}  (privileged: {priv}, conf={sub["privileged_confidence"].iloc[0]:.4f}):')
            for _, row in sub.sort_values('disparate_impact_ratio').iterrows():
                lines.append(
                    f'    vs {row["unprivileged_value"]:<33}  '
                    f'DIR={_fmt_dir(row["disparate_impact_ratio"])}  '
                    f'SPD={row["statistical_parity_difference"]:+.4f}'
                )
            lines.append('')

    # --- Lift Disparity ---
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
                priv_tag = ' [privileged]' if row['value'] == privileged_values.get(attr) else ''
                pval_str = (
                    f'  p={row["pval_vs_privileged"]:.4f}'
                    if pd.notna(row.get('pval_vs_privileged')) else ''
                )
                lines.append(
                    f'    {row["value"]:<35}{priv_tag}  '
                    f'mean_lift={row["mean_lift"]:.4f}'
                    f'{pval_str}'
                )
            lines.append('')

    # --- Intersectional ---
    lines += [hr, '5. INTERSECTIONAL ANALYSIS', hr, '']
    if intersect_df.empty:
        lines.append('  No intersectional rules (not enough co-occurrences of both attributes).')
    else:
        lines.append(
            f'  {len(intersect_df)} (SEX, RAC1P) group combinations '
            f'appear in rules together.\n'
        )
        lines.append(
            f'  {"SEX":<25}  {"RAC1P":<30}  {"n_rules":>7}  '
            f'{"mean_conf":>9}  {"mean_lift":>9}'
        )
        lines.append('  ' + '-' * 85)
        for _, row in intersect_df.sort_values('mean_confidence', ascending=False).iterrows():
            attr1_col = intersect_df.columns[0]
            attr2_col = intersect_df.columns[1]
            lines.append(
                f'  {str(row[attr1_col]):<25}  {str(row[attr2_col]):<30}  '
                f'{int(row["n_rules"]):>7}  '
                f'{row["mean_confidence"]:>9.4f}  '
                f'{row["mean_lift"]:>9.4f}'
            )
        lines.append('')

    # --- Population-level ---
    lines += [hr, '6. POPULATION-LEVEL FAIRNESS (from original dataset)', hr, '']
    if pop_df.empty:
        lines.append('  No dataset provided (use --dataset flag or dataset_paths argument).')
    else:
        pop_violations = pop_df['fairness_violation'].sum()
        lines.append(
            f'  Population DIR violations (DIR < {DIR_THRESHOLD}) : {pop_violations}'
        )
        lines.append('')
        for attr in sensitive_attrs:
            sub = pop_df[pop_df['attribute'] == attr]
            if sub.empty:
                continue
            lines.append(f'  {attr}:')
            for _, row in sub.sort_values('positive_rate', ascending=False).iterrows():
                priv_tag = ' [privileged]' if row['value'] == privileged_values.get(attr) else ''
                dir_str = (
                    f'DIR={_fmt_dir(row["disparate_impact_ratio"])}  '
                    if not row['is_privileged'] else ''
                )
                lines.append(
                    f'    {row["value"]:<35}{priv_tag}  '
                    f'rate={row["positive_rate"]:.4f}  '
                    f'{dir_str}'
                    f'(n={row["n_samples"]:,})'
                )
            lines.append('')

    lines += [hr, 'END OF REPORT', hr]

    report_path = output_dir / 'fairness_report.txt'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'  [fairness] Report written → {report_path}')


# ---------------------------------------------------------------------------
# Main orchestrator for a single rules source
# ---------------------------------------------------------------------------

def analyse_rules(
    rules_source: Path,
    output_dir: Path,
    sensitive_attrs: list[str]      = None,
    privileged_values: dict[str, str] = None,
    outcome_label: str              = DEFAULT_OUTCOME_LABEL,
    positive_outcome: str           = DEFAULT_POSITIVE_OUTCOME,
    dataset_paths: list[Path]       = None,
) -> dict[str, pd.DataFrame]:
    """
    Run the full fairness analysis for a single rules source (file or directory).

    Parameters
    ----------
    rules_source : Path
        Either a single rules.csv or a directory (scanned recursively).
    output_dir : Path
        Directory where all outputs will be written.
    sensitive_attrs : list[str]
        Sensitive attribute names. Defaults to ``['SEX', 'RAC1P']``.
    privileged_values : dict[str, str]
        Privileged group per attribute.
    outcome_label : str
        Feature name of the outcome (default: ``INCOME_ABOVE_THRESHOLD``).
    positive_outcome : str
        String value representing the positive outcome (default: ``'1'``).
    dataset_paths : list[Path] or None
        Paths to original dataset CSVs for population-level analysis.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary of result DataFrames keyed by metric name.
    """
    if sensitive_attrs is None:
        sensitive_attrs = DEFAULT_SENSITIVE_ATTRS
    if privileged_values is None:
        privileged_values = DEFAULT_PRIVILEGED_VALUES

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # ── Load rules ──────────────────────────────────────────────────────────
    print(f'\n[fairness] Loading rules from: {rules_source}')
    if rules_source.is_dir():
        rules_df = collect_rules_from_dir(rules_source)
        desc = str(rules_source)
    else:
        rules_df = load_rules(rules_source)
        desc = str(rules_source)

    if rules_df.empty:
        print('[fairness] No rules loaded — nothing to analyse.')
        return {}

    print(f'[fairness] {len(rules_df):,} rules loaded.')

    # ── Rule-level metrics ──────────────────────────────────────────────────
    print('[fairness] Computing rule coverage …')
    coverage_df = compute_rule_coverage(rules_df, sensitive_attrs)

    print('[fairness] Computing confidence parity …')
    conf_parity_df = compute_confidence_parity(
        rules_df, sensitive_attrs, privileged_values, outcome_label, positive_outcome
    )

    print('[fairness] Computing support parity …')
    support_parity_df = compute_support_parity(rules_df, sensitive_attrs, privileged_values)

    print('[fairness] Computing lift disparity …')
    lift_df = compute_lift_disparity(rules_df, sensitive_attrs, privileged_values)

    print('[fairness] Computing disparate impact ratio …')
    dir_df = compute_disparate_impact(conf_parity_df, privileged_values)

    # ── Intersectional ──────────────────────────────────────────────────────
    intersect_df = pd.DataFrame()
    if len(sensitive_attrs) >= 2:
        print(f'[fairness] Computing intersectional analysis ({sensitive_attrs[0]} × {sensitive_attrs[1]}) …')
        intersect_df = compute_intersectional_analysis(
            rules_df, sensitive_attrs[0], sensitive_attrs[1],
            outcome_label, positive_outcome,
        )

    # ── Population-level ────────────────────────────────────────────────────
    pop_df = pd.DataFrame()
    if dataset_paths:
        frames = []
        for dp in dataset_paths:
            print(f'[fairness] Computing population fairness from: {dp}')
            pf = compute_population_fairness(
                dp, sensitive_attrs, privileged_values, outcome_label, positive_outcome
            )
            if not pf.empty:
                pf['dataset'] = dp.name
                frames.append(pf)
        if frames:
            pop_df = pd.concat(frames, ignore_index=True)

    # ── Summary table ───────────────────────────────────────────────────────
    summary_df = compute_aggregate_summary(dir_df, coverage_df, lift_df)

    # ── Save CSVs ───────────────────────────────────────────────────────────
    def _save(df: pd.DataFrame, name: str) -> None:
        if not df.empty:
            path = output_dir / name
            df.to_csv(path, index=False)
            print(f'  [fairness] Saved → {path}')

    _save(coverage_df,       'rule_coverage.csv')
    _save(conf_parity_df,    'confidence_parity.csv')
    _save(support_parity_df, 'support_parity.csv')
    _save(lift_df,           'lift_disparity.csv')
    _save(dir_df,            'disparate_impact.csv')
    _save(intersect_df,      'intersectional_analysis.csv')
    _save(pop_df,            'population_fairness.csv')
    _save(summary_df,        'fairness_metrics.csv')

    # ── Plots ────────────────────────────────────────────────────────────────
    print('[fairness] Generating plots …')
    plot_coverage_barplot(coverage_df, privileged_values, plots_dir)
    plot_dir_barplot(dir_df, plots_dir)
    plot_confidence_boxplot(rules_df, sensitive_attrs, privileged_values, plots_dir)
    plot_lift_disparity_barplot(lift_df, privileged_values, plots_dir)
    if intersect_df is not None and not intersect_df.empty and len(sensitive_attrs) >= 2:
        plot_intersectional_heatmap(
            intersect_df, sensitive_attrs[0], sensitive_attrs[1], plots_dir
        )
    if not pop_df.empty:
        plot_population_fairness(pop_df, privileged_values, plots_dir)

    # ── Report ───────────────────────────────────────────────────────────────
    write_fairness_report(
        output_dir=output_dir,
        rules_path_desc=desc,
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

    print(f'\n[fairness] Analysis complete. Results in: {output_dir}\n')

    return {
        'coverage':         coverage_df,
        'confidence_parity': conf_parity_df,
        'support_parity':   support_parity_df,
        'lift_disparity':   lift_df,
        'disparate_impact': dir_df,
        'intersectional':   intersect_df,
        'population':       pop_df,
        'summary':          summary_df,
    }


# ---------------------------------------------------------------------------
# Pipeline entry point — mirrors main() signature of the other pipeline scripts
# ---------------------------------------------------------------------------

def main(
    regions: list[str]               = None,
    k_values: list[int]              = None,
    base_dir: Path                   = None,
    sensitive_attrs: list[str]       = None,
    privileged_values: dict[str, str] = None,
    outcome_label: str               = DEFAULT_OUTCOME_LABEL,
    positive_outcome: str            = DEFAULT_POSITIVE_OUTCOME,
    # Grid parameters — accepted but not used (kept for signature parity)
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
) -> None:
    """
    Pipeline entry point for the fairness analysis step.

    Signature mirrors ``microscopic_experiment_association_rules_values.main()``
    so that main.py can invoke this step with the same parameter dict without
    any adapter code.  The grid parameters (sup_*, conf_*, lift_*) are accepted
    for signature compatibility but are not used — fairness analysis is
    performed on whatever rules exist.

    Parameters
    ----------
    regions : list[str] or None
        Regions to process.  Defaults to ``['northeast', 'south']`` when None.
    k_values : list[int] or None
        CF neighbourhood sizes.  Defaults to ``[1, 3, 5, 7]`` when None.
    base_dir : Path or None
        Project root.  Auto-detected for Kaggle/Colab when None.
    sensitive_attrs : list[str] or None
        Sensitive attribute names.  Defaults to ``['SEX', 'RAC1P']``.
    privileged_values : dict[str, str] or None
        Privileged group per attribute.  Defaults to
        ``{'SEX': 'Male', 'RAC1P': 'White-Alone'}``.
    outcome_label : str
        Outcome feature name (default: ``INCOME_ABOVE_THRESHOLD``).
    positive_outcome : str
        Positive-outcome value string (default: ``'1'``).
    """
    # ── Resolve base_dir ────────────────────────────────────────────────────
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
    if sensitive_attrs is None:
        sensitive_attrs = DEFAULT_SENSITIVE_ATTRS
    if privileged_values is None:
        privileged_values = DEFAULT_PRIVILEGED_VALUES

    results_dir = base_dir / 'results'
    data_dir    = base_dir / 'data'

    for region in regions:
        print('\n' + '=' * 70)
        print(f'FAIRNESS ANALYSIS — {region.upper()}')
        print('=' * 70)

        # Locate rules source: scan all value-level association rule outputs
        ar_values_dir = results_dir / region / 'association_rules_values'
        if not ar_values_dir.exists():
            print(
                f'  [fairness] No association_rules_values directory found for '
                f'{region}; skipping.'
            )
            continue

        # Locate original dataset(s) for this region
        dataset_paths: list[Path] = sorted(
            data_dir.glob(f'*{region}*.csv')
        ) if data_dir.exists() else []

        output_dir = results_dir / region / 'fairness_analysis'

        analyse_rules(
            rules_source    = ar_values_dir,
            output_dir      = output_dir,
            sensitive_attrs = sensitive_attrs,
            privileged_values = privileged_values,
            outcome_label   = outcome_label,
            positive_outcome = positive_outcome,
            dataset_paths   = dataset_paths if dataset_paths else None,
        )

    print('\n' + '=' * 70)
    print('Fairness analysis done.')
    print('=' * 70 + '\n')


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Fairness analysis on association rules CSV files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        '--rules', required=True, type=Path,
        help='Path to a single rules.csv or a directory (scanned recursively).',
    )
    p.add_argument(
        '--dataset', type=Path, action='append', dest='datasets', default=[],
        metavar='PATH',
        help='Original dataset CSV for population-level metrics. '
             'Can be specified multiple times.',
    )
    p.add_argument(
        '--output_dir', type=Path, default=None,
        help='Output directory (default: <rules_parent>/fairness_analysis/).',
    )
    p.add_argument(
        '--sensitive_attrs', nargs='+', default=DEFAULT_SENSITIVE_ATTRS,
        metavar='ATTR',
        help=f'Sensitive attribute names (default: {DEFAULT_SENSITIVE_ATTRS}).',
    )
    p.add_argument(
        '--privileged', nargs='+', default=[], metavar='ATTR=VALUE',
        help='Privileged values as ATTR=VALUE pairs, e.g. SEX=Male RAC1P=White-Alone.',
    )
    p.add_argument(
        '--outcome_label', default=DEFAULT_OUTCOME_LABEL,
        help=f'Outcome column/item name (default: {DEFAULT_OUTCOME_LABEL}).',
    )
    p.add_argument(
        '--positive_outcome', default=DEFAULT_POSITIVE_OUTCOME,
        help=f'Positive-outcome value string (default: {DEFAULT_POSITIVE_OUTCOME}).',
    )
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
        rules_source      = args.rules,
        output_dir        = out,
        sensitive_attrs   = args.sensitive_attrs,
        privileged_values = priv_vals,
        outcome_label     = args.outcome_label,
        positive_outcome  = args.positive_outcome,
        dataset_paths     = args.datasets or None,
    )