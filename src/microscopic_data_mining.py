"""
src/microscopic_data_mining.py
───────────────────────────────
Stage 4 of the ACS Income pipeline: Microscopic Association Rule Mining (ARM).

Relation to stage 3 (macroscopic ARM)
──────────────────────────────────────
Stage 3 discards feature values and works only with feature *labels*
(e.g. the transaction {SCHL, WKHP}).  Stage 4 is the *microscopic* companion:
it works with full "LABEL=value" tokens (e.g. {SCHL=Bachelors-Degree,
WKHP=Full-Time}) so the rules carry specific value-level information.

The microscopic analysis is anchored to the macroscopic rules: for each
macroscopic rule (antecedent_labels → consequent_labels) we select only the
itemset rows that contain at least one item belonging to that rule (i.e. at
least one token whose label matches a label in the antecedent or consequent).
This keeps the microscopic analysis focused on the same feature relationships
identified at the coarser level.

Input
─────
  • Stage-2 itemset CSV  (feature_importance_itemsets_k<N>[suffix].csv  or
    the combined file) — same file used by stage 3, but now the full
    "LABEL=value" tokens are retained instead of being reduced to labels.
  • Stage-3 macroscopic rules CSV  (association_rules/k<N>/arm[suffix]_rules.csv
    or the all_k equivalent) — used to derive the anchor label sets.

Processing flow (per macroscopic rule, per k)
─────────────────────────────────────────────
  1. Parse the macroscopic rule → extract the set of feature labels that appear
     in its antecedent and/or consequent.
  2. Filter the itemset CSV: keep only rows whose itemset contains at least one
     token with a matching label.
  3. Build microscopic transactions: each retained row becomes a list of full
     "LABEL=value" tokens (sorted, deduplicated).
  4. Run the same adaptive FP-Growth grid search used in stage 3 on these
     microscopic transactions.
  5. Annotate every surviving microscopic rule with:
       • macro_antecedents  — the macroscopic antecedent it is anchored to
       • macro_consequents  — the macroscopic consequent it is anchored to
       • macro_rule_id      — index of the macroscopic rule (0-based)

Output directory structure
──────────────────────────
All outputs land under the stage-3 association_rules directory:

  <output_dir>/association_rules/
  ├── k<N>/
  │   ├── micro/
  │   │   ├── micro_rules[suffix].csv         ← all microscopic rules
  │   │   ├── micro_grid_summary[suffix].csv
  │   │   └── heatmaps/
  │   │       ├── heatmap_support_confidence[suffix].png
  │   │       ├── heatmap_support_lift[suffix].png
  │   │       └── heatmap_confidence_lift[suffix].png
  │   └── ...  (existing macroscopic outputs)
  └── all_k/
      ├── micro/
      │   ├── micro_rules[suffix].csv
      │   ├── micro_grid_summary[suffix].csv
      │   └── heatmaps/
      └── ...

Column layout in micro_rules.csv
─────────────────────────────────
  macro_rule_id        — 0-based index of the macroscopic rule that anchored
                         the transaction filter
  macro_antecedents    — macroscopic antecedent label(s), e.g. "SCHL"
  macro_consequents    — macroscopic consequent label(s), e.g. "WKHP"
  k_value              — k value (per-k files only)
  antecedents          — microscopic antecedent "LABEL=value" item(s)
  consequents          — microscopic consequent "LABEL=value" item(s)
  antecedent support   — P(antecedent)
  consequent support   — P(consequent)
  support              — P(antecedent ∪ consequent)
  confidence           — P(consequent | antecedent)
  lift                 — observed / expected co-occurrence
  leverage             — P(A∪C) − P(A)·P(C)
  conviction           — (1 − P(C)) / (1 − confidence)
  lift_type            — "positive_correlation" or "negative_correlation"
  grid_min_support     — min_support threshold that produced this rule
  grid_min_confidence  — min_confidence threshold that produced this rule
  filter_*             — active filter thresholds (self-documenting)

Skip-if-exists behaviour
────────────────────────
If micro_rules[suffix].csv already exists in a k-folder the entire microscopic
run for that k is skipped.  Delete the file to force a re-run.

Performance design
──────────────────
  1. Macroscopic rules CSV read once per k; label sets pre-computed into
     frozensets for O(1) membership tests.
  2. Itemset CSV parsed once per k with the same vectorised explode+groupby
     pipeline used in stage 3.
  3. Per-rule transaction filtering uses a vectorised pandas mask on the
     exploded token frame — no Python loop over rows.
  4. Grid search reuses the adaptive strategy from stage 3 (vectorised path
     for few items, threaded path for many items).
  5. Results across all macroscopic rules are concatenated and deduplicated
     before writing.

Dependencies
────────────
  pip install mlxtend pandas numpy matplotlib seaborn
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Re-use the shared infrastructure from stage 3.
from src.macroscopic_data_mining import (
    DEFAULT_LIFT_INDEPENDENCE_LOW,
    DEFAULT_LIFT_INDEPENDENCE_HIGH,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MAX_CONFIDENCE,
    DEFAULT_MIN_SUPPORT,
    DEFAULT_MAX_SUPPORT,
    _DEFAULT_ARM_WORKERS,
    _FOLDER_ARM,
    _FOLDER_FEATURE_IMP,
    _FOLDER_HEATMAPS,
    _arm_root,
    _k_dir,
    _all_k_dir,
    _heatmap_dir,
    _build_boolean_matrix,
    _mine_frequent_itemsets,
    _annotate_lift_type,
    _compute_data_driven_steps,
    _linspace,
    _build_grid,
    generate_heatmaps,
    run_grid_search,
)

logger = logging.getLogger("src.microscopic_data_mining")

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

# Sub-folder name for microscopic outputs inside each k-folder.
_FOLDER_MICRO = "micro"

# ─────────────────────────────────────────────────────────────────────────────
# Default parameters (inherit stage-3 defaults)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MICRO_MIN_SUPPORT         = DEFAULT_MIN_SUPPORT
DEFAULT_MICRO_MAX_SUPPORT         = DEFAULT_MAX_SUPPORT
DEFAULT_MICRO_SUPPORT_STEP        = None   # data-driven

DEFAULT_MICRO_MIN_CONFIDENCE      = DEFAULT_MIN_CONFIDENCE
DEFAULT_MICRO_MAX_CONFIDENCE      = DEFAULT_MAX_CONFIDENCE
DEFAULT_MICRO_CONFIDENCE_STEP     = None   # data-driven

DEFAULT_MICRO_LIFT_LOW  = DEFAULT_LIFT_INDEPENDENCE_LOW
DEFAULT_MICRO_LIFT_HIGH = DEFAULT_LIFT_INDEPENDENCE_HIGH


# ─────────────────────────────────────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _micro_dir(k_or_allk_dir: Path) -> Path:
    """Return <k_dir>/micro/ or <all_k_dir>/micro/, creating if absent."""
    p = k_or_allk_dir / _FOLDER_MICRO
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Macroscopic rules loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_macro_rules(macro_rules_path: Path) -> pd.DataFrame:
    """
    Load the macroscopic rules CSV produced by stage 3.

    Returns a DataFrame with at minimum columns:
      macro_rule_id, macro_antecedents (str), macro_consequents (str)
    and the full set of metric/filter columns from the macroscopic run.

    Raises FileNotFoundError if the file does not exist.
    """
    if not macro_rules_path.exists():
        raise FileNotFoundError(
            f"Macroscopic rules file not found: {macro_rules_path}.  "
            "Make sure stage 3 (macroscopic_data_mining.py) has completed."
        )

    try:
        df = pd.read_csv(macro_rules_path)
    except pd.errors.EmptyDataError:
        logger.warning("Macroscopic rules file %s is empty.", macro_rules_path.name)
        return pd.DataFrame()

    if df.empty:
        logger.warning("Macroscopic rules file %s has no data rows.", macro_rules_path.name)
        return pd.DataFrame()

    # Rename for clarity in the microscopic context.
    rename = {}
    if "antecedents" in df.columns:
        rename["antecedents"] = "macro_antecedents"
    if "consequents" in df.columns:
        rename["consequents"] = "macro_consequents"
    df = df.rename(columns=rename)

    # Add a stable 0-based rule index.
    df.insert(0, "macro_rule_id", range(len(df)))

    logger.info(
        "Loaded %d macroscopic rules from %s.", len(df), macro_rules_path.name
    )
    return df


def _macro_label_sets(macro_rules: pd.DataFrame) -> list[frozenset[str]]:
    """
    For each macroscopic rule return the frozenset of feature labels that appear
    in its antecedent OR consequent.

    Input format: antecedent/consequent are strings like "SCHL" or "SCHL & WKHP".
    """
    label_sets: list[frozenset[str]] = []
    for row in macro_rules.itertuples(index=False):
        row_dict = row._asdict()
        labels: set[str] = set()
        for col in ("macro_antecedents", "macro_consequents"):
            val = str(row_dict.get(col, ""))
            if val and val.lower() not in ("nan", ""):
                for part in val.split(" & "):
                    lbl = part.strip()
                    if lbl:
                        labels.add(lbl)
        label_sets.append(frozenset(labels))
    return label_sets


# ─────────────────────────────────────────────────────────────────────────────
# Microscopic transaction loading and filtering
# ─────────────────────────────────────────────────────────────────────────────

def load_micro_itemsets(csv_path: Path) -> pd.DataFrame:
    """
    Load the stage-2 itemset CSV and return a DataFrame with columns:
      _txn_idx  — original row index (one per boundary instance)
      token     — each "LABEL=value" token (one row per token per transaction)
      label     — the LABEL part (before "=")

    This is the exploded form used for vectorised per-rule filtering.
    The full "LABEL=value" tokens are retained (not reduced to labels).

    Returns an empty DataFrame if the file is empty or has no data.
    """
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        logger.warning("File %s is empty.", csv_path.name)
        return pd.DataFrame()

    if df.empty or "itemset" not in df.columns:
        logger.warning("File %s has no usable itemset data.", csv_path.name)
        return pd.DataFrame()

    df = df[["itemset"]].dropna().reset_index(drop=True)
    df.index.name = "_txn_idx"

    exploded = df["itemset"].str.split().explode().reset_index()
    exploded.columns = ["_txn_idx", "token"]
    exploded = exploded[exploded["token"].str.contains("=", na=False)].copy()
    exploded["label"] = exploded["token"].str.split("=", n=1).str[0]
    exploded = exploded.drop_duplicates(subset=["_txn_idx", "token"])

    logger.info(
        "Loaded %d tokens from %d transactions in %s.",
        len(exploded),
        exploded["_txn_idx"].nunique(),
        csv_path.name,
    )
    return exploded


def _filter_transactions_for_rule(
    exploded: pd.DataFrame,
    label_set: frozenset[str],
) -> list[list[str]]:
    """
    From the exploded token DataFrame, select all transactions (rows) that
    contain at least one token whose label is in *label_set*.

    Returns microscopic transactions: each transaction is a sorted list of
    unique "LABEL=value" tokens from the matching rows.

    Parameters
    ----------
    exploded   : Output of load_micro_itemsets().
    label_set  : Frozenset of feature labels from the macroscopic rule's
                 antecedent ∪ consequent.
    """
    if exploded.empty or not label_set:
        return []

    # Vectorised mask: find transactions containing at least one matching label.
    mask = exploded["label"].isin(label_set)
    matching_txn_ids = set(exploded.loc[mask, "_txn_idx"].unique())

    if not matching_txn_ids:
        return []

    # Keep ALL tokens of the matching transactions.
    subset = exploded[exploded["_txn_idx"].isin(matching_txn_ids)]

    if subset.empty:
        return []

    # Build token lists using numpy lexsort+split — faster than groupby+apply
    # for large subsets (~4× speedup on 43k+ transaction sets).
    idx_arr   = subset["_txn_idx"].values
    tok_arr   = subset["token"].values
    order     = np.lexsort((tok_arr, idx_arr))
    idx_s     = idx_arr[order]
    tok_s     = tok_arr[order]
    split_pts = np.where(np.diff(idx_s))[0] + 1
    return [sorted(chunk.tolist()) for chunk in np.split(tok_s, split_pts)]


# ─────────────────────────────────────────────────────────────────────────────
# CSV formatting
# ─────────────────────────────────────────────────────────────────────────────

def _format_micro_rules_for_csv(
    rules_df: pd.DataFrame,
    lift_independence_low: float,
    lift_independence_high: float,
    min_support: Optional[float] = None,
    max_support: Optional[float] = None,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
) -> pd.DataFrame:
    """
    Prepare the microscopic rules DataFrame for CSV output.

    Applies the same formatting as the macroscopic version but prepends the
    macro_rule_id / macro_antecedents / macro_consequents columns so every
    row traces back to its macroscopic anchor rule.
    """
    df = rules_df.copy()

    # Convert frozenset columns to readable strings.
    for col in ("antecedents", "consequents"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: " & ".join(sorted(x)) if isinstance(x, frozenset) else str(x)
            )

    # Add lift_type.
    df = _annotate_lift_type(df, lift_independence_low, lift_independence_high)

    # Rename internal grid columns.
    df = df.rename(columns={
        "grid_support":    "grid_min_support",
        "grid_confidence": "grid_min_confidence",
    })

    # Add global filter thresholds.
    if min_support    is not None: df["filter_min_support"]    = min_support
    if max_support    is not None: df["filter_max_support"]    = max_support
    if min_confidence is not None: df["filter_min_confidence"] = min_confidence
    if max_confidence is not None: df["filter_max_confidence"] = max_confidence
    df["filter_lift_kept_below"] = lift_independence_low
    df["filter_lift_kept_above"] = lift_independence_high
    df["filter_lift_discarded"]  = f"[{lift_independence_low}, {lift_independence_high}]"

    # Drop unwanted columns.
    _DROP = {"zhangs_metric", "jaccard", "certainty", "kulczynski", "representativity"}
    df = df.drop(columns=[c for c in _DROP if c in df.columns])

    # Sanitise inf / -inf / NaN in numeric columns (e.g. conviction → inf
    # when confidence = 1.0).  Replace infinities with NaN so to_csv()
    # writes an empty cell rather than the string "inf".
    _FLOAT_COLS = [
        "antecedent support", "consequent support",
        "support", "confidence", "lift", "leverage", "conviction",
        "grid_min_support", "grid_min_confidence",
    ]
    for col in _FLOAT_COLS:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # Canonical column order — macro provenance columns come first.
    _ORDERED = [
        "macro_rule_id", "macro_antecedents", "macro_consequents",
        "k_value",
        "antecedents", "consequents",
        "antecedent support", "consequent support",
        "support", "confidence", "lift", "leverage", "conviction", "lift_type",
        "grid_min_support", "grid_min_confidence",
        "filter_min_support", "filter_max_support",
        "filter_min_confidence", "filter_max_confidence",
        "filter_lift_kept_below", "filter_lift_kept_above",
        "filter_lift_discarded",
    ]
    present = [c for c in _ORDERED if c in df.columns]
    extra   = [c for c in df.columns if c not in present]
    return df[present + extra]


def _save_micro_results(
    all_rules: pd.DataFrame,
    grid_summary: pd.DataFrame,
    dest_dir: Path,
    filename_stem: str,
    lift_independence_low: float,
    lift_independence_high: float,
    min_support: Optional[float]    = None,
    max_support: Optional[float]    = None,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
) -> None:
    """Write micro_rules and micro_grid_summary CSVs to dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    rules_path   = dest_dir / f"{filename_stem}_rules.csv"
    summary_path = dest_dir / f"{filename_stem}_grid_summary.csv"

    rules_csv = _format_micro_rules_for_csv(
        all_rules,
        lift_independence_low=lift_independence_low,
        lift_independence_high=lift_independence_high,
        min_support=min_support,
        max_support=max_support,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )
    rules_csv.to_csv(rules_path, index=False, float_format="%.6f")
    logger.info("    Micro rules   → %s  (%d rows)", rules_path.name, len(rules_csv))

    gs = grid_summary.copy()
    if min_support    is not None: gs["filter_min_support"]    = min_support
    if max_support    is not None: gs["filter_max_support"]    = max_support
    if min_confidence is not None: gs["filter_min_confidence"] = min_confidence
    if max_confidence is not None: gs["filter_max_confidence"] = max_confidence
    gs["filter_lift_kept_below"] = lift_independence_low
    gs["filter_lift_kept_above"] = lift_independence_high
    gs["filter_lift_discarded"]  = f"[{lift_independence_low}, {lift_independence_high}]"
    gs.to_csv(summary_path, index=False, float_format="%.6f")
    logger.info("    Micro summary → %s  (%d cells)", summary_path.name, len(grid_summary))


# ─────────────────────────────────────────────────────────────────────────────
# Core: microscopic ARM for a single macroscopic rule
# ─────────────────────────────────────────────────────────────────────────────

def _run_micro_for_macro_rule(
    macro_rule: pd.Series,
    label_set: frozenset[str],
    exploded_tokens: pd.DataFrame,
    min_support: float,
    max_support: float,
    support_step: Optional[float],
    min_confidence: float,
    max_confidence: float,
    confidence_step: Optional[float],
    lift_independence_low: float,
    lift_independence_high: float,
    n_workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the microscopic grid search for a single macroscopic rule.

    Returns (micro_rules_df, grid_summary_df).  Both are annotated with
    macro_rule_id, macro_antecedents, macro_consequents.  Returns empty
    DataFrames if no transactions survive the filter or no rules are found.
    """
    macro_id   = int(macro_rule["macro_rule_id"])
    macro_ant  = str(macro_rule.get("macro_antecedents", ""))
    macro_cons = str(macro_rule.get("macro_consequents", ""))

    logger.info(
        "    macro rule %d: [%s] → [%s]  |  label filter: %s",
        macro_id, macro_ant, macro_cons, sorted(label_set),
    )

    # Filter transactions to those relevant to this macroscopic rule.
    transactions = _filter_transactions_for_rule(exploded_tokens, label_set)

    if not transactions:
        logger.info(
            "      No transactions contain any label from this rule — skipping."
        )
        return pd.DataFrame(), pd.DataFrame()

    logger.info("      %d transactions selected for microscopic ARM.", len(transactions))

    all_rules, grid_summary, _freq_micro = run_grid_search(
        transactions=transactions,
        min_support=min_support,
        max_support=max_support,
        support_step=support_step,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        confidence_step=confidence_step,
        lift_independence_low=lift_independence_low,
        lift_independence_high=lift_independence_high,
        n_workers=n_workers,
    )

    if all_rules.empty:
        logger.info("      No microscopic rules found.")
        return pd.DataFrame(), grid_summary

    # Annotate with macroscopic provenance.
    all_rules = all_rules.copy()
    all_rules.insert(0, "macro_consequents",  macro_cons)
    all_rules.insert(0, "macro_antecedents",  macro_ant)
    all_rules.insert(0, "macro_rule_id",      macro_id)

    grid_summary = grid_summary.copy()
    grid_summary.insert(0, "macro_consequents", macro_cons)
    grid_summary.insert(0, "macro_antecedents", macro_ant)
    grid_summary.insert(0, "macro_rule_id",     macro_id)

    logger.info("      → %d microscopic rules found.", len(all_rules))
    return all_rules, grid_summary


# ─────────────────────────────────────────────────────────────────────────────
# Per-k runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_micro_for_k(
    k_val: int,
    output_dir: Path,
    suffix: str,
    macro_rules: pd.DataFrame,
    min_support: float,
    max_support: float,
    support_step: Optional[float],
    min_confidence: float,
    max_confidence: float,
    confidence_step: Optional[float],
    lift_independence_low: float,
    lift_independence_high: float,
    n_workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full microscopic ARM for a single k value.

    Loads the per-k itemset file (or falls back to the combined file),
    iterates over all macroscopic rules, runs the microscopic grid search
    for each, aggregates results, saves to association_rules/k<N>/micro/,
    and returns (all_micro_rules_df, all_grid_summary_df).
    """
    logger.info("  ── micro k=%d ─────────────────────────────────────", k_val)

    # Locate the itemset CSV (search output_dir and feature_importance/).
    search_dirs = [output_dir, output_dir / _FOLDER_FEATURE_IMP]
    csv_path: Optional[Path] = None
    for d in search_dirs:
        candidate = d / f"feature_importance_itemsets_k{k_val}{suffix}.csv"
        if candidate.exists():
            csv_path = candidate
            break
    if csv_path is None:
        # Fallback: combined file.
        for d in search_dirs:
            candidate = d / f"feature_importance_itemsets{suffix}.csv"
            if candidate.exists():
                csv_path = candidate
                break
    if csv_path is None:
        logger.error(
            "    No itemset CSV found for k=%d suffix='%s' — skipping.", k_val, suffix
        )
        return pd.DataFrame(), pd.DataFrame()

    # Load the full exploded token frame once; reuse for all macro rules.
    exploded_tokens = load_micro_itemsets(csv_path)
    if exploded_tokens.empty:
        logger.warning("    No usable tokens for k=%d — skipping.", k_val)
        return pd.DataFrame(), pd.DataFrame()

    # Pre-compute label sets for all macroscopic rules.
    label_sets = _macro_label_sets(macro_rules)

    all_rules_list:   list[pd.DataFrame] = []
    all_summary_list: list[pd.DataFrame] = []

    for macro_rule, label_set in zip(macro_rules.itertuples(index=False), label_sets):
        # Convert namedtuple to dict-like access via _asdict().
        macro_row = macro_rule._asdict()
        rules_i, summary_i = _run_micro_for_macro_rule(
            macro_rule=pd.Series(macro_row),
            label_set=label_set,
            exploded_tokens=exploded_tokens,
            min_support=min_support,
            max_support=max_support,
            support_step=support_step,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            confidence_step=confidence_step,
            lift_independence_low=lift_independence_low,
            lift_independence_high=lift_independence_high,
            n_workers=n_workers,
        )
        if not rules_i.empty:
            all_rules_list.append(rules_i)
        if not summary_i.empty:
            all_summary_list.append(summary_i)

    # Aggregate.
    if all_rules_list:
        combined_rules = pd.concat(all_rules_list, ignore_index=True)
        # Deduplicate on (macro_rule_id, antecedents, consequents) — the same
        # micro rule may appear across different grid cells.
        key_cols = ["macro_rule_id", "antecedents", "consequents"]
        key_cols = [c for c in key_cols if c in combined_rules.columns]
        if key_cols:
            def _fs_str(x: object) -> str:
                return "|".join(sorted(x)) if isinstance(x, frozenset) else str(x)
            combined_rules["_dedup_key"] = (
                combined_rules["macro_rule_id"].astype(str)
                + "||"
                + combined_rules["antecedents"].apply(_fs_str)
                + "→"
                + combined_rules["consequents"].apply(_fs_str)
            )
            combined_rules = (
                combined_rules
                .drop_duplicates(subset="_dedup_key", keep="first")
                .drop(columns="_dedup_key")
                .reset_index(drop=True)
            )
        # Annotate with k value.
        combined_rules.insert(0, "k_value", k_val)
    else:
        combined_rules = pd.DataFrame()

    if all_summary_list:
        combined_summary = pd.concat(all_summary_list, ignore_index=True)
        combined_summary.insert(0, "k_value", k_val)
    else:
        combined_summary = pd.DataFrame(
            columns=["k_value", "macro_rule_id", "min_support", "min_confidence", "n_rules"]
        )

    # Save per-k micro outputs.
    k_out  = _k_dir(output_dir, k_val)
    micro_out = _micro_dir(k_out)
    stem = f"micro{suffix}"

    _save_micro_results(
        combined_rules, combined_summary, micro_out, stem,
        lift_independence_low=lift_independence_low,
        lift_independence_high=lift_independence_high,
        min_support=min_support, max_support=max_support,
        min_confidence=min_confidence, max_confidence=max_confidence,
    )

    _hm_grid: pd.DataFrame
    if (not combined_summary.empty
            and {"min_support", "min_confidence", "n_rules"} <= set(combined_summary.columns)):
        _hm_grid = (
            combined_summary
            .groupby(["min_support", "min_confidence"], as_index=False)["n_rules"]
            .sum()
        )
    else:
        _hm_grid = pd.DataFrame(columns=["min_support", "min_confidence", "n_rules"])

    generate_heatmaps(
        all_rules=combined_rules,
        grid_summary=_hm_grid,
        heatmap_dir=_heatmap_dir(micro_out),
        suffix=suffix,
        lift_independence_low=lift_independence_low,
        lift_independence_high=lift_independence_high,
        k_label=f"micro k={k_val}",
        min_support=min_support, max_support=max_support,
        min_confidence=min_confidence, max_confidence=max_confidence,
    )

    return combined_rules, combined_summary


# ─────────────────────────────────────────────────────────────────────────────
# Helpers to locate macroscopic rules files
# ─────────────────────────────────────────────────────────────────────────────

def _find_macro_rules_path(output_dir: Path, k_val: int, suffix: str) -> Optional[Path]:
    """
    Locate the macroscopic rules CSV for a given k value.

    Tries, in order:
      association_rules/k<N>/arm[suffix]_rules.csv
      association_rules/all_k/arm[suffix]_all_k_rules.csv  (fallback)
    """
    per_k = _k_dir(output_dir, k_val) / f"arm{suffix}_rules.csv"
    if per_k.exists():
        return per_k
    all_k = _all_k_dir(output_dir) / f"arm{suffix}_all_k_rules.csv"
    if all_k.exists():
        logger.info(
            "    Per-k macro rules not found for k=%d; using all_k rules.", k_val
        )
        return all_k
    return None


def _discover_k_values_micro(output_dir: Path, suffix: str) -> list[int]:
    """
    Discover k values for which macroscopic rules already exist.
    Returns sorted list of k values found in association_rules/k<N>/.
    """
    arm_root = _arm_root(output_dir)
    pattern = re.compile(r"^k(\d+)$")
    found: list[int] = []
    if arm_root.is_dir():
        for d in arm_root.iterdir():
            m = pattern.match(d.name)
            if m and d.is_dir():
                rules_file = d / f"arm{suffix}_rules.csv"
                if rules_file.exists():
                    found.append(int(m.group(1)))
    return sorted(found)


# ─────────────────────────────────────────────────────────────────────────────
# Main stage-4 runner
# ─────────────────────────────────────────────────────────────────────────────

def run_microscopic_mining(
    output_dir: Path,
    original_class: list[int],
    k_value: Optional[int]              = None,
    min_support: float                  = DEFAULT_MICRO_MIN_SUPPORT,
    max_support: float                  = DEFAULT_MICRO_MAX_SUPPORT,
    support_step: Optional[float]       = DEFAULT_MICRO_SUPPORT_STEP,
    min_confidence: float               = DEFAULT_MICRO_MIN_CONFIDENCE,
    max_confidence: float               = DEFAULT_MICRO_MAX_CONFIDENCE,
    confidence_step: Optional[float]    = DEFAULT_MICRO_CONFIDENCE_STEP,
    lift_independence_low: float        = DEFAULT_MICRO_LIFT_LOW,
    lift_independence_high: float       = DEFAULT_MICRO_LIFT_HIGH,
    n_workers: int                      = _DEFAULT_ARM_WORKERS,
) -> None:
    """
    Entry point for stage 4, called by main.py after stage 3 completes.

    For each class in *original_class*:
      1. Discovers k values for which macroscopic rules exist.
      2. For each k: loads macroscopic rules → filters itemset transactions
         per rule → runs microscopic grid search → saves results.
      3. Aggregates all per-k microscopic rules into association_rules/all_k/micro/.

    Skip-if-exists: if micro_rules[suffix].csv already exists in a k-folder
    the microscopic run for that k is skipped.  The all_k aggregation is
    repeated whenever any per-k result is new.

    Parameters
    ----------
    output_dir             : Stage-2/3 output directory.
    original_class         : List of class indices ([0], [1], or [0, 1]).
    k_value                : If set, process only that k value.
    min_support            : Lower bound of the support grid.
    max_support            : Upper bound filter for support.
    support_step           : Step size (None = data-driven).
    min_confidence         : Lower bound of the confidence grid.
    max_confidence         : Upper bound filter for confidence.
    confidence_step        : Step size (None = data-driven).
    lift_independence_low  : Lower boundary of the lift independence interval.
    lift_independence_high : Upper boundary of the lift independence interval.
    n_workers              : Thread-pool size (parallel path only).
    """
    logger.info("═" * 62)
    logger.info("  ACS INCOME PIPELINE  —  stage 4: microscopic ARM")
    logger.info("═" * 62)
    logger.info("  Output dir             : %s", output_dir.resolve())
    logger.info("  k selector             : %s",
                k_value if k_value else "auto (all k with macro rules)")
    logger.info("  Support  grid          : [%.3f, %.3f]  step=%s",
                min_support, max_support,
                f"{support_step:.4f}" if support_step is not None else "auto")
    logger.info("  Confidence grid        : [%.3f, %.3f]  step=%s",
                min_confidence, max_confidence,
                f"{confidence_step:.4f}" if confidence_step is not None else "auto")
    logger.info("  Lift independence zone : [%.2f, %.2f]  → discarded",
                lift_independence_low, lift_independence_high)
    logger.info("  Workers                : %d", n_workers)
    logger.info("═" * 62)

    orig_classes_requested = sorted(set(original_class))
    suffix_map = {
        c: (f"_class{c}" if len(orig_classes_requested) > 1 else "")
        for c in orig_classes_requested
    }

    for orig_cls in orig_classes_requested:
        suffix = suffix_map[orig_cls]

        logger.info("── Class %d ──────────────────────────────────────────", orig_cls)

        # Determine k values to process.
        if k_value is not None:
            k_values_to_run = [k_value]
        else:
            k_values_to_run = _discover_k_values_micro(output_dir, suffix)
            if not k_values_to_run:
                logger.warning(
                    "No macroscopic rules found for suffix='%s' — "
                    "run stage 3 first.", suffix
                )
                continue

        logger.info("  k values to process: %s", k_values_to_run)

        all_micro_rules_list:   list[pd.DataFrame] = []
        all_micro_summary_list: list[pd.DataFrame] = []

        for k_val in k_values_to_run:
            k_out        = _k_dir(output_dir, k_val)
            micro_out    = _micro_dir(k_out)
            micro_rules_path = micro_out / f"micro{suffix}_rules.csv"

            # Skip-if-exists for this k.
            if micro_rules_path.exists():
                logger.info(
                    "  Skipping micro k=%d: %s already exists.",
                    k_val, micro_rules_path.name,
                )
                try:
                    ex_rules   = pd.read_csv(micro_rules_path)
                    ex_summary = pd.read_csv(micro_out / f"micro{suffix}_grid_summary.csv")
                    if not ex_rules.empty:
                        all_micro_rules_list.append(ex_rules)
                    if not ex_summary.empty:
                        all_micro_summary_list.append(ex_summary)
                except (pd.errors.EmptyDataError, FileNotFoundError):
                    pass
                continue

            # Locate macroscopic rules for this k.
            macro_path = _find_macro_rules_path(output_dir, k_val, suffix)
            if macro_path is None:
                logger.warning(
                    "  No macroscopic rules found for k=%d suffix='%s' — skipping.",
                    k_val, suffix,
                )
                continue

            macro_rules = _load_macro_rules(macro_path)
            if macro_rules.empty:
                logger.warning("  Macroscopic rules file is empty for k=%d.", k_val)
                continue

            rules_k, summary_k = _run_micro_for_k(
                k_val=k_val,
                output_dir=output_dir,
                suffix=suffix,
                macro_rules=macro_rules,
                min_support=min_support, max_support=max_support,
                support_step=support_step,
                min_confidence=min_confidence, max_confidence=max_confidence,
                confidence_step=confidence_step,
                lift_independence_low=lift_independence_low,
                lift_independence_high=lift_independence_high,
                n_workers=n_workers,
            )

            if not rules_k.empty:
                all_micro_rules_list.append(rules_k)
            if not summary_k.empty:
                all_micro_summary_list.append(summary_k)

        # ── Aggregate all-k micro results ─────────────────────────────────────
        logger.info("── Aggregating all-k micro results for class %d …", orig_cls)
        all_k_micro_out = _micro_dir(_all_k_dir(output_dir))

        if all_micro_rules_list:
            combined_rules = pd.concat(all_micro_rules_list, ignore_index=True)
        else:
            combined_rules = pd.DataFrame()

        if all_micro_summary_list:
            _concat_summary = pd.concat(all_micro_summary_list, ignore_index=True)
            grp_cols = ["macro_rule_id", "macro_antecedents", "macro_consequents",
                        "min_support", "min_confidence"]
            grp_cols = [c for c in grp_cols if c in _concat_summary.columns]
            if grp_cols and "n_rules" in _concat_summary.columns:
                combined_summary = (
                    _concat_summary
                    .groupby(grp_cols, as_index=False)["n_rules"]
                    .sum()
                )
            else:
                combined_summary = _concat_summary
        else:
            combined_summary = pd.DataFrame()

        _save_micro_results(
            combined_rules, combined_summary, all_k_micro_out,
            f"micro{suffix}_all_k",
            lift_independence_low=lift_independence_low,
            lift_independence_high=lift_independence_high,
            min_support=min_support, max_support=max_support,
            min_confidence=min_confidence, max_confidence=max_confidence,
        )

        _hm_grid_allk: pd.DataFrame
        if (not combined_summary.empty
                and {"min_support", "min_confidence", "n_rules"} <= set(combined_summary.columns)):
            _hm_grid_allk = (
                combined_summary
                .groupby(["min_support", "min_confidence"], as_index=False)["n_rules"]
                .sum()
            )
        else:
            _hm_grid_allk = pd.DataFrame(columns=["min_support", "min_confidence", "n_rules"])

        generate_heatmaps(
            all_rules=combined_rules,
            grid_summary=_hm_grid_allk,
            heatmap_dir=_heatmap_dir(all_k_micro_out),
            suffix=suffix,
            lift_independence_low=lift_independence_low,
            lift_independence_high=lift_independence_high,
            k_label="micro all k",
            min_support=min_support, max_support=max_support,
            min_confidence=min_confidence, max_confidence=max_confidence,
        )

    logger.info("═" * 62)
    logger.info("  Stage 4 (microscopic ARM) completed.")
    logger.info("  Outputs in: %s", (_arm_root(output_dir)).resolve())
    logger.info("═" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument definitions (consumed by main.py's build_parser)
# ─────────────────────────────────────────────────────────────────────────────

def add_micro_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add stage-4 (microscopic ARM) arguments to an existing ArgumentParser.

    Called by main.py's build_parser() to keep all CLI logic in one place.
    All arguments are optional with sensible defaults.
    """
    micro = parser.add_argument_group(
        "Micro ARM hyperparameters  (stage 4 — microscopic association rule mining)"
    )

    micro.add_argument(
        "--micro-min-support", type=float,
        default=DEFAULT_MICRO_MIN_SUPPORT, metavar="S",
        help=f"Minimum support for the microscopic FP-Growth grid search.  "
             f"Default: {DEFAULT_MICRO_MIN_SUPPORT}.",
    )
    micro.add_argument(
        "--micro-max-support", type=float,
        default=DEFAULT_MICRO_MAX_SUPPORT, metavar="S",
        help=f"Maximum support upper-bound filter.  Default: {DEFAULT_MICRO_MAX_SUPPORT}.",
    )
    micro.add_argument(
        "--micro-support-step", type=float, default=None, metavar="S",
        help="Step size for the microscopic support grid.  "
             "Default: auto-computed from transaction data.",
    )
    micro.add_argument(
        "--micro-min-confidence", type=float,
        default=DEFAULT_MICRO_MIN_CONFIDENCE, metavar="C",
        help=f"Minimum confidence for the microscopic grid search.  "
             f"Default: {DEFAULT_MICRO_MIN_CONFIDENCE}.",
    )
    micro.add_argument(
        "--micro-max-confidence", type=float,
        default=DEFAULT_MICRO_MAX_CONFIDENCE, metavar="C",
        help=f"Maximum confidence upper-bound filter.  "
             f"Default: {DEFAULT_MICRO_MAX_CONFIDENCE}.",
    )
    micro.add_argument(
        "--micro-confidence-step", type=float, default=None, metavar="C",
        help="Step size for the microscopic confidence grid.  "
             "Default: auto-computed from transaction data.",
    )
    micro.add_argument(
        "--micro-lift-low", type=float,
        default=DEFAULT_MICRO_LIFT_LOW, metavar="L",
        help=f"Lower boundary of the lift independence interval for microscopic rules.  "
             f"Default: {DEFAULT_MICRO_LIFT_LOW}.",
    )
    micro.add_argument(
        "--micro-lift-high", type=float,
        default=DEFAULT_MICRO_LIFT_HIGH, metavar="L",
        help=f"Upper boundary of the lift independence interval for microscopic rules.  "
             f"Default: {DEFAULT_MICRO_LIFT_HIGH}.",
    )
    micro.add_argument(
        "--micro-k", type=int, default=None, metavar="K",
        help="If set, run microscopic ARM only for this k value.  "
             "Default: process all k values for which macroscopic rules exist.",
    )
    micro.add_argument(
        "--micro-workers", type=int,
        default=_DEFAULT_ARM_WORKERS, metavar="N",
        help=f"Thread-pool size for the microscopic grid-search (parallel path only).  "
             f"Default: {_DEFAULT_ARM_WORKERS} (auto-detected).",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry-point guard
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print(
        "microscopic_data_mining.py is stage 4 of the pipeline and cannot be\n"
        "run independently — it depends on stage 3 macroscopic rules.\n\n"
        "Run the full pipeline via:\n"
        "  python -m src.main [OPTIONS]\n\n"
        "Stage-4 specific options:\n"
        "  --micro-min-support        (default: 0.05)\n"
        "  --micro-max-support        (default: 1.00)\n"
        "  --micro-support-step       (default: auto)\n"
        "  --micro-min-confidence     (default: 0.50)\n"
        "  --micro-max-confidence     (default: 1.00)\n"
        "  --micro-confidence-step    (default: auto)\n"
        "  --micro-lift-low           (default: 0.75)\n"
        "  --micro-lift-high          (default: 1.25)\n"
        f"  --micro-workers            (default: {_DEFAULT_ARM_WORKERS}, auto-detected)\n"
        "  --micro-k                  (default: None — all k with macro rules)\n",
        file=sys.stderr,
    )
    sys.exit(1)