"""
src/macroscopic_data_mining.py
──────────────────────────────
Stage 3 of the ACS Income pipeline: Macroscopic Association Rule Mining (ARM).

This module consumes the itemset CSV files produced by stage 2 (BoCSoR,
feature_importance.py) and mines association rules using FP-Growth.

Macroscopic analysis
────────────────────
Each row of the stage-2 itemset file contains a space-separated string of
tokens in the form "FEATURE=value" (e.g. "SCHL=Bachelors-Degree WKHP=Full-Time").
For the *macroscopic* view only the **feature labels** are retained — the
values are discarded.  This means each transaction becomes the set of feature
*names* that were relevant for a given boundary instance (e.g. {"SCHL", "WKHP"}).

Association rule mining
───────────────────────
Frequent itemsets are mined with the FP-Growth algorithm (mlxtend).
Rules are evaluated with three metrics:

  support    – fraction of transactions that contain both antecedent and
               consequent.  Filters: [min_support, max_support].

  confidence – P(consequent | antecedent).
               Filters: [min_confidence, max_confidence].

  lift       – ratio of observed co-occurrence to expected under independence.
               lift = support(A ∪ C) / (support(A) × support(C))
               Filter: lift < lift_independence_low  OR
                       lift > lift_independence_high
               This keeps:
                 • positive correlations  (lift > 1.25)
                 • negative correlations  (lift < 0.75)
               and **discards** independence / near-independence rules.
               The range [0.75, 1.25] is configurable via CLI.

Output directory structure
──────────────────────────
All outputs land under the stage-2 output directory in two sub-folders:

  <output_dir>/
  ├── feature_importance/          ← stage-2 files are moved/referenced here
  └── association_rules/
      ├── k<N>/                    ← one sub-folder per k value
      │   ├── arm_rules[suffix].csv
      │   ├── arm_grid_summary[suffix].csv
      │   └── heatmaps/
      │       ├── heatmap_support_confidence[suffix].png
      │       ├── heatmap_support_lift[suffix].png
      │       └── heatmap_confidence_lift[suffix].png
      └── all_k/                   ← combined across all k values
          ├── arm_rules_all_k[suffix].csv
          ├── arm_grid_summary_all_k[suffix].csv
          └── heatmaps/
              ├── heatmap_support_confidence[suffix].png
              ├── heatmap_support_lift[suffix].png
              └── heatmap_confidence_lift[suffix].png

Heatmaps
────────
For each k value (and for the combined all-k run) three heatmaps are generated:

  1. Support × Confidence  — rows = support thresholds, cols = confidence thresholds.
  2. Support × Lift         — rows = support thresholds, cols = lift bins.
  3. Confidence × Lift      — rows = confidence thresholds, cols = lift bins.

Cell colour encodes the number of rules found at that parameter combination.
Darker cells = more rules.  The lift independence window (the discarded range)
is visually annotated on the lift axis with a hatched band so it is immediately
clear that no rules appear in that region.

Grid search
───────────
A grid search over all (min_support, min_confidence) pairs is run per k value.
FP-Growth is executed once per distinct support threshold and the result is
cached.  ``association_rules()`` is then called once per support level at the
global min_confidence floor, and per-cell confidence/lift filtering is applied
as vectorised numpy operations — no thread pool needed.

Performance design
──────────────────
  1. Vectorised transaction parsing via pandas explode + groupby.
  2. Boolean matrix built once with numpy and reused across support thresholds.
  3. FP-Growth cached per distinct min_support value.
  4. Adaptive rule generation strategy — chosen at runtime by probing the
     actual rule volume at the lowest support threshold:
       • Few rules (≤ 500 per support level, typical with ≤ ~7 items):
         association_rules() called once per support level at the global
         min_confidence floor; per-cell filtering is a vectorised numpy mask.
         Thread scheduling overhead would dominate over useful work here.
       • Many rules (> 500 per support level, typical with many columns):
         ThreadPoolExecutor evaluates each (support, confidence) cell
         concurrently; rule generation cost justifies parallelism.
  5. Vectorised deduplication via serialised frozenset keys + drop_duplicates.
  6. Single-pass filter: support, confidence, lift combined in one boolean mask.

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

logger = logging.getLogger("src.macroscopic_data_mining")

# Silence matplotlib's verbose DEBUG output (font scoring, cache lookups, etc.)
# which floods the log when the root logger is set to DEBUG.
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Default grid-search parameters
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MIN_SUPPORT         = 0.05
DEFAULT_MAX_SUPPORT         = 1.00
DEFAULT_SUPPORT_STEP        = None   # None → computed from transactions (data-driven)

DEFAULT_MIN_CONFIDENCE      = 0.50
DEFAULT_MAX_CONFIDENCE      = 1.00
DEFAULT_CONFIDENCE_STEP     = None   # None → computed from transactions (data-driven)

# Target number of grid steps along each axis when using data-driven step.
# Chosen to produce a grid dense enough to capture meaningful variation
# without redundant cells (empirically: 30–50 levels per axis is ideal).
_GRID_TARGET_STEPS = 40

DEFAULT_LIFT_INDEPENDENCE_LOW  = 0.75
DEFAULT_LIFT_INDEPENDENCE_HIGH = 1.25

DEFAULT_K_VALUE: Optional[int] = None   # None → process ALL k values found

_ARM_WORKER_CEILING  = 16
_DEFAULT_ARM_WORKERS = max(1, min(_ARM_WORKER_CEILING, (os.cpu_count() or 4) - 2))

# Sub-folder names inside the stage-2 output directory.
_FOLDER_FEATURE_IMP  = "feature_importance"
_FOLDER_ARM          = "association_rules"
_FOLDER_HEATMAPS     = "heatmaps"
_FOLDER_ALL_K        = "all_k"


# ─────────────────────────────────────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _arm_root(output_dir: Path) -> Path:
    """Return <output_dir>/association_rules/, creating it if absent."""
    p = output_dir / _FOLDER_ARM
    p.mkdir(parents=True, exist_ok=True)
    return p


def _k_dir(output_dir: Path, k_val: int) -> Path:
    """Return <output_dir>/association_rules/k<N>/."""
    p = _arm_root(output_dir) / f"k{k_val}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _all_k_dir(output_dir: Path) -> Path:
    """Return <output_dir>/association_rules/all_k/."""
    p = _arm_root(output_dir) / _FOLDER_ALL_K
    p.mkdir(parents=True, exist_ok=True)
    return p


def _heatmap_dir(parent: Path) -> Path:
    """Return <parent>/heatmaps/, creating it if absent."""
    p = parent / _FOLDER_HEATMAPS
    p.mkdir(parents=True, exist_ok=True)
    return p


def _feature_imp_dir(output_dir: Path) -> Path:
    """Return <output_dir>/feature_importance/, creating it if absent."""
    p = output_dir / _FOLDER_FEATURE_IMP
    p.mkdir(parents=True, exist_ok=True)
    return p


def _move_feature_importance_files(output_dir: Path) -> None:
    """
    Move all stage-2 output files from *output_dir* into the
    ``feature_importance/`` sub-folder so the directory tree is tidy.

    Files matched:
      • feature_importance*.csv
      • feature_importance*.md
      • bocsor_summary*.md

    Already-moved files (already inside feature_importance/) are left alone.
    """
    fi_dir = _feature_imp_dir(output_dir)
    patterns = [
        "feature_importance*.csv",
        "feature_importance*.md",
        "bocsor_summary*.md",
    ]
    moved = 0
    for pattern in patterns:
        for src in output_dir.glob(pattern):
            # Skip files already inside the sub-folder.
            if src.parent == fi_dir:
                continue
            dst = fi_dir / src.name
            if dst.exists():
                # Destination already present — remove the stale copy in root.
                src.unlink()
            else:
                src.rename(dst)
            moved += 1
    if moved:
        logger.info(
            "Moved %d stage-2 file(s) → %s/", moved, fi_dir.name
        )


# ─────────────────────────────────────────────────────────────────────────────
# Discover available k values from stage-2 output
# ─────────────────────────────────────────────────────────────────────────────

def _discover_k_values(output_dir: Path, suffix: str) -> list[int]:
    """
    Scan *output_dir* and its ``feature_importance/`` sub-folder for per-k
    itemset files and return sorted k values.

    Looks for files matching:
        feature_importance_itemsets_k<N><suffix>.csv

    Searches both the root output directory and the feature_importance/
    sub-folder so that discovery works regardless of whether the move has
    already been performed.

    Returns an empty list if no per-k files are found.
    """
    pattern = re.compile(
        r"^feature_importance_itemsets_k(\d+)" + re.escape(suffix) + r"\.csv$"
    )
    found: list[int] = []
    search_dirs = [output_dir, output_dir / _FOLDER_FEATURE_IMP]
    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for f in search_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                k = int(m.group(1))
                if k not in found:
                    found.append(k)
    return sorted(found)


def _build_input_path(
    output_dir: Path,
    suffix: str,
    k_value: Optional[int],
) -> Path:
    """
    Resolve the stage-2 itemset CSV for a given k value (or the combined file).

    Searches both *output_dir* and its ``feature_importance/`` sub-folder so
    that file resolution works regardless of whether the post-processing move
    has already been performed (files may be in either location).

    Resolution order for each location:
      1. Per-k file  feature_importance_itemsets_k<N><suffix>.csv
      2. Combined    feature_importance_itemsets<suffix>.csv

    Raises FileNotFoundError if neither is found in either location.
    """
    search_dirs = [output_dir, output_dir / _FOLDER_FEATURE_IMP]

    if k_value is not None:
        for d in search_dirs:
            candidate = d / f"feature_importance_itemsets_k{k_value}{suffix}.csv"
            if candidate.exists():
                return candidate
        logger.warning(
            "Per-k file feature_importance_itemsets_k%d%s.csv not found "
            "in output_dir or feature_importance/; falling back to combined file.",
            k_value, suffix,
        )

    for d in search_dirs:
        combined = d / f"feature_importance_itemsets{suffix}.csv"
        if combined.exists():
            return combined

    raise FileNotFoundError(
        f"No itemset CSV found for suffix='{suffix}' "
        f"(tried k_value={k_value} and combined file in {output_dir} "
        f"and {output_dir / _FOLDER_FEATURE_IMP}).  "
        "Make sure stage 2 (feature_importance.py) has completed."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Transaction loading
# ─────────────────────────────────────────────────────────────────────────────

def load_itemsets(csv_path: Path, k_value: Optional[int] = None) -> list[list[str]]:
    """
    Load a stage-2 itemset CSV and return macroscopic transactions.

    Each transaction is a **sorted list of unique feature labels** extracted
    from the ``itemset`` column (tokens before the ``=`` separator).

    Parsing is fully vectorised via pandas explode + groupby — no Python
    row loop.

    Parameters
    ----------
    csv_path : Path to a ``feature_importance_itemsets*.csv`` file.
    k_value  : If given, only rows with this ``k_value`` are used.

    Returns
    -------
    List of transactions (each transaction is a sorted list of label strings).
    """
    # Guard: empty file (e.g. k=1 produces 0 boundary instances → 0 rows).
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        logger.warning(
            "File %s is empty (0 rows) — no transactions available.", csv_path.name
        )
        return []

    # Guard: non-empty file but no data rows (only header).
    if df.empty:
        logger.warning(
            "File %s contains a header but no data rows.", csv_path.name
        )
        return []

    missing = {"itemset"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Expected column(s) {missing} not found in {csv_path}.\n"
            f"Available columns: {list(df.columns)}"
        )

    if k_value is not None:
        if "k_value" not in df.columns:
            raise ValueError(
                f"Column 'k_value' not found in {csv_path} — "
                "cannot filter by k_value."
            )
        df = df[df["k_value"] == k_value]
        if df.empty:
            logger.warning("No rows with k_value=%d in %s.", k_value, csv_path)
            return []

    df = df[["itemset"]].dropna().reset_index(drop=True)
    df.index.name = "_txn_idx"

    exploded = df["itemset"].str.split().explode().reset_index()
    exploded.columns = ["_txn_idx", "token"]
    exploded = exploded[exploded["token"].str.contains("=", na=False)].copy()
    exploded["label"] = exploded["token"].str.split("=", n=1).str[0]
    exploded = exploded.drop_duplicates(subset=["_txn_idx", "label"])

    grouped = (
        exploded.groupby("_txn_idx", sort=True)["label"]
        .apply(lambda s: sorted(s.tolist()))
    )
    transactions: list[list[str]] = grouped.tolist()

    logger.info(
        "Loaded %d transactions from %s%s.",
        len(transactions),
        csv_path.name,
        f" (k={k_value})" if k_value is not None else "",
    )
    return transactions


# ─────────────────────────────────────────────────────────────────────────────
# Boolean matrix
# ─────────────────────────────────────────────────────────────────────────────

def _build_boolean_matrix(
    transactions: list[list[str]],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the one-hot boolean matrix for FP-Growth.

    Uses numpy advanced indexing with pre-computed row/column index arrays —
    no Python loop over transactions.

    Returns (bool_df, sorted_item_names).
    """
    all_items: list[str] = sorted({item for txn in transactions for item in txn})
    item_index: dict[str, int] = {item: i for i, item in enumerate(all_items)}
    n_txn  = len(transactions)
    n_item = len(all_items)

    # Build parallel row/col index arrays for all (transaction, item) pairs.
    row_idx = np.fromiter(
        (r for r, txn in enumerate(transactions) for _ in txn),
        dtype=np.intp,
        count=sum(len(t) for t in transactions),
    )
    col_idx = np.fromiter(
        (item_index[it] for txn in transactions for it in txn),
        dtype=np.intp,
        count=len(row_idx),
    )

    arr = np.zeros((n_txn, n_item), dtype=bool)
    arr[row_idx, col_idx] = True
    return pd.DataFrame(arr, columns=all_items), all_items


# ─────────────────────────────────────────────────────────────────────────────
# FP-Growth + rule generation
# ─────────────────────────────────────────────────────────────────────────────

def _mine_frequent_itemsets(bool_df: pd.DataFrame, min_support: float) -> pd.DataFrame:
    """Run FP-Growth on the pre-built boolean matrix."""
    try:
        from mlxtend.frequent_patterns import fpgrowth
    except ImportError as exc:
        raise ImportError("pip install mlxtend") from exc

    if bool_df.empty:
        return pd.DataFrame(columns=["support", "itemsets"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fpgrowth(bool_df, min_support=min_support, use_colnames=True)


# ─────────────────────────────────────────────────────────────────────────────
# Grid search
# ─────────────────────────────────────────────────────────────────────────────

def _linspace(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    v = start
    while v <= stop + 1e-9:
        values.append(round(v, 6))
        v += step
    return values


def _compute_data_driven_steps(
    transactions: list[list[str]],
    min_support: float,
    max_support: float,
    min_confidence: float,
    max_confidence: float,
) -> tuple[float, float]:
    """
    Derive grid step sizes from the natural granularity of the transaction data.

    Strategy
    ────────
    The minimum meaningful step on the support axis is ``1 / n_transactions``
    (one transaction changes the support by exactly this amount).  However, with
    few distinct items the number of *distinct* support values after FP-Growth is
    much smaller than n_transactions — typically of order (n_items choose k).
    Rather than targeting the finest possible resolution we aim for
    ``_GRID_TARGET_STEPS`` evenly-spaced levels across each axis range, rounded
    to the nearest "human-readable" value from a fixed candidate set.

    Candidate steps (human-readable fractions):
        0.005, 0.01, 0.02, 0.025, 0.04, 0.05, 0.1

    The smallest candidate that still produces ≥ _GRID_TARGET_STEPS levels
    across the range is chosen.  If n_transactions is very small the step floor
    is raised to ``1 / n_transactions`` so we never request more levels than
    the data can provide.

    Parameters
    ----------
    transactions   : Loaded macroscopic transactions.
    min_support    : Lower bound of the support range.
    max_support    : Upper bound of the support range.
    min_confidence : Lower bound of the confidence range.
    max_confidence : Upper bound of the confidence range.

    Returns
    -------
    (support_step, confidence_step) — both rounded to 6 decimal places.
    """
    n = max(1, len(transactions))

    # Minimum step imposed by the data (one transaction = one unit of support).
    data_floor = round(1.0 / n, 6)

    # Candidate human-readable steps, from finest to coarsest.
    candidates = [0.005, 0.01, 0.02, 0.025, 0.04, 0.05, 0.1]

    def _pick(range_width: float) -> float:
        ideal = range_width / _GRID_TARGET_STEPS
        # Use the finest candidate that is ≥ both the data floor and the ideal.
        for c in candidates:
            if c >= data_floor and c >= ideal:
                return c
        return candidates[-1]   # fallback: coarsest candidate

    sup_step  = _pick(max_support    - min_support)
    conf_step = _pick(max_confidence - min_confidence)

    logger.info(
        "  Data-driven grid steps: support=%.4f  confidence=%.4f  "
        "(n_transactions=%d, floor=%.6f, target_steps=%d)",
        sup_step, conf_step, n, data_floor, _GRID_TARGET_STEPS,
    )
    return sup_step, conf_step


def _build_grid(
    min_support: float, max_support: float, support_step: float,
    min_confidence: float, max_confidence: float, confidence_step: float,
) -> list[tuple[float, float]]:
    return list(itertools.product(
        _linspace(min_support,    max_support,    support_step),
        _linspace(min_confidence, max_confidence, confidence_step),
    ))


def run_grid_search(
    transactions: list[list[str]],
    min_support:            float         = DEFAULT_MIN_SUPPORT,
    max_support:            float         = DEFAULT_MAX_SUPPORT,
    support_step:           Optional[float] = DEFAULT_SUPPORT_STEP,
    min_confidence:         float         = DEFAULT_MIN_CONFIDENCE,
    max_confidence:         float         = DEFAULT_MAX_CONFIDENCE,
    confidence_step:        Optional[float] = DEFAULT_CONFIDENCE_STEP,
    lift_independence_low:  float         = DEFAULT_LIFT_INDEPENDENCE_LOW,
    lift_independence_high: float         = DEFAULT_LIFT_INDEPENDENCE_HIGH,
    n_workers:              int           = _DEFAULT_ARM_WORKERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full grid search over (support, confidence) parameter space.

    Adaptive strategy
    ─────────────────
    The optimal implementation depends on how many rules FP-Growth produces,
    which in turn depends on the number of distinct feature labels (= n_items)
    in the transaction set.  This varies with ``--columns``:

    • Few items  (e.g. 3 columns → ≤ 12 rules/level):
      ``association_rules()`` is called once per support level at the global
      min_confidence floor.  Per-cell filtering is a vectorised numpy mask.
      Thread scheduling overhead would exceed useful work here.

    • Many items (e.g. 10 columns → up to thousands of rules/level):
      A ``ThreadPoolExecutor`` calls ``association_rules()`` concurrently for
      each (support, confidence) grid cell.  Rule generation cost is large
      enough that parallelism pays off.

    The threshold (``_VECTOR_RULE_THRESHOLD = 500``) is measured by probing
    the actual rule count at the lowest support value before starting the grid,
    so the decision is purely data-driven.

    Step resolution
    ───────────────
    When support_step or confidence_step is None (the default), the step is
    computed automatically via ``_compute_data_driven_steps()``.

    Parameters
    ----------
    n_workers : Thread-pool size used in the parallel path only.
                Ignored when the vectorised path is selected.

    Returns (all_rules_df, grid_summary_df, freq_itemsets_at_min_support).
    freq_itemsets_at_min_support is the FP-Growth result at the lowest support
    threshold — used by callers to save the frequent itemsets CSV.
    """
    # ── Resolve data-driven steps if not explicitly provided ──────────────────
    if support_step is None or confidence_step is None:
        auto_sup, auto_conf = _compute_data_driven_steps(
            transactions, min_support, max_support, min_confidence, max_confidence,
        )
        if support_step is None:
            support_step = auto_sup
        if confidence_step is None:
            confidence_step = auto_conf

    grid = _build_grid(
        min_support, max_support, support_step,
        min_confidence, max_confidence, confidence_step,
    )
    unique_supports    = sorted({s for s, _ in grid})
    unique_confidences = sorted({c for _, c in grid})
    logger.info(
        "  Grid: %d cells (%d sup × %d conf), workers=%d.",
        len(grid), len(unique_supports), len(unique_confidences), n_workers,
    )

    bool_df, _ = _build_boolean_matrix(transactions)

    freq_cache: dict[float, pd.DataFrame] = {}
    for sup in unique_supports:
        freq_cache[sup] = _mine_frequent_itemsets(bool_df, min_support=sup)

    # Lazy import — kept here to avoid catboost/sklearn overhead on stage-1 runs.
    try:
        from mlxtend.frequent_patterns import association_rules as _ar
    except ImportError as exc:
        raise ImportError("pip install mlxtend") from exc

    # ── Step 3: adaptive rule generation + per-cell filtering ────────────────
    #
    # The optimal strategy depends on n_items (= number of distinct feature
    # labels in the transaction set), which is not known at import time and
    # varies with --columns:
    #
    #   Few items  (≤ _VECTOR_RULE_THRESHOLD total rules at min support)
    #     → Call association_rules() ONCE per support level at the global
    #       min_confidence floor, cache the full rules DataFrame, then apply
    #       all per-cell filters as vectorised numpy masks.  Thread scheduling
    #       overhead would exceed useful work here.
    #
    #   Many items (> _VECTOR_RULE_THRESHOLD total rules at min support)
    #     → Use a ThreadPoolExecutor to call association_rules() concurrently
    #       for each (support, confidence) grid cell.  Each call is expensive
    #       enough that parallelism pays off.
    #
    # The threshold is measured empirically after the first FP-Growth run, so
    # the decision is data-driven and independent of any assumption about the
    # number of columns used.

    # Probe: generate rules at the lowest support and lowest confidence to
    # measure worst-case rule volume for this dataset.
    _VECTOR_RULE_THRESHOLD = 500   # rules per support level

    probe_freq = freq_cache[unique_supports[0]]
    probe_rules_count = 0
    if not probe_freq.empty:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _probe = _ar(probe_freq, metric="confidence",
                             min_threshold=min_confidence)
                probe_rules_count = len(_probe)
            except Exception:
                probe_rules_count = 0

    use_vectorised = probe_rules_count <= _VECTOR_RULE_THRESHOLD
    logger.info(
        "  Rule generation strategy: %s  "
        "(probe rules at min_support=%.4f: %d, threshold=%d)",
        "vectorised" if use_vectorised else "parallel",
        unique_supports[0], probe_rules_count, _VECTOR_RULE_THRESHOLD,
    )

    cell_results: dict[tuple[float, float], pd.DataFrame] = {}

    if use_vectorised:
        # ── Vectorised path ───────────────────────────────────────────────────
        # Call association_rules() once per support level at global min_confidence,
        # then filter each cell with numpy boolean masks — no thread overhead.
        rules_cache: dict[float, pd.DataFrame] = {}
        for sup in unique_supports:
            fi = freq_cache[sup]
            if fi.empty:
                rules_cache[sup] = pd.DataFrame()
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    r = _ar(fi, metric="confidence",
                            min_threshold=min_confidence)
                except Exception:
                    r = pd.DataFrame()
            rules_cache[sup] = r

        for sup, conf in grid:
            base = rules_cache[sup]
            if base.empty:
                cell_results[(sup, conf)] = base
                continue
            mask = (
                (base["support"]    <= max_support)
                & (base["confidence"] >= conf)
                & (base["confidence"] <= max_confidence)
                & (
                    (base["lift"] < lift_independence_low)
                    | (base["lift"] > lift_independence_high)
                )
            )
            filtered = base.loc[mask]
            if not filtered.empty:
                # assign() returns a copy only when there are actual rows,
                # avoiding an unconditional copy of the full DataFrame.
                filtered = filtered.assign(grid_support=sup, grid_confidence=conf)
            cell_results[(sup, conf)] = filtered

    else:
        # ── Parallel path ─────────────────────────────────────────────────────
        # Each (support, confidence) cell calls association_rules() independently.
        # Rule generation is expensive enough that parallelism pays off.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _eval_cell_parallel(
            sup: float, conf: float, freq_itemsets: pd.DataFrame,
        ) -> tuple[float, float, pd.DataFrame]:
            if freq_itemsets.empty:
                return sup, conf, pd.DataFrame()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    rules = _ar(freq_itemsets, metric="confidence",
                                min_threshold=conf)
                except Exception:
                    return sup, conf, pd.DataFrame()
            if rules.empty:
                return sup, conf, pd.DataFrame()
            mask = (
                (rules["support"]    <= max_support)
                & (rules["confidence"] <= max_confidence)
                & (
                    (rules["lift"] < lift_independence_low)
                    | (rules["lift"] > lift_independence_high)
                )
            )
            filtered = rules.loc[mask]
            if not filtered.empty:
                filtered = filtered.copy()
                filtered["grid_support"]    = sup
                filtered["grid_confidence"] = conf
            return sup, conf, filtered

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_map = {
                executor.submit(
                    _eval_cell_parallel, sup, conf, freq_cache[sup]
                ): (sup, conf)
                for sup, conf in grid
            }
            for future in as_completed(future_map):
                sup, conf, rules_df = future.result()
                cell_results[(sup, conf)] = rules_df

    grid_summary = pd.DataFrame([
        {"min_support": sup, "min_confidence": conf,
         "n_rules": len(cell_results[(sup, conf)])}
        for sup, conf in grid
    ])

    non_empty = [df for df in cell_results.values() if not df.empty]
    if not non_empty:
        all_rules_df = pd.DataFrame(columns=[
            "antecedents", "consequents", "antecedent support",
            "consequent support", "support", "confidence", "lift",
            "leverage", "conviction", "zhangs_metric",
            "grid_support", "grid_confidence",
        ])
    else:
        combined = pd.concat(non_empty, ignore_index=True)
        combined["_rule_key"] = (
            combined["antecedents"].apply(
                lambda fs: "|".join(sorted(fs)) if isinstance(fs, frozenset) else str(fs)
            )
            + "→"
            + combined["consequents"].apply(
                lambda fs: "|".join(sorted(fs)) if isinstance(fs, frozenset) else str(fs)
            )
        )
        all_rules_df = (
            combined
            .drop_duplicates(subset="_rule_key", keep="first")
            .drop(columns="_rule_key")
            .reset_index(drop=True)
        )

    logger.info("  → %d unique rules found.", len(all_rules_df))
    return all_rules_df, grid_summary, freq_cache[unique_supports[0]]


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap generation
# ─────────────────────────────────────────────────────────────────────────────

# Number of bins for the lift axis in heatmaps.
_LIFT_N_BINS = 20


def _lift_bins(
    all_rules: pd.DataFrame,
    lift_independence_low: float,
    lift_independence_high: float,
) -> tuple[np.ndarray, list[str]]:
    """
    Build lift bin edges that always include the independence window boundaries
    as explicit edges, so the filtered band is clearly visible in the heatmap.

    Returns (edges, tick_labels).
    """
    if all_rules.empty or "lift" not in all_rules.columns:
        edges = np.linspace(0.0, 2.0, _LIFT_N_BINS + 1)
    else:
        lift_min = max(0.0, float(all_rules["lift"].min()) - 0.05)
        lift_max = float(all_rules["lift"].max()) + 0.05
        # Always include the independence boundary values as edges.
        raw_edges = np.linspace(lift_min, lift_max, _LIFT_N_BINS + 1)
        extra = [lift_independence_low, lift_independence_high]
        edges = np.unique(np.sort(np.concatenate([raw_edges, extra])))

    labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(edges) - 1)]
    return edges, labels


def _make_heatmap(
    pivot: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Path,
    subtitle: str = "",
    lift_independence_low: Optional[float] = None,
    lift_independence_high: Optional[float] = None,
    lift_axis: Optional[str] = None,
) -> None:
    """
    Render and save a single heatmap.

    Parameters
    ----------
    pivot                  : Pivot table with n_rules as values.
    title                  : Plot title (bold, large).
    subtitle               : Secondary line below the title showing active filters.
    xlabel / ylabel        : Axis labels.
    save_path              : Output PNG path.
    lift_independence_low  : If given, draw a hatched band on the lift axis.
    lift_independence_high : Upper bound of that band.
    lift_axis              : 'x' or 'y' — which axis carries the lift values.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import seaborn as sns
    except ImportError as exc:
        raise ImportError(
            "matplotlib and seaborn are required for heatmaps.\n"
            "  pip install matplotlib seaborn"
        ) from exc

    fig, ax = plt.subplots(
        figsize=(max(10, pivot.shape[1] * 0.35), max(8, pivot.shape[0] * 0.28))
    )

    # Disable per-cell annotation when the grid is too dense to be readable.
    annotate = pivot.shape[0] * pivot.shape[1] <= 400

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="Blues",          # Darker = more rules.
        linewidths=0.15 if annotate else 0.0,
        linecolor="white",
        annot=annotate,
        fmt="d",
        cbar_kws={"label": "Number of rules"},
        square=False,
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=4)
    if subtitle:
        ax.text(
            0.5, 1.01, subtitle,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=7.5, color="#555555",
            style="italic",
        )
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    # Subsample tick labels when there are too many to read comfortably.
    _MAX_TICKS = 20
    for axis_obj, n_labels in (
        (ax.xaxis, pivot.shape[1]),
        (ax.yaxis, pivot.shape[0]),
    ):
        if n_labels > _MAX_TICKS:
            step = max(1, n_labels // _MAX_TICKS)
            ticks     = axis_obj.get_major_ticks()
            ticklabels = [t.label1 for t in ticks]
            for i, tick in enumerate(ticks):
                tick.set_visible(i % step == 0)

    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)

    # ── Annotate the lift independence window on the correct axis ─────────────
    if (
        lift_independence_low is not None
        and lift_independence_high is not None
        and lift_axis is not None
    ):
        labels_on_axis = (
            list(pivot.columns) if lift_axis == "x" else list(pivot.index)
        )

        def _label_in_window(lbl: str) -> bool:
            """True if the bin label's lower edge is inside the independence window."""
            try:
                lo = float(str(lbl).split("–")[0])
                hi = float(str(lbl).split("–")[1])
                # The bin overlaps the independence window.
                return lo < lift_independence_high and hi > lift_independence_low
            except Exception:
                return False

        window_indices = [i for i, lbl in enumerate(labels_on_axis) if _label_in_window(lbl)]

        for idx in window_indices:
            if lift_axis == "x":
                ax.add_patch(mpatches.Rectangle(
                    (idx, 0), 1, len(pivot.index),
                    fill=True, facecolor="red", alpha=0.12,
                    hatch="///", edgecolor="red", linewidth=0,
                    zorder=3,
                ))
            else:
                ax.add_patch(mpatches.Rectangle(
                    (0, idx), len(pivot.columns), 1,
                    fill=True, facecolor="red", alpha=0.12,
                    hatch="///", edgecolor="red", linewidth=0,
                    zorder=3,
                ))

        if window_indices:
            legend_patch = mpatches.Patch(
                facecolor="red", alpha=0.25, hatch="///", edgecolor="red",
                label=f"Lift independence window [{lift_independence_low:.2f}, "
                      f"{lift_independence_high:.2f}] — discarded",
            )
            ax.legend(
                handles=[legend_patch],
                loc="upper right",
                fontsize=8,
                framealpha=0.85,
            )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("    Heatmap saved → %s", save_path.name)


def generate_heatmaps(
    all_rules: pd.DataFrame,
    grid_summary: pd.DataFrame,
    heatmap_dir: Path,
    suffix: str,
    lift_independence_low: float,
    lift_independence_high: float,
    k_label: str,
    min_support: Optional[float]    = None,
    max_support: Optional[float]    = None,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
) -> None:
    """
    Generate the three heatmaps for a given k (or all-k) run.

    Each heatmap shows how many rules were found at each parameter combination.
    A subtitle records the active filter thresholds so every PNG is self-
    documenting.

    Heatmap 1 — Support × Confidence  (grid-threshold view)
    ─────────────────────────────────────────────────────────
    rows  = min_support thresholds used in the grid
    cols  = min_confidence thresholds used in the grid
    cell  = n_rules found at that (min_support, min_confidence) pair
            (sourced from grid_summary, which already has the exact counts)

    Heatmap 2 — Support × Lift  (actual-metric view)
    ──────────────────────────────────────────────────
    rows  = actual rule support values (binned)
    cols  = actual rule lift values (binned, with independence window hatched)
    cell  = number of surviving rules in that (support_bin, lift_bin) cell

    Heatmap 3 — Confidence × Lift  (actual-metric view)
    ─────────────────────────────────────────────────────
    rows  = actual rule confidence values (binned)
    cols  = actual rule lift values (binned, with independence window hatched)
    cell  = number of surviving rules in that (confidence_bin, lift_bin) cell
    """
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    # ── Subtitle: active filter thresholds ────────────────────────────────────
    parts: list[str] = []
    if min_support is not None and max_support is not None:
        parts.append(f"support ∈ [{min_support:.3f}, {max_support:.3f}]")
    if min_confidence is not None and max_confidence is not None:
        parts.append(f"confidence ∈ [{min_confidence:.3f}, {max_confidence:.3f}]")
    parts.append(
        f"lift kept: <{lift_independence_low:.2f} (neg.) "
        f"or >{lift_independence_high:.2f} (pos.) | "
        f"discarded: [{lift_independence_low:.2f}, {lift_independence_high:.2f}]"
    )
    subtitle = "   ·   ".join(parts)

    # ── Heatmap 1: Support × Confidence (grid thresholds, from grid_summary) ──
    if not grid_summary.empty:
        # Use the actual column names regardless of whether they come from
        # grid_summary (min_support/min_confidence) or from a rules CSV that
        # was already renamed (grid_min_support/grid_min_confidence).
        sup_col  = "min_support"    if "min_support"    in grid_summary.columns else "grid_min_support"
        conf_col = "min_confidence" if "min_confidence" in grid_summary.columns else "grid_min_confidence"

        pivot_sc = grid_summary.pivot_table(
            index=sup_col,
            columns=conf_col,
            values="n_rules",
            aggfunc="sum",
            fill_value=0,
        )
        pivot_sc.index   = [f"{v:.3f}" for v in pivot_sc.index]
        pivot_sc.columns = [f"{v:.3f}" for v in pivot_sc.columns]
        pivot_sc.index.name   = "min_support (threshold)"
        pivot_sc.columns.name = "min_confidence (threshold)"

        _make_heatmap(
            pivot=pivot_sc,
            title=f"Rules found per (min_support, min_confidence) threshold  [{k_label}]",
            subtitle=subtitle,
            xlabel="min_confidence threshold",
            ylabel="min_support threshold",
            save_path=heatmap_dir / f"heatmap_support_confidence{suffix}.png",
        )

    # ── Heatmaps 2 & 3 use actual rule metric values ──────────────────────────
    if all_rules.empty or "lift" not in all_rules.columns:
        logger.info("    No rules with lift values — skipping lift heatmaps.")
        return

    # Detect renamed columns (rules may come from a post-format DataFrame).
    sup_col_r  = "support"
    conf_col_r = "confidence"
    if sup_col_r not in all_rules.columns or conf_col_r not in all_rules.columns:
        logger.info("    Missing support/confidence columns — skipping lift heatmaps.")
        return

    lift_edges, lift_labels = _lift_bins(
        all_rules, lift_independence_low, lift_independence_high
    )

    def _bin_axis(series: pd.Series, n_bins: int = 20) -> tuple[list[str], pd.Series]:
        """Bin a continuous series into n_bins equal-width bins, return labels and binned series."""
        lo, hi = float(series.min()), float(series.max())
        if lo == hi:
            edges = np.array([lo - 0.01, hi + 0.01])
        else:
            edges = np.linspace(lo, hi, n_bins + 1)
        edges = np.round(edges, 4)
        labels = [f"{edges[i]:.3f}–{edges[i+1]:.3f}" for i in range(len(edges) - 1)]
        binned = pd.cut(series, bins=edges, labels=labels,
                        include_lowest=True).astype(str)
        return labels, binned

    rules_work = all_rules.copy()

    # Bin the lift axis (shared across heatmaps 2 and 3).
    rules_work["lift_bin"] = pd.cut(
        rules_work["lift"],
        bins=lift_edges,
        labels=lift_labels,
        include_lowest=True,
    ).astype(str)

    # ── Heatmap 2: Support × Lift (actual values) ─────────────────────────────
    sup_labels, rules_work["sup_bin"] = _bin_axis(rules_work[sup_col_r])

    pivot_sl = (
        rules_work.groupby(["sup_bin", "lift_bin"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=sup_labels,  fill_value=0)
        .reindex(columns=lift_labels, fill_value=0)
    )
    pivot_sl.index.name   = "support (actual)"
    pivot_sl.columns.name = "lift (actual)"

    _make_heatmap(
        pivot=pivot_sl,
        title=f"Rules by actual Support × Lift  [{k_label}]",
        subtitle=subtitle,
        xlabel="lift (actual value, binned)",
        ylabel="support (actual value, binned)",
        save_path=heatmap_dir / f"heatmap_support_lift{suffix}.png",
        lift_independence_low=lift_independence_low,
        lift_independence_high=lift_independence_high,
        lift_axis="x",
    )

    # ── Heatmap 3: Confidence × Lift (actual values) ──────────────────────────
    conf_labels, rules_work["conf_bin"] = _bin_axis(rules_work[conf_col_r])

    pivot_cl = (
        rules_work.groupby(["conf_bin", "lift_bin"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=conf_labels,  fill_value=0)
        .reindex(columns=lift_labels, fill_value=0)
    )
    pivot_cl.index.name   = "confidence (actual)"
    pivot_cl.columns.name = "lift (actual)"

    _make_heatmap(
        pivot=pivot_cl,
        title=f"Rules by actual Confidence × Lift  [{k_label}]",
        subtitle=subtitle,
        xlabel="lift (actual value, binned)",
        ylabel="confidence (actual value, binned)",
        save_path=heatmap_dir / f"heatmap_confidence_lift{suffix}.png",
        lift_independence_low=lift_independence_low,
        lift_independence_high=lift_independence_high,
        lift_axis="x",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_lift_type(
    rules_df: pd.DataFrame,
    lift_independence_low: float,
    lift_independence_high: float,
) -> pd.DataFrame:
    """
    Add a ``lift_type`` column that makes the filter decision self-documenting:

      "positive_correlation"  — lift > lift_independence_high  (kept)
      "negative_correlation"  — lift < lift_independence_low   (kept)
      "independent"           — lift in [low, high]            (discarded; should
                                never appear in filtered output)
    """
    df = rules_df.copy()
    if "lift" not in df.columns:
        return df
    conditions = [
        df["lift"] > lift_independence_high,
        df["lift"] < lift_independence_low,
    ]
    choices = ["positive_correlation", "negative_correlation"]
    df["lift_type"] = np.select(conditions, choices, default="independent")
    return df


def _format_rules_for_csv(
    rules_df: pd.DataFrame,
    lift_independence_low: float  = DEFAULT_LIFT_INDEPENDENCE_LOW,
    lift_independence_high: float = DEFAULT_LIFT_INDEPENDENCE_HIGH,
    min_support: Optional[float]    = None,
    max_support: Optional[float]    = None,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
) -> pd.DataFrame:
    """
    Prepare the rules DataFrame for CSV output.

    Column layout (in order)
    ────────────────────────
    antecedents          — feature label(s) in the antecedent (human-readable)
    consequents          — feature label(s) in the consequent
    antecedent support   — P(antecedent)
    consequent support   — P(consequent)
    support              — P(antecedent ∪ consequent)  [actual rule metric]
    confidence           — P(consequent | antecedent)  [actual rule metric]
    lift                 — observed / expected co-occurrence [actual rule metric]
    leverage             — P(A∪C) - P(A)·P(C)
    conviction           — (1 - P(C)) / (1 - confidence)
    lift_type            — "positive_correlation" or "negative_correlation"
    grid_min_support     — min_support threshold at which this rule was found
    grid_min_confidence  — min_confidence threshold at which this rule was found
    filter_min_support   — global lower bound of the support grid
    filter_max_support   — global upper bound of the support grid
    filter_min_confidence — global lower bound of the confidence grid
    filter_max_confidence — global upper bound of the confidence grid
    filter_lift_kept_below   — lift < this → kept (negative correlation)
    filter_lift_kept_above   — lift > this → kept (positive correlation)
    filter_lift_discarded    — interval [low, high] that was discarded
    k_value              — k value (present only in per-k files)

    Columns dropped (not requested)
    ────────────────────────────────
    zhangs_metric, jaccard, certainty, kulczynski, representativity
    """
    df = rules_df.copy()

    # ── Convert frozenset columns to readable strings ─────────────────────────
    for col in ("antecedents", "consequents"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: " & ".join(sorted(x)) if isinstance(x, frozenset) else str(x)
            )

    # ── Add lift_type ─────────────────────────────────────────────────────────
    df = _annotate_lift_type(df, lift_independence_low, lift_independence_high)

    # ── Rename grid annotation columns to be self-explanatory ────────────────
    df = df.rename(columns={
        "grid_support":    "grid_min_support",
        "grid_confidence": "grid_min_confidence",
    })

    # ── Add global filter thresholds as per-row columns ──────────────────────
    if min_support    is not None: df["filter_min_support"]    = min_support
    if max_support    is not None: df["filter_max_support"]    = max_support
    if min_confidence is not None: df["filter_min_confidence"] = min_confidence
    if max_confidence is not None: df["filter_max_confidence"] = max_confidence
    df["filter_lift_kept_below"]  = lift_independence_low
    df["filter_lift_kept_above"]  = lift_independence_high
    df["filter_lift_discarded"]   = f"[{lift_independence_low}, {lift_independence_high}]"

    # ── Drop columns that are not needed ─────────────────────────────────────
    _DROP = {
        "zhangs_metric",
        "jaccard", "certainty", "kulczynski", "representativity",
    }
    df = df.drop(columns=[c for c in _DROP if c in df.columns])

    # ── Sanitise inf / -inf / NaN in numeric columns ──────────────────────────
    # conviction = (1-P(C))/(1-conf) → inf when confidence=1.0 (mathematically
    # correct but unreadable in CSV).  Replace with a large finite sentinel and
    # replace NaN with empty string equivalent for float columns.
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

    # ── Enforce canonical column order ────────────────────────────────────────
    _ORDERED = [
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
    df = df[present + extra]

    return df


def _save_rules(
    all_rules: pd.DataFrame,
    grid_summary: pd.DataFrame,
    dest_dir: Path,
    filename_stem: str,
    lift_independence_low: float    = DEFAULT_LIFT_INDEPENDENCE_LOW,
    lift_independence_high: float   = DEFAULT_LIFT_INDEPENDENCE_HIGH,
    min_support: Optional[float]    = None,
    max_support: Optional[float]    = None,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
) -> None:
    """
    Write arm_rules and arm_grid_summary CSVs to dest_dir.

    Rules CSV
    ─────────
    Each row contains the actual rule metrics (support, confidence, lift),
    antecedent/consequent supports, lift_type, the grid thresholds at which
    the rule was found (grid_min_support, grid_min_confidence), and the global
    filter bounds — making every row fully self-describing.
    Columns not requested (zhangs_metric, jaccard, certainty,
    kulczynski, representativity) are dropped.

    Grid summary CSV
    ────────────────
    One row per (min_support, min_confidence) grid cell with n_rules count
    plus the global filter parameters as extra columns.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    rules_path   = dest_dir / f"{filename_stem}_rules.csv"
    summary_path = dest_dir / f"{filename_stem}_grid_summary.csv"

    rules_csv = _format_rules_for_csv(
        all_rules,
        lift_independence_low=lift_independence_low,
        lift_independence_high=lift_independence_high,
        min_support=min_support,
        max_support=max_support,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )
    rules_csv.to_csv(rules_path, index=False, float_format="%.6f")
    logger.info("    Rules   → %s  (%d rows)", rules_path.name, len(rules_csv))

    gs = grid_summary.copy()
    if min_support    is not None: gs["filter_min_support"]    = min_support
    if max_support    is not None: gs["filter_max_support"]    = max_support
    if min_confidence is not None: gs["filter_min_confidence"] = min_confidence
    if max_confidence is not None: gs["filter_max_confidence"] = max_confidence
    gs["filter_lift_kept_below"]  = lift_independence_low
    gs["filter_lift_kept_above"]  = lift_independence_high
    gs["filter_lift_discarded"]   = f"[{lift_independence_low}, {lift_independence_high}]"
    gs.to_csv(summary_path, index=False, float_format="%.6f")
    logger.info("    Summary → %s  (%d cells)", summary_path.name, len(grid_summary))


# ─────────────────────────────────────────────────────────────────────────────
# Per-k runner
# ─────────────────────────────────────────────────────────────────────────────

def _save_frequent_itemsets(
    freq_itemsets: pd.DataFrame,
    dest_dir: Path,
    filename_stem: str,
    k_val: int,
    min_support: float,
) -> None:
    """
    Save the frequent itemsets DataFrame (output of FP-Growth at min_support)
    to a CSV file.

    Columns written
    ───────────────
    k_value   — k value for this run
    itemset   — frozenset serialised as " & "-joined sorted tokens
    support   — itemset support (fraction of transactions)

    Sorted descending by support so the most frequent itemsets appear first.
    """
    if freq_itemsets is None or freq_itemsets.empty:
        logger.info("    Frequent itemsets: none at min_support=%.4f", min_support)
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{filename_stem}_frequent_itemsets.csv"

    df = freq_itemsets.copy()
    if "itemsets" in df.columns:
        df["itemsets"] = df["itemsets"].apply(
            lambda x: " & ".join(sorted(x)) if isinstance(x, frozenset) else str(x)
        )
    df.insert(0, "k_value", k_val)
    df = df.sort_values("support", ascending=False).reset_index(drop=True)
    df.to_csv(path, index=False, float_format="%.6f")
    logger.info(
        "    Frequent itemsets → %s  (%d itemsets at min_support=%.4f)",
        path.name, len(df), min_support,
    )


def _run_for_k(
    k_val: int,
    output_dir: Path,
    suffix: str,
    min_support: float, max_support: float, support_step: Optional[float],
    min_confidence: float, max_confidence: float, confidence_step: Optional[float],
    lift_independence_low: float, lift_independence_high: float,
    n_workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full grid search for a single k value.

    Loads transactions from the per-k itemset file, runs the grid search,
    saves results + heatmaps + frequent itemsets CSV into association_rules/k<N>/,
    and returns (all_rules_df, grid_summary_df) for later aggregation.
    """
    logger.info("  ── k=%d ──────────────────────────────────────────", k_val)

    csv_path = _build_input_path(output_dir, suffix, k_val)
    transactions = load_itemsets(csv_path, k_value=None)  # file already filtered
    if not transactions:
        logger.warning("    No transactions for k=%d; skipping.", k_val)
        empty_rules = pd.DataFrame()
        empty_summary = pd.DataFrame(columns=["min_support", "min_confidence", "n_rules"])
        return empty_rules, empty_summary

    all_rules, grid_summary, freq_itemsets = run_grid_search(
        transactions=transactions,
        min_support=min_support, max_support=max_support, support_step=support_step,
        min_confidence=min_confidence, max_confidence=max_confidence,
        confidence_step=confidence_step,
        lift_independence_low=lift_independence_low,
        lift_independence_high=lift_independence_high,
        n_workers=n_workers,
    )

    # Annotate with the k value so the combined dataset is traceable.
    if not all_rules.empty:
        all_rules = all_rules.copy()
        all_rules.insert(0, "k_value", k_val)
    if not grid_summary.empty:
        grid_summary = grid_summary.copy()
        grid_summary.insert(0, "k_value", k_val)

    # Save per-k outputs.
    k_output_dir = _k_dir(output_dir, k_val)
    _save_rules(
        all_rules, grid_summary, k_output_dir, f"arm{suffix}",
        lift_independence_low=lift_independence_low,
        lift_independence_high=lift_independence_high,
        min_support=min_support, max_support=max_support,
        min_confidence=min_confidence, max_confidence=max_confidence,
    )
    _save_frequent_itemsets(
        freq_itemsets, k_output_dir, f"arm{suffix}", k_val, min_support,
    )

    generate_heatmaps(
        all_rules=all_rules,
        grid_summary=grid_summary,
        heatmap_dir=_heatmap_dir(k_output_dir),
        suffix=suffix,
        lift_independence_low=lift_independence_low,
        lift_independence_high=lift_independence_high,
        k_label=f"k={k_val}",
        min_support=min_support, max_support=max_support,
        min_confidence=min_confidence, max_confidence=max_confidence,
    )

    return all_rules, grid_summary


# ─────────────────────────────────────────────────────────────────────────────
# Main stage-3 runner (called from main.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_macroscopic_mining(
    output_dir: Path,
    original_class: list[int],
    k_value: Optional[int]              = DEFAULT_K_VALUE,
    min_support: float                  = DEFAULT_MIN_SUPPORT,
    max_support: float                  = DEFAULT_MAX_SUPPORT,
    support_step: Optional[float]       = DEFAULT_SUPPORT_STEP,
    min_confidence: float               = DEFAULT_MIN_CONFIDENCE,
    max_confidence: float               = DEFAULT_MAX_CONFIDENCE,
    confidence_step: Optional[float]    = DEFAULT_CONFIDENCE_STEP,
    lift_independence_low: float        = DEFAULT_LIFT_INDEPENDENCE_LOW,
    lift_independence_high: float       = DEFAULT_LIFT_INDEPENDENCE_HIGH,
    n_workers: int                      = _DEFAULT_ARM_WORKERS,
) -> None:
    """
    Entry point for stage 3, called by main.py after stage 2 completes.

    For each class in *original_class* the function:
      1. Discovers all available per-k itemset files (or uses a single k if
         --arm-k is specified).
      2. Runs the grid search for each k separately → saves rules + heatmaps
         under association_rules/k<N>/.
      3. Aggregates all per-k rules into a combined dataset → saves under
         association_rules/all_k/.

    Skip-if-exists: if association_rules/all_k/arm[suffix]_all_k_rules.csv already
    exists the entire class is skipped.  Individual per-k runs are skipped if
    their arm[suffix]_rules.csv already exists, or if a sentinel file
    .arm[suffix]_done marks them as previously processed with zero results.

    Parameters
    ----------
    output_dir             : Stage-2 output directory (contains itemset CSVs).
    original_class         : List of class indices to process ([0], [1], [0,1]).
    k_value                : If set, process only that k value; otherwise all
                             k values found in output_dir are processed.
    min_support            : Lower bound of the support grid.
    max_support            : Upper bound filter for support.
    support_step           : Step size for the support grid.
    min_confidence         : Lower bound of the confidence grid.
    max_confidence         : Upper bound filter for confidence.
    confidence_step        : Step size for the confidence grid.
    lift_independence_low  : Lower boundary of the independence lift interval.
    lift_independence_high : Upper boundary of the independence lift interval.
    n_workers              : Thread-pool size for the parallel confidence sweep.
    """
    logger.info("═" * 62)
    logger.info("  ACS INCOME PIPELINE  —  stage 3: macroscopic ARM")
    logger.info("═" * 62)
    logger.info("  Output dir             : %s", output_dir.resolve())
    logger.info("  k selector             : %s",
                k_value if k_value else "auto (all k files found)")
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

    # Ensure the feature_importance sub-folder exists.
    _feature_imp_dir(output_dir)

    orig_classes_requested = sorted(set(original_class))
    suffix_map = {
        c: (f"_class{c}" if len(orig_classes_requested) > 1 else "")
        for c in orig_classes_requested
    }

    for orig_cls in orig_classes_requested:
        suffix = suffix_map[orig_cls]

        # Skip-if-exists: keyed on the all_k combined rules file.
        all_k_rules_path = _all_k_dir(output_dir) / f"arm{suffix}_all_k_rules.csv"
        if all_k_rules_path.exists():
            logger.info(
                "Skipping stage 3 class %d: %s already exists.",
                orig_cls, all_k_rules_path.name,
            )
            continue

        logger.info("── Class %d ──────────────────────────────────────────", orig_cls)

        # Determine which k values to process.
        if k_value is not None:
            k_values_to_run = [k_value]
        else:
            k_values_to_run = _discover_k_values(output_dir, suffix)
            if not k_values_to_run:
                logger.warning(
                    "No per-k itemset files found for suffix='%s'.  "
                    "Falling back to combined itemset file.",
                    suffix,
                )
                # Use combined file, treat as a single pseudo-k labelled 0.
                k_values_to_run = [0]

        logger.info("  k values to process: %s", k_values_to_run)

        all_rules_list:   list[pd.DataFrame] = []
        all_summary_list: list[pd.DataFrame] = []

        for k_val in k_values_to_run:
            # For the fallback (k=0) we use the combined file.
            effective_k = k_val if k_val != 0 else None

            if effective_k is not None:
                k_out = _k_dir(output_dir, k_val)
                k_rules_path   = k_out / f"arm{suffix}_rules.csv"
                k_summary_path = k_out / f"arm{suffix}_grid_summary.csv"
                # Sentinel file written when a k is processed but yields no rules.
                k_sentinel     = k_out / f".arm{suffix}_done"

                if k_rules_path.exists():
                    logger.info("  Skipping k=%d: output already exists.", k_val)
                    # Safe reload: skip empty files (0-row k runs).
                    try:
                        existing = pd.read_csv(k_rules_path)
                        existing_summary = pd.read_csv(k_summary_path)
                        if not existing.empty:
                            all_rules_list.append(existing)
                        if not existing_summary.empty:
                            all_summary_list.append(existing_summary)
                    except pd.errors.EmptyDataError:
                        pass   # truly empty file — nothing to aggregate
                    continue

                if k_sentinel.exists():
                    logger.info("  Skipping k=%d: previously processed, no rules found.", k_val)
                    continue

            try:
                if effective_k is not None:
                    rules_k, summary_k = _run_for_k(
                        k_val=k_val,
                        output_dir=output_dir,
                        suffix=suffix,
                        min_support=min_support, max_support=max_support,
                        support_step=support_step,
                        min_confidence=min_confidence, max_confidence=max_confidence,
                        confidence_step=confidence_step,
                        lift_independence_low=lift_independence_low,
                        lift_independence_high=lift_independence_high,
                        n_workers=n_workers,
                    )
                    # Write sentinel so empty-result k values are not reprocessed.
                    if rules_k.empty:
                        k_sentinel.touch()
                else:
                    # Combined-file fallback.
                    logger.info("  ── combined (all k) ──────────────────────────────")
                    csv_path = _build_input_path(output_dir, suffix, None)
                    transactions = load_itemsets(csv_path)
                    if not transactions:
                        logger.warning("  No transactions; skipping.")
                        continue
                    rules_k, summary_k, _freq_combined = run_grid_search(
                        transactions=transactions,
                        min_support=min_support, max_support=max_support,
                        support_step=support_step,
                        min_confidence=min_confidence, max_confidence=max_confidence,
                        confidence_step=confidence_step,
                        lift_independence_low=lift_independence_low,
                        lift_independence_high=lift_independence_high,
                        n_workers=n_workers,
                    )

            except FileNotFoundError as exc:
                logger.error("  %s", exc)
                continue

            if not rules_k.empty:
                all_rules_list.append(rules_k)
            if not summary_k.empty:
                all_summary_list.append(summary_k)

        # ── Aggregate and save combined all-k results ─────────────────────────
        logger.info("── Aggregating all-k results for class %d …", orig_cls)
        all_k_out = _all_k_dir(output_dir)

        if all_rules_list:
            combined_rules = pd.concat(all_rules_list, ignore_index=True)
            # Deduplicate on rule identity (ignoring k_value column).
            key_cols = [c for c in ("antecedents", "consequents") if c in combined_rules.columns]
            if key_cols:
                def _fs_key(x: object) -> str:
                    if isinstance(x, frozenset):
                        return "|".join(sorted(x))
                    return str(x)
                combined_rules["_rule_key"] = (
                    combined_rules[key_cols[0]].apply(_fs_key)
                    + "→"
                    + combined_rules[key_cols[1]].apply(_fs_key)
                )
                combined_rules = (
                    combined_rules
                    .drop_duplicates(subset="_rule_key", keep="first")
                    .drop(columns="_rule_key")
                    .reset_index(drop=True)
                )
        else:
            combined_rules = pd.DataFrame()

        if all_summary_list:
            combined_summary = (
                pd.concat(all_summary_list, ignore_index=True)
                .groupby(["min_support", "min_confidence"], as_index=False)["n_rules"]
                .sum()
            )
        else:
            combined_summary = pd.DataFrame(
                columns=["min_support", "min_confidence", "n_rules"]
            )

        _save_rules(
            combined_rules, combined_summary, all_k_out, f"arm{suffix}_all_k",
            lift_independence_low=lift_independence_low,
            lift_independence_high=lift_independence_high,
            min_support=min_support, max_support=max_support,
            min_confidence=min_confidence, max_confidence=max_confidence,
        )

        generate_heatmaps(
            all_rules=combined_rules,
            grid_summary=combined_summary,
            heatmap_dir=_heatmap_dir(all_k_out),
            suffix=suffix,
            lift_independence_low=lift_independence_low,
            lift_independence_high=lift_independence_high,
            k_label="all k",
            min_support=min_support, max_support=max_support,
            min_confidence=min_confidence, max_confidence=max_confidence,
        )

    # ── Move stage-2 files into feature_importance/ now that all reads are done ─
    _move_feature_importance_files(output_dir)

    logger.info("═" * 62)
    logger.info("  Stage 3 (ARM) completed.")
    logger.info("  Outputs in: %s", (_arm_root(output_dir)).resolve())
    logger.info("═" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument definitions (consumed by main.py's build_parser)
# ─────────────────────────────────────────────────────────────────────────────

def add_arm_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add stage-3 (ARM) arguments to an existing ArgumentParser.

    Called by main.py's build_parser() to keep all CLI logic in one place.
    All arguments are optional with sensible defaults.
    """
    arm = parser.add_argument_group(
        "ARM hyperparameters  (stage 3 — macroscopic association rule mining)"
    )

    arm.add_argument(
        "--arm-min-support", type=float, default=DEFAULT_MIN_SUPPORT, metavar="S",
        help=f"Minimum support for FP-Growth grid search.  Default: {DEFAULT_MIN_SUPPORT}.",
    )
    arm.add_argument(
        "--arm-max-support", type=float, default=DEFAULT_MAX_SUPPORT, metavar="S",
        help=f"Maximum support upper-bound filter.  Default: {DEFAULT_MAX_SUPPORT}.",
    )
    arm.add_argument(
        "--arm-support-step", type=float, default=None, metavar="S",
        help=(
            "Step size for the support grid.  "
            "Default: auto-computed from the transaction data "
            f"(targets ~{_GRID_TARGET_STEPS} levels, rounded to nearest "
            "human-readable fraction: 0.005, 0.01, 0.02, 0.025, 0.04, 0.05, 0.1)."
        ),
    )
    arm.add_argument(
        "--arm-min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE, metavar="C",
        help=f"Minimum confidence for the grid search.  Default: {DEFAULT_MIN_CONFIDENCE}.",
    )
    arm.add_argument(
        "--arm-max-confidence", type=float, default=DEFAULT_MAX_CONFIDENCE, metavar="C",
        help=f"Maximum confidence upper-bound filter.  Default: {DEFAULT_MAX_CONFIDENCE}.",
    )
    arm.add_argument(
        "--arm-confidence-step", type=float, default=None, metavar="C",
        help=(
            "Step size for the confidence grid.  "
            "Default: auto-computed from the transaction data "
            f"(same algorithm as --arm-support-step, ~{_GRID_TARGET_STEPS} levels)."
        ),
    )
    arm.add_argument(
        "--arm-lift-low", type=float, default=DEFAULT_LIFT_INDEPENDENCE_LOW, metavar="L",
        help=(
            f"Lower boundary of the lift independence interval.  Rules with "
            f"lift >= this AND <= --arm-lift-high are discarded.  "
            f"Default: {DEFAULT_LIFT_INDEPENDENCE_LOW}."
        ),
    )
    arm.add_argument(
        "--arm-lift-high", type=float, default=DEFAULT_LIFT_INDEPENDENCE_HIGH, metavar="L",
        help=(
            f"Upper boundary of the lift independence interval.  "
            f"Default: {DEFAULT_LIFT_INDEPENDENCE_HIGH}."
        ),
    )
    arm.add_argument(
        "--arm-k", type=int, default=None, metavar="K",
        help=(
            "Process only this k value (uses feature_importance_itemsets_k<K>.csv).  "
            "Default: process all k values found in the output directory."
        ),
    )
    arm.add_argument(
        "--arm-workers", type=int, default=_DEFAULT_ARM_WORKERS, metavar="N",
        help=(
            f"Thread-pool size for the ARM grid-search confidence sweep.  "
            f"Default: {_DEFAULT_ARM_WORKERS} (auto-detected)."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry-point guard
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print(
        "macroscopic_data_mining.py is stage 3 of the pipeline and cannot be\n"
        "run independently — the input file path depends on the full set of\n"
        "pipeline parameters (states, years, columns, percentile, …).\n\n"
        "Run the full pipeline via:\n"
        "  python -m src.main [OPTIONS]\n\n"
        "Stage-3 specific options:\n"
        "  --arm-min-support        (default: 0.05)\n"
        "  --arm-max-support        (default: 1.00)\n"
        "  --arm-support-step       (default: 0.05)\n"
        "  --arm-min-confidence     (default: 0.50)\n"
        "  --arm-max-confidence     (default: 1.00)\n"
        "  --arm-confidence-step    (default: 0.10)\n"
        "  --arm-lift-low           (default: 0.75)\n"
        "  --arm-lift-high          (default: 1.25)\n"
        f"  --arm-workers            (default: {_DEFAULT_ARM_WORKERS}, auto-detected)\n"
        "  --arm-k                  (default: None — process all k values found)\n",
        file=sys.stderr,
    )
    sys.exit(1)