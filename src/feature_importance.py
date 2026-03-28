"""
src/feature_importance.py
─────────────────────────
Stage 2 of the ACS Income pipeline: Boundary Crossing Solo Ratio (BoCSoR) XAI.

Adapted from:
  Alfeo et al. (2023) "From local counterfactuals to global feature importance:
  efficient, robust, and model-agnostic explanations for brain connectivity networks"
  Computer Methods and Programs in Biomedicine 236, 107550.

How the original paper uses data
─────────────────────────────────
The paper states that BoCSoR operates on the training set.  In practice the
training set serves as the *starting pool* — the actual counterfactuals and
the instances used for relevance testing are **synthetic**:

  1. Boundary instances are selected from the training set (percentile filter
     on inter-class distance) — these are real training rows.
  2. For each boundary instance, k nearest neighbours of the opposite class
     are retrieved from the training set — these are also real.
  3. Intermediate points are generated via np.linspace between the boundary
     instance and each neighbour — these are **synthetic**, not in any dataset.
  4. The first synthetic midpoint classified as the opposite class becomes
     the closestCF — a **synthetic** point near the decision boundary.
  5. Feature substitution on closestCF creates yet another **synthetic**
     instance, on which model.predict() is called — the model is thus
     probed on an instance it has never seen during training.

The model is therefore queried on synthetic, unseen instances to explore
its decision surface.  This is functionally equivalent to probing with test
data, except the probe points are constructed strategically near the
boundary rather than sampled randomly.

Adaptations for fully-categorical data
───────────────────────────────────────
* Boundary instances and the opposite-class pool come from the TRAINING SET,
  exactly as in the paper.  The test set is used only to compute a held-out
  accuracy estimate.

* No interpolation of intermediate points (not meaningful for categorical
  features).  Instead of generating synthetic midpoints (Algorithm 1 of the
  paper), counterfactuals are the K nearest neighbours of the opposite class
  in hybrid-encoded Manhattan space **that differ in at least one feature**
  (distance > 0).  Cross-class duplicates (identical feature vectors,
  different label) are skipped by over-querying the NN index and filtering.
  This guarantees every counterfactual has ≥ 1 feature to substitute,
  producing clean itemsets for downstream ARM (stages 3–4).

* Feature substitution (Algorithm 2) creates synthetic instances just as in
  the paper: each modified counterfactual (one feature swapped back to the
  original value) is an instance that likely does not exist in the training
  set.  model.predict() is called on these synthetic instances to determine
  which features are relevant — again probing the model on unseen data.

* Hybrid encoding (new):
    - Ordinal columns (AGEP, SCHL, WKHP, ...): rank-based encoding
      normalised per-column via min-max to [0, 1].
    - Nominal columns (all others): one-hot encoding divided by 2.
      Within a single nominal column two samples either share the same
      category (Manhattan distance 0) or differ (Manhattan distance
      0.5 + 0.5 = 1.0, equivalent to Hamming distance 1).
    Both groups therefore contribute values in [0, 1] per original column,
    making ordinal and nominal columns commensurable.

* Hybrid Manhattan distance (raw sum, no global normalisation):
      dist(a,b) = Σ_i |enc_i(a) - enc_i(b)|
    where enc_i ∈ [0, 1] for every encoded column i.  A single nominal
    feature change contributes 1.0; a single ordinal step contributes
    1/(n_levels - 1).  No division by n_cols — the sum is interpretable
    directly as "how many feature-equivalent changes apart".

* Multi-k evaluation via --k:
      Single value K  -> auto-expanded to all odd integers 1..K (plus K if
                         even).  --k 11 -> [1, 3, 5, 7, 9, 11]
      Multiple values -> used as-is.  --k 1 5 11 -> [1, 5, 11]

* Itemset output: one row per boundary instance per k value.  All relevant
  features are on the same row.  Columns: k_value, instance_index,
  features (space-separated names), itemset (space-separated FEATURE=value
  tokens, ARM-ready).  One combined file plus one file per k value.

* Two BoCSoR indices (computed per k):

  1. Per-CF label-level index (bocsor_label_importance.csv):
     For each feature LABEL, counts how many times it is relevant across
     ALL (boundary_instance, counterfactual) swap-and-restore tests,
     divided by the total number of counterfactuals tested.
         BoCSoR_label_k(SCHL) = n_times_SCHL_relevant / n_total_CFs
     This distinguishes a feature relevant for every CF (strong signal)
     from one relevant for a single distant CF (weak signal).

  2. Per-CF value-level index (bocsor_value_importance.csv):
     Same as above but keyed on LABEL=value tokens:
         BoCSoR_value_k(SCHL=Bachelors) = n_times_relevant / n_total_CFs

  The old union-based index (feature_importance.csv) is retained for
  backward compatibility.  It counts boundary instances (not CFs) and
  uses the union of relevant features across all k CFs per instance.

Parallelism and performance
────────────────────────────
* Start method: multiprocessing uses "fork" so worker processes inherit the
  parent's memory (rank maps, encoded arrays, model) without reimporting.
* Brute-force NN index (sklearn, Manhattan metric): built once on the cf-class
  instances and inherited by workers via fork.  Uses BLAS-accelerated distance
  computation — faster than BallTree at high dimensions (147 encoded columns)
  due to the curse of dimensionality.  Used for both boundary selection and
  batched k-NN queries in each worker (one BLAS call per chunk, not per
  instance).
* Batch predict: for each boundary instance the relevance check builds a
  batch of all modified instances (one per differing feature, across all k
  counterfactuals) and calls model.predict() once, instead of one call per
  feature per counterfactual.
* Boundary instance processing: chunks dispatched via ProcessPoolExecutor.
* CatBoost inference uses thread_count=1 per worker to avoid competing pools.
* Worker count defaults to (cpu_count - 2), capped at 14.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import multiprocessing
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from src.constants import (
    AGEP_LABELS,
    WKHP_LABELS,
    SCHL_MAP,
)

logger = logging.getLogger("src.feature_importance")

def _compute_default_workers() -> int:
    """
    Return the default number of worker processes for BoCSoR.

    Formula: max(1, min(14, cpu_count - 2))
    Reserves 2 logical CPUs for the OS and the main process, caps at 14
    to avoid diminishing returns from CatBoost's internal thread pools.
    """
    return max(1, min(14, (os.cpu_count() or 4) - 2))

_DEFAULT_WORKERS = _compute_default_workers()


# ─────────────────────────────────────────────────────────────────────────────
# Ordered category definitions
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Ordered category definitions
#
# Imported from constants.py to avoid duplication: if category labels or
# orderings change in constants.py this dict updates automatically.
# ─────────────────────────────────────────────────────────────────────────────

ORDERED_CATEGORIES: dict[str, list[str]] = {
    "AGEP": AGEP_LABELS,
    "WKHP": WKHP_LABELS,
    # SCHL_MAP keys are already in ascending code order (1–24), so the
    # values() list preserves the semantic educational attainment ordering.
    "SCHL": list(SCHL_MAP.values()),
}


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid encoding  (ordinal → rank-normalised | nominal → one-hot)
# ─────────────────────────────────────────────────────────────────────────────

def build_rank_maps(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """
    Build {column -> {category_label -> rank}} for ORDINAL columns only.

    Ordinal columns are those listed in ORDERED_CATEGORIES; they use the
    declared semantic order (rank 1 = lowest).  Ranks are 1-based and must
    be built from training data only.

    Nominal columns are NOT included here — they are handled by
    build_nominal_maps / encode_hybrid.
    """
    rank_maps: dict[str, dict[str, int]] = {}
    for col in df.columns:
        if col not in ORDERED_CATEGORIES:
            continue
        present = set(df[col].dropna().unique())
        ordered = [c for c in ORDERED_CATEGORIES[col] if c in present]
        ordered += sorted(present - set(ordered))   # unseen values at the end
        rank_maps[col] = {cat: rank for rank, cat in enumerate(ordered, start=1)}
    return rank_maps


def build_nominal_maps(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Build {column -> sorted_categories} for NOMINAL (non-ordinal) columns.

    The sorted list of categories observed in the training data defines the
    one-hot column order.  Must be built from training data only.
    """
    nominal_maps: dict[str, list[str]] = {}
    for col in df.columns:
        if col in ORDERED_CATEGORIES:
            continue
        cats = sorted(df[col].dropna().unique().tolist())
        nominal_maps[col] = cats
    return nominal_maps


def encode_hybrid(
    df: pd.DataFrame,
    rank_maps: dict[str, dict[str, int]],
    nominal_maps: dict[str, list[str]],
    feature_cols: list[str],
) -> np.ndarray:
    """
    Encode a categorical DataFrame into a float32 matrix where every column
    contributes values in [0, 1], making ordinal and nominal distances
    commensurable.

    Ordinal columns (in rank_maps)
    ──────────────────────────────
    Rank-based encoding normalised per-column with min-max to [0, 1]:
        encoded = (rank - 1) / (max_rank - 1)   if max_rank > 1, else 0.
    Unknown values fall back to 0.5 (mid-range neutral).

    Nominal columns (in nominal_maps)
    ──────────────────────────────────
    One-hot encoding, one column per category.  Within a single original
    column the Manhattan distance between two samples is either 0 (same
    category, bits identical) or 2 (different categories, two bits differ).
    Each one-hot column therefore contributes 0 or 1 after dividing by 2 — 
    still in [0, 1].  The /2 normalisation is applied here so that the
    encoded values are 0.0 or 0.5 per bit; the Manhattan distance
    then recovers values in {0, 1} per original nominal column.

    Column order
    ────────────
    Ordinal columns appear in feature_cols order; nominal one-hot columns are
    appended immediately after their source column, in category sort order.
    The `encoded_col_names` helper mirrors this order.

    Parameters
    ----------
    df           : DataFrame whose rows are to be encoded.
    rank_maps    : Output of build_rank_maps (training data only).
    nominal_maps : Output of build_nominal_maps (training data only).
    feature_cols : Ordered list of original feature column names.

    Returns
    -------
    (n_samples, n_encoded_cols) float32 array.
    """
    n = len(df)
    cols_out: list[np.ndarray] = []

    for col in feature_cols:
        if col in rank_maps:
            # ── Ordinal: rank → min-max normalised to [0, 1] ─────────────────
            # Vectorised via pd.Series.map(dict) (C-optimised in pandas) +
            # fillna for unknown values, avoiding a Python-level lambda per row.
            rmap     = rank_maps[col]
            max_rank = max(rmap.values())
            denom    = float(max_rank - 1) if max_rank > 1 else 1.0
            fallback = 0.5
            # Build a pre-normalised mapping: {category -> normalised_rank}.
            norm_map = {cat: (rank - 1) / denom for cat, rank in rmap.items()}
            vals = (
                df[col]
                .map(norm_map)          # C-optimised dict lookup
                .fillna(fallback)       # unknown categories → mid-range
                .to_numpy(dtype=np.float32)
            )
            cols_out.append(vals.reshape(-1, 1))
        else:
            # ── Nominal: one-hot / 2  → each bit in {0.0, 0.5} ──────────────
            cats = nominal_maps.get(col, [])
            col_vals = df[col].to_numpy()
            for cat in cats:
                bit = (col_vals == cat).astype(np.float32) * 0.5
                cols_out.append(bit.reshape(-1, 1))

    if not cols_out:
        return np.empty((n, 0), dtype=np.float32)
    return np.concatenate(cols_out, axis=1)


def encoded_col_names(
    feature_cols: list[str],
    rank_maps: dict[str, dict[str, int]],
    nominal_maps: dict[str, list[str]],
) -> list[str]:
    """
    Return the list of encoded column names in the same order as encode_hybrid.

    Useful for debugging / logging.  Ordinal columns keep their name;
    nominal columns become 'COL=cat' strings.
    """
    names: list[str] = []
    for col in feature_cols:
        if col in rank_maps:
            names.append(col)
        else:
            for cat in nominal_maps.get(col, []):
                names.append(f"{col}={cat}")
    return names


def n_encoded_cols(
    feature_cols: list[str],
    rank_maps: dict[str, dict[str, int]],
    nominal_maps: dict[str, list[str]],
) -> int:
    """
    Return the total number of columns produced by encode_hybrid.

    Utility function — not currently called by the pipeline but retained
    for diagnostics and potential future use.
    """
    total = 0
    for col in feature_cols:
        if col in rank_maps:
            total += 1
        else:
            total += len(nominal_maps.get(col, []))
    return total




# ─────────────────────────────────────────────────────────────────────────────
# k-value expansion
# ─────────────────────────────────────────────────────────────────────────────

def expand_k(k_values: list[int]) -> list[int]:
    """
    Convert the raw --k argument into the list of neighbourhood sizes to use.

    Single value K  ->  all odd integers from 1 to K, plus K if even.
        --k 11  ->  [1, 3, 5, 7, 9, 11]
        --k 4   ->  [1, 3, 4]
        --k 1   ->  [1]

    Multiple values  ->  deduplicated, sorted, used as-is.
        --k 1 5 11  ->  [1, 5, 11]

    Raises ValueError if any value < 1.
    """
    invalid = [v for v in k_values if v < 1]
    if invalid:
        raise ValueError(f"--k values must be >= 1.  Invalid: {invalid}.")
    if len(k_values) == 1:
        k_max    = k_values[0]
        expanded = list(range(1, k_max + 1, 2))
        if k_max % 2 == 0 and k_max not in expanded:
            expanded.append(k_max)
        return expanded
    seen: dict[int, None] = {}
    for v in k_values:
        seen[v] = None
    return sorted(seen.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Boundary instance selection (computed once, shared across all k values)
# ─────────────────────────────────────────────────────────────────────────────

def build_nn_index(X_enc: np.ndarray) -> NearestNeighbors:
    """
    Build a brute-force nearest-neighbour index on hybrid-encoded data.

    Uses sklearn NearestNeighbors with algorithm='brute' and Manhattan
    metric.  At high dimensions (e.g. 147 encoded columns), brute-force
    BLAS distance computation is significantly faster than tree-based
    methods (BallTree/KDTree) which degrade due to the curse of
    dimensionality.

    The index is built once in the main process and inherited by worker
    processes via fork — no serialisation overhead.

    Each encoded column contributes values in [0, 1]:
        dist(a, b) = Σ_i |enc_i(a) - enc_i(b)|
    A single nominal feature change contributes 1.0; a single ordinal
    step contributes 1/(n_levels - 1).
    """
    nn = NearestNeighbors(algorithm="brute", metric="manhattan", n_jobs=1)
    nn.fit(X_enc.astype(np.float32))
    return nn


def select_boundary_instances(
    X_enc: np.ndarray,
    y_true: np.ndarray,
    y_pred_train: np.ndarray,
    percentile_th: float,
    original_class: int,
    cf_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, NearestNeighbors]:
    """
    Compute boundary instances and counterfactual indices once.

    Mirrors the authors' original approach (CounterfactualExplainerByProximity):
    1. Separate instances by TRUE label (y_true), not model predictions, so
       that the class pools are stable and independent of the model.
    2. Keep only orig-class instances that the model ALSO predicts as
       original_class (i.e. correctly classified).  Misclassified instances
       would lie on the wrong side of the decision boundary and their
       counterfactual direction would be semantically inverted.
    3. Select boundary instances as those whose distance to the nearest
       true-cf-class neighbour is <= the percentile_th-th percentile.
       (Original authors use <=; we align with that convention.)

    The NN index is built on the counterfactual-class instances and returned
    so that worker processes can reuse it for the k-NN step (Algorithm 1).

    Returns
    -------
    (boundary_indices, cf_indices, X_enc_cf, cf_nn)
      boundary_indices : row indices in X_enc of boundary instances.
      cf_indices       : row indices in X_enc of true-cf-class instances.
      X_enc_cf         : encoded submatrix for cf-class instances.
      cf_nn            : NearestNeighbors index built on cf-class instances.
    """
    # Separate by TRUE label — same as the original authors.
    orig_indices_true = np.where(y_true == original_class)[0]
    cf_indices        = np.where(y_true == cf_class)[0]

    if len(orig_indices_true) == 0 or len(cf_indices) == 0:
        empty_nn = NearestNeighbors(algorithm="brute", metric="manhattan")
        empty_nn.fit(np.zeros((1, X_enc.shape[1]), dtype=np.float32))
        return (np.array([], dtype=int), cf_indices,
                np.empty((0, X_enc.shape[1]), dtype=np.float32), empty_nn)

    # Keep only correctly classified orig-class instances.
    correct_mask  = y_pred_train[orig_indices_true] == original_class
    orig_indices  = orig_indices_true[correct_mask]

    if len(orig_indices) == 0:
        empty_nn = NearestNeighbors(algorithm="brute", metric="manhattan")
        empty_nn.fit(np.zeros((1, X_enc.shape[1]), dtype=np.float32))
        return (np.array([], dtype=int), cf_indices,
                np.empty((0, X_enc.shape[1]), dtype=np.float32), empty_nn)

    X_enc_orig = X_enc[orig_indices]
    X_enc_cf   = X_enc[cf_indices]

    # Build NN index on the true-cf-class instances.
    cf_nn = build_nn_index(X_enc_cf)

    # Query k=1 to find each orig-class instance's nearest cf-class neighbour.
    X_orig_q = X_enc_orig.astype(np.float32)
    min_dists, _ = cf_nn.kneighbors(X_orig_q, n_neighbors=1)   # (N_orig, 1)
    min_dists    = min_dists[:, 0]                               # (N_orig,)

    # Use <= to match the original authors' convention.
    threshold = float(np.percentile(min_dists, percentile_th))
    b_local   = np.where(min_dists <= threshold)[0]

    boundary_indices = orig_indices[b_local]
    return boundary_indices, cf_indices, X_enc_cf, cf_nn


# ─────────────────────────────────────────────────────────────────────────────
# Worker function — runs in a subprocess, uses batched predict
# ─────────────────────────────────────────────────────────────────────────────

def _process_boundary_chunk(
    chunk_indices: list[int],
    X_enc: np.ndarray,
    X_enc_cf: np.ndarray,
    cf_global_indices: np.ndarray,
    X_train_values: np.ndarray,
    feature_cols: list[str],
    model: CatBoostClassifier,
    k: int,
    original_class: int,
    cf_nn: NearestNeighbors,
) -> tuple[dict[str, int], list[dict], list[dict], dict[str, int], dict[str, int], int]:
    """
    Process one chunk of boundary instances in a worker process.

    For each boundary instance:
      1. Find the k nearest neighbours from the counterfactual class that
         differ in at least one feature (distance > 0).  The NN index is
         queried once for the ENTIRE chunk in a single BLAS call, then
         results are filtered per-instance to skip cross-class duplicates
         (identical features, different label).  This guarantees every
         counterfactual has ≥ 1 feature to substitute.
      2. Build a batch of all modified instances needed for the relevance
         check across all k counterfactuals, then call model.predict() once
         on the entire batch (batched Algorithm 2).
      3. For each CF independently, record which features are relevant
         (per-CF tracking for the new BoCSoR index).
      4. Take the UNION of relevant features across all k counterfactuals
         and emit one itemset row per boundary instance (input for ARM).

    The model is inherited from the parent process via fork — no disk I/O.

    Returns
    -------
    (importance_counts, itemset_rows, distance_rows,
     label_relevance, value_relevance, n_cf_tested)
    """
    n_features = len(feature_cols)
    importance_counts: dict[str, int] = {f: 0 for f in feature_cols}
    itemset_rows: list[dict] = []
    distance_rows: list[dict] = []

    # New per-CF counters for the reformulated BoCSoR index.
    # label_relevance:  {feature_label: count of (instance, CF) pairs where relevant}
    # value_relevance:  {"LABEL=value": count of (instance, CF) pairs where relevant}
    # n_cf_tested:      total number of counterfactuals tested across all instances
    label_relevance: dict[str, int] = {f: 0 for f in feature_cols}
    value_relevance: dict[str, int] = {}
    n_cf_tested: int = 0

    n_cf = len(cf_global_indices)

    feat_to_idx: dict[str, int] = {f: i for i, f in enumerate(feature_cols)}

    # ── Batch k-NN query for ALL instances in this chunk at once ──────────
    # A single BLAS call computes the entire (chunk_size × n_cf) distance
    # matrix, which is orders of magnitude faster than chunk_size separate
    # single-row queries.
    k_query = min(max(k * 50, 200), n_cf)
    chunk_queries = X_enc[chunk_indices].astype(np.float32)
    all_dists, all_tops = cf_nn.kneighbors(chunk_queries, n_neighbors=k_query)

    for row_i, b_idx in enumerate(chunk_indices):
        dists_row = all_dists[row_i]
        tops_row  = all_tops[row_i]

        # Filter: keep only counterfactuals that differ in ≥ 1 feature.
        nonzero = dists_row > 0.0
        top_nz  = tops_row[nonzero]
        dist_nz = dists_row[nonzero]

        k_act = min(k, len(top_nz))
        if k_act == 0:
            continue   # all cf-class neighbours are identical — skip

        cf_idxs  = [int(cf_global_indices[i]) for i in top_nz[:k_act]]
        cf_dists = dist_nz[:k_act].tolist()

        instance_vals = X_train_values[b_idx]

        # Record one distance row per (boundary_instance, k_neighbour) pair.
        # All distances here are guaranteed > 0.
        for rank, (cf_idx, dist) in enumerate(zip(cf_idxs, cf_dists), start=1):
            cf_vals = X_train_values[cf_idx]
            n_diff = int(sum(1 for fi in range(n_features)
                             if cf_vals[fi] != instance_vals[fi]))
            distance_rows.append({
                "instance_index":    int(b_idx),
                "cf_index":          cf_idx,
                "k_neighbour_rank":  rank,
                "distance":          round(float(dist), 6),
                "n_diff_features":   n_diff,
            })

        # ── Algorithm 2: batched relevance check ─────────────────────────────
        # Build one modified instance per (counterfactual, differing_feature)
        # pair, collect their (cf_idx, fi) metadata, then call predict once.
        batch_rows:  list[np.ndarray] = []
        batch_meta:  list[tuple[int, int]] = []   # (cf_idx, feature_index)

        for cf_idx in cf_idxs:
            cf_vals = X_train_values[cf_idx]
            for fi in range(n_features):
                if cf_vals[fi] == instance_vals[fi]:
                    continue
                modified    = cf_vals.copy()
                modified[fi] = instance_vals[fi]
                batch_rows.append(modified)
                batch_meta.append((cf_idx, fi))

        if not batch_rows:
            continue

        # Single predict call for all (counterfactual, feature) pairs.
        batch_array = np.array(batch_rows, dtype=object)
        preds       = model.predict(batch_array, thread_count=1).ravel()

        # Track CFs tested for the reformulated index denominator.
        n_cf_tested += k_act

        # ── Per-CF relevance tracking (new BoCSoR index) ─────────────────
        # For each CF independently, record which features are relevant.
        # This feeds the reformulated index: count / total_CFs_tested.
        per_cf_relevant: dict[int, set[str]] = {}  # cf_idx → {relevant features}
        for pred, (cf_idx, fi) in zip(preds, batch_meta):
            if int(pred) == original_class:
                per_cf_relevant.setdefault(cf_idx, set()).add(feature_cols[fi])

        for cf_idx, feats in per_cf_relevant.items():
            for feat in feats:
                label_relevance[feat] += 1
                token = f"{feat}={instance_vals[feat_to_idx[feat]]}"
                value_relevance[token] = value_relevance.get(token, 0) + 1

        # ── Union across all k CFs (existing logic for itemsets / ARM) ───
        relevant_union: set[str] = set()
        for pred, (_, fi) in zip(preds, batch_meta):
            if int(pred) == original_class:
                relevant_union.add(feature_cols[fi])

        if relevant_union:
            # One row per boundary instance — all relevant features on the same
            # record.  "itemset" is a space-separated string of FEATURE=value
            # tokens (sorted for reproducibility), ready for ARM.
            items_str = " ".join(
                f"{feat}={instance_vals[feat_to_idx[feat]]}"
                for feat in sorted(relevant_union)
            )
            itemset_rows.append({
                "instance_index": int(b_idx),
                "features":       " ".join(sorted(relevant_union)),
                "itemset":        items_str,
            })
            for feat in relevant_union:
                importance_counts[feat] += 1

    return importance_counts, itemset_rows, distance_rows, label_relevance, value_relevance, n_cf_tested


# ─────────────────────────────────────────────────────────────────────────────
# BoCSoR — single k, parallel over boundary chunks
# ─────────────────────────────────────────────────────────────────────────────

def run_bocsor_single_k(
    model: CatBoostClassifier,
    X_train: pd.DataFrame,
    X_enc: np.ndarray,
    X_train_values: np.ndarray,
    boundary_indices: np.ndarray,
    cf_indices: np.ndarray,
    X_enc_cf: np.ndarray,
    cf_nn: NearestNeighbors,
    feature_cols: list[str],
    k: int,
    original_class: int,
    n_workers: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict, pd.Series, pd.Series]:
    """
    Run BoCSoR for a single k value.

    Returns
    -------
    (itemsets_df, feat_imp, distances_df, filter_stats, label_imp, value_imp)
      itemsets_df   : union-based itemsets for ARM (stages 3-4).
      feat_imp      : old union-based BoCSoR index (backward compat).
      distances_df  : per (instance, CF) distances.
      filter_stats  : dict with boundary/filtered/relevant counts.
      label_imp     : new per-CF BoCSoR index at LABEL level.
      value_imp     : new per-CF BoCSoR index at LABEL=value level.
    """
    logger.info(
        "  k=%d | boundary instances: %d (class %d)",
        k, len(boundary_indices), original_class,
    )
    if len(boundary_indices) == 0:
        empty_label = pd.Series(0.0, index=feature_cols, name="BoCSoR_label")
        empty_value = pd.Series(dtype=float, name="BoCSoR_value")
        return (
            pd.DataFrame(columns=["instance_index", "features", "itemset"]),
            pd.Series(0.0, index=feature_cols, name="BoCSoR_importance"),
            pd.DataFrame(columns=["instance_index", "cf_index",
                                   "k_neighbour_rank", "distance", "n_diff_features"]),
            {"k": k, "boundary_instances": 0, "instances_with_cf": 0,
             "instances_filtered_dist0": 0, "pct_filtered": 0.0,
             "instances_with_relevant_features": 0, "total_cf_tested": 0},
            empty_label,
            empty_value,
        )

    # ── Chunk and dispatch ────────────────────────────────────────────────────
    n_chunks   = min(n_workers, len(boundary_indices))
    chunk_size = math.ceil(len(boundary_indices) / n_chunks)
    chunks     = [
        boundary_indices[i : i + chunk_size].tolist()
        for i in range(0, len(boundary_indices), chunk_size)
    ]

    total_importance: dict[str, int] = {f: 0 for f in feature_cols}
    all_rows:      list[dict] = []
    all_dist_rows: list[dict] = []
    n_with_cf = 0

    # Aggregators for the reformulated per-CF BoCSoR index.
    total_label_rel: dict[str, int] = {f: 0 for f in feature_cols}
    total_value_rel: dict[str, int] = {}
    total_cf_tested: int = 0

    _fork_ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=n_chunks, mp_context=_fork_ctx) as executor:
        futures = {
            executor.submit(
                _process_boundary_chunk,
                chunk,
                X_enc,
                X_enc_cf,
                cf_indices,
                X_train_values,
                feature_cols,
                model,
                k,
                original_class,
                cf_nn,
            ): chunk
            for chunk in chunks
        }
        for future in as_completed(futures):
            imp_c, rows_c, dist_c, lab_c, val_c, ncf_c = future.result()
            for feat, cnt in imp_c.items():
                total_importance[feat] += cnt
            all_rows.extend(rows_c)
            all_dist_rows.extend(dist_c)
            n_with_cf += len({r["instance_index"] for r in rows_c})
            # Aggregate per-CF counters.
            for feat, cnt in lab_c.items():
                total_label_rel[feat] = total_label_rel.get(feat, 0) + cnt
            for token, cnt in val_c.items():
                total_value_rel[token] = total_value_rel.get(token, 0) + cnt
            total_cf_tested += ncf_c

    logger.info(
        "  k=%d | instances with >=1 counterfactual: %d / %d",
        k, n_with_cf, len(boundary_indices),
    )
    n_filtered = len(boundary_indices) - len(
        {r["instance_index"] for r in all_dist_rows}
    )
    if n_filtered > 0:
        logger.info(
            "  k=%d | instances skipped (all %d neighbours at distance 0): %d / %d (%.1f%%)",
            k, k, n_filtered, len(boundary_indices),
            n_filtered / len(boundary_indices) * 100,
        )

    filter_stats = {
        "k": k,
        "boundary_instances": len(boundary_indices),
        "instances_with_cf": len({r["instance_index"] for r in all_dist_rows}),
        "instances_filtered_dist0": n_filtered,
        "pct_filtered": round(n_filtered / max(len(boundary_indices), 1) * 100, 2),
        "instances_with_relevant_features": n_with_cf,
        "total_cf_tested": total_cf_tested,
    }

    # ── Old index (union-based, kept for backward compatibility) ──────────
    feat_imp = (
        pd.Series(total_importance, name="BoCSoR_importance")
        / max(n_with_cf, 1)
    ).sort_values(ascending=False)

    # ── New BoCSoR index: per-CF label-level ──────────────────────────────
    # BoCSoR_k(LABEL) = times LABEL was relevant across all (instance, CF)
    #                    pairs / total CFs tested.
    label_imp = (
        pd.Series(total_label_rel, name="BoCSoR_label")
        / max(total_cf_tested, 1)
    ).sort_values(ascending=False)

    # ── New BoCSoR index: per-CF value-level ──────────────────────────────
    # BoCSoR_k(LABEL=value) = times that specific value was relevant /
    #                          total CFs tested.
    value_imp = (
        pd.Series(total_value_rel, name="BoCSoR_value")
        / max(total_cf_tested, 1)
    ).sort_values(ascending=False)

    logger.info(
        "  k=%d | total CFs tested: %d | label index top: %s=%.4f | value index top: %s=%.4f",
        k, total_cf_tested,
        label_imp.index[0] if len(label_imp) > 0 else "N/A",
        label_imp.iloc[0] if len(label_imp) > 0 else 0,
        value_imp.index[0] if len(value_imp) > 0 else "N/A",
        value_imp.iloc[0] if len(value_imp) > 0 else 0,
    )

    dist_df = pd.DataFrame(all_dist_rows) if all_dist_rows else pd.DataFrame(
        columns=["instance_index", "cf_index", "k_neighbour_rank", "distance", "n_diff_features"]
    )
    if not dist_df.empty:
        dist_df = (
            dist_df
            .sort_values(["instance_index", "k_neighbour_rank"], ascending=True)
            .reset_index(drop=True)
        )
    return pd.DataFrame(all_rows), feat_imp, dist_df, filter_stats, label_imp, value_imp


# ─────────────────────────────────────────────────────────────────────────────
# BoCSoR — multi-k entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_bocsor_multi_k(
    model: CatBoostClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    y_pred_train: np.ndarray,
    rank_maps: dict[str, dict[str, int]],
    feature_cols: list[str],
    k_values: list[int],
    percentile_th: float = 20.0,
    original_class: int = 0,
    cf_class: int = 1,
    n_workers: int = _DEFAULT_WORKERS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame], pd.DataFrame,
           pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run BoCSoR once per value in k_values on the training set.

    Key optimisations vs. the naive loop:
    - Hybrid encoding of X_train: computed once, reused for all k values.
      Ordinal columns: rank-based, min-max normalised to [0,1].
      Nominal columns: one-hot / 2, each bit in {0.0, 0.5}.
    - Boundary selection via brute-force NN: BLAS-accelerated distance
      computation, efficient at 147 encoded dimensions.
      Pools separated by TRUE label; only correctly-classified orig-class
      instances are candidates.  Built once, reused for all k values.
    - k-NN (Algorithm 1) in workers: batched brute-force query — one BLAS
      call per chunk instead of one per boundary instance.
    - Workers inherit the NN index from the main process via fork.
    - If zero boundary instances are found, all k values are skipped
      immediately without any tree query per k.
    - Relevance check uses batched model.predict() instead of one call
      per (counterfactual, feature) pair.

    Returns
    -------
    (all_itemsets_df, importance_df, per_k_itemsets, distances_df,
     filter_stats_df, label_imp_df, value_imp_df)
      all_itemsets_df  : columns [k_value, instance_index, features, itemset].
      importance_df    : old union-based BoCSoR (backward compat), columns k_1…k_N.
      per_k_itemsets   : {k -> itemset DataFrame for that k only}.
      distances_df     : columns [k_value, instance_index, cf_index,
                         k_neighbour_rank, distance, n_diff_features].
      filter_stats_df  : one row per k with boundary/filtered/relevant counts.
      label_imp_df     : new per-CF BoCSoR at LABEL level, columns k_1…k_N.
      value_imp_df     : new per-CF BoCSoR at LABEL=value level, columns k_1…k_N.
    """
    logger.info(
        "BoCSoR on TRAINING SET: k=%s  class %d->%d  "
        "percentile=%.1f  workers=%d",
        k_values, original_class, cf_class, percentile_th, n_workers,
    )

    # ── Hybrid encoding: built once, reused for all k values ─────────────────
    nominal_maps = build_nominal_maps(X_train[feature_cols])
    X_enc        = encode_hybrid(X_train[feature_cols], rank_maps, nominal_maps, feature_cols)
    X_train_values = (
        X_train[feature_cols].reset_index(drop=True).to_numpy(dtype=object)
    )

    # ── Compute boundary instances once for all k ─────────────────────────────
    t0 = time.perf_counter()
    boundary_indices, cf_indices, X_enc_cf, cf_nn = select_boundary_instances(
        X_enc=X_enc,
        y_true=y_train.to_numpy().astype(int),
        y_pred_train=y_pred_train,
        percentile_th=percentile_th,
        original_class=original_class,
        cf_class=cf_class,
    )
    t_boundary = time.perf_counter() - t0
    logger.info(
        "Boundary selection: %d instances  (class %d, pct=%.1f, %.1fs)",
        len(boundary_indices), original_class, percentile_th, t_boundary,
    )

    if len(boundary_indices) == 0:
        logger.warning(
            "No boundary instances for class %d -> %d.  "
            "All k values skipped.",
            original_class, cf_class,
        )
        empty_imp = pd.Series(0.0, index=feature_cols, name="BoCSoR_importance")
        empty_df  = pd.DataFrame(columns=["k_value", "instance_index", "features", "itemset"])
        imp_df    = pd.DataFrame(
            {f"k_{k}": empty_imp for k in k_values}
        ).rename_axis("feature")
        empty_dist = pd.DataFrame(
            columns=["k_value", "instance_index", "cf_index",
                     "k_neighbour_rank", "distance", "n_diff_features"]
        )
        empty_fstats = pd.DataFrame(
            columns=["k", "boundary_instances", "instances_with_cf",
                     "instances_filtered_dist0", "pct_filtered",
                     "instances_with_relevant_features", "total_cf_tested"]
        )
        empty_label_imp = pd.DataFrame(
            {f"k_{k}": pd.Series(0.0, index=feature_cols) for k in k_values}
        ).rename_axis("feature")
        empty_value_imp = pd.DataFrame(
            columns=[f"k_{k}" for k in k_values]
        ).rename_axis("feature_value")
        return (empty_df, imp_df, {k: empty_df.copy() for k in k_values},
                empty_dist, empty_fstats, empty_label_imp, empty_value_imp)

    # ── Run once per k (boundary instances reused) ────────────────────────────
    all_itemsets:    list[pd.DataFrame]   = []
    all_distances:   list[pd.DataFrame]   = []
    all_filter_stats: list[dict]          = []
    importance_dict: dict[int, pd.Series] = {}
    label_imp_dict:  dict[int, pd.Series] = {}
    value_imp_dict:  dict[int, pd.Series] = {}

    for k in k_values:
        logger.info("── k=%d ──────────────────────────────────────────────", k)
        t_k = time.perf_counter()
        itemsets_df, feat_imp, dist_df, fstats, label_imp, value_imp = run_bocsor_single_k(
            model=model,
            X_train=X_train,
            X_enc=X_enc,
            X_train_values=X_train_values,
            boundary_indices=boundary_indices,
            cf_indices=cf_indices,
            X_enc_cf=X_enc_cf,
            cf_nn=cf_nn,
            feature_cols=feature_cols,
            k=k,
            original_class=original_class,
            n_workers=n_workers,
        )
        all_filter_stats.append(fstats)
        logger.info("  k=%d | elapsed: %.1fs", k, time.perf_counter() - t_k)
        # Always tag with k_value — even if empty — so pd.concat produces
        # a consistent column schema across all k values.
        if not itemsets_df.empty:
            itemsets_df.insert(0, "k_value", k)
        else:
            itemsets_df = pd.DataFrame(
                columns=["k_value", "instance_index", "features", "itemset"]
            )
        if not dist_df.empty:
            dist_df.insert(0, "k_value", k)
        else:
            dist_df = pd.DataFrame(
                columns=["k_value", "instance_index", "cf_index",
                         "k_neighbour_rank", "distance", "n_diff_features"]
            )
        all_itemsets.append(itemsets_df)
        all_distances.append(dist_df)
        importance_dict[k] = feat_imp
        label_imp_dict[k]  = label_imp
        value_imp_dict[k]  = value_imp

    combined = pd.concat(all_itemsets, ignore_index=True)
    imp_df   = pd.DataFrame(importance_dict).rename_axis("feature")
    imp_df.columns = [f"k_{k}" for k in imp_df.columns]

    # New BoCSoR indices: label-level and value-level, one column per k.
    label_imp_df = pd.DataFrame(label_imp_dict).rename_axis("feature")
    label_imp_df.columns = [f"k_{k}" for k in label_imp_df.columns]

    # Value-level: union of all tokens seen across all k, fill missing with 0.
    all_tokens = sorted({t for s in value_imp_dict.values() for t in s.index})
    value_imp_df = pd.DataFrame(
        {f"k_{k}": pd.Series(v, dtype=float).reindex(all_tokens, fill_value=0.0)
         for k, v in value_imp_dict.items()}
    ).rename_axis("feature_value")

    per_k_itemsets: dict[int, pd.DataFrame] = {
        k: df for k, df in zip(k_values, all_itemsets)
    }

    distances_df = pd.concat(all_distances, ignore_index=True)
    if not distances_df.empty:
        distances_df = (
            distances_df
            .sort_values(
                ["k_value", "instance_index", "k_neighbour_rank"],
                ascending=True,
            )
            .reset_index(drop=True)
        )

    filter_stats_df = pd.DataFrame(all_filter_stats)
    return (combined, imp_df, per_k_itemsets, distances_df,
            filter_stats_df, label_imp_df, value_imp_df)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _infer_target_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.startswith("income_over_")]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"Multiple target-like columns: {candidates}.")
    raise ValueError(
        "No 'income_over_*' column found. "
        "Ensure the file was produced by stage 1 (src/main.py)."
    )


def load_split_data(
    train_path: Optional[Path],
    test_path: Optional[Path],
    dataset_path: Optional[Path],
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    """
    Return (X_train, X_test, y_train, y_test, target_col).

    Pre-split files are preferred.  A single dataset file is split internally
    with a warning recommending stage-1 regeneration.
    """
    if dataset_path is not None:
        logger.warning(
            "Single dataset file '%s' provided. An internal %d/%d stratified "
            "split will be performed (seed=%d). For reproducible results, "
            "regenerate via stage 1 with --test-size %.2f and pass the "
            "resulting files with --train / --test.",
            dataset_path.name,
            int((1 - test_size) * 100), int(test_size * 100),
            random_seed, test_size,
        )
        df           = pd.read_csv(dataset_path, dtype=str)
        target_col   = _infer_target_column(df)
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols]
        y = df[target_col].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_seed,
        )
        return X_train, X_test, y_train, y_test, target_col

    if train_path is None or test_path is None:
        raise ValueError("Provide either --dataset or both --train and --test.")

    train_df = pd.read_csv(train_path, dtype=str)
    test_df  = pd.read_csv(test_path,  dtype=str)
    target_col  = _infer_target_column(train_df)
    feat_train  = [c for c in train_df.columns if c != target_col]
    feat_test   = [c for c in test_df.columns  if c != target_col]
    if set(feat_train) != set(feat_test):
        raise ValueError(
            "Train and test files have different feature columns.\n"
            f"  Train only: {sorted(set(feat_train) - set(feat_test))}\n"
            f"  Test only : {sorted(set(feat_test) - set(feat_train))}"
        )
    return (
        train_df[feat_train],
        test_df[feat_test],
        train_df[target_col].astype(int),
        test_df[target_col].astype(int),
        target_col,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────────────────────────────────────

def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_features: list[str],
    random_seed: int = 42,
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    verbose: bool = False,
    early_stopping_rounds: int | None = None,
) -> CatBoostClassifier:
    """
    Train a CatBoostClassifier on the (fully categorical) training set.

    CatBoost is the right classifier here because:
    - It accepts raw string-valued categorical columns with no manual encoding,
      avoiding the dimensionality explosion of one-hot encoding on columns
      such as OCCP (23 values) or POBP (80+ values).
    - Its ordered target statistics handle high-cardinality categories robustly
      without introducing target leakage.
    - Alfeo et al. (2023) found CatBoost to be the most accurate of the
      three classifiers they tested on the HCP benchmark tasks.

    Parameters
    ----------
    early_stopping_rounds : If set, training stops when the validation loss
                            does not improve for this many consecutive rounds.
                            An internal 80/20 stratified split is used as the
                            eval set.  None = disabled (train for all iterations).
    """
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        cat_features=cat_features,
        random_seed=random_seed,
        verbose=verbose,
        eval_metric="Accuracy",
        task_type="CPU",
    )
    logger.info(
        "Training CatBoost: iterations=%d, lr=%.4f, depth=%d, "
        "cat_features=%d, early_stopping=%s.",
        iterations, learning_rate, depth, len(cat_features),
        early_stopping_rounds if early_stopping_rounds else "disabled",
    )

    if early_stopping_rounds is not None:
        # Hold out 20% of training data as an internal validation set so
        # CatBoost can monitor the eval metric and stop early.
        #
        # Note for BoCSoR: y_pred_train is computed on all of X_train
        # (including this 20% eval split) after training completes, so
        # boundary instances are selected from the full training set.
        # The 20% eval rows are technically slightly less "known" to the
        # model than the 80% training rows, but this is a minor effect
        # and standard practice for early-stopped gradient boosting.
        from sklearn.model_selection import train_test_split as _tts
        X_tr, X_val, y_tr, y_val = _tts(
            X_train, y_train,
            test_size=0.2,
            stratify=y_train,
            random_state=random_seed,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=early_stopping_rounds,
        )
    else:
        model.fit(X_train, y_train)
    return model


class MLPWrapper:
    """
    Wraps sklearn MLPClassifier to expose the same predict() interface as CatBoost.

    MLPClassifier requires numeric inputs, so categorical features are
    integer-encoded: ordinal columns use their semantic rank from rank_maps;
    nominal columns use 1-based lexicographic rank from nominal_maps.
    Features are then standardised (zero-mean, unit-variance) using statistics
    computed at fit time, stored in the wrapper so predict() can apply the
    same transform.

    The CatBoost-specific ``thread_count`` keyword argument is silently
    dropped so that _process_boundary_chunk can call model.predict() with
    the same signature for both classifiers.
    """

    def __init__(
        self,
        mlp_model: object,
        rank_maps: dict[str, dict[str, int]],
        nominal_maps: dict[str, list[str]],
        feature_cols: list[str],
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> None:
        self._model = mlp_model
        self._rank_maps = rank_maps
        self._nominal_maps = nominal_maps
        self._feat_cols = feature_cols
        self._mean = mean
        self._std = std
        # Build a combined int-encoding map: ordinal columns use their
        # semantic rank; nominal columns use 1-based lexicographic index.
        # Unknown categories fall back to 0 (a dedicated sentinel value).
        self._enc: dict[str, dict[str, int]] = {}
        for col in feature_cols:
            if col in rank_maps:
                self._enc[col] = rank_maps[col]
            else:
                cats = nominal_maps.get(col, [])
                self._enc[col] = {cat: i + 1 for i, cat in enumerate(cats)}

    def _encode(self, X: np.ndarray) -> np.ndarray:
        """
        Convert an (N, F) object array of strings to a float64 array.

        Unknown category values are mapped to 0.  If standardisation
        statistics are available the output is standardised.

        Uses pd.Series.map(dict) for arrays with > 1000 rows (C-optimised
        dict lookup in pandas) and falls back to list comprehension for
        small batches where pandas overhead would dominate.
        """
        n_rows = X.shape[0]
        out = np.zeros(X.shape, dtype=np.float64)

        if n_rows > 1000:
            # Large array: pd.Series.map(dict) is ~5× faster than list comp.
            for j, col in enumerate(self._feat_cols):
                rmap = self._enc[col]
                out[:, j] = (
                    pd.Series(X[:, j])
                    .map(rmap)
                    .fillna(0)
                    .to_numpy(dtype=np.float64)
                )
        else:
            # Small batch (worker predict): list comp avoids pandas overhead.
            for j, col in enumerate(self._feat_cols):
                rmap = self._enc[col]
                out[:, j] = np.array(
                    [rmap.get(v, 0) for v in X[:, j]], dtype=np.float64
                )

        if self._mean is not None and self._std is not None:
            out = (out - self._mean) / self._std
        return out

    def predict(self, X, **kwargs) -> np.ndarray:
        """
        Return integer class labels (0 or 1).

        Accepts either a pandas DataFrame or a numpy object array of raw
        string category values.
        """
        kwargs.pop("thread_count", None)  # CatBoost-specific — ignored
        if isinstance(X, pd.DataFrame):
            arr = X[self._feat_cols].to_numpy(dtype=object)
        else:
            arr = np.asarray(X, dtype=object)
        X_enc = self._encode(arr)
        labels = self._model.predict(X_enc)
        return np.asarray(labels).astype(int)


def train_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    rank_maps: dict[str, dict[str, int]],
    nominal_maps: dict[str, list[str]],
    feature_cols: list[str],
    random_seed: int = 42,
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    verbose: bool = False,
    early_stopping_rounds: int | None = None,
) -> MLPWrapper:
    """
    Train an sklearn MLPClassifier on the (fully categorical) training set
    and return it wrapped in an MLPWrapper for a uniform predict() interface.

    The MLP provides a fundamentally different decision boundary geometry
    compared to tree-based classifiers (smooth non-linear surfaces vs.
    axis-aligned splits), making it an ideal complement to CatBoost for
    verifying that BoCSoR results are model-agnostic.

    Hyperparameter mapping from tree parameters
    ────────────────────────────────────────────
    - iterations → max_iter (training epochs)
    - learning_rate → learning_rate_init
    - depth → two hidden layers of width 2^depth (captures model capacity)
    - early_stopping_rounds → early_stopping=True with n_iter_no_change

    Parameters
    ----------
    rank_maps            : {column -> {label -> rank}} from build_rank_maps().
    nominal_maps         : {column -> sorted_categories} from build_nominal_maps().
    feature_cols         : Ordered list of feature column names.
    early_stopping_rounds: Stop if validation loss does not improve for N
                           consecutive rounds.  An internal 10% stratified
                           split is used as the eval set.  None = disabled.
    """
    from sklearn.neural_network import MLPClassifier

    hidden_size = max(16, 2 ** depth)
    hidden_layers = (hidden_size, hidden_size)

    mlp_model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        max_iter=iterations,
        learning_rate_init=learning_rate,
        random_state=random_seed,
        verbose=verbose,
        early_stopping=early_stopping_rounds is not None,
        n_iter_no_change=early_stopping_rounds if early_stopping_rounds else 10,
        validation_fraction=0.1,
    )

    logger.info(
        "Training MLP: hidden_layers=%s, max_iter=%d, lr=%.4f, "
        "early_stopping=%s.",
        hidden_layers, iterations, learning_rate,
        f"{early_stopping_rounds} rounds" if early_stopping_rounds else "disabled",
    )

    # Build wrapper and encode the training data.
    wrapper = MLPWrapper(mlp_model, rank_maps, nominal_maps, feature_cols)
    X_arr = X_train[feature_cols].to_numpy(dtype=object)
    X_enc = wrapper._encode(X_arr)  # (N, F) float64

    # Standardise: compute mean/std on training data and store in wrapper.
    mean = X_enc.mean(axis=0)
    std = X_enc.std(axis=0)
    std[std == 0] = 1.0  # avoid division by zero for constant columns
    X_enc = (X_enc - mean) / std
    wrapper._mean = mean
    wrapper._std = std

    mlp_model.fit(X_enc, y_train)
    return wrapper


def train_model(
    classifier: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    rank_maps: dict[str, dict[str, int]],
    nominal_maps: dict[str, list[str]],
    feature_cols: list[str],
    random_seed: int = 42,
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    verbose: bool = False,
    early_stopping_rounds: int | None = None,
) -> object:
    """
    Dispatch to train_catboost or train_mlp based on *classifier*.

    Parameters
    ----------
    classifier   : "catboost" or "mlp".
    rank_maps    : Ordinal-only rank maps from build_rank_maps().
                   Required for MLP encoding; unused for CatBoost.
    nominal_maps : Nominal category lists from build_nominal_maps().
                   Required for MLP encoding; unused for CatBoost.

    Returns
    -------
    A trained model with a uniform predict(X) -> np.ndarray[int] interface.
    """
    if classifier == "catboost":
        return train_catboost(
            X_train=X_train,
            y_train=y_train,
            cat_features=feature_cols,
            random_seed=random_seed,
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            verbose=verbose,
            early_stopping_rounds=early_stopping_rounds,
        )
    elif classifier == "mlp":
        return train_mlp(
            X_train=X_train,
            y_train=y_train,
            rank_maps=rank_maps,
            nominal_maps=nominal_maps,
            feature_cols=feature_cols,
            random_seed=random_seed,
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            verbose=verbose,
            early_stopping_rounds=early_stopping_rounds,
        )
    else:
        raise ValueError(
            f"Unknown classifier '{classifier}'. "
            "Choose 'catboost' or 'mlp'."
        )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.feature_importance",
        description=(
            "ACS Income pipeline — stage 2: BoCSoR feature importance\n"
            "Global feature importance from local counterfactuals on the\n"
            "TRAINING SET of a fully-categorical ACS dataset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input (choose one):
  Pre-split (recommended):
    --train data/train_2024_ALL_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv \\
    --test  data/test_2024_ALL_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv

  Single file (split internally -- not recommended):
    --dataset data/dataset_2024_ALL_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv

Examples:
  python -m src.feature_importance \\
      --train data/train_2024_NY_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv \\
      --test  data/test_2024_NY_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv

  python -m src.feature_importance \\
      --train data/train_2024_ALL_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv \\
      --test  data/test_2024_ALL_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv \\
      --k 11 --percentile 20 --output-dir results/

  python -m src.feature_importance \\
      --train data/train_2024_ALL_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv \\
      --test  data/test_2024_ALL_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv \\
      --k 1 5 11 --original-class 0 1 --workers 12
        """,
    )

    inp = parser.add_argument_group("Input")
    inp.add_argument("--train",      type=Path, default=None, metavar="CSV")
    inp.add_argument("--test",       type=Path, default=None, metavar="CSV")
    inp.add_argument("--dataset",    type=Path, default=None, metavar="CSV",
                     help="Single unsplit CSV (prefer --train/--test).")
    inp.add_argument("--split-size", type=float, default=0.2, metavar="FRACTION",
                     help="Test fraction when --dataset is used.  Default: 0.2.")

    boc = parser.add_argument_group("BoCSoR hyperparameters")
    boc.add_argument(
        "--k", nargs="+", type=int, default=[11], metavar="K",
        help=(
            "Neighbourhood size(s). Single value K -> auto-expand to all odd "
            "integers 1..K (e.g. --k 11 -> 1 3 5 7 9 11). Multiple values "
            "used as-is (e.g. --k 1 5 11). Default: 11."
        ),
    )
    boc.add_argument(
        "--percentile", type=float, default=20.0, metavar="PCT",
        help=(
            "Percentile for boundary instance selection (0-100). Training "
            "instances whose distance to their nearest opposite-class neighbour "
            "is below this percentile are treated as boundary instances. "
            "Default: 20."
        ),
    )
    boc.add_argument(
        "--original-class", nargs="+", type=int, default=[0],
        choices=[0, 1], metavar="C",
        help=(
            "Class(es) whose boundary instances to explain.  "
            "0 = income <= threshold (default).  "
            "1 = income > threshold.  "
            "0 1 = both classes (produces separate output files per class).  "
            "Default: 0."
        ),
    )

    clf = parser.add_argument_group("Classifier selection")
    clf.add_argument(
        "--classifier", choices=["catboost", "mlp"], default="catboost",
        help=(
            "Classifier to train.  "
            "'catboost' (default) accepts raw string categoricals natively.  "
            "'mlp' (Multi-Layer Perceptron) provides a fundamentally different "
            "decision boundary geometry, useful for verifying model-agnosticity."
        ),
    )

    cb = parser.add_argument_group("Classifier hyperparameters  (shared by CatBoost and MLP)")
    cb.add_argument("--cb-iterations", type=int,   default=500,  metavar="N",
                    help="Boosting rounds / training epochs.  Default: 500.")
    cb.add_argument("--cb-lr",         type=float, default=0.05, metavar="LR",
                    help="Learning rate.  Default: 0.05.")
    cb.add_argument("--cb-depth",      type=int,   default=6,    metavar="D",
                    help="Tree depth / hidden layer size exponent.  Default: 6.")
    cb.add_argument("--cb-early-stopping", type=int, default=0, metavar="N",
                    help=(
                        "Stop training if the eval loss does not improve for N "
                        "consecutive rounds.  0 disables early stopping.  "
                        "When enabled, 20%% of the training data is held out as "
                        "an internal validation split (CatBoost) or 10%% (MLP).  "
                        "Default: 0 (disabled)."
                    ))
    cb.add_argument("--cb-verbose",    action="store_true",
                    help="Print CatBoost training progress.")

    out = parser.add_argument_group("Output and performance")
    out.add_argument("--output-dir", type=Path, default=Path("results"), metavar="DIR",
                     help="Output directory (created if absent).  Default: results/.")
    out.add_argument(
        "--workers", type=int, default=_DEFAULT_WORKERS, metavar="N",
        help=(
            f"Worker processes for parallel boundary processing.  "
            f"Default: {_DEFAULT_WORKERS}."
        ),
    )
    out.add_argument("--seed", type=int, default=42,
                     help="Random seed.  Default: 42.")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity.  Default: INFO.",
    )
    return parser


def main() -> None:
    multiprocessing.set_start_method("fork", force=True)

    parser = build_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.dataset is None and (args.train is None or args.test is None):
        parser.error("Provide either --dataset OR both --train and --test.")
    if args.dataset is not None and (args.train is not None or args.test is not None):
        parser.error("--dataset is mutually exclusive with --train / --test.")
    if args.workers < 1:
        parser.error("--workers must be >= 1.")

    try:
        k_values = expand_k(args.k)
    except ValueError as exc:
        parser.error(str(exc))

    logger.info("=" * 62)
    logger.info("  ACS INCOME PIPELINE  --  stage 2: BoCSoR feature importance")
    logger.info("=" * 62)
    logger.info("  Analysis on      : TRAINING SET")
    logger.info("  k values         : %s  (from --k %s)", k_values, args.k)
    logger.info("  Percentile       : %.1f%%", args.percentile)
    logger.info("  Classes          : %s", args.original_class)
    logger.info("  Workers          : %d", args.workers)
    logger.info("  Output dir       : %s", args.output_dir.resolve())
    logger.info("=" * 62)

    t_start = time.perf_counter()

    X_train, X_test, y_train, y_test, target_col = load_split_data(
        train_path=args.train,
        test_path=args.test,
        dataset_path=args.dataset,
        test_size=args.split_size,
        random_seed=args.seed,
    )
    feature_cols = list(X_train.columns)
    logger.info("Train: %d rows  |  Test: %d rows  |  Features: %d",
                len(X_train), len(X_test), len(feature_cols))
    logger.info("Target: %s  |  Features: %s", target_col, feature_cols)

    logger.info("Building rank maps (ordinal columns only) ...")
    rank_maps    = build_rank_maps(X_train[feature_cols])
    nominal_maps = build_nominal_maps(X_train[feature_cols])

    model = train_model(
        classifier=args.classifier,
        X_train=X_train,
        y_train=y_train,
        rank_maps=rank_maps,
        nominal_maps=nominal_maps,
        feature_cols=feature_cols,
        random_seed=args.seed,
        iterations=args.cb_iterations,
        learning_rate=args.cb_lr,
        depth=args.cb_depth,
        verbose=args.cb_verbose,
        early_stopping_rounds=(
            args.cb_early_stopping if args.cb_early_stopping > 0 else None
        ),
    )

    y_pred_train = model.predict(X_train).astype(int).ravel()
    y_pred_test  = model.predict(X_test).astype(int).ravel()
    train_acc = (y_pred_train == y_train.values).mean()
    test_acc  = (y_pred_test  == y_test.values).mean()
    logger.info(
        "%s accuracy -- train: %.4f  |  test (held-out): %.4f",
        args.classifier.capitalize(), train_acc, test_acc,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Skip-if-exists check ───────────────────────────────────────────────
    # Determine which classes still need to be processed.  For each requested
    # class, check whether feature_importance[_classN].csv already exists.
    # If all requested outputs are present, skip stage 2 entirely.
    orig_classes_requested = sorted(set(args.original_class))
    suffix_map = {
        c: (f"_class{c}" if len(orig_classes_requested) > 1 else "")
        for c in orig_classes_requested
    }
    orig_classes_todo = [
        c for c in orig_classes_requested
        if not (args.output_dir / f"feature_importance{suffix_map[c]}.csv").exists()
    ]
    if not orig_classes_todo:
        logger.info(
            "Skipping stage 2: all output files already exist in %s.",
            args.output_dir.resolve(),
        )
        return
    skipped = set(orig_classes_requested) - set(orig_classes_todo)
    if skipped:
        logger.info(
            "Skipping class(es) %s: output files already exist.",
            sorted(skipped),
        )

    # Deduplicate and sort the requested classes; for each, the counterfactual
    # class is the other one (binary classification: 0 <-> 1).
    orig_classes = orig_classes_todo

    # Collect results for the summary report.
    summary_data: list[dict] = []

    for orig_cls in orig_classes:
        cf_cls = 1 - orig_cls
        # suffix_map is keyed on orig_classes_requested (the full set),
        # so it is correct even when only a subset of classes is processed.
        suffix = suffix_map[orig_cls]

        logger.info(
            "== BoCSoR: class %d boundary (counterfactual class %d) ==",
            orig_cls, cf_cls,
        )
        t_dir = time.perf_counter()
        (all_itemsets_df, importance_df, per_k_itemsets, distances_df,
         filter_stats_df, label_imp_df, value_imp_df) = run_bocsor_multi_k(
            model=model,
            X_train=X_train,
            y_train=y_train,
            y_pred_train=y_pred_train,
            rank_maps=rank_maps,
            feature_cols=feature_cols,
            k_values=k_values,
            percentile_th=args.percentile,
            original_class=orig_cls,
            cf_class=cf_cls,
            n_workers=args.workers,
        )
        elapsed_dir = time.perf_counter() - t_dir

        # ── Save itemsets ──────────────────────────────────────────────────
        itemsets_path = args.output_dir / f"feature_importance_itemsets{suffix}.csv"
        all_itemsets_df.to_csv(itemsets_path, index=False)
        logger.info("Itemsets (all k) -> %s  (%d rows)", itemsets_path, len(all_itemsets_df))

        for k_val, k_df in per_k_itemsets.items():
            k_path = args.output_dir / f"feature_importance_itemsets_k{k_val}{suffix}.csv"
            k_df.to_csv(k_path, index=False)
            logger.info("  k=%d -> %s  (%d rows)", k_val, k_path.name, len(k_df))

        # ── Save feature importance (old union-based, backward compat) ────
        imp_path = args.output_dir / f"feature_importance{suffix}.csv"
        importance_df.reset_index().to_csv(imp_path, index=False)
        logger.info("Importance (union) -> %s", imp_path)

        # ── Save new BoCSoR indices ───────────────────────────────────────
        label_path = args.output_dir / f"bocsor_label_importance{suffix}.csv"
        label_imp_df.reset_index().to_csv(label_path, index=False)
        logger.info("BoCSoR label importance -> %s", label_path)

        value_path = args.output_dir / f"bocsor_value_importance{suffix}.csv"
        value_imp_df.reset_index().to_csv(value_path, index=False)
        logger.info("BoCSoR value importance -> %s", value_path)

        # Distances
        dist_path = args.output_dir / f"bocsor_distances{suffix}.csv"
        distances_df.to_csv(dist_path, index=False)
        logger.info("Distances -> %s  (%d rows)", dist_path, len(distances_df))

        # Filter stats
        fstats_path = args.output_dir / f"bocsor_filter_stats{suffix}.csv"
        filter_stats_df.to_csv(fstats_path, index=False)
        logger.info("Filter stats -> %s", fstats_path)

        # ── Distance histograms (saved to plots/ subfolder) ───────────────
        plot_distance_histograms(
            distances_df, args.output_dir, suffix=suffix,
        )

        logger.info("Feature importance summary (class %d):", orig_cls)
        for k_col in importance_df.columns:
            top = importance_df[k_col].sort_values(ascending=False)
            logger.info("  %s:", k_col)
            for feat, score in top.items():
                logger.info("    %-40s %.4f", feat, score)

        if not all_itemsets_df.empty:
            logger.info(
                "\nItemset preview class %d (first 10 rows):\n%s",
                orig_cls, all_itemsets_df.head(10).to_string(index=False),
            )

        summary_data.append({
            "orig_cls":      orig_cls,
            "importance_df": importance_df,
            "elapsed_dir":   elapsed_dir,
        })

    # ── Write summary report ──────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_start
    _write_summary(
        output_dir=args.output_dir,
        train_path=args.train or args.dataset,
        test_path=args.test,
        target_col=target_col,
        feature_cols=feature_cols,
        n_train=len(X_train),
        n_test=len(X_test),
        train_acc=train_acc,
        test_acc=test_acc,
        k_values=k_values,
        percentile_th=args.percentile,
        summary_data=summary_data,
        total_elapsed=total_elapsed,
        classifier=args.classifier,
        cb_iterations=args.cb_iterations,
        cb_lr=args.cb_lr,
        cb_depth=args.cb_depth,
        cb_early_stopping=args.cb_early_stopping,
        n_workers=args.workers,
    )

    logger.info("=" * 62)
    logger.info("  Stage 2 complete.  Outputs: %s", args.output_dir.resolve())
    logger.info("  Total elapsed: %.1fs", total_elapsed)
    logger.info("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# Distance histograms
# ─────────────────────────────────────────────────────────────────────────────

def plot_distance_histograms(
    distances_df: pd.DataFrame,
    output_dir: Path,
    suffix: str = "",
    bins: int = 80,
) -> None:
    """
    Plot per-rank distance histograms and differing-feature breakdowns.

    Uses the max-k run data (which contains ranks 1 through max_k) to
    show how distance and feature differences grow from the 1st to the
    k-th nearest counterfactual.

    Produces 3 types of PNG:
      - Per-rank histogram (stacked by n_diff_features): one PNG per rank.
      - Combined grid: all ranks on a single figure.
      - Stacked percentage bars: % of counterfactuals with 1, 2, 3, …
        differing features, one bar per rank.

    Parameters
    ----------
    distances_df : DataFrame with columns [k_value, instance_index,
                   cf_index, k_neighbour_rank, distance, n_diff_features].
    output_dir   : Directory where PNGs are saved.
    suffix       : Filename suffix (e.g. "_class0", "_class1").
    bins         : Number of histogram bins.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if distances_df.empty:
        logger.info("No distances to plot (empty DataFrame).")
        return

    # Defensive: exclude any residual distance = 0 rows.
    distances_df = distances_df[distances_df["distance"] > 0.0]
    if distances_df.empty:
        logger.info("No non-zero distances to plot.")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    k_values = sorted(distances_df["k_value"].unique())

    # Use the largest k run — contains ranks 1 through max_k.
    max_k = max(k_values)
    max_k_df = distances_df[distances_df["k_value"] == max_k]
    ranks = sorted(max_k_df["k_neighbour_rank"].unique())

    if not ranks:
        logger.info("No distance data for k=%d — skipping plots.", max_k)
        return

    # Shared bin edges (linear scale) across all ranks for visual comparison.
    global_min = max_k_df["distance"].min()
    global_max = max_k_df["distance"].max()
    bin_edges = np.linspace(global_min, global_max, bins + 1)

    has_ndiff = "n_diff_features" in max_k_df.columns

    # Color palette for n_diff_features stacking.
    _colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
               "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
               "#CCB974", "#64B5CD"]
    if has_ndiff:
        max_diff = int(max_k_df["n_diff_features"].max())
        diff_values = list(range(1, max_diff + 1))
    else:
        diff_values = []

    # ── One PNG per neighbour rank (stacked by n_diff_features) ──────────
    for rank in ranks:
        rank_df = max_k_df[max_k_df["k_neighbour_rank"] == rank]
        dists = rank_df["distance"].to_numpy()
        ordinal = f"{rank}{'st' if rank == 1 else 'nd' if rank == 2 else 'rd' if rank == 3 else 'th'}"

        fig, ax = plt.subplots(figsize=(16, 9))

        if has_ndiff and len(diff_values) > 1:
            # Stacked histogram: one layer per n_diff_features value.
            layers = []
            labels = []
            for d in diff_values:
                layer = rank_df.loc[rank_df["n_diff_features"] == d, "distance"].to_numpy()
                if len(layer) > 0:
                    layers.append(layer)
                    labels.append(f"{d} feat.")
            ax.hist(layers, bins=bin_edges, stacked=True, edgecolor="black",
                    linewidth=0.4, alpha=0.85,
                    color=_colors[:len(layers)], label=labels)
            ax.legend(fontsize=10, title="Diff. features")
        else:
            ax.hist(dists, bins=bin_edges, edgecolor="black", alpha=0.75,
                    color="#4C72B0")

        ax.axvline(float(np.median(dists)), color="red", linestyle="--",
                   linewidth=1, label=f"median = {np.median(dists):.3f}")
        ax.legend(fontsize=10)

        # Dense ticks + grid for readability.
        from matplotlib.ticker import MultipleLocator, AutoMinorLocator
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="minor", length=3)
        ax.tick_params(which="major", length=6)
        ax.grid(axis="both", which="major", alpha=0.25, linewidth=0.5)

        ax.set_xlabel("Hybrid Manhattan distance to counterfactual (1.0 = one nominal change)",
                       fontsize=12)
        ax.set_ylabel("Number of instances", fontsize=12)
        ax.set_title(
            f"BoCSoR: distance to the {ordinal} nearest counterfactual\n"
            f"({len(dists):,} instances, "
            f"range [{dists.min():.3f}, {dists.max():.3f}])",
            fontsize=14,
        )
        fig.tight_layout()

        path = plots_dir / f"bocsor_distance_histogram_rank{rank}{suffix}.png"
        fig.savefig(path, dpi=250)
        plt.close(fig)
        logger.info("Distance histogram rank=%d -> %s", rank, path.name)

    # ── Combined figure: one subplot per rank ────────────────────────────
    if len(ranks) > 1:
        from matplotlib.ticker import MultipleLocator, AutoMinorLocator
        n_r = len(ranks)
        n_cols_grid = min(3, n_r)
        n_rows = math.ceil(n_r / n_cols_grid)
        fig, axes = plt.subplots(
            n_rows, n_cols_grid, figsize=(8 * n_cols_grid, 6 * n_rows),
            squeeze=False,
        )

        for idx, rank in enumerate(ranks):
            ax = axes[idx // n_cols_grid][idx % n_cols_grid]
            rank_df = max_k_df[max_k_df["k_neighbour_rank"] == rank]
            dists = rank_df["distance"].to_numpy()

            if has_ndiff and len(diff_values) > 1:
                layers = []
                labels = []
                for d in diff_values:
                    layer = rank_df.loc[rank_df["n_diff_features"] == d, "distance"].to_numpy()
                    if len(layer) > 0:
                        layers.append(layer)
                        labels.append(f"{d}f")
                ax.hist(layers, bins=bin_edges, stacked=True, edgecolor="black",
                        linewidth=0.3, alpha=0.85,
                        color=_colors[:len(layers)], label=labels)
                ax.legend(fontsize=7)
            else:
                ax.hist(dists, bins=bin_edges, edgecolor="black", alpha=0.75,
                        color="#4C72B0")

            ax.xaxis.set_major_locator(MultipleLocator(0.2))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.grid(axis="both", which="major", alpha=0.2, linewidth=0.5)

            ordinal = f"{rank}{'st' if rank == 1 else 'nd' if rank == 2 else 'rd' if rank == 3 else 'th'}"
            ax.set_title(f"{ordinal} nearest CF  (n={len(dists):,}, med={np.median(dists):.3f})")
            ax.set_xlabel("Distance (1.0 = 1 nom. change)")
            ax.set_ylabel("Instances")

        for idx in range(n_r, n_rows * n_cols_grid):
            axes[idx // n_cols_grid][idx % n_cols_grid].set_visible(False)

        fig.suptitle(
            f"BoCSoR: distance to k-th nearest counterfactual (k=1…{max_k}){suffix}",
            fontsize=14, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        path = plots_dir / f"bocsor_distance_histograms_per_rank{suffix}.png"
        fig.savefig(path, dpi=250)
        plt.close(fig)
        logger.info("Per-rank distance histograms -> %s", path.name)

    # ── Stacked bars: % of counterfactuals by n_diff at each rank ────────
    if has_ndiff and len(ranks) > 1:
        pct_data: dict[int, list[float]] = {d: [] for d in diff_values}
        for rank in ranks:
            sub = max_k_df.loc[max_k_df["k_neighbour_rank"] == rank, "n_diff_features"]
            total = len(sub)
            counts = sub.value_counts()
            for d in diff_values:
                pct_data[d].append(counts.get(d, 0) / total * 100 if total else 0)

        fig, ax = plt.subplots(figsize=(max(10, len(ranks) * 0.9), 7))
        x = np.arange(len(ranks))
        bar_width = 0.65
        bottom = np.zeros(len(ranks))

        for i, d in enumerate(diff_values):
            vals = np.array(pct_data[d])
            color = _colors[i % len(_colors)]
            ax.bar(x, vals, bar_width, bottom=bottom,
                   label=f"{d} feature{'s' if d > 1 else ''}",
                   color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
            for j, v in enumerate(vals):
                if v > 5:
                    ax.text(x[j], bottom[j] + v / 2, f"{v:.0f}%",
                            ha="center", va="center", fontsize=7, fontweight="bold",
                            color="white")
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in ranks])
        ax.set_xlabel("k-th nearest counterfactual")
        ax.set_ylabel("Percentage of counterfactuals")
        ax.set_title(f"Differing features: 1st → {max_k}th nearest counterfactual{suffix}")
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()

        path = plots_dir / f"bocsor_diff_features_pct{suffix}.png"
        fig.savefig(path, dpi=250)
        plt.close(fig)
        logger.info("Diff features stacked bars -> %s", path.name)


# ─────────────────────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────────────────────

def _write_summary(
    output_dir: Path,
    train_path,
    test_path,
    target_col: str,
    feature_cols: list[str],
    n_train: int,
    n_test: int,
    train_acc: float,
    test_acc: float,
    k_values: list[int],
    percentile_th: float,
    summary_data: list[dict],
    total_elapsed: float,
    classifier: str,
    cb_iterations: int,
    cb_lr: float,
    cb_depth: int,
    cb_early_stopping: int,
    n_workers: int,
) -> None:
    """
    Write a human-readable Markdown summary of the BoCSoR run to
    results/bocsor_summary.md.
    """
    from datetime import datetime

    lines: list[str] = []
    a = lines.append

    a("# BoCSoR Feature Importance — Run Summary")
    a("")
    a(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"**Total runtime:** {total_elapsed:.1f}s")
    a("")

    # ── Dataset ───────────────────────────────────────────────────────────────
    a("## Dataset")
    a("")
    a(f"| | |")
    a(f"|---|---|")
    a(f"| Train file | `{Path(train_path).name if train_path else 'n/a'}` |")
    a(f"| Test file  | `{Path(test_path).name  if test_path  else 'n/a'}` |")
    a(f"| Train rows | {n_train:,} |")
    a(f"| Test rows  | {n_test:,} |")
    a(f"| Target     | `{target_col}` |")
    a(f"| Features   | {', '.join(f'`{f}`' for f in feature_cols)} |")
    a("")

    # ── Classifier ────────────────────────────────────────────────────────────
    a(f"## {classifier.capitalize()} Classifier")
    a("")
    a(f"| | |")
    a(f"|---|---|")
    a(f"| Iterations      | {cb_iterations} |")
    a(f"| Learning rate   | {cb_lr} |")
    a(f"| Tree depth      | {cb_depth} |")
    a(f"| Early stopping  | {'disabled' if cb_early_stopping == 0 else f'{cb_early_stopping} rounds'} |")
    a(f"| Train accuracy  | {train_acc:.4f} |")
    a(f"| Test accuracy   | {test_acc:.4f} |")
    a("")

    # ── BoCSoR configuration ──────────────────────────────────────────────────
    a("## BoCSoR Configuration")
    a("")
    a(f"| | |")
    a(f"|---|---|")
    a(f"| k values    | {k_values} |")
    a(f"| Percentile  | {percentile_th}% |")
    a(f"| Workers     | {n_workers} |")
    directions = " | ".join(
        f"class {e['orig_cls']} → class {1 - e['orig_cls']}"
        for e in summary_data
    )
    a(f"| Boundaries  | {directions} |")
    a("")
    a("")

    # ── Feature importance per boundary direction ─────────────────────────────
    a("## Feature Importance Results")
    a("")
    a("> BoCSoR score = fraction of boundary instances for which the feature")
    a("> appears in the union of relevant features across all k counterfactuals.")
    a("> Values in [0, 1]. Higher = more important for crossing the decision boundary.")
    a("")

    # Group summary_data by direction (each direction appears once in summary_data
    # with its importance_df; we de-duplicate).
    for entry in summary_data:
        orig_cls    = entry["orig_cls"]
        cf_cls      = 1 - orig_cls
        imp_df      = entry["importance_df"]
        elapsed_dir = entry["elapsed_dir"]

        a(f"### Class {orig_cls} → Class {cf_cls}  *(elapsed: {elapsed_dir:.1f}s)*")
        a("")

        # Table header
        header = "| Feature |" + "".join(f" {col} |" for col in imp_df.columns)
        sep    = "|---|" + "---|" * len(imp_df.columns)
        a(header)
        a(sep)

        # Sort features by the last (largest) k column for stable display.
        last_col = imp_df.columns[-1]
        for feat in imp_df[last_col].sort_values(ascending=False).index:
            row = f"| `{feat}` |"
            for col in imp_df.columns:
                row += f" {imp_df.loc[feat, col]:.4f} |"
            a(row)
        a("")

        # Stability note: flag features whose rank changes across k values.
        ranks = imp_df.rank(ascending=False, method="min")
        unstable = [
            feat for feat in imp_df.index
            if ranks.loc[feat].max() - ranks.loc[feat].min() > 1
        ]
        if unstable:
            a(f"WARNING: Rank-unstable features (rank varies across k values): "
              f"{', '.join(f'`{f}`' for f in unstable)}")
            a("")
        else:
            a("All features maintain a consistent rank across all k values.")
            a("")

    # ── Output files ──────────────────────────────────────────────────────────
    a("## Output Files")
    a("")
    a("| File | Description |")
    a("|---|---|")
    a("| `feature_importance.csv` | BoCSoR scores — rows: features, columns: k_1…k_N |")
    a("| `feature_importance_itemsets.csv` | Itemsets for all k merged — columns: k_value, instance_index, features, itemset |")
    for k in k_values:
        a(f"| `feature_importance_itemsets_k{k}.csv` | Itemsets for k={k} only |")
    a("| `bocsor_distances.csv` | Distances — columns: k_value, instance_index, cf_index, k_neighbour_rank, distance |")
    a("| `bocsor_summary.md` | This file |")
    a("")

    summary_path = output_dir / "bocsor_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary -> %s", summary_path)


if __name__ == "__main__":
    main()