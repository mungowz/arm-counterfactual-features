"""
src/feature_importance.py
─────────────────────────
Stage 2 of the ACS Income pipeline: Boundary Crossing Solo Ratio (BoCSoR) XAI.

Adapted from:
  Alfeo et al. (2023) "From local counterfactuals to global feature importance:
  efficient, robust, and model-agnostic explanations for brain connectivity networks"
  Computer Methods and Programs in Biomedicine 236, 107550.

Adaptations for fully-categorical data
───────────────────────────────────────
* Analysis runs on the TRAINING SET.  The classifier has full knowledge of
  training instances, so boundary instances found there are the most
  informative for explaining the model's decision surface.  The test set is
  used only to compute a held-out accuracy estimate.

* No interpolation of intermediate points (not meaningful for categorical
  features).  Counterfactuals are the K nearest neighbours of the opposite
  class in rank-encoded Manhattan space.

* Rank-based encoding:
    - Ordinal columns (AGEP, SCHL, WKHP) use the declared semantic order.
    - All other columns use lexicographic (alphabetical) order — consistent
      but arbitrary, no semantic meaning for nominal features.

* Manhattan distance normalised to [0, 2]:
      dist(a,b) = 2 x sum|rank_i(a) - rank_i(b)| / sum(max_rank_i)

* Multi-k evaluation via --k:
      Single value K  -> auto-expanded to all odd integers 1..K (plus K if
                         even).  --k 11 -> [1, 3, 5, 7, 9, 11]
      Multiple values -> used as-is.  --k 1 5 11 -> [1, 5, 11]

* Itemset output: one row per boundary instance per k value.  All relevant
  features are on the same row.  Columns: k_value, instance_index,
  features (space-separated names), itemset (space-separated FEATURE=value
  tokens, ARM-ready).  One combined file plus one file per k value.

Parallelism and performance
────────────────────────────
* Start method: multiprocessing uses "fork" so worker processes inherit the
  parent's memory (rank maps, encoded arrays, model) without reimporting.
* BallTree (sklearn, Manhattan metric): built once on the cf-class instances
  and inherited by workers via fork.  Used for both boundary selection
  (O(N log N) instead of O(N²)) and k-NN queries in each worker (O(k log N)).
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
from sklearn.neighbors import BallTree

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
# Rank encoding
# ─────────────────────────────────────────────────────────────────────────────

def build_rank_maps(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """
    Build {column -> {category_label -> rank}} from the training DataFrame.

    Ordinal columns use the semantic order in ORDERED_CATEGORIES (rank 1 =
    lowest, rank N = highest).  All other columns use lexicographic order.
    Ranks are 1-based.  Must be built from training data only.
    """
    rank_maps: dict[str, dict[str, int]] = {}
    for col in df.columns:
        present = set(df[col].dropna().unique())
        if col in ORDERED_CATEGORIES:
            ordered = [c for c in ORDERED_CATEGORIES[col] if c in present]
            ordered += sorted(present - set(ordered))
        else:
            ordered = sorted(present)
        rank_maps[col] = {cat: rank for rank, cat in enumerate(ordered, start=1)}
    return rank_maps


def encode_ranks(
    df: pd.DataFrame,
    rank_maps: dict[str, dict[str, int]],
) -> np.ndarray:
    """
    Convert a categorical DataFrame to a (n_samples, n_features) int32 matrix.

    Values absent from the rank map fall back to the median rank of that
    column (neutral fallback, avoids NaN in distance calculations).
    """
    result = np.empty((len(df), len(df.columns)), dtype=np.int32)
    for j, col in enumerate(df.columns):
        rmap     = rank_maps[col]
        fallback = int(np.median(list(rmap.values())))
        result[:, j] = (
            df[col].map(lambda v, rm=rmap, fb=fallback: rm.get(v, fb)).to_numpy()
        )
    return result


def max_rank_per_column(rank_maps: dict[str, dict[str, int]]) -> np.ndarray:
    """Return a (F,) float64 array of maximum rank for each column."""
    return np.array([max(rm.values()) for rm in rank_maps.values()], dtype=np.float64)




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

def build_balltree(X_enc: np.ndarray, max_ranks: np.ndarray) -> BallTree:
    """
    Build a BallTree on rank-encoded data using normalised Manhattan distance.

    The tree is built once in the main process and inherited by worker
    processes via fork — no serialisation overhead.

    We use the "chebyshev" leaf_size default and Manhattan (L1) metric.
    The normalisation (divide by norm_factor, multiply by 2) is applied to
    the input before building so that the tree returns distances in [0, 2],
    consistent with the formula: dist(a,b) = 2 × Σ|rank_i(a)-rank_i(b)| / Σmax_rank_i.
    """
    norm_factor = float(max_ranks.sum())
    X_norm = X_enc.astype(np.float32) / norm_factor * 2.0
    return BallTree(X_norm, metric="manhattan", leaf_size=40)


def select_boundary_instances(
    X_enc: np.ndarray,
    y_true: np.ndarray,
    y_pred_train: np.ndarray,
    max_ranks: np.ndarray,
    percentile_th: float,
    original_class: int,
    cf_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, BallTree]:
    """
    Compute boundary instances and counterfactual indices once using a BallTree.

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

    The BallTree is built on the counterfactual-class instances and returned
    so that worker processes can reuse it for the k-NN step (Algorithm 1).

    Parameters
    ----------
    y_true        : True class labels (used to separate class pools).
    y_pred_train  : Model predictions on the training set (used to filter
                    misclassified instances from the boundary candidates).

    Returns
    -------
    (boundary_indices, cf_indices, X_enc_cf, cf_tree)
      boundary_indices : row indices in X_enc of boundary instances.
      cf_indices       : row indices in X_enc of true-cf-class instances.
      X_enc_cf         : rank-encoded submatrix for cf-class instances.
      cf_tree          : BallTree built on cf-class instances (normalised).
    """
    # Separate by TRUE label — same as the original authors.
    orig_indices_true = np.where(y_true == original_class)[0]
    cf_indices        = np.where(y_true == cf_class)[0]

    if len(orig_indices_true) == 0 or len(cf_indices) == 0:
        empty_tree = BallTree(np.zeros((1, X_enc.shape[1]), dtype=np.float32),
                              metric="manhattan")
        return (np.array([], dtype=int), cf_indices,
                np.empty((0, X_enc.shape[1]), dtype=np.int32), empty_tree)

    # Keep only correctly classified orig-class instances.
    # A misclassified instance would already be on the CF side of the
    # decision boundary; its counterfactual direction is inverted and
    # its relevant features would be meaningless.
    correct_mask  = y_pred_train[orig_indices_true] == original_class
    orig_indices  = orig_indices_true[correct_mask]

    if len(orig_indices) == 0:
        empty_tree = BallTree(np.zeros((1, X_enc.shape[1]), dtype=np.float32),
                              metric="manhattan")
        return (np.array([], dtype=int), cf_indices,
                np.empty((0, X_enc.shape[1]), dtype=np.int32), empty_tree)

    X_enc_orig = X_enc[orig_indices]
    X_enc_cf   = X_enc[cf_indices]

    # Build BallTree on the true-cf-class instances (normalised coordinates).
    cf_tree = build_balltree(X_enc_cf, max_ranks)

    # For each correctly-classified orig-class instance find its nearest
    # true-cf-class neighbour (k=1).  Instances below the distance threshold
    # are the boundary instances.
    norm_factor = float(max_ranks.sum())
    X_orig_norm = X_enc_orig.astype(np.float32) / norm_factor * 2.0
    min_dists, _ = cf_tree.query(X_orig_norm, k=1)   # (N_orig, 1)
    min_dists    = min_dists[:, 0]                     # (N_orig,)

    # Use <= to match the original authors' convention.
    threshold = float(np.percentile(min_dists, percentile_th))
    b_local   = np.where(min_dists <= threshold)[0]

    boundary_indices = orig_indices[b_local]
    return boundary_indices, cf_indices, X_enc_cf, cf_tree


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
    norm_factor: float,
    cf_tree: BallTree,
) -> tuple[dict[str, int], list[dict], list[dict]]:
    """
    Process one chunk of boundary instances in a worker process.

    For each boundary instance:
      1. Find k nearest neighbours from the counterfactual class
         (Algorithm 1 — no interpolation for categorical data).
      2. Build a batch of all modified instances needed for the relevance
         check across all k counterfactuals, then call model.predict() once
         on the entire batch (batched Algorithm 2).
      3. Take the UNION of relevant features across all k counterfactuals
         and emit one row per boundary instance with all relevant features.

    The model is inherited from the parent process via fork — no disk I/O.

    Parameters
    ----------
    chunk_indices     : Row indices (in X_enc / X_train_values) to process.
    X_enc             : Full rank-encoded training matrix (read-only).
    X_enc_cf          : Rank-encoded submatrix of counterfactual-class rows.
    cf_global_indices : Maps position in X_enc_cf -> row in X_enc.
    X_train_values    : (n_train, n_features) object array of string labels.
    feature_cols      : Ordered feature column names.
    model             : CatBoostClassifier inherited from parent via fork.
    k                 : Number of nearest counterfactuals to consider.
    original_class    : Class label of the boundary instances.
    norm_factor       : sum(max_rank_i), precomputed for distance normalisation.
    cf_tree           : BallTree built on cf-class instances (normalised coords).
                        Inherited via fork — used for O(k log N) k-NN queries.

    Returns
    -------
    (importance_counts, itemset_rows)
      importance_counts : {feature -> count of boundary instances in this
                           chunk for which the feature is in the relevant union}.
      itemset_rows      : one dict per boundary instance with relevant features.
      distance_rows     : one dict per (boundary_instance, k_neighbour) pair
                          with fields instance_index, cf_index,
                          k_neighbour_rank (1=closest), distance.
    """
    n_features = len(feature_cols)
    importance_counts: dict[str, int] = {f: 0 for f in feature_cols}
    itemset_rows: list[dict] = []
    distance_rows: list[dict] = []

    n_cf = len(cf_global_indices)

    for b_idx in chunk_indices:
        # ── Algorithm 1: k-NN via BallTree — O(k log N_cf) ───────────────────
        query_norm = (X_enc[b_idx].astype(np.float32)
                      / norm_factor * 2.0).reshape(1, -1)
        k_act          = min(k, n_cf)
        dists, top_local = cf_tree.query(query_norm, k=k_act)   # (1, k_act)
        cf_idxs = [int(cf_global_indices[i]) for i in top_local[0]]
        cf_dists = dists[0].tolist()   # normalised Manhattan distances

        if not cf_idxs:
            continue

        instance_vals = X_train_values[b_idx]

        # Record one distance row per (boundary_instance, k_neighbour) pair.
        for rank, (cf_idx, dist) in enumerate(zip(cf_idxs, cf_dists), start=1):
            distance_rows.append({
                "instance_index":    int(b_idx),
                "cf_index":          cf_idx,
                "k_neighbour_rank":  rank,   # 1 = closest, 2 = second closest, …
                "distance":          round(float(dist), 6),
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

        # Collect relevant features (union across all k counterfactuals).
        relevant_union: set[str] = set()
        for pred, (_, fi) in zip(preds, batch_meta):
            if int(pred) == original_class:
                relevant_union.add(feature_cols[fi])

        # Pre-compute a {feature_name: column_index} map to avoid O(n)
        # list.index() calls inside the relevance-collection loop below.
        feat_to_idx: dict[str, int] = {f: i for i, f in enumerate(feature_cols)}

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

    return importance_counts, itemset_rows, distance_rows


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
    cf_tree: BallTree,
    feature_cols: list[str],
    rank_maps: dict[str, dict[str, int]],
    k: int,
    original_class: int,
    n_workers: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Run BoCSoR for a single k value.

    Boundary instances and counterfactual indices are passed in (precomputed
    by the caller) so that the expensive boundary distance computation is not
    recomputed for every k.

    Returns
    -------
    (itemsets_df, feature_importance, distances_df)
    """
    norm_factor = float(max_rank_per_column(rank_maps).sum())

    logger.info(
        "  k=%d | boundary instances: %d (class %d)",
        k, len(boundary_indices), original_class,
    )
    if len(boundary_indices) == 0:
        return (
            pd.DataFrame(columns=["instance_index", "features", "itemset"]),
            pd.Series(0.0, index=feature_cols, name="BoCSoR_importance"),
            pd.DataFrame(columns=["instance_index", "cf_index",
                                   "k_neighbour_rank", "distance"]),
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
                norm_factor,
                cf_tree,
            ): chunk
            for chunk in chunks
        }
        for future in as_completed(futures):
            imp_c, rows_c, dist_c = future.result()
            for feat, cnt in imp_c.items():
                total_importance[feat] += cnt
            all_rows.extend(rows_c)
            all_dist_rows.extend(dist_c)
            n_with_cf += len({r["instance_index"] for r in rows_c})

    logger.info(
        "  k=%d | instances with >=1 counterfactual: %d / %d",
        k, n_with_cf, len(boundary_indices),
    )

    # Normalise by n_with_cf (instances that produced ≥ 1 relevant feature)
    # rather than len(boundary_indices).  Instances for which no feature swap
    # changed the prediction contribute no signal, so excluding them from the
    # denominator keeps scores comparable across datasets with different
    # counterfactual densities.  This choice is intentional and documented here.
    feat_imp = (
        pd.Series(total_importance, name="BoCSoR_importance")
        / max(n_with_cf, 1)
    ).sort_values(ascending=False)

    dist_df = pd.DataFrame(all_dist_rows) if all_dist_rows else pd.DataFrame(
        columns=["instance_index", "cf_index", "k_neighbour_rank", "distance"]
    )
    return pd.DataFrame(all_rows), feat_imp, dist_df


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
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame], pd.DataFrame]:
    """
    Run BoCSoR once per value in k_values on the training set.

    Key optimisations vs. the naive loop:
    - Rank encoding of X_train: computed once, reused for all k values.
    - Boundary selection via BallTree: O(N log N) instead of O(N²).
      Pools separated by TRUE label; only correctly-classified orig-class
      instances are candidates.  Built once, reused for all k values.
    - k-NN (Algorithm 1) in workers: BallTree query O(k log N_cf)
      instead of a linear scan O(N_cf) per boundary instance.
    - Workers inherit the BallTree from the main process via fork.
    - If zero boundary instances are found, all k values are skipped
      immediately without any tree query per k.
    - Relevance check uses batched model.predict() instead of one call
      per (counterfactual, feature) pair.

    Returns
    -------
    (all_itemsets_df, importance_df, per_k_itemsets, distances_df)
      all_itemsets_df : columns [k_value, instance_index, features, itemset].
      importance_df   : features x k_values wide table (columns k_1, k_3, ...).
      per_k_itemsets  : {k -> itemset DataFrame for that k only}.
      distances_df    : columns [k_value, instance_index, cf_index,
                        k_neighbour_rank, distance] — one row per
                        (boundary_instance, k_neighbour) pair.
    """
    logger.info(
        "BoCSoR on TRAINING SET: k=%s  class %d->%d  "
        "percentile=%.1f  workers=%d",
        k_values, original_class, cf_class, percentile_th, n_workers,
    )

    max_ranks = max_rank_per_column(rank_maps)
    X_enc     = encode_ranks(X_train[feature_cols], rank_maps)
    X_train_values = (
        X_train[feature_cols].reset_index(drop=True).to_numpy(dtype=object)
    )

    # ── Compute boundary instances once for all k ─────────────────────────────
    t0 = time.perf_counter()
    boundary_indices, cf_indices, X_enc_cf, cf_tree = select_boundary_instances(
        X_enc=X_enc,
        y_true=y_train.to_numpy().astype(int),
        y_pred_train=y_pred_train,
        max_ranks=max_ranks,
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
                     "k_neighbour_rank", "distance"]
        )
        return empty_df, imp_df, {k: empty_df.copy() for k in k_values}, empty_dist

    # ── Run once per k (boundary instances reused) ────────────────────────────
    all_itemsets:    list[pd.DataFrame]   = []
    all_distances:   list[pd.DataFrame]   = []
    importance_dict: dict[int, pd.Series] = {}

    for k in k_values:
        logger.info("── k=%d ──────────────────────────────────────────────", k)
        t_k = time.perf_counter()
        itemsets_df, feat_imp, dist_df = run_bocsor_single_k(
            model=model,
            X_train=X_train,
            X_enc=X_enc,
            X_train_values=X_train_values,
            boundary_indices=boundary_indices,
            cf_indices=cf_indices,
            X_enc_cf=X_enc_cf,
            cf_tree=cf_tree,
            feature_cols=feature_cols,
            rank_maps=rank_maps,
            k=k,
            original_class=original_class,
            n_workers=n_workers,
        )
        logger.info("  k=%d | elapsed: %.1fs", k, time.perf_counter() - t_k)
        if not itemsets_df.empty:
            itemsets_df.insert(0, "k_value", k)
        if not dist_df.empty:
            dist_df.insert(0, "k_value", k)
        all_itemsets.append(itemsets_df)
        all_distances.append(dist_df)
        importance_dict[k] = feat_imp

    combined = pd.concat(all_itemsets, ignore_index=True)
    imp_df   = pd.DataFrame(importance_dict).rename_axis("feature")
    imp_df.columns = [f"k_{k}" for k in imp_df.columns]

    # per_k_itemsets: {k -> itemset DataFrame for that k only}
    per_k_itemsets: dict[int, pd.DataFrame] = {
        k: df for k, df in zip(k_values, all_itemsets)
    }

    # distances_df: all k merged, columns [k_value, instance_index,
    # cf_index, k_neighbour_rank, distance]
    distances_df = pd.concat(all_distances, ignore_index=True)

    return combined, imp_df, per_k_itemsets, distances_df


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


class LightGBMWrapper:
    """
    Wraps LGBMClassifier to expose the same predict() interface as CatBoost.

    LightGBM requires integer-encoded categorical inputs and returns class
    probabilities from predict(), while CatBoost accepts raw string DataFrames
    and returns class labels directly.  This wrapper handles both differences
    so the rest of the pipeline (BoCSoR workers, accuracy computation) can
    call model.predict() uniformly regardless of which classifier was trained.

    Encoding uses the rank_maps already computed for the distance calculation,
    avoiding a second encoding step and keeping the integer representation
    consistent throughout stage 2.

    The wrapper silently drops the CatBoost-specific `thread_count` keyword
    argument so that _process_boundary_chunk can call model.predict() with the
    same signature for both classifiers.
    """

    def __init__(
        self,
        lgbm_model: object,
        rank_maps: dict[str, dict[str, int]],
        feature_cols: list[str],
    ) -> None:
        self._model      = lgbm_model
        self._rank_maps  = rank_maps
        self._feat_cols  = feature_cols

    def _encode(self, X: np.ndarray) -> np.ndarray:
        """Convert an (N, F) object array of strings to int32 via rank_maps."""
        out = np.empty(X.shape, dtype=np.int32)
        for j, col in enumerate(self._feat_cols):
            rmap     = self._rank_maps[col]
            fallback = int(np.median(list(rmap.values())))
            out[:, j] = np.array(
                [rmap.get(v, fallback) for v in X[:, j]], dtype=np.int32
            )
        return out

    def predict(self, X, **kwargs) -> np.ndarray:
        """
        Return integer class labels (0 or 1).

        Accepts either a pandas DataFrame or a numpy object array of raw
        string category values.  The CatBoost-specific `thread_count` kwarg
        is silently ignored.
        """
        kwargs.pop("thread_count", None)   # CatBoost-specific — not used by LightGBM
        if isinstance(X, pd.DataFrame):
            arr = X[self._feat_cols].to_numpy(dtype=object)
        else:
            arr = np.asarray(X, dtype=object)
        X_enc = self._encode(arr)
        proba = self._model.predict(X_enc, num_threads=1)
        return (proba >= 0.5).astype(int)


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    rank_maps: dict[str, dict[str, int]],
    feature_cols: list[str],
    random_seed: int = 42,
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    verbose: bool = False,
    early_stopping_rounds: int | None = None,
) -> LightGBMWrapper:
    """
    Train a LightGBM classifier on the (fully categorical) training set and
    return it wrapped in a LightGBMWrapper for a uniform predict() interface.

    LightGBM is a leaf-wise gradient boosting framework that can match or
    exceed CatBoost accuracy on tabular data while producing decision
    boundaries that are typically more fine-grained, which benefits BoCSoR
    by increasing the number of boundary instances that produce relevant
    counterfactuals (especially at small k values).

    Categorical features are integer-encoded via the rank_maps already
    computed for the distance calculation.  The `depth` parameter maps to
    LightGBM's `max_depth`; LightGBM's leaf-wise growth is controlled
    additionally via `num_leaves` (set to 2^depth - 1 to mirror CatBoost
    behaviour by default).

    Parameters
    ----------
    rank_maps            : {column -> {label -> rank}} from build_rank_maps().
                           Used to encode string categories to integers.
    feature_cols         : Ordered list of feature column names.
    early_stopping_rounds: Stop if validation metric does not improve for N
                           consecutive rounds.  An internal 80/20 stratified
                           split is used as the eval set.  None = disabled.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "LightGBM is required for --classifier lightgbm.\n"
            "  pip install lightgbm"
        ) from exc

    num_leaves = max(2, 2 ** depth - 1)
    lgbm_model = lgb.LGBMClassifier(
        n_estimators=iterations,
        learning_rate=learning_rate,
        max_depth=depth,
        num_leaves=num_leaves,
        random_state=random_seed,
        verbose=-1 if not verbose else 1,
        n_jobs=1,   # thread control handled externally via worker processes
    )

    logger.info(
        "Training LightGBM: n_estimators=%d, lr=%.4f, max_depth=%d, "
        "num_leaves=%d, early_stopping=%s.",
        iterations, learning_rate, depth, num_leaves,
        early_stopping_rounds if early_stopping_rounds else "disabled",
    )

    # Encode training data via rank_maps.
    wrapper = LightGBMWrapper(lgbm_model, rank_maps, feature_cols)
    X_arr   = X_train[feature_cols].to_numpy(dtype=object)
    X_enc   = wrapper._encode(X_arr)

    if early_stopping_rounds is not None:
        from sklearn.model_selection import train_test_split as _tts
        X_tr, X_val, y_tr, y_val = _tts(
            X_enc, y_train,
            test_size=0.2,
            stratify=y_train,
            random_state=random_seed,
        )
        lgbm_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=verbose)],
        )
    else:
        lgbm_model.fit(X_enc, y_train)

    return wrapper


def train_model(
    classifier: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    rank_maps: dict[str, dict[str, int]],
    feature_cols: list[str],
    random_seed: int = 42,
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    verbose: bool = False,
    early_stopping_rounds: int | None = None,
) -> object:
    """
    Dispatch to train_catboost or train_lightgbm based on *classifier*.

    Parameters
    ----------
    classifier : "catboost" or "lightgbm".
    rank_maps  : Required for LightGBM encoding; unused for CatBoost.

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
    elif classifier == "lightgbm":
        return train_lightgbm(
            X_train=X_train,
            y_train=y_train,
            rank_maps=rank_maps,
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
            "Choose 'catboost' or 'lightgbm'."
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
        "--classifier", choices=["catboost", "lightgbm"], default="catboost",
        help=(
            "Gradient boosting classifier to train.  "
            "'catboost' (default) accepts raw string categoricals natively.  "
            "'lightgbm' uses leaf-wise growth and often produces finer-grained "
            "decision boundaries, increasing counterfactual yield at small k."
        ),
    )

    cb = parser.add_argument_group("Classifier hyperparameters  (shared by CatBoost and LightGBM)")
    cb.add_argument("--cb-iterations", type=int,   default=500,  metavar="N",
                    help="Boosting rounds (epochs).  Default: 500.")
    cb.add_argument("--cb-lr",         type=float, default=0.05, metavar="LR",
                    help="Learning rate.  Default: 0.05.")
    cb.add_argument("--cb-depth",      type=int,   default=6,    metavar="D",
                    help="Tree depth.  Default: 6.")
    cb.add_argument("--cb-early-stopping", type=int, default=0, metavar="N",
                    help=(
                        "Stop training if the eval loss does not improve for N "
                        "consecutive rounds.  0 disables early stopping.  "
                        "When enabled, 20%% of the training data is held out as "
                        "an internal validation split.  Default: 0 (disabled)."
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

    logger.info("Building rank maps ...")
    rank_maps = build_rank_maps(X_train)

    model = train_model(
        classifier=args.classifier,
        X_train=X_train,
        y_train=y_train,
        rank_maps=rank_maps,
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
        all_itemsets_df, importance_df, per_k_itemsets, distances_df = run_bocsor_multi_k(
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
        # Combined file: all k merged.  Columns: [k_value, instance_index,
        # features, itemset].  One row per boundary instance per k value.
        itemsets_path = args.output_dir / f"feature_importance_itemsets{suffix}.csv"
        all_itemsets_df.to_csv(itemsets_path, index=False)
        logger.info("Itemsets (all k) -> %s  (%d rows)", itemsets_path, len(all_itemsets_df))

        # Per-k files: feature_importance_itemsets_k<N>[_class<C>].csv
        # Columns: [k_value, instance_index, features, itemset].
        for k_val, k_df in per_k_itemsets.items():
            k_path = args.output_dir / f"feature_importance_itemsets_k{k_val}{suffix}.csv"
            k_df.to_csv(k_path, index=False)
            logger.info("  k=%d -> %s  (%d rows)", k_val, k_path.name, len(k_df))

        # ── Save feature importance table ──────────────────────────────────
        # Columns: [feature, k_1, k_3, …, k_N].  One row per feature.
        imp_path = args.output_dir / f"feature_importance{suffix}.csv"
        importance_df.reset_index().to_csv(imp_path, index=False)
        logger.info("Importance -> %s", imp_path)

        # Distances: one row per (boundary_instance, k_neighbour) pair.
        # Columns: [k_value, instance_index, cf_index, k_neighbour_rank, distance].
        dist_path = args.output_dir / f"bocsor_distances{suffix}.csv"
        distances_df.to_csv(dist_path, index=False)
        logger.info("Distances -> %s  (%d rows)", dist_path, len(distances_df))

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
    a("## CatBoost Classifier")
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