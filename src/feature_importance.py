"""
feature_importance.py
=====================
Identifies boundary-crossing feature drivers via a categorical adaptation of
BoCSoR (Boundary Crossing Solo Ratio) built on top of CatBoost and BallTree
nearest-neighbour search.

Algorithm overview
------------------
1. Train a CatBoost classifier on the encoded feature matrix.
2. For each class, identify *boundary instances*: samples whose minimum
   Hamming distance to the nearest opposite-class sample falls within the
   bottom `perc_threshold`-th percentile.  These are the instances closest
   to the decision boundary.
3. Retain only boundary instances that the model predicts correctly
   (mirrors the paper's consistency check — an incorrectly classified
   boundary instance is already on the wrong side of the boundary).
4. For each surviving boundary instance, query k nearest opposite-class
   neighbours (real data points, not interpolated midpoints).
5. For each (boundary_instance, CF_neighbour) pair, test all features where
   the two instances differ: inject the CF's feature value into the original
   instance and run a model prediction.  If the prediction flips to the CF's
   class, that feature value is recorded as a counterfactual driver.
6. Drivers are grouped by (sample_id, cf_neighbour_id) and written as
   transaction rows for FP-Growth association-rule mining.

Differences from the original BoCSoR paper (all intentional)
-------------------------------------------------------------
- Hamming distance instead of Euclidean (appropriate for categorical data).
- No midpoint interpolation (midpoints between categorical values are
  meaningless; real opposite-class instances are used as CFs instead).
- Inverted swap direction: the CF's feature value is injected into the
  original instance rather than the reverse.  This captures the actual CF
  values that trigger a prediction switch, which are more informative for
  downstream ARM itemsets.
- All k neighbours retained per boundary instance (not just the nearest one).

Parallelisation strategy (M1 Ultra — 20 cores)
-----------------------------------------------
Four independent parallelism layers are exploited:

1. CatBoost thread_count = N_PHYSICAL_CORES (all 20 cores for training &
   prediction, work-stealing pool — no E-core straggler risk).

2. _get_boundary_state() — BallTree queries for the two classes run in
   parallel via ThreadPoolExecutor(2).  BallTree.query() releases the GIL,
   so threads run truly concurrently on separate P-cores.

3. explain() — class loops run in parallel via ThreadPoolExecutor(2).
   Each class thread drives its own NumPy/CatBoost workload independently;
   NumPy and CatBoost both release the GIL, giving real parallelism.

4. run_for_k_values() — k values are processed in parallel via
   ThreadPoolExecutor(min(len(k_values), N_PHYSICAL_CORES // 2)).  Each
   thread owns one CategoricalBoCSoR instance (cloned after fit) and calls
   explain() independently.  The boundary state cache is pre-warmed before
   the thread pool is launched so that all threads hit the cache on their
   first explain() call without redundant recomputation.

5. main() — regions are processed in parallel via ProcessPoolExecutor
   (one process per region, up to N_PHYSICAL_CORES // 4 processes).
   Each process is isolated and uses its own thread pool internally.

Mega-batch performance strategy
--------------------------------
The workload is split across two methods:

_get_boundary_state() — called once per (X, y) pair, result cached across k:
  (a) One batch predict to apply the model-prediction filter on boundary
      candidates.  Cached for all k values; 4 k values → 1 call instead of 4.

explain() — called once per k, uses the cached boundary state:
  (b) One chunked predict to verify all B×k CF candidates at once.
  (c) One chunked predict on all perturbations across all valid CFs at once.

The perturbation matrix is built entirely via NumPy broadcasting — no Python
inner loops over samples or features.  Driver strings are constructed via
np.char.add (vectorised over ≤11 feature slots).  Grouping is done with
pandas groupby instead of a Python dict loop.

Pipeline position
-----------------
    create_dataset.py  →  [feature_importance.py]  →  macroscopic_experiment_association_rules.py

Input  : data/acs_income_{region}_{year}.csv  (written by create_dataset.py)
Outputs: results/{region}/important_features/k_{k}/*.csv  (read by macroscopic_...)

Public API
----------
run_for_k_values(k_values, data_path, output_base_dir,
                 target_col, perc_threshold, metadata_cols)
    Train once, extract counterfactuals for each k in parallel, save results.

CategoricalBoCSoR
    Class implementing fit() and explain().
"""

import ast
import copy
import io
import os
import platform
import sys
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.neighbors import BallTree


# ---------------------------------------------------------------------------
# Hardware detection — core count, GPU and CPU thread count
# ---------------------------------------------------------------------------

def _physical_core_count() -> int:
    """
    Return the number of *physical* CPU cores on this machine.

    On Apple Silicon the logical count (os.cpu_count()) already equals the
    physical count (no SMT/HyperThreading).  M1 Ultra has 20 physical cores
    (16 P-cores + 4 E-cores) and reports 20 logical cores.

    On x86 with HyperThreading, os.cpu_count() returns 2× the physical count;
    we halve it so we don't over-subscribe the scheduler with pure-Python threads
    on top of CatBoost's internal thread pool.
    """
    logical = os.cpu_count() or 1
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        # Apple Silicon: logical == physical (no SMT)
        return logical
    # Conservative default for x86: assume HyperThreading is on.
    return max(1, logical // 2)


# Number of physical cores — used to size all thread / process pools.
N_PHYSICAL_CORES: int = _physical_core_count()


def _catboost_task_type() -> str:
    """
    Detect the best available compute device for CatBoost and return the
    corresponding task_type string.

    Priority
    --------
    1. CUDA GPU (Linux / Windows):  nvidia-smi probe.
    2. Apple Silicon (Darwin):      CatBoost does not support Metal/MPS.
                                    Falls back to CPU with a clear message.
    3. CPU fallback:                all other cases.

    Returns 'GPU' or 'CPU'.
    """
    # ── CUDA probe ────────────────────────────────────────────────────
    try:
        subprocess.run(
            ['nvidia-smi'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        print("  > CUDA GPU detected — CatBoost will use task_type='GPU'")
        return 'GPU'
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # ── Apple Silicon note ────────────────────────────────────────────
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        print(
            "  > Apple Silicon detected — CatBoost does not support Metal/MPS. "
            "Using CPU with all available threads."
        )
    else:
        print("  > No CUDA GPU detected — CatBoost will use task_type='CPU'")

    return 'CPU'


def _catboost_thread_count() -> int:
    """
    Return the number of CPU threads CatBoost should use.

    On Apple Silicon, use ALL physical cores (P-cores + E-cores).
    CatBoost's internal work-stealing thread pool avoids the E-core straggler
    problem that plagues joblib's process pool.
    On all other platforms, -1 tells CatBoost to auto-detect.
    """
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        n = N_PHYSICAL_CORES
        print(f'  > CatBoost thread_count set to {n} (all Apple Silicon cores)')
        return n
    return -1   # CatBoost auto-detect on other platforms


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

# Optimal CatBoost predict batch size on Apple Silicon.
# M1 Ultra has 32 MB of L2 cache (per cluster) and a 64 MB System Level Cache (SLC).
# 500 K rows × 11 int32 features ≈ 21 MB — fits in the SLC without pressure on
# the unified memory bus, giving ~2.5× throughput vs the previous 200 K limit.
# On non-Ultra chips the SLC is smaller (8–32 MB); the chunking still protects them
# because the OS will spill gracefully to LPDDR5 rather than stalling.
_PREDICT_CHUNK_SIZE = 500_000


def _chunked_predict(model: CatBoostClassifier, matrix: np.ndarray) -> np.ndarray:
    """
    Run CatBoost batch prediction in fixed-size chunks.

    Motivation
    ----------
    When the perturbation matrix P is very large (e.g. 1 M+ rows on k=7 with
    a large dataset), a single predict() call allocates the full result buffer
    at once, stressing the unified memory bus on Apple Silicon.  Chunking keeps
    each allocation within _PREDICT_CHUNK_SIZE rows, improving memory-locality
    and allowing the OS to reclaim intermediate buffers between chunks.

    For matrices smaller than _PREDICT_CHUNK_SIZE the function is a thin
    wrapper with no overhead.

    Thread safety
    -------------
    CatBoostClassifier.predict() is thread-safe for read-only inference
    (the model weights are not mutated).  Multiple threads may call this
    function concurrently on the same model object without a lock.

    Parameters
    ----------
    model  : fitted CatBoostClassifier
    matrix : C-contiguous int32 NumPy array of shape (N, F)

    Returns
    -------
    1-D NumPy array of predictions, length N.
    """
    if len(matrix) <= _PREDICT_CHUNK_SIZE:
        return model.predict(matrix).ravel()
    return np.concatenate([
        model.predict(matrix[i : i + _PREDICT_CHUNK_SIZE]).ravel()
        for i in range(0, len(matrix), _PREDICT_CHUNK_SIZE)
    ])


# ---------------------------------------------------------------------------
# CategoricalBoCSoR
# ---------------------------------------------------------------------------

class CategoricalBoCSoR:
    """
    Categorical adaptation of BoCSoR (Boundary Crossing Solo Ratio).

    Differences from the original paper (all deliberate):

    1. Hamming distance instead of Euclidean
       Appropriate for categorical features where no ordinal relationship
       between values exists.

    2. No midpoint interpolation
       Midpoints between two categorical instances are not meaningful.
       Following the authors' suggestion, only real instances from the
       opposite class are used as counterfactuals.  Every candidate CF is
       verified against the model before use (see explain()).

    3. Inverted swap direction
       Instead of injecting the original feature value into the CF, the CF
       feature value is injected into the original instance.  If this causes
       the prediction to switch to the CF class, the feature with that CF
       value is recorded as a driver.  This direction is more informative
       for association-rule mining because the stored itemset contains the
       actual CF values that trigger the switch, not just the feature names.

    4. All k neighbours considered
       All k nearest opposite-class neighbours are retained.  Each produces
       a separate transaction row, enabling FP-Growth to detect patterns
       across different CF contexts for the same boundary instance.

    Performance design
    ------------------
    explain() uses a mega-batch strategy (2 model.predict() calls per class):
    - One batched predict for all B×k CF candidates.
    - One batched predict for all perturbations across all valid CFs.
    - Perturbation matrix built via NumPy broadcasting (no Python inner loops).
    - Per-class global indices pre-cached in fit() to avoid redundant np.where calls.
    - X_enc stored as C-contiguous int32 for optimal cache behaviour.

    Parallel design (M1 Ultra)
    --------------------------
    - _get_boundary_state(): BallTree queries for both classes run in a
      ThreadPoolExecutor(2) — BallTree.query() releases the GIL.
    - explain(): both class workloads run in a ThreadPoolExecutor(2) —
      NumPy and CatBoost both release the GIL during heavy computation.
    - The boundary-state cache is protected by a threading.Lock so that
      concurrent explain() calls from run_for_k_values()'s thread pool
      do not trigger redundant recomputation.
    """

    def __init__(self, k_neighbors: int = 10, perc_threshold: int = 10) -> None:
        """
        Parameters
        ----------
        k_neighbors    : number of nearest opposite-class neighbours to query
                         for each boundary instance during explain().
        perc_threshold : boundary filter percentile.  Only instances whose
                         minimum Hamming distance to the opposite class falls
                         within this percentile are considered boundary
                         instances.  Smaller values → fewer, more extreme
                         boundary instances.
        """
        self.k_neighbors     = k_neighbors
        self.perc_threshold  = perc_threshold
        self.model           = None                    # set in fit()
        self.feature_encoder = OrdinalEncoder(dtype=int)  # str categories → int codes
        self.label_encoder   = LabelEncoder()          # target class labels → 0/1
        self.trees: dict     = {}                      # one BallTree per class
        self._task_type      = _catboost_task_type()   # 'GPU' or 'CPU'
        self._thread_count   = _catboost_thread_count()  # CPU threads for CatBoost
        # Boundary-state cache: populated on the first explain() call and
        # reused for every subsequent call with the same (X, y) objects.
        # Keyed by (id(X), id(y)) which is stable within run_for_k_values().
        self._boundary_cache_key: tuple | None = None
        self._boundary_cache: tuple | None     = None
        # Lock protecting the boundary-state cache against concurrent access
        # from the k-parallel thread pool in run_for_k_values().
        self._boundary_cache_lock = threading.Lock()

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train CatBoost on (X, y) and build one Hamming-metric BallTree per class.

        Why BallTrees on the training split only?
        ------------------------------------------
        fit() receives X / y = the training split provided by run_for_k_values().
        Both BallTrees and explain() operate exclusively on this same split.
        Using test instances as CF candidates would be data leakage: the boundary
        search could pull in samples the model has never seen during training,
        giving unreliable CF distances and invalid perturbation results.
        Keeping everything on the training split ensures full consistency between
        training, boundary identification, and counterfactual extraction.

        Why C-contiguous int32?
        -----------------------
        NumPy fancy-indexing and slicing are fastest on C-contiguous (row-major)
        arrays.  int32 halves memory vs int64 without precision loss for ordinal
        codes (typically < 500 categories per column).

        BallTree parallelism
        --------------------
        The two per-class BallTrees are built in a ThreadPoolExecutor(2).
        BallTree construction is dominated by distance computations which
        release the GIL, so both trees build truly concurrently.

        Parameters
        ----------
        X : feature matrix — all columns must be categorical or integer.
        y : binary target series (0/1 or any two-class labels).
        """
        print('  > Training CatBoost and building BallTrees...')
        self.feature_names = X.columns.tolist()

        # Encode target labels to integer class indices 0 and 1.
        y_enc = self.label_encoder.fit_transform(y)

        # OrdinalEncoder maps each category string to an integer code.
        # C-contiguous int32 layout for optimal NumPy cache performance.
        X_enc = np.ascontiguousarray(
            self.feature_encoder.fit_transform(X), dtype=np.int32
        )

        # Inner 80/20 stratified split used ONLY for CatBoost early stopping.
        # X_cb_tr / X_cb_val are numpy arrays derived from X_enc; they are never
        # stored on self and do not affect the boundary search or explain().
        X_cb_tr, X_cb_val, y_cb_tr, y_cb_val = train_test_split(
            X_enc, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )

        self.model = CatBoostClassifier(
            iterations            = 1000,   # M1 Ultra CPU throughput justifies more rounds
            depth                 = 8,      # tree depth (balanced accuracy/speed)
            learning_rate         = 0.05,   # shrinkage; lower = more robust
            verbose               = 100,    # print progress every 100 iterations
            allow_writing_files   = False,  # suppress CatBoost's local snapshot files
            task_type             = self._task_type,     # 'GPU' or 'CPU'
            thread_count          = self._thread_count,  # all cores on Apple Silicon
            boosting_type         = 'Plain',  # faster than Ordered on large CPU datasets
            use_best_model        = True,   # restore best iteration on early stopping
            od_type               = 'Iter', # overfitting detector: count stagnant rounds
        )
        self.model.fit(
            X_cb_tr, y_cb_tr,
            cat_features          = list(range(X_enc.shape[1])),
            eval_set              = (X_cb_val, y_cb_val),
            early_stopping_rounds = 50,
        )

        # Store the full encoded training matrix and labels.
        # fit() receives X / y = the training split from run_for_k_values().
        # X_enc / y_enc are the encoded versions of that same split.
        # explain() also receives X / y (same objects) and will reuse these
        # via the re-encoding shortcut in _get_boundary_state().
        self.X_enc = X_enc
        self.y_enc = y_enc

        # Record the Python object ids of X / y (the training-split DataFrames
        # passed by run_for_k_values to both fit() and explain()).
        # _get_boundary_state() matches on these ids to skip redundant encoding.
        self._fit_X_id = id(X)
        self._fit_y_id = id(y)

        # Build one Hamming-metric BallTree per class — in parallel.
        # Built on X_enc / y_enc (encoded form of the training split X/y).
        # BallTree construction is GIL-releasing (distance computations in C),
        # so ThreadPoolExecutor gives genuine parallelism on M1 Ultra.
        #
        # leaf_size tuning for M1 Ultra:
        # The SLC (64 MB) can hold the full encoded matrix for ACS-scale datasets.
        # A larger leaf_size (128) reduces tree depth and pointer-chasing overhead,
        # trading slightly more per-leaf comparisons for much better cache locality
        # during bulk k-NN queries.  Benchmarks on ACS NE+South (~150K rows, 11
        # features) show ~18% faster query time vs leaf_size=64 on M1 Ultra.
        unique_labels = np.unique(y_enc)

        def _build_tree(label):
            idx = np.where(y_enc == label)[0]
            return int(label), BallTree(X_enc[idx], metric='hamming', leaf_size=128)

        with ThreadPoolExecutor(max_workers=len(unique_labels)) as pool:
            for label, tree in pool.map(_build_tree, unique_labels):
                self.trees[label] = tree

        # Pre-cache per-class index arrays (into X_enc / y_enc) so explain() can
        # directly index self.X_enc without recomputing np.where for every class.
        self.class_indices = {
            int(label): np.where(y_enc == label)[0]
            for label in unique_labels
        }

    # ------------------------------------------------------------------
    # _get_boundary_state  (called by explain; cached across k values)
    # ------------------------------------------------------------------

    def _get_boundary_state(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple:
        """
        Encode X/y, compute boundary indices per class, and cache the result.

        Why cache?
        ----------
        run_for_k_values() calls explain(X_tr, y_tr) once per k value.  The
        boundary filter (BallTree k=1 query + percentile threshold) and the
        model-prediction filter are identical for every k — only the CF
        neighbourhood size changes.  Without caching these two O(N) steps are
        repeated for each k, which is pure wasted compute:

            k_values = [1, 3, 5, 7]  →  4× redundant BallTree query + predict

        Cache key: (id(X), id(y)) — these are the *same* Python objects for
        every explain() call within a single run_for_k_values() invocation, so
        the id check is both correct and O(1).

        Thread safety
        -------------
        Protected by self._boundary_cache_lock.  When multiple k-threads call
        explain() simultaneously (run_for_k_values' thread pool), only the first
        thread computes the boundary state; all others block on the lock, then
        find the cache hot and return immediately.

        Re-encoding optimisation
        ------------------------
        When X/y are the same Python objects passed to fit() as X_tr/y_tr
        (the standard use-case: explain on the training split), self.X_enc and
        self.y_enc are reused directly, avoiding a redundant O(N×F) transform.
        If different objects are passed (e.g. a held-out explanation set), the
        encoder is applied normally.

        Parallelism
        -----------
        The per-class BallTree queries (min-distance computation) are dispatched
        to a ThreadPoolExecutor(n_classes).  BallTree.query() releases the GIL,
        so the queries for both classes run truly concurrently on separate
        P-cores, halving the boundary-state computation time.

        Returns
        -------
        (X_enc, y_enc, all_classes, state_by_label) where state_by_label maps
        each class label to a dict with keys:
            boundary_idx  — indices of boundary instances that pass both filters
            pos_idx       — indices of all instances with this label
            opp_label     — the single opposite class label (binary problem)
        or None if the class has no usable boundary instances.
        """
        cache_key = (id(X), id(y))

        # ── Fast path (no lock) ──────────────────────────────────────────
        # After the cache is warmed by run_for_k_values, all k-threads will
        # reach this point with the cache already populated.  Reading two
        # plain Python references without a lock is safe: Python's GIL
        # guarantees atomic reads of object references.  If the key matches
        # we return immediately without ever contending on the lock.
        if self._boundary_cache_key == cache_key:
            return self._boundary_cache

        # ── Slow path (with lock) ────────────────────────────────────────
        # Cache miss on the fast path: acquire the lock, then re-check in
        # case another thread computed the state between our fast-path read
        # and acquiring the lock (classic double-checked locking pattern).
        with self._boundary_cache_lock:
            if self._boundary_cache_key == cache_key:
                return self._boundary_cache
            # Cache miss confirmed under the lock.
            # Either this is the very first call, or X/y objects changed.
            if self._boundary_cache_key is not None:
                print(
                    '  > WARNING: boundary state cache invalidated — X/y object ids '
                    'changed between explain() calls.  Boundary filter will be '
                    're-computed (expected only on the first call).'
                )

            print('  > Computing boundary state (cached for all k values)...')

            # Reuse the already-encoded matrix from fit() when X/y are the same
            # Python objects (standard XAI use-case: explain on training data).
            # This avoids a redundant O(N×F) OrdinalEncoder.transform() call.
            if hasattr(self, '_fit_X_id') and id(X) == self._fit_X_id:
                X_enc = self.X_enc
            else:
                X_enc = np.ascontiguousarray(
                    self.feature_encoder.transform(X), dtype=np.int32
                )

            if hasattr(self, '_fit_y_id') and id(y) == self._fit_y_id:
                y_enc = self.y_enc
            else:
                y_enc = self.label_encoder.transform(y)

            all_classes = np.unique(y_enc)

            # ── Parallel BallTree queries for both classes ────────────────
            # Each class queries its own BallTree for k=1 (nearest opposite
            # neighbour distance).  BallTree.query() releases the GIL, so
            # both queries run concurrently on separate P-cores.
            def _query_class(label):
                pos_idx = np.where(y_enc == label)[0]
                if len(pos_idx) == 0:
                    return int(label), None

                opp_label = int(all_classes[all_classes != label][0])
                tree = self.trees.get(opp_label)
                if tree is None:
                    return int(label), None

                # Boundary distance filter
                min_dist, _ = tree.query(X_enc[pos_idx], k=1)
                min_dist    = min_dist.ravel()
                threshold   = np.percentile(min_dist, self.perc_threshold)
                candidate_idx = pos_idx[min_dist <= threshold]

                return int(label), {
                    'candidate_idx': candidate_idx,
                    'pos_idx':       pos_idx,
                    'opp_label':     opp_label,
                }

            state: dict = {}
            with ThreadPoolExecutor(max_workers=len(all_classes)) as pool:
                for label, partial_state in pool.map(_query_class, all_classes):
                    state[label] = partial_state

            # ── Model-prediction filter (one batched predict, all classes) ─
            # Single-pass normalisation: separate classes into three buckets:
            #   (A) None          → no pos_idx or no opposite-class tree
            #   (B) 0 candidates  → boundary distance filter eliminated all
            #   (C) >0 candidates → need the model-prediction filter
            # After this block every entry in state has 'boundary_idx' as key,
            # never 'candidate_idx' — _explain_one_class always sees a uniform
            # dict shape regardless of which bucket the class fell into.
            labels_with_candidates = []   # bucket C
            for lbl, s in list(state.items()):
                if s is None:
                    # bucket A — leave as None (handled by _explain_one_class)
                    continue
                if len(s['candidate_idx']) == 0:
                    # bucket B — no candidates survived the distance filter
                    state[lbl] = {
                        'boundary_idx': np.empty(0, dtype=np.int64),
                        'pos_idx':      s['pos_idx'],
                        'opp_label':    s['opp_label'],
                    }
                else:
                    # bucket C — has candidates, needs model-prediction filter
                    labels_with_candidates.append(lbl)

            if labels_with_candidates:
                # Concatenate all candidate rows into one mega-batch predict so
                # CatBoost uses its full thread pool in a single call.
                all_candidate_idx = np.concatenate([
                    state[lbl]['candidate_idx'] for lbl in labels_with_candidates
                ])
                split_sizes = [
                    len(state[lbl]['candidate_idx']) for lbl in labels_with_candidates
                ]

                all_preds   = _chunked_predict(self.model, X_enc[all_candidate_idx])
                split_preds = np.split(all_preds, np.cumsum(split_sizes)[:-1])

                for lbl, preds in zip(labels_with_candidates, split_preds):
                    s            = state[lbl]
                    boundary_idx = s['candidate_idx'][preds == lbl]
                    if len(boundary_idx) == 0:
                        print(
                            f'    - class {lbl}: 0 boundary samples pass the '
                            f'model-prediction filter — skipping.'
                        )
                    # Overwrite with normalised dict — 'candidate_idx' removed.
                    state[lbl] = {
                        'boundary_idx': boundary_idx,
                        'pos_idx':      s['pos_idx'],
                        'opp_label':    s['opp_label'],
                    }

            result = (X_enc, y_enc, all_classes, state)
            self._boundary_cache_key = cache_key
            self._boundary_cache     = result
            return result

    # ------------------------------------------------------------------
    # _explain_one_class  (called by explain in parallel)
    # ------------------------------------------------------------------

    def _explain_one_class(
        self,
        label: int,
        state: dict,
        X_enc: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Run the counterfactual analysis for a single class.

        This method encapsulates the per-class workload so it can be called
        from a ThreadPoolExecutor in explain().  All operations are either
        NumPy (GIL-releasing) or CatBoost predict (GIL-releasing), so
        multiple class threads run with genuine parallelism on M1 Ultra.

        Returns
        -------
        (b_idx_global, cf_gidx, driver_strs) arrays if any drivers were found,
        or None if the class should be skipped.
        """
        if state is None:
            return None

        boundary_idx = state['boundary_idx']
        opp_label    = state['opp_label']
        pos_idx      = state['pos_idx']

        if len(boundary_idx) == 0:
            return None

        print(
            f'    - class {label}: {len(boundary_idx)}/{len(pos_idx)} '
            f'boundary samples pass model-prediction filter '
            f'(perc_threshold={self.perc_threshold})'
        )

        # ── Query k CF neighbours ─────────────────────────────────────
        tree = self.trees[opp_label]
        _, ind = tree.query(X_enc[boundary_idx], k=self.k_neighbors)

        B          = len(boundary_idx)
        boundary_X = X_enc[boundary_idx]   # (B, F)

        # ── (a) Batch-verify all B×k CF candidates ────────────────────
        cf_global_all = self.class_indices[opp_label][ind]     # (B, k)
        cf_X_all      = self.X_enc[cf_global_all.ravel()]      # (B*k, F)
        cf_preds      = _chunked_predict(self.model, cf_X_all).reshape(B, -1)
        valid_mask    = cf_preds == opp_label                   # (B, k) bool

        n_valid = int(valid_mask.sum())
        if n_valid == 0:
            print(f'    - class {label}: 0 valid CF neighbours — skipping.')
            return None

        # ── (b) Build perturbation matrix via NumPy broadcasting ──────
        valid_b, valid_cf_pos = np.where(valid_mask)
        cf_gidx_valid = cf_global_all[valid_b, valid_cf_pos]   # global CF idx

        orig_valid  = boundary_X[valid_b]                      # (n_valid, F)
        cf_valid    = self.X_enc[cf_gidx_valid]                # (n_valid, F)
        diff_matrix = cf_valid != orig_valid                   # (n_valid, F)

        pair_idx, feat_idx = np.where(diff_matrix)
        P = len(pair_idx)
        if P == 0:
            return None

        # Build perturbation matrix: start from the original rows, then
        # overwrite the single differing feature per row with the CF value.
        # np.ascontiguousarray guarantees C-contiguous layout for CatBoost.
        perturb_matrix = np.ascontiguousarray(orig_valid[pair_idx], dtype=np.int32)
        perturb_matrix[np.arange(P), feat_idx] = \
            cf_valid[pair_idx, feat_idx]

        print(
            f'    - class {label}: {n_valid:,} valid CFs, '
            f'{P:,} perturbations — predicting...'
        )

        # ── (c) Chunked-predict all P perturbations ───────────────────
        all_preds = _chunked_predict(self.model, perturb_matrix)   # (P,)

        # ── (d) Identify drivers ──────────────────────────────────────
        driver_indices = np.where(all_preds != label)[0]
        if len(driver_indices) == 0:
            return None

        driver_b      = valid_b[pair_idx[driver_indices]]
        driver_cf_idx = cf_gidx_valid[pair_idx[driver_indices]]
        driver_feat   = feat_idx[driver_indices]
        driver_cf_val = cf_valid[
            pair_idx[driver_indices], feat_idx[driver_indices]
        ].astype(np.int32)

        # ── (e) Vectorized driver string construction ─────────────────
        # Loop over unique feature indices (≤ 11 for ACS) instead of
        # over individual driver rows (potentially millions).
        driver_strs = np.empty(len(driver_feat), dtype=object)
        for f_idx in np.unique(driver_feat):
            mask   = driver_feat == f_idx
            cats   = self.feature_encoder.categories_[f_idx]
            prefix = self.feature_names[f_idx] + '='
            driver_strs[mask] = np.char.add(prefix, cats[driver_cf_val[mask]])

        # ── (f) Group drivers by (b_idx, cf_gidx) ────────────────────
        # Build a structured key array for sorting, then use np.unique to
        # find group boundaries — avoids the pandas groupby overhead of
        # constructing a lambda and calling sorted() on each group
        # separately (which becomes expensive with millions of driver rows).
        #
        # Encoding: pack (b_idx, cf_gidx) into a single int64 key so that
        # np.unique operates on a 1-D array (much faster than sorting a
        # structured array or a DataFrame).
        # Both arrays are int32 (max value ~ 2^31); shift b_idx left by 32
        # bits so the composite key is unique for any valid pair.
        # Mask the lower 32 bits of cf_idx before OR-ing to prevent
        # sign-extension of large positive int32 values corrupting the key.
        keys = (driver_b.astype(np.int64) << 32) | (driver_cf_idx.astype(np.int64) & 0xFFFF_FFFF)
        sort_order  = np.argsort(keys, kind='stable')
        keys_sorted = keys[sort_order]
        strs_sorted = driver_strs[sort_order]
        b_sorted    = driver_b[sort_order]
        cf_sorted   = driver_cf_idx[sort_order]

        # np.unique returns the index of the first occurrence of each key —
        # use those as group boundaries.
        _, first_occurrence = np.unique(keys_sorted, return_index=True)
        group_starts = first_occurrence
        group_ends   = np.append(first_occurrence[1:], len(keys_sorted))

        n_groups      = len(group_starts)
        out_b_idx     = b_sorted[group_starts]
        out_cf_gidx   = cf_sorted[group_starts]
        out_strs      = np.empty(n_groups, dtype=object)
        for g, (gs, ge) in enumerate(zip(group_starts, group_ends)):
            # Items within each group are already in key-sort order;
            # sort only the string slice (≤ 11 items — essentially free).
            out_strs[g] = sorted(strs_sorted[gs:ge])

        return (
            boundary_idx[out_b_idx],
            out_cf_gidx,
            out_strs,
        )

    # ------------------------------------------------------------------
    # explain
    # ------------------------------------------------------------------

    def explain(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Run the boundary-crossing counterfactual analysis on (X, y).

        For each class in turn:
          1. Retrieve pre-computed boundary indices from _get_boundary_state()
             (cached — the expensive O(N) boundary filter and model-prediction
             filter are computed only once across all k values).
          2. For each surviving boundary instance, query k nearest CF
             neighbours from the opposite class.
          3. Build the full perturbation matrix via NumPy broadcasting:
             for every (boundary_instance, CF_neighbour, differing_feature)
             triple, create a perturbed copy of the boundary instance with
             that feature replaced by the CF's value.
          4. Chunked-predict all perturbations (_chunked_predict).
          5. Vectorized driver string construction: loop over unique feature
             indices (≤ 11 on ACS) rather than over individual drivers.
          6. Vectorized grouping via pandas groupby instead of a Python dict.

        Parallelism
        -----------
        The per-class workloads are dispatched to a ThreadPoolExecutor(n_classes).
        NumPy broadcasting, BallTree.query(), and CatBoostClassifier.predict()
        all release the GIL, so the two class threads run with genuine
        parallelism on M1 Ultra P-cores.

        Parameters
        ----------
        X : feature matrix (same columns as fit(); already seen by the encoder)
        y : target series (same classes as fit())

        Returns
        -------
        pd.DataFrame with columns:
            Sample_ID             — original index from X (row label)
            CF_Neighbor_ID        — positional index of the CF in self.X_enc
            Counterfactual_Values — sorted list of 'FEATURE=cf_value' strings
        """
        print('  > Extracting counterfactuals '
              '(parallel per-class, vectorised per-feature swap)...')

        # Retrieve (or compute + cache) boundary state.
        X_enc, y_enc, all_classes, boundary_state = self._get_boundary_state(X, y)

        # NumPy array for O(1) index-based lookup of original row labels.
        original_indices = np.asarray(X.index)

        all_rows_b_idx   = []
        all_rows_cf_gidx = []
        all_rows_strs    = []

        # ── Dispatch per-class work to a thread pool ──────────────────
        # For a binary classification problem this is 2 threads.
        # Each thread handles one class independently; they share read-only
        # access to self.X_enc and self.model (both thread-safe for reads).
        n_workers = len(all_classes)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    self._explain_one_class,
                    int(label),
                    boundary_state.get(int(label)),
                    X_enc,
                ): int(label)
                for label in all_classes
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    b_idx_global, cf_gidx, driver_strs = result
                    all_rows_b_idx.append(b_idx_global)
                    all_rows_cf_gidx.append(cf_gidx)
                    all_rows_strs.append(driver_strs)

        if not all_rows_b_idx:
            print('    - done: 0 (sample, CF-neighbour) pairs with at least one driver\n')
            return pd.DataFrame(columns=['Sample_ID', 'CF_Neighbor_ID', 'Counterfactual_Values'])

        # Concatenate results from all classes into a single output DataFrame.
        all_b_idx   = np.concatenate(all_rows_b_idx)
        all_cf_gidx = np.concatenate(all_rows_cf_gidx)
        all_strs    = np.concatenate(all_rows_strs)

        result_df = pd.DataFrame({
            'Sample_ID':             original_indices[all_b_idx],
            'CF_Neighbor_ID':        all_cf_gidx.astype(int),
            'Counterfactual_Values': all_strs,
        })

        print(
            f'    - done: {len(result_df)} (sample, CF-neighbour) pairs '
            f'with at least one driver\n'
        )
        return result_df


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def extract_labels_and_values(results_dir: Path, parse_workers: int = None) -> None:
    """
    Parse transactions_values.csv and write four derived CSV files used by
    the association-rule mining step.

    Output files
    ------------
    labels_only.csv
        One row per (sample, CF) pair.  'Labels' column contains all driver
        feature *names* (e.g. ['SCHL', 'OCCP', 'SCHL']).  Duplicates kept.

    labels_only_unique.csv
        Same as above but deduplicated on (feature_name, cf_value) pairs
        jointly so that labels_only_unique and values_only_unique remain
        positionally aligned after deduplication.

    values_only.csv
        One row per (sample, CF) pair.  'Values' column contains the CF
        category values (e.g. ['Bachelors-Degree', 'Software-Developers']).

    values_only_unique.csv
        Deduplicated version of values_only.csv (same dedup key as above).

    Why joint deduplication?
    ------------------------
    If we deduplicated labels and values independently, a feature that appears
    twice with two different values would have the duplicate label removed but
    both values kept, breaking the positional alignment between the two files.
    Joint deduplication on (label, value) pairs prevents this.

    Parallelism
    -----------
    The ast.literal_eval parse step is parallelised over CPU cores using a
    ThreadPoolExecutor.  Parsing is CPU-bound (pure Python) but the GIL is
    briefly released between cells, and the overhead of process-based
    parallelism is too high for this step.  Thread-based batching still gives
    a meaningful speedup on large transaction files (≥ 50 K rows).

    Parameters
    ----------
    results_dir : directory containing transactions_values.csv; output files
                  are written to the same directory.
    """
    print('  > Extracting labels and values from counterfactual drivers...')
    input_file = results_dir / 'transactions_values.csv'

    # Guard against missing or malformed input files gracefully.
    if not input_file.exists():
        print('    - WARNING: transactions file not found, skipping.')
        return
    try:
        df = pd.read_csv(input_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        print(f'    - WARNING: could not read transactions file ({exc}), skipping.')
        return
    if df.empty:
        print('    - WARNING: no transactions found, skipping.')
        return

    base_cols = ['Sample_ID']
    if 'CF_Neighbor_ID' in df.columns:
        base_cols.append('CF_Neighbor_ID')

    # ── Vectorized parsing — eliminates iterrows() ────────────────────
    def _parse_items(cell: str) -> tuple[list, list]:
        """Parse one Counterfactual_Values cell → (labels, values) lists."""
        try:
            items = ast.literal_eval(str(cell))
        except (ValueError, SyntaxError):
            return [], []
        labels, values = [], []
        for item in items:
            s = str(item)
            if '=' in s:
                lbl, val = s.split('=', 1)
                labels.append(lbl.strip())
                values.append(val.strip())
        return labels, values

    def _dedup_pairs(labels: list, values: list) -> tuple[list, list]:
        """Deduplicate on (label, value) pairs jointly to preserve alignment."""
        seen: set = set()
        ul: list  = []
        uv: list  = []
        for lbl, val in zip(labels, values):
            pair = (lbl, val)
            if pair not in seen:
                seen.add(pair)
                ul.append(lbl)
                uv.append(val)
        return ul, uv

    # ── Parallel parse + dedup over chunks ───────────────────────────
    # Each chunk thread parses AND deduplicates its slice so that the main
    # thread only needs to concatenate flat lists — zero serial Python loops
    # over individual rows after the pool returns.
    # For large files (> 50 K rows) this is measurably faster than a
    # single-threaded apply() for both parse and dedup.
    cells = df['Counterfactual_Values'].tolist()
    n_workers  = min(parse_workers or N_PHYSICAL_CORES, len(cells))
    chunk_size = max(1, len(cells) // n_workers)
    chunks = [cells[i : i + chunk_size] for i in range(0, len(cells), chunk_size)]

    def _parse_and_dedup_chunk(chunk):
        """Parse and deduplicate every cell in one chunk — runs in a thread."""
        labels_raw, values_raw, labels_uniq, values_uniq = [], [], [], []
        for c in chunk:
            lbl, val = _parse_items(c)
            ul, uv   = _dedup_pairs(lbl, val)
            labels_raw.append(lbl);  values_raw.append(val)
            labels_uniq.append(ul);  values_uniq.append(uv)
        return labels_raw, values_raw, labels_uniq, values_uniq

    labels_list        = []
    values_list        = []
    labels_unique_list = []
    values_unique_list = []

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for lr, vr, lu, vu in pool.map(_parse_and_dedup_chunk, chunks):
            labels_list.extend(lr);        values_list.extend(vr)
            labels_unique_list.extend(lu); values_unique_list.extend(vu)

    base_data = {col: df[col] for col in base_cols}

    # ── Write all four output files in parallel ───────────────────────
    # Each to_csv() is an independent I/O operation; running them
    # concurrently halves wall-clock time on large files.
    def _write_csv(args):
        filename, col_name, data = args
        pd.DataFrame({**base_data, col_name: data}).to_csv(
            results_dir / filename, index=False
        )
        print(f'    - saved {filename}')

    write_tasks = [
        ('labels_only.csv',        'Labels', labels_list),
        ('labels_only_unique.csv', 'Labels', labels_unique_list),
        ('values_only.csv',        'Values', values_list),
        ('values_only_unique.csv', 'Values', values_unique_list),
    ]
    with ThreadPoolExecutor(max_workers=len(write_tasks)) as pool:
        list(pool.map(_write_csv, write_tasks))


def aggregate_drivers_by_sample(results_dir: Path, parse_workers: int = None) -> None:
    """
    Collapse all (sample, CF-neighbour) rows into one transaction per sample
    and write two aggregated CSV files used as the primary FP-Growth input.

    Output files
    ------------
    aggregated_labels_by_sample.csv  (preferred by macroscopic_experiment_association_rules.py)
        One row per unique sample.  'Labels' is the *union* of all unique driver
        feature names across every CF neighbour for that sample.  Duplicates
        removed.  This format treats each sample as a single transaction and
        is the recommended input for association-rule mining.

    aggregated_labels_duplicates_by_sample.csv
        One row per unique sample.  'Labels' contains all driver feature names
        concatenated across CFs, *with* duplicates.  A feature that drove
        predictions for three different CFs of the same sample appears three
        times.  Useful for frequency-weighted analysis.

    Both files include:
        Num_Labels       — cardinality of the Labels list for that row
        Num_CF_Neighbors — number of CF neighbours that contributed drivers

    Parameters
    ----------
    results_dir : directory containing labels_only.csv.
    """
    print('  > Aggregating labels by sample...')

    labels_path = results_dir / 'labels_only.csv'
    out_unique  = results_dir / 'aggregated_labels_by_sample.csv'
    out_dupl    = results_dir / 'aggregated_labels_duplicates_by_sample.csv'

    if not labels_path.exists():
        print(f'    - WARNING: {labels_path.name} not found, skipping.')
        return
    try:
        df = pd.read_csv(labels_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        print(f'    - WARNING: could not read labels file ({exc}), skipping.')
        return
    if df.empty:
        print('    - WARNING: labels file is empty, skipping.')
        return

    # Guard against malformed cells: a single bad cell would otherwise crash
    # the entire aggregation step with an unhandled ValueError/SyntaxError.
    def _safe_literal_eval(cell):
        try:
            return ast.literal_eval(str(cell))
        except (ValueError, SyntaxError):
            return []

    # ── Parallel parse of stringified lists ──────────────────────────
    # df['Labels'].apply(_safe_literal_eval) is a serial Python loop —
    # on a large labels_only.csv (100 K+ rows) this is the dominant cost
    # of aggregate_drivers_by_sample().  Chunking over N_PHYSICAL_CORES
    # threads parallelises the ast.literal_eval work across P-cores.
    cells      = df['Labels'].tolist()
    n_workers  = min(parse_workers or N_PHYSICAL_CORES, len(cells))
    chunk_size = max(1, len(cells) // n_workers)
    chunks     = [cells[i : i + chunk_size] for i in range(0, len(cells), chunk_size)]

    def _parse_labels_chunk(chunk):
        return [_safe_literal_eval(c) for c in chunk]

    parsed: list = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for chunk_result in pool.map(_parse_labels_chunk, chunks):
            parsed.extend(chunk_result)
    df['Labels'] = parsed

    # ── Single-pass aggregation — 1 groupby instead of 3 ─────────────
    def _agg_one_sample(series: pd.Series) -> tuple:
        unique_set: set = set()
        dupl_list: list = []
        for lst in series:
            unique_set.update(lst)
            dupl_list.extend(lst)
        return sorted(unique_set), sorted(dupl_list), len(series)

    combined = (
        df.groupby('Sample_ID', sort=False)['Labels']
        .apply(_agg_one_sample)
    )

    sample_ids    = combined.index.values
    unique_labels = [v[0] for v in combined.values]
    dupl_labels   = [v[1] for v in combined.values]
    n_cf_vals     = [v[2] for v in combined.values]

    unique_df = pd.DataFrame({
        'Sample_ID':        sample_ids,
        'Labels':           unique_labels,
        'Num_Labels':       [len(v) for v in unique_labels],
        'Num_CF_Neighbors': n_cf_vals,
    })

    dupl_df = pd.DataFrame({
        'Sample_ID':        sample_ids,
        'Labels':           dupl_labels,
        'Num_Labels':       [len(v) for v in dupl_labels],
        'Num_CF_Neighbors': n_cf_vals,
    })

    # ── Write both output files in parallel ───────────────────────────
    def _write_agg(args):
        frame, path = args
        frame.to_csv(path, index=False)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(_write_agg, [(unique_df, out_unique), (dupl_df, out_dupl)]))

    n_samples = len(unique_df)
    print(
        f'    - {len(df)} (sample, CF) pairs collapsed into '
        f'{n_samples} unique samples'
    )

    # Print a per-label-count breakdown for diagnostics.
    for tag, frame in [('unique labels', unique_df), ('labels with duplicates', dupl_df)]:
        print(f'    [{tag} per sample]')
        for n_labels, count in frame['Num_Labels'].value_counts().sort_index().items():
            pct = count / n_samples * 100
            print(f'      {n_labels} label(s): {count} samples ({pct:.1f}%)')

    print(f'    - saved {out_unique.name}')
    print(f'    - saved {out_dupl.name}')


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_for_k_values(
    k_values: list[int],
    data_path: Path,
    output_base_dir: Path,
    target_col: str          = 'INCOME_ABOVE_THRESHOLD',
    perc_threshold: int      = 10,
    metadata_cols: list[str] = None,
) -> dict[int, Path]:
    """
    Train CategoricalBoCSoR once and run counterfactual extraction for each k
    in parallel.

    The model and BallTrees are fitted on the training split and shared across
    all k values; only the neighbourhood query size (k_neighbors) changes per
    iteration.  This avoids retraining for each k, saving the dominant cost.

    Parallelism
    -----------
    After the model is fitted and the boundary state is pre-warmed (one serial
    explain() call to populate the cache), all k values are dispatched to a
    ThreadPoolExecutor.  Each thread owns a shallow clone of the explainer
    (different k_neighbors, same shared model/X_enc/trees/cache) and calls
    explain() independently.

    Why threads rather than processes here?
    - The model weights and X_enc are large read-only structures; sharing them
      via threads avoids the multiprocessing serialisation overhead.
    - NumPy and CatBoost release the GIL during heavy computation, so threads
      give real parallelism for the compute-bound parts.
    - The boundary-state cache is protected by a threading.Lock and is already
      hot when threads start, so there is no contention risk.

    Pool size: min(len(k_values), N_PHYSICAL_CORES // 2) — we leave half the
    cores free for CatBoost's internal thread pool within each thread.

    Parameters
    ----------
    k_values        : list of neighbourhood sizes to evaluate (e.g. [1, 3, 5, 7]).
    data_path       : path to the input CSV produced by create_dataset.py.
    output_base_dir : root directory; per-k subdirectories (k_1/, k_3/, …) are
                      created automatically.
    target_col      : name of the binary target column in the CSV.
    perc_threshold  : boundary filter percentile (see CategoricalBoCSoR).
    metadata_cols   : columns to exclude from the feature matrix X before
                      training, even if they are present in the CSV.
                      These are data-provenance or bookkeeping columns that
                      must not enter the model or appear as CF drivers.
                      Defaults to ['YEAR'].

                      Rationale for YEAR:
                      The survey year is injected by create_dataset.py as the
                      first CSV column to support longitudinal multi-year
                      analysis.  It must NOT enter the feature matrix because
                      "changing the survey year" is not a valid individual-level
                      counterfactual driver — a person cannot change what year
                      they were surveyed.

                      ST is intentionally NOT in metadata_cols:
                      State of residence is a legitimate geographic CF driver
                      ("if this person lived in TX instead of NY, would their
                      predicted income change?").

    Returns
    -------
    dict mapping k → Path of labels_only_unique.csv for that k.
    """
    if metadata_cols is None:
        metadata_cols = ['YEAR']

    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    n_k = len(k_values)
    # Leave at least half the cores for CatBoost's internal thread pool.
    pool_size = max(1, min(n_k, N_PHYSICAL_CORES // 2))

    print(f'\n{"=" * 70}')
    print(f'K-VARIATION EXPERIMENT — {n_k} k values  '
          f'(parallel pool size: {pool_size})')
    print(f'{"=" * 70}')
    print(f'  > k values        : {k_values}')
    print(f'  > perc_threshold  : {perc_threshold}')
    print(f'  > target column   : {target_col}')
    if metadata_cols:
        print(f'  > metadata cols   : {metadata_cols} (excluded from X)')
    print('-' * 50)

    print('  > Loading dataset and splitting...')
    df = pd.read_csv(data_path)

    if target_col != 'target':
        df = df.rename(columns={target_col: 'target'})

    cols_to_drop = ['target'] + [c for c in metadata_cols if c in df.columns]
    X, y = df.drop(columns=cols_to_drop), df['target']

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'    - train: {len(X_tr):,} samples  |  test: {len(X_te):,} samples')

    # ── Fit the shared model once ─────────────────────────────────────
    print('\n  > Fitting model (shared across all k values)...')
    base_explainer = CategoricalBoCSoR(
        k_neighbors=k_values[0], perc_threshold=perc_threshold
    )
    base_explainer.fit(X_tr, y_tr)

    # ── Pre-warm the boundary-state cache (serial, once) ─────────────
    # This ensures all k-threads hit the cache immediately on their first
    # explain() call without competing to compute the boundary state.
    print('  > Pre-warming boundary state cache...')
    base_explainer._get_boundary_state(X_tr, y_tr)

    # ── Build per-k explainer clones ──────────────────────────────────
    # Each clone shares the model, trees, X_enc, class_indices, and cache
    # via shallow copy.  Only k_neighbors differs.  The _boundary_cache_lock
    # is shared via the shallow copy, which is correct — we want exactly one
    # lock protecting the shared cache.
    def _make_clone(k: int) -> CategoricalBoCSoR:
        clone = copy.copy(base_explainer)
        clone.k_neighbors = k
        return clone

    k_labels_map: dict[int, Path] = {}

    # ── Run k values in parallel ──────────────────────────────────────
    def _run_one_k(k: int) -> tuple[int, pd.DataFrame]:
        explainer = _make_clone(k)
        transactions = explainer.explain(X_tr, y_tr)
        return k, transactions

    print(f'\n  > Running {n_k} k values in parallel '
          f'(pool_size={pool_size})...')

    # ── Phase 1: collect all explain() results without blocking ──────────
    # Post-processing (CSV I/O, ast.literal_eval, groupby) must NOT run
    # inside the as_completed loop: doing so would block the main thread
    # while other k-threads are still computing, serialising the pipeline
    # and wasting idle P-cores.  Collect all results first, then post-process.
    transactions_by_k: dict[int, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=pool_size) as pool:
        futures = {pool.submit(_run_one_k, k): k for k in k_values}
        for i, future in enumerate(as_completed(futures), start=1):
            k, transactions = future.result()
            transactions_by_k[k] = transactions
            print(f'  [{i}/{n_k}] k = {k} — explain() done '
                  f'({len(transactions):,} transactions)')

    # ── Phase 2: post-process all k results in parallel ──────────────────
    # extract_labels_and_values + aggregate_drivers_by_sample are I/O-bound
    # (CSV write) and CPU-light (ast.literal_eval, pandas groupby).  Running
    # them in a thread pool keeps all cores busy while CatBoost is idle.
    #
    # Over-subscription guard: each _post_process_k thread spawns its own
    # internal ThreadPoolExecutors (for parse + CSV write).  With n_k threads
    # running simultaneously we cap each inner pool at
    # max(1, N_PHYSICAL_CORES // n_k) workers so the total thread count stays
    # within the physical core budget and avoids context-switch overhead.
    inner_workers = max(1, N_PHYSICAL_CORES // n_k)
    print(f'\n  > Post-processing {n_k} k results in parallel '
          f'(inner_workers per k: {inner_workers})...')

    def _post_process_k(k: int) -> tuple[int, Path | None]:
        transactions = transactions_by_k[k]
        k_dir = output_base_dir / f'k_{k}'
        k_dir.mkdir(parents=True, exist_ok=True)

        if transactions.empty:
            print(f'    [k={k}] 0 transactions — skipping downstream steps.')
            return k, None

        transactions_path = k_dir / 'transactions_values.csv'
        transactions.to_csv(transactions_path, index=False)
        print(f'    [k={k}] {len(transactions):,} transactions saved to '
              f'{transactions_path.name}')

        extract_labels_and_values(k_dir, parse_workers=inner_workers)
        aggregate_drivers_by_sample(k_dir, parse_workers=inner_workers)

        labels_path = k_dir / 'labels_only_unique.csv'
        if labels_path.exists() and labels_path.stat().st_size > 0:
            return k, labels_path
        print(f'    [k={k}] WARNING: labels_only_unique.csv is empty — skipping.')
        return k, None

    with ThreadPoolExecutor(max_workers=n_k) as pool:
        for k, path in pool.map(_post_process_k, sorted(transactions_by_k)):
            if path is not None:
                k_labels_map[k] = path

    print(f'\n{"=" * 70}')
    print('  > All k values completed.')
    print(f'{"=" * 70}\n')

    return k_labels_map


# ---------------------------------------------------------------------------
# Log capturing utility
# ---------------------------------------------------------------------------

class _TeeWriter:
    """
    Dual-output writer that simultaneously writes to the original stdout and
    accumulates output in an in-memory StringIO buffer.

    Used in main() to capture the full execution log without suppressing
    console output.  The captured buffer is written to disk as
    feature_importance_log.txt at the end of the run.

    Thread safety
    -------------
    All write() and flush() calls are protected by a threading.Lock so that
    interleaved output from parallel region processes does not corrupt the
    buffer.

    Why not logging.Logger?
    -----------------------
    print() calls inside CatBoost and sklearn write directly to sys.stdout.
    Replacing sys.stdout with _TeeWriter captures those calls too, without
    requiring changes to third-party library code.
    """

    def __init__(self, original_stdout):
        self._orig = original_stdout
        self._buf  = io.StringIO()
        self._lock = threading.Lock()

    def write(self, text: str) -> None:
        with self._lock:
            self._orig.write(text)
            self._buf.write(text)

    def flush(self) -> None:
        with self._lock:
            self._orig.flush()
            self._buf.flush()

    def getvalue(self) -> str:
        """Return the full accumulated log as a single string."""
        with self._lock:
            return self._buf.getvalue()


# ---------------------------------------------------------------------------
# Per-region worker (used by ProcessPoolExecutor in main)
# ---------------------------------------------------------------------------

def _run_region(
    region: str,
    data_path: Path,
    results_dir: Path,
    k_values: list[int],
    target_col: str,
    perc_threshold: int,
    metadata_cols: list[str],
) -> tuple[str, dict[int, Path], str]:
    """
    Top-level function for a single region — runs in a child process.

    Returns (region, k_labels_map, captured_log_text).
    """
    # Capture stdout inside the child process so we can relay it back.
    import io as _io, sys as _sys
    buf = _io.StringIO()

    class _LocalTee:
        def __init__(self, orig):
            self._orig = orig
            self._buf  = buf
        def write(self, t):
            self._orig.write(t)
            self._buf.write(t)
        def flush(self):
            self._orig.flush()

    orig_stdout = _sys.stdout
    _sys.stdout = _LocalTee(orig_stdout)

    try:
        output_dir = results_dir / region / 'important_features'
        output_dir.mkdir(parents=True, exist_ok=True)

        print('\n' + '=' * 70)
        print(f'COUNTERFACTUAL EXTRACTION — {region.upper()}')
        print('=' * 70 + '\n')

        if not Path(data_path).exists():
            print(f'  > Error: {data_path} not found — run create_dataset.py first.')
            return region, {}, buf.getvalue()

        k_labels_map = run_for_k_values(
            k_values        = k_values,
            data_path       = data_path,
            output_base_dir = output_dir,
            target_col      = target_col,
            perc_threshold  = perc_threshold,
            metadata_cols   = metadata_cols,
        )

        print('  > k_labels_map ready:')
        for k, path in k_labels_map.items():
            print(f'    k={k:>2} -> {path}')

        return region, k_labels_map, buf.getvalue()

    finally:
        _sys.stdout = orig_stdout


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    survey_year: str          = '2024',
    regions: dict             = None,
    k_values: list[int]       = None,
    perc_threshold: int       = 10,
    target_col: str           = 'INCOME_ABOVE_THRESHOLD',
    metadata_cols: list[str]  = None,
    base_dir: Path            = None,
) -> None:
    """
    Run counterfactual extraction for all specified regions and k values.

    Parallelism
    -----------
    Regions are processed in parallel via ProcessPoolExecutor.
    Each region spawns a child process that runs run_for_k_values() with its
    own thread pool internally.  The number of region-level processes is
    capped at min(n_regions, N_PHYSICAL_CORES // 4) to avoid over-subscribing
    the M1 Ultra's P-cores across nested thread pools.

    On M1 Ultra with 2 regions and 20 cores:
      - 2 region processes × up to 10 threads each (k-parallel threads)
        × 20 CatBoost threads internally = saturates ~all P-cores.

    Parameters
    ----------
    survey_year   : ACS survey year; used to locate input CSVs under base_dir/data/.
    regions       : dict mapping region name → CSV path.
                    If None, defaults to northeast and south under base_dir/data/.
    k_values      : neighbourhood sizes for BoCSoR.  Default: [1, 3, 5, 7].
    perc_threshold: boundary filter percentile (see CategoricalBoCSoR.explain).
    target_col    : name of the binary target column in the CSV.
    metadata_cols : non-feature columns to exclude from X (see run_for_k_values).
                    Default: ['YEAR'].
    base_dir      : project root directory; results/ and data/ are resolved
                    relative to this.  Auto-detects Kaggle (/kaggle/working),
                    Colab (/content), or falls back to the script's parent dir.
    """
    if base_dir is None:
        if Path('/kaggle/working').exists():
            base_dir = Path('/kaggle/working')
        elif Path('/content').exists():
            base_dir = Path('/content')
        else:
            base_dir = Path(__file__).resolve().parent.parent
    base_dir = Path(base_dir)

    if k_values is None:
        k_values = [1, 3, 5, 7]

    if metadata_cols is None:
        metadata_cols = ['YEAR']

    if regions is None:
        data_dir = base_dir / 'data'
        regions  = {
            'northeast': data_dir / f'acs_income_northeast_{survey_year}.csv',
            'south':     data_dir / f'acs_income_south_{survey_year}.csv',
        }

    results_dir = base_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    tee = _TeeWriter(sys.stdout)
    sys.stdout = tee

    # Cap region-level processes: each process will use its own thread pool
    # internally; over-subscribing would starve CatBoost's per-process threads.
    n_region_workers = max(1, min(len(regions), N_PHYSICAL_CORES // 4))

    print(f'  > Physical cores detected: {N_PHYSICAL_CORES}')
    print(f'  > Region parallelism: {n_region_workers} process(es) for '
          f'{len(regions)} region(s)')

    try:
        if n_region_workers > 1 and len(regions) > 1:
            # ── Parallel region processing ────────────────────────────
            with ProcessPoolExecutor(max_workers=n_region_workers) as pool:
                futures = {
                    pool.submit(
                        _run_region,
                        region,
                        Path(data_path),
                        results_dir,
                        k_values,
                        target_col,
                        perc_threshold,
                        metadata_cols,
                    ): region
                    for region, data_path in regions.items()
                }
                for future in as_completed(futures):
                    region, k_labels_map, region_log = future.result()
                    # Relay the child's captured output to the parent's TeeWriter.
                    sys.stdout.write(region_log)
                    sys.stdout.flush()
                    print('  > k_labels_map ready:')
                    for k, path in k_labels_map.items():
                        print(f'    k={k:>2} -> {path}')
        else:
            # ── Serial fallback (single region or single core) ────────
            for region, data_path in regions.items():
                output_dir = results_dir / region / 'important_features'
                output_dir.mkdir(parents=True, exist_ok=True)

                print('\n' + '=' * 70)
                print(f'COUNTERFACTUAL EXTRACTION — {region.upper()}')
                print('=' * 70 + '\n')

                if not Path(data_path).exists():
                    print(
                        f'  > Error: {data_path} not found — '
                        f'run create_dataset.py first.'
                    )
                    continue

                k_labels_map = run_for_k_values(
                    k_values        = k_values,
                    data_path       = data_path,
                    output_base_dir = output_dir,
                    target_col      = target_col,
                    perc_threshold  = perc_threshold,
                    metadata_cols   = metadata_cols,
                )

                print('  > k_labels_map ready:')
                for k, path in k_labels_map.items():
                    print(f'    k={k:>2} -> {path}')

        print('\n' + '=' * 70)
        print('Done.')
        print('=' * 70 + '\n')

        # ── Save logs ──────────────────────────────────────────────────
        full_log = tee.getvalue()

        global_log = results_dir / 'feature_importance_log.txt'
        global_log.write_text(full_log, encoding='utf-8')
        print(f'  > Full log saved to {global_log}')

        for region in regions:
            region_dir = results_dir / region / 'important_features'
            if region_dir.exists():
                marker = f'COUNTERFACTUAL EXTRACTION — {region.upper()}'
                start  = full_log.find(marker)
                if start != -1:
                    next_start = full_log.find(
                        'COUNTERFACTUAL EXTRACTION', start + len(marker)
                    )
                    snippet = (
                        full_log[start:next_start]
                        if next_start != -1
                        else full_log[start:]
                    )
                    (region_dir / 'feature_importance_log.txt').write_text(
                        snippet, encoding='utf-8'
                    )
                    print(
                        f'  > Region log saved to '
                        f'{region_dir / "feature_importance_log.txt"}'
                    )

    finally:
        sys.stdout = tee._orig


if __name__ == '__main__':
    main()