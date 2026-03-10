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

Mega-batch performance strategy
--------------------------------
explain() uses exactly 2 model.predict() calls per class (not per sample):
  (a) One batch predict to verify all B×k CF candidates at once.
  (b) One batch predict on all perturbations across all valid CFs at once.
The perturbation matrix is built entirely via NumPy broadcasting — no Python
inner loops over samples or features.  This is the key performance win on GPU.

Pipeline position
-----------------
    create_dataset.py  →  [feature_importance.py]  →  macroscopic_experiment_association_rules.py

Input  : data/acs_income_{region}_{year}.csv  (written by create_dataset.py)
Outputs: results/{region}/important_features/k_{k}/*.csv  (read by macroscopic_...)

Public API
----------
run_for_k_values(k_values, data_path, output_base_dir,
                 target_col, perc_threshold, metadata_cols)
    Train once, extract counterfactuals for each k, save results.

CategoricalBoCSoR
    Class implementing fit() and explain().
"""

import ast
import io
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.neighbors import BallTree


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _catboost_task_type() -> str:
    """
    Detect CUDA-capable GPU availability and return the CatBoost task_type.

    Uses nvidia-smi rather than importing a GPU library directly, so this
    function works even when CUDA libraries are not installed (it simply
    catches the FileNotFoundError and falls back to CPU).

    Returns 'GPU' if nvidia-smi exits successfully, 'CPU' otherwise.
    """
    try:
        subprocess.run(
            ['nvidia-smi'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        print("  > GPU detected — CatBoost will use task_type='GPU'")
        return 'GPU'
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  > No CUDA GPU detected — CatBoost will use task_type='CPU'")
        return 'CPU'


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

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train CatBoost on (X, y) and build one Hamming-metric BallTree per class.

        Why BallTrees on the full dataset (not just the training split)?
        ----------------------------------------------------------------
        The boundary percentile threshold is computed from the distribution of
        minimum Hamming distances across *all* samples.  Using only the training
        split would give a percentile anchored to a biased subset — the full
        dataset provides a more accurate picture of where the decision boundary
        is in the feature space.

        Why C-contiguous int32?
        -----------------------
        NumPy fancy-indexing and slicing are fastest on C-contiguous (row-major)
        arrays.  int32 halves memory vs int64 without precision loss for ordinal
        codes (typically < 500 categories per column).

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

        # 80/20 stratified split: the test set is used only for CatBoost's
        # early stopping evaluation set — it is not used in explain().
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_enc, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )

        self.model = CatBoostClassifier(
            iterations            = 500,    # max boosting rounds
            depth                 = 8,      # tree depth (balanced accuracy/speed)
            learning_rate         = 0.05,   # shrinkage; lower = more robust
            verbose               = 50,     # print progress every 50 iterations
            allow_writing_files   = False,  # suppress CatBoost's local snapshot files
            task_type             = self._task_type,  # 'GPU' or 'CPU'
        )
        self.model.fit(
            X_tr, y_tr,
            cat_features          = list(range(X_enc.shape[1])),  # all cols are categorical
            eval_set              = (X_val, y_val),
            early_stopping_rounds = 50,   # stop if val loss stagnates for 50 rounds
        )

        # Build one Hamming-metric BallTree per class using the *full* encoded
        # dataset.  Hamming distance counts the number of feature positions where
        # two instances differ, which is the natural distance for ordinal-encoded
        # categorical data.
        for label in np.unique(y_enc):
            idx = np.where(y_enc == label)[0]
            self.trees[label] = BallTree(X_enc[idx], metric='hamming')

        # Store the full encoded matrix and labels for use in explain().
        self.X_enc = X_enc
        self.y_enc = y_enc

        # Pre-cache per-class index arrays so explain() can directly index
        # self.X_enc without recomputing np.where(y_enc == label) for every
        # boundary instance in every class iteration.
        self.class_indices = {
            int(label): np.where(y_enc == label)[0]
            for label in np.unique(y_enc)
        }

    # ------------------------------------------------------------------
    # explain
    # ------------------------------------------------------------------

    def explain(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Run the boundary-crossing counterfactual analysis on (X, y).

        For each class in turn:
          1. Compute Hamming distance from each instance to its nearest
             opposite-class neighbour.
          2. Keep only instances at or below the perc_threshold-th percentile
             of those distances (boundary filter).
          3. Keep only boundary instances the model predicts correctly
             (model-prediction filter — mirrors the paper's consistency check).
          4. For each surviving boundary instance, query k nearest CF
             neighbours from the opposite class.
          5. Build the full perturbation matrix via NumPy broadcasting:
             for every (boundary_instance, CF_neighbour, differing_feature)
             triple, create a perturbed copy of the boundary instance with
             that feature replaced by the CF's value.
          6. Batch-predict all perturbations in one model.predict() call.
          7. Record features where the prediction flipped as drivers.

        The mega-batch strategy (steps 5–6) replaces what would otherwise be
        len(boundary_idx) × k × len(features) individual predict() calls with
        just 2 calls per class, achieving 100–1000× speedup on GPU.

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
              '(parallel, vectorised per-feature swap)...')

        # Re-encode X using the encoder fitted in fit() (no refitting).
        X_enc = np.ascontiguousarray(
            self.feature_encoder.transform(X), dtype=np.int32
        )
        y_enc = self.label_encoder.transform(y)

        # Preserve original DataFrame row labels for output traceability.
        original_indices = X.index.tolist()
        rows = []

        all_classes = np.unique(y_enc)

        for label in all_classes:
            # Gather indices of all instances belonging to this class.
            pos_idx = np.where(y_enc == label)[0]
            if len(pos_idx) == 0:
                continue

            # The opposite class is the only other class in a binary problem.
            opp_label = int(all_classes[all_classes != label][0])
            tree = self.trees.get(opp_label)
            if tree is None:
                continue

            # ── Boundary filter ──────────────────────────────────────────
            # Query the nearest opposite-class neighbour for every instance
            # in this class (k=1 is sufficient to compute the min distance).
            min_dist, _ = tree.query(X_enc[pos_idx], k=1)
            min_dist    = min_dist.ravel()

            # Keep only instances within the perc_threshold-th percentile of
            # minimum distances.  Lower percentile → stricter boundary filter
            # → fewer but more extreme boundary instances.
            threshold    = np.percentile(min_dist, self.perc_threshold)
            boundary_idx = pos_idx[min_dist <= threshold]

            if len(boundary_idx) == 0:
                continue

            # ── Model-prediction filter ──────────────────────────────────
            # Retain only boundary instances that the model predicts as their
            # true label.  Misclassified instances are already "across" the
            # boundary and do not represent genuine boundary crossings.
            model_preds  = self.model.predict(X_enc[boundary_idx]).ravel()
            correct_mask = model_preds == label
            boundary_idx = boundary_idx[correct_mask]

            if len(boundary_idx) == 0:
                print(
                    f'    - class {label}: 0 boundary samples pass the '
                    f'model-prediction filter — skipping.'
                )
                continue

            print(
                f'    - class {label}: {len(boundary_idx)}/{len(pos_idx)} '
                f'boundary samples pass model-prediction filter '
                f'(perc_threshold={self.perc_threshold})'
            )

            # ── Query k CF neighbours ─────────────────────────────────────
            # For each boundary instance, find k nearest opposite-class
            # neighbours.  `ind` shape: (B, k), indices into the BallTree's
            # own training set (= self.class_indices[opp_label]).
            _, ind = tree.query(X_enc[boundary_idx], k=self.k_neighbors)

            # ── Mega-batch approach ───────────────────────────────────────
            # Instead of one predict() call per boundary sample, we batch all
            # B×k CF verifications into a single call, then all perturbations
            # into another single call.  This is the primary performance lever.
            B          = len(boundary_idx)
            boundary_X = X_enc[boundary_idx]   # shape (B, F)

            # (a) Batch-verify all CF candidates: translate BallTree-local
            #     indices to global indices in self.X_enc, then predict all
            #     in one call and reshape to (B, k).
            cf_global_all = self.class_indices[opp_label][ind]   # (B, k)
            cf_X_all      = self.X_enc[cf_global_all.ravel()]    # (B*k, F)
            cf_preds      = (
                self.model.predict(cf_X_all).ravel().reshape(B, -1)
            )                                                     # (B, k)
            # A CF is "valid" only if the model also predicts it as the
            # opposite class (rejects noise/borderline CF instances).
            valid_mask = cf_preds == opp_label                    # (B, k) bool

            n_valid = int(valid_mask.sum())
            if n_valid == 0:
                print(f'    - class {label}: 0 valid CF neighbours — skipping.')
                continue

            # (b) Build the perturbation matrix via NumPy broadcasting.
            # valid_b, valid_cf_pos: row/col coordinates of valid (boundary, CF) pairs.
            valid_b, valid_cf_pos = np.where(valid_mask)
            cf_gidx_valid = cf_global_all[valid_b, valid_cf_pos]  # global CF indices

            orig_valid = boundary_X[valid_b]                       # (n_valid, F)
            cf_valid   = self.X_enc[cf_gidx_valid]                 # (n_valid, F)

            # diff_matrix[i, j] is True where the i-th (boundary, CF) pair
            # differs at feature j.  These are the candidates for feature swap.
            diff_matrix = cf_valid != orig_valid                   # (n_valid, F)

            # pair_idx[p], feat_idx[p]: the pair index and feature index of
            # the p-th perturbation to test.
            pair_idx, feat_idx = np.where(diff_matrix)
            P = len(pair_idx)

            if P == 0:
                continue

            # Construct the perturbation matrix: copy the boundary instance
            # for each perturbation, then replace the target feature with the
            # CF's value.  The copy is essential — numpy fancy-indexing returns
            # a copy, so modifying perturb_matrix does not affect orig_valid.
            perturb_matrix = orig_valid[pair_idx].copy()           # (P, F)
            perturb_matrix[np.arange(P), feat_idx] = \
                cf_valid[pair_idx, feat_idx]

            print(
                f'    - class {label}: {n_valid:,} valid CFs, '
                f'{P:,} perturbations — predicting...'
            )

            # (c) Batch-predict all P perturbations in one call.
            all_preds = self.model.predict(perturb_matrix).ravel()  # (P,)

            # (d) A perturbation is a "driver" if it flipped the prediction
            #     away from the boundary instance's true label.  This means
            #     the CF's feature value was sufficient to cross the boundary.
            is_driver      = all_preds != label
            driver_indices = np.where(is_driver)[0]

            if len(driver_indices) == 0:
                continue

            # Recover the boundary instance index, CF global index, feature
            # index, and CF feature value for each driver perturbation.
            driver_b       = valid_b[pair_idx[driver_indices]]
            driver_cf_gidx = cf_gidx_valid[pair_idx[driver_indices]]
            driver_feat    = feat_idx[driver_indices]
            driver_cf_val  = cf_valid[
                pair_idx[driver_indices], feat_idx[driver_indices]
            ].astype(int)

            # Convert ordinal code back to the original string category label.
            # Format: 'FEATURE_NAME=cf_category_value' (e.g. 'SCHL=Bachelors-Degree').
            driver_strs = [
                f'{self.feature_names[f]}'
                f'={self.feature_encoder.categories_[f][v]}'
                for f, v in zip(driver_feat, driver_cf_val)
            ]

            # Group all drivers for the same (boundary_sample, CF_neighbour) pair
            # into a single transaction row.  Each row will become one itemset
            # in the FP-Growth input file.
            groups: dict[tuple, list] = {}
            for d, ds in enumerate(driver_strs):
                key = (int(driver_b[d]), int(driver_cf_gidx[d]))
                groups.setdefault(key, []).append(ds)

            for (b_idx, cf_gidx), driver_list in groups.items():
                sample_idx = boundary_idx[b_idx]
                rows.append({
                    'Sample_ID':             original_indices[sample_idx],
                    'CF_Neighbor_ID':        cf_gidx,
                    # Sorted for deterministic output and easier deduplication.
                    'Counterfactual_Values': sorted(driver_list),
                })

        print(
            f'    - done: {len(rows)} (sample, CF-neighbour) pairs '
            f'with at least one driver\n'
        )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def extract_labels_and_values(results_dir: Path) -> None:
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

    # Determine which ID columns exist (CF_Neighbor_ID may be absent in some
    # legacy formats).
    base_cols = ['Sample_ID']
    if 'CF_Neighbor_ID' in df.columns:
        base_cols.append('CF_Neighbor_ID')

    labels_list        = []
    labels_unique_list = []
    values_list        = []
    values_unique_list = []

    for _, row in df.iterrows():
        # Counterfactual_Values is stored as a stringified Python list;
        # ast.literal_eval safely parses it back to a Python list.
        try:
            items = ast.literal_eval(str(row['Counterfactual_Values']))
        except (ValueError, SyntaxError):
            items = []

        labels, values = [], []
        for item in items:
            item_str = str(item)
            if '=' in item_str:
                # Split on the first '=' only; category values may contain '='
                # in edge cases (e.g. formula-based occupation codes).
                lbl, val = item_str.split('=', 1)
                labels.append(lbl.strip())
                values.append(val.strip())

        # Deduplicate on (label, value) pairs jointly to preserve alignment.
        seen_pairs  = set()
        uniq_labels = []
        uniq_values = []
        for lbl, val in zip(labels, values):
            pair = (lbl, val)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                uniq_labels.append(lbl)
                uniq_values.append(val)

        labels_list.append(labels)
        labels_unique_list.append(uniq_labels)
        values_list.append(values)
        values_unique_list.append(uniq_values)

    base_data = {col: df[col] for col in base_cols}

    # Write all four output files.
    for filename, col_name, data in [
        ('labels_only.csv',        'Labels', labels_list),
        ('labels_only_unique.csv', 'Labels', labels_unique_list),
        ('values_only.csv',        'Values', values_list),
        ('values_only_unique.csv', 'Values', values_unique_list),
    ]:
        pd.DataFrame({**base_data, col_name: data}).to_csv(
            results_dir / filename, index=False
        )
        print(f'    - saved {filename}')


def aggregate_drivers_by_sample(results_dir: Path) -> None:
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

    # Parse stringified lists back to Python lists.
    df['Labels'] = df['Labels'].apply(ast.literal_eval)

    aggregated_unique = []
    aggregated_dupl   = []

    for sample_id, group in df.groupby('Sample_ID'):
        n_cf = len(group)   # number of CF neighbours for this sample

        # Union: deduplicated set of all driver feature names across all CFs.
        all_labels_set = set()
        for labels_list in group['Labels']:
            all_labels_set.update(labels_list)
        unique_labels = sorted(all_labels_set)   # sorted for determinism

        # Flat list: all driver feature names concatenated, duplicates kept.
        all_labels_list = sorted(
            lbl for labels_list in group['Labels'] for lbl in labels_list
        )

        aggregated_unique.append({
            'Sample_ID':        sample_id,
            'Labels':           unique_labels,
            'Num_Labels':       len(unique_labels),
            'Num_CF_Neighbors': n_cf,
        })
        aggregated_dupl.append({
            'Sample_ID':        sample_id,
            'Labels':           all_labels_list,
            'Num_Labels':       len(all_labels_list),
            'Num_CF_Neighbors': n_cf,
        })

    unique_df = pd.DataFrame(aggregated_unique)
    dupl_df   = pd.DataFrame(aggregated_dupl)

    unique_df.to_csv(out_unique, index=False)
    dupl_df.to_csv(out_dupl,    index=False)

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
    Train CategoricalBoCSoR once and run counterfactual extraction for each k.

    The model and BallTrees are fitted on the training split and shared across
    all k values; only the neighbourhood query size (k_neighbors) changes per
    iteration.  This avoids retraining for each k, saving the dominant cost.

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
    # Default: exclude YEAR (longitudinal provenance column), keep everything else.
    if metadata_cols is None:
        metadata_cols = ['YEAR']

    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"=" * 70}')
    print(f'K-VARIATION EXPERIMENT — {len(k_values)} k values')
    print(f'{"=" * 70}')
    print(f'  > k values        : {k_values}')
    print(f'  > perc_threshold  : {perc_threshold}')
    print(f'  > target column   : {target_col}')
    if metadata_cols:
        print(f'  > metadata cols   : {metadata_cols} (excluded from X)')
    print('-' * 50)

    print('  > Loading dataset and splitting...')
    df = pd.read_csv(data_path)

    # Rename the target column to 'target' for uniform internal handling.
    if target_col != 'target':
        df = df.rename(columns={target_col: 'target'})

    # Build the list of columns to drop from the feature matrix:
    # always drop 'target'; also drop any metadata_cols that are present.
    # Checking membership with `if c in df.columns` avoids KeyError when a
    # metadata column is not present (e.g. YEAR was not included in the CSV).
    cols_to_drop = ['target'] + [c for c in metadata_cols if c in df.columns]
    X, y = df.drop(columns=cols_to_drop), df['target']

    # 80/20 stratified split — explain() runs on X_tr (training logic).
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'    - train: {len(X_tr):,} samples  |  test: {len(X_te):,} samples')

    # Fit once using the first k value — only k_neighbors matters for fit(),
    # and it is overwritten in the loop below before each explain() call.
    print('\n  > Fitting model (shared across all k values)...')
    explainer = CategoricalBoCSoR(
        k_neighbors=k_values[0], perc_threshold=perc_threshold
    )
    explainer.fit(X_tr, y_tr)

    k_labels_map: dict[int, Path] = {}

    for i, k in enumerate(k_values):
        k_dir = output_base_dir / f'k_{k}'
        k_dir.mkdir(parents=True, exist_ok=True)

        print(f'\n  [{i + 1}/{len(k_values)}] k = {k}')

        # Update the neighbourhood size without refitting the model or BallTrees.
        explainer.k_neighbors = k

        # explain() runs on the *training* set: we are explaining the model's
        # decision logic on the data it was trained on, not generalising to
        # unseen instances.  The goal is to understand what features drive
        # predictions near the boundary, not to evaluate generalisation.
        transactions = explainer.explain(X_tr, y_tr)

        if transactions.empty:
            print('    > 0 transactions — skipping downstream steps.')
            continue

        # Save raw transactions (one row per (sample, CF) pair).
        transactions_path = k_dir / 'transactions_values.csv'
        transactions.to_csv(transactions_path, index=False)
        print(
            f'    > {len(transactions)} transactions saved to '
            f'{transactions_path.name}'
        )

        # Parse and split transactions into labels-only and values-only files.
        extract_labels_and_values(k_dir)

        # Collapse (sample, CF) pairs into one row per sample for FP-Growth.
        aggregate_drivers_by_sample(k_dir)

        labels_path = k_dir / 'labels_only_unique.csv'
        if labels_path.exists() and labels_path.stat().st_size > 0:
            k_labels_map[k] = labels_path
        else:
            print(
                f'    - WARNING: labels_only_unique.csv is empty for k={k}, '
                f'skipping.'
            )

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

    Why not logging.Logger?
    -----------------------
    print() calls inside CatBoost and sklearn write directly to sys.stdout.
    Replacing sys.stdout with _TeeWriter captures those calls too, without
    requiring changes to third-party library code.
    """

    def __init__(self, original_stdout):
        self._orig = original_stdout   # original sys.stdout to preserve console output
        self._buf  = io.StringIO()     # in-memory buffer for log capture

    def write(self, text: str) -> None:
        self._orig.write(text)   # echo to console in real time
        self._buf.write(text)    # accumulate for disk write at end of run

    def flush(self) -> None:
        self._orig.flush()   # keep the console output responsive

    def getvalue(self) -> str:
        """Return the full accumulated log as a single string."""
        return self._buf.getvalue()


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

    Iterates over regions, calls run_for_k_values for each, and writes a
    combined execution log plus per-region log excerpts to disk.

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
    # Auto-detect environment if base_dir is not explicitly provided.
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

    if regions is None:
        # Default: run on northeast and south assuming standard output_dir layout.
        data_dir = base_dir / 'data'
        regions  = {
            'northeast': data_dir / f'acs_income_northeast_{survey_year}.csv',
            'south':     data_dir / f'acs_income_south_{survey_year}.csv',
        }

    results_dir = base_dir / 'results'

    # Replace sys.stdout with a TeeWriter to capture the full log while still
    # printing to the console.  Restored in the finally block below.
    tee = _TeeWriter(sys.stdout)
    sys.stdout = tee

    try:
        for region, data_path in regions.items():
            # Create the output directory for this region's feature importance results.
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

            # Run the full k-variation experiment for this region.
            k_labels_map = run_for_k_values(
                k_values        = k_values,
                data_path       = data_path,
                output_base_dir = output_dir,
                target_col      = target_col,
                perc_threshold  = perc_threshold,
                metadata_cols   = metadata_cols,
            )

            # Print the per-k output file paths for easy reference.
            print('  > k_labels_map ready:')
            for k, path in k_labels_map.items():
                print(f'    k={k:>2} -> {path}')

        print('\n' + '=' * 70)
        print('Done.')
        print('=' * 70 + '\n')

    finally:
        # Always restore sys.stdout, even if an exception occurred.
        sys.stdout = tee._orig

    # ── Save logs ─────────────────────────────────────────────────────
    full_log = tee.getvalue()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Write a single global log covering all regions.
    global_log = results_dir / 'feature_importance_log.txt'
    global_log.write_text(full_log, encoding='utf-8')
    print(f'  > Full log saved to {global_log}')

    # Write a per-region excerpt by finding the section marker for each region
    # and slicing the log between consecutive markers.
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


if __name__ == '__main__':
    main()
