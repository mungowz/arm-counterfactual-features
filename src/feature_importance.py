import ast
import pandas as pd
import numpy as np
import subprocess
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.neighbors import BallTree
from pathlib import Path
from joblib import Parallel, delayed


# ---------------------------------------------------------------------------
# Utility: detect GPU availability for CatBoost
# ---------------------------------------------------------------------------

def _catboost_task_type():
    """
    Detect whether a CUDA-capable GPU is available and return the appropriate
    CatBoost task_type string ('GPU' or 'CPU').
    """
    try:
        subprocess.run(
            ['nvidia-smi'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        print("  > GPU detected — CatBoost will use task_type='GPU'")
        return 'GPU'
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  > No CUDA GPU detected — CatBoost will use task_type='CPU'")
        return 'CPU'


# ---------------------------------------------------------------------------
# Core class: CategoricalBoCSoR
# ---------------------------------------------------------------------------

class CategoricalBoCSoR:
    """
    Categorical adaptation of BoCSoR (Boundary Crossing Solo Ratio).

    Differences from the original paper and code (all deliberate):

    1. Hamming distance instead of Euclidean — appropriate for categorical
       features where no ordinal relationship between values exists.

    2. No midpoint interpolation — midpoints between two categorical instances
       are not meaningful. Following the authors' suggestion, only real
       instances from the opposite class are used as counterfactuals. To
       preserve the guarantee that the selected CF is predicted as the CF
       class by the model, every candidate CF is verified against the model
       before being used (see _process_single_sample).

    3. Inverted swap direction — instead of injecting the original feature
       value into the CF (as in the paper), we inject the CF feature value
       into the original instance. The check is symmetric: if injecting the
       CF value into the original causes the prediction to switch to the CF
       class, that feature (with that specific CF value) is a driver. This
       direction is more informative for association-rule mining because the
       stored itemset contains the actual CF values that trigger the switch,
       not just the feature names.

    4. All k neighbours considered — instead of selecting only the single
       closest counterfactual, all k neighbours are retained. Each produces
       a separate transaction row, enabling FP-Growth to detect patterns
       across different CF contexts for the same boundary instance.

    Performance optimisations (M2-oriented):
    - model.predict calls reduced from O(1 + 2k) to O(3) per boundary sample:
        * 1 call for the original instance
        * 1 batched call for all k CF verifications at once
        * 1 batched call for ALL perturbations across ALL valid CFs at once
    - Perturbation matrix built via NumPy broadcasting (no Python loops),
      exploiting the M2's NEON SIMD units for the array operations
    - class_global_indices pre-cached in fit() so workers never recompute
      np.where(y_enc == label) on every call
    - X_enc stored as C-contiguous int32 array for optimal cache behaviour
    - joblib backend set to 'loky' (true multiprocessing) to bypass the GIL
      and spread load across all M2 performance + efficiency cores
    """

    def __init__(self, k_neighbors=10, perc_threshold=10):
        self.k_neighbors     = k_neighbors
        self.perc_threshold  = perc_threshold
        self.model           = None
        self.feature_encoder = OrdinalEncoder(dtype=int)
        self.label_encoder   = LabelEncoder()
        self.trees           = {}   # one BallTree (Hamming) per class
        self._task_type      = _catboost_task_type()

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Train CatBoost on (X, y) and build one BallTree per class.

        Parameters
        ----------
        X : pd.DataFrame  — feature matrix (categorical and/or integer columns)
        y : pd.Series     — binary target (0/1)
        """
        print("  > Training CatBoost and building BallTrees...")
        self.feature_names = X.columns.tolist()

        y_enc = self.label_encoder.fit_transform(y)

        # Store as C-contiguous int32: faster numpy slicing and better cache
        # locality on M2 when building perturbation matrices in workers
        X_enc = np.ascontiguousarray(
            self.feature_encoder.fit_transform(X), dtype=np.int32
        )

        # Internal train/validation split for CatBoost early-stopping
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_enc, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )

        self.model = CatBoostClassifier(
            iterations=100, depth=6, learning_rate=0.1,
            verbose=0, allow_writing_files=False,
            task_type=self._task_type
        )
        self.model.fit(
            X_tr, y_tr,
            cat_features=list(range(X_enc.shape[1])),
            eval_set=(X_val, y_val)
        )

        # BallTrees built on ALL encoded training data so the boundary-
        # filtering percentile reflects the full training distribution.
        for label in np.unique(y_enc):
            idx = np.where(y_enc == label)[0]
            self.trees[label] = BallTree(X_enc[idx], metric='hamming')

        self.X_enc = X_enc
        self.y_enc = y_enc

        # OPTIMISATION: pre-cache per-class global indices so workers never
        # recompute np.where(y_enc == label) on every parallel call.
        self.class_indices = {
            int(label): np.where(y_enc == label)[0]
            for label in np.unique(y_enc)
        }

    # ------------------------------------------------------------------
    # _process_single_sample  (worker called by joblib)
    # ------------------------------------------------------------------

    def _process_single_sample(self, sample_idx, orig_enc_sample, ind_row,
                                opp_label):
        """
        Process one boundary instance against its k CF neighbours.

        Performance profile (per call):
            model.predict calls : 3  (was 1 + 2k in the naive version)
            numpy ops           : fully vectorised, no Python inner loops

        Steps
        -----
        1. Predict orig_pred with a single call.
        2. Retrieve all k CF candidates; verify all of them in ONE batched
           predict call; discard those not predicted as opp_label.
        3. Build the full perturbation matrix in one NumPy operation:
              shape = (total_diff_positions_across_valid_CFs, n_features)
           using np.tile + advanced indexing — no Python loop over features.
        4. Predict all perturbations in ONE batched call.
        5. Group drivers by CF neighbour.

        Returns
        -------
        sample_idx : int
        per_neighbor_results : list of (cf_global_idx, sorted_driver_list)
        """
        orig      = orig_enc_sample          # C-contiguous int32, already copied
        orig_pred = self.model.predict([orig])[0]

        # Map BallTree-local indices to global row indices (pre-cached)
        global_ind = self.class_indices[int(opp_label)][ind_row]
        neighbors  = self.X_enc[global_ind]  # shape (k, n_features)

        # ------------------------------------------------------------------
        # OPTIMISATION 1 — batch verify all k CF candidates in one call
        # ------------------------------------------------------------------
        cf_preds    = self.model.predict(neighbors).ravel()
        valid_mask  = (cf_preds == opp_label)
        valid_global = global_ind[valid_mask]
        valid_neigh  = neighbors[valid_mask]   # shape (n_valid, n_features)

        if len(valid_neigh) == 0:
            return sample_idx, []

        # ------------------------------------------------------------------
        # OPTIMISATION 2 — vectorised perturbation matrix construction
        #
        # diff_matrix[i, j] is True where valid CF i differs from orig at j.
        # np.where returns two aligned arrays:
        #   cf_row[p]  = which valid CF the p-th perturbation belongs to
        #   feat_col[p] = which feature is swapped in the p-th perturbation
        # ------------------------------------------------------------------
        diff_matrix            = valid_neigh != orig          # (n_valid, n_feat)
        cf_row, feat_col       = np.where(diff_matrix)        # both shape (P,)
        n_perturbations        = len(cf_row)

        if n_perturbations == 0:
            return sample_idx, []

        # Build (P, n_features) matrix: start from P copies of orig, then
        # inject each CF value at the corresponding feature position.
        # np.tile + advanced indexing: no Python loop, pure NumPy.
        perturb_matrix = np.tile(orig, (n_perturbations, 1))
        perturb_matrix[np.arange(n_perturbations), feat_col] = \
            valid_neigh[cf_row, feat_col]

        # ------------------------------------------------------------------
        # OPTIMISATION 3 — one single predict call for ALL perturbations
        # ------------------------------------------------------------------
        all_preds = self.model.predict(perturb_matrix).ravel()

        # ------------------------------------------------------------------
        # Group drivers by CF neighbour
        # ------------------------------------------------------------------
        # Build driver strings for all perturbations (vectorised decode)
        driver_strings = [
            f"{self.feature_names[feat_col[p]]}"
            f"={self.feature_encoder.categories_[feat_col[p]][valid_neigh[cf_row[p], feat_col[p]]]}"
            for p in range(n_perturbations)
        ]

        # Collect drivers per (valid) CF neighbour
        cf_drivers: dict[int, list] = {}
        for p, (pred, driver_str) in enumerate(zip(all_preds, driver_strings)):
            if pred != orig_pred:
                gidx = int(valid_global[cf_row[p]])
                cf_drivers.setdefault(gidx, []).append(driver_str)

        per_neighbor_results = [
            (gidx, sorted(drivers))
            for gidx, drivers in cf_drivers.items()
        ]

        return sample_idx, per_neighbor_results

    # ------------------------------------------------------------------
    # explain
    # ------------------------------------------------------------------

    def explain(self, X, y):
        """
        Run the boundary-crossing analysis on (X, y) — training data only.

        For each class label:
          1. Compute Hamming distance from each instance to its nearest
             opposite-class neighbour.
          2. Keep only instances whose distance is <= the perc_threshold-th
             percentile (boundary filter).
          3. Keep only boundary instances the model predicts correctly
             (paper consistency: mirrors the original explain_decision_boundary
             check before calling explain_sample).
          4. For each surviving boundary instance, query k nearest CF
             neighbours and run the vectorised per-feature swap in parallel
             across all M2 cores (loky backend = true multiprocessing).

        Returns
        -------
        pd.DataFrame with columns:
            Sample_ID             — original DataFrame index
            CF_Neighbor_ID        — positional index in self.X_enc
            Counterfactual_Values — sorted list of "FEATURE=cf_value" strings
        """
        print("  > Extracting counterfactuals "
              "(parallel + vectorised per-feature swap)...")

        X_enc = np.ascontiguousarray(
            self.feature_encoder.transform(X), dtype=np.int32
        )
        y_enc = self.label_encoder.transform(y)

        original_indices = X.index.tolist()
        rows = []

        all_classes = np.unique(y_enc)

        for label in all_classes:
            pos_idx = np.where(y_enc == label)[0]
            if len(pos_idx) == 0:
                continue

            opp_label = int(all_classes[all_classes != label][0])
            tree = self.trees.get(opp_label)
            if tree is None:
                continue

            # --------------------------------------------------------------
            # Step 1 — Boundary filter (Hamming percentile)
            # --------------------------------------------------------------
            min_dist_to_opp, _ = tree.query(X_enc[pos_idx], k=1)
            min_dist_to_opp    = min_dist_to_opp.ravel()
            dist_threshold     = np.percentile(min_dist_to_opp,
                                               self.perc_threshold)

            boundary_mask      = min_dist_to_opp <= dist_threshold
            boundary_pos_idx   = pos_idx[boundary_mask]

            if len(boundary_pos_idx) == 0:
                continue

            # --------------------------------------------------------------
            # Step 2 — Model-prediction filter
            # Keep only boundary instances the model predicts as `label`.
            # Mirrors the original authors' check in explain_decision_boundary:
            #   if self.model.predict(sample.values) == self.original_class
            # --------------------------------------------------------------
            model_preds      = self.model.predict(
                X_enc[boundary_pos_idx]
            ).ravel()
            correct_mask     = (model_preds == label)
            boundary_pos_idx = boundary_pos_idx[correct_mask]

            if len(boundary_pos_idx) == 0:
                print(f"    - class {label}: 0 boundary samples pass the "
                      f"model-prediction filter — skipping.")
                continue

            print(f"    - class {label}: {len(boundary_pos_idx)}"
                  f"/{len(pos_idx)} boundary samples pass model-prediction "
                  f"filter (perc_threshold={self.perc_threshold})")

            # --------------------------------------------------------------
            # Step 3 — k nearest CF neighbours for each boundary instance
            # --------------------------------------------------------------
            _, ind = tree.query(X_enc[boundary_pos_idx], k=self.k_neighbors)

            # --------------------------------------------------------------
            # Step 4 — Parallel dispatch across all M2 cores
            # 'loky' spawns real processes → true parallelism, no GIL.
            # CatBoost supports pickle so workers can deserialise the model.
            # --------------------------------------------------------------
            parallel_results = Parallel(n_jobs=-1, backend="loky")(
                delayed(self._process_single_sample)(
                    sample_idx,
                    X_enc[sample_idx].copy(),   # copy → worker-safe
                    ind[i],
                    opp_label
                )
                for i, sample_idx in enumerate(boundary_pos_idx)
            )

            for sample_idx, per_neighbor_results in parallel_results:
                for cf_global_idx, found_drivers in per_neighbor_results:
                    rows.append({
                        'Sample_ID':             original_indices[sample_idx],
                        'CF_Neighbor_ID':        cf_global_idx,
                        'Counterfactual_Values': found_drivers
                    })

        print(f"    - done: {len(rows)} (sample, CF-neighbour) pairs "
              f"with at least one driver\n")
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def extract_labels_and_values(results_dir):
    """
    Parse transactions_values.csv and produce four additional CSVs:

        labels_only.csv          — per (sample, CF): all driver feature names
        labels_only_unique.csv   — per (sample, CF): unique driver feature names
        values_only.csv          — per (sample, CF): all CF category values
        values_only_unique.csv   — per (sample, CF): unique CF category values

    Deduplication is performed on (label, value) pairs jointly so that
    labels_only_unique and values_only_unique remain positionally aligned.
    """
    print("  > Extracting labels and values from counterfactual drivers...")
    input_file = results_dir / "transactions_values.csv"

    if not input_file.exists():
        print("    - WARNING: transactions file not found, skipping.")
        return
    try:
        df = pd.read_csv(input_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"    - WARNING: could not read transactions file ({e}), "
              f"skipping.")
        return
    if df.empty:
        print("    - WARNING: no transactions found, skipping.")
        return

    base_col_names = ['Sample_ID']
    if 'CF_Neighbor_ID' in df.columns:
        base_col_names.append('CF_Neighbor_ID')

    labels_list        = []
    labels_unique_list = []
    values_list        = []
    values_unique_list = []

    for _, row in df.iterrows():
        try:
            items = ast.literal_eval(str(row['Counterfactual_Values']))
        except (ValueError, SyntaxError):
            items = []

        labels, values = [], []
        for item in items:
            item_str = str(item)
            if '=' in item_str:
                lbl, val = item_str.split('=', 1)
                labels.append(lbl.strip())
                values.append(val.strip())

        # Deduplicate on (label, value) pairs jointly — keeps the two lists
        # positionally aligned (labels_only_unique[i] ↔ values_only_unique[i])
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

    base_data = {col: df[col] for col in base_col_names}

    outputs = [
        ("labels_only.csv",        "Labels", labels_list),
        ("labels_only_unique.csv", "Labels", labels_unique_list),
        ("values_only.csv",        "Values", values_list),
        ("values_only_unique.csv", "Values", values_unique_list),
    ]
    for filename, col_name, data in outputs:
        path = results_dir / filename
        pd.DataFrame({**base_data, col_name: data}).to_csv(path, index=False)
        print(f"    - saved {filename}")


def aggregate_drivers_by_sample(results_dir):
    """
    Collapse all (sample, CF-neighbour) rows into one transaction per sample,
    producing two aggregated CSV files for ARM on feature labels:

    aggregated_labels_by_sample.csv
        One row per sample. 'Labels' contains the UNION of all unique driver
        feature names across every CF neighbour of that sample — duplicates
        removed. This is the recommended input for FP-Growth: each row is one
        transaction (itemset) of feature names.

    aggregated_labels_duplicates_by_sample.csv
        One row per sample. 'Labels' contains ALL driver feature names
        including duplicates (i.e. if a feature appears as a driver for
        multiple CF neighbours of the same sample, it is listed multiple
        times). Useful for weighted ARM or frequency analysis.

    Both files share the same columns:
        Sample_ID        — original row index in the dataset
        Labels           — list of feature name drivers (see above)
        Num_Labels       — cardinality of the label list
        Num_CF_Neighbors — number of CF neighbours that contributed drivers
    """
    print("  > Aggregating labels by sample...")

    # Read from labels_only.csv (with duplicates across CF neighbours) so we
    # can build both the unique and duplicate aggregations from a single source
    labels_path = results_dir / "labels_only.csv"
    out_unique   = results_dir / "aggregated_labels_by_sample.csv"
    out_dupl     = results_dir / "aggregated_labels_duplicates_by_sample.csv"

    if not labels_path.exists():
        print(f"    - WARNING: {labels_path.name} not found, skipping.")
        return
    try:
        df = pd.read_csv(labels_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"    - WARNING: could not read labels file ({e}), skipping.")
        return
    if df.empty:
        print("    - WARNING: labels file is empty, skipping.")
        return

    df['Labels'] = df['Labels'].apply(ast.literal_eval)

    aggregated_unique = []
    aggregated_dupl   = []

    for sample_id, group in df.groupby('Sample_ID'):
        n_cf_neighbors = len(group)

        # --- Unique: union of all driver labels, duplicates removed ----------
        all_labels_set = set()
        for labels_list in group['Labels']:
            all_labels_set.update(labels_list)
        unique_labels = sorted(all_labels_set)

        # --- With duplicates: concatenate all label lists across CF neighbours
        # preserving multiplicity (a feature that drives the switch for 3 CFs
        # of the same sample appears 3 times)
        all_labels_list = []
        for labels_list in group['Labels']:
            all_labels_list.extend(labels_list)
        # Sort for reproducibility while keeping duplicates
        all_labels_list_sorted = sorted(all_labels_list)

        aggregated_unique.append({
            'Sample_ID':        sample_id,
            'Labels':           unique_labels,
            'Num_Labels':       len(unique_labels),
            'Num_CF_Neighbors': n_cf_neighbors
        })
        aggregated_dupl.append({
            'Sample_ID':        sample_id,
            'Labels':           all_labels_list_sorted,
            'Num_Labels':       len(all_labels_list_sorted),
            'Num_CF_Neighbors': n_cf_neighbors
        })

    unique_df = pd.DataFrame(aggregated_unique)
    dupl_df   = pd.DataFrame(aggregated_dupl)

    unique_df.to_csv(out_unique, index=False)
    dupl_df.to_csv(out_dupl,   index=False)

    n_samples = len(unique_df)
    print(f"    - {len(df)} (sample, CF) pairs collapsed into "
          f"{n_samples} unique samples")

    print(f"    [unique labels per sample]")
    for n_labels, count in (unique_df['Num_Labels']
                             .value_counts().sort_index().items()):
        pct = count / n_samples * 100
        print(f"      {n_labels} label(s): {count} samples ({pct:.1f}%)")
    print(f"    - saved to {out_unique.name}")

    print(f"    [labels with duplicates per sample]")
    for n_labels, count in (dupl_df['Num_Labels']
                             .value_counts().sort_index().items()):
        pct = count / n_samples * 100
        print(f"      {n_labels} label(s): {count} samples ({pct:.1f}%)")
    print(f"    - saved to {out_dupl.name}")


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_for_k_values(k_values, data_path, output_base_dir,
                     target_col='INCOME_ABOVE_THRESHOLD', perc_threshold=10):
    """
    Run CategoricalBoCSoR for each value of k in k_values.

    The model and BallTrees are trained once on the training split and reused
    across all k values; only the neighbourhood query size changes per run.

    Parameters
    ----------
    k_values        : list[int] — neighbourhood sizes to test
    data_path       : Path      — path to the input CSV file
    output_base_dir : Path      — root directory for result sub-folders
    target_col      : str       — name of the target column in the CSV
    perc_threshold  : int       — percentile threshold for boundary filtering

    Returns
    -------
    dict mapping k -> Path of the corresponding labels_only_unique.csv
    """
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"K-VARIATION EXPERIMENT — {len(k_values)} values of k")
    print(f"{'='*70}")
    print(f"  > k values       : {k_values}")
    print(f"  > perc_threshold : {perc_threshold}")
    print(f"  > target column  : {target_col}")
    print("-" * 50)

    print("  > Loading dataset and splitting...")
    df = pd.read_csv(data_path)

    if target_col != 'target':
        df = df.rename(columns={target_col: 'target'})

    X, y = df.drop(columns=['target']), df['target']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    - train: {len(X_tr)} samples  |  test: {len(X_te)} samples")

    # Fit once — model and BallTrees shared across all k iterations
    print(f"\n  > Fitting model (shared across all k values)...")
    explainer = CategoricalBoCSoR(
        k_neighbors=k_values[0], perc_threshold=perc_threshold
    )
    explainer.fit(X_tr, y_tr)

    k_labels_map = {}

    for i, k in enumerate(k_values):
        k_dir = output_base_dir / f"k_{k}"
        k_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{i+1}/{len(k_values)}] k = {k}")
        explainer.k_neighbors = k

        # explain() is called on training data only — we explain the
        # classifier's logic, not its generalisation on unseen instances
        transactions = explainer.explain(X_tr, y_tr)

        if transactions.empty:
            print("    > 0 transactions — skipping CSV write and downstream "
                  "steps.")
            continue

        transactions_path = k_dir / "transactions_values.csv"
        transactions.to_csv(transactions_path, index=False)
        print(f"    > {len(transactions)} transactions saved to "
              f"{transactions_path.name}")

        extract_labels_and_values(k_dir)
        aggregate_drivers_by_sample(k_dir)

        labels_path = k_dir / "labels_only_unique.csv"
        if labels_path.exists() and labels_path.stat().st_size > 0:
            k_labels_map[k] = labels_path
        else:
            print(f"    - WARNING: labels_only_unique.csv empty for k={k}, "
                  f"skipping.")

    print(f"\n{'='*70}")
    print("  > All k values done.")
    print(f"{'='*70}\n")

    return k_labels_map


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if Path("/content").exists():
        base_dir = Path("/content")       # Google Colab
    else:
        base_dir = Path(__file__).resolve().parent.parent   # local

    data_dir    = base_dir / "data"
    results_dir = base_dir / "results"

    regions = {
        'northeast': data_dir / "acs_income_northeast_2024.csv",
        'south':     data_dir / "acs_income_south_2024.csv",
    }

    k_values       = [1, 3, 5, 7]
    perc_threshold = 10
    target_col     = 'INCOME_ABOVE_THRESHOLD'

    for region, data_path in regions.items():
        output_dir = results_dir / region / "important_features"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print(f"COUNTERFACTUAL EXTRACTION — {region.upper()}")
        print("=" * 70 + "\n")

        if not data_path.exists():
            print(f"  > Error: {data_path.name} not found — "
                  f"run create_dataset.py first.")
            continue

        k_labels_map = run_for_k_values(
            k_values, data_path, output_dir,
            target_col=target_col,
            perc_threshold=perc_threshold
        )

        print("  > k_labels_map ready:")
        for k, path in k_labels_map.items():
            print(f"    k={k:>2} -> {path}")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70 + "\n")