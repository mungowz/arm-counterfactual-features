"""
feature_importance.py
=====================
Identifies boundary-crossing feature drivers via a categorical adaptation of
BoCSoR (Boundary Crossing Solo Ratio) built on top of CatBoost and BallTree
nearest-neighbour search.

The algorithm trains a CatBoost classifier, identifies instances near the
decision boundary (Hamming-distance percentile filter), queries k opposite-
class neighbours for each boundary instance, and records which feature values
cause the model prediction to flip — producing a transaction table suitable
for FP-Growth association-rule mining.

Public API
----------
run_for_k_values(k_values, data_path, output_base_dir,
                 target_col, perc_threshold)
    Train once, run counterfactual extraction for each k, and save results.

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
from joblib import Parallel, delayed


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _catboost_task_type() -> str:
    """
    Detect CUDA-capable GPU availability and return the appropriate CatBoost
    task_type string ('GPU' or 'CPU').
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
       verified against the model before use (see _process_single_sample).

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
    - model.predict calls reduced from O(1 + 2k) to O(3) per boundary sample:
        * 1 call for the original instance
        * 1 batched call for all k CF verifications
        * 1 batched call for all perturbations across all valid CFs
    - Perturbation matrix built via NumPy broadcasting (no Python inner loops).
    - Per-class global indices pre-cached in fit() so workers never recompute
      np.where(y_enc == label) on every parallel call.
    - X_enc stored as C-contiguous int32 for optimal cache behaviour.
    - joblib 'loky' backend provides true multiprocessing, bypassing the GIL.
    """

    def __init__(self, k_neighbors: int = 10, perc_threshold: int = 10) -> None:
        self.k_neighbors     = k_neighbors
        self.perc_threshold  = perc_threshold
        self.model           = None
        self.feature_encoder = OrdinalEncoder(dtype=int)
        self.label_encoder   = LabelEncoder()
        self.trees: dict     = {}   # one BallTree (Hamming metric) per class
        self._task_type      = _catboost_task_type()

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train CatBoost on (X, y) and build one BallTree per class.

        Parameters
        ----------
        X : feature matrix (categorical and/or integer columns)
        y : binary target (0/1)
        """
        print('  > Training CatBoost and building BallTrees...')
        self.feature_names = X.columns.tolist()

        y_enc = self.label_encoder.fit_transform(y)

        # C-contiguous int32: faster NumPy slicing and better cache locality
        # when building perturbation matrices in worker processes.
        X_enc = np.ascontiguousarray(
            self.feature_encoder.fit_transform(X), dtype=np.int32
        )

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_enc, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )

        self.model = CatBoostClassifier(
            iterations=100, depth=6, learning_rate=0.1,
            verbose=0, allow_writing_files=False,
            task_type=self._task_type,
        )
        self.model.fit(
            X_tr, y_tr,
            cat_features=list(range(X_enc.shape[1])),
            eval_set=(X_val, y_val),
        )

        # BallTrees built on the full encoded training set so the boundary
        # percentile reflects the complete training distribution.
        for label in np.unique(y_enc):
            idx = np.where(y_enc == label)[0]
            self.trees[label] = BallTree(X_enc[idx], metric='hamming')

        self.X_enc = X_enc
        self.y_enc = y_enc

        # Pre-cache per-class global indices so worker processes never
        # recompute np.where(y_enc == label) on every parallel call.
        self.class_indices = {
            int(label): np.where(y_enc == label)[0]
            for label in np.unique(y_enc)
        }

    # ------------------------------------------------------------------
    # _process_single_sample  (worker dispatched by joblib)
    # ------------------------------------------------------------------

    def _process_single_sample(
        self,
        sample_idx: int,
        orig_enc_sample: np.ndarray,
        ind_row: np.ndarray,
        opp_label: int,
    ):
        """
        Process one boundary instance against its k CF neighbours.

        Performance profile per call:
            model.predict calls : 3  (vs. 1 + 2k in the naive version)
            NumPy ops           : fully vectorised, no Python inner loops

        Steps
        -----
        1.  Predict orig_pred with a single call.
        2.  Retrieve all k CF candidates; verify all of them in one batched
            predict call; discard those not predicted as opp_label.
        3.  Build the full perturbation matrix in one NumPy operation:
                shape = (total_diff_positions_across_valid_CFs, n_features)
            using np.tile + advanced indexing.
        4.  Predict all perturbations in one batched call.
        5.  Group driver strings by CF neighbour.

        Returns
        -------
        sample_idx           : int
        per_neighbor_results : list of (cf_global_idx, sorted_driver_list)
        """
        orig      = orig_enc_sample
        orig_pred = self.model.predict([orig])[0]

        # Map BallTree-local indices to global row indices (pre-cached).
        global_ind = self.class_indices[int(opp_label)][ind_row]
        neighbors  = self.X_enc[global_ind]       # shape: (k, n_features)

        # Batch-verify all k CF candidates in one predict call.
        cf_preds     = self.model.predict(neighbors).ravel()
        valid_mask   = cf_preds == opp_label
        valid_global = global_ind[valid_mask]
        valid_neigh  = neighbors[valid_mask]       # shape: (n_valid, n_features)

        if len(valid_neigh) == 0:
            return sample_idx, []

        # Build perturbation matrix via NumPy broadcasting.
        # diff_matrix[i, j] is True where valid CF i differs from orig at j.
        # np.where returns aligned (cf_row, feat_col) index arrays.
        diff_matrix      = valid_neigh != orig          # (n_valid, n_feat)
        cf_row, feat_col = np.where(diff_matrix)        # both: shape (P,)
        n_perturbations  = len(cf_row)

        if n_perturbations == 0:
            return sample_idx, []

        # Build (P, n_features) matrix: start from P copies of orig, then
        # inject each CF value at the corresponding feature position.
        perturb_matrix = np.tile(orig, (n_perturbations, 1))
        perturb_matrix[np.arange(n_perturbations), feat_col] = \
            valid_neigh[cf_row, feat_col]

        # Predict all perturbations in one batched call.
        all_preds = self.model.predict(perturb_matrix).ravel()

        # Build driver strings for all perturbations (vectorised decode).
        driver_strings = [
            f'{self.feature_names[feat_col[p]]}'
            f'={self.feature_encoder.categories_[feat_col[p]][valid_neigh[cf_row[p], feat_col[p]]]}'
            for p in range(n_perturbations)
        ]

        # Collect drivers per valid CF neighbour.
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

    def explain(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Run boundary-crossing analysis on (X, y).

        For each class label:
          1. Compute Hamming distance to the nearest opposite-class instance.
          2. Keep only instances whose distance ≤ the perc_threshold-th
             percentile (boundary filter).
          3. Keep only boundary instances the model predicts correctly,
             mirroring the consistency check in the original paper.
          4. For each surviving boundary instance, query k nearest CF
             neighbours and run the vectorised per-feature swap in parallel.

        Returns
        -------
        pd.DataFrame with columns:
            Sample_ID             — original DataFrame index
            CF_Neighbor_ID        — positional index in self.X_enc
            Counterfactual_Values — sorted list of 'FEATURE=cf_value' strings
        """
        print('  > Extracting counterfactuals '
              '(parallel, vectorised per-feature swap)...')

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

            # Boundary filter: keep instances within the perc_threshold-th
            # percentile of Hamming distance to the opposite class.
            min_dist, _ = tree.query(X_enc[pos_idx], k=1)
            min_dist    = min_dist.ravel()
            threshold   = np.percentile(min_dist, self.perc_threshold)

            boundary_idx = pos_idx[min_dist <= threshold]
            if len(boundary_idx) == 0:
                continue

            # Model-prediction filter: retain only instances predicted as
            # their true label (mirrors the original paper's consistency check).
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

            # Query k nearest CF neighbours for each boundary instance.
            _, ind = tree.query(X_enc[boundary_idx], k=self.k_neighbors)

            # Dispatch in parallel; 'loky' spawns real processes (no GIL).
            parallel_results = Parallel(n_jobs=-1, backend='loky')(
                delayed(self._process_single_sample)(
                    sample_idx,
                    X_enc[sample_idx].copy(),   # copy: worker-safe
                    ind[i],
                    opp_label,
                )
                for i, sample_idx in enumerate(boundary_idx)
            )

            for sample_idx, per_neighbor_results in parallel_results:
                for cf_global_idx, found_drivers in per_neighbor_results:
                    rows.append({
                        'Sample_ID':             original_indices[sample_idx],
                        'CF_Neighbor_ID':        cf_global_idx,
                        'Counterfactual_Values': found_drivers,
                    })

        print(f'    - done: {len(rows)} (sample, CF-neighbour) pairs '
              f'with at least one driver\n')
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def extract_labels_and_values(results_dir: Path) -> None:
    """
    Parse transactions_values.csv and write four derived CSV files:

        labels_only.csv          per (sample, CF): all driver feature names
        labels_only_unique.csv   per (sample, CF): unique driver feature names
        values_only.csv          per (sample, CF): all CF category values
        values_only_unique.csv   per (sample, CF): unique CF category values

    Deduplication is performed on (label, value) pairs jointly so that
    labels_only_unique and values_only_unique remain positionally aligned.
    """
    print('  > Extracting labels and values from counterfactual drivers...')
    input_file = results_dir / 'transactions_values.csv'

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

        # Deduplicate on (label, value) pairs jointly to keep the two lists
        # positionally aligned after deduplication.
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
    Collapse all (sample, CF-neighbour) rows into one transaction per sample,
    writing two aggregated CSV files for ARM on feature labels.

    aggregated_labels_by_sample.csv
        One row per sample.  'Labels' contains the union of all unique driver
        feature names across every CF neighbour — duplicates removed.
        Recommended input for FP-Growth.

    aggregated_labels_duplicates_by_sample.csv
        One row per sample.  'Labels' contains all driver feature names
        including duplicates (a feature appearing as a driver for multiple CFs
        of the same sample is listed multiple times).

    Both files share columns:
        Sample_ID        — original row index in the dataset
        Labels           — list of feature-name drivers (see above)
        Num_Labels       — cardinality of the label list
        Num_CF_Neighbors — number of CF neighbours that contributed drivers
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

    df['Labels'] = df['Labels'].apply(ast.literal_eval)

    aggregated_unique = []
    aggregated_dupl   = []

    for sample_id, group in df.groupby('Sample_ID'):
        n_cf = len(group)

        # Union of all driver labels across CFs — duplicates removed.
        all_labels_set = set()
        for labels_list in group['Labels']:
            all_labels_set.update(labels_list)
        unique_labels = sorted(all_labels_set)

        # All driver labels concatenated across CFs — duplicates kept.
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
    print(f'    - {len(df)} (sample, CF) pairs collapsed into '
          f'{n_samples} unique samples')

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
    target_col: str  = 'INCOME_ABOVE_THRESHOLD',
    perc_threshold: int = 10,
) -> dict[int, Path]:
    """
    Train CategoricalBoCSoR once and run counterfactual extraction for each k.

    The model and BallTrees are fitted on the training split and shared across
    all k values; only the neighbourhood query size changes per iteration.

    Parameters
    ----------
    k_values        : neighbourhood sizes to evaluate
    data_path       : path to the input CSV
    output_base_dir : root directory for per-k result sub-folders
    target_col      : name of the target column in the CSV
    perc_threshold  : boundary filter percentile

    Returns
    -------
    dict mapping k → Path of labels_only_unique.csv for that k
    """
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"=" * 70}')
    print(f'K-VARIATION EXPERIMENT — {len(k_values)} k values')
    print(f'{"=" * 70}')
    print(f'  > k values        : {k_values}')
    print(f'  > perc_threshold  : {perc_threshold}')
    print(f'  > target column   : {target_col}')
    print('-' * 50)

    print('  > Loading dataset and splitting...')
    df = pd.read_csv(data_path)

    if target_col != 'target':
        df = df.rename(columns={target_col: 'target'})

    X, y = df.drop(columns=['target']), df['target']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'    - train: {len(X_tr):,} samples  |  test: {len(X_te):,} samples')

    # Fit once — model and BallTrees are shared across all k iterations.
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
        explainer.k_neighbors = k

        # explain() runs on the training set: we explain the classifier's
        # decision logic, not its generalisation to unseen instances.
        transactions = explainer.explain(X_tr, y_tr)

        if transactions.empty:
            print('    > 0 transactions — skipping downstream steps.')
            continue

        transactions_path = k_dir / 'transactions_values.csv'
        transactions.to_csv(transactions_path, index=False)
        print(f'    > {len(transactions)} transactions saved to '
              f'{transactions_path.name}')

        extract_labels_and_values(k_dir)
        aggregate_drivers_by_sample(k_dir)

        labels_path = k_dir / 'labels_only_unique.csv'
        if labels_path.exists() and labels_path.stat().st_size > 0:
            k_labels_map[k] = labels_path
        else:
            print(f'    - WARNING: labels_only_unique.csv is empty for k={k}, '
                  f'skipping.')

    print(f'\n{"=" * 70}')
    print('  > All k values completed.')
    print(f'{"=" * 70}\n')

    return k_labels_map


# ---------------------------------------------------------------------------
# Log capturing utility
# ---------------------------------------------------------------------------

class _TeeWriter:
    """Write to the original stdout and simultaneously capture in a StringIO buffer."""

    def __init__(self, original_stdout):
        self._orig = original_stdout
        self._buf  = io.StringIO()

    def write(self, text: str) -> None:
        self._orig.write(text)
        self._buf.write(text)

    def flush(self) -> None:
        self._orig.flush()

    def getvalue(self) -> str:
        return self._buf.getvalue()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    survey_year: str       = '2024',
    regions: dict          = None,
    k_values: list[int]    = None,
    perc_threshold: int    = 10,
    target_col: str        = 'INCOME_ABOVE_THRESHOLD',
    base_dir: Path         = None,
) -> None:
    """
    Run counterfactual extraction for all specified regions and k values.

    Parameters
    ----------
    survey_year     : ACS survey year, used to locate input CSV files.
    regions         : mapping of region name → CSV path.  If None, defaults
                      to Northeast and South under base_dir/data/.
    k_values        : neighbourhood sizes for BoCSoR.
    perc_threshold  : boundary filter percentile.
    target_col      : name of the binary target column in the CSV.
    base_dir        : project root directory.  Defaults to two levels above
                      this file when run standalone.
    """
    if base_dir is None:
        base_dir = (
            Path('/content')
            if Path('/content').exists()
            else Path(__file__).resolve().parent.parent
        )
    base_dir = Path(base_dir)

    if k_values is None:
        k_values = [1, 3, 5, 7]

    if regions is None:
        data_dir = base_dir / 'data'
        regions  = {
            'northeast': data_dir / f'acs_income_northeast_{survey_year}.csv',
            'south':     data_dir / f'acs_income_south_{survey_year}.csv',
        }

    results_dir = base_dir / 'results'

    tee = _TeeWriter(sys.stdout)
    sys.stdout = tee

    try:
        for region, data_path in regions.items():
            output_dir = results_dir / region / 'important_features'
            output_dir.mkdir(parents=True, exist_ok=True)

            print('\n' + '=' * 70)
            print(f'COUNTERFACTUAL EXTRACTION — {region.upper()}')
            print('=' * 70 + '\n')

            if not Path(data_path).exists():
                print(f'  > Error: {data_path} not found — '
                      f'run create_dataset.py first.')
                continue

            k_labels_map = run_for_k_values(
                k_values       = k_values,
                data_path      = data_path,
                output_base_dir= output_dir,
                target_col     = target_col,
                perc_threshold = perc_threshold,
            )

            print('  > k_labels_map ready:')
            for k, path in k_labels_map.items():
                print(f'    k={k:>2} -> {path}')

        print('\n' + '=' * 70)
        print('Done.')
        print('=' * 70 + '\n')

    finally:
        sys.stdout = tee._orig

    # Save full execution log and per-region excerpts.
    full_log = tee.getvalue()
    results_dir.mkdir(parents=True, exist_ok=True)

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
                print(f'  > Region log saved to '
                      f'{region_dir / "feature_importance_log.txt"}')


if __name__ == '__main__':
    main()
