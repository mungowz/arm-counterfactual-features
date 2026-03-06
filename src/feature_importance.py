# Install the CatBoost library (if needed)
# !pip install catboost --quiet

import ast
import pandas as pd
import numpy as np
import re
import subprocess
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.neighbors import BallTree
from pathlib import Path

# NUOVA IMPORTAZIONE PER IL MULTIPROCESSING
from joblib import Parallel, delayed

def _catboost_task_type():
    """
    Detect whether a CUDA-capable GPU is available and return the appropriate
    CatBoost task_type string.
    """
    try:
        subprocess.run(
            ['nvidia-smi'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=True
        )
        print("  > GPU detected — CatBoost will use task_type='GPU'")
        return 'GPU'
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Sul tuo Mac M2 stamperà questo e userà la CPU (molto più adatta per questo task)
        print("  > No CUDA GPU detected — CatBoost will use task_type='CPU'")
        return 'CPU'


class CategoricalBoCSoR:
    def __init__(self, k_neighbors=10, perc_threshold=10):
        self.k_neighbors    = k_neighbors
        self.perc_threshold = perc_threshold
        self.model          = None
        self.feature_encoder = OrdinalEncoder(dtype=int)
        self.label_encoder   = LabelEncoder()
        self.trees  = {}
        self._task_type = _catboost_task_type()

    def fit(self, X, y):
        print("  > Training CatBoost and building BallTrees...")
        self.feature_names = X.columns.tolist()

        y_enc = self.label_encoder.fit_transform(y)
        X_enc = self.feature_encoder.fit_transform(X)

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

        for label in np.unique(y_enc):
            idx = np.where(y_enc == label)[0]
            self.trees[label] = BallTree(X_enc[idx], metric='hamming')

        self.X_enc = X_enc
        self.y_enc = y_enc

    # --- NUOVO METODO HELPER PER IL MULTIPROCESSING ---
    def _process_single_sample(self, sample_idx, orig_enc_sample, ind_row, opp_label):
        """
        Processes a single boundary sample in parallel across M2 cores.
        Returns (sample_idx, per_neighbor_results) where per_neighbor_results
        is a list of (cf_global_idx, set_of_counterfactual_drivers), one entry
        per neighbor that produced at least one driver.

        sample_idx      — positional index into X_enc
        orig_enc_sample — encoded feature vector for this sample (copy, not a view)
        ind_row         — 1-D array of k BallTree-local indices for this sample's neighbors
        opp_label       — encoded label of the opposite class
        """
        orig = orig_enc_sample
        orig_pred = self.model.predict([orig])[0]

        global_ind = np.where(self.y_enc == opp_label)[0][ind_row]
        neighbors  = self.X_enc[global_ind]

        per_neighbor_results = []

        for cf_global_idx, cand in zip(global_ind, neighbors):
            diff_positions = np.where(cand != orig)[0]
            if len(diff_positions) == 0:
                continue

            batch_samples = []
            batch_labels  = []

            for f_idx in diff_positions:
                perturbed         = orig.copy()
                perturbed[f_idx]  = cand[f_idx]
                batch_samples.append(perturbed)

                val_str = self.feature_encoder.categories_[f_idx][cand[f_idx]]
                batch_labels.append(f"{self.feature_names[f_idx]}={val_str}")

            preds = self.model.predict(np.array(batch_samples))

            found_drivers = set()
            for pred, label_str in zip(preds, batch_labels):
                if pred != orig_pred:
                    found_drivers.add(label_str)

            if found_drivers:
                per_neighbor_results.append((int(cf_global_idx), found_drivers))

        return sample_idx, per_neighbor_results
    # ---------------------------------------------------

    def explain(self, X, y):
        print("  > Extracting counterfactuals (PARALLEL batched per-feature swap)...")
        X_enc  = self.feature_encoder.transform(X)
        y_enc  = self.label_encoder.transform(y)

        original_indices = X.index.tolist()
        rows = []

        all_classes = np.unique(y_enc)

        for label in all_classes:
            pos_idx = np.where(y_enc == label)[0]
            if len(pos_idx) == 0:
                continue

            opp_label = all_classes[all_classes != label][0]
            tree = self.trees.get(opp_label)
            if tree is None:
                continue

            min_dist_to_opp, _ = tree.query(X_enc[pos_idx], k=1)
            min_dist_to_opp    = min_dist_to_opp.ravel()
            dist_threshold     = np.percentile(min_dist_to_opp, self.perc_threshold)
            boundary_mask      = min_dist_to_opp <= dist_threshold
            boundary_pos_idx   = pos_idx[boundary_mask]

            if len(boundary_pos_idx) == 0:
                continue

            print(f"    - class {label}: {len(boundary_pos_idx)}/{len(pos_idx)} "
                  f"boundary samples (perc_threshold={self.perc_threshold})")

            _, ind = tree.query(X_enc[boundary_pos_idx], k=self.k_neighbors)

            # --- MAGIA DEL MULTIPROCESSING SUL MAC M2 ---
            # n_jobs=-1 dice a joblib di usare TUTTI i core fisici e logici disponibili.
            # backend="threading" evita il pickle di self.model (CatBoost rilascia il GIL)
            parallel_results = Parallel(n_jobs=-1, backend="threading")(
                delayed(self._process_single_sample)(
                    sample_idx, X_enc[sample_idx], ind[i], opp_label
                )
                for i, sample_idx in enumerate(boundary_pos_idx)
            )

            # Raccogliamo i risultati: una riga per coppia (campione, vicino CF)
            for sample_idx, per_neighbor_results in parallel_results:
                for cf_global_idx, found_drivers in per_neighbor_results:
                    rows.append({
                        'Sample_ID': original_indices[sample_idx],
                        'CF_Neighbor_ID': cf_global_idx,
                        'Counterfactual_Values': list(found_drivers)
                    })

        print(f"    - done, {len(rows)} (sample, neighbor) pairs have at least one "
              f"counterfactual driver\n")
        return pd.DataFrame(rows)


def extract_labels(results_dir):
    print("  > Extracting labels from counterfactual values...")
    input_file    = results_dir / "transactions_values.csv"
    output_file   = results_dir / "labels_only.csv"
    output_unique = results_dir / "labels_only_unique.csv"

    # Guard robusto: cattura sia file mancante che CSV vuoto/solo-header
    if not input_file.exists():
        print("    - WARNING: transactions file not found, skipping.")
        return

    try:
        df = pd.read_csv(input_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"    - WARNING: could not read transactions file ({e}), skipping.")
        return

    if df.empty:
        print("    - WARNING: no transactions found, skipping label extraction.")
        return

    labels_list        = []
    labels_unique_list = []

    has_cf_col = 'CF_Neighbor_ID' in df.columns

    for _, row in df.iterrows():
        labels = re.findall(r'([A-Z]\w*)=', str(row['Counterfactual_Values']))
        labels_list.append(labels)

        seen   = set()
        unique = []
        for label in labels:
            if label not in seen:
                seen.add(label)
                unique.append(label)
        labels_unique_list.append(unique)

    base_cols = {'Sample_ID': df['Sample_ID']}
    if has_cf_col:
        base_cols['CF_Neighbor_ID'] = df['CF_Neighbor_ID']

    pd.DataFrame({**base_cols, 'Labels': labels_list}).to_csv(
        output_file, index=False
    )
    print(f"    - labels (with duplicates) saved to {output_file.name}")

    pd.DataFrame({**base_cols, 'Labels': labels_unique_list}).to_csv(
        output_unique, index=False
    )
    print(f"    - labels (unique) saved to {output_unique.name}")


def aggregate_drivers_by_sample(results_dir):
    """
    Aggregate drivers by sample for association rule mining.

    Reads labels_only_unique.csv (one row per sample-CF pair) and groups all
    drivers by sample into a single transaction. This enables FP-Growth to
    discover patterns like "when SCHL changes, OCCP also often changes".
    """
    print("  > Aggregating drivers by sample...")

    labels_path = results_dir / "labels_only_unique.csv"
    output_path = results_dir / "aggregated_drivers_by_sample.csv"

    if not labels_path.exists():
        print(f"    - WARNING: {labels_path.name} not found, skipping aggregation.")
        return

    try:
        df = pd.read_csv(labels_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"    - WARNING: could not read labels file ({e}), skipping aggregation.")
        return

    if df.empty:
        print("    - WARNING: labels file is empty, skipping aggregation.")
        return

    # Parse string representations of lists to actual lists
    df['Labels'] = df['Labels'].apply(ast.literal_eval)

    # Group by Sample_ID and collect all unique drivers
    aggregated = []
    for sample_id, group in df.groupby('Sample_ID'):
        # Flatten all label lists and remove duplicates while preserving order
        all_drivers = set()
        for labels_list in group['Labels']:
            all_drivers.update(labels_list)

        aggregated.append({
            'Sample_ID': sample_id,
            'Drivers': sorted(list(all_drivers)),  # sorted for reproducibility
            'Num_Drivers': len(all_drivers),
            'Num_CF_Neighbors': len(group)
        })

    result_df = pd.DataFrame(aggregated)
    result_df.to_csv(output_path, index=False)

    print(f"    - Aggregated {len(df)} (sample, neighbor) pairs into {len(result_df)} unique samples")

    # Statistics
    dist = result_df['Num_Drivers'].value_counts().sort_index().to_dict()
    for n_drivers, count in sorted(dist.items()):
        pct = count / len(result_df) * 100
        print(f"      {n_drivers} driver(s): {count} samples ({pct:.1f}%)")

    print(f"    - aggregated CSV saved to {output_path.name}")


def run_for_k_values(k_values, data_path, output_base_dir, perc_threshold=10):
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"K-VARIATION EXPERIMENT — {len(k_values)} values of k")
    print(f"{'='*70}")
    print(f"  > k values:       {k_values}")
    print(f"  > perc_threshold: {perc_threshold}")
    print("-" * 50)

    print("  > Loading dataset and splitting...")
    df   = pd.read_csv(data_path)
    X, y = df.drop(columns=["target"]), df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    - train: {len(X_tr)} samples, test: {len(X_te)} samples")

    print(f"\n  > Fitting model (shared across all k values)...")
    explainer = CategoricalBoCSoR(k_neighbors=k_values[0], perc_threshold=perc_threshold)
    explainer.fit(X_tr, y_tr)

    k_labels_map = {}

    for i, k in enumerate(k_values):
        k_dir = output_base_dir / f"k_{k}"
        k_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{i+1}/{len(k_values)}] k = {k}")

        explainer.k_neighbors = k

        transactions = explainer.explain(X_tr, y_tr)

        transactions_path = k_dir / "transactions_values.csv"

        # Scriviamo il CSV SOLO se ci sono transazioni effettive
        if not transactions.empty:
            transactions.to_csv(transactions_path, index=False)
            print(f"    > {len(transactions)} transactions saved to "
                  f"{transactions_path.name}")
            extract_labels(k_dir)
            aggregate_drivers_by_sample(k_dir)
        else:
            print(f"    > 0 transactions — skipping CSV write and label extraction.")

        labels_path = k_dir / "labels_only_unique.csv"
        if labels_path.exists() and labels_path.stat().st_size > 0:
            k_labels_map[k] = labels_path
        else:
            print(f"    - WARNING: no labels for k={k}, skipping.")

    print(f"\n{'='*70}")
    print("  > All k values done.")
    print(f"{'='*70}\n")

    return k_labels_map


if __name__ == "__main__":
    if Path("/content").exists():
        base_dir = Path("/content")
    else:
        base_dir = Path(__file__).resolve().parent.parent

    data_dir    = base_dir / "data"
    results_dir = base_dir / "results"

    regions = {
        'northeast': data_dir / "ACSIncome_northeast_2018_balanced.csv",
        'south':     data_dir / "ACSIncome_south_2018_balanced.csv",
    }

    k_values       = [1, 3, 5, 7]
    perc_threshold = 10  

    for region, data_path in regions.items():
        important_features_dir = results_dir / region / "important_features"
        important_features_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print(f"COUNTERFACTUAL EXTRACTION — {region.upper()}")
        print("="*70 + "\n")

        if not data_path.exists():
            print(f"  > Error: {data_path.name} not found — run create_dataset.py first.")
            continue

        k_labels_map = run_for_k_values(
            k_values, data_path, important_features_dir, perc_threshold
        )

        print("  > k_labels_map ready:")
        for k, path in k_labels_map.items():
            print(f"    k={k:>2} -> {path}")

    print("\n" + "="*70)
    print("Done.")
    print("="*70 + "\n")