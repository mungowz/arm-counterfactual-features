# Install CatBoost, run this cell in a Jupyter notebook or Colab environment
# !pip install catboost --quiet


import pandas as pd
import numpy as np
import re
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.neighbors import BallTree
from pathlib import Path


class CategoricalBoCSoR:
    """
    BoCSoR adapted for fully categorical datasets.
    Hamming distance + BallTree for nearest neighbor search in discrete space.

    Faithfully follows the CounterfactualExplainerByProximity approach from the
    original paper, with three adaptations for the discrete/categorical setting:

      1. Hamming distance replaces Euclidean — the only meaningful metric when
         all features are nominal categories.
      2. No synthetic interpolation — np.linspace cannot be applied to discrete
         values, so each k neighbor is used directly as a candidate switch point.
         All k neighbors are tested (rather than just the closest one) to maximise
         coverage for global feature importance extraction via FP-Growth.
      3. Swap direction — the original code starts from the switch point and
         restores features to the original one at a time. Here we start from the
         original and swap features to the candidate's value one at a time. Both
         directions isolate single-feature contributions; the swap-from-original
         direction is the natural choice when the switch point is a real data
         point rather than a synthetic interpolation.
    """

    def __init__(self, k_neighbors=10, perc_threshold=10):
        self.k_neighbors    = k_neighbors
        self.perc_threshold = perc_threshold
        self.model          = None
        self.feature_encoder = OrdinalEncoder(dtype=int)
        self.label_encoder   = LabelEncoder()
        self.trees = {}

    def fit(self, X, y):
        """
        Train CatBoost and build one BallTree per class for counterfactual search.

        The internal 80/20 split is used exclusively for CatBoost's eval_set.
        BallTrees are built on the FULL encoded set passed to fit() — matching
        the original authors' approach where find_counterfactuals() queries
        self.x (the complete training set), not a sub-split of it.
        """
        print("  > Training CatBoost and building BallTrees...")
        self.feature_names = X.columns.tolist()

        y_enc = self.label_encoder.fit_transform(y)
        X_enc = self.feature_encoder.fit_transform(X)

        # internal split only for CatBoost's eval_set — not used for BallTrees
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_enc, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )

        self.model = CatBoostClassifier(
            iterations=100, depth=6, learning_rate=0.1,
            verbose=0, allow_writing_files=False
        )
        self.model.fit(
            X_tr, y_tr,
            cat_features=list(range(X_enc.shape[1])),
            eval_set=(X_val, y_val)
        )

        # BallTrees built on the FULL encoded set, not just X_tr.
        # Storing self.X_enc / self.y_enc lets explain() map BallTree-local
        # indices back to global rows and apply the perc_threshold filter.
        for label in np.unique(y_enc):
            idx = np.where(y_enc == label)[0]
            self.trees[label] = BallTree(X_enc[idx], metric='hamming')

        self.X_enc = X_enc
        self.y_enc = y_enc

    def explain(self, X, y):
        """
        Extract per-feature counterfactual drivers for all boundary samples.

        Pipeline (mirrors explain_decision_boundary + explain_sample):

          1. Boundary filter (perc_threshold):
             For each class, compute the minimum Hamming distance from every
             sample to the nearest neighbor in the opposite class. Keep only
             samples within the perc_threshold-th percentile of that distance
             distribution — these are the samples closest to the decision
             boundary and most informative for global feature importance.

          2. Neighbor search:
             For each boundary sample, query the k nearest neighbors in the
             opposite class using the pre-built BallTree.

          3. Per-feature swap → predict → restore:
             For every candidate neighbor and every feature position where
             candidate differs from original:
               a. swap    — set test_sample[f] = candidate[f]
               b. predict — ask CatBoost whether the class flips
               c. record  — if flip, save feature f as a counterfactual driver
               d. restore — reset test_sample[f] = original[f] before next f

             The restore step guarantees each feature is tested in complete
             isolation: at predict time, test_sample differs from original
             in exactly one feature position.
        """
        print("  > Extracting counterfactuals (per-feature swap)...")
        X_enc  = self.feature_encoder.transform(X)
        y_enc  = self.label_encoder.transform(y)

        # key by the original DataFrame index so Sample_ID is traceable
        # back to the balanced CSV row, not just a positional offset in X_enc
        original_indices = X.index.tolist()
        results = {original_indices[i]: set() for i in range(len(X_enc))}

        # derive the opposite class without assuming labels are {0, 1}:
        # all_classes is a sorted 1-D array from np.unique, so
        # all_classes[all_classes != label] gives the complement safely
        # for any binary encoding
        all_classes = np.unique(y_enc)

        for label in all_classes:
            pos_idx = np.where(y_enc == label)[0]
            if len(pos_idx) == 0:
                continue

            opp_label = all_classes[all_classes != label][0]
            tree = self.trees.get(opp_label)
            if tree is None:
                continue

            # --- boundary filter (perc_threshold) ---
            # query each sample's distance to its single nearest neighbor in
            # the opposite class, then keep only those within the percentile
            # threshold — equivalent to explain_decision_boundary()'s cdist
            # + np.percentile filter, adapted for Hamming via BallTree
            min_dist_to_opp, _ = tree.query(X_enc[pos_idx], k=1)
            min_dist_to_opp    = min_dist_to_opp.ravel()
            dist_threshold     = np.percentile(min_dist_to_opp, self.perc_threshold)
            boundary_mask      = min_dist_to_opp <= dist_threshold
            boundary_pos_idx   = pos_idx[boundary_mask]

            if len(boundary_pos_idx) == 0:
                continue

            print(f"    - class {label}: {len(boundary_pos_idx)}/{len(pos_idx)} "
                  f"boundary samples (perc_threshold={self.perc_threshold})")

            # query k neighbors for boundary samples only
            _, ind = tree.query(X_enc[boundary_pos_idx], k=self.k_neighbors)

            for i, sample_idx in enumerate(boundary_pos_idx):
                orig      = X_enc[sample_idx]
                orig_pred = self.model.predict([orig])[0]  # cached once per sample

                # map BallTree-local indices back to self.X_enc global indices
                global_ind = np.where(self.y_enc == opp_label)[0][ind[i]]
                neighbors  = self.X_enc[global_ind]

                for cand in neighbors:
                    diff_positions = np.where(cand != orig)[0]
                    if len(diff_positions) == 0:
                        continue  # identical sample — nothing to test

                    # one working copy per candidate; the restore step keeps it
                    # identical to orig between feature iterations
                    test_sample = orig.copy()

                    for f_idx in diff_positions:
                        test_sample[f_idx] = cand[f_idx]        # a. swap

                        if self.model.predict([test_sample])[0] != orig_pred:
                            # c. flip confirmed — record the feature
                            val_str = self.feature_encoder.categories_[f_idx][cand[f_idx]]
                            results[original_indices[sample_idx]].add(
                                f"{self.feature_names[f_idx]}={val_str}"
                            )

                        test_sample[f_idx] = orig[f_idx]        # d. restore

        rows = [
            {'Sample_ID': sid, 'Counterfactual_Values': list(v)}
            for sid, v in results.items() if v
        ]
        print(f"    - done, {len(rows)} boundary samples have at least one "
              f"counterfactual driver\n")
        return pd.DataFrame(rows)


def extract_labels(results_dir):
    """
    Pull just the feature names out of the counterfactual values and save them
    as transaction lists for FP-Growth. Saves two versions: one with duplicates
    (labels_only.csv) and one without (labels_only_unique.csv).
    """
    print("  > Extracting labels from counterfactual values...")

    input_file    = results_dir / "transactions_values.csv"
    output_file   = results_dir / "labels_only.csv"
    output_unique = results_dir / "labels_only_unique.csv"

    df = pd.read_csv(input_file)

    labels_list        = []
    labels_unique_list = []

    for _, row in df.iterrows():
        labels = re.findall(r'([A-Z]\w*)=', str(row['Counterfactual_Values']))
        labels_list.append(labels)

        # deduplicate while keeping the original order
        seen   = set()
        unique = []
        for label in labels:
            if label not in seen:
                seen.add(label)
                unique.append(label)
        labels_unique_list.append(unique)

    pd.DataFrame({'Sample_ID': df['Sample_ID'], 'Labels': labels_list}).to_csv(
        output_file, index=False
    )
    print(f"    - labels (with duplicates) saved to {output_file.name}")

    pd.DataFrame({'Sample_ID': df['Sample_ID'], 'Labels': labels_unique_list}).to_csv(
        output_unique, index=False
    )
    print(f"    - labels (unique) saved to {output_unique.name}")


def run_for_k_values(k_values, data_path, output_base_dir, perc_threshold=10):
    """
    Run the full pipeline (fit + explain + extract_labels) for each k in k_values.

    The train/test split and model fit are performed once before the loop so that:
      - all k values see identical train/test partitions (fair comparison);
      - CatBoost is not redundantly re-trained for each k, since the model
        depends only on the data and random_state, not on k_neighbors.

    explain() is called on X_tr (the training set), matching the original
    authors' approach where CounterfactualExplainerByProximity is initialised
    with X_train and the boundary analysis is run on training samples.

    Returns a dict {k: path_to_labels_only_unique.csv} that can be passed
    directly to run_k_comparison() in association_rules.py.
    """
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"K-VARIATION EXPERIMENT — {len(k_values)} values of k")
    print(f"{'='*70}")
    print(f"  > k values:       {k_values}")
    print(f"  > perc_threshold: {perc_threshold}")
    print("-" * 50)

    # split once, reuse for all k — otherwise the comparison isn't fair
    print("  > Loading dataset and splitting...")
    df   = pd.read_csv(data_path)
    X, y = df.drop(columns=["target"]), df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    - train: {len(X_tr)} samples, test: {len(X_te)} samples")

    # fit CatBoost and build BallTrees once — k_neighbors only affects how many
    # neighbors are queried in explain(), not the model or the trees themselves
    print(f"\n  > Fitting model (shared across all k values)...")
    explainer = CategoricalBoCSoR(k_neighbors=k_values[0], perc_threshold=perc_threshold)
    explainer.fit(X_tr, y_tr)

    k_labels_map = {}

    for i, k in enumerate(k_values):
        k_dir = output_base_dir / f"k_{k}"
        k_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{i+1}/{len(k_values)}] k = {k}")

        # update k_neighbors in place — no re-fit needed
        explainer.k_neighbors = k

        # explain() is called on the TRAINING set, matching the original paper's
        # approach of running the boundary analysis on training samples
        transactions = explainer.explain(X_tr, y_tr)

        transactions_path = k_dir / "transactions_values.csv"
        transactions.to_csv(transactions_path, index=False)
        print(f"    > {len(transactions)} transactions saved to "
              f"{transactions_path.name}")

        extract_labels(k_dir)
        k_labels_map[k] = k_dir / "labels_only_unique.csv"

    print(f"\n{'='*70}")
    print("  > All k values done.")
    print(f"{'='*70}\n")

    return k_labels_map


if __name__ == "__main__":
    # when run standalone, processes both regions independently
    # expects the balanced CSVs produced by create_dataset.py to already exist
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

    k_values      = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    perc_threshold = 10  # matches original paper: explain_decision_boundary(perc_threshold=10)

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