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
    """
    def __init__(self, k_neighbors=10, perc_threshold=10):
        self.k_neighbors = k_neighbors
        self.perc_threshold = perc_threshold
        self.model = None
        self.feature_encoder = OrdinalEncoder(dtype=int)
        self.label_encoder = LabelEncoder()
        self.trees = {}

    def fit(self, X, y):
        """Train CatBoost and build one BallTree per class for counterfactual search."""
        print("  > Training CatBoost and building BallTrees...")
        self.feature_names = X.columns.tolist()

        # need integer encoding for both CatBoost cat_features and Hamming distance
        y_enc = self.label_encoder.fit_transform(y)
        X_enc = self.feature_encoder.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(
            X_enc, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )

        self.model = CatBoostClassifier(
            iterations=100, depth=6, learning_rate=0.1,
            verbose=0, allow_writing_files=False
        )
        self.model.fit(
            X_train, y_train,
            cat_features=list(range(X_enc.shape[1])),
            eval_set=(X_val, y_val)
        )

        # one BallTree per class — at query time we always search the opposite class
        for label in np.unique(y_enc):
            idx = np.where(y_train == label)[0]
            self.trees[label] = BallTree(X_train[idx], metric='hamming')

        self.X_train_enc = X_train
        self.y_train_enc = y_train

    def explain(self, X_test, y_test):
        """
        Extract 1-sparse counterfactuals for each test sample.
        A counterfactual is 1-sparse if it differs from the original by exactly 1 feature
        — these are the most actionable explanations near the decision boundary.
        """
        print("  > Extracting 1-sparse counterfactuals...")
        X_test_enc = self.feature_encoder.transform(X_test)
        y_test_enc = self.label_encoder.transform(y_test)

        results = {i: set() for i in range(len(X_test_enc))}

        for label in np.unique(y_test_enc):
            idx = np.where(y_test_enc == label)[0]
            if len(idx) == 0:
                continue

            # look for neighbors in the opposite class
            opp_label = 1 - label
            tree = self.trees.get(opp_label)
            if tree is None:
                continue

            dist, ind = tree.query(X_test_enc[idx], k=self.k_neighbors)

            for i, sample_idx in enumerate(idx):
                orig = X_test_enc[sample_idx]
                neighbors = self.X_train_enc[
                    np.where(self.y_train_enc == opp_label)[0][ind[i]]
                ]

                dists = np.sum(neighbors != orig, axis=1)
                min_dist = np.min(dists)

                # skip if no 1-sparse counterfactual exists among the k neighbors
                if min_dist != 1:
                    continue

                for cand in neighbors[dists == min_dist]:
                    f_idx = np.where(cand != orig)[0][0]
                    val_str = self.feature_encoder.categories_[f_idx][cand[f_idx]]
                    results[sample_idx].add(f"{self.feature_names[f_idx]}={val_str}")

        rows = [{'Sample_ID': k, 'Counterfactual_Values': list(v)} for k, v in results.items() if v]
        print(f"    - done, {len(rows)} samples have at least one counterfactual\n")
        return pd.DataFrame(rows)


def extract_labels(results_dir):
    """
    Pull just the feature names out of the counterfactual values and save them
    as transaction lists for FP-Growth. Saves two versions: one with duplicates
    (labels_only.csv) and one without (labels_only_unique.csv).
    """
    print("  > Extracting labels from counterfactual values...")

    input_file = results_dir / "transactions_values.csv"
    output_file = results_dir / "labels_only.csv"
    output_unique = results_dir / "labels_only_unique.csv"

    df = pd.read_csv(input_file)

    labels_list = []
    labels_unique_list = []

    for _, row in df.iterrows():
        labels = re.findall(r'([A-Z]\w*)=', str(row['Counterfactual_Values']))
        labels_list.append(labels)

        # deduplicate while keeping the original order
        seen = set()
        unique = []
        for label in labels:
            if label not in seen:
                seen.add(label)
                unique.append(label)
        labels_unique_list.append(unique)

    pd.DataFrame({'Sample_ID': df['Sample_ID'], 'Labels': labels_list}).to_csv(output_file, index=False)
    print(f"    - labels (with duplicates) saved to {output_file.name}")

    pd.DataFrame({'Sample_ID': df['Sample_ID'], 'Labels': labels_unique_list}).to_csv(output_unique, index=False)
    print(f"    - labels (unique) saved to {output_unique.name}")


def run_for_k_values(k_values, data_path, output_base_dir):
    """
    Run the full pipeline (fit + explain + extract_labels) for each k in k_values.
    The train/test split is done once before the loop so all k values see the same data.

    Returns a dict {k: path_to_labels_only_unique.csv} that can be passed
    directly to run_k_comparison() in association_rules.py.
    """
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"K-VARIATION EXPERIMENT — {len(k_values)} values of k")
    print(f"{'='*70}")
    print(f"  > k values: {k_values}")
    print("-" * 50)

    # split once, reuse for all k — otherwise the comparison isn't fair
    print("  > Loading dataset and splitting...")
    df = pd.read_csv(data_path)
    X, y = df.drop(columns=["target"]), df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"    - train: {len(X_tr)} samples, test: {len(X_te)} samples")

    k_labels_map = {}

    for i, k in enumerate(k_values):
        k_dir = output_base_dir / f"k_{k}"
        k_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{i+1}/{len(k_values)}] k = {k}")

        explainer = CategoricalBoCSoR(k_neighbors=k, perc_threshold=10)
        explainer.fit(X_tr, y_tr)
        transactions = explainer.explain(X_te, y_te)

        transactions_path = k_dir / "transactions_values.csv"
        transactions.to_csv(transactions_path, index=False)
        print(f"    > {len(transactions)} transactions saved to {transactions_path.name}")

        extract_labels(k_dir)
        k_labels_map[k] = k_dir / "labels_only_unique.csv"

    print(f"\n{'='*70}")
    print("  > All k values done.")
    print(f"{'='*70}\n")

    return k_labels_map


if __name__ == "__main__":
    if Path("/content").exists():
        data_dir = Path("/content/data")
        results_dir = Path("/content/results")
        important_features_dir = Path("/content/results/important_features")
    else:
        base_dir = Path(__file__).resolve().parent.parent
        data_dir = base_dir / "data"
        results_dir = base_dir / "results"
        important_features_dir = results_dir / "important_features"

    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    important_features_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / "ACSIncome_NY_2018_categorized.csv"

    if not data_path.exists():
        print(f"  > Error: dataset not found at {data_path.name}")
    else:
        # single run with default k=10
        print("\n" + "="*70)
        print("COUNTERFACTUAL EXTRACTION FOR FEATURE IMPORTANCE")
        print("="*70 + "\n")

        out_path = important_features_dir / "transactions_values.csv"

        print("  > Loading and splitting dataset...")
        df = pd.read_csv(data_path)
        X, y = df.drop(columns=["target"]), df["target"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"    - train: {len(X_tr)} samples, test: {len(X_te)} samples\n")

        explainer = CategoricalBoCSoR(k_neighbors=10, perc_threshold=10)
        explainer.fit(X_tr, y_tr)
        transactions = explainer.explain(X_te, y_te)

        transactions.to_csv(out_path, index=False)
        print(f"  > Transactions saved to {out_path.name}\n")

        extract_labels(important_features_dir)

        # k-variation experiment — odd values from 1 to 19
        # pass k_labels_map to run_k_comparison() in association_rules.py
        k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        k_labels_map = run_for_k_values(k_values, data_path, important_features_dir)

        print("  > k_labels_map ready:")
        for k, path in k_labels_map.items():
            print(f"    k={k:>2} -> {path}")

    print("\n" + "="*70)
    print("Done.")
    print("="*70 + "\n")