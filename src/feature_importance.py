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
    Implementation of BoCSoR adapted for fully categorical datasets.
    Uses Hamming distance and BallTree for nearest neighbor queries in discrete space.
    """
    def __init__(self, k_neighbors=10, perc_threshold=10):
        self.k_neighbors = k_neighbors
        self.perc_threshold = perc_threshold
        self.model = None
        self.feature_encoder = OrdinalEncoder(dtype=int)
        self.label_encoder = LabelEncoder()
        self.trees = {}

    def fit(self, X, y):
        """
        Train the CatBoost model and build BallTrees for counterfactual extraction.

        Args:
            X: Feature matrix (pandas DataFrame)
            y: Target vector

        Returns:
            None
        """
        print("  > Training CatBoost and building BallTrees...")
        self.feature_names = X.columns.tolist()

        # Encode target and features to integers for CatBoost and Hamming distance
        y_enc = self.label_encoder.fit_transform(y)
        X_enc = self.feature_encoder.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(X_enc, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

        # Train model specifying all features are categorical
        self.model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=0, allow_writing_files=False)
        self.model.fit(X_train, y_train, cat_features=list(range(X_enc.shape[1])), eval_set=(X_val, y_val))

        # Build a separate BallTree for each target class using Hamming distance
        for label in np.unique(y_enc):
            idx = np.where(y_train == label)[0]
            self.trees[label] = BallTree(X_train[idx], metric='hamming')

        self.X_train_enc = X_train
        self.y_train_enc = y_train

    def explain(self, X_test, y_test):
        """
        Extract 1-sparse counterfactuals for feature importance.

        Uses nearest neighbor search to find counterfactuals near the decision boundary.

        Args:
            X_test: Test feature matrix
            y_test: Test target vector

        Returns:
            DataFrame with Sample_ID and extracted Counterfactual_Values
        """
        print("  > Extracting 1-sparse counterfactuals near the decision boundary...")
        X_test_enc = self.feature_encoder.transform(X_test)
        y_test_enc = self.label_encoder.transform(y_test)

        results = {i: set() for i in range(len(X_test_enc))}

        for label in np.unique(y_test_enc):
            idx = np.where(y_test_enc == label)[0]
            if len(idx) == 0:
                continue

            # Query the tree for the opposite class
            opp_label = 1 - label
            tree = self.trees.get(opp_label)
            if tree is None:
                continue

            # Retrieve the k nearest neighbors from the opposite class
            dist, ind = tree.query(X_test_enc[idx], k=self.k_neighbors)

            for i, sample_idx in enumerate(idx):
                orig_sample = X_test_enc[sample_idx]
                neighbors_idx = ind[i]
                neighbors = self.X_train_enc[np.where(self.y_train_enc == opp_label)[0][neighbors_idx]]

                # Compute distances to neighbors
                dists = np.sum(neighbors != orig_sample, axis=1)
                min_dist = np.min(dists)

                # 1-sparse extraction: only counterfactuals differing by exactly 1 feature
                if min_dist != 1:
                    continue

                closest = neighbors[dists == min_dist]

                for cand in closest:
                    diff_mask = cand != orig_sample
                    f_idx = np.where(diff_mask)[0][0]

                    # Use inverse transform to get the original categorical value
                    val_str = self.feature_encoder.categories_[f_idx][cand[f_idx]]
                    results[sample_idx].add(f"{self.feature_names[f_idx]}={val_str}")

        # Format output as a list of itemsets for FP-growth
        rows = [{'Sample_ID': k, 'Counterfactual_Values': list(v)} for k, v in results.items() if v]
        print("    - Extraction completed successfully.\n")
        return pd.DataFrame(rows)


def extract_labels(results_dir):
    """
    Extract only the labels from counterfactual values and save to CSV files.

    Generates two versions:
    - labels_only.csv: Raw labels with possible duplicates
    - labels_only_unique.csv: Deduplicated labels preserving order

    Args:
        results_dir: Path to results directory

    Returns:
        None (saves results to CSV files)
    """
    print("  > Extracting labels from counterfactual values...")

    input_file = results_dir / "transactions_values.csv"
    output_file = results_dir / "labels_only.csv"
    output_file_unique = results_dir / "labels_only_unique.csv"

    df = pd.read_csv(input_file)

    labels_list = []
    labels_unique_list = []

    for idx, row in df.iterrows():
        values_str = str(row['Counterfactual_Values'])
        labels = re.findall(r'([A-Z]\w*)=', values_str)
        labels_list.append(labels)

        # Remove duplicates while preserving order
        seen = set()
        unique_labels = []
        for label in labels:
            if label not in seen:
                seen.add(label)
                unique_labels.append(label)
        labels_unique_list.append(unique_labels)

    # Save labels with duplicates
    output_df = pd.DataFrame({
        'Sample_ID': df['Sample_ID'],
        'Labels': labels_list
    })
    output_df.to_csv(output_file, index=False)
    print(f"    - Labels (with duplicates) saved to {Path(output_file).name}")

    # Save labels without duplicates
    output_df_unique = pd.DataFrame({
        'Sample_ID': df['Sample_ID'],
        'Labels': labels_unique_list
    })
    output_df_unique.to_csv(output_file_unique, index=False)
    print(f"    - Labels (unique only) saved to {Path(output_file_unique).name}")


if __name__ == "__main__":
    # Detect the environment (Local vs Colab) and set paths accordingly
    if Path("/content").exists():
        data_dir = Path("/content/data")
        results_dir = Path("/content/results")
        important_features_dir = Path("/content/results/important_features")
    else:
        base_dir = Path(__file__).resolve().parent.parent
        data_dir = base_dir / "data"
        results_dir = base_dir / "results"
        important_features_dir = results_dir / "important_features"

    # Ensure directories exist
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    important_features_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("COUNTERFACTUAL EXTRACTION FOR FEATURE IMPORTANCE")
    print("="*70 + "\n")

    data_path = data_dir / "ACSIncome_NY_2018_categorized.csv"
    out_path = important_features_dir / "transactions_values.csv"

    if not data_path.exists():
        print(f"  > Error: Required dataset not found at {data_path.name}")
    else:
        # Load data and split
        print("  > Loading and splitting dataset...")
        df = pd.read_csv(data_path)
        X, y = df.drop(columns=['target']), df['target']

        # Use standard split - model evaluated on unseen data
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"    - Training set: {len(X_tr)} samples")
        print(f"    - Test set: {len(X_te)} samples\n")

        # Train explainer and extract counterfactuals
        explainer = CategoricalBoCSoR(k_neighbors=10, perc_threshold=10)
        explainer.fit(X_tr, y_tr)
        transactions = explainer.explain(X_te, y_te)

        transactions.to_csv(out_path, index=False)
        print(f"  > Counterfactual transactions saved to {out_path.name}\n")

        # Extract labels from counterfactuals
        extract_labels(important_features_dir)

    print("="*70)
    print("Execution completed successfully.")
    print("="*70 + "\n")