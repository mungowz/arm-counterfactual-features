import pandas as pd
import numpy as np
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
        print("Training CatBoost and building BallTrees...")
        self.feature_names = X.columns.tolist()
        
        # Encode target and features to int for CatBoost and Hamming distance
        y_enc = self.label_encoder.fit_transform(y)
        X_enc = self.feature_encoder.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(X_enc, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

        # Train model specifying all features are categorical
        self.model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=0, allow_writing_files=False)
        self.model.fit(X_train, y_train, cat_features=list(range(X_train.shape[1])))

        # Build a separate tree for each class to quickly find opposite-class neighbors
        for cls in np.unique(y_enc):
            self.trees[cls] = BallTree(X_train[y_train == cls], metric='hamming')

        self.X_train_encoded = X_train
        self.y_train_encoded = y_train
        return self

    def explain(self, X_test, y_test):
        print("Extracting 1-sparse counterfactuals near the decision boundary...")
        X_enc = self.feature_encoder.transform(X_test).astype(int)
        y_enc = self.label_encoder.transform(y_test)

        # Compute distance to the closest counterfactual to identify boundary samples
        dists = []
        for i in range(len(X_enc)):
            target_cls = 1 - y_enc[i]
            d, _ = self.trees[target_cls].query(X_enc[i].reshape(1,-1), k=1)
            dists.append(d[0][0])

        # Filter instances based on the percentile threshold
        threshold = np.percentile(dists, self.perc_threshold)
        boundary_idx = [i for i, d in enumerate(dists) if d <= threshold]

        results = {}
        for idx in boundary_idx:
            sample = X_enc[idx].copy()
            orig_label = y_enc[idx]
            target_cls = 1 - orig_label

            # Get k neighbors from the target class
            _, n_idx = self.trees[target_cls].query(sample.reshape(1,-1), k=self.k_neighbors)
            candidates = self.X_train_encoded[self.y_train_encoded == target_cls][n_idx[0]]

            results[idx] = set()
            for cand in candidates:
                diff_feats = np.where(sample != cand)[0]
                
                # Test single feature swaps to see if they cross the boundary
                for f_idx in diff_feats:
                    test_sample = sample.copy()
                    test_sample[f_idx] = cand[f_idx]
                    
                    if self.model.predict(test_sample.reshape(1,-1))[0] != orig_label:
                        # Revert the integer code to the original string label
                        val_str = self.feature_encoder.categories_[f_idx][cand[f_idx]]
                        results[idx].add(f"{self.feature_names[f_idx]}={val_str}")

        # Format output as a list of itemsets for FP-growth
        rows = [{'Sample_ID': k, 'Counterfactual_Values': list(v)} for k, v in results.items() if v]
        return pd.DataFrame(rows)


if __name__ == "__main__":
    current_dir = Path.cwd()
    data_dir = current_dir / "data"
    if not data_dir.exists(): 
        data_dir = Path("/content/data")

    df = pd.read_csv(data_dir / "ACSIncome_NY_2018_categorized.csv")
    X, y = df.drop(columns=['target']), df['target']

    # Using standard split, the model is evaluated on unseen data
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    explainer = CategoricalBoCSoR().fit(X_tr, y_tr)
    transactions = explainer.explain(X_te, y_te)

    output_path = data_dir / "transactions_values.csv"
    transactions.to_csv(output_path, index=False)
    print(f"Counterfactual transactions extracted and saved to {output_path}")