import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree

df = pd.read_csv('data/ACSIncome_northeast_2018_balanced.csv')
X, y = df.drop(columns=['target']), df['target']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

enc = OrdinalEncoder(dtype=int)
X_enc = enc.fit_transform(X_tr)
le = LabelEncoder()
y_enc = le.fit_transform(y_tr)

for label in np.unique(y_enc):
    pos_idx = np.where(y_enc == label)[0]
    opp_label = 1 - label
    opp_idx = np.where(y_enc == opp_label)[0]
    tree = BallTree(X_enc[opp_idx], metric='hamming')

    min_dist, _ = tree.query(X_enc[pos_idx], k=1)
    min_dist = min_dist.ravel()
    thresh = np.percentile(min_dist, 10)
    boundary = pos_idx[min_dist <= thresh]

    _, ind = tree.query(X_enc[boundary], k=5)

    all_diffs = []
    for i, si in enumerate(boundary[:500]):
        for j in range(5):
            cf_idx = opp_idx[ind[i, j]]
            n_diff = int(np.sum(X_enc[si] != X_enc[cf_idx]))
            all_diffs.append(n_diff)

    all_diffs = np.array(all_diffs)
    vals, counts = np.unique(all_diffs, return_counts=True)
    dist_dict = dict(zip(vals.tolist(), counts.tolist()))
    print(f"Classe {label} -> CF classe {opp_label}:")
    print(f"  Feature diverse per coppia: min={all_diffs.min()} max={all_diffs.max()} media={all_diffs.mean():.2f}")
    print(f"  Distribuzione: {dist_dict}")
    print()
