import sys
from pathlib import Path

# add the scripts folder to the path so imports work regardless of where main.py is called from
sys.path.insert(0, str(Path(__file__).resolve().parent))

from create_dataset import create_region_dataset, categorize_dataset, balance_datasets, REGIONS
from feature_importance import run_for_k_values
from macroscopic_experiment_association_rules import run_k_comparison

import pandas as pd


if __name__ == "__main__":

    if Path("/content").exists():
        base_dir = Path("/content")
    else:
        base_dir = Path(__file__).resolve().parent.parent

    data_dir = base_dir / "data"
    results_dir = base_dir / "results"

    for d in [data_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # -- STEP 1: data preparation ---------------------------------------------
    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION — NORTHEAST vs SOUTH 2018")
    print("="*70 + "\n")

    for region, states in REGIONS.items():
        raw_csv = data_dir / f"ACSIncome_{region}_2018_clean.csv"
        cat_csv = data_dir / f"ACSIncome_{region}_2018_categorized.csv"

        print(f"  > Region: {region.upper()} {states}")

        if not raw_csv.exists():
            df_raw = create_region_dataset(states)
            df_raw.to_csv(raw_csv, index=False)
            print(f"    - raw saved to {raw_csv.name}\n")
        else:
            print(f"    - raw already exists ({raw_csv.name}), skipping.\n")

        if not cat_csv.exists():
            categorize_dataset(raw_csv, cat_csv)
            print()
        else:
            print(f"    - categorized already exists ({cat_csv.name}), skipping.\n")

    # balance the two regions to the same size (stratified on target)
    ne_csv = data_dir / "ACSIncome_northeast_2018_categorized.csv"
    so_csv = data_dir / "ACSIncome_south_2018_categorized.csv"
    ne_bal = data_dir / "ACSIncome_northeast_2018_balanced.csv"
    so_bal = data_dir / "ACSIncome_south_2018_balanced.csv"

    if not ne_bal.exists() or not so_bal.exists():
        print("  > Balancing datasets...")
        df_ne = pd.read_csv(ne_csv)
        df_so = pd.read_csv(so_csv)
        df_ne, df_so = balance_datasets(df_ne, df_so)
        df_ne.to_csv(ne_bal, index=False)
        df_so.to_csv(so_bal, index=False)
        print()
    else:
        print("  > Balanced datasets already exist, skipping.\n")

    # -- STEP 2 & 3: counterfactual extraction + association rules per region -
    region_configs = {
        'northeast': ne_bal,
        'south':     so_bal,
    }

    for region, data_path in region_configs.items():
        important_features_dir = results_dir / region / "important_features"
        ar_output_dir = results_dir / region / "association_rules"

        for d in [important_features_dir, ar_output_dir]:
            d.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print(f"STEP 2: COUNTERFACTUAL EXTRACTION — {region.upper()}")
        print("="*70 + "\n")

        k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        k_labels_map = run_for_k_values(k_values, data_path, important_features_dir)

        print("\n" + "="*70)
        print(f"STEP 3: ASSOCIATION RULES — {region.upper()}")
        print("="*70 + "\n")

        # sup_delta=0.02 but sup_min will be calibrated per k — the fixed
        # floor passed here only applies when auto_calibrate=False
        # conf_min=0.50 and sup_min auto-calibrated per k, but expected to
        # stabilise around 0.10 — justified by ~29k transactions per region
        # (a 10% floor means a pair must appear in ~2,900 transactions)
        run_k_comparison(
            k_labels_map=k_labels_map,
            output_dir=ar_output_dir,
            auto_calibrate=True,
            sup_delta=0.02,
            conf_min=0.50, conf_max=1.00, conf_delta=0.05,
            lift_min=0.0,  lift_delta=0.05,
            lift_neutral_half_window=0.25,
        )

    print("\n" + "="*70)
    print("ALL STEPS COMPLETED")
    print("="*70)
    print(f"\n  > data:             {data_dir}/")
    print(f"  > northeast results:{results_dir}/northeast/")
    print(f"  > south results:    {results_dir}/south/\n")