import sys
from pathlib import Path
from create_dataset import create_ny_2018_dataset, categorize_dataset
from feature_importance import run_for_k_values
from macroscopic_experiment_association_rules import run_k_comparison

# add the scripts folder to the path so imports work regardless of where main.py is called from
sys.path.insert(0, str(Path(__file__).resolve().parent))


if __name__ == "__main__":

    if Path("/content").exists():
        base_dir = Path("/content")
    else:
        base_dir = Path(__file__).resolve().parent.parent

    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    important_features_dir = results_dir / "important_features"
    ar_output_dir = important_features_dir / "association_rules"

    for d in [data_dir, results_dir, important_features_dir, ar_output_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # -- STEP 1: data preparation ---------------------------------------------
    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION")
    print("="*70 + "\n")

    raw_csv = data_dir / "ACSIncome_NY_2018_clean.csv"
    cat_csv = data_dir / "ACSIncome_NY_2018_categorized.csv"

    if not raw_csv.exists():
        df_raw = create_ny_2018_dataset()
        df_raw.to_csv(raw_csv, index=False)
        print(f"  > Raw dataset saved to {raw_csv.name}\n")
    else:
        print(f"  > Raw dataset already exists ({raw_csv.name}), skipping.\n")

    if not cat_csv.exists():
        categorize_dataset(raw_csv, cat_csv)
        print()
    else:
        print(f"  > Categorized dataset already exists ({cat_csv.name}), skipping.\n")

    # -- STEP 2: counterfactual extraction for all k values -------------------
    print("\n" + "="*70)
    print("STEP 2: COUNTERFACTUAL EXTRACTION (k-variation)")
    print("="*70 + "\n")

    k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    k_labels_map = run_for_k_values(k_values, cat_csv, important_features_dir)

    # -- STEP 3: association rules for all k values ---------------------------
    print("\n" + "="*70)
    print("STEP 3: ASSOCIATION RULES — K-VARIATION EXPERIMENT")
    print("="*70 + "\n")

    # auto_calibrate=True recomputes sup_min, sup_max, lift_max for each k
    # since item frequencies change with k — see docs/parameter_rationale.md
    run_k_comparison(
        k_labels_map=k_labels_map,
        output_dir=ar_output_dir,
        auto_calibrate=True,
        sup_delta=0.02,
        conf_min=0.10, conf_max=1.00, conf_delta=0.05,
        lift_min=0.0, lift_delta=0.05,
        lift_neutral_half_window=0.25,
    )

    print("\n" + "="*70)
    print("ALL STEPS COMPLETED")
    print("="*70)
    print(f"\n  > data: {data_dir}/")
    print(f"  > counterfactuals: {important_features_dir}/")
    print(f"  > association rules: {ar_output_dir}/")
    print(f"  > cross-k results: {ar_output_dir}/k_comparison/\n")