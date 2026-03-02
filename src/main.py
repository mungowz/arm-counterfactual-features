# Necessary libraries for the pipeline. Ensure these are installed before running the scripts.
# !pip install catboost folktables scikit-learn pandas numpy mlxtend

import os
import warnings
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import create_dataset
import feature_importance
import data_mining

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")

def main():
    # Set up paths for both local and Colab environments
    if Path("/content").exists():
        base_dir = Path("/content")
    else:
        base_dir = Path(__file__).resolve().parent.parent
        
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"

    # Ensure directories exist
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    raw_csv = data_dir / "ACSIncome_NY_2018_clean.csv"
    cat_csv = data_dir / "ACSIncome_NY_2018_categorized.csv"
    transactions_file = results_dir / "transactions_values.csv"
    rules_file = results_dir / "microscopic_level_association_rules.csv"

    print("--- Data Preparation ---")
    if not raw_csv.exists():
        df_raw = create_dataset.create_ny_2018_dataset()
        df_raw.to_csv(raw_csv, index=False)
    else:
        print(f"Raw dataset already exists at {raw_csv}")
    
    if not cat_csv.exists():
        create_dataset.categorize_dataset(raw_csv, cat_csv)
    else:
        print(f"Categorized dataset already exists at {cat_csv}")

    print("\n--- Counterfactual Extraction ---")
    if not transactions_file.exists():
        df = pd.read_csv(cat_csv)
        X, y = df.drop(columns=['target']), df['target']
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        explainer = feature_importance.CategoricalBoCSoR(k_neighbors=10, perc_threshold=10)
        explainer.fit(X_tr, y_tr)
        transactions = explainer.explain(X_te, y_te)
        
        transactions.to_csv(transactions_file, index=False)
        print(f"Counterfactual transactions saved to {transactions_file}")
    else:
        print(f"Transactions already exist at {transactions_file}")

    print("\n--- Data Mining & Fairness Audit ---\n")
    
    # 1. Microscopic Data Mining (Label=Value)
    print("[1/3] Microscopic Analysis (Label=Value)")
    miner = data_mining.InteractionMiner(min_support=0.01, min_confidence=0.1)
    miner.execute(transactions_file, results_dir)

    # 2. Macroscopic Data Mining (Quantitative & Weighted Label Multiplicity)
    print("[2/3] Macroscopic Analysis (Label:Count)")
    macro_miner = data_mining.MacroscopicMiner(min_support=0.01, min_confidence=0.1, min_w_support=0.005)
    macro_miner.execute(transactions_file, results_dir)

    # 3. Audits and Visualizations
    if rules_file.exists():
        print("[3/3] Fairness & Bias Audit")
        sens_features = ['SEX', 'RAC1P']

        # Direct bias audit
        auditor_sens = data_mining.SensitiveAuditMiner(sensitive_features=sens_features)
        if auditor_sens.load_rules(rules_file): 
            auditor_sens.run_audit(results_dir)

        # Conditional fairness audit
        auditor_fair = data_mining.FairnessAuditor(sensitive_features=sens_features)
        fairness_report = auditor_fair.audit_rules(rules_file, cat_csv)
        
        if not fairness_report.empty:
            fairness_report = fairness_report.sort_values(by='Conf_Difference', ascending=False)
            fairness_report.to_csv(results_dir / "fairness_audit_results.csv", index=False)
            print("    - Fairness audit report saved.")
            
            data_mining.FairnessVisualizer.plot_bias_barchart(fairness_report, results_dir, top_n=10)

        # Proxy variable detection
        proxy_detector = data_mining.SensitiveProxyDetector(sensitive_features=sens_features)
        proxy_report = proxy_detector.detect_proxies(rules_file, cat_csv, lift_threshold=1.5)
        
        if not proxy_report.empty:
            proxy_report.to_csv(results_dir / "proxy_variables_detected.csv", index=False)
            print("    - Proxy variables report saved.")

    print("\nPipeline execution completed successfully.")

if __name__ == "__main__":
    main()