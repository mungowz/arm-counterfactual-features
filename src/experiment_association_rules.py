import pandas as pd
import json
from pathlib import Path
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import datetime


def load_transactions(labels_csv_path):
    """
    Load transactions from the labels_only.csv file.
    Each row contains a Sample_ID and its associated Labels (as a string representation of a list).
    Returns a list of lists where each inner list is a transaction (itemset).
    """
    df = pd.read_csv(labels_csv_path)

    transactions = []
    for idx, row in df.iterrows():
        labels_str = str(row['Labels'])
        try:
            labels = eval(labels_str)
            if isinstance(labels, list):
                transactions.append(labels)
        except:
            continue

    return transactions


def run_experiment(transactions, min_support, min_confidence, min_lift, experiment_name, output_dir):
    """
    Run a single association rules mining experiment with given parameters.

    Args:
        transactions: List of lists representing transactions
        min_support: Minimum support threshold
        min_confidence: Minimum confidence threshold
        min_lift: Minimum lift threshold
        experiment_name: Name of the experiment
        output_dir: Directory to save results

    Returns:
        Dictionary with experiment results and statistics
    """
    print(f"\n{'='*70}")
    print(f"Experiment: {experiment_name}")
    print(f"Params: support={min_support}, confidence={min_confidence}, lift={min_lift}")
    print(f"{'='*70}")

    # Transform transactions into one-hot encoded DataFrame
    print("  > Encoding transactions...")
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # Apply FP-Growth algorithm
    print("  > Running FP-Growth algorithm...")
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    print(f"  > Found {len(frequent_itemsets)} frequent itemsets")

    # Generate association rules
    if len(frequent_itemsets) == 0:
        print("  > Warning: No frequent itemsets found")
        return {
            'experiment_name': experiment_name,
            'min_support': min_support,
            'min_confidence': min_confidence,
            'min_lift': min_lift,
            'num_frequent_itemsets': 0,
            'num_rules': 0
        }

    print("  > Generating association rules...")
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    # Filter by lift
    rules = rules[rules['lift'] >= min_lift]
    rules = rules.sort_values('lift', ascending=False)

    print(f"  > Generated {len(rules)} association rules")

    # Create experiment directory
    exp_dir = Path(output_dir) / f"{experiment_name.replace(' ', '_').replace('=', '')}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    if len(rules) > 0:
        # Formatted output
        formatted_rules = pd.DataFrame()
        formatted_rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        formatted_rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        formatted_rules['support'] = rules['support'].round(4)
        formatted_rules['confidence'] = rules['confidence'].round(4)
        formatted_rules['lift'] = rules['lift'].round(4)

        formatted_rules.to_csv(exp_dir / "rules.csv", index=False)

        # Detailed output
        detailed_rules = pd.DataFrame()
        detailed_rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        detailed_rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        detailed_rules['support'] = rules['support']
        detailed_rules['confidence'] = rules['confidence']
        detailed_rules['lift'] = rules['lift']
        detailed_rules['leverage'] = rules['leverage']
        detailed_rules['conviction'] = rules['conviction']

        detailed_rules.to_csv(exp_dir / "rules_detailed.csv", index=False)

        # Summary
        with open(exp_dir / "summary.txt", 'w') as f:
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Parameters:\n")
            f.write(f"  Min Support: {min_support}\n")
            f.write(f"  Min Confidence: {min_confidence}\n")
            f.write(f"  Min Lift: {min_lift}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Frequent Itemsets: {len(frequent_itemsets)}\n")
            f.write(f"  Association Rules: {len(rules)}\n\n")
            f.write(f"Statistics:\n")
            f.write(f"  Avg Support: {rules['support'].mean():.4f}\n")
            f.write(f"  Avg Confidence: {rules['confidence'].mean():.4f}\n")
            f.write(f"  Avg Lift: {rules['lift'].mean():.4f}\n")
            f.write(f"  Lift Range: {rules['lift'].min():.4f} - {rules['lift'].max():.4f}\n\n")
            f.write(f"Top 10 Rules (by Lift):\n")
            f.write(f"{'-'*60}\n")
            for idx, row in formatted_rules.head(10).iterrows():
                f.write(f"{idx+1}. {row['antecedents']} => {row['consequents']}\n")
                f.write(f"   Support: {row['support']:.4f} | Confidence: {row['confidence']:.4f} | Lift: {row['lift']:.4f}\n\n")

        print(f"  > Results saved to: {exp_dir.name}/")

    # Return statistics
    result = {
        'experiment_name': experiment_name,
        'min_support': min_support,
        'min_confidence': min_confidence,
        'min_lift': min_lift,
        'num_frequent_itemsets': len(frequent_itemsets),
        'num_rules': len(rules),
    }

    if len(rules) > 0:
        result.update({
            'avg_support': rules['support'].mean(),
            'avg_confidence': rules['confidence'].mean(),
            'avg_lift': rules['lift'].mean(),
            'min_lift_value': rules['lift'].min(),
            'max_lift_value': rules['lift'].max(),
        })

    return result


def load_configs(config_path):
    """Load experiment configurations from JSON file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config['experiments']


def save_comparison_report(results, output_dir):
    """Save a comparison report of all experiments."""
    report_path = Path(output_dir) / "experiments_comparison.csv"

    df = pd.DataFrame(results)
    df.to_csv(report_path, index=False)

    print(f"\n{'='*70}")
    print("Comparison Report")
    print(f"{'='*70}\n")
    print(df.to_string(index=False))

    # Summary statistics
    summary_path = Path(output_dir) / "experiments_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("EXPERIMENTS COMPARISON SUMMARY\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total Experiments: {len(results)}\n\n")
        f.write("Results:\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write("Insights:\n")
        f.write(f"  - Most rules generated: {df.loc[df['num_rules'].idxmax(), 'experiment_name']} ({df['num_rules'].max()} rules)\n")
        f.write(f"  - Fewest rules generated: {df.loc[df['num_rules'].idxmin(), 'experiment_name']} ({df['num_rules'].min()} rules)\n")
        if 'avg_lift' in df.columns:
            f.write(f"  - Highest avg lift: {df.loc[df['avg_lift'].idxmax(), 'experiment_name']} ({df['avg_lift'].max():.4f})\n")

    print(f"\n  > Comparison saved to: {report_path.name}")
    print(f"  > Summary saved to: {summary_path.name}\n")


if __name__ == "__main__":
    # Detect the environment (Local vs Colab) and set paths accordingly
    if Path("/content").exists():
        base_dir = Path("/content")
    else:
        base_dir = Path(__file__).resolve().parent.parent

    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    config_path = Path(__file__).resolve().parent / "experiment_configs.json"

    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("ASSOCIATION RULES MINING - EXPERIMENTS")
    print("="*70)

    labels_csv = results_dir / "labels_only_unique.csv"

    if not labels_csv.exists():
        print(f"  > Error: Labels file not found at {labels_csv}")
    elif not config_path.exists():
        print(f"  > Error: Configuration file not found at {config_path}")
        print(f"  > Please create {config_path.name} with your experiment configurations")
    else:
        # Load transactions
        print(f"\n  > Loading transactions from {labels_csv.name}...")
        transactions = load_transactions(labels_csv)
        print(f"  > Loaded {len(transactions)} transactions")

        if len(transactions) == 0:
            print("  > No valid transactions found")
        else:
            # Load experiment configurations
            print(f"\n  > Loading configurations from {config_path.name}...")
            try:
                experiments = load_configs(config_path)
                print(f"  > Found {len(experiments)} experiment(s) to run")
            except json.JSONDecodeError as e:
                print(f"  > Error parsing configuration file: {e}")
                exit(1)

            # Create experiments output directory
            exp_output_dir = results_dir / "experiments"
            exp_output_dir.mkdir(parents=True, exist_ok=True)

            # Run experiments
            results = []
            for exp in experiments:
                result = run_experiment(
                    transactions,
                    min_support=exp['min_support'],
                    min_confidence=exp['min_confidence'],
                    min_lift=exp['min_lift'],
                    experiment_name=exp['name'],
                    output_dir=exp_output_dir
                )
                results.append(result)

            # Save comparison report
            save_comparison_report(results, exp_output_dir)

    print("\nExecution completed successfully.\n")
