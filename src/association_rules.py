import pandas as pd
import numpy as np
from pathlib import Path
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder


def load_transactions(labels_csv_path):
    """
    Load transactions from the labels_only.csv file.
    Each row contains a Sample_ID and its associated Labels (as a string representation of a list).
    Returns a list of lists where each inner list is a transaction (itemset).
    """
    print(f"  > Loading transactions from {Path(labels_csv_path).name}...")
    df = pd.read_csv(labels_csv_path)

    transactions = []
    for idx, row in df.iterrows():
        labels_str = str(row['Labels'])
        # Parse the string representation of list (e.g., "['SCHL', 'COW']")
        try:
            # Use eval to safely convert string representation to list
            # In production, consider using ast.literal_eval instead
            labels = eval(labels_str)
            if isinstance(labels, list):
                transactions.append(labels)
        except:
            # Skip rows that can't be parsed
            continue

    print(f"  > Loaded {len(transactions)} transactions")
    return transactions


def generate_frequent_itemsets(transactions, min_support=0.05):
    """
    Generate frequent itemsets using FP-Growth algorithm.

    Args:
        transactions: List of lists, where each inner list is a transaction
        min_support: Minimum support threshold (default 5%)

    Returns:
        DataFrame with frequent itemsets and their support values
    """
    print(f"  > Generating frequent itemsets with min_support={min_support}...")

    # Transform transactions into one-hot encoded DataFrame
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # Apply FP-Growth algorithm
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)

    if len(frequent_itemsets) == 0:
        print("  > Warning: No frequent itemsets found with the specified min_support")
    else:
        print(f"  > Found {len(frequent_itemsets)} frequent itemsets")

    return frequent_itemsets


def generate_association_rules(frequent_itemsets, min_confidence=0.5, min_lift=1.0):
    """
    Generate association rules from frequent itemsets.

    Args:
        frequent_itemsets: DataFrame of frequent itemsets from fpgrowth
        min_confidence: Minimum confidence threshold (default 50%)
        min_lift: Minimum lift threshold (default 1.0)

    Returns:
        DataFrame with association rules and their metrics
    """
    print(f"  > Generating association rules...")
    print(f"    - min_confidence={min_confidence}, min_lift={min_lift}")

    if len(frequent_itemsets) == 0:
        print("  > No frequent itemsets to generate rules from")
        return pd.DataFrame()

    # Generate rules with support, confidence, and lift metrics
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    if len(rules) == 0:
        print("  > Warning: No rules generated with the specified min_confidence")
        return pd.DataFrame()

    # Filter by lift threshold
    rules = rules[rules['lift'] >= min_lift]

    # Sort rules by lift in descending order
    rules = rules.sort_values('lift', ascending=False)

    print(f"  > Generated {len(rules)} association rules")

    return rules


def format_rules_output(rules):
    """
    Format association rules for better readability.

    Args:
        rules: DataFrame with association rules

    Returns:
        DataFrame with formatted and renamed columns
    """
    if len(rules) == 0:
        return pd.DataFrame()

    # Create a more readable version of the rules
    formatted_rules = pd.DataFrame()
    formatted_rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    formatted_rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    formatted_rules['support'] = rules['support'].round(4)
    formatted_rules['confidence'] = rules['confidence'].round(4)
    formatted_rules['lift'] = rules['lift'].round(4)

    return formatted_rules


def export_results(rules, formatted_rules, output_dir, min_support, min_confidence, min_lift):
    """
    Export results to CSV files.

    Args:
        rules: Original rules DataFrame from association_rules
        formatted_rules: Formatted rules DataFrame
        output_dir: Path to output directory
        min_support: min_support threshold used
        min_confidence: min_confidence threshold used
        min_lift: min_lift threshold used
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save formatted rules (human-readable)
    formatted_output = output_dir / "association_rules.csv"
    formatted_rules.to_csv(formatted_output, index=False)
    print(f"  > Formatted rules saved to {formatted_output.name}")

    # Save detailed rules with all metrics from mlxtend
    detailed_output = output_dir / "association_rules_detailed.csv"

    if len(rules) > 0:
        detailed_rules = pd.DataFrame()
        detailed_rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        detailed_rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        detailed_rules['support'] = rules['support']
        detailed_rules['confidence'] = rules['confidence']
        detailed_rules['lift'] = rules['lift']
        detailed_rules['leverage'] = rules['leverage']
        detailed_rules['conviction'] = rules['conviction']

        detailed_rules.to_csv(detailed_output, index=False)
        print(f"  > Detailed rules saved to {detailed_output.name}")

    # Save summary statistics
    summary_output = output_dir / "association_rules_summary.txt"
    with open(summary_output, 'w') as f:
        f.write("=== Association Rules Summary ===\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  - Min Support: {min_support}\n")
        f.write(f"  - Min Confidence: {min_confidence}\n")
        f.write(f"  - Min Lift: {min_lift}\n\n")
        f.write(f"Results:\n")
        f.write(f"  - Total Rules Generated: {len(formatted_rules)}\n")

        if len(formatted_rules) > 0:
            f.write(f"\nRules Statistics:\n")
            f.write(f"  - Average Support: {rules['support'].mean():.4f}\n")
            f.write(f"  - Average Confidence: {rules['confidence'].mean():.4f}\n")
            f.write(f"  - Average Lift: {rules['lift'].mean():.4f}\n")
            f.write(f"  - Min/Max Lift: {rules['lift'].min():.4f} / {rules['lift'].max():.4f}\n")

        f.write(f"\nTop 10 Rules (by Lift):\n")
        for idx, row in formatted_rules.head(10).iterrows():
            f.write(f"  {idx+1}. {row['antecedents']} -> {row['consequents']}\n")
            f.write(f"     Support: {row['support']:.4f}, Confidence: {row['confidence']:.4f}, Lift: {row['lift']:.4f}\n")

    print(f"  > Summary saved to {summary_output.name}")


if __name__ == "__main__":
    # Detect the environment (Local vs Colab) and set paths accordingly
    if Path("/content").exists():
        results_dir = Path("/content/results")
    else:
        base_dir = Path(__file__).resolve().parent.parent
        results_dir = base_dir / "results"

    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Association Rules Mining (FP-Growth) ---\n")

    labels_csv = results_dir / "labels_only_unique.csv"

    if not labels_csv.exists():
        print(f"  > Error: Labels file not found at {labels_csv}")
    else:
        # Load transactions
        transactions = load_transactions(labels_csv)

        if len(transactions) == 0:
            print("  > No valid transactions found")
        else:
            # Parameters for association rule mining
            min_support = 0.05  # 5% minimum support
            min_confidence = 0.5  # 50% minimum confidence
            min_lift = 1.0  # Minimum lift threshold

            # Generate frequent itemsets
            frequent_itemsets = generate_frequent_itemsets(transactions, min_support=min_support)

            if len(frequent_itemsets) > 0:
                # Generate association rules
                rules = generate_association_rules(
                    frequent_itemsets,
                    min_confidence=min_confidence,
                    min_lift=min_lift
                )

                if len(rules) > 0:
                    # Format and display rules
                    formatted_rules = format_rules_output(rules)

                    print("\n--- Top 10 Association Rules ---\n")
                    print(formatted_rules.head(10).to_string(index=False))

                    # Export results
                    export_results(rules, formatted_rules, results_dir, min_support, min_confidence, min_lift)
                else:
                    print("  > No association rules generated with specified thresholds")
            else:
                print("  > No frequent itemsets generated")

    print("\nExecution completed successfully.\n")
