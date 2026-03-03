import ast
import datetime
import shutil
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

matplotlib.use('Agg')  # Non-interactive backend — safe for script execution without a display


def extract_labels(labels_only_path):
    """
    Extract and one-hot encode labels from counterfactual transactions.
    Each row contains a Sample_ID and its associated Labels (as a string representation of a list).

    Args:
        labels_only_path: Path to the labels-only CSV file.

    Returns:
        DataFrame with one-hot encoded labels for association rule mining.
    """
    print("  > Extracting labels from counterfactual transactions...")
    df = pd.read_csv(labels_only_path)
    itemsets_lists = df['Labels'].apply(ast.literal_eval)

    # One-hot encode itemsets for FP-Growth
    te = TransactionEncoder()
    te_ary = te.fit(itemsets_lists).transform(itemsets_lists)
    encoded_df = pd.DataFrame(te_ary, columns=te.columns_)

    print("  > Sample of encoded labels:")
    print(encoded_df.head())
    print("-" * 50)

    return encoded_df


def cleanup_empty_folders(output_dir):
    """
    Remove conf_* folders that contain no rules after filtering, then remove
    sup_* folders that are left with no conf_* subfolders.
    Called automatically at the end of explore_association_rules().

    Args:
        output_dir: Root folder of the exploration output (same as passed to explore_association_rules()).

    Returns:
        Tuple (removed_conf, removed_sup) with the counts of deleted folders.
    """
    output_dir = Path(output_dir)
    removed_conf = 0
    removed_sup  = 0

    for sup_dir in sorted(output_dir.glob("sup_*")):
        if not sup_dir.is_dir():
            continue

        for conf_dir in sorted(sup_dir.glob("conf_*")):
            if not conf_dir.is_dir():
                continue

            # A conf folder is considered empty if rules.csv is missing or has no data rows
            rules_csv = conf_dir / "rules.csv"
            is_empty  = (
                not rules_csv.exists()
                or pd.read_csv(rules_csv).empty
            )

            if is_empty:
                shutil.rmtree(conf_dir)
                removed_conf += 1

        # Remove the sup folder if no conf subfolders remain
        remaining_conf = list(sup_dir.glob("conf_*"))
        if not remaining_conf:
            shutil.rmtree(sup_dir)
            removed_sup += 1

    return removed_conf, removed_sup


def grid_search_fpgrowth_delta(df, sup_min, sup_max, sup_delta,
                               conf_min, conf_max, conf_delta,
                               lift_min, lift_max, lift_delta):
    """
    Perform a grid search over support, confidence, and lift for FP-Growth association rule mining.
    Returns an in-memory summary DataFrame without saving any files.
    Use explore_association_rules() for the full exploration with file output.

    Args:
        df: DataFrame with one-hot encoded itemsets.
        sup_min, sup_max, sup_delta: Min, max, and step for support.
        conf_min, conf_max, conf_delta: Min, max, and step for confidence.
        lift_min, lift_max, lift_delta: Min, max, and step for lift.

    Returns:
        DataFrame summarizing the number of rules, max lift, and mean confidence
        for each combination of parameters, sorted by number of rules and lift descending.
    """
    print(f"\n{'='*70}")
    print("GRID SEARCH: FP-GROWTH ASSOCIATION RULES (IN-MEMORY)")
    print(f"{'='*70}")

    # Build grids — add half-delta to stop so the max value is always included
    support_grid = np.round(np.arange(sup_min,  sup_max  + sup_delta  / 2, sup_delta),  4)
    confidence_grid = np.round(np.arange(conf_min, conf_max + conf_delta / 2, conf_delta), 4)
    lift_grid = np.round(np.arange(lift_min, lift_max + lift_delta / 2, lift_delta), 4)

    print(f"  > Support grid: {support_grid}")
    print(f"  > Confidence grid: {confidence_grid}")
    print(f"  > Lift grid: {lift_grid}")
    print("-" * 50)

    results = []

    for min_sup in support_grid:

        frequent_itemsets = fpgrowth(df, min_support=min_sup, use_colnames=True)

        if len(frequent_itemsets) == 0:
            continue

        for min_conf in confidence_grid:

            try:
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
            except ValueError:
                continue

            if len(rules) == 0:
                continue

            for min_lift in lift_grid:

                filtered  = rules[rules['lift'] >= min_lift]
                num_rules = len(filtered)

                results.append({
                    'Support': min_sup,
                    'Confidence': min_conf,
                    'Lift': min_lift,
                    'Number_of_Rules': num_rules,
                    'Max_Lift': round(filtered['lift'].max(), 4) if num_rules > 0 else 0,
                    'Mean_Confidence': round(filtered['confidence'].mean(), 4) if num_rules > 0 else 0,
                })

    print("  > Grid search completed.")

    df_results = pd.DataFrame(results)

    if not df_results.empty:
        df_results = df_results.sort_values(
            by=['Number_of_Rules', 'Lift'], ascending=[False, False]
        )

    return df_results


def explore_association_rules(df, output_dir,
                              sup_min, sup_max, sup_delta,
                              conf_min, conf_max, conf_delta,
                              lift_min, lift_max, lift_delta,
                              lift_neutral_half_window=0.25):
    """
    Exhaustive grid search over support, confidence, and lift for FP-Growth
    association rule mining, saving detailed results to disk for every combination.

    Rules whose lift falls within the neutral window [1 - lift_neutral_half_window,
    1 + lift_neutral_half_window] are excluded before saving, as they indicate
    near-statistical-independence and carry no associative signal.
    After generation, empty folders (no rules surviving the filter) are automatically
    removed by cleanup_empty_folders().†

    Args:
        df: One-hot encoded DataFrame from extract_labels().
        output_dir: Root folder where all output will be written.
        sup_min, sup_max, sup_delta: Min, max, and step for support.
        conf_min, conf_max, conf_delta: Min, max, and step for confidence.
        lift_min, lift_max, lift_delta: Min, max, and step for lift.
        lift_neutral_half_window: Half-width of the neutral lift window to exclude.
            Rules with lift in [1 - half_window, 1 + half_window] are dropped.
            Default is 0.25, producing a window of [0.75, 1.25].

    Returns:
        Summary DataFrame (also saved to output_dir/summary.csv).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build grids — add half-delta to stop so the max value is always included
    support_grid = np.round(np.arange(sup_min,  sup_max  + sup_delta  / 2, sup_delta),  4)
    confidence_grid = np.round(np.arange(conf_min, conf_max + conf_delta / 2, conf_delta), 4)
    lift_grid = np.round(np.arange(lift_min, lift_max + lift_delta / 2, lift_delta), 4)

    total_combinations = len(support_grid) * len(confidence_grid) * len(lift_grid)
    lift_window_lo = round(1.0 - lift_neutral_half_window, 4)
    lift_window_hi = round(1.0 + lift_neutral_half_window, 4)

    print(f"\n{'='*70}")
    print("FULL EXPLORATION: FP-GROWTH ASSOCIATION RULES")
    print(f"{'='*70}")
    print(f"  > Support: {len(support_grid)} values  [{support_grid[0]} ... {support_grid[-1]}, step={sup_delta}]")
    print(f"  > Confidence: {len(confidence_grid)} values  [{confidence_grid[0]} ... {confidence_grid[-1]}, step={conf_delta}]")
    print(f"  > Lift: {len(lift_grid)} values  [{lift_grid[0]} ... {lift_grid[-1]}, step={lift_delta}]")
    print(f"  > Neutral lift window (excluded): [{lift_window_lo}, {lift_window_hi}]")
    print(f"  > Max combinations to evaluate: {total_combinations:,}")
    print("-" * 50)

    summary_rows = []
    total_sup = len(support_grid)

    for sup_idx, min_sup in enumerate(support_grid, start=1):

        sup_label = f"{min_sup:.2f}"
        sup_dir = output_dir / f"sup_{sup_label}"
        sup_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{sup_idx}/{total_sup}] Support = {min_sup}")

        # Find frequent itemsets — these depend only on the support threshold
        print("    > Running FP-Growth algorithm...")
        frequent_itemsets = fpgrowth(df, min_support=min_sup, use_colnames=True)
        print(f"    > Found {len(frequent_itemsets)} frequent itemsets")

        if len(frequent_itemsets) == 0:
            print("    > Warning: No frequent itemsets found — skipping this support value.")
            continue

        # Enrich frequent itemsets with string representation and length
        fi = frequent_itemsets.copy()
        fi['itemset_str'] = fi['itemsets'].apply(lambda x: ', '.join(sorted(x)))
        fi['itemset_length'] = fi['itemsets'].apply(len)
        fi = fi[['itemset_str', 'itemset_length', 'support']]
        fi = fi.sort_values(by=['itemset_length', 'support'], ascending=[True, False])

        # Save frequent itemsets CSV
        fi.to_csv(sup_dir / "frequent_itemsets.csv", index=False)

        # Count itemsets by length for reporting
        itemsets_by_len = fi['itemset_length'].value_counts().sort_index().to_dict()
        len_summary = ', '.join([f"len={k}: {v}" for k, v in itemsets_by_len.items()])
        print(f"    > Itemset breakdown: {len_summary}")

        # Save frequent itemsets summary .txt
        with open(sup_dir / "frequent_itemsets_summary.txt", 'w') as f:
            f.write(f"Frequent Itemsets Summary\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Parameters:\n")
            f.write(f"  Min Support: {min_sup}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Total Frequent Itemsets: {len(frequent_itemsets)}\n")
            for length, count in itemsets_by_len.items():
                f.write(f"  Itemsets of length {length}: {count}\n")
            f.write(f"\nAll Frequent Itemsets (sorted by length, then support desc):\n")
            f.write(f"{'-'*60}\n")
            for _, row in fi.iterrows():
                f.write(f"  [{row['itemset_str']}]  support={row['support']:.4f}  length={row['itemset_length']}\n")

        # Generate association rules for each confidence threshold
        for min_conf in confidence_grid:

            conf_label = f"{min_conf:.2f}"
            conf_dir = sup_dir / f"conf_{conf_label}"

            try:
                rules = association_rules(
                    frequent_itemsets,
                    metric="confidence",
                    min_threshold=min_conf
                )
            except ValueError:
                # mlxtend raises ValueError when no rules can be generated
                continue

            if len(rules) == 0:
                continue

            # Remove rules whose lift falls inside the neutral window [lo, hi]:
            # these indicate near-statistical-independence and carry no associative signal.
            rules = rules[
                (rules['lift'] < lift_window_lo) | (rules['lift'] > lift_window_hi)
            ]

            if len(rules) == 0:
                continue

            rules = rules.sort_values('lift', ascending=False)

            # Build compact rules DataFrame
            formatted_rules = pd.DataFrame()
            formatted_rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
            formatted_rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
            formatted_rules['support'] = rules['support'].round(4)
            formatted_rules['confidence'] = rules['confidence'].round(4)
            formatted_rules['lift'] = rules['lift'].round(4)

            # Build detailed rules DataFrame with full metrics and itemset lengths
            detailed_rules = pd.DataFrame()
            detailed_rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
            detailed_rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
            detailed_rules['antecedent_length'] = rules['antecedents'].apply(len)
            detailed_rules['consequent_length'] = rules['consequents'].apply(len)
            detailed_rules['rule_length'] = detailed_rules['antecedent_length'] + detailed_rules['consequent_length']
            detailed_rules['antecedent support'] = rules['antecedent support'].values
            detailed_rules['consequent support'] = rules['consequent support'].values
            detailed_rules['support'] = rules['support'].values
            detailed_rules['confidence'] = rules['confidence'].values
            detailed_rules['lift'] = rules['lift'].values
            detailed_rules['leverage'] = rules['leverage'].values
            detailed_rules['conviction'] = rules['conviction'].values

            conf_dir.mkdir(parents=True, exist_ok=True)
            formatted_rules.to_csv(conf_dir / "rules.csv", index=False)
            detailed_rules.to_csv(conf_dir / "rules_detailed.csv", index=False)

            # Save per-(sup, conf) summary .txt
            with open(conf_dir / "summary.txt", 'w') as f:
                f.write(f"Association Rules Summary\n")
                f.write(f"{'='*60}\n\n")
                f.write(f"Parameters:\n")
                f.write(f"  Min Support: {min_sup}\n")
                f.write(f"  Min Confidence: {min_conf}\n")
                f.write(f"  Neutral Lift Window (excluded): [{lift_window_lo}, {lift_window_hi}]\n\n")
                f.write(f"Results:\n")
                f.write(f"  Frequent Itemsets: {len(frequent_itemsets)}\n")
                f.write(f"  Association Rules: {len(rules)}\n\n")
                f.write(f"Statistics:\n")
                f.write(f"  Avg Support: {rules['support'].mean():.4f}\n")
                f.write(f"  Avg Confidence: {rules['confidence'].mean():.4f}\n")
                f.write(f"  Avg Lift: {rules['lift'].mean():.4f}\n")
                f.write(f"  Lift Range: {rules['lift'].min():.4f} - {rules['lift'].max():.4f}\n")
                f.write(f"  Avg Rule Length: {detailed_rules['rule_length'].mean():.2f}\n\n")
                f.write(f"Top 10 Rules (by Lift):\n")
                f.write(f"{'-'*60}\n")
                for idx, row in formatted_rules.head(10).iterrows():
                    f.write(f"{idx + 1}. {row['antecedents']} => {row['consequents']}\n")
                    f.write(f"   Support: {row['support']:.4f} | Confidence: {row['confidence']:.4f} | Lift: {row['lift']:.4f}\n\n")

            print(f"    > [conf={min_conf}] {len(rules)} rules — saved to {conf_dir.relative_to(output_dir)}/")

            # Accumulate one summary row per lift threshold
            for min_lift in lift_grid:

                filtered  = rules[rules['lift'] >= min_lift]
                num_rules = len(filtered)

                rule_lengths = (
                    filtered['antecedents'].apply(len) + filtered['consequents'].apply(len)
                ) if num_rules > 0 else pd.Series(dtype=float)

                summary_rows.append({
                    'Support': min_sup,
                    'Confidence': min_conf,
                    'Lift_threshold': min_lift,
                    'Number_of_Rules': num_rules,
                    'Max_Lift': round(filtered['lift'].max(), 4) if num_rules > 0 else None,
                    'Min_Lift': round(filtered['lift'].min(), 4) if num_rules > 0 else None,
                    'Avg_Lift': round(filtered['lift'].mean(), 4) if num_rules > 0 else None,
                    'Avg_Confidence': round(filtered['confidence'].mean(), 4) if num_rules > 0 else None,
                    'Avg_Support': round(filtered['support'].mean(), 4) if num_rules > 0 else None,
                    'Avg_Rule_Length': round(rule_lengths.mean(), 4) if num_rules > 0 else None,
                    'Max_Rule_Length': int(rule_lengths.max()) if num_rules > 0 else None,
                    'Num_Frequent_Itemsets': len(frequent_itemsets),
                    'Num_FI_length_1': itemsets_by_len.get(1, 0),
                    'Num_FI_length_2': itemsets_by_len.get(2, 0),
                    'Num_FI_length_3plus': sum(v for k, v in itemsets_by_len.items() if k >= 3),
                })

    print(f"\n{'='*70}")
    print("  > Exploration completed. Building summary...")

    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=['Number_of_Rules', 'Max_Lift'], ascending=[False, False]
        ).reset_index(drop=True)

    summary_df.to_csv(output_dir / "summary.csv", index=False)

    combinations_with_rules = int((summary_df['Number_of_Rules'] > 0).sum()) if not summary_df.empty else 0

    # Save human-readable exploration summary .txt
    with open(output_dir / "exploration_summary.txt", 'w') as f:
        f.write("FULL EXPLORATION SUMMARY\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Parameter Grids:\n")
        f.write(f"  Support: {len(support_grid)} values [{support_grid[0]} ... {support_grid[-1]}, step={sup_delta}]\n")
        f.write(f"  Confidence: {len(confidence_grid)} values [{confidence_grid[0]} ... {confidence_grid[-1]}, step={conf_delta}]\n")
        f.write(f"  Lift: {len(lift_grid)} values [{lift_grid[0]} ... {lift_grid[-1]}, step={lift_delta}]\n")
        f.write(f"  Neutral Lift Window (excluded): [{lift_window_lo}, {lift_window_hi}]\n\n")
        f.write("Results:\n")
        f.write(f"  Total combinations evaluated: {len(summary_df):,}\n")
        f.write(f"  Combinations with >= 1 rule: {combinations_with_rules:,}\n\n")
        if not summary_df.empty and combinations_with_rules > 0:
            best = summary_df.iloc[0]
            f.write("Top Combination (by Number of Rules, then Max Lift):\n")
            f.write(f"{'-'*60}\n")
            f.write(f"  Support: {best['Support']}\n")
            f.write(f"  Confidence: {best['Confidence']}\n")
            f.write(f"  Lift threshold: {best['Lift_threshold']}\n")
            f.write(f"  Number of Rules: {best['Number_of_Rules']}\n")
            f.write(f"  Max Lift: {best['Max_Lift']}\n")
            f.write(f"  Avg Lift: {best['Avg_Lift']}\n")

    # Remove empty folders produced by the neutral-window filter
    print("  > Cleaning up empty folders...")
    removed_conf, removed_sup = cleanup_empty_folders(output_dir)
    print(f"  > Removed {removed_conf} empty conf folder(s) and {removed_sup} empty sup folder(s)")

    # Generate heatmaps for all three parameter-pair combinations
    plot_heatmaps(summary_df, output_dir)

    print(f"  > Summary CSV saved → {output_dir / 'summary.csv'}")
    print(f"  > Summary TXT saved → {output_dir / 'exploration_summary.txt'}")
    print(f"  > Total combinations: {len(summary_df):,}")
    print(f"  > With >= 1 rule: {combinations_with_rules:,}")

    return summary_df



def plot_heatmaps(summary_df, output_dir, lift_display_step=0.1):
    """
    Generate three heatmaps from the exploration summary, one for each pair of
    parameters: (support, confidence), (support, lift) and (confidence, lift).

    For each cell the value shown is the maximum Number_of_Rules across all values
    of the third parameter. Darker cells indicate more rules; lighter cells fewer or none.
    All heatmaps use a horizontal layout: lift is always placed on the x-axis to
    avoid overly tall figures caused by the fine lift grid (step 0.05).
    For the same reason, lift values are aggregated into coarser display bins
    controlled by lift_display_step before plotting.

    Heatmaps are saved as PNG files inside output_dir/heatmaps/:
        heatmap_support_confidence.png
        heatmap_support_lift.png
        heatmap_confidence_lift.png

    Args:
        summary_df: Summary DataFrame returned by explore_association_rules().
        output_dir: Root folder of the exploration output (same as passed to explore_association_rules()).
        lift_display_step: Bin width used to coarsen the lift axis for display only (default 0.1).
            The underlying data is not changed; only the visual granularity is reduced.
    """
    if summary_df.empty:
        print("  > Warning: Summary DataFrame is empty — skipping heatmap generation.")
        return

    output_dir  = Path(output_dir)
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    print("  > Generating heatmaps...")

    # Coarsen lift values for display: bin each Lift_threshold into the nearest
    # multiple of lift_display_step so the lift axis stays readable.
    df = summary_df.copy()
    df['Lift_display'] = (
        (df['Lift_threshold'] / lift_display_step).round() * lift_display_step
    ).round(4)

    # For each heatmap: (x_col, y_col, filename_suffix, use_coarsened_lift_for_x, use_coarsened_lift_for_y)
    # Lift is always placed on the x-axis to keep all heatmaps horizontal.
    configs = [
        # x_col            y_col         suffix                   x_is_lift  y_is_lift
        ('Confidence',     'Support',     'support_confidence',    False,     False),
        ('Lift_display',   'Support',     'support_lift',          True,      False),
        ('Lift_display',   'Confidence',  'confidence_lift',       True,      False),
    ]

    for x_col, y_col, suffix, x_is_lift, _ in configs:

        pivot = (
            df
            .groupby([y_col, x_col])['Number_of_Rules']
            .max()
            .unstack(level=x_col)
            .sort_index(ascending=False)   # higher y values at the top
            .fillna(0)
            .astype(int)
        )

        n_cols = len(pivot.columns)
        n_rows = len(pivot.index)

        # Horizontal layout: width scales with columns, height with rows
        fig_w = max(10, n_cols * 0.75)
        fig_h = max(4,  n_rows * 0.55)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        img = ax.imshow(pivot.values, aspect='auto', cmap='YlOrBr', interpolation='nearest')

        # X-axis ticks — rotate lift labels less aggressively since they are on x
        x_labels = [f"{v:.2f}" for v in pivot.columns]
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(x_labels, rotation=40, ha='right', fontsize=8)

        # Y-axis ticks
        y_labels = [f"{v:.2f}" for v in pivot.index]
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(y_labels, fontsize=8)

        x_display = 'Lift' if x_is_lift else x_col
        ax.set_xlabel(x_display, fontsize=11, labelpad=8)
        ax.set_ylabel(y_col,     fontsize=11, labelpad=8)
        ax.set_title(
            f"Max Number of Rules — {y_col} vs {x_display}\n"
            f"(darker = more rules; aggregated over the third parameter)",
            fontsize=11, pad=14
        )

        # Annotate non-zero cells; switch text color based on background brightness
        max_val = pivot.values.max() if pivot.values.max() > 0 else 1
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                val = pivot.values[row_idx, col_idx]
                if val > 0:
                    txt_color = 'white' if (val / max_val) > 0.55 else 'black'
                    ax.text(col_idx, row_idx, str(val),
                            ha='center', va='center', fontsize=7, color=txt_color)

        cbar = plt.colorbar(img, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label('Number of Rules', fontsize=9)
        plt.tight_layout()

        out_path = heatmap_dir / f"heatmap_{suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"    > Saved: heatmaps/heatmap_{suffix}.png")

    print(f"  > Heatmaps saved → {heatmap_dir}/")

if __name__ == "__main__":
    # Detect the environment (Local vs Colab) and set paths accordingly
    if Path("/content").exists():
        base_dir = Path("/content")
    else:
        base_dir = Path(__file__).resolve().parent.parent

    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    important_features_dir = results_dir / "important_features"
    ar_output_dir = results_dir / "association_rules"

    for d in [data_dir, results_dir, important_features_dir, ar_output_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("MACROSCOPIC EXPERIMENT: ASSOCIATION RULES - FULL EXPLORATION")
    print("="*70)

    labels_csv = important_features_dir / "labels_only_unique.csv"

    if not labels_csv.exists():
        print(f"  > Error: Labels file not found at {labels_csv}")
    else:
        # Extract and encode labels from counterfactual transactions
        print(f"\n  > Loading transactions from {labels_csv.name}...")
        df_encoded = extract_labels(labels_csv)
        print(f"  > Loaded {len(df_encoded)} transactions")

        # Run full exploration with file output
        # For the full rationale behind each parameter choice see: docs/parameter_rationale.md
        summary = explore_association_rules(
            df         = df_encoded,
            output_dir = ar_output_dir,
            # Support: SEX pairwise floor (~2%) → observed ceiling at 0.16 (from sup=0.18 only len=1 itemsets remain, 
            # making rule generation impossible).
            sup_min=0.02,  sup_max=0.16,  sup_delta=0.02,
            # Confidence: full [0.10, 1.00] range to include both weak and deterministic rules.
            conf_min=0.10, conf_max=1.00, conf_delta=0.05,
            # Lift: 0.0 is the true theoretical minimum (items never co-occur); 2.5 covers the observed max (1.97) with margin; 
            # finer step (0.05) adds resolution now that the range is narrower.
            lift_min=0.0,  lift_max=2.5,  lift_delta=0.05,
            # Neutral window: rules with lift in [0.75, 1.25] are excluded (near-independence).
            lift_neutral_half_window=0.25,
        )

        print(f"\n{'='*70}")
        print("TOP 10 COMBINATIONS BY NUMBER OF RULES AND LIFT")
        print(f"{'='*70}\n")
        print(summary.head(10).to_string(index=False))

    print("\nExecution completed successfully.\n")