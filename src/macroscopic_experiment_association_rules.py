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

# need this to avoid display errors when running headless (e.g. on the server)
matplotlib.use('Agg')


def extract_labels(labels_only_path):
    """
    Read the labels CSV and one-hot encode each transaction for FP-Growth.
    The 'Labels' column contains lists stored as strings, so we eval them first.

    Args:
        labels_only_path: path to labels_only_unique.csv
    Returns:
        one-hot encoded DataFrame ready for fpgrowth()
    """
    print("  > Loading and encoding labels...")
    df = pd.read_csv(labels_only_path)
    itemsets = df['Labels'].apply(ast.literal_eval)

    te = TransactionEncoder()
    te_ary = te.fit(itemsets).transform(itemsets)
    encoded_df = pd.DataFrame(te_ary, columns=te.columns_)

    print("  > First few encoded rows:")
    print(encoded_df.head())
    print("-" * 50)

    return encoded_df


def cleanup_empty_folders(output_dir):
    """
    After filtering, some conf_* folders end up with no rules — just delete them.
    If a sup_* folder loses all its conf_* subfolders, delete that too.

    Args:
        output_dir: root output folder (same one passed to explore_association_rules)
    Returns:
        (n_conf_removed, n_sup_removed)
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

            # if rules.csv doesn't exist or is empty → folder is useless
            rules_csv = conf_dir / "rules.csv"
            is_empty  = not rules_csv.exists() or pd.read_csv(rules_csv).empty

            if is_empty:
                shutil.rmtree(conf_dir)
                removed_conf += 1

        # clean up the sup folder too if nothing is left inside
        if not list(sup_dir.glob("conf_*")):
            shutil.rmtree(sup_dir)
            removed_sup += 1

    return removed_conf, removed_sup


def grid_search_fpgrowth_delta(df, sup_min, sup_max, sup_delta,
                               conf_min, conf_max, conf_delta,
                               lift_min, lift_max, lift_delta):
    """
    Quick in-memory grid search — no files saved. Useful for a fast overview
    before running the full exploration. Use explore_association_rules() if
    you want everything saved to disk.

    Args:
        df: one-hot encoded DataFrame from extract_labels()
        sup_min, sup_max, sup_delta: support range and step
        conf_min, conf_max, conf_delta: confidence range and step
        lift_min, lift_max, lift_delta: lift range and step
    Returns:
        summary DataFrame sorted by number of rules and lift, descending
    """
    print(f"\n{'='*70}")
    print("GRID SEARCH: FP-GROWTH (IN-MEMORY, NO FILES SAVED)")
    print(f"{'='*70}")

    # +half delta so the max is always included (floating point safety)
    support_grid    = np.round(np.arange(sup_min,  sup_max  + sup_delta  / 2, sup_delta),  4)
    confidence_grid = np.round(np.arange(conf_min, conf_max + conf_delta / 2, conf_delta), 4)
    lift_grid       = np.round(np.arange(lift_min, lift_max + lift_delta / 2, lift_delta), 4)

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
                # mlxtend throws this when there are no valid rules
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

    print("  > Done.")

    df_results = pd.DataFrame(results)

    if not df_results.empty:
        df_results = df_results.sort_values(
            by=['Number_of_Rules', 'Lift'], ascending=[False, False]
        )

    return df_results


def deduplicate_symmetric_rules(rules_df):
    """
    lift(A→B) == lift(B→A) by definition, so keeping both directions is redundant.
    For each symmetric pair we keep the direction with higher confidence, since
    confidence is asymmetric and tells us which direction is actually predictive.
    Tiebreak: shorter antecedent (simpler rule).

    Negative-correlation rules (lift < 1) go through the same process — we still
    want to keep them, just avoid the duplicate direction.

    Args:
        rules_df: DataFrame with columns antecedents_str, consequents_str,
                  confidence, antecedent_length
    Returns:
        deduplicated DataFrame
    """
    if rules_df.empty:
        return rules_df

    seen = set()
    keep = []

    # sort so the better direction (higher conf, shorter antecedent) comes first
    df = rules_df.sort_values(
        by=['confidence', 'antecedent_length'],
        ascending=[False, True]
    ).reset_index(drop=True)

    for _, row in df.iterrows():
        ant = row['antecedents_str']
        con = row['consequents_str']
        pair = (ant, con)
        rev = (con, ant)

        if rev in seen:
            # already saved the better direction, skip this one
            continue

        seen.add(pair)
        keep.append(row)

    return pd.DataFrame(keep).reset_index(drop=True)


def plot_heatmaps(summary_df, output_dir, lift_display_step=0.1):
    """
    Generate 3 heatmaps from the summary: support-confidence, support-lift,
    confidence-lift. Each cell shows the max number of rules across the third
    parameter (darker = more rules).

    Lift is always on the x-axis to keep the plots horizontal — putting it on y
    with step=0.05 makes the figure absurdly tall. We also bin lift values at
    lift_display_step resolution just for readability (data is unchanged).

    Output: heatmaps/ folder inside output_dir with 3 PNG files.

    Args:
        summary_df: summary DataFrame from explore_association_rules()
        output_dir: same root folder used in explore_association_rules()
        lift_display_step: bin width for the lift axis (display only, default 0.1)
    """
    if summary_df.empty:
        print("  > No data in summary, skipping heatmaps.")
        return

    output_dir = Path(output_dir)
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    print("  > Generating heatmaps...")

    # bin lift values so we don't get 51 tick labels on the x-axis
    df = summary_df.copy()
    df['Lift_display'] = (
        (df['Lift_threshold'] / lift_display_step).round() * lift_display_step
    ).round(4)

    # (x_col, y_col, file suffix, is_lift_on_x)
    configs = [
        ('Confidence', 'Support', 'support_confidence', False),
        ('Lift_display', 'Support', 'support_lift', True),
        ('Lift_display', 'Confidence', 'confidence_lift', True),
    ]

    for x_col, y_col, suffix, x_is_lift in configs:

        # pivot: rows = y (high at top), cols = x, values = max rules
        pivot = (
            df
            .groupby([y_col, x_col])['Number_of_Rules']
            .max()
            .unstack(level=x_col)
            .sort_index(ascending=False)
            .fillna(0)
            .astype(int)
        )

        n_cols = len(pivot.columns)
        n_rows = len(pivot.index)

        fig_w = max(10, n_cols * 0.75)
        fig_h = max(4,  n_rows * 0.55)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        img = ax.imshow(pivot.values, aspect='auto', cmap='YlOrBr', interpolation='nearest')

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], rotation=40, ha='right', fontsize=8)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=8)

        x_label = 'Lift' if x_is_lift else x_col
        ax.set_xlabel(x_label, fontsize=11, labelpad=8)
        ax.set_ylabel(y_col,   fontsize=11, labelpad=8)
        ax.set_title(
            f"Max Number of Rules — {y_col} vs {x_label}\n"
            f"(darker = more rules; max over the third parameter)",
            fontsize=11, pad=14
        )

        # annotate each cell — skip zeros to keep it clean
        # text is white on dark cells, black on light ones
        max_val = pivot.values.max() if pivot.values.max() > 0 else 1
        for ri in range(n_rows):
            for ci in range(n_cols):
                val = pivot.values[ri, ci]
                if val > 0:
                    txt_color = 'white' if (val / max_val) > 0.55 else 'black'
                    ax.text(ci, ri, str(val),
                            ha='center', va='center', fontsize=7, color=txt_color)

        cbar = plt.colorbar(img, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label('Number of Rules', fontsize=9)
        plt.tight_layout()

        out_path = heatmap_dir / f"heatmap_{suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"    > Saved heatmaps/heatmap_{suffix}.png")

    print(f"  > All heatmaps saved to {heatmap_dir}/")


def explore_association_rules(df, output_dir,
                              sup_min, sup_max, sup_delta,
                              conf_min, conf_max, conf_delta,
                              lift_min, lift_max, lift_delta,
                              lift_neutral_half_window=0.25):
    """
    Full grid search over support x confidence x lift. For each (sup, conf) pair:
      - run FP-Growth to get frequent itemsets
      - generate association rules
      - drop rules in the neutral lift window (near-independence, not useful)
      - remove symmetric duplicates (keep higher-confidence direction)
      - save rules.csv, rules_detailed.csv, summary.txt

    Also saves a global summary.csv and exploration_summary.txt at the root,
    then cleans up empty folders and generates the heatmaps.

    Args:
        df: one-hot encoded DataFrame from extract_labels()
        output_dir: where to save everything
        sup_min/max/delta, conf_min/max/delta, lift_min/max/delta: grid params
        lift_neutral_half_window: rules with lift in [1-hw, 1+hw] are dropped (default 0.25 → window [0.75, 1.25])
    Returns:
        summary DataFrame (also saved to summary.csv)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # +half delta so the max is always included
    support_grid = np.round(np.arange(sup_min,  sup_max  + sup_delta  / 2, sup_delta),  4)
    confidence_grid = np.round(np.arange(conf_min, conf_max + conf_delta / 2, conf_delta), 4)
    lift_grid = np.round(np.arange(lift_min, lift_max + lift_delta / 2, lift_delta), 4)

    total_combos = len(support_grid) * len(confidence_grid) * len(lift_grid)
    lift_window_lo = round(1.0 - lift_neutral_half_window, 4)
    lift_window_hi = round(1.0 + lift_neutral_half_window, 4)

    print(f"\n{'='*70}")
    print("FP-GROWTH ASSOCIATION RULES")
    print(f"{'='*70}")
    print(f"  > Support: {len(support_grid)} values [{support_grid[0]} ... {support_grid[-1]}, step={sup_delta}]")
    print(f"  > Confidence: {len(confidence_grid)} values [{confidence_grid[0]} ... {confidence_grid[-1]}, step={conf_delta}]")
    print(f"  > Lift: {len(lift_grid)} values [{lift_grid[0]} ... {lift_grid[-1]}, step={lift_delta}]")
    print(f"  > Neutral lift window excluded: [{lift_window_lo}, {lift_window_hi}]")
    print(f"  > Total combinations to check: {total_combos:,}")
    print("-" * 50)

    summary_rows = []
    total_sup = len(support_grid)

    for sup_idx, min_sup in enumerate(support_grid, start=1):

        sup_label = f"{min_sup:.2f}"
        sup_dir = output_dir / f"sup_{sup_label}"
        sup_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{sup_idx}/{total_sup}] Support = {min_sup}")

        print("    > Running FP-Growth...")
        frequent_itemsets = fpgrowth(df, min_support=min_sup, use_colnames=True)
        print(f"    > Found {len(frequent_itemsets)} frequent itemsets")

        if len(frequent_itemsets) == 0:
            print("    > No frequent itemsets — skipping.")
            continue

        # add readable string and length columns
        fi = frequent_itemsets.copy()
        fi['itemset_str'] = fi['itemsets'].apply(lambda x: ', '.join(sorted(x)))
        fi['itemset_length'] = fi['itemsets'].apply(len)
        fi = fi[['itemset_str', 'itemset_length', 'support']]
        fi = fi.sort_values(by=['itemset_length', 'support'], ascending=[True, False])

        fi.to_csv(sup_dir / "frequent_itemsets.csv", index=False)

        # breakdown by length (useful to spot when only len=1 itemsets survive)
        itemsets_by_len = fi['itemset_length'].value_counts().sort_index().to_dict()
        len_summary = ', '.join([f"len={k}: {v}" for k, v in itemsets_by_len.items()])
        print(f"    > Breakdown: {len_summary}")

        with open(sup_dir / "frequent_itemsets_summary.txt", 'w') as f:
            f.write(f"Frequent Itemsets Summary\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Parameters:\n")
            f.write(f"  Min Support: {min_sup}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Total Frequent Itemsets: {len(frequent_itemsets)}\n")
            for length, count in itemsets_by_len.items():
                f.write(f"  Itemsets of length {length}: {count}\n")
            f.write(f"\nAll Frequent Itemsets (by length, then support desc):\n")
            f.write(f"{'-'*60}\n")
            for _, row in fi.iterrows():
                f.write(f"  [{row['itemset_str']}]  support={row['support']:.4f}  length={row['itemset_length']}\n")

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
                continue

            if len(rules) == 0:
                continue

            # drop near-independent rules (lift ≈ 1, not useful)
            rules = rules[
                (rules['lift'] < lift_window_lo) | (rules['lift'] > lift_window_hi)
            ]

            if len(rules) == 0:
                continue

            rules = rules.sort_values('lift', ascending=False)

            # remove symmetric duplicates before saving — lift is the same in both
            # directions, so we just keep the direction with higher confidence
            _tmp = pd.DataFrame({
                'antecedents_str': rules['antecedents'].apply(lambda x: ', '.join(sorted(x))),
                'consequents_str': rules['consequents'].apply(lambda x: ', '.join(sorted(x))),
                'confidence': rules['confidence'].values,
                'antecedent_length': rules['antecedents'].apply(len).values,
                '_idx': range(len(rules)),
            })
            _keep_idx = deduplicate_symmetric_rules(_tmp)['_idx'].tolist()
            rules = rules.iloc[_keep_idx].reset_index(drop=True)

            # compact output
            formatted_rules = pd.DataFrame()
            formatted_rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
            formatted_rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
            formatted_rules['support'] = rules['support'].round(4)
            formatted_rules['confidence'] = rules['confidence'].round(4)
            formatted_rules['lift'] = rules['lift'].round(4)

            # detailed output — includes itemset lengths and all mlxtend metrics
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

            # one summary row per lift threshold
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
    print("  > Exploration done, building summary...")

    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=['Number_of_Rules', 'Max_Lift'], ascending=[False, False]
        ).reset_index(drop=True)

    summary_df.to_csv(output_dir / "summary.csv", index=False)

    combos_with_rules = int((summary_df['Number_of_Rules'] > 0).sum()) if not summary_df.empty else 0

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
        f.write(f"  Combinations with >= 1 rule: {combos_with_rules:,}\n\n")
        if not summary_df.empty and combos_with_rules > 0:
            best = summary_df.iloc[0]
            f.write("Best combination (most rules, then highest lift):\n")
            f.write(f"{'-'*60}\n")
            f.write(f"  Support: {best['Support']}\n")
            f.write(f"  Confidence: {best['Confidence']}\n")
            f.write(f"  Lift threshold: {best['Lift_threshold']}\n")
            f.write(f"  Number of Rules: {best['Number_of_Rules']}\n")
            f.write(f"  Max Lift: {best['Max_Lift']}\n")
            f.write(f"  Avg Lift: {best['Avg_Lift']}\n")

    print("  > Cleaning up empty folders...")
    removed_conf, removed_sup = cleanup_empty_folders(output_dir)
    print(f"  > Removed {removed_conf} empty conf folder(s) and {removed_sup} empty sup folder(s)")

    plot_heatmaps(summary_df, output_dir)

    print(f"  > Summary saved to {output_dir / 'summary.csv'}")
    print(f"  > Exploration summary saved to {output_dir / 'exploration_summary.txt'}")
    print(f"  > Total combinations: {len(summary_df):,}")
    print(f"  > With >= 1 rule: {combos_with_rules:,}")

    return summary_df


if __name__ == "__main__":
    # local vs Colab path setup
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
    print("MACROSCOPIC EXPERIMENT: ASSOCIATION RULES EXPLORATION")
    print("="*70)

    labels_csv = important_features_dir / "labels_only_unique.csv"

    if not labels_csv.exists():
        print(f"  > Error: file not found at {labels_csv}")
    else:
        print(f"\n  > Loading transactions from {labels_csv.name}...")
        df_encoded = extract_labels(labels_csv)
        print(f"  > Loaded {len(df_encoded)} transactions")

        # full grid search — see docs/parameter_rationale.md for why these values
        summary = explore_association_rules(
            df         = df_encoded,
            output_dir = ar_output_dir,
            # sup ceiling at 0.16: from 0.18 onward FP-Growth only finds len=1 itemsets
            # (can't generate any rules from a single item), confirmed empirically
            sup_min=0.02,  sup_max=0.16,  sup_delta=0.02,
            # full confidence range, want to see everything from weak to deterministic
            conf_min=0.10, conf_max=1.00, conf_delta=0.05,
            # lift 0.0 → 2.5: 0.0 is the true min (items never co-occur),
            # 2.5 gives some headroom over the observed max of 1.97,
            # finer step (0.05) since the range is now much narrower
            lift_min=0.0,  lift_max=2.5,  lift_delta=0.05,
            # drop rules with lift in [0.75, 1.25] — basically independent, not interesting
            lift_neutral_half_window=0.25,
        )

        print(f"\n{'='*70}")
        print("TOP 10 COMBINATIONS BY NUMBER OF RULES AND LIFT")
        print(f"{'='*70}\n")
        print(summary.head(10).to_string(index=False))

    print("\nExecution completed successfully.\n")