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

# headless backend — needed when running on a server without a display
matplotlib.use('Agg')


def extract_labels(labels_only_path):
    """
    Read the labels CSV and one-hot encode each transaction for FP-Growth.
    The Labels column contains lists stored as strings, so we eval them first.
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
    Returns (n_conf_removed, n_sup_removed).
    """
    output_dir = Path(output_dir)
    removed_conf = 0
    removed_sup = 0

    for sup_dir in sorted(output_dir.glob("sup_*")):
        if not sup_dir.is_dir():
            continue

        for conf_dir in sorted(sup_dir.glob("conf_*")):
            if not conf_dir.is_dir():
                continue

            rules_csv = conf_dir / "rules.csv"
            if not rules_csv.exists() or pd.read_csv(rules_csv).empty:
                shutil.rmtree(conf_dir)
                removed_conf += 1

        if not list(sup_dir.glob("conf_*")):
            shutil.rmtree(sup_dir)
            removed_sup += 1

    return removed_conf, removed_sup


def grid_search_fpgrowth_delta(df, 
                               sup_min, sup_max, sup_delta,
                               conf_min, conf_max, conf_delta,
                               lift_min, lift_max, lift_delta):
    """
    Quick in-memory grid search, no files saved. Useful for a fast sanity check
    before running the full exploration. Use explore_association_rules() if you
    want everything written to disk.
    """
    print(f"\n{'='*70}")
    print("GRID SEARCH: FP-GROWTH (IN-MEMORY)")
    print(f"{'='*70}")

    # +half delta so the max endpoint is always included
    support_grid = np.round(np.arange(sup_min,  sup_max  + sup_delta  / 2, sup_delta),  4)
    confidence_grid = np.round(np.arange(conf_min, conf_max + conf_delta / 2, conf_delta), 4)
    lift_grid = np.round(np.arange(lift_min, lift_max + lift_delta / 2, lift_delta), 4)

    print(f"  > support grid: {support_grid}")
    print(f"  > confidence grid: {confidence_grid}")
    print(f"  > lift grid: {lift_grid}")
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
                filtered = rules[rules['lift'] >= min_lift]
                n = len(filtered)
                results.append({
                    'Support': min_sup,
                    'Confidence': min_conf,
                    'Lift': min_lift,
                    'Number_of_Rules': n,
                    'Max_Lift': round(filtered['lift'].max(), 4) if n > 0 else 0,
                    'Mean_Confidence': round(filtered['confidence'].mean(), 4) if n > 0 else 0,
                })

    print("  > Done.")

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values(by=['Number_of_Rules', 'Lift'], ascending=[False, False])

    return df_results


def deduplicate_symmetric_rules(rules_df):
    """
    lift(A->B) == lift(B->A) by definition, so keeping both directions is just
    redundant noise. For each pair we keep the direction with higher confidence
    since that's the one with actual predictive power. Tiebreak: shorter antecedent.

    Negative-correlation rules (lift < 1) go through the same process — we still
    want them, just not both directions.
    """
    if rules_df.empty:
        return rules_df

    seen = set()
    keep = []

    # sort so the better direction comes first
    df = rules_df.sort_values(
        by=['confidence', 'antecedent_length'],
        ascending=[False, True]
    ).reset_index(drop=True)

    for _, row in df.iterrows():
        ant = row['antecedents_str']
        con = row['consequents_str']

        if (con, ant) in seen:
            continue

        seen.add((ant, con))
        keep.append(row)

    return pd.DataFrame(keep).reset_index(drop=True)


def plot_heatmaps(summary_df, output_dir, lift_display_step=0.1):
    """
    3 heatmaps from the summary: support-confidence, support-lift, confidence-lift.
    Each cell = max rules over the third parameter (darker = more rules).

    Lift is always on the x-axis to keep the plots horizontal — putting it on y
    with step=0.05 makes the figure absurdly tall. Values are also binned at
    lift_display_step resolution just for readability.
    """
    if summary_df.empty:
        print("  > Summary is empty, skipping heatmaps.")
        return

    output_dir = Path(output_dir)
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    print("  > Generating heatmaps...")

    df = summary_df.copy()
    df['Lift_display'] = (
        (df['Lift_threshold'] / lift_display_step).round() * lift_display_step
    ).round(4)

    # (x_col, y_col, file suffix, lift_on_x?)
    configs = [
        ('Confidence', 'Support', 'support_confidence', False),
        ('Lift_display', 'Support', 'support_lift', True),
        ('Lift_display', 'Confidence', 'confidence_lift', True),
    ]

    for x_col, y_col, suffix, x_is_lift in configs:
        pivot = (
            df.groupby([y_col, x_col])['Number_of_Rules']
            .max()
            .unstack(level=x_col)
            .sort_index(ascending=False)
            .fillna(0)
            .astype(int)
        )

        # trim trailing zero columns on the lift axis — keep up to the last
        # column with at least one rule so the drop-off is still visible
        # but the empty desert after it is removed
        if x_is_lift:
            last_nonzero = int(np.where((pivot != 0).any(axis=0).values)[0].max()) if (pivot != 0).any(axis=0).any() else len(pivot.columns) - 1
            pivot = pivot.iloc[:, :last_nonzero + 1]

        n_cols = len(pivot.columns)
        n_rows = len(pivot.index)
        fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.75), max(4, n_rows * 0.55)))

        img = ax.imshow(pivot.values, aspect='auto', cmap='YlOrBr', interpolation='nearest')

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], rotation=40, ha='right', fontsize=8)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=8)

        x_label = 'Lift' if x_is_lift else x_col
        ax.set_xlabel(x_label, fontsize=11, labelpad=8)
        ax.set_ylabel(y_col, fontsize=11, labelpad=8)
        ax.set_title(
            f"Max Number of Rules — {y_col} vs {x_label}\n"
            f"(darker = more rules; max over the third parameter)",
            fontsize=11, pad=14
        )

        max_val = pivot.values.max() if pivot.values.max() > 0 else 1
        for ri in range(n_rows):
            for ci in range(n_cols):
                val = pivot.values[ri, ci]
                if val > 0:
                    txt_color = 'white' if (val / max_val) > 0.55 else 'black'
                    ax.text(ci, ri, str(val), ha='center', va='center', fontsize=7, color=txt_color)

        cbar = plt.colorbar(img, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label('Number of Rules', fontsize=9)
        plt.tight_layout()

        fig.savefig(heatmap_dir / f"heatmap_{suffix}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    > saved heatmaps/heatmap_{suffix}.png")

    print(f"  > heatmaps saved to {heatmap_dir}/")


def explore_association_rules(df, output_dir,
                              sup_min, sup_max, sup_delta,
                              conf_min, conf_max, conf_delta,
                              lift_min, lift_max, lift_delta,
                              lift_neutral_half_window=0.25):
    """
    Full grid search over support x confidence x lift using FP-Growth.

    For each (sup, conf) pair:
      - run FP-Growth to get frequent itemsets
      - generate rules
      - drop near-independent rules (lift in neutral window)
      - remove symmetric duplicates (keep the direction with higher confidence)
      - save rules.csv, rules_detailed.csv, summary.txt

    Also saves a global summary.csv + exploration_summary.txt, cleans up
    empty folders, and generates the heatmaps.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # +half delta so the max endpoint is always included
    support_grid = np.round(np.arange(sup_min,  sup_max  + sup_delta  / 2, sup_delta),  4)
    confidence_grid = np.round(np.arange(conf_min, conf_max + conf_delta / 2, conf_delta), 4)
    lift_grid = np.round(np.arange(lift_min, lift_max + lift_delta / 2, lift_delta), 4)

    lift_window_lo = round(1.0 - lift_neutral_half_window, 4)
    lift_window_hi = round(1.0 + lift_neutral_half_window, 4)
    total_combos = len(support_grid) * len(confidence_grid) * len(lift_grid)

    print(f"\n{'='*70}")
    print("FP-GROWTH ASSOCIATION RULES")
    print(f"{'='*70}")
    print(f"  > support: {len(support_grid)} values [{support_grid[0]} ... {support_grid[-1]}, step={sup_delta}]")
    print(f"  > confidence: {len(confidence_grid)} values [{confidence_grid[0]} ... {confidence_grid[-1]}, step={conf_delta}]")
    print(f"  > lift: {len(lift_grid)} values [{lift_grid[0]} ... {lift_grid[-1]}, step={lift_delta}]")
    print(f"  > neutral window excluded: [{lift_window_lo}, {lift_window_hi}]")
    print(f"  > total combinations: {total_combos:,}")
    print("-" * 50)

    summary_rows = []

    for sup_idx, min_sup in enumerate(support_grid, start=1):
        sup_label = f"{min_sup:.2f}"
        sup_dir = output_dir / f"sup_{sup_label}"
        sup_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{sup_idx}/{len(support_grid)}] support = {min_sup}")
        print("    > running FP-Growth...")

        frequent_itemsets = fpgrowth(df, min_support=min_sup, use_colnames=True)
        print(f"    > found {len(frequent_itemsets)} frequent itemsets")

        if len(frequent_itemsets) == 0:
            print("    > no frequent itemsets, skipping.")
            continue

        fi = frequent_itemsets.copy()
        fi['itemset_str'] = fi['itemsets'].apply(lambda x: ', '.join(sorted(x)))
        fi['itemset_length'] = fi['itemsets'].apply(len)
        fi = fi[['itemset_str', 'itemset_length', 'support']]
        fi = fi.sort_values(by=['itemset_length', 'support'], ascending=[True, False])

        fi.to_csv(sup_dir / "frequent_itemsets.csv", index=False)

        itemsets_by_len = fi['itemset_length'].value_counts().sort_index().to_dict()
        print(f"    > breakdown: {', '.join(f'len={k}: {v}' for k, v in itemsets_by_len.items())}")

        with open(sup_dir / "frequent_itemsets_summary.txt", 'w') as f:
            f.write("Frequent Itemsets Summary\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Parameters:\n  Min Support: {min_sup}\n\n")
            f.write(f"Results:\n  Total: {len(frequent_itemsets)}\n")
            for length, count in itemsets_by_len.items():
                f.write(f"  len={length}: {count}\n")
            f.write(f"\nAll Frequent Itemsets (by length, then support desc):\n{'-'*60}\n")
            for _, row in fi.iterrows():
                f.write(f"  [{row['itemset_str']}]  support={row['support']:.4f}  length={row['itemset_length']}\n")

        for min_conf in confidence_grid:
            conf_label = f"{min_conf:.2f}"
            conf_dir = sup_dir / f"conf_{conf_label}"

            try:
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
            except ValueError:
                continue

            if len(rules) == 0:
                continue

            # drop near-independent rules (lift ~= 1, not useful)
            rules = rules[(rules['lift'] < lift_window_lo) | (rules['lift'] > lift_window_hi)]

            if len(rules) == 0:
                continue

            rules = rules.sort_values('lift', ascending=False)

            # remove symmetric duplicates — lift is symmetric so we only need
            # the direction with higher confidence
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
            fmt = pd.DataFrame()
            fmt['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
            fmt['consequents'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
            fmt['support'] = rules['support'].round(4)
            fmt['confidence'] = rules['confidence'].round(4)
            fmt['lift'] = rules['lift'].round(4)

            # detailed output with all mlxtend metrics and itemset lengths
            det = pd.DataFrame()
            det['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
            det['consequents'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
            det['antecedent_length'] = rules['antecedents'].apply(len)
            det['consequent_length'] = rules['consequents'].apply(len)
            det['rule_length'] = det['antecedent_length'] + det['consequent_length']
            det['antecedent support'] = rules['antecedent support'].values
            det['consequent support'] = rules['consequent support'].values
            det['support'] = rules['support'].values
            det['confidence'] = rules['confidence'].values
            det['lift'] = rules['lift'].values
            det['leverage'] = rules['leverage'].values
            det['conviction'] = rules['conviction'].values

            conf_dir.mkdir(parents=True, exist_ok=True)
            fmt.to_csv(conf_dir / "rules.csv", index=False)
            det.to_csv(conf_dir / "rules_detailed.csv", index=False)

            with open(conf_dir / "summary.txt", 'w') as f:
                f.write("Association Rules Summary\n")
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
                f.write(f"  Avg Rule Length: {det['rule_length'].mean():.2f}\n\n")
                f.write(f"Top 10 Rules (by Lift):\n{'-'*60}\n")
                for idx, row in fmt.head(10).iterrows():
                    f.write(f"{idx+1}. {row['antecedents']} => {row['consequents']}\n")
                    f.write(f"   support={row['support']:.4f} | confidence={row['confidence']:.4f} | lift={row['lift']:.4f}\n\n")

            print(f"    > [conf={min_conf}] {len(rules)} rules saved to {conf_dir.relative_to(output_dir)}/")

            # one summary row per lift threshold
            for min_lift in lift_grid:
                filtered = rules[rules['lift'] >= min_lift]
                n = len(filtered)
                rl = (filtered['antecedents'].apply(len) + filtered['consequents'].apply(len)) if n > 0 else pd.Series(dtype=float)

                summary_rows.append({
                    'Support': min_sup,
                    'Confidence': min_conf,
                    'Lift_threshold': min_lift,
                    'Number_of_Rules': n,
                    'Max_Lift': round(filtered['lift'].max(), 4) if n > 0 else None,
                    'Min_Lift': round(filtered['lift'].min(), 4) if n > 0 else None,
                    'Avg_Lift': round(filtered['lift'].mean(), 4) if n > 0 else None,
                    'Avg_Confidence': round(filtered['confidence'].mean(), 4) if n > 0 else None,
                    'Avg_Support': round(filtered['support'].mean(), 4) if n > 0 else None,
                    'Avg_Rule_Length': round(rl.mean(), 4) if n > 0 else None,
                    'Max_Rule_Length': int(rl.max()) if n > 0 else None,
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
        f.write(f"  Total combinations: {len(summary_df):,}\n")
        f.write(f"  With >= 1 rule: {combos_with_rules:,}\n\n")
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
    print(f"  > Removed {removed_conf} conf folder(s) and {removed_sup} sup folder(s)")

    plot_heatmaps(summary_df, output_dir)

    print(f"  > summary saved to {output_dir / 'summary.csv'}")
    print(f"  > total combinations: {len(summary_df):,}  |  with rules: {combos_with_rules:,}")

    return summary_df


def calibrate_parameters(encoded_df, sup_delta=0.02, lift_delta=0.05):
    """
    Auto-calibrate sup_min, sup_max and lift_max from the actual item frequencies.
    Call this before explore_association_rules() when k changes so the grid always
    covers the right range without manual tuning.

    sup_min  = expected joint support of the two rarest items (under independence),
               rounded down to the nearest sup_delta step
    sup_max  = last threshold where FP-Growth still finds at least one 2-itemset
    lift_max = 1 / support(rarest item), rounded up to nearest 0.5, capped at 10
    """
    print("  > Calibrating parameters from item frequencies...")

    item_supports = encoded_df.mean().sort_values()
    print("  > Item supports:")
    for item, sup in item_supports.items():
        print(f"    {item}: {sup:.4f}")

    rarest = item_supports.iloc[0]
    second = item_supports.iloc[1]

    # pairwise floor under independence, rounded down to sup_delta
    raw_sup_min = rarest * second
    sup_min = max(round(np.floor(raw_sup_min / sup_delta) * sup_delta, 4), sup_delta)

    # scan upward until 2-itemsets disappear
    scan_grid = np.round(np.arange(sup_min, rarest * 1.05 + sup_delta, sup_delta), 4)
    sup_max = sup_min
    for t in scan_grid:
        fi = fpgrowth(encoded_df, min_support=t, use_colnames=True)
        if fi.empty:
            break
        if fi['itemsets'].apply(len).max() >= 2:
            sup_max = t

    # theoretical lift ceiling, round up to nearest 0.5
    raw_lift_max = 1.0 / rarest
    lift_max = min(round(np.ceil(raw_lift_max * 2) / 2, 1), 10.0)

    # if sup_max == sup_min it means FP-Growth never found a 2-itemset at any
    # threshold — this k produces only single-item transactions (too sparse)
    if sup_max == sup_min:
        print(f"  > Warning: no 2-itemsets found at any support threshold for this k.")
        print(f"    Transactions are too sparse to generate association rules.")
        print("-" * 50)
        return None

    params = {
        'sup_min': sup_min,
        'sup_max': sup_max,
        'sup_delta': sup_delta,
        'lift_min': 0.0,
        'lift_max': lift_max,
        'lift_delta': lift_delta,
    }

    print(f"  > calibrated: sup_min={sup_min} (raw={raw_sup_min:.4f}), "
          f"sup_max={sup_max}, lift_max={lift_max} (raw ceiling={raw_lift_max:.2f})")
    print("-" * 50)

    return params


def run_k_comparison(k_labels_map, output_dir,
                     auto_calibrate=True,
                     sup_min=0.02,  sup_max=0.16,  sup_delta=0.02,
                     conf_min=0.10, conf_max=1.00,  conf_delta=0.05,
                     lift_min=0.0,  lift_max=2.5,   lift_delta=0.05,
                     lift_neutral_half_window=0.25):
    """
    Run explore_association_rules for each k and produce a cross-k comparison.

    If auto_calibrate=True, sup_min/sup_max/lift_max are recomputed for each k
    from the actual item frequencies — recommended since k changes the transaction
    distribution. conf_* and lift_delta are shared across all k values.

    Cross-k outputs saved in output_dir/k_comparison/:
      k_comparison_summary.csv / .txt
      heatmap_k_support.png, heatmap_k_confidence.png, heatmap_k_lift.png
    """
    output_dir = Path(output_dir)
    comp_dir = output_dir / "k_comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"K-VARIATION EXPERIMENT — {len(k_labels_map)} values of k")
    print(f"{'='*70}")
    print(f"  > k values: {sorted(k_labels_map.keys())}")
    print(f"  > auto_calibrate: {auto_calibrate}")
    print("-" * 50)

    k_summaries = {}
    comparison_rows = []

    for k in sorted(k_labels_map.keys()):
        labels_csv = Path(k_labels_map[k])
        k_dir = output_dir / f"k_{k}"
        k_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  k = {k}  |  {labels_csv.name}")
        print(f"{'='*70}")

        if not labels_csv.exists():
            print(f"  > file not found, skipping k={k}.")
            continue

        df_encoded = extract_labels(labels_csv)
        item_supports = df_encoded.mean().sort_values()

        if auto_calibrate:
            params = calibrate_parameters(df_encoded, sup_delta=sup_delta, lift_delta=lift_delta)
            if params is None:
                print(f"  > Skipping k={k} — not enough co-occurrences to generate rules.")
                continue
            k_sup_min = params['sup_min']
            k_sup_max = params['sup_max']
            k_lift_max = params['lift_max']
        else:
            k_sup_min = sup_min
            k_sup_max = sup_max
            k_lift_max = lift_max

        summary_df = explore_association_rules(
            df=df_encoded,
            output_dir=k_dir,
            sup_min=k_sup_min, sup_max=k_sup_max, sup_delta=sup_delta,
            conf_min=conf_min, conf_max=conf_max, conf_delta=conf_delta,
            lift_min=lift_min, lift_max=k_lift_max, lift_delta=lift_delta,
            lift_neutral_half_window=lift_neutral_half_window,
        )

        k_summaries[k] = summary_df

        has_rules_col = not summary_df.empty and 'Number_of_Rules' in summary_df.columns
        with_rules = summary_df[summary_df['Number_of_Rules'] > 0] if has_rules_col else pd.DataFrame()
        comparison_rows.append({
            'k': k,
            'n_transactions': len(df_encoded),
            'n_items': df_encoded.shape[1],
            'rarest_item_support': round(item_supports.iloc[0], 4),
            'sup_min_used': k_sup_min,
            'sup_max_used': k_sup_max,
            'lift_max_used': k_lift_max,
            'total_combos': len(summary_df),
            'combos_with_rules': len(with_rules),
            'max_rules_any_combo': int(summary_df['Number_of_Rules'].max()) if has_rules_col else 0,
            'avg_lift_best_combo': round(with_rules['Avg_Lift'].max(), 4) if not with_rules.empty else None,
            'max_lift_observed': round(summary_df['Max_Lift'].max(), 4) if has_rules_col else None,
        })

    print(f"\n{'='*70}")
    print("  > Building cross-k comparison...")

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(comp_dir / "k_comparison_summary.csv", index=False)

    with open(comp_dir / "k_comparison_summary.txt", "w") as f:
        f.write("K-VARIATION EXPERIMENT SUMMARY\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"k values tested: {sorted(k_labels_map.keys())}\n")
        f.write(f"auto_calibrate:  {auto_calibrate}\n\n")
        f.write(f"Results per k:\n{'-'*60}\n")
        f.write(comp_df.to_string(index=False))
        f.write("\n\n")
        if not comp_df.empty:
            best_k = comp_df.loc[comp_df['max_rules_any_combo'].idxmax(), 'k']
            f.write(f"Most rules: k={best_k} ({comp_df['max_rules_any_combo'].max()} rules at best combo)\n")

    print("  > Saved k_comparison_summary.csv and .txt")

    # cross-k heatmaps
    all_summaries = []
    for k, sdf in k_summaries.items():
        tmp = sdf.copy()
        tmp['k'] = k
        all_summaries.append(tmp)

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined['Lift_display'] = (
            (combined['Lift_threshold'] / 0.1).round() * 0.1
        ).round(4)

        heatmap_configs = [
            ('Support', 'k_support'),
            ('Confidence', 'k_confidence'),
            ('Lift_display', 'k_lift'),
        ]

        for x_col, suffix in heatmap_configs:
            x_label = 'Lift' if 'lift' in suffix else x_col

            pivot = (
                combined.groupby(['k', x_col])['Number_of_Rules']
                .max()
                .unstack(level=x_col)
                .sort_index(ascending=False)
                .fillna(0)
                .astype(int)
            )

            # same trim as plot_heatmaps — keep up to the last non-zero lift column
            if 'lift' in suffix:
                last_nonzero = int(np.where((pivot != 0).any(axis=0).values)[0].max()) if (pivot != 0).any(axis=0).any() else len(pivot.columns) - 1
                pivot = pivot.iloc[:, :last_nonzero + 1]

            n_cols = len(pivot.columns)
            n_rows = len(pivot.index)
            fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.75), max(4, n_rows * 0.6)))

            img = ax.imshow(pivot.values, aspect='auto', cmap='YlOrBr', interpolation='nearest')

            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(
                [f"{v:.2f}" if isinstance(v, float) else str(v) for v in pivot.columns],
                rotation=40, ha='right', fontsize=8
            )
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels([f"k={v}" for v in pivot.index], fontsize=9)

            ax.set_xlabel(x_label, fontsize=11, labelpad=8)
            ax.set_ylabel('k', fontsize=11, labelpad=8)
            ax.set_title(
                f"Max Number of Rules — k vs {x_label}\n"
                f"(darker = more rules; max over the other two parameters)",
                fontsize=11, pad=14
            )

            max_val = pivot.values.max() if pivot.values.max() > 0 else 1
            for ri in range(n_rows):
                for ci in range(n_cols):
                    val = pivot.values[ri, ci]
                    if val > 0:
                        txt_color = 'white' if (val / max_val) > 0.55 else 'black'
                        ax.text(ci, ri, str(val), ha='center', va='center', fontsize=7, color=txt_color)

            cbar = plt.colorbar(img, ax=ax, fraction=0.025, pad=0.02)
            cbar.set_label('Number of Rules', fontsize=9)
            plt.tight_layout()

            fig.savefig(comp_dir / f"heatmap_{suffix}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    > saved k_comparison/heatmap_{suffix}.png")

    print(f"  > Cross-k comparison saved to {comp_dir}/")
    return k_summaries


if __name__ == "__main__":
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
        print(f"  > Error: file not found at {labels_csv}")
    else:
        print(f"\n  > Loading transactions from {labels_csv.name}...")
        df_encoded = extract_labels(labels_csv)
        print(f"  > Loaded {len(df_encoded)} transactions")

        # see docs/parameter_rationale.md for why these values
        summary = explore_association_rules(
            df=df_encoded,
            output_dir=ar_output_dir,
            # sup ceiling at 0.16: from 0.18 onward only len=1 itemsets survive
            sup_min=0.02,  sup_max=0.16,  sup_delta=0.02,
            # full confidence range
            conf_min=0.10, conf_max=1.00, conf_delta=0.05,
            # lift from 0 (true min) to 2.5 (headroom over observed max of 1.97)
            lift_min=0.0,  lift_max=2.5,  lift_delta=0.05,
            # drop rules with lift in [0.75, 1.25] — basically independent
            lift_neutral_half_window=0.25,
        )

        print(f"\n{'='*70}")
        print("TOP 10 COMBINATIONS BY NUMBER OF RULES AND LIFT")
        print(f"{'='*70}\n")
        print(summary.head(10).to_string(index=False))

    print("\nDone.\n")