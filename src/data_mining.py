import os
import warnings
import logging
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from pathlib import Path
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Suppress standard warnings to keep the terminal output clean
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")
logging.getLogger('jupyter_client').setLevel(logging.ERROR)

class InteractionMiner:
    """ Mines association rules from the extracted counterfactual transactions. """
    def __init__(self, min_support=0.01, min_confidence=0.1):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.te = TransactionEncoder()

    def execute(self, filepath, output_dir):
        print(f"Mining frequent itemsets from {filepath}...")
        df = pd.read_csv(filepath)
        df['Counterfactual_Values'] = df['Counterfactual_Values'].apply(ast.literal_eval)
        
        # Drop empty transactions
        transactions = df[df['Counterfactual_Values'].map(len) > 0]['Counterfactual_Values'].tolist()
        df_enc = pd.DataFrame(self.te.fit(transactions).transform(transactions), columns=self.te.columns_)

        # FP-growth is used here as it evaluates the FP-Tree faster than Apriori for this data density
        freq_itemsets = fpgrowth(df_enc, min_support=self.min_support, use_colnames=True)
        rules = association_rules(freq_itemsets, metric="confidence", min_threshold=self.min_confidence)

        '''
        if not rules.empty:
            # Filter out rules where antecedent and consequent belong to the same base feature
            def is_cross_feature(row):
                ant_feats = set(i.split('=')[0] for i in row['antecedents'])
                con_feats = set(i.split('=')[0] for i in row['consequents'])
                return len(ant_feats.intersection(con_feats)) == 0

            rules = rules[rules.apply(is_cross_feature, axis=1)]
            rules = rules.sort_values(by='lift', ascending=False)
        '''
            
        freq_itemsets.to_csv(output_dir / "frequent_values.csv", index=False)
        if not rules.empty:
            rules.to_csv(output_dir / "microscopic_level_association_rules.csv", index=False)
        print("Rule mining completed.")

class SensitiveAuditMiner:
    """ Checks if sensitive attributes are explicitly present in the extracted rules. """
    def __init__(self, sensitive_features=['SEX', 'RAC1P']):
        self.sensitive_features = sensitive_features
        self.df_rules = None

    def load_rules(self, file_path):
        if not Path(file_path).exists():
            return False
        self.df_rules = pd.read_csv(file_path)
        return True

    def run_audit(self, output_dir):
        print(f"Auditing for direct bias (features: {self.sensitive_features})...")
        
        def contains_sensitive(row):
            combined_rule = str(row['antecedents']) + str(row['consequents'])
            return any(f"{feat}=" in combined_rule for feat in self.sensitive_features)

        sensitive_subset = self.df_rules[self.df_rules.apply(contains_sensitive, axis=1)].copy()
        sensitive_subset = sensitive_subset.sort_values(by='lift', ascending=False)

        if not sensitive_subset.empty:
            sensitive_subset.to_csv(output_dir / "sensitive_audit_report.csv", index=False)
            print("Direct sensitive rules found and saved.")
        else:
            print("No direct sensitive rules found.")

class FairnessAuditor:
    """ Computes the worst-case confidence difference across multiple sensitive features. """
    def __init__(self, sensitive_features=['SEX', 'RAC1P']):
        self.sensitive_features = sensitive_features

    def audit_rules(self, rules_path, data_path):
        print(f"Evaluating conditional fairness across {self.sensitive_features} groups...")
        df_rules = pd.read_csv(rules_path)
        df_data = pd.read_csv(data_path)
        audit_results = []

        for _, rule in df_rules.iterrows():
            ant_str = str(rule['antecedents']).replace("frozenset({", "").replace("})", "").replace("'", "")
            con_str = str(rule['consequents']).replace("frozenset({", "").replace("})", "").replace("'", "")

            antecedents = [a.strip() for a in ant_str.split(',') if a.strip()]
            consequents = [c.strip() for c in con_str.split(',') if c.strip()]

            def get_mask(df, items):
                mask = pd.Series([True] * len(df))
                for item in items:
                    if '=' in item:
                        feat, val = item.split('=', 1)
                        mask &= (df[feat] == val)
                return mask

            # Evaluate the rule against every specified sensitive feature
            for sens_feat in self.sensitive_features:
                groups = df_data[sens_feat].dropna().unique()
                group_metrics = {}

                # Calculate confidence for each demographic sub-group
                for g in groups:
                    group_data = df_data[df_data[sens_feat] == g]
                    g_ant_mask = get_mask(group_data, antecedents)
                    g_con_mask = get_mask(group_data, consequents)

                    support_count = g_ant_mask.sum()
                    confidence = (g_ant_mask & g_con_mask).sum() / support_count if support_count > 0 else 0
                    group_metrics[g] = confidence

                # Compute the Worst-Case Disparity comparing most favored vs most penalized
                if len(group_metrics) >= 2:
                    max_g = max(group_metrics, key=group_metrics.get)
                    min_g = min(group_metrics, key=group_metrics.get)
                    
                    v_max = group_metrics[max_g]
                    v_min = group_metrics[min_g]
                    
                    conf_diff = abs(v_max - v_min)
                    
                    if v_max > 0 and v_min > 0: 
                        disp_impact = min(v_min/v_max, v_max/v_min)
                    elif v_max == 0 and v_min == 0: 
                        disp_impact = 1.0
                    else: 
                        disp_impact = 0.0

                    audit_results.append({
                        'Rule': f"{ant_str} -> {con_str}",
                        'Sensitive_Feature': sens_feat,
                        'Favored_Group': max_g,
                        'Penalized_Group': min_g,
                        'Max_Conf': round(v_max, 3),
                        'Min_Conf': round(v_min, 3),
                        'Conf_Difference': round(conf_diff, 3),
                        'Disparate_Impact': round(disp_impact, 3)
                    })

        return pd.DataFrame(audit_results)

class SensitiveProxyDetector:
    """ Identifies if an antecedent acts as a statistical proxy for a sensitive attribute. """
    def __init__(self, sensitive_features=['SEX', 'RAC1P']):
        self.sensitive_features = sensitive_features

    def detect_proxies(self, rules_path, data_path, lift_threshold=1.5):
        print(f"Calculating Proxy Lift for {self.sensitive_features}...")
        df_rules = pd.read_csv(rules_path)
        df_data = pd.read_csv(data_path)
        proxy_results = []

        base_rates = {}
        for feat in self.sensitive_features:
            base_rates[feat] = df_data[feat].value_counts(normalize=True).to_dict()

        for _, rule in df_rules.iterrows():
            ant_str = str(rule['antecedents']).replace("frozenset({", "").replace("})", "").replace("'", "")
            antecedents = [a.strip() for a in ant_str.split(',') if a.strip()]

            mask = pd.Series([True] * len(df_data))
            for item in antecedents:
                if '=' in item:
                    feat, val = item.split('=', 1)
                    mask &= (df_data[feat] == val)
            
            support_count = mask.sum()
            # Ignore combinations with too little support to avoid statistical noise
            if support_count < 10: 
                continue
                
            subset = df_data[mask]

            for sens_feat in self.sensitive_features:
                subset_dist = subset[sens_feat].value_counts(normalize=True).to_dict()
                
                for sens_val, prob_given_ant in subset_dist.items():
                    prob_base = base_rates[sens_feat].get(sens_val, 0)
                    
                    if prob_base > 0:
                        proxy_lift = prob_given_ant / prob_base
                        if proxy_lift >= lift_threshold:
                            proxy_results.append({
                                'Model_Condition (Antecedent)': ant_str,
                                'Acts_as_Proxy_For': f"{sens_feat}={sens_val}",
                                'Base_Probability': round(prob_base, 3),
                                'Proxy_Probability': round(prob_given_ant, 3),
                                'Proxy_Lift': round(proxy_lift, 2)
                            })

        report_df = pd.DataFrame(proxy_results)
        if not report_df.empty:
            report_df = report_df.sort_values(by='Proxy_Lift', ascending=False).drop_duplicates()
        return report_df

class FairnessVisualizer:
    """ Plots horizontal bar charts visualizing the worst-case confidence disparities. """
    @staticmethod
    def plot_bias_barchart(report_df, output_dir, top_n=10):
        print("Generating fairness visualizations...")

        if report_df.empty:
            print("No data available to plot.")
            return

        top_rules = report_df.head(top_n).copy()
        # Reverse order so the highest bias is at the top of the plot
        top_rules = top_rules.iloc[::-1]

        def format_rule(row):
            rule_text = row['Rule']
            if ' -> ' in rule_text:
                ant, con = rule_text.split(' -> ')
                ant_wrapped = '\n'.join(textwrap.wrap(ant, width=35))
                con_wrapped = '\n'.join(textwrap.wrap(con, width=35))
                rule_fmt = f"{ant_wrapped}\n -> {con_wrapped}"
            else:
                rule_fmt = '\n'.join(textwrap.wrap(rule_text, width=40))
            
            # Prepend the sensitive feature being analyzed for clarity
            return f"[{row['Sensitive_Feature']}]\n{rule_fmt}"

        rules = top_rules.apply(format_rule, axis=1).values
        val_max = top_rules['Max_Conf'].values
        val_min = top_rules['Min_Conf'].values
        label_max = top_rules['Favored_Group'].values
        label_min = top_rules['Penalized_Group'].values

        y = np.arange(len(rules))
        width = 0.35

        # Dynamic figure height based on the number of rules
        fig, ax = plt.subplots(figsize=(12, max(8, len(rules) * 1.0)))
        
        bars_max = ax.barh(y - width/2, val_max, width, color='skyblue', label='Favored Group')
        bars_min = ax.barh(y + width/2, val_min, width, color='salmon', label='Penalized Group')

        ax.set_xlabel('Rule Confidence', fontsize=12)
        ax.set_title('Top Biased Rules: Worst-Case Confidence Disparity', fontsize=14, pad=15)
        ax.set_yticks(y)
        ax.set_yticklabels(rules, fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        # Annotate bars with the specific demographic group names
        for i, (b_max, b_min) in enumerate(zip(bars_max, bars_min)):
            ax.text(b_max.get_width() + 0.01, b_max.get_y() + b_max.get_height()/2,
                    label_max[i], va='center', ha='left', fontsize=9, color='black')
            ax.text(b_min.get_width() + 0.01, b_min.get_y() + b_min.get_height()/2,
                    label_min[i], va='center', ha='left', fontsize=9, color='black')

        # Extend x-axis slightly to ensure annotations are not cut off
        ax.set_xlim(0, max(val_max) * 1.15)

        plt.tight_layout()
        plot_path = output_dir / "fairness_bias_barchart.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    # Detect the environment (Local vs Colab)
    if Path("/content").exists():
        data_dir = Path("/content/data")
        results_dir = Path("/content/results")
    else:
        base_dir = Path(__file__).resolve().parent.parent
        data_dir = base_dir / "data"
        results_dir = base_dir / "results"

    # Ensure that the folders always exist
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    transactions_file = results_dir / "transactions_values.csv"
    rules_file = results_dir / "microscopic_level_association_rules.csv"
    categorized_data = data_dir / "ACSIncome_NY_2018_categorized.csv" 

    if transactions_file.exists():
        InteractionMiner(min_support=0.01, min_confidence=0.1).execute(transactions_file, results_dir)

    if rules_file.exists():
        auditor_sens = SensitiveAuditMiner(sensitive_features=['SEX', 'RAC1P'])
        if auditor_sens.load_rules(rules_file): 
            auditor_sens.run_audit(results_dir)

    if rules_file.exists() and categorized_data.exists():
        # Passing multiple features to compute worst-case disparity across both Race and Sex
        auditor_fair = FairnessAuditor(sensitive_features=['SEX', 'RAC1P'])
        fairness_report = auditor_fair.audit_rules(rules_file, categorized_data)
        fairness_report = fairness_report.sort_values(by='Conf_Difference', ascending=False)
        
        output_fairness_file = results_dir / "fairness_audit_results.csv"
        fairness_report.to_csv(output_fairness_file, index=False)
        print("Fairness audit completed.")

        proxy_detector = SensitiveProxyDetector(sensitive_features=['SEX', 'RAC1P'])
        proxy_report = proxy_detector.detect_proxies(rules_file, categorized_data, lift_threshold=1.5)
        
        if not proxy_report.empty:
            proxy_report.to_csv(results_dir / "proxy_variables_detected.csv", index=False)
            print("Proxy detection completed.")

        if not fairness_report.empty:
            FairnessVisualizer.plot_bias_barchart(fairness_report, results_dir, top_n=10)