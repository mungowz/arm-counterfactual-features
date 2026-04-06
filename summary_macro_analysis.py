import pandas as pd
import numpy as np
from pathlib import Path
import re
import itertools
import argparse

def extract_macro_features(item_string):
    if pd.isna(item_string): return set()
    s = str(item_string).replace("frozenset", "")
    return set(re.findall(r'[A-Za-z0-9_]+', s))

def main():
    parser = argparse.ArgumentParser(description="Analisi Macroscopica XAI - Grid Search")
    parser.add_argument("--min-supp", type=float, default=0.05, help="Supporto minimo di partenza")
    parser.add_argument("--max-supp", type=float, default=0.20, help="Supporto massimo")
    parser.add_argument("--step-supp", type=float, default=0.05, help="Step per il supporto")
    
    parser.add_argument("--min-conf", type=float, default=0.50, help="Confidenza minima di partenza")
    parser.add_argument("--max-conf", type=float, default=1.00, help="Confidenza massima")
    parser.add_argument("--step-conf", type=float, default=0.10, help="Step per la confidenza")
    
    parser.add_argument("--lift", type=float, default=1.25, help="Soglia minima per il Lift")
    args = parser.parse_args()

    # Generazione griglie (aggiungiamo step/2 al max per assicurarci che numpy includa l'ultimo valore)
    support_thresholds = np.arange(args.min_supp, args.max_supp + (args.step_supp/2), args.step_supp).round(3).tolist()
    confidence_thresholds = np.arange(args.min_conf, args.max_conf + (args.step_conf/2), args.step_conf).round(3).tolist()
    lift_thresholds = [args.lift]

    print("="*50)
    print(" ANALISI MACROSCOPICA")
    print("="*50)
    print(f"Griglia Supporto: {support_thresholds}")
    print(f"Griglia Confidenza: {confidence_thresholds}")
    print(f"Lift: >= {args.lift}")
    print("-" * 50)

    actionable_vars = {"SCHL", "COW", "WKHP", "OCCP"}
    sensitive_vars = {"SEX", "RAC1P"}
    allowed_fairness_vars = actionable_vars.union(sensitive_vars)

    base_dir = Path("results")
    all_files = list(base_dir.rglob("arm_*rules.csv")) 
    print(f"Trovati {len(all_files)} file. Elaborazione in corso...\n")

    risultati = []
    
    totale_assoluto_regole = 0
    totale_assoluto_act = 0
    totale_assoluto_fair = 0

    for file_path in all_files:
        try:
            parts = file_path.parts
            stato, percentile, modello = parts[1], int(parts[5].replace("pct", "")), parts[6]
            k_val = "all" if parts[-2] == "all_k" else parts[-2].replace("k", "")
        except Exception:
            continue

        df = pd.read_csv(file_path)
        if df.empty: continue

        df['all_features'] = df.apply(
            lambda row: extract_macro_features(row['antecedents']).union(extract_macro_features(row['consequents'])), axis=1
        )
        df['is_actionable'] = df['all_features'].apply(lambda x: x.issubset(actionable_vars) and len(x) > 0)
        
        def is_fairness_rule(features_set):
            if not features_set.issubset(allowed_fairness_vars): return False
            return (len(features_set.intersection(actionable_vars)) > 0) and (len(features_set.intersection(sensitive_vars)) > 0)

        df['is_fairness'] = df['all_features'].apply(is_fairness_rule)

        # Calcolo Totale Globale sull'universo di partenza
        df_base = df[(df['support'] >= args.min_supp) & (df['confidence'] >= args.min_conf) & (df['lift'] >= args.lift)]
        totale_assoluto_regole += len(df_base)
        totale_assoluto_act += df_base['is_actionable'].sum()
        totale_assoluto_fair += df_base['is_fairness'].sum()

        for min_sup, min_conf, min_lift in itertools.product(support_thresholds, confidence_thresholds, lift_thresholds):
            df_filtered = df[(df['support'] >= min_sup) & (df['confidence'] >= min_conf) & (df['lift'] >= min_lift)]
            totale = len(df_filtered)
            if totale == 0: continue
                
            act_count = df_filtered['is_actionable'].sum()
            fair_count = df_filtered['is_fairness'].sum()
            
            if act_count == 0 and fair_count == 0:
                continue
            
            risultati.append({
                "Stato": stato, "Modello": modello, "Percentile": percentile, "K": k_val, 
                "Supp": min_sup, "Conf": min_conf, "Lift": min_lift,
                "Tot_Regole": totale, "Regole_Act": act_count, "Regole_Fair": fair_count,
                "%_Actionable": round((act_count / totale) * 100, 2),
                "%_Fairness": round((fair_count / totale) * 100, 2)
            })

    if risultati:
        summary_df = pd.DataFrame(risultati).sort_values(by=["Stato", "Modello", "Percentile", "K", "Supp", "Conf"])
        
        pct_act_glob = round((totale_assoluto_act / totale_assoluto_regole) * 100, 2) if totale_assoluto_regole > 0 else 0
        pct_fair_glob = round((totale_assoluto_fair / totale_assoluto_regole) * 100, 2) if totale_assoluto_regole > 0 else 0
        
        global_row = pd.DataFrame([{
            "Stato": ">> TOTALE GLOBALE <<", "Modello": "-", "Percentile": "-", "K": "-",
            "Supp": args.min_supp, "Conf": args.min_conf, "Lift": args.lift,
            "Tot_Regole": totale_assoluto_regole, "Regole_Act": totale_assoluto_act, "Regole_Fair": totale_assoluto_fair,
            "%_Actionable": pct_act_glob, "%_Fairness": pct_fair_glob
        }])
        
        summary_df = pd.concat([summary_df, global_row], ignore_index=True)
        
        out_file = f"SUMMARY_macro_Sup{args.min_supp}_Conf{args.min_conf}.csv"
        summary_df.to_csv(out_file, index=False)
        print(f"Salvato con successo: {out_file}")
    else:
        print("Nessun risultato compatibile.")

if __name__ == "__main__":
    main()