import pandas as pd
import numpy as np
from pathlib import Path
import itertools
import re
import argparse

def parse_macro_string_to_set(s):
    if pd.isna(s): return set()
    return set(re.findall(r'[A-Za-z0-9_]+', str(s).replace("frozenset", "")))

def parse_micro_string_to_macro_set(s):
    if pd.isna(s): return set()
    return {item.split("=")[0].strip() for item in str(s).split(" & ")}

def set_to_string(feature_set):
    return "{" + ", ".join(sorted(list(feature_set))) + "}"

def main():
    parser = argparse.ArgumentParser(description="Analisi Microscopica XAI - Grid Search")
    
    # Parametri per la grid search MICRO
    parser.add_argument("--min-supp", type=float, default=0.01, help="Supporto min MICRO")
    parser.add_argument("--max-supp", type=float, default=0.05, help="Supporto max MICRO")
    parser.add_argument("--step-supp", type=float, default=0.01, help="Step supporto MICRO")
    
    parser.add_argument("--min-conf", type=float, default=0.30, help="Confidenza min MICRO")
    parser.add_argument("--max-conf", type=float, default=0.80, help="Confidenza max MICRO")
    parser.add_argument("--step-conf", type=float, default=0.10, help="Step confidenza MICRO")
    
    parser.add_argument("--lift", type=float, default=1.25, help="Lift min MICRO")
    
    # Parametri per il "Cancello" MACRO
    parser.add_argument("--use-macro-filter", action="store_true", help="Se abilitato, processa solo micro-regole con madri forti")
    parser.add_argument("--macro-min-supp", type=float, default=0.05, help="Supporto min MACRO per il filtro")
    parser.add_argument("--macro-min-conf", type=float, default=0.50, help="Confidenza min MACRO per il filtro")
    parser.add_argument("--macro-lift", type=float, default=1.25, help="Lift min MACRO per il filtro")
    
    args = parser.parse_args()

    support_thresholds = np.arange(args.min_supp, args.max_supp + (args.step_supp/2), args.step_supp).round(3).tolist()
    confidence_thresholds = np.arange(args.min_conf, args.max_conf + (args.step_conf/2), args.step_conf).round(3).tolist()
    lift_thresholds = [args.lift]

    actionable_vars = {"SCHL", "COW", "WKHP", "OCCP"}
    sensitive_vars = {"SEX", "RAC1P"}
    allowed_fairness_vars = actionable_vars.union(sensitive_vars)

    print("="*50)
    print(" ANALISI MICROSCOPICA")
    print("="*50)
    if args.use_macro_filter:
        print(f"Filtro MACRO ATTIVO (Sopravvivono solo madri con Sup>={args.macro_min_supp}, Conf>={args.macro_min_conf})")
    else:
        print("Filtro MACRO DISATTIVATO (Analisi su tutte le micro-regole)")
    print("-" * 50)

    base_dir = Path("results")
    all_micro_files = list(base_dir.rglob("micro_*rules.csv")) 
    print(f"Trovati {len(all_micro_files)} file micro. Elaborazione in corso...\n")

    risultati = []

    for file_path in all_micro_files:
        try:
            parts = file_path.parts
            stato, percentile, modello = parts[1], int(parts[5].replace("pct", "")), parts[6]
            k_val = "all" if parts[-2] == "all_k" else parts[-2].replace("k", "")
            classe = "class1" if "class1" in file_path.name else "class0"
        except Exception:
            continue

        valid_macro_rules = set()
        if args.use_macro_filter:
            macro_files = list(file_path.parent.glob("arm_*rules.csv"))
            if macro_files:
                df_macro = pd.read_csv(macro_files[0])
                if not df_macro.empty:
                    df_m_filt = df_macro[
                        (df_macro['support'] >= args.macro_min_supp) & 
                        (df_macro['confidence'] >= args.macro_min_conf) & 
                        (df_macro['lift'] >= args.macro_lift)
                    ]
                    m_ant_str = df_m_filt['antecedents'].apply(lambda x: set_to_string(parse_macro_string_to_set(x)))
                    m_cons_str = df_m_filt['consequents'].apply(lambda x: set_to_string(parse_macro_string_to_set(x)))
                    valid_macro_rules = set(m_ant_str + " -> " + m_cons_str)
            else:
                continue # Salta se manca il file macro

        df_micro = pd.read_csv(file_path)
        if df_micro.empty: continue

        df_micro['m_ant_set'] = df_micro['antecedents'].apply(parse_micro_string_to_macro_set)
        df_micro['m_cons_set'] = df_micro['consequents'].apply(parse_micro_string_to_macro_set)
        df_micro['macro_rule_string'] = df_micro['m_ant_set'].apply(set_to_string) + " -> " + df_micro['m_cons_set'].apply(set_to_string)

        if args.use_macro_filter:
            df_micro = df_micro[df_micro['macro_rule_string'].isin(valid_macro_rules)]
        if df_micro.empty: continue

        for min_sup, min_conf, min_lift in itertools.product(support_thresholds, confidence_thresholds, lift_thresholds):
            mask = (df_micro['support'] >= min_sup) & (df_micro['confidence'] >= min_conf) & (df_micro['lift'] >= min_lift)
            df_filtered = df_micro[mask]
            if df_filtered.empty: continue
                
            counts = df_filtered['macro_rule_string'].value_counts()
            
            for rule_name, count in counts.items():
                ant_str, cons_str = rule_name.split(" -> ")
                full_features = parse_macro_string_to_set(ant_str).union(parse_macro_string_to_set(cons_str))
                
                is_actionable = full_features.issubset(actionable_vars) and len(full_features) > 0
                is_fairness = full_features.issubset(allowed_fairness_vars) and \
                              (len(full_features.intersection(actionable_vars)) > 0) and \
                              (len(full_features.intersection(sensitive_vars)) > 0)
                
                if not is_actionable and not is_fairness:
                    continue
                    
                risultati.append({
                    "Stato": stato, "Modello": modello, "Percentile": percentile,
                    "K": k_val, "Classe": classe, "Macro_Rule_Madre": rule_name,
                    "Tipo_Regola": "Actionable" if is_actionable else "Fairness",
                    "Supp_Micro": min_sup, "Conf_Micro": min_conf, "Lift_Micro": min_lift,
                    "Tot_Micro_Regole": count
                })

    if risultati:
        summary_df = pd.DataFrame(risultati).sort_values(
            by=["Stato", "Modello", "Percentile", "K", "Classe", "Macro_Rule_Madre", "Supp_Micro", "Conf_Micro"]
        )
        out_file = f"SUMMARY_micro_Sup{args.min_supp}_Conf{args.min_conf}.csv"
        if args.use_macro_filter: out_file = out_file.replace(".csv", "_FILTERED.csv")
            
        summary_df.to_csv(out_file, index=False)
        print(f"Salvato con successo: {out_file}")
    else:
        print("Nessun risultato compatibile.")

if __name__ == "__main__":
    main()