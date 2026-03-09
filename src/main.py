"""
main.py — Pipeline orchestrator
================================
Esegue in sequenza (o singolarmente) i tre moduli del progetto:

  Step 1 — create_dataset.py
            Scarica i dati ACS e produce due CSV in data/.

  Step 2 — feature_importance.py
            Addestra CatBoost + BoCSoR ed estrae le feature driver di
            confine per ogni regione e ogni valore di k.

  Step 3 — macroscopic_experiment_association_rules.py
            Applica FP-Growth alle transazioni prodotte dal passo 2 e
            salva le regole associative (+ heatmap) in results/.

Utilizzo rapido
---------------
  # Esegui l'intera pipeline
  python main.py

  # Esegui solo uno step (es. solo il passo 2)
  python main.py --steps 2

  # Esegui un sottoinsieme di passi (es. 2 e 3)
  python main.py --steps 2 3

  # Forza la riesecuzione anche se i file di output esistono già
  python main.py --force

  # Mostra la configurazione senza eseguire nulla
  python main.py --dry-run

Opzioni disponibili
-------------------
  --steps   {1,2,3} [...]   Passi da eseguire (default: tutti e tre).
  --force                   Salta i controlli di pre-esistenza degli output
                            e riesegui sempre.
  --dry-run                 Stampa il piano di esecuzione senza eseguire nulla.
  --survey-year YEAR        Anno del sondaggio ACS (default: 2024).
  --k-values K [K ...]      Valori di k per BoCSoR (default: 1 3 5 7).
  --perc-threshold N        Percentile per il filtro di confine (default: 10).
  --regions {northeast,south} [...]
                            Regioni da processare (default: entrambe).
  --auto-calibrate          (Step 3) Calibrazione automatica dei parametri ARM.
  --no-auto-calibrate       (Step 3) Usa i parametri manuali definiti in
                            ARM_MANUAL_PARAMS nel corpo di questo file.
"""

import argparse
import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path


# ---------------------------------------------------------------------------
# Posizione dei sorgenti — modifica se i file si trovano altrove
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent

SRC = {
    1: _HERE / "create_dataset.py",
    2: _HERE / "feature_importance.py",
    3: _HERE / "macroscopic_experiment_association_rules.py",
}

STEP_NAMES = {
    1: "Create Dataset",
    2: "Feature Importance (BoCSoR)",
    3: "Association Rules (FP-Growth)",
}

# ---------------------------------------------------------------------------
# Parametri di default — modificabili da CLI o qui sotto
# ---------------------------------------------------------------------------

DEFAULT_SURVEY_YEAR    = "2024"
DEFAULT_K_VALUES       = [1, 3, 5, 7]
DEFAULT_PERC_THRESHOLD = 10
DEFAULT_REGIONS        = ["northeast", "south"]

# Parametri ARM usati solo quando --no-auto-calibrate è attivo
ARM_MANUAL_PARAMS = dict(
    sup_min   = 0.02, sup_max   = 0.50, sup_delta   = 0.02,
    conf_min  = 0.05, conf_max  = 1.00, conf_delta  = 0.05,
    lift_min  = 0.0,  lift_max  = 5.0,  lift_delta  = 0.05,
    lift_neutral_half_window = 0.25,
)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _banner(msg: str, char: str = "=", width: int = 70) -> None:
    print(f"\n{char * width}")
    print(f"  {msg}")
    print(f"{char * width}")


def _load_module(path: Path, name: str):
    """Carica un file .py come modulo senza eseguire il blocco __main__."""
    spec   = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_source_files() -> bool:
    """Verifica che tutti i sorgenti esistano; restituisce False se manca qualcosa."""
    ok = True
    for step, path in SRC.items():
        if not path.exists():
            print(f"  [ERRORE] Sorgente Step {step} non trovato: {path}")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Controlli di pre-condizione per ogni step
# ---------------------------------------------------------------------------

def _outputs_step1_exist(survey_year: str) -> bool:
    """True se entrambi i CSV del dataset esistono già."""
    data_dir = _HERE / "data"
    ne = data_dir / f"acs_income_northeast_{survey_year}.csv"
    s  = data_dir / f"acs_income_south_{survey_year}.csv"
    return ne.exists() and s.exists()


def _outputs_step2_exist(regions: list, k_values: list, survey_year: str) -> bool:
    """True se almeno un file transactions_values.csv esiste per ogni regione/k."""
    results_dir = _HERE / "results"
    for region in regions:
        for k in k_values:
            p = results_dir / region / "important_features" / f"k_{k}" / "transactions_values.csv"
            if not p.exists():
                return False
    return True


def _inputs_step2_exist(regions: list, survey_year: str) -> bool:
    """True se i CSV richiesti da Step 2 esistono."""
    data_dir = _HERE / "data"
    mapping = {
        "northeast": data_dir / f"acs_income_northeast_{survey_year}.csv",
        "south":     data_dir / f"acs_income_south_{survey_year}.csv",
    }
    ok = True
    for region in regions:
        if region in mapping and not mapping[region].exists():
            print(f"  [AVVISO] Input mancante per Step 2: {mapping[region]}")
            ok = False
    return ok


def _inputs_step3_exist(regions: list, k_values: list) -> bool:
    """True se almeno un file labels esiste per ogni regione/k richiesti."""
    results_dir = _HERE / "results"
    ok = True
    for region in regions:
        for k in k_values:
            base = results_dir / region / "important_features" / f"k_{k}"
            found = (base / "aggregated_labels_by_sample.csv").exists() or \
                    (base / "labels_only_unique.csv").exists()
            if not found:
                print(f"  [AVVISO] Input mancante per Step 3: {base}/ "
                      f"(aggregated_labels_by_sample.csv o labels_only_unique.csv)")
                ok = False
    return ok


# ---------------------------------------------------------------------------
# Runner per ogni step
# ---------------------------------------------------------------------------

def run_step1(survey_year: str, force: bool) -> bool:
    """
    Esegue create_dataset.py:main() dopo aver aggiornato SURVEY_YEAR nel modulo.
    """
    _banner(f"STEP 1 — {STEP_NAMES[1]}")

    if not force and _outputs_step1_exist(survey_year):
        print(f"  > Output già presenti (survey_year={survey_year}).")
        print(f"    Usa --force per sovrascriverli.")
        return True

    mod = _load_module(SRC[1], "create_dataset")
    # Override del parametro anno nel modulo caricato
    mod.SURVEY_YEAR = survey_year

    t0 = time.perf_counter()
    try:
        mod.main()
    except Exception:
        print("\n  [ERRORE] Step 1 fallito:")
        traceback.print_exc()
        return False

    print(f"\n  > Step 1 completato in {time.perf_counter() - t0:.1f}s")
    return True


def run_step2(survey_year: str, k_values: list, perc_threshold: int,
              regions: list, force: bool) -> bool:
    """
    Esegue feature_importance.py ripetendo il loop per ogni regione,
    rispettando i parametri k_values e perc_threshold passati da CLI.
    """
    _banner(f"STEP 2 — {STEP_NAMES[2]}")

    if not _inputs_step2_exist(regions, survey_year):
        print("  [ERRORE] Input mancanti — esegui prima Step 1.")
        return False

    if not force and _outputs_step2_exist(regions, k_values, survey_year):
        print("  > Output già presenti per tutte le regioni/k richiesti.")
        print("    Usa --force per sovrascriverli.")
        return True

    mod = _load_module(SRC[2], "feature_importance")

    base_dir    = _HERE
    data_dir    = base_dir / "data"
    results_dir = base_dir / "results"

    regions_map = {
        "northeast": data_dir / f"acs_income_northeast_{survey_year}.csv",
        "south":     data_dir / f"acs_income_south_{survey_year}.csv",
    }

    # Tee per il log (già presente nel modulo originale)
    import io as _io
    class _TeeWriter:
        def __init__(self, orig):
            self._orig = orig
            self._buf  = _io.StringIO()
        def write(self, text):
            self._orig.write(text)
            self._buf.write(text)
        def flush(self):
            self._orig.flush()
        def getvalue(self):
            return self._buf.getvalue()

    tee = _TeeWriter(sys.stdout)
    sys.stdout = tee

    success = True
    t0 = time.perf_counter()
    try:
        for region in regions:
            data_path  = regions_map.get(region)
            output_dir = results_dir / region / "important_features"
            output_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "=" * 70)
            print(f"COUNTERFACTUAL EXTRACTION — {region.upper()}")
            print("=" * 70 + "\n")

            if data_path is None or not data_path.exists():
                print(f"  > Errore: {data_path} non trovato — salto regione.")
                continue

            k_labels_map = mod.run_for_k_values(
                k_values       = k_values,
                data_path      = data_path,
                output_base_dir= output_dir,
                target_col     = "INCOME_ABOVE_THRESHOLD",
                perc_threshold = perc_threshold,
            )

            print("  > k_labels_map pronto:")
            for k, path in k_labels_map.items():
                print(f"    k={k:>2} -> {path}")

    except Exception:
        print("\n  [ERRORE] Step 2 fallito:")
        traceback.print_exc()
        success = False
    finally:
        sys.stdout = tee._orig

    # Salva log
    full_log = tee.getvalue()
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "feature_importance_log.txt"
    log_path.write_text(full_log, encoding="utf-8")
    print(f"  > Log salvato in {log_path}")

    for region in regions:
        region_dir = results_dir / region / "important_features"
        if region_dir.exists():
            marker = f"COUNTERFACTUAL EXTRACTION — {region.upper()}"
            start  = full_log.find(marker)
            if start != -1:
                next_start = full_log.find("COUNTERFACTUAL EXTRACTION", start + len(marker))
                snippet    = full_log[start:next_start] if next_start != -1 else full_log[start:]
                (region_dir / "feature_importance_log.txt").write_text(snippet, encoding="utf-8")

    if success:
        print(f"\n  > Step 2 completato in {time.perf_counter() - t0:.1f}s")
    return success


def run_step3(k_values: list, regions: list, auto_calibrate: bool, force: bool) -> bool:
    """
    Esegue macroscopic_experiment_association_rules.py per ogni regione.
    """
    _banner(f"STEP 3 — {STEP_NAMES[3]}")

    if not _inputs_step3_exist(regions, k_values):
        print("  [ERRORE] Input mancanti — esegui prima Step 2.")
        return False

    mod = _load_module(SRC[3], "macroscopic_experiment_association_rules")

    base_dir    = _HERE
    results_dir = base_dir / "results"

    # Parametri ARM
    p = ARM_MANUAL_PARAMS
    conf = dict(
        auto_calibrate          = auto_calibrate,
        sup_min                 = p["sup_min"],
        sup_max                 = p["sup_max"],
        sup_delta               = p["sup_delta"],
        conf_min                = p["conf_min"],
        conf_max                = p["conf_max"],
        conf_delta              = p["conf_delta"],
        lift_min                = p["lift_min"],
        lift_max                = p["lift_max"],
        lift_delta              = p["lift_delta"],
        lift_neutral_half_window= p["lift_neutral_half_window"],
    )

    exp_label = mod._experiment_label(
        auto_calibrate          = auto_calibrate,
        sup_min                 = p["sup_min"],
        sup_max                 = p["sup_max"],
        sup_delta               = p["sup_delta"],
        conf_min                = p["conf_min"],
        conf_max                = p["conf_max"],
        conf_delta              = p["conf_delta"],
        lift_min                = p["lift_min"],
        lift_max                = p["lift_max"],
        lift_delta              = p["lift_delta"],
        lift_neutral_half_window= p["lift_neutral_half_window"],
    )

    success = True
    t0 = time.perf_counter()
    try:
        import datetime as _dt
        for region in regions:
            important_features_dir = results_dir / region / "important_features"
            ar_output_dir = results_dir / region / "association_rules" / exp_label
            ar_output_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "=" * 70)
            print(f"ASSOCIATION RULES — {region.upper()}")
            print(f"Experiment: {exp_label}")
            print("=" * 70 + "\n")

            # Costruzione k_labels_map (aggregated > original)
            k_labels_map = {}
            for k in k_values:
                p_agg  = important_features_dir / f"k_{k}" / "aggregated_labels_by_sample.csv"
                p_orig = important_features_dir / f"k_{k}" / "labels_only_unique.csv"
                if p_agg.exists():
                    k_labels_map[k] = p_agg
                elif p_orig.exists():
                    k_labels_map[k] = p_orig

            if not k_labels_map:
                print(f"  > Nessun file labels trovato in {important_features_dir}")
                print("    Esegui prima Step 2.")
                continue

            print(f"  > Labels trovati per k = {sorted(k_labels_map.keys())}")

            k_summaries = mod.run_k_comparison(
                k_labels_map = k_labels_map,
                output_dir   = ar_output_dir,
                **conf,
            )

            # Post-run summary
            k_max_per_k = {
                k: int(sdf["Number_of_Rules"].max())
                for k, sdf in k_summaries.items()
                if not sdf.empty and "Number_of_Rules" in sdf.columns
            }
            sum_rules = sum(k_max_per_k.values())
            max_rules = max(k_max_per_k.values()) if k_max_per_k else 0
            best_k    = max(k_max_per_k, key=k_max_per_k.get) if k_max_per_k else None

            log_path = ar_output_dir / "experiment_log.txt"
            with open(log_path, "a") as f:
                f.write(f"\n{'='*70}\n")
                f.write("Post-run Summary (main.py):\n")
                f.write(f"{'-'*60}\n")
                f.write(f"  k values ran              : {sorted(k_summaries.keys())}\n")
                f.write(f"  Sum of max rules across k : {sum_rules}\n")
                f.write(f"  Max rules in best combo   : {max_rules}  (k={best_k})\n")
                f.write(f"  Completed at              : "
                        f"{_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    except Exception:
        print("\n  [ERRORE] Step 3 fallito:")
        traceback.print_exc()
        success = False

    if success:
        print(f"\n  > Step 3 completato in {time.perf_counter() - t0:.1f}s")
    return success


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog        = "main.py",
        description = "Pipeline orchestrator: create_dataset → feature_importance → association_rules",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
Esempi:
  python main.py                        # intera pipeline
  python main.py --steps 1              # solo download dataset
  python main.py --steps 2 3            # solo BoCSoR + ARM (dataset già presente)
  python main.py --steps 3 --no-auto-calibrate   # ARM con parametri manuali
  python main.py --force                # riesegui tutto sovrascrivendo
  python main.py --dry-run              # mostra il piano senza eseguire
        """,
    )

    parser.add_argument(
        "--steps", nargs="+", type=int, choices=[1, 2, 3],
        default=[1, 2, 3],
        metavar="{1,2,3}",
        help="Passi da eseguire. Default: tutti (1 2 3).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Riesegui anche se gli output esistono già.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Mostra il piano di esecuzione senza eseguire nulla.",
    )
    parser.add_argument(
        "--survey-year", default=DEFAULT_SURVEY_YEAR,
        metavar="YEAR",
        help=f"Anno ACS (default: {DEFAULT_SURVEY_YEAR}).",
    )
    parser.add_argument(
        "--k-values", nargs="+", type=int, default=DEFAULT_K_VALUES,
        metavar="K",
        help=f"Valori di k per BoCSoR (default: {DEFAULT_K_VALUES}).",
    )
    parser.add_argument(
        "--perc-threshold", type=int, default=DEFAULT_PERC_THRESHOLD,
        metavar="N",
        help=f"Percentile per boundary filter (default: {DEFAULT_PERC_THRESHOLD}).",
    )
    parser.add_argument(
        "--regions", nargs="+", choices=["northeast", "south"],
        default=DEFAULT_REGIONS,
        help=f"Regioni da processare (default: {DEFAULT_REGIONS}).",
    )

    cal_group = parser.add_mutually_exclusive_group()
    cal_group.add_argument(
        "--auto-calibrate", dest="auto_calibrate", action="store_true", default=True,
        help="(Step 3) Calibrazione automatica dei parametri ARM (default).",
    )
    cal_group.add_argument(
        "--no-auto-calibrate", dest="auto_calibrate", action="store_false",
        help="(Step 3) Usa i parametri manuali in ARM_MANUAL_PARAMS.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    _banner("PIPELINE ORCHESTRATOR", width=70)
    print(f"  Steps selezionati : {sorted(args.steps)}")
    print(f"  Survey year       : {args.survey_year}")
    print(f"  k values          : {args.k_values}")
    print(f"  Perc threshold    : {args.perc_threshold}")
    print(f"  Regioni           : {args.regions}")
    print(f"  Auto-calibrate    : {args.auto_calibrate}")
    print(f"  Force             : {args.force}")
    print(f"  Dry-run           : {args.dry_run}")
    print(f"  Working dir       : {_HERE}")

    if not _check_source_files():
        print("\n  [ERRORE] Uno o più sorgenti mancanti. Interruzione.")
        sys.exit(1)

    if args.dry_run:
        _banner("DRY RUN — nessun comando verrà eseguito", char="-")
        for step in sorted(args.steps):
            print(f"  [Step {step}] {STEP_NAMES[step]}  → {SRC[step].name}")
        print()
        return

    total_t0 = time.perf_counter()
    results  = {}

    for step in sorted(args.steps):
        if step == 1:
            results[1] = run_step1(
                survey_year = args.survey_year,
                force       = args.force,
            )
        elif step == 2:
            results[2] = run_step2(
                survey_year     = args.survey_year,
                k_values        = args.k_values,
                perc_threshold  = args.perc_threshold,
                regions         = args.regions,
                force           = args.force,
            )
        elif step == 3:
            results[3] = run_step3(
                k_values       = args.k_values,
                regions        = args.regions,
                auto_calibrate = args.auto_calibrate,
                force          = args.force,
            )

        if not results[step]:
            print(f"\n  [ERRORE] Step {step} non completato. Interruzione della pipeline.")
            break

    # Riepilogo finale
    elapsed = time.perf_counter() - total_t0
    _banner("RIEPILOGO FINALE")
    for step, ok in sorted(results.items()):
        status = "✓  OK" if ok else "✗  FALLITO"
        print(f"  Step {step} ({STEP_NAMES[step]}): {status}")
    print(f"\n  Tempo totale: {elapsed:.1f}s")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()