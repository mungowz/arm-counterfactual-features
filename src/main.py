"""
main.py — Pipeline orchestrator
================================
Runs the three pipeline modules in sequence or individually.
"""

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Source file locations
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent

_SRC = {
    1: _HERE / 'create_dataset.py',
    2: _HERE / 'feature_importance.py',
    3: _HERE / 'macroscopic_experiment_association_rules.py',
}

_STEP_NAMES = {
    1: 'Create Dataset',
    2: 'Feature Importance (CategoricalBoCSoR)',
    3: 'Association Rules (FP-Growth)',
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _banner(msg: str, char: str = '=', width: int = 70) -> None:
    print(f'\n{char * width}')
    print(f'  {msg}')
    print(f'{char * width}')

def _load_module(path: Path, name: str):
    spec   = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _check_source_files() -> bool:
    ok = True
    for step, path in _SRC.items():
        if not path.exists():
            print(f'  [ERROR] Source for Step {step} not found: {path}')
            ok = False
    return ok

# ---------------------------------------------------------------------------
# Output pre-existence checks
# ---------------------------------------------------------------------------
def _outputs_step1_exist(survey_year: str, output_dir: str, regions: list[str]) -> bool:
    for reg in regions:
        p = Path(output_dir) / f'acs_income_{reg}_{survey_year}.csv'
        if not p.exists():
            return False
    return True

def _outputs_step2_exist(regions: list[str], k_values: list[int]) -> bool:
    results_dir = _HERE.parent / 'results'
    for region in regions:
        for k in k_values:
            p = results_dir / region / 'important_features' / f'k_{k}' / 'transactions_values.csv'
            if not p.exists():
                return False
    return True

def _inputs_step2_exist(regions: list[str], survey_year: str, output_dir: str) -> bool:
    data_dir = Path(output_dir)
    ok = True
    for region in regions:
        p = data_dir / f'acs_income_{region}_{survey_year}.csv'
        if not p.exists():
            print(f'  [WARNING] Missing input for Step 2: {p}')
            ok = False
    return ok

def _inputs_step3_exist(regions: list[str], k_values: list[int]) -> bool:
    results_dir = _HERE.parent / 'results'
    ok = True
    for region in regions:
        for k in k_values:
            base  = results_dir / region / 'important_features' / f'k_{k}'
            found = (base / 'aggregated_labels_by_sample.csv').exists() or (base / 'labels_only_unique.csv').exists()
            if not found:
                print(f'  [WARNING] Missing input for Step 3: {base}/ (aggregated_labels_by_sample.csv or labels_only_unique.csv)')
                ok = False
    return ok

# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------
def run_step1(args: argparse.Namespace) -> bool:
    _banner(f'STEP 1 — {_STEP_NAMES[1]}')
    if not args.force and _outputs_step1_exist(args.survey_year, args.output_dir, args.regions):
        print(f'  > Output files already present (survey_year={args.survey_year}). Use --force to overwrite.')
        return True

    mod = _load_module(_SRC[1], 'create_dataset')
    t0 = time.perf_counter()
    try:
        mod.main(
            survey_year                = args.survey_year,
            horizon                    = args.horizon,
            random_seed                = args.random_seed,
            output_dir                 = args.output_dir,
            income_threshold_northeast = args.income_threshold_ne,
            income_threshold_south     = args.income_threshold_south,
            income_threshold_usa       = args.income_threshold_usa,
            regions_to_build           = args.regions,
        )
    except Exception:
        print('\n  [ERROR] Step 1 failed:')
        traceback.print_exc()
        return False
    print(f'\n  > Step 1 completed in {time.perf_counter() - t0:.1f}s')
    return True

def run_step2(args: argparse.Namespace) -> bool:
    _banner(f'STEP 2 — {_STEP_NAMES[2]}')
    if not _inputs_step2_exist(args.regions, args.survey_year, args.output_dir):
        print('  [ERROR] Required inputs are missing — run Step 1 first.')
        return False
    if not args.force and _outputs_step2_exist(args.regions, args.k_values):
        print('  > Output files already present for all regions and k values. Use --force to overwrite.')
        return True

    mod = _load_module(_SRC[2], 'feature_importance')
    data_dir = Path(args.output_dir)
    regions  = {r: data_dir / f'acs_income_{r}_{args.survey_year}.csv' for r in args.regions}

    t0 = time.perf_counter()
    try:
        mod.main(
            survey_year    = args.survey_year,
            regions        = regions,
            k_values       = args.k_values,
            perc_threshold = args.perc_threshold,
            target_col     = args.target_col,
            base_dir       = _HERE.parent,
        )
    except Exception:
        print('\n  [ERROR] Step 2 failed:')
        traceback.print_exc()
        return False
    print(f'\n  > Step 2 completed in {time.perf_counter() - t0:.1f}s')
    return True

def run_step3(args: argparse.Namespace) -> bool:
    _banner(f'STEP 3 — {_STEP_NAMES[3]}')
    if not _inputs_step3_exist(args.regions, args.k_values):
        print('  [ERROR] Required inputs are missing — run Step 2 first.')
        return False

    mod = _load_module(_SRC[3], 'macroscopic_experiment_association_rules')
    t0 = time.perf_counter()
    try:
        mod.main(
            regions                  = args.regions,
            k_values                 = args.k_values,
            auto_calibrate           = args.auto_calibrate,
            sup_min                  = args.sup_min,
            sup_max                  = args.sup_max,
            sup_delta                = args.sup_delta,
            conf_min                 = args.conf_min,
            conf_max                 = args.conf_max,
            conf_delta               = args.conf_delta,
            lift_min                 = args.lift_min,
            lift_max                 = args.lift_max,
            lift_delta               = args.lift_delta,
            lift_neutral_half_window = args.lift_neutral_half_window,
            base_dir                 = _HERE.parent,
        )
    except Exception:
        print('\n  [ERROR] Step 3 failed:')
        traceback.print_exc()
        return False
    print(f'\n  > Step 3 completed in {time.perf_counter() - t0:.1f}s')
    return True

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog            = 'main.py',
        description     = 'Pipeline orchestrator: create_dataset → feature_importance → association_rules',
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--steps', nargs='+', type=int, choices=[1, 2, 3], default=[1, 2, 3], metavar='{1,2,3}')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--dry-run', action='store_true')

    g1 = parser.add_argument_group('Step 1: dataset')
    g1.add_argument('--survey-year', default='2024')
    g1.add_argument('--horizon', default='1-Year')
    g1.add_argument('--random-seed', type=int, default=42)
    g1.add_argument('--output-dir', default='data')
    g1.add_argument('--income-threshold-ne', type=int, default=110_000)
    g1.add_argument('--income-threshold-south', type=int, default=90_000)
    g1.add_argument('--income-threshold-usa', type=int, default=100_000, help='USA global income threshold in USD.')

    g2 = parser.add_argument_group('Step 2: feature importance')
    g2.add_argument('--k-values', nargs='+', type=int, default=[1, 3, 5, 7])
    g2.add_argument('--perc-threshold', type=int, default=10)
    g2.add_argument('--target-col', default='INCOME_ABOVE_THRESHOLD')

    parser.add_argument(
        '--regions', nargs='+', choices=['northeast', 'south', 'usa'],
        default=['northeast', 'south'], help='Regions to process (default: northeast south).'
    )

    g3  = parser.add_argument_group('Step 3: association rules')
    cal = g3.add_mutually_exclusive_group()
    cal.add_argument('--auto-calibrate', dest='auto_calibrate', action='store_true', default=True)
    cal.add_argument('--no-auto-calibrate', dest='auto_calibrate', action='store_false')
    g3.add_argument('--sup-min', type=float, default=0.02)
    g3.add_argument('--sup-max', type=float, default=0.50)
    g3.add_argument('--sup-delta', type=float, default=0.02)
    g3.add_argument('--conf-min', type=float, default=0.05)
    g3.add_argument('--conf-max', type=float, default=1.00)
    g3.add_argument('--conf-delta', type=float, default=0.05)
    g3.add_argument('--lift-min', type=float, default=0.0)
    g3.add_argument('--lift-max', type=float, default=5.0)
    g3.add_argument('--lift-delta', type=float, default=0.05)
    g3.add_argument('--lift-neutral-half-window', type=float, default=0.25)

    return parser.parse_args()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    _banner('PIPELINE ORCHESTRATOR')
    print(f'  Steps selected               : {sorted(args.steps)}')
    print(f'  Regions                      : {args.regions}')
    print(f'  Income threshold — Northeast : ${args.income_threshold_ne:,}')
    print(f'  Income threshold — South     : ${args.income_threshold_south:,}')
    print(f'  Income threshold — USA       : ${args.income_threshold_usa:,}')

    if not _check_source_files():
        print('\n  [ERROR] One or more source files are missing. Aborting.')
        sys.exit(1)

    if args.dry_run:
        _banner('DRY RUN — no commands will be executed', char='-')
        for step in sorted(args.steps):
            print(f'  [Step {step}] {_STEP_NAMES[step]}  →  {_SRC[step].name}')
        print()
        return

    _RUNNERS = {1: run_step1, 2: run_step2, 3: run_step3}
    total_t0 = time.perf_counter()
    results = {}

    for step in sorted(args.steps):
        results[step] = _RUNNERS[step](args)
        if not results[step]:
            print(f'\n  [ERROR] Step {step} did not complete successfully. Pipeline interrupted.')
            break

    elapsed = time.perf_counter() - total_t0
    _banner('SUMMARY')
    for step, ok in sorted(results.items()):
        status = '✓  OK' if ok else '✗  FAILED'
        print(f'  Step {step} ({_STEP_NAMES[step]}): {status}')
    print(f'\n  Total elapsed time: {elapsed:.1f}s')
    if not all(results.values()):
        sys.exit(1)

if __name__ == '__main__':
    main()