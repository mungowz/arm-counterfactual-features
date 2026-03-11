"""
main.py — Pipeline orchestrator
================================
Runs the three pipeline modules in sequence or individually via a unified CLI.

Pipeline
--------
    Step 1: create_dataset.py
        Download ACS PUMS data, encode features, write one CSV per region.

    Step 2: feature_importance.py
        Train CategoricalBoCSoR, extract counterfactual drivers, write
        transaction CSVs for each (region, k) combination.

    Step 3: macroscopic_experiment_association_rules.py
        Run FP-Growth association-rule mining on the transaction CSVs,
        produce rules + heatmaps for each (region, k) combination.

Design principles
-----------------
- All three step modules are loaded at runtime via importlib.util so that
  this orchestrator does not need to import them at module level.  This
  means missing dependencies in one module do not prevent the others from
  running.
- Output pre-existence checks: if outputs already exist and --force is not
  set, the step is skipped with a message.  This makes re-running cheaper
  when only one step needs to be repeated.
- All paths are anchored to _HERE (the directory containing this file),
  not to the current working directory.  This means the pipeline can be
  invoked from any directory.
- The banner prints only the thresholds for the selected regions to avoid
  cluttering the output with irrelevant defaults.

CLI usage (quick reference)
----------------------------
  # Full pipeline, default regions (NE + South):
  python main.py

  # All three regions:
  python main.py --regions northeast south usa

  # USA only, skip dataset download (CSV already present):
  python main.py --regions usa --steps 2 3

  # Force re-run all steps even if outputs exist:
  python main.py --force

  # Dry run — print plan without executing:
  python main.py --dry-run

  # Custom k values and boundary percentile:
  python main.py --k-values 3 7 --perc-threshold 5

  # ARM with manual grid (disable auto-calibration):
  python main.py --no-auto-calibrate --sup-min 0.05 --sup-max 0.40

  # Exclude additional columns from feature matrix:
  python main.py --metadata-cols YEAR STATE_GROUP

  # Exclude nothing from feature matrix:
  python main.py --metadata-cols
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

# _HERE: absolute path to the directory containing this script.
# All relative paths in the pipeline are resolved against _HERE, not against
# the current working directory.  This makes the orchestrator CWD-independent.
_HERE = Path(__file__).resolve().parent

# Map step numbers to their source file paths.  Used by _check_source_files()
# and _load_module() to load each step module at runtime.
_SRC = {
    1: _HERE / 'create_dataset.py',
    2: _HERE / 'feature_importance.py',
    3: _HERE / 'macroscopic_experiment_association_rules.py',
}

# Human-readable step names used in the banner and summary output.
_STEP_NAMES = {
    1: 'Create Dataset',
    2: 'Feature Importance (CategoricalBoCSoR)',
    3: 'Association Rules (FP-Growth)',
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _banner(msg: str, char: str = '=', width: int = 70) -> None:
    """Print a prominent section header to stdout."""
    print(f'\n{char * width}')
    print(f'  {msg}')
    print(f'{char * width}')


def _load_module(path: Path, name: str):
    """
    Dynamically load a Python source file as a module without installing it.

    Using importlib.util.spec_from_file_location allows each step module to
    be loaded on demand rather than at import time.  Benefits:
    - A broken dependency in one module (e.g. missing CUDA library) does not
      crash the orchestrator when that step is not being run.
    - The module's __file__ attribute is set correctly, so its own
      Path(__file__).resolve().parent resolves as expected.

    Parameters
    ----------
    path : absolute path to the .py file
    name : module name to register in sys.modules
    """
    spec   = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module   # register before exec so relative imports and re-use work correctly
    spec.loader.exec_module(module)
    return module


def _check_source_files() -> bool:
    """
    Verify that all step source files exist on disk.

    Returns True if all files are present, False otherwise.
    Prints an error message for each missing file so the user knows
    exactly which script is absent.
    """
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
    """
    Return True if all expected Step 1 output CSVs already exist on disk.

    Used to skip Step 1 when re-running the pipeline after a completed download.
    The check verifies one CSV per requested region; if any is missing, the
    step is not considered complete.

    Parameters
    ----------
    survey_year : ACS survey year string (e.g. '2024')
    output_dir  : directory where CSVs are expected
    regions     : list of region names (e.g. ['northeast', 'south'])
    """
    for reg in regions:
        p = Path(output_dir) / f'acs_income_{reg}_{survey_year}.csv'
        if not p.exists():
            return False
    return True


def _outputs_step2_exist(regions: list[str], k_values: list[int]) -> bool:
    """
    Return True if all expected Step 2 transaction CSVs exist for every
    (region, k) combination.

    The sentinel file checked is transactions_values.csv inside each k_{k}/
    subdirectory.  If it exists, the assumption is that all downstream files
    (labels_only_unique.csv, aggregated_labels_by_sample.csv, etc.) are also
    present from the same run.

    Parameters
    ----------
    regions  : list of region names
    k_values : list of k values used in the experiment
    """
    results_dir = _HERE.parent / 'results'
    for region in regions:
        for k in k_values:
            p = (
                results_dir / region / 'important_features'
                / f'k_{k}' / 'transactions_values.csv'
            )
            if not p.exists():
                return False
    return True


def _inputs_step2_exist(
    regions: list[str], survey_year: str, output_dir: str
) -> bool:
    """
    Verify that all input CSVs required by Step 2 are present.

    Called before attempting to run Step 2 to give an informative error
    instead of a cryptic FileNotFoundError inside feature_importance.py.

    Returns True only if every region's CSV exists.
    """
    # Resolve path consistently with run_step2 (Path(args.output_dir) — no CWD magic).
    data_dir = Path(output_dir)
    ok = True
    for region in regions:
        p = data_dir / f'acs_income_{region}_{survey_year}.csv'
        if not p.exists():
            print(f'  [WARNING] Missing input for Step 2: {p}')
            ok = False
    return ok


def _outputs_step3_exist(regions: list[str], k_values: list[int]) -> bool:
    """
    Return True if all expected Step 3 output rule CSVs exist for every
    (region, k) combination.

    NOTE: Step 3 currently has no skip logic — ARM is fast enough that
    re-running is not a burden.  This function is provided for symmetry
    with the other output-check helpers and to make adding a future
    --skip-arm flag straightforward without restructuring run_step3().

    The sentinel file checked is association_rules.csv inside each k_{k}/
    subdirectory under results/<region>/association_rules/.
    """
    results_dir = _HERE.parent / 'results'
    for region in regions:
        for k in k_values:
            p = results_dir / region / 'association_rules' / f'k_{k}' / 'association_rules.csv'
            if not p.exists():
                return False
    return True


def _inputs_step3_exist(regions: list[str], k_values: list[int]) -> bool:
    """
    Verify that Step 3 has at least one valid input file per region.

    It is normal for low k values (especially k=1) to produce no transactions
    — and therefore no output files — because a single CF neighbour rarely
    drives enough co-occurring features to form association rules.  Those k
    values are legitimately absent and should not block Step 3.

    Logic
    -----
    - For each k, check whether aggregated_labels_by_sample.csv or
      labels_only_unique.csv exists (same preference order as Step 3).
    - A WARNING is printed for each missing k (informational only).
    - Returns False only if NO k value has any valid input for a given region
      (i.e. Step 2 has never been run or failed entirely for that region).
      In that case Step 3 cannot do anything useful and is blocked.
    """
    results_dir = _HERE.parent / 'results'
    ok = True
    for region in regions:
        valid_ks   = []
        missing_ks = []
        for k in k_values:
            base  = results_dir / region / 'important_features' / f'k_{k}'
            found = (
                (base / 'aggregated_labels_by_sample.csv').exists()
                or (base / 'labels_only_unique.csv').exists()
            )
            if found:
                valid_ks.append(k)
            else:
                missing_ks.append(k)

        if missing_ks:
            # Low-k values often produce no output — warn but do not block.
            print(
                f'  [INFO] {region}: no input files for k={missing_ks} '
                f'(normal for low k — sparse transactions produce no drivers). '
                f'Step 3 will skip those k values.'
            )
        if not valid_ks:
            # Every k is missing for this region — Step 2 was never run or failed.
            print(
                f'  [ERROR] {region}: no valid input files found for any k value. '
                f'Run Step 2 first.'
            )
            ok = False
        else:
            print(f'  [INFO] {region}: valid inputs found for k={valid_ks}.')
    return ok

# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def run_step1(args: argparse.Namespace) -> bool:
    """
    Run create_dataset.py — download and preprocess ACS data.

    Skip condition: all expected output CSVs already exist and --force is
    not set.  This avoids re-downloading multi-GB ACS archives on re-runs.

    Parameters
    ----------
    args : parsed CLI namespace containing all Step 1 parameters.

    Returns True on success, False on failure.
    """
    _banner(f'STEP 1 — {_STEP_NAMES[1]}')

    # Skip if outputs already exist, unless the user explicitly requests a re-run.
    if not args.force and _outputs_step1_exist(
        args.survey_year, args.output_dir, args.regions
    ):
        print(
            f'  > Output files already present (survey_year={args.survey_year}). '
            f'Use --force to overwrite.'
        )
        return True

    mod = _load_module(_SRC[1], 'create_dataset')
    t0  = time.perf_counter()
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
    """
    Run feature_importance.py — CategoricalBoCSoR counterfactual extraction.

    Prerequisites: Step 1 output CSVs must exist for all requested regions.
    Skip condition: all expected output transaction CSVs exist and --force
    is not set.

    Parameters
    ----------
    args : parsed CLI namespace containing all Step 2 parameters.

    Returns True on success, False on failure.
    """
    _banner(f'STEP 2 — {_STEP_NAMES[2]}')

    # Abort early with a clear error if the input CSVs are missing.
    if not _inputs_step2_exist(args.regions, args.survey_year, args.output_dir):
        print('  [ERROR] Required inputs are missing — run Step 1 first.')
        return False

    # Skip if outputs already exist, unless forced.
    if not args.force and _outputs_step2_exist(args.regions, args.k_values):
        print(
            '  > Output files already present for all regions and k values. '
            'Use --force to overwrite.'
        )
        return True

    mod = _load_module(_SRC[2], 'feature_importance')

    # Build the regions dict expected by feature_importance.main():
    # {region_name: Path_to_CSV}.
    data_dir = Path(args.output_dir)
    regions  = {
        r: data_dir / f'acs_income_{r}_{args.survey_year}.csv'
        for r in args.regions
    }

    t0 = time.perf_counter()
    try:
        mod.main(
            survey_year    = args.survey_year,
            regions        = regions,
            k_values       = args.k_values,
            perc_threshold = args.perc_threshold,
            target_col     = args.target_col,
            metadata_cols  = args.metadata_cols,   # columns excluded from feature matrix X
            base_dir       = _HERE.parent,                # results/ and data/ resolved under _HERE.parent (project root)
        )
    except Exception:
        print('\n  [ERROR] Step 2 failed:')
        traceback.print_exc()
        return False

    print(f'\n  > Step 2 completed in {time.perf_counter() - t0:.1f}s')
    return True


def run_step3(args: argparse.Namespace) -> bool:
    """
    Run macroscopic_experiment_association_rules.py — FP-Growth ARM.

    Prerequisites: Step 2 output CSVs must exist for all (region, k) pairs.
    No skip condition: ARM is fast enough that re-running is not a burden,
    and the user may want to re-run with different grid parameters.

    Parameters
    ----------
    args : parsed CLI namespace containing all Step 3 parameters.

    Returns True on success, False on failure.
    """
    _banner(f'STEP 3 — {_STEP_NAMES[3]}')

    # Abort early if input files are missing.
    if not _inputs_step3_exist(args.regions, args.k_values):
        print('  [ERROR] Required inputs are missing — run Step 2 first.')
        return False

    mod = _load_module(_SRC[3], 'macroscopic_experiment_association_rules')
    t0  = time.perf_counter()
    try:
        mod.main(
            regions                  = args.regions,
            k_values                 = args.k_values,
            auto_calibrate           = args.auto_calibrate,
            # Support grid
            sup_min                  = args.sup_min,
            sup_max                  = args.sup_max,
            sup_delta                = args.sup_delta,
            # Confidence grid
            conf_min                 = args.conf_min,
            conf_max                 = args.conf_max,
            conf_delta               = args.conf_delta,
            # Lift grid
            lift_min                 = args.lift_min,
            lift_max                 = args.lift_max,
            lift_delta               = args.lift_delta,
            lift_neutral_half_window = args.lift_neutral_half_window,
            base_dir                 = _HERE.parent,   # results/ and data/ resolved under _HERE.parent (project root)
        )
    except Exception:
        print('\n  [ERROR] Step 3 failed:')
        traceback.print_exc()
        return False

    print(f'\n  > Step 3 completed in {time.perf_counter() - t0:.1f}s')
    return True

# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments and return a populated Namespace.

    Arguments are grouped by pipeline step for readability in --help output.
    All defaults match the documented experiment configuration so that running
    `python main.py` with no arguments produces a sensible baseline run.
    """
    parser = argparse.ArgumentParser(
        prog            = 'main.py',
        description     = (
            'Pipeline orchestrator: '
            'create_dataset → feature_importance → association_rules'
        ),
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )

    # ── Global options ─────────────────────────────────────────────────
    parser.add_argument(
        '--steps', nargs='+', type=int, choices=[1, 2, 3],
        default=[1, 2, 3], metavar='{1,2,3}',
        help='Steps to run (default: all three).',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force overwrite of existing output files.',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print execution plan without actually running any step.',
    )
    parser.add_argument(
        '--regions', nargs='+',
        choices=['northeast', 'south', 'usa'],
        default=['northeast', 'south'],
        help='Regions to process (default: northeast south).',
    )

    # ── Step 1: dataset parameters ─────────────────────────────────────
    g1 = parser.add_argument_group('Step 1: dataset')
    g1.add_argument(
        '--survey-year', default='2024',
        help='ACS survey year (default: 2024).',
    )
    g1.add_argument(
        '--horizon', default='1-Year',
        help='ACS survey horizon (default: 1-Year).',
    )
    g1.add_argument(
        '--random-seed', type=int, default=42,
        help='Random seed for undersampling and train/test split (default: 42).',
    )
    g1.add_argument(
        '--output-dir', default=str(_HERE.parent / 'data'),
        help='Directory for dataset CSV files (default: <project_dir>/data, i.e. one level above src/).',
    )
    g1.add_argument(
        '--income-threshold-ne', type=int, default=110_000,
        help='Annual income threshold (USD) for the Northeast positive class (default: $110,000).',
    )
    g1.add_argument(
        '--income-threshold-south', type=int, default=90_000,
        help='Annual income threshold (USD) for the South positive class (default: $90,000).',
    )
    g1.add_argument(
        '--income-threshold-usa', type=int, default=100_000,
        help='Annual income threshold (USD) for the USA positive class (default: $100,000).',
    )

    # ── Step 2: feature importance parameters ──────────────────────────
    g2 = parser.add_argument_group('Step 2: feature importance')
    g2.add_argument(
        '--k-values', nargs='+', type=int, default=[1, 3, 5, 7],
        help='CF neighbourhood sizes (default: 1 3 5 7).',
    )
    g2.add_argument(
        '--perc-threshold', type=int, default=10,
        help='Boundary filter percentile (default: 10).',
    )
    g2.add_argument(
        '--target-col', default='INCOME_ABOVE_THRESHOLD',
        help='Binary target column name in the CSV (default: INCOME_ABOVE_THRESHOLD).',
    )
    g2.add_argument(
        '--metadata-cols', nargs='*', default=['YEAR'], metavar='COL',
        help=(
            'Columns excluded from feature matrix X in Step 2 '
            '(data-provenance columns, not predictive features). '
            'Default: YEAR. '
            'Note: ST (state) is intentionally kept as a feature by default; '
            'add --metadata-cols YEAR ST to exclude it too. '
            'Pass --metadata-cols with no arguments to exclude nothing.'
        ),
    )

    # ── Step 3: association rules parameters ───────────────────────────
    g3  = parser.add_argument_group('Step 3: association rules')

    # --auto-calibrate and --no-auto-calibrate are mutually exclusive;
    # auto-calibrate is the default.
    cal = g3.add_mutually_exclusive_group()
    cal.add_argument(
        '--auto-calibrate', dest='auto_calibrate',
        action='store_true', default=True,
        help='Calibrate support/lift/confidence bounds from data (default).',
    )
    cal.add_argument(
        '--no-auto-calibrate', dest='auto_calibrate',
        action='store_false',
        help='Use manual grid bounds instead of auto-calibration.',
    )

    # Manual grid bounds (used when --no-auto-calibrate is set, or as
    # floor/ceiling values when auto-calibration is active).
    g3.add_argument('--sup-min',   type=float, default=0.02,
                    help='Min support (default: 0.02).')
    g3.add_argument('--sup-max',   type=float, default=0.50,
                    help='Max support (default: 0.50).')
    g3.add_argument('--sup-delta', type=float, default=0.02,
                    help='Support grid step (default: 0.02).')
    g3.add_argument('--conf-min',   type=float, default=0.05,
                    help='Min confidence / calibration floor (default: 0.05).')
    g3.add_argument('--conf-max',   type=float, default=1.00,
                    help='Max confidence (default: 1.00).')
    g3.add_argument('--conf-delta', type=float, default=0.05,
                    help='Confidence grid step (default: 0.05).')
    g3.add_argument('--lift-min',   type=float, default=0.0,
                    help='Min lift — 0.0 includes negative correlations (default: 0.0).')
    g3.add_argument('--lift-max',   type=float, default=5.0,
                    help='Max lift (default: 5.0).')
    g3.add_argument('--lift-delta', type=float, default=0.05,
                    help='Lift grid step (default: 0.05).')
    g3.add_argument('--lift-neutral-half-window', type=float, default=0.25,
                    help=(
                        'Half-width of the neutral lift window to exclude '
                        '(default: 0.25 → window [0.75, 1.25]).'
                    ))

    return parser.parse_args()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Parse arguments, print the configuration banner, and run the selected steps.

    Each step runner (run_step1, run_step2, run_step3) returns True on success
    and False on failure.  The pipeline halts at the first failed step to
    prevent downstream steps from running on incomplete inputs.

    Exit codes:
        0  — all selected steps completed successfully
        1  — at least one step failed
    """
    args = _parse_args()

    # Ensure the script's own directory is on sys.path so that step modules
    # can import sibling modules (e.g. 'from utils import ...') without error.
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))

    # ── Configuration banner ──────────────────────────────────────────
    # Print only the thresholds for the regions the user actually selected
    # to avoid showing irrelevant defaults in the output.
    _banner('PIPELINE ORCHESTRATOR')
    print(f'  Steps selected               : {sorted(args.steps)}')
    print(f'  Regions                      : {args.regions}')
    print(f'  Survey year                  : {args.survey_year}')
    print(f'  Horizon                      : {args.horizon}')
    print(f'  Random seed                  : {args.random_seed}')
    print(f'  Output dir (datasets)        : {args.output_dir}')
    if 'northeast' in args.regions:
        print(f'  Income threshold — Northeast : ${args.income_threshold_ne:,}')
    if 'south' in args.regions:
        print(f'  Income threshold — South     : ${args.income_threshold_south:,}')
    if 'usa' in args.regions:
        print(f'  Income threshold — USA       : ${args.income_threshold_usa:,}')
    print(f'  k values                     : {args.k_values}')
    print(f'  Perc threshold               : {args.perc_threshold}')
    print(f'  Target column                : {args.target_col}')
    print(f'  Auto-calibrate (ARM)         : {args.auto_calibrate}')
    print(f'  Force re-run                 : {args.force}')
    print(f'  Dry-run                      : {args.dry_run}')
    print(f'  Script directory (src/)      : {_HERE}')

    # ── Dry-run mode: print plan and exit ─────────────────────────────
    # Source file check is intentionally skipped here: dry-run only prints
    # the execution plan and should not fail due to missing scripts.
    if args.dry_run:
        _banner('DRY RUN — no commands will be executed', char='-')
        for step in sorted(args.steps):
            print(f'  [Step {step}] {_STEP_NAMES[step]}  →  {_SRC[step].name}')
        print()
        return

    # Verify all source files exist before doing any work.
    if not _check_source_files():
        print('\n  [ERROR] One or more source files are missing. Aborting.')
        sys.exit(1)

    # ── Execute selected steps in order ───────────────────────────────
    _RUNNERS = {1: run_step1, 2: run_step2, 3: run_step3}
    total_t0  = time.perf_counter()
    results   = {}

    for step in sorted(args.steps):
        results[step] = _RUNNERS[step](args)
        if not results[step]:
            # One step failed — halt the pipeline to avoid running downstream
            # steps on incomplete or missing inputs.
            print(
                f'\n  [ERROR] Step {step} did not complete successfully. '
                f'Pipeline interrupted.'
            )
            break

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.perf_counter() - total_t0
    _banner('SUMMARY')
    for step, ok in sorted(results.items()):
        status = '✓  OK' if ok else '✗  FAILED'
        print(f'  Step {step} ({_STEP_NAMES[step]}): {status}')
    print(f'\n  Total elapsed time: {elapsed:.1f}s')

    # Exit with code 1 if any step failed so calling scripts can detect failure.
    if not all(results.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()