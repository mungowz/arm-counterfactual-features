"""
main.py — Pipeline orchestrator
================================
Runs the five pipeline modules in sequence or individually via a unified CLI.

Pipeline
--------
    Step 1: create_dataset.py
        Download ACS PUMS data, encode features, write one CSV per region.

    Step 2: feature_importance.py
        Train CategoricalBoCSoR, extract counterfactual drivers, write
        transaction CSVs for each (region, k) combination.

    Step 3: macroscopic_experiment_association_rules.py
        Run FP-Growth association-rule mining on label-level transaction CSVs,
        produce label-level rules + heatmaps for each (region, k) combination.

    Step 4: microscopic_experiment_association_rules_values.py
        Run FP-Growth association-rule mining at the value level (LABEL=value
        items), filtered to the labels that appear in Step 3 rules.
        Produces value-level rules + heatmaps under association_rules_values/.
        Requires Step 3 outputs (rules.csv files) to derive the active labels.

    Step 5: fairness_analysis.py
        Compute fairness metrics on the value-level rules produced by Step 4.
        Metrics include disparate impact ratio (4/5 rule), statistical parity
        difference, confidence/support/lift parity, and intersectional analysis
        for every sensitive attribute (default: SEX, RAC1P).
        When the original dataset CSVs are available (auto-discovered from the
        data directory, or supplied explicitly via --fairness-datasets), also
        computes population-level DIR and positive rates directly from the data.
        Outputs: fairness_report.txt, per-metric CSVs, and plots under
        results/{region}/fairness_analysis/.

Design principles
-----------------
- All step modules are loaded at runtime via importlib.util so that this
  orchestrator does not need to import them at module level.  This means
  missing dependencies in one module do not prevent the others from running.
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

  # All five regions:
  python main.py --regions northeast south usa

  # USA only, skip dataset download (CSV already present):
  python main.py --regions usa --steps 2 3 4 5

  # Only value-level ARM (Step 4), after Steps 1-3 already ran:
  python main.py --steps 4

  # Only fairness analysis (Step 5), after Steps 1-4 already ran:
  python main.py --steps 5

  # Force re-run all steps even if outputs exist:
  python main.py --force

  # Dry run — print plan without executing:
  python main.py --dry-run

  # Custom k values and boundary percentile:
  python main.py --k-values 3 7 --perc-threshold 5

  # ARM with manual grid (disable auto-calibration, applies to Steps 3 and 4):
  python main.py --no-auto-calibrate --sup-min 0.05 --sup-max 0.40

  # Exclude additional columns from feature matrix:
  python main.py --metadata-cols YEAR STATE_GROUP

  # Exclude nothing from feature matrix:
  python main.py --metadata-cols

  # Fairness analysis with custom sensitive attributes and privileged groups:
  python main.py --steps 5 --fairness-sensitive-attrs SEX RAC1P \\
      --fairness-privileged SEX=Male RAC1P=White-Alone

  # Fairness analysis with explicit dataset CSVs (overrides auto-discovery):
  python main.py --steps 5 \\
      --fairness-datasets data/acs_income_northeast_2024.csv \\
                          data/acs_income_south_2024.csv
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
    4: _HERE / 'microscopic_experiment_association_rules_values.py',
    5: _HERE / 'fairness_analysis.py',
}

# Human-readable step names used in the banner and summary output.
_STEP_NAMES = {
    1: 'Create Dataset',
    2: 'Feature Importance (CategoricalBoCSoR)',
    3: 'Association Rules — label level (FP-Growth)',
    4: 'Association Rules — value level (FP-Growth)',
    5: 'Fairness Analysis — disparate impact, parity, intersectional',
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


def _check_source_files(selected_steps: list[int] | None = None) -> bool:
    """
    Verify that the source files for the selected steps exist on disk.

    Only the steps that will actually be executed are checked: there is no
    reason to abort because an unrelated script is missing.  If
    *selected_steps* is None, all entries in _SRC are checked (used only
    in tests / dry-run contexts where the step list is not yet known).

    Returns True if all relevant files are present, False otherwise.
    Prints an error message for each missing file so the user knows
    exactly which script is absent.
    """
    steps_to_check = (
        {s: _SRC[s] for s in selected_steps if s in _SRC}
        if selected_steps is not None
        else _SRC
    )
    ok = True
    for step, path in steps_to_check.items():
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


def _outputs_step4_exist(regions: list[str], k_values: list[int]) -> bool:
    """
    Return True if all expected Step 4 output summary CSVs exist for every
    (region, k) combination.

    The sentinel file is summary.csv inside any k_{k}/ subdirectory under
    results/<region>/association_rules_values/.  The experiment-label component
    of the path is not known here (it encodes grid parameters), so we use
    rglob to detect any matching file.
    """
    results_dir = _HERE.parent / 'results'
    for region in regions:
        for k in k_values:
            ar_base = results_dir / region / 'association_rules_values'
            found   = ar_base.exists() and any(ar_base.rglob(f'k_{k}/summary.csv'))
            if not found:
                return False
    return True


def _inputs_step4_exist(regions: list[str], k_values: list[int]) -> bool:
    """
    Verify that Step 4 has at least one valid input for each region.

    Step 4 requires:
      - transactions_values.csv from Step 2 (one per (region, k))
      - at least one rules.csv from Step 3 (produced under
        association_rules/{exp_label}/k_{k}/sup_*/conf_*/)

    It is normal for low-k values to produce no Step 3 rules (sparse
    transactions).  A WARNING is printed for each missing k but the step is
    only blocked if NO k value has any rules.csv for a given region.
    """
    results_dir = _HERE.parent / 'results'
    ok = True
    for region in regions:
        has_tv      = False   # any transactions_values.csv for this region
        has_rules   = False   # any rules.csv from Step 3 for this region

        for k in k_values:
            tv_path = (
                results_dir / region / 'important_features'
                / f'k_{k}' / 'transactions_values.csv'
            )
            if tv_path.exists():
                has_tv = True

            ar_base = results_dir / region / 'association_rules'
            if ar_base.exists() and any(ar_base.rglob(f'k_{k}/sup_*/conf_*/rules.csv')):
                has_rules = True

        if not has_tv:
            print(
                f'  [ERROR] {region}: no transactions_values.csv found for any k. '
                f'Run Step 2 first.'
            )
            ok = False
        elif not has_rules:
            print(
                f'  [ERROR] {region}: no rules.csv from Step 3 found for any k. '
                f'Run Step 3 first (Step 4 needs label-level rules to filter '
                f'value-level items).'
            )
            ok = False
        else:
            print(f'  [INFO] {region}: transactions_values.csv and Step 3 rules found.')
    return ok


def _outputs_step3_exist(regions: list[str], k_values: list[int]) -> bool:
    """
    Return True if all expected Step 3 output rule CSVs exist for every
    (region, k) combination.

    NOTE: Step 3 currently has no skip logic — ARM is fast enough that
    re-running is not a burden.  This function is provided for symmetry
    with the other output-check helpers and to make adding a future
    --skip-arm flag straightforward without restructuring run_step3().

    The sentinel file checked is summary.csv inside k_{k}/ under any
    experiment-label subdirectory of results/<region>/association_rules/.
    The experiment label is not known at this point (it encodes the grid
    parameters), so we use rglob to find any matching file.
    """
    results_dir = _HERE.parent / 'results'
    for region in regions:
        for k in k_values:
            ar_base = results_dir / region / 'association_rules'
            found   = ar_base.exists() and any(ar_base.rglob(f'k_{k}/summary.csv'))
            if not found:
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
            metadata_cols  = args.metadata_cols if args.metadata_cols is not None else [],
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

def run_step4(args: argparse.Namespace) -> bool:
    """
    Run microscopic_experiment_association_rules_values.py — value-level ARM.

    Prerequisites:
      - Step 2 output (transactions_values.csv) for all (region, k) pairs.
      - Step 3 output (rules.csv) for at least one (region, k) pair — needed
        to derive the active label set that filters value-level items.

    No skip condition: like Step 3, value-level ARM is fast enough that
    re-running is not a burden, and the user may want to re-run with
    different grid parameters.

    Parameters
    ----------
    args : parsed CLI namespace containing all Step 4 parameters.
           Step 4 reuses the same grid parameters as Step 3 (same argparse
           group) so the CLI surface stays minimal.

    Returns True on success, False on failure.
    """
    _banner(f'STEP 4 — {_STEP_NAMES[4]}')

    # Abort early if required inputs are missing.
    if not _inputs_step4_exist(args.regions, args.k_values):
        print('  [ERROR] Required inputs are missing — run Steps 2 and 3 first.')
        return False

    mod = _load_module(_SRC[4], 'microscopic_experiment_association_rules_values')
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
            base_dir                 = _HERE.parent,
        )
    except Exception:
        print('\n  [ERROR] Step 4 failed:')
        traceback.print_exc()
        return False

    print(f'\n  > Step 4 completed in {time.perf_counter() - t0:.1f}s')
    return True


# ---------------------------------------------------------------------------
# Step 5 helpers and runner
# ---------------------------------------------------------------------------

def _outputs_step5_exist(regions: list[str]) -> bool:
    """
    Return True if a fairness_report.txt already exists for every region.

    The sentinel file is fairness_report.txt inside
    results/<region>/fairness_analysis/.  If it is present the step is
    considered complete (unless --force is set).
    """
    results_dir = _HERE.parent / 'results'
    for region in regions:
        p = results_dir / region / 'fairness_analysis' / 'fairness_report.txt'
        if not p.exists():
            return False
    return True


def _inputs_step5_exist(regions: list[str]) -> bool:
    """
    Verify that Step 5 has at least one value-level rules.csv for each region.

    Step 5 requires the output of Step 4 (association_rules_values/ subtree).
    A WARNING is printed for missing regions; the step is blocked only when NO
    rules.csv is found at all for a given region.
    """
    results_dir = _HERE.parent / 'results'
    ok = True
    for region in regions:
        ar_base = results_dir / region / 'association_rules_values'
        if ar_base.exists() and any(ar_base.rglob('rules.csv')):
            print(f'  [INFO] {region}: value-level rules.csv found for Step 5.')
        else:
            print(
                f'  [ERROR] {region}: no value-level rules.csv found under '
                f'{ar_base}. Run Step 4 first.'
            )
            ok = False
    return ok


def _resolve_fairness_datasets(args: argparse.Namespace) -> list[Path]:
    """
    Return the list of dataset CSV paths to pass to the fairness module.

    Resolution order:
    1. ``--fairness-datasets`` explicit paths (absolute or relative to CWD).
       Each path must exist; a WARNING is printed for any that do not.
    2. Auto-discovery: every CSV matching ``acs_income_{region}_{year}.csv``
       in ``args.output_dir`` for the requested regions and survey year.
    3. Fallback: all CSVs in ``args.output_dir`` whose name starts with
       ``acs_income_``, regardless of region or year.

    Returns an empty list when nothing is found so the fairness module can
    still run on rules alone (population-level metrics will simply be skipped).
    """
    # ── Explicit paths ────────────────────────────────────────────────────
    if getattr(args, 'fairness_datasets', None):
        paths: list[Path] = []
        for raw in args.fairness_datasets:
            p = Path(raw)
            if not p.is_absolute():
                p = Path.cwd() / p
            if p.exists():
                paths.append(p)
            else:
                print(f'  [WARNING] --fairness-datasets: file not found: {p}')
        return paths

    # ── Auto-discovery: region + year specific ────────────────────────────
    data_dir    = Path(args.output_dir)
    survey_year = args.survey_year
    paths = [
        data_dir / f'acs_income_{region}_{survey_year}.csv'
        for region in args.regions
    ]
    found = [p for p in paths if p.exists()]
    if found:
        return found

    # ── Fallback: any acs_income_*.csv in data_dir ────────────────────────
    fallback = sorted(data_dir.glob('acs_income_*.csv'))
    if fallback:
        print(
            f'  [INFO] No region-specific dataset CSVs found for year '
            f'{survey_year}; using all acs_income_*.csv files in {data_dir}:'
        )
        for p in fallback:
            print(f'    {p}')
    return fallback


def run_step5(args: argparse.Namespace) -> bool:
    """
    Run fairness_analysis.py — demographic fairness metrics.

    Prerequisites:
      - Step 4 output (value-level rules.csv files) must exist for at least
        one (region, k) pair.
      - Original dataset CSVs are optional: if present (auto-discovered from
        args.output_dir or supplied via --fairness-datasets), population-level
        DIR and positive rates are also computed.

    Skip condition: fairness_report.txt already exists for all regions and
    --force is not set.

    Parameters
    ----------
    args : parsed CLI namespace.

    Returns True on success, False on failure.
    """
    _banner(f'STEP 5 — {_STEP_NAMES[5]}')

    if not _inputs_step5_exist(args.regions):
        print('  [ERROR] Required inputs are missing — run Step 4 first.')
        return False

    if not args.force and _outputs_step5_exist(args.regions):
        print(
            '  > Fairness report already present for all regions. '
            'Use --force to overwrite.'
        )
        return True

    # ── Resolve sensitive-attribute configuration ─────────────────────────
    sensitive_attrs: list[str] = args.fairness_sensitive_attrs

    # Parse ATTR=VALUE pairs from --fairness-privileged into a dict.
    privileged_values: dict[str, str] = {}
    for token in getattr(args, 'fairness_privileged', []):
        if '=' in token:
            attr, val = token.split('=', 1)
            privileged_values[attr.strip()] = val.strip()
        else:
            print(
                f'  [WARNING] --fairness-privileged: ignored token '
                f'"{token}" (expected format ATTR=VALUE).'
            )

    # Fall back to module defaults when the user supplied nothing.
    if not privileged_values:
        privileged_values = None   # let the module use its own defaults

    # ── Resolve dataset paths ─────────────────────────────────────────────
    dataset_paths = _resolve_fairness_datasets(args)
    if dataset_paths:
        print('  [INFO] Dataset CSVs for population-level fairness:')
        for p in dataset_paths:
            print(f'    {p}')
    else:
        print(
            '  [INFO] No dataset CSVs found — population-level metrics will '
            'be skipped (rule-level metrics still computed).'
        )

    mod = _load_module(_SRC[5], 'fairness_analysis')
    t0  = time.perf_counter()
    try:
        mod.main(
            regions           = args.regions,
            k_values          = args.k_values,
            base_dir          = _HERE.parent,
            sensitive_attrs   = sensitive_attrs if sensitive_attrs else None,
            privileged_values = privileged_values,
            outcome_label     = args.fairness_outcome_label,
            positive_outcome  = args.fairness_positive_outcome,
            # Grid params — accepted for signature parity, not used by Step 5.
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
        )
    except Exception:
        print('\n  [ERROR] Step 5 failed:')
        traceback.print_exc()
        return False

    print(f'\n  > Step 5 completed in {time.perf_counter() - t0:.1f}s')
    return True

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
            'create_dataset → feature_importance → association_rules → fairness_analysis'
        ),
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )

    # ── Global options ─────────────────────────────────────────────────
    parser.add_argument(
        '--steps', nargs='+', type=int, choices=[1, 2, 3, 4, 5],
        default=[1, 2, 3, 4, 5], metavar='{1,2,3,4,5}',
        help='Steps to run (default: all five).',
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

    # ── Steps 3 & 4: association rules parameters ──────────────────────
    g3  = parser.add_argument_group(
        'Steps 3 & 4: association rules',
        'Parameters apply to both the label-level (Step 3) and value-level '
        '(Step 4) ARM passes.',
    )

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

    # ── Step 5: fairness analysis parameters ───────────────────────────
    g5 = parser.add_argument_group(
        'Step 5: fairness analysis',
        'Parameters for the demographic fairness analysis step.',
    )
    g5.add_argument(
        '--fairness-sensitive-attrs', nargs='+',
        default=['SEX', 'RAC1P'], metavar='ATTR',
        help=(
            'Sensitive attribute names to analyse (default: SEX RAC1P). '
            'Must match LABEL names used in the LABEL=value items of the rules.'
        ),
    )
    g5.add_argument(
        '--fairness-privileged', nargs='+', default=[], metavar='ATTR=VALUE',
        help=(
            'Privileged group for each sensitive attribute, as ATTR=VALUE pairs '
            '(default: SEX=Male RAC1P=White-Alone). '
            'The disparate impact ratio is computed relative to these groups. '
            'Example: --fairness-privileged SEX=Male RAC1P=White-Alone'
        ),
    )
    g5.add_argument(
        '--fairness-outcome-label', default='INCOME_ABOVE_THRESHOLD',
        help=(
            'Feature name of the binary outcome in LABEL=value items '
            '(default: INCOME_ABOVE_THRESHOLD).  Also used as the column name '
            'when reading the original dataset CSVs for population-level metrics.'
        ),
    )
    g5.add_argument(
        '--fairness-positive-outcome', default='1',
        help=(
            'String value that represents the positive outcome '
            '(default: "1").  Must match the encoding in both the rules and '
            'the original dataset CSV.'
        ),
    )
    g5.add_argument(
        '--fairness-datasets', nargs='+', default=[], metavar='PATH',
        help=(
            'Explicit paths to one or more original dataset CSVs used for '
            'population-level fairness metrics (DIR, SPD, positive rates). '
            'Paths may be absolute or relative to the current working directory. '
            'When omitted, dataset CSVs are auto-discovered from --output-dir '
            'by matching acs_income_{region}_{survey-year}.csv; if none are '
            'found there, all acs_income_*.csv files in that directory are used '
            'as a fallback.  Pass an empty list to suppress auto-discovery and '
            'run rule-level metrics only.'
        ),
    )

    return parser.parse_args()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Parse arguments, print the configuration banner, and run the selected steps.

    Each step runner (run_step1 … run_step5) returns True on success and False
    on failure.  The pipeline halts at the first failed step to prevent
    downstream steps from running on incomplete inputs.

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
    if not args.auto_calibrate:
        print(f'  Support grid                 : [{args.sup_min}, {args.sup_max}] step={args.sup_delta}')
        print(f'  Confidence grid              : [{args.conf_min}, {args.conf_max}] step={args.conf_delta}')
        print(f'  Lift grid                    : [{args.lift_min}, {args.lift_max}] step={args.lift_delta} window=±{args.lift_neutral_half_window}')
    if 5 in args.steps:
        print(f'  Fairness sensitive attrs     : {args.fairness_sensitive_attrs}')
        priv_str = ' '.join(args.fairness_privileged) if args.fairness_privileged else '(module defaults)'
        print(f'  Fairness privileged groups   : {priv_str}')
        print(f'  Fairness outcome label       : {args.fairness_outcome_label}={args.fairness_positive_outcome}')
        ds_str = ' '.join(args.fairness_datasets) if args.fairness_datasets else '(auto-discovered)'
        print(f'  Fairness dataset CSVs        : {ds_str}')
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
    if not _check_source_files(selected_steps=args.steps):
        print('\n  [ERROR] One or more source files are missing. Aborting.')
        sys.exit(1)

    # ── Execute selected steps in order ───────────────────────────────
    _RUNNERS = {1: run_step1, 2: run_step2, 3: run_step3, 4: run_step4, 5: run_step5}
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