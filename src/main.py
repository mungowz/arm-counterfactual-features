"""
src/main.py
───────────
Command-line entry point for the ACS Income pipeline.

This file orchestrates both pipeline stages:

  Stage 1 — dataset creation  (src/create_dataset.py)
    Downloads ACS PUMS data via folktables, decodes categorical features,
    discretises continuous variables, and writes train/test CSV files.
    When --states is a group name (e.g. northeast) the group name is used
    in the output filename instead of the individual state codes.

  Stage 2 — feature importance  (src/feature_importance.py)
    Trains a CatBoost classifier on the stage-1 output and computes global
    feature importance using the BoCSoR algorithm (Alfeo et al., 2023).
    Results are saved under <output-dir>/<states_tag>/<years_tag>/.

Usage
─────
From the project root:
    python -m src.main [OPTIONS]

Examples
────────
    # Both stages — dataset created if missing, BoCSoR skipped if output exists
    python -m src.main --states northeast --years 2024

    # Custom BoCSoR settings
    python -m src.main --states CA NY TX --columns ALL \\
                        --k 11 --percentile 20

    # Multiple years
    python -m src.main --states ALL --years 2021 2022 2023 2024

    python -m src.main --help
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.constants import (       # noqa: E402
    USA_STATES,
    STATE_GROUPS,
    INCOME_FEATURES,
    DEFAULT_COLUMNS,
)
from src.create_dataset import create_dataset  # noqa: E402

VALID_HORIZONS   = ("1-Year", "5-Year")
VALID_SURVEYS    = ("person", "household")
_ALL_STATE_CODES: set[str] = set(USA_STATES) | {"AK", "DC", "PR"}

# Default worker count used for both multi-year stage-1 parallelism and
# BoCSoR stage-2 processing.  Formula: max(1, min(14, cpu_count - 2)).
# Reserves 2 logical CPUs for the OS and the main process; caps at 14
# to avoid competing CatBoost thread pools degrading throughput.
_DEFAULT_WORKERS = max(1, min(14, (os.cpu_count() or 4) - 2))

logger = logging.getLogger("src.main")


# ─────────────────────────────────────────────────────────────────────────────
# Argument resolution helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_states(raw: list[str], horizon: str) -> list[str] | None:
    """
    Convert the raw --states argument into a list of state codes or None.

    Accepted inputs
    ───────────────
    "ALL"         → None  (downstream uses the full USA_STATES list)
    group name    → expanded via STATE_GROUPS
    state codes   → validated against the full set of known state codes

    Parameters
    ----------
    raw     : Unprocessed token list from argparse.
    horizon : Survey horizon, used to validate Alaska compatibility.

    Returns
    -------
    Sorted list of two-letter state codes, or None for all states.

    Raises
    ------
    ValueError
        On unrecognised state codes, mixed group/code input, or Alaska
        with an incompatible horizon.
    """
    if raw == ["ALL"]:
        return None

    if len(raw) == 1 and raw[0].lower() in STATE_GROUPS:
        group_name = raw[0].lower()
        states     = list(STATE_GROUPS[group_name])
        logger.info("State group '%s' expanded to: %s", group_name, states)
    else:
        groups_found = [s for s in raw if s.lower() in STATE_GROUPS]
        if groups_found:
            raise ValueError(
                f"Cannot mix group names and individual state codes. "
                f"Group tokens found: {groups_found}."
            )
        normalized = [s.upper() for s in raw]
        invalid    = [s for s in normalized if s not in _ALL_STATE_CODES]
        if invalid:
            raise ValueError(
                f"Unrecognised state codes: {invalid}.\n"
                f"Available groups: {sorted(STATE_GROUPS)}.\n"
                f"Valid state codes: {sorted(_ALL_STATE_CODES)}."
            )
        states = normalized

    if "AK" in states and horizon == "1-Year":
        raise ValueError(
            "Alaska (AK) is not supported with horizon='1-Year': the Census "
            "Bureau does not publish 1-Year estimates for areas with fewer "
            "than 65,000 inhabitants.  Use --horizon 5-Year or remove AK."
        )
    return states


def resolve_columns(raw: list[str] | None) -> list[str] | None:
    """
    Convert the raw --columns argument into a column list or None.

    Mapping
    ───────
    None      → DEFAULT_COLUMNS  (COW, SCHL, WKHP)
    ["ALL"]   → None  (retain all feature columns)
    list      → validated against INCOME_FEATURES

    Raises
    ------
    ValueError
        If any supplied column name is not in INCOME_FEATURES.
    """
    if raw is None:
        return list(DEFAULT_COLUMNS)
    if raw == ["ALL"]:
        return None

    valid   = set(INCOME_FEATURES)
    invalid = [c for c in raw if c not in valid]
    if invalid:
        raise ValueError(
            f"Invalid column names: {invalid}. "
            f"Available columns: {sorted(valid)}."
        )
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Multi-year process worker  (stage 1)
# ─────────────────────────────────────────────────────────────────────────────

def _process_year(
    year: int,
    horizon: str,
    survey: str,
    states: list[str] | None,
    threshold: float,
    random_seed: int,
    data_dir: Path,
    keep_columns: list[str] | None,
    log_level: str,
    states_label: str | None = None,
) -> tuple[int, int, int]:
    """
    Worker function executed in a separate process for a single survey year.

    Each worker configures its own logging handler because file descriptors
    are not reliably inherited across processes on all platforms.

    Returns
    -------
    (year, n_train_rows, n_test_rows)
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )
    _, train_df, test_df = create_dataset(
        survey_year=year,
        horizon=horizon,
        survey=survey,
        states=states,
        threshold=threshold,
        random_seed=random_seed,
        data_dir=data_dir,
        keep_columns=keep_columns,
        states_label=states_label,
    )
    return year, len(train_df), len(test_df)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_feature_importance(
    train_path: Path,
    test_path: Path,
    output_dir: Path,
    k: list[int],
    percentile: float,
    cb_iterations: int,
    cb_lr: float,
    cb_depth: int,
    cb_verbose: bool,
    cb_early_stopping: int,
    n_workers: int,
    random_seed: int,
    log_level: str,
) -> None:
    """
    Invoke stage 2 (BoCSoR feature importance) programmatically.

    Called after stage 1 completes for each survey year.
    Both boundary directions (class 0→1 and class 1→0) are always computed
    and saved to separate files (_class0 / _class1 suffixes).

    Parameters
    ----------
    train_path           : Path to the train CSV produced by stage 1.
    test_path            : Path to the test CSV produced by stage 1.
    output_dir           : Directory where BoCSoR results will be written.
    k                    : Raw list from --k (e.g. [11] or [1, 5, 11]).
                           Passed to expand_k(): single value K is auto-expanded
                           to all odd integers 1..K; multiple values used as-is.
    percentile           : Percentile threshold for boundary instance selection.
    cb_iterations        : CatBoost boosting rounds.
    cb_lr                : CatBoost learning rate.
    cb_depth             : CatBoost tree depth.
    cb_verbose           : Whether CatBoost prints training progress.
    cb_early_stopping    : Stop training if validation loss does not improve
                           for this many rounds (0 = disabled).
    random_seed          : Random seed for CatBoost.
    log_level            : Logging level string (e.g. "INFO").
    """
    # Lazy import to keep stage-1-only runs free of catboost/sklearn overhead.
    from src.feature_importance import (
        load_split_data,
        build_rank_maps,
        train_catboost,
        run_bocsor_multi_k,
        expand_k,
        _compute_default_workers,
    )

    k_values = expand_k(k)

    logger.info("═" * 62)
    logger.info("  ACS INCOME PIPELINE  —  stage 2: feature importance (BoCSoR)")
    logger.info("═" * 62)
    logger.info("  k values             : %s  (from --k %s)", k_values, k)
    logger.info("  Percentile threshold : %.1f%%", percentile)
    logger.info("  Output directory     : %s", output_dir.resolve())
    logger.info("═" * 62)

    X_train, X_test, y_train, y_test, target_col = load_split_data(
        train_path=train_path,
        test_path=test_path,
        dataset_path=None,
        random_seed=random_seed,
    )
    feature_cols = list(X_train.columns)
    logger.info("Train: %d rows  |  Test: %d rows  |  Features: %d",
                len(X_train), len(X_test), len(feature_cols))
    logger.info("Target: %s  |  Features: %s", target_col, feature_cols)

    logger.info("Building rank maps from training data …")
    rank_maps = build_rank_maps(X_train)

    model = train_catboost(
        X_train=X_train,
        y_train=y_train,
        cat_features=feature_cols,
        random_seed=random_seed,
        iterations=cb_iterations,
        learning_rate=cb_lr,
        depth=cb_depth,
        verbose=cb_verbose,
        early_stopping_rounds=cb_early_stopping if cb_early_stopping > 0 else None,
    )
    y_pred_test  = model.predict(X_test).astype(int).ravel()
    y_pred_train = model.predict(X_train).astype(int).ravel()
    test_acc  = (y_pred_test  == y_test.values).mean()
    train_acc = (y_pred_train == y_train.values).mean()
    logger.info("CatBoost accuracy — train: %.4f  |  test: %.4f", train_acc, test_acc)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Always explain both boundary directions (0→1 and 1→0).
    # Each class gets its own _class0 / _class1 suffixed output files.
    orig_classes_requested = [0, 1]
    suffix_map = {c: f"_class{c}" for c in orig_classes_requested}
    orig_classes_todo = [
        c for c in orig_classes_requested
        if not (output_dir / f"feature_importance{suffix_map[c]}.csv").exists()
    ]
    if not orig_classes_todo:
        logger.info(
            "Skipping stage 2: all output files already exist in %s.",
            output_dir.resolve(),
        )
        return
    skipped = set(orig_classes_requested) - set(orig_classes_todo)
    if skipped:
        logger.info("Skipping class(es) %s: output already exists.", sorted(skipped))

    orig_classes = orig_classes_todo
    for orig_cls in orig_classes:
        cf_cls = 1 - orig_cls
        # suffix_map is built on orig_classes_requested (full set) so it
        # remains correct even when only a subset of classes is processed.
        suffix = suffix_map[orig_cls]

        logger.info(
            "== BoCSoR: class %d boundary (counterfactual class %d) ==",
            orig_cls, cf_cls,
        )
        all_itemsets_df, importance_df, per_k_itemsets, distances_df = run_bocsor_multi_k(
            model=model,
            X_train=X_train,
            y_train=y_train,
            y_pred_train=y_pred_train,
            rank_maps=rank_maps,
            feature_cols=feature_cols,
            k_values=k_values,
            percentile_th=percentile,
            original_class=orig_cls,
            cf_class=cf_cls,
            n_workers=n_workers,
        )

        itemsets_path = output_dir / f"feature_importance_itemsets{suffix}.csv"
        all_itemsets_df.to_csv(itemsets_path, index=False)
        logger.info("Itemsets (all k) -> %s  (%d rows)", itemsets_path, len(all_itemsets_df))

        for k_val, k_df in per_k_itemsets.items():
            k_path = output_dir / f"feature_importance_itemsets_k{k_val}{suffix}.csv"
            k_df.to_csv(k_path, index=False)
            logger.info("  k=%d -> %s  (%d rows)", k_val, k_path.name, len(k_df))

        imp_path = output_dir / f"feature_importance{suffix}.csv"
        importance_df.reset_index().to_csv(imp_path, index=False)
        logger.info("Importance -> %s", imp_path)

        # Distances: one row per (boundary_instance, k_neighbour) pair.
        # Columns: k_value, instance_index, cf_index, k_neighbour_rank, distance.
        dist_path = output_dir / f"bocsor_distances{suffix}.csv"
        distances_df.to_csv(dist_path, index=False)
        logger.info("Distances -> %s  (%d rows)", dist_path, len(distances_df))

        logger.info("Feature importance summary (class %d):", orig_cls)
        for k_col in importance_df.columns:
            top = importance_df[k_col].sort_values(ascending=False)
            logger.info("  %s:", k_col)
            for feat, score in top.items():
                logger.info("    %-40s %.4f", feat, score)

    logger.info("═" * 62)
    logger.info("  Stage 2 completed successfully.")
    logger.info("  Outputs in: %s", output_dir.resolve())
    logger.info("═" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    groups_str = ", ".join(sorted(STATE_GROUPS))
    cols_str   = ", ".join(INCOME_FEATURES)

    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="ACS Income pipeline — stage 1 (dataset creation) and/or stage 2 (BoCSoR feature importance)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Predefined state groups:
  {groups_str}

Available feature columns (--columns):
  {cols_str}
  or: ALL  (retain every feature column)

The pipeline always runs both stages.  If the stage-1 CSV files already
exist they are loaded directly (no download).  If the stage-2 output files
already exist that class is skipped.  Re-runs are therefore always safe.

Examples:
  # Full pipeline — dataset + BoCSoR
  python -m src.main --states northeast --years 2024

  # Custom BoCSoR settings
  python -m src.main --states CA NY TX --columns ALL \\
                      --k 11 --percentile 20

  # Multiple years
  python -m src.main --states ALL --years 2021 2022 2023 2024
        """,
    )

    # ── ACS parameters (stage 1) ──────────────────────────────────────────────
    acs = parser.add_argument_group("ACS parameters  (stage 1)")
    acs.add_argument(
        "--years", nargs="+", type=int, default=[2024], metavar="YEAR",
        help="Survey year(s) in the range 2014–2024.  Default: 2024.",
    )
    acs.add_argument(
        "--horizon", choices=VALID_HORIZONS, default="1-Year",
        help=(
            "ACS survey horizon.  Note: '1-Year' excludes Alaska (AK).  "
            "Default: 1-Year."
        ),
    )
    acs.add_argument(
        "--survey", choices=VALID_SURVEYS, default="person",
        help="Survey unit of analysis.  Default: person.",
    )
    acs.add_argument(
        "--states", nargs="+", default=["ALL"], metavar="STATE_OR_GROUP",
        help=(
            "Individual state codes (e.g. CA NY TX), a single predefined "
            "group name (e.g. northeast), or ALL.  Default: ALL."
        ),
    )

    # ── Task parameters (stage 1) ─────────────────────────────────────────────
    task = parser.add_argument_group("Task parameters  (stage 1)")
    task.add_argument(
        "--threshold", type=float, default=100_000.0, metavar="DOLLARS",
        help=(
            "Annual personal income threshold in U.S. dollars (PINCP field).  "
            "The binary target is 1 if PINCP > threshold.  Default: 100000."
        ),
    )

    # ── Output column selection (stage 1) ─────────────────────────────────────
    cols = parser.add_argument_group("Output column selection  (stage 1)")
    cols.add_argument(
        "--columns", nargs="+", default=None, metavar="COL",
        help=(
            f"Feature columns to retain in the output CSV.  "
            f"Default: {DEFAULT_COLUMNS}.  "
            f"Pass ALL to keep every feature column."
        ),
    )

    # ── Train / test split seed (stage 1) ────────────────────────────────────
    split = parser.add_argument_group("Train / test split  (stage 1)")
    split.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the stratified split and CatBoost.  Default: 42.",
    )

    # ── Input / output ────────────────────────────────────────────────────────
    io = parser.add_argument_group("Input / output")
    io.add_argument(
        "--data-dir", type=Path, default=Path("data"), metavar="DIR",
        help=(
            "Root output directory for stage 1.  Raw PUMS files → <dir>/raw/.  "
            "Processed CSVs → <dir>/.  Default: data/."
        ),
    )
    io.add_argument(
        "--output-dir", type=Path, default=Path("results"), metavar="DIR",
        help=(
            "Output directory for stage 2 results (feature importance CSVs "
            "and itemset CSVs).  Created if absent.  Default: results/."
        ),
    )

    # ── BoCSoR hyperparameters (stage 2) ──────────────────────────────────────
    boc = parser.add_argument_group("BoCSoR hyperparameters  (stage 2)")
    boc.add_argument(
        "--k", nargs="+", type=int, default=[11], metavar="K",
        help=(
            "Neighbourhood size(s) for the counterfactual search.  "
            "A single value K is auto-expanded to all odd integers 1..K "
            "(e.g. --k 11 -> 1 3 5 7 9 11).  "
            "Multiple values are used as-is (e.g. --k 1 5 11).  "
            "Values must be >= 1.  Default: 11."
        ),
    )
    boc.add_argument(
        "--percentile", type=float, default=20.0, metavar="PCT",
        help=(
            "Percentile threshold for boundary instance selection (0–100).  "
            "Instances whose distance to the nearest opposite-class instance "
            "is below this percentile are treated as boundary instances.  "
            "Default: 20."
        ),
    )

    # ── CatBoost hyperparameters (stage 2) ────────────────────────────────────
    cb = parser.add_argument_group("CatBoost hyperparameters  (stage 2)")
    cb.add_argument("--cb-iterations", type=int,   default=500,  metavar="N",
                    help="Boosting rounds.  Default: 500.")
    cb.add_argument("--cb-lr",         type=float, default=0.05, metavar="LR",
                    help="Learning rate.  Default: 0.05.")
    cb.add_argument("--cb-depth",      type=int,   default=6,    metavar="D",
                    help="Tree depth.  Default: 6.")
    cb.add_argument("--cb-early-stopping", type=int, default=0, metavar="N",
                    help=(
                        "Stop training if the loss does not improve for N consecutive "
                        "rounds (early stopping).  0 disables early stopping.  "
                        "When enabled, 20%% of the training set is used as an internal "
                        "validation split.  Default: 0 (disabled)."
                    ))
    cb.add_argument("--cb-verbose",    action="store_true",
                    help="Print CatBoost training progress.")

    # ── Performance (stage 1) ─────────────────────────────────────────────────
    perf = parser.add_argument_group("Performance  (stage 1)")
    perf.add_argument(
        "--workers", type=int, default=_DEFAULT_WORKERS, metavar="N",
        help=(
            f"Number of parallel worker processes when processing multiple "
            f"years.  Each year runs as an independent process, bypassing "
            f"the GIL.  Ignored when a single year is specified.  "
            f"Default: {_DEFAULT_WORKERS}."
        ),
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity level.  Default: INFO.",
    )

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: infer stage-1 output paths
# ─────────────────────────────────────────────────────────────────────────────

def _infer_split_paths(
    data_dir: Path,
    year: int,
    states: list[str] | None,
    threshold: float,
    horizon: str,
    survey: str,
    keep_columns: list[str] | None,
    states_label: str | None = None,
) -> tuple[Path, Path]:
    """
    Reconstruct the train/test file paths that create_dataset writes.

    Mirrors the naming convention in create_dataset.py (step 8):

        {prefix}_{year}_{states_tag}_{horizon_tag}_{survey}_thr{threshold}_cols{cols_tag}.csv

    When states_label is provided it is used in place of the sorted
    state codes (e.g. 'northeast' instead of 'CT_MA_ME_NH_NJ_NY_PA_RI_VT').

    Examples
    --------
        train_2024_NY_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv
        train_2024_northeast_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv
        test_2024_ALL_1Y_person_thr75000_colsALL.csv

    Parameters
    ----------
    data_dir     : Root output directory passed to create_dataset.
    year         : Survey year.
    states       : Expanded state codes, or None for all states.
    threshold    : Income threshold.
    horizon      : Survey horizon ("1-Year" or "5-Year").
    survey       : Survey unit ("person" or "household").
    keep_columns : Feature columns retained (None means ALL).
    states_label : Group name to use in the filename instead of state codes.
    """
    if states_label is not None:
        states_tag = states_label
    elif states is None:
        states_tag = "ALL"
    else:
        states_tag = "_".join(sorted(states))
    horizon_tag = horizon.replace("-", "").replace("Year", "Y")   # "1-Year" -> "1Y"
    cols_tag    = "-".join(sorted(keep_columns)) if keep_columns else "ALL"
    stem = (
        f"{year}_{states_tag}"
        f"_{horizon_tag}_{survey}"
        f"_thr{int(threshold)}"
        f"_cols{cols_tag}"
    )
    return data_dir / f"train_{stem}.csv", data_dir / f"test_{stem}.csv"


def _build_output_dir(
    base_dir: Path,
    states: list[str] | None,
    raw_states_arg: list[str],
    years: list[int],
    keep_columns: list[str] | None,
    percentile: float,
) -> Path:
    """
    Build the stage-2 output directory path, embedding the state scope,
    year range, and feature columns so that results from different
    configurations never overwrite each other.

    Directory structure
    -------------------
        <base_dir>/<states_tag>/<years_tag>/cols<cols_tag>/

    <states_tag> rules
    ------------------
    - Recognised group name (e.g. "northeast") → used as-is.
    - ALL states → "ALL".
    - Individual codes → sorted and joined by "_" (e.g. "CA_NY_TX").

    <years_tag> rules
    -----------------
    - Single year       → the year itself (e.g. "2024").
    - Contiguous range  → "<first>-<last>" (e.g. "2021-2024").
    - Non-contiguous    → years joined by "_" (e.g. "2021_2023").

    <cols_tag> rules
    ----------------
    - keep_columns list → columns sorted and joined by "-" (e.g. "COW-SCHL-WKHP").
    - None (all columns) → "ALL".

    <pct_tag> rules
    ---------------
    - Percentile value as integer (e.g. 20 → "pct20").

    Examples
    --------
        results/northeast/2024/colsCOW-SCHL-WKHP/pct20/
        results/ALL/2021-2024/colsCOW-OCCP-SCHL-WKHP/pct10/
        results/CA_NY_TX/2024/colsALL/pct20/
    """
    # ── States tag ────────────────────────────────────────────────────────────
    if raw_states_arg == ["ALL"] or states is None:
        states_tag = "ALL"
    elif (len(raw_states_arg) == 1
          and raw_states_arg[0].lower() in STATE_GROUPS):
        # Preserve the group name exactly as typed by the user.
        states_tag = raw_states_arg[0].lower()
    else:
        states_tag = "_".join(sorted(states))

    # ── Years tag ─────────────────────────────────────────────────────────────
    sorted_years = sorted(years)
    if len(sorted_years) == 1:
        years_tag = str(sorted_years[0])
    elif sorted_years == list(range(sorted_years[0], sorted_years[-1] + 1)):
        # Contiguous range.
        years_tag = f"{sorted_years[0]}-{sorted_years[-1]}"
    else:
        years_tag = "_".join(str(y) for y in sorted_years)

    # ── Columns tag ───────────────────────────────────────────────────────────
    cols_tag = "-".join(sorted(keep_columns)) if keep_columns else "ALL"

    # ── Percentile tag ────────────────────────────────────────────────────────
    pct_tag = f"pct{int(percentile)}"

    return base_dir / states_tag / years_tag / f"cols{cols_tag}" / pct_tag


def _resolve_states_label(raw_states_arg: list[str]) -> str | None:
    """
    Return a short label for the states scope used in filenames.

    If the raw --states argument is a recognised group/region/division name
    that label is returned as-is (e.g. "northeast").  If "ALL" is passed,
    None is returned so create_dataset uses "ALL".  For individual state
    codes None is returned so the codes are used directly.
    """
    if raw_states_arg == ["ALL"]:
        return None
    if (len(raw_states_arg) == 1
            and raw_states_arg[0].lower() in STATE_GROUPS):
        return raw_states_arg[0].lower()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    multiprocessing.set_start_method("fork", force=True)
    parser = build_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Argument validation ───────────────────────────────────────────────────
    try:
        states       = resolve_states(args.states, args.horizon)
        keep_columns = resolve_columns(args.columns)
    except ValueError as exc:
        logger.error("%s", exc)
        parser.print_usage()
        sys.exit(1)

    states_label = _resolve_states_label(args.states)

    invalid_years = [y for y in args.years if not (2014 <= y <= 2024)]
    if invalid_years:
        logger.error("Year(s) out of supported range (2014–2024): %s", invalid_years)
        sys.exit(1)

    invalid_k = [v for v in args.k if v < 1]
    if invalid_k:
        logger.error("--k values must be >= 1.  Invalid: %s", invalid_k)
        sys.exit(1)

    if not (0.0 < args.percentile <= 100.0):
        logger.error("--percentile must be in the range (0, 100].")
        sys.exit(1)

    if args.threshold <= 0:
        logger.error("--threshold must be a positive value.")
        sys.exit(1)

    # ── Configuration summary ─────────────────────────────────────────────────
    n_states = len(states) if states else len(USA_STATES)
    xai_output_dir = _build_output_dir(
        args.output_dir, states, args.states, args.years,
        keep_columns, args.percentile,
    )
    logger.info("═" * 62)
    logger.info("  ACS INCOME PIPELINE")
    logger.info("═" * 62)
    logger.info("  Year(s)        : %s", args.years)
    logger.info("  Horizon        : %s", args.horizon)
    logger.info("  Survey         : %s", args.survey)
    logger.info("  States         : %s (%d)", states or "ALL", n_states)
    logger.info("  Threshold      : $%.0f", args.threshold)
    logger.info("  Output columns : %s", keep_columns or "ALL")
    logger.info("  Workers        : %d (auto-detected: %d)", args.workers, _DEFAULT_WORKERS)
    logger.info("  Data dir       : %s", args.data_dir.resolve())
    logger.info("  BoCSoR k       : %s", args.k)
    logger.info("  BoCSoR pct     : %.1f%%", args.percentile)
    logger.info("  XAI output dir : %s", xai_output_dir.resolve())
    logger.info("═" * 62)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1: dataset creation (skipped automatically if output exists)
    # ─────────────────────────────────────────────────────────────────────────
    if len(args.years) == 1:
        year = args.years[0]
        logger.info("── Year %d ──────────────────────────────────────────", year)
        try:
            dataset_df, train_df, test_df = create_dataset(
                survey_year=year,
                horizon=args.horizon,
                survey=args.survey,
                states=states,
                threshold=args.threshold,
                random_seed=args.seed,
                data_dir=args.data_dir,
                keep_columns=keep_columns,
                states_label=states_label,
            )
            logger.info(
                "Year %d complete → train=%d rows, test=%d rows.",
                year, len(train_df), len(test_df),
            )
        except Exception as exc:
            logger.error("Error processing year %d: %s", year, exc)
            raise

        # ── Stage 2 (single year) ─────────────────────────────────────────────
        train_path, test_path = _infer_split_paths(
            args.data_dir, year, states, args.threshold,
            args.horizon, args.survey, keep_columns,
            states_label=states_label,
        )
        _run_feature_importance(
            train_path=train_path,
            test_path=test_path,
            output_dir=xai_output_dir,
            k=args.k,
            percentile=args.percentile,
            cb_iterations=args.cb_iterations,
            cb_lr=args.cb_lr,
            cb_depth=args.cb_depth,
            cb_verbose=args.cb_verbose,
            cb_early_stopping=args.cb_early_stopping,
            n_workers=args.workers,
            random_seed=args.seed,
            log_level=args.log_level,
        )

    else:
        # Multiple years: one process per year (CPU-bound + independent I/O).
        workers = min(args.workers, len(args.years))
        logger.info(
            "Starting parallel execution: %d years across %d worker(s).",
            len(args.years), workers,
        )

        common_kwargs = dict(
            horizon=args.horizon,
            survey=args.survey,
            states=states,
            threshold=args.threshold,
            random_seed=args.seed,
            data_dir=args.data_dir,
            keep_columns=keep_columns,
            log_level=args.log_level,
            states_label=states_label,
        )

        completed: dict[int, tuple[int, int]] = {}
        failed:    list[int] = []

        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_year = {
                executor.submit(_process_year, year=y, **common_kwargs): y
                for y in args.years
            }
            for future in as_completed(future_to_year):
                y = future_to_year[future]
                try:
                    _, n_train, n_test = future.result()
                    completed[y] = (n_train, n_test)
                    logger.info(
                        "Year %d complete → train=%d rows, test=%d rows.",
                        y, n_train, n_test,
                    )
                except Exception as exc:
                    logger.error("Year %d failed: %s", y, exc)
                    failed.append(y)

        if failed:
            logger.error("The following years encountered errors: %s", sorted(failed))
            sys.exit(1)

        # ── Stage 2 (multi-year: one run per year, sequential) ────────────────
        for year in sorted(completed):
            logger.info("── Stage 2: year %d ─────────────────────────────────", year)
            train_path, test_path = _infer_split_paths(
                args.data_dir, year, states, args.threshold,
                args.horizon, args.survey, keep_columns,
                states_label=states_label,
            )
            year_output_dir = xai_output_dir / str(year)
            _run_feature_importance(
                train_path=train_path,
                test_path=test_path,
                output_dir=year_output_dir,
                k=args.k,
                percentile=args.percentile,
                cb_iterations=args.cb_iterations,
                cb_lr=args.cb_lr,
                cb_depth=args.cb_depth,
                cb_verbose=args.cb_verbose,
                cb_early_stopping=args.cb_early_stopping,
                n_workers=args.workers,
                random_seed=args.seed,
                log_level=args.log_level,
            )

    logger.info("═" * 62)
    logger.info("  Pipeline completed successfully.")
    logger.info("═" * 62)


if __name__ == "__main__":
    main()