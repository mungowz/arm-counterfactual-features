"""
src/main.py
───────────
Command-line entry point for the ACS Income pipeline — stage 1: dataset creation.

Usage
─────
From the project root:
    python -m src.main [OPTIONS]

Or directly:
    python src/main.py [OPTIONS]

Examples
────────
    python -m src.main
    python -m src.main --states northeast --years 2024
    python -m src.main --states south --threshold 75000
    python -m src.main --states CA NY TX --columns COW SCHL MAR SEX RAC1P
    python -m src.main --states ALL --columns ALL
    python -m src.main --years 2021 2022 2023 2024 --states midwest
    python -m src.main --help
"""

from __future__ import annotations

import argparse
import logging
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

# Default number of worker processes for multi-year parallel execution.
# Capped to avoid saturating the system with CPU-bound subprocesses.
_DEFAULT_YEAR_WORKERS = min(4, os.cpu_count() or 2)

logger = logging.getLogger("src.main")


# ─────────────────────────────────────────────────────────────
# Argument resolution helpers
# ─────────────────────────────────────────────────────────────

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
        On unrecognized state codes, mixed group/code input, or Alaska
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
                f"Unrecognized state codes: {invalid}.\n"
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


# ─────────────────────────────────────────────────────────────
# Multi-year process worker
# ─────────────────────────────────────────────────────────────

def _process_year(
    year: int,
    horizon: str,
    survey: str,
    states: list[str] | None,
    threshold: float,
    test_size: float,
    random_seed: int,
    data_dir: Path,
    keep_columns: list[str] | None,
    log_level: str,
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
    train_df, test_df = create_dataset(
        survey_year=year,
        horizon=horizon,
        survey=survey,
        states=states,
        threshold=threshold,
        test_size=test_size,
        random_seed=random_seed,
        data_dir=data_dir,
        keep_columns=keep_columns,
    )
    return year, len(train_df), len(test_df)


# ─────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    groups_str = ", ".join(sorted(STATE_GROUPS))
    cols_str   = ", ".join(INCOME_FEATURES)

    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="ACS Income pipeline — stage 1: dataset creation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Predefined state groups:
  {groups_str}

Available feature columns (--columns):
  {cols_str}
  or: ALL  (retain every feature column)

Examples:
  python -m src.main --states northeast --years 2024
  python -m src.main --states CA NY TX --threshold 75000 --columns COW SCHL MAR
  python -m src.main --states ALL --columns ALL --years 2023 2024
        """,
    )

    acs = parser.add_argument_group("ACS parameters")
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

    task = parser.add_argument_group("Task parameters")
    task.add_argument(
        "--threshold", type=float, default=100_000.0, metavar="DOLLARS",
        help=(
            "Annual personal income threshold in U.S. dollars (PINCP field).  "
            "The binary target is 1 if PINCP > threshold.  Default: 100000."
        ),
    )

    cols = parser.add_argument_group("Output column selection")
    cols.add_argument(
        "--columns", nargs="+", default=None, metavar="COL",
        help=(
            f"Feature columns to retain in the output CSV.  "
            f"Default: {DEFAULT_COLUMNS}.  "
            f"Pass ALL to keep every feature column."
        ),
    )

    split = parser.add_argument_group("Train / test split")
    split.add_argument(
        "--test-size", type=float, default=0.0, metavar="FRACTION",
        help=(
            "Fraction of the dataset reserved for the test split (0.0–1.0).  "
            "0.0 produces a single dataset_*.csv with no split.  Default: 0.0."
        ),
    )
    split.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the stratified split.  Default: 42.",
    )

    io = parser.add_argument_group("Input / output")
    io.add_argument(
        "--data-dir", type=Path, default=Path("data"), metavar="DIR",
        help=(
            "Root output directory.  Raw PUMS files → <dir>/raw/.  "
            "Processed CSVs → <dir>/.  Default: data/."
        ),
    )

    perf = parser.add_argument_group("Performance")
    perf.add_argument(
        "--workers", type=int, default=_DEFAULT_YEAR_WORKERS, metavar="N",
        help=(
            f"Number of parallel worker processes when processing multiple "
            f"years.  Each year runs as an independent process, bypassing "
            f"the GIL.  Ignored when a single year is specified.  "
            f"Default: {_DEFAULT_YEAR_WORKERS}."
        ),
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity level.  Default: INFO.",
    )

    return parser


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Argument validation ───────────────────────────────────
    try:
        states       = resolve_states(args.states, args.horizon)
        keep_columns = resolve_columns(args.columns)
    except ValueError as exc:
        logger.error("%s", exc)
        parser.print_usage()
        sys.exit(1)

    invalid_years = [y for y in args.years if not (2014 <= y <= 2024)]
    if invalid_years:
        logger.error("Year(s) out of supported range (2014–2024): %s", invalid_years)
        sys.exit(1)

    if not (0.0 <= args.test_size < 1.0):
        logger.error("--test-size must be in the range [0.0, 1.0).")
        sys.exit(1)

    if args.threshold <= 0:
        logger.error("--threshold must be a positive value.")
        sys.exit(1)

    # ── Configuration summary ─────────────────────────────────
    n_states = len(states) if states else len(USA_STATES)
    logger.info("═" * 62)
    logger.info("  ACS INCOME PIPELINE  —  stage 1: create_dataset")
    logger.info("═" * 62)
    logger.info("  Year(s)        : %s", args.years)
    logger.info("  Horizon        : %s", args.horizon)
    logger.info("  Survey         : %s", args.survey)
    logger.info("  States         : %s (%d)", states or "ALL", n_states)
    logger.info("  Threshold      : $%.0f", args.threshold)
    logger.info("  Output columns : %s", keep_columns or "ALL")
    logger.info("  Test split     : %.0f%%", args.test_size * 100)
    logger.info("  Random seed    : %d", args.seed)
    logger.info("  Workers        : %d", args.workers if len(args.years) > 1 else 1)
    logger.info("  Output dir     : %s", args.data_dir.resolve())
    logger.info("═" * 62)

    # ── Execution ─────────────────────────────────────────────
    if len(args.years) == 1:
        # Single year: run in the current process to avoid fork/spawn overhead.
        year = args.years[0]
        logger.info("── Year %d ──────────────────────────────────────────", year)
        try:
            train_df, test_df = create_dataset(
                survey_year=year,
                horizon=args.horizon,
                survey=args.survey,
                states=states,
                threshold=args.threshold,
                test_size=args.test_size,
                random_seed=args.seed,
                data_dir=args.data_dir,
                keep_columns=keep_columns,
            )
            logger.info(
                "Year %d complete → train=%d rows, test=%d rows.",
                year, len(train_df), len(test_df),
            )
        except Exception as exc:
            logger.error("Error processing year %d: %s", year, exc)
            raise

    else:
        # Multiple years: one process per year (CPU-bound + independent I/O).
        # ProcessPoolExecutor bypasses the GIL and exploits separate CPU cores.
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
            test_size=args.test_size,
            random_seed=args.seed,
            data_dir=args.data_dir,
            keep_columns=keep_columns,
            log_level=args.log_level,
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

    logger.info("═" * 62)
    logger.info("  Pipeline completed successfully.")
    logger.info("═" * 62)


if __name__ == "__main__":
    main()
