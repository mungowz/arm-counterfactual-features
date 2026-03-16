"""
src/create_dataset.py
─────────────────────
Pipeline stage 1 — ACS Income dataset creation.

Processing steps
────────────────
1. Download raw ACS PUMS data into data/raw/ (explicit root_dir; no
   files are written to the user's home directory).
2. Normalize the raw DataFrame: rename RELSHIPP → RELP when present.
   The Census Bureau renamed the household-relationship variable in
   2019, but the folktables BasicProblem always references "RELP".
3. Build the ACSIncome task with a configurable income threshold and
   extract the feature matrix, label vector, and sensitive-attribute
   group via df_to_pandas().
4. Categorize all feature columns:
     - AGEP  : binned into six career-stage bands
     - WKHP  : binned into six work-hours bands
     - All remaining categorical columns decoded to human-readable
       string labels using the ACS PUMS code maps in constants.py.
       Result columns use pd.Categorical to reduce memory footprint.
5. Optionally filter columns: keep only `keep_columns` + target.
6. Split into train and test sets with stratification on the target.
7. Write output CSV files to data/.

Parallelization strategy
────────────────────────
- Per-state download  : ThreadPoolExecutor (I/O-bound; one HTTP
                         request per state file, all independent).
- Column categorization: ThreadPoolExecutor (independent per-column
                         transforms; NumPy-backed pandas operations
                         release the GIL, enabling true concurrency).
- CSV write (split)   : two threads write train and test files
                         simultaneously.
- Multi-year execution: ProcessPoolExecutor in main.py (CPU-bound;
                         each year is a fully independent workload
                         that bypasses the GIL via separate processes).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import folktables
import numpy as np
import pandas as pd
from folktables import ACSDataSource
from sklearn.model_selection import train_test_split

from src.constants import (
    AGEP_BINS, AGEP_LABELS,
    WKHP_BINS, WKHP_LABELS,
    COW_MAP, SCHL_MAP, MAR_MAP,
    POBP_MAP, RELP_MAP, SEX_MAP, RAC1P_MAP,
    COLUMN_FALLBACKS,
    INCOME_FEATURES,
    DEFAULT_COLUMNS,
    occp_to_major_group,
    OCCP_FALLBACK,
    OCCP_MAJOR_GROUP_RANGES,
)

logger = logging.getLogger(__name__)

# Maximum worker counts for concurrent operations.
# Both are capped at the number of logical CPUs to avoid over-subscribing
# the OS scheduler.
_IO_WORKERS  = min(16, os.cpu_count() or 4)
_CAT_WORKERS = min(8,  os.cpu_count() or 4)


# ─────────────────────────────────────────────────────────────
# 1. Raw DataFrame normalization
# ─────────────────────────────────────────────────────────────

def normalize_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename RELSHIPP → RELP on the raw ACS DataFrame when necessary.

    Background
    ──────────
    The Census Bureau renamed the household-relationship column between
    survey years: raw PUMS files use "RELP" for years ≤ 2018 and
    "RELSHIPP" for years ≥ 2019.  The official folktables source (acs.py)
    always references "RELP" in the BasicProblem feature list but does
    not perform the rename automatically.  Without this normalization
    step, df_to_pandas() raises a KeyError for any 2019–2024 dataset.

    Parameters
    ----------
    df : Raw DataFrame returned by ACSDataSource.get_data().

    Returns
    -------
    DataFrame with the household-relationship column consistently
    named "RELP", regardless of the original survey year.
    """
    if "RELSHIPP" in df.columns and "RELP" not in df.columns:
        df = df.rename(columns={"RELSHIPP": "RELP"})
        logger.debug("Renamed RELSHIPP → RELP (ACS data year ≥ 2019).")
    return df


# ─────────────────────────────────────────────────────────────
# 2. Parallel per-state download
# ─────────────────────────────────────────────────────────────

def _download_one_state(
    state: str,
    survey_year: int,
    horizon: str,
    survey: str,
    raw_dir: Path,
) -> pd.DataFrame:
    """Download and return the raw PUMS DataFrame for a single state."""
    source = ACSDataSource(
        survey_year=survey_year,
        horizon=horizon,
        survey=survey,
        root_dir=str(raw_dir),
    )
    return source.get_data(states=[state], download=True)


def download_data(
    survey_year: int,
    horizon: str,
    survey: str,
    states: list[str] | None,
    raw_dir: Path,
) -> pd.DataFrame:
    """
    Download ACS PUMS data for the requested states.

    Single-state and all-states requests use the direct ACSDataSource
    path to avoid thread-pool overhead.  Multi-state requests are
    parallelized with a ThreadPoolExecutor: each state corresponds to
    a separate HTTP download, making this operation I/O-bound and
    therefore well-suited to thread-level concurrency.

    Parameters
    ----------
    survey_year : ACS survey year (2014–2024).
    horizon     : Survey horizon ("1-Year" or "5-Year").
    survey      : Survey unit ("person" or "household").
    states      : List of two-letter state codes, or None for all states.
    raw_dir     : Directory where raw PUMS files are cached.

    Returns
    -------
    Concatenated raw DataFrame for all requested states.

    Raises
    ------
    RuntimeError
        If one or more per-state downloads fail.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not states or len(states) == 1:
        source = ACSDataSource(
            survey_year=survey_year,
            horizon=horizon,
            survey=survey,
            root_dir=str(raw_dir),
        )
        df = source.get_data(states=states, download=True)
        logger.info("Download complete: %d rows × %d columns.", *df.shape)
        return df

    n_workers = min(_IO_WORKERS, len(states))
    logger.info(
        "Parallel download: %d states, %d workers.", len(states), n_workers,
    )

    frames: dict[str, pd.DataFrame] = {}
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_state = {
            executor.submit(
                _download_one_state,
                state, survey_year, horizon, survey, raw_dir,
            ): state
            for state in states
        }
        for future in as_completed(future_to_state):
            state = future_to_state[future]
            try:
                frames[state] = future.result()
                logger.debug("  ✓ %s (%d rows)", state, len(frames[state]))
            except Exception as exc:
                logger.error("  ✗ %s: %s", state, exc)
                failures.append(state)

    if failures:
        raise RuntimeError(
            f"Download failed for the following states: {sorted(failures)}"
        )

    # Concatenate in the original order for reproducibility.
    df = pd.concat(
        [frames[s] for s in states if s in frames], ignore_index=True,
    )
    logger.info("Download complete: %d rows × %d columns.", *df.shape)
    return df


# ─────────────────────────────────────────────────────────────
# 3. ACSIncome task construction
# ─────────────────────────────────────────────────────────────

def build_acs_income_task(threshold: float) -> folktables.BasicProblem:
    """
    Return an ACSIncome-equivalent BasicProblem with a configurable
    income threshold.

    The task predicts whether an individual's annual personal income
    (PINCP) exceeds `threshold` dollars.  All other parameters mirror
    the official ACSIncome definition in folktables/acs.py.

    Parameters
    ----------
    threshold : Income threshold in U.S. dollars.

    Returns
    -------
    folktables.BasicProblem
    """
    return folktables.BasicProblem(
        features=INCOME_FEATURES,
        target="PINCP",
        target_transform=lambda x: x > threshold,
        group="RAC1P",
        preprocess=folktables.adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )


# ─────────────────────────────────────────────────────────────
# 4. Parallel column categorization
# ─────────────────────────────────────────────────────────────

def _bin_column(series: pd.Series, bins: list, labels: list[str]) -> pd.Categorical:
    """Bin a continuous Series into ordered categorical bands."""
    return pd.cut(series, bins=bins, labels=labels, right=True)


def _map_column(
    series: pd.Series,
    mapping: dict[str, str],
    fallback: str,
) -> pd.Categorical:
    """
    Decode a numeric-code Series to human-readable string labels.

    Codes absent from `mapping` are replaced by `fallback`.
    The result is stored as pd.Categorical to reduce memory usage and
    accelerate downstream groupby and value_counts operations.
    """
    decoded = series.astype(np.int32).astype(str).map(mapping).fillna(fallback)
    categories = sorted(set(mapping.values()) | {fallback})
    return pd.Categorical(decoded, categories=categories)


def _categorize_occp(series: pd.Series) -> pd.Categorical:
    """
    Map ACS PUMS OCCP codes to BLS OEWS major-group labels.

    Uses range-based lookup via occp_to_major_group() rather than a
    dictionary map, because OCCP codes are contiguous numeric ranges
    rather than a sparse set of individual keys.

    Parameters
    ----------
    series : Integer or float Series of raw OCCP codes.

    Returns
    -------
    pd.Categorical with the 23 BLS major-group labels as categories.
    """
    labels = series.astype(int).map(occp_to_major_group)
    all_categories = sorted({label for *_, label in OCCP_MAJOR_GROUP_RANGES} | {OCCP_FALLBACK})
    return pd.Categorical(labels, categories=all_categories)


def _build_column_transforms() -> list[tuple[str, Callable[[pd.Series], pd.Series]]]:
    """Return the ordered list of (column_name, transform_function) pairs."""
    return [
        ("AGEP",  lambda s: _bin_column(s, AGEP_BINS, AGEP_LABELS)),
        ("WKHP",  lambda s: _bin_column(s, WKHP_BINS, WKHP_LABELS)),
        ("COW",   lambda s: _map_column(s, COW_MAP,   "Unknown")),
        ("SCHL",  lambda s: _map_column(s, SCHL_MAP,  "Unknown")),
        ("MAR",   lambda s: _map_column(s, MAR_MAP,   "Unknown")),
        ("OCCP",  _categorize_occp),                                  # range-based BLS major groups
        ("POBP",  lambda s: _map_column(s, POBP_MAP,  COLUMN_FALLBACKS["POBP"])),
        ("RELP",  lambda s: _map_column(s, RELP_MAP,  COLUMN_FALLBACKS["RELP"])),
        ("SEX",   lambda s: _map_column(s, SEX_MAP,   "Unknown")),
        ("RAC1P", lambda s: _map_column(s, RAC1P_MAP, "Unknown")),
    ]


def categorize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply binning and label decoding to all feature columns in parallel.

    Column-level transforms are fully independent of one another and are
    therefore dispatched to a ThreadPoolExecutor.  The underlying NumPy
    array operations (astype, pd.cut, Series.map) release the GIL,
    enabling genuine concurrency across threads.

    Parameters
    ----------
    df : Feature DataFrame produced by task.df_to_pandas().

    Returns
    -------
    DataFrame with identical column names but categorical/string values.
    """
    transforms = [
        (col, fn) for col, fn in _build_column_transforms()
        if col in df.columns
    ]

    results: dict[str, pd.Series] = {}

    def _apply(
        col: str, fn: Callable, series: pd.Series,
    ) -> tuple[str, pd.Series]:
        return col, fn(series)

    with ThreadPoolExecutor(max_workers=min(_CAT_WORKERS, len(transforms))) as executor:
        futures = {
            executor.submit(_apply, col, fn, df[col].copy()): col
            for col, fn in transforms
        }
        for future in as_completed(futures):
            col, result = future.result()
            results[col] = result

    # Reassemble in the original column order.
    out = df.copy()
    for col in df.columns:
        if col in results:
            out[col] = results[col]
    return out


# ─────────────────────────────────────────────────────────────
# 5. Column selection
# ─────────────────────────────────────────────────────────────

def select_columns(
    df: pd.DataFrame,
    keep_columns: list[str] | None,
    target_col: str,
) -> pd.DataFrame:
    """
    Retain only `keep_columns` plus the target column.

    Parameters
    ----------
    df           : Fully categorized feature DataFrame with target appended.
    keep_columns : Feature columns to retain.  Pass None to keep all columns.
    target_col   : Name of the binary target column (always preserved).

    Returns
    -------
    Filtered DataFrame.

    Raises
    ------
    ValueError
        If any column in `keep_columns` is not present in `df`.
    """
    if keep_columns is None:
        return df

    missing = [c for c in keep_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Requested columns not found in the dataset: {missing}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )
    cols = list(keep_columns)
    if target_col not in cols:
        cols.append(target_col)
    return df[cols]


# ─────────────────────────────────────────────────────────────
# 6. Main pipeline function
# ─────────────────────────────────────────────────────────────

def create_dataset(
    survey_year: int,
    horizon: str,
    survey: str,
    states: list[str] | None,
    threshold: float,
    test_size: float,
    random_seed: int,
    data_dir: Path,
    keep_columns: list[str] | None = DEFAULT_COLUMNS,
    states_label: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the full dataset-creation pipeline for a single survey year.

    Parameters
    ----------
    survey_year  : ACS survey year (2014–2024).
    horizon      : Survey horizon — "1-Year" or "5-Year".
    survey       : Survey unit — "person" or "household".
    states       : Two-letter state codes to include, or None for all
                   states in USA_STATES.
    threshold    : Annual personal income threshold in U.S. dollars.
                   The binary target label is 1 if PINCP > threshold.
    test_size    : Fraction of the dataset reserved for the test split.
                   0.0 produces a single output file with no split.
    random_seed  : Random seed for the stratified train/test split.
    data_dir     : Root output directory.  Raw PUMS files are cached in
                   <data_dir>/raw/; processed CSVs are written to
                   <data_dir>/.
    keep_columns : Feature columns retained in the output CSV.
                   Defaults to DEFAULT_COLUMNS = ["COW", "SCHL", "WKHP"].
                   Pass None to retain all feature columns.
    states_label : Optional human-readable label for the states scope used
                   in the output filename instead of the full list of codes.
                   Pass the group/region/division name (e.g. "northeast") so
                   that the filename reads train_2024_northeast_… rather than
                   train_2024_CT_MA_ME_….  If None the codes are used.

    Returns
    -------
    (train_df, test_df) : Processed DataFrames.
        When test_size == 0.0, test_df is an empty DataFrame with the
        correct column schema.
    """
    data_dir = Path(data_dir)
    raw_dir  = data_dir / "raw"

    # ── Pre-flight: skip if output files already exist ────────
    # Reconstruct the expected output filename using the same logic
    # as step 8 below.  If the file(s) already exist, load and
    # return them directly, skipping download + encoding entirely.
    if states_label is not None:
        _states_tag = states_label
    elif states is None:
        _states_tag = "ALL"
    else:
        _states_tag = "_".join(sorted(states))
    _horizon_tag = horizon.replace("-", "").replace("Year", "Y")
    _cols_tag    = "-".join(keep_columns) if keep_columns else "ALL"
    _stem = (
        f"{survey_year}_{_states_tag}"
        f"_{_horizon_tag}_{survey}"
        f"_thr{int(threshold)}"
        f"_cols{_cols_tag}"
    )
    if test_size > 0.0:
        _train_path = data_dir / f"train_{_stem}.csv"
        _test_path  = data_dir / f"test_{_stem}.csv"
        if _train_path.exists() and _test_path.exists():
            logger.info(
                "Skipping stage 1: output files already exist.\n"
                "  Train → %s\n  Test  → %s",
                _train_path, _test_path,
            )
            train_df = pd.read_csv(_train_path, dtype=str)
            test_df  = pd.read_csv(_test_path,  dtype=str)
            # Re-cast the target column to int8 to match normal pipeline output.
            _col_target = f"income_over_{int(threshold)}"
            if _col_target in train_df.columns:
                train_df[_col_target] = train_df[_col_target].astype(np.int8)
                test_df[_col_target]  = test_df[_col_target].astype(np.int8)
            return train_df, test_df
    else:
        _dataset_path = data_dir / f"dataset_{_stem}.csv"
        if _dataset_path.exists():
            logger.info(
                "Skipping stage 1: output file already exists.\n"
                "  Dataset → %s",
                _dataset_path,
            )
            train_df = pd.read_csv(_dataset_path, dtype=str)
            _col_target = f"income_over_{int(threshold)}"
            if _col_target in train_df.columns:
                train_df[_col_target] = train_df[_col_target].astype(np.int8)
            return train_df, pd.DataFrame()

    # ── Step 1: Download raw PUMS data ────────────────────────
    acs_data = download_data(survey_year, horizon, survey, states, raw_dir)

    # ── Step 2: Normalize column names ───────────────────────
    acs_data = normalize_raw_columns(acs_data)

    # ── Step 3: Extract features and labels ──────────────────
    logger.info("Building ACSIncome task | threshold=$%.0f", threshold)
    task = build_acs_income_task(threshold)
    features, labels, _ = task.df_to_pandas(acs_data)
    logger.info("Feature matrix: %d rows × %d columns.", *features.shape)

    # ── Step 4: Categorize feature columns ───────────────────
    logger.info("Categorizing feature columns…")
    features_cat = categorize(features)

    # ── Step 5: Append binary target column ──────────────────
    col_target = f"income_over_{int(threshold)}"
    features_cat[col_target] = labels.astype(np.int8).values
    logger.info(
        "Dataset: %d samples | positive class: %.2f%%.",
        len(features_cat),
        features_cat[col_target].mean() * 100,
    )

    # ── Step 6: Filter output columns ────────────────────────
    dataset = select_columns(features_cat, keep_columns, col_target)
    retained = [c for c in dataset.columns if c != col_target]
    logger.info("Retained columns: %s + [%s]", retained, col_target)

    # ── Step 7: Stratified train / test split ─────────────────
    X = dataset.drop(columns=[col_target])
    y = dataset[col_target]

    if test_size > 0.0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_seed,
            stratify=y,
        )
    else:
        X_train, y_train = X, y
        X_test,  y_test  = X.iloc[:0], y.iloc[:0]

    logger.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    # ── Step 8: Write output CSV files ───────────────────────
    # File name encodes every parameter that affects the dataset content,
    # so runs with different configurations never overwrite each other.
    #
    # Pattern:
    #   {prefix}_{year}_{states_tag}_{horizon_tag}_{survey}_thr{threshold}_cols{cols_tag}.csv
    #
    # {states_tag} is states_label when supplied (e.g. 'northeast'),
    # 'ALL' when states is None, or sorted state codes joined by '_'.
    #
    # Examples:
    #   train_2024_NY_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv
    #   train_2024_northeast_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv
    #   dataset_2024_ALL_1Y_person_thr75000_colsALL.csv

    # Use the caller-supplied label (e.g. a group/region/division name) when
    # available so the filename reads train_2024_northeast_… rather than
    # train_2024_CT_MA_ME_NH_NJ_NY_PA_RI_VT_…
    if states_label is not None:
        states_tag = states_label
    elif states is None:
        states_tag = "ALL"
    else:
        states_tag = "_".join(sorted(states))
    horizon_tag = horizon.replace("-", "").replace("Year", "Y")   # "1-Year" → "1Y"
    cols_tag    = (
        "-".join(keep_columns) if keep_columns else "ALL"
    )
    stem = (
        f"{survey_year}_{states_tag}"
        f"_{horizon_tag}_{survey}"
        f"_thr{int(threshold)}"
        f"_cols{cols_tag}"
    )

    train_df             = X_train.copy()
    train_df[col_target] = y_train.values
    test_df              = X_test.copy()
    test_df[col_target]  = y_test.values

    def _safe_path(directory: Path, name: str) -> Path:
        """
        Return a path that does not already exist.
        If <name>.csv is taken, append _2, _3, … until a free slot is found.
        """
        candidate = directory / f"{name}.csv"
        if not candidate.exists():
            return candidate
        counter = 2
        while True:
            candidate = directory / f"{name}_{counter}.csv"
            if not candidate.exists():
                return candidate
            counter += 1

    if test_size > 0.0:
        train_path = _safe_path(data_dir, f"train_{stem}")
        test_path  = _safe_path(data_dir, f"test_{stem}")

        # Write both files concurrently.
        with ThreadPoolExecutor(max_workers=2) as executor:
            f_train = executor.submit(train_df.to_csv, train_path, index=False)
            f_test  = executor.submit(test_df.to_csv,  test_path,  index=False)
            f_train.result()
            f_test.result()

        logger.info("✓ Train   → %s", train_path)
        logger.info("✓ Test    → %s", test_path)
    else:
        dataset_path = _safe_path(data_dir, f"dataset_{stem}")
        train_df.to_csv(dataset_path, index=False)
        logger.info("✓ Dataset → %s", dataset_path)

    return train_df, test_df