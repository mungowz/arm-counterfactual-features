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
                logger.debug("  OK %s (%d rows)", state, len(frames[state]))
            except Exception as exc:
                logger.error("  FAIL %s: %s", state, exc)
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

def _build_acs_raw_task() -> folktables.BasicProblem:
    """
    Return an ACSIncome-equivalent task that keeps PINCP as a raw
    continuous value (no binarisation).

    Used internally to extract raw income before applying the dead zone
    filter and then binarising manually.
    """
    return folktables.BasicProblem(
        features=INCOME_FEATURES,
        target="PINCP",
        target_transform=None,     # raw continuous PINCP
        group="RAC1P",
        preprocess=folktables.adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )


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
    return pd.cut(series, bins=bins, labels=labels, right=True, include_lowest=True)


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

    Uses an integer-keyed mapping internally: this avoids the expensive
    .astype(str) step which creates ~1M Python string objects on large
    datasets.  Integer hashing is also faster than string hashing inside
    pandas .map().
    """
    int_mapping = {int(k): v for k, v in mapping.items()}
    decoded = series.astype(np.int32).map(int_mapping).fillna(fallback)
    categories = sorted(set(mapping.values()) | {fallback})
    return pd.Categorical(decoded, categories=categories)


def _categorize_occp(series: pd.Series) -> pd.Categorical:
    """
    Map ACS PUMS OCCP codes to BLS OEWS major-group labels.

    Uses vectorised np.searchsorted on the pre-computed lower-bound array
    instead of a per-row Python function call — ~10× faster on 1M+ rows.

    Parameters
    ----------
    series : Integer or float Series of raw OCCP codes.

    Returns
    -------
    pd.Categorical with the 23 BLS major-group labels as categories.
    """
    codes = series.to_numpy(dtype=np.int64)

    # Vectorised range lookup: for each code, find the candidate range
    # index via searchsorted, then verify the code falls within the
    # upper bound of that range.
    lower_bounds = np.array([lo for lo, _, _ in OCCP_MAJOR_GROUP_RANGES], dtype=np.int64)
    upper_bounds = np.array([hi for _, hi, _ in OCCP_MAJOR_GROUP_RANGES], dtype=np.int64)
    range_labels = [lbl for _, _, lbl in OCCP_MAJOR_GROUP_RANGES]

    # searchsorted with side='right' gives the insertion point; subtract 1
    # to get the index of the range whose lower bound is <= code.
    idx = np.searchsorted(lower_bounds, codes, side="right") - 1

    # Vectorised label assignment via fancy indexing (no Python loop).
    # Build a label lookup with OCCP_FALLBACK at position -1 and len(ranges).
    label_lookup = np.array(range_labels + [OCCP_FALLBACK], dtype=object)
    # Codes outside any range → fallback (last position in label_lookup).
    fallback_idx = len(range_labels)

    valid = (codes > 0) & (idx >= 0) & (idx < len(upper_bounds))
    valid[valid] &= codes[valid] <= upper_bounds[idx[valid]]

    # Map: valid indices → label_lookup[idx], invalid → fallback.
    mapped_idx = np.where(valid, idx, fallback_idx)
    result = label_lookup[mapped_idx]

    all_categories = sorted(set(range_labels) | {OCCP_FALLBACK})
    return pd.Categorical(result, categories=all_categories)


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
# 6. Shared filename stem builder
# ─────────────────────────────────────────────────────────────

def build_dataset_stem(
    survey_year: int,
    states: list[str] | None,
    horizon: str,
    survey: str,
    threshold: float,
    keep_columns: list[str] | None,
    states_label: str | None = None,
    margin: float = 0.0,
) -> str:
    """
    Build the filename stem shared by dataset, train and test CSV files.

    Pattern:
        {year}_{states}_{horizon}_{survey}_thr{threshold}[_dz{margin}]_cols{cols}

    The dead zone tag (dz{margin}) is included only when margin > 0.

    Examples:
        2024_ALL_1Y_person_thr94200_dz600_colsALL
        2024_NY_1Y_person_thr94400_colsALL           (no dead zone)
        2024_northeast_1Y_person_thr100700_dz1200_colsCOW-SCHL-WKHP

    This function is the single source of truth for filename construction.
    It is also imported by main.py (_infer_split_paths) to ensure consistency.
    """
    if states_label is not None:
        states_tag = states_label
    elif states is None:
        states_tag = "ALL"
    else:
        states_tag = "_".join(sorted(states))
    horizon_tag = horizon.replace("-", "").replace("Year", "Y")  # "1-Year" → "1Y"
    cols_tag = "-".join(sorted(keep_columns)) if keep_columns else "ALL"
    dz_tag = f"_dz{int(margin)}" if margin > 0 else ""
    return (
        f"{survey_year}_{states_tag}"
        f"_{horizon_tag}_{survey}"
        f"_thr{int(threshold)}"
        f"{dz_tag}"
        f"_cols{cols_tag}"
    )


# ─────────────────────────────────────────────────────────────
# 7. Main pipeline function
# ─────────────────────────────────────────────────────────────

def create_dataset(
    survey_year: int,
    horizon: str,
    survey: str,
    states: list[str] | None,
    threshold: float,
    random_seed: int,
    data_dir: Path,
    keep_columns: list[str] | None = None,
    states_label: str | None = None,
    margin: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    random_seed  : Random seed for the stratified 80/20 train/test split.
    data_dir     : Root output directory.  Raw PUMS files are cached in
                   <data_dir>/raw/; processed CSVs are written to
                   <data_dir>/.
    keep_columns : Feature columns retained in the output CSV.
                   None (default) retains all feature columns.
                   Pass a list to keep only specific columns.
    states_label : Optional group/region name used in the output filename
                   instead of individual state codes (e.g. "northeast").
                   If None the sorted state codes are used.
    margin       : Dead zone half-width in dollars.  Individuals with
                   PINCP in [threshold − margin, threshold + margin] are
                   excluded from the dataset.  0 = disabled.

    Returns
    -------
    (dataset_df, train_df, test_df)
        dataset_df : full unsplit dataset.
        train_df   : stratified training split (80 %).
        test_df    : stratified test split (20 %).
    """
    # Resolve keep_columns: None → all columns (no filtering).
    # When a list is provided, sort for consistent filename tags.
    if keep_columns is not None:
        keep_columns = sorted(keep_columns)

    data_dir = Path(data_dir)
    raw_dir  = data_dir / "raw"

    # ── Pre-flight: skip if output files already exist ────────
    # Reconstruct the expected output filename using build_dataset_stem().
    # If all three files exist, load and return them directly, skipping
    # download + encoding entirely.
    _stem = build_dataset_stem(
        survey_year, states, horizon, survey, threshold,
        keep_columns, states_label, margin=margin,
    )
    _dataset_path = data_dir / f"dataset_{_stem}.csv"
    _train_path   = data_dir / f"train_{_stem}.csv"
    _test_path    = data_dir / f"test_{_stem}.csv"
    if _dataset_path.exists() and _train_path.exists() and _test_path.exists():
        logger.info(
            "Skipping stage 1: output files already exist.\n"
            "  Dataset → %s\n  Train → %s\n  Test  → %s",
            _dataset_path, _train_path, _test_path,
        )
        _col_target = f"income_over_{int(threshold)}"
        dataset_df = pd.read_csv(_dataset_path, dtype=str)
        train_df   = pd.read_csv(_train_path,   dtype=str)
        test_df    = pd.read_csv(_test_path,    dtype=str)
        for _df in (dataset_df, train_df, test_df):
            if _col_target in _df.columns:
                _df[_col_target] = _df[_col_target].astype(np.int8)
        return dataset_df, train_df, test_df

    # ── Step 1: Download raw PUMS data ────────────────────────
    acs_data = download_data(survey_year, horizon, survey, states, raw_dir)

    # ── Step 2: Normalize column names ───────────────────────
    acs_data = normalize_raw_columns(acs_data)

    # ── Step 3: Extract features and raw income ────────────────
    # Use the raw task (no binarisation) so we can apply the dead zone
    # filter before binarising the target.
    logger.info("Building ACSIncome task | threshold=$%.0f | margin=$%.0f", threshold, margin)
    task_raw = _build_acs_raw_task()
    features, raw_pincp, _ = task_raw.df_to_pandas(acs_data)
    logger.info("Feature matrix: %d rows × %d columns.", *features.shape)

    # Log the number of missing values replaced by the postprocess
    # (np.nan_to_num converts NaN → -1 in folktables).
    sentinel_counts = (features == -1).sum()
    if sentinel_counts.any():
        for col, cnt in sentinel_counts.items():
            if cnt > 0:
                logger.warning(
                    "  Column '%s': %d rows had NaN (replaced with -1 by postprocess).",
                    col, cnt,
                )

    # ── Step 3b: Dead zone filter ────────────────────────────
    # Exclude individuals whose income falls within ±margin of the
    # threshold.  These are ambiguous cases where the binarised label
    # depends on noise rather than structural differences in features.
    # The margin is derived from the ACS Margin of Error for median
    # family income, propagated through the Pew upper-income formula.
    if margin > 0:
        raw_vals = raw_pincp.to_numpy().ravel().astype(float)
        lower = threshold - margin
        upper = threshold + margin
        keep_mask = (raw_vals <= lower) | (raw_vals > upper)
        n_before  = len(features)
        features  = features.loc[keep_mask].reset_index(drop=True)
        raw_pincp = raw_pincp.loc[keep_mask].reset_index(drop=True)
        n_excluded = n_before - len(features)
        logger.info(
            "Dead zone [$ %.0f, $ %.0f]: excluded %d / %d samples (%.1f%%).",
            lower, upper, n_excluded, n_before, n_excluded / n_before * 100,
        )

    # ── Step 4: Categorize feature columns ───────────────────
    logger.info("Categorizing feature columns…")
    features_cat = categorize(features)

    # ── Step 5: Binarise target and append ───────────────────
    col_target = f"income_over_{int(threshold)}"
    labels = (raw_pincp.to_numpy().ravel().astype(float) > threshold).astype(np.int8)
    features_cat[col_target] = labels
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=random_seed,
        stratify=y,
    )

    logger.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    # ── Step 8: Write output CSV files ───────────────────────
    # File name encodes every parameter that affects the dataset content,
    # so runs with different configurations never overwrite each other.
    #
    # Pattern:
    #   {prefix}_{year}_{states}_{horizon}_{survey}_thr{threshold}_cols{cols}.csv
    #
    # Examples:
    #   dataset_2024_CA_NY_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv
    #   dataset_2024_ALL_1Y_person_thr75000_colsALL.csv
    #   train_2024_northeast_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv
    #   test_2024_northeast_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv

    stem = build_dataset_stem(
        survey_year, states, horizon, survey, threshold,
        keep_columns, states_label, margin=margin,
    )

    dataset_df           = dataset.copy()
    train_df             = X_train.copy()
    train_df[col_target] = y_train.values
    test_df              = X_test.copy()
    test_df[col_target]  = y_test.values

    dataset_path = data_dir / f"dataset_{stem}.csv"
    train_path   = data_dir / f"train_{stem}.csv"
    test_path    = data_dir / f"test_{stem}.csv"

    # Write all three files concurrently.
    with ThreadPoolExecutor(max_workers=3) as executor:
        f_ds    = executor.submit(dataset_df.to_csv, dataset_path, index=False)
        f_train = executor.submit(train_df.to_csv,   train_path,   index=False)
        f_test  = executor.submit(test_df.to_csv,    test_path,    index=False)
        f_ds.result()
        f_train.result()
        f_test.result()

    logger.info("Dataset -> %s", dataset_path)
    logger.info("Train   -> %s", train_path)
    logger.info("Test    -> %s", test_path)

    return dataset_df, train_df, test_df