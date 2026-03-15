# ACS Income Dataset Pipeline

A command-line pipeline for building binary-classification datasets from the
U.S. Census Bureau's American Community Survey (ACS) Public Use Microdata Sample
(PUMS), using the [folktables](https://github.com/socialfoundations/folktables)
library.

The pipeline predicts whether an individual's annual personal income (`PINCP`)
exceeds a configurable threshold.  All categorical features are decoded from
numeric ACS codes to human-readable string labels, and continuous features are
discretized into meaningful bands.

---

## Project structure

```
project/
├── src/
│   ├── __init__.py
│   ├── constants.py       # State groups, feature set, bin configs, code maps
│   ├── create_dataset.py  # Core pipeline logic (download → encode → split → save)
│   └── main.py            # CLI entry point
└── data/
    ├── raw/               # Cached raw PUMS files (auto-created)
    └── *.csv              # Processed output datasets
```

---

## Requirements

```
folktables
pandas
numpy
scikit-learn
```

Install with:

```bash
pip install folktables pandas numpy scikit-learn
```

---

## Quick start

```bash
# Default: all 49 states, year 2024, threshold $100k, columns COW + SCHL + WKHP
python -m src.main

# Northeast region, single year
python -m src.main --states northeast --years 2024

# Custom states, custom threshold, all feature columns
python -m src.main --states CA NY TX --threshold 75000 --columns ALL

# Four years in parallel, Midwest region
python -m src.main --years 2021 2022 2023 2024 --states midwest

# With a stratified 80/20 train-test split
python -m src.main --states northeast --test-size 0.2
```

---

## Output files

| `--test-size` | Files produced |
|---|---|
| `0.0` (default) | `data/dataset_<year>_<states>_thr<threshold>.csv` |
| `> 0.0` | `data/train_<year>_<states>_thr<threshold>.csv` + `data/test_<year>_<states>_thr<threshold>.csv` |

The train/test split is **stratified** on the binary target column, preserving
the positive-class proportion across both sets.

---

## CLI reference

### ACS parameters

| Option | Type | Default | Description |
|---|---|---|---|
| `--years` | int (one or more) | `2024` | ACS survey year(s). Supported range: **2014–2024**. Pass multiple values to process several years (e.g. `--years 2022 2023 2024`). |
| `--horizon` | `1-Year` \| `5-Year` | `1-Year` | ACS survey horizon. **1-Year** excludes Alaska (AK); use **5-Year** if Alaska coverage is required. |
| `--survey` | `person` \| `household` | `person` | Unit of analysis for the ACS survey. |
| `--states` | str (one or more) | `ALL` | States to include. Accepts individual two-letter codes (`CA NY TX`), a single predefined group name (`northeast`), or `ALL` for all 49 supported states. See [State groups](#state-groups) below. |

### Task parameters

| Option | Type | Default | Description |
|---|---|---|---|
| `--threshold` | float | `100000` | Annual personal income threshold in U.S. dollars (`PINCP` field). The binary target label is **1** if `PINCP > threshold`, **0** otherwise. |

### Output column selection

| Option | Type | Default | Description |
|---|---|---|---|
| `--columns` | str (one or more) | `COW SCHL WKHP` | Feature columns to retain in the output CSV. Pass `ALL` to keep every feature column. See [Available feature columns](#available-feature-columns) below. |

### Train / test split

| Option | Type | Default | Description |
|---|---|---|---|
| `--test-size` | float | `0.0` | Fraction of the dataset reserved for the test set (e.g. `0.2` for an 80/20 split). `0.0` produces a single output file. The split is stratified on the target column. |
| `--seed` | int | `42` | Random seed for reproducibility of the stratified split. |

### Input / output

| Option | Type | Default | Description |
|---|---|---|---|
| `--data-dir` | path | `data/` | Root output directory. Raw PUMS files are cached in `<dir>/raw/`; processed CSVs are written directly to `<dir>/`. |

### Performance

| Option | Type | Default | Description |
|---|---|---|---|
| `--workers` | int | `min(4, cpu_count)` | Number of parallel worker **processes** when multiple years are requested. Each year runs as an independent process, fully bypassing the GIL. Ignored when a single year is specified. |

### Logging

| Option | Values | Default | Description |
|---|---|---|---|
| `--log-level` | `DEBUG` `INFO` `WARNING` `ERROR` | `INFO` | Logging verbosity. Use `DEBUG` to inspect per-column categorization and per-state download progress. |

---

## State groups

Predefined groups follow the official U.S. Census Bureau geographic
classification.

### 4 Census regions

| Group | States |
|---|---|
| `northeast` | CT ME MA NH NJ NY PA RI VT |
| `midwest` | IL IN IA KS MI MN MO NE ND OH SD WI |
| `south` | AL AR DE FL GA KY LA MD MS NC OK SC TN TX VA WV |
| `west` | AZ CA CO HI ID MT NV NM OR UT WA WY |

### 9 Census divisions

| Group | States |
|---|---|
| `new_england` | CT ME MA NH RI VT |
| `middle_atlantic` | NJ NY PA |
| `east_north_central` | IL IN MI OH WI |
| `west_north_central` | IA KS MN MO NE ND SD |
| `south_atlantic` | DE FL GA MD NC SC VA WV |
| `east_south_central` | AL KY MS TN |
| `west_south_central` | AR LA OK TX |
| `mountain` | AZ CO ID MT NV NM UT WY |
| `pacific` | CA HI OR WA |

### Convenience aliases

| Group | States |
|---|---|
| `sunbelt` | AZ CA FL GA NM NV SC TX |
| `rust_belt` | IL IN MI MO NY OH PA WI |
| `great_plains` | IA KS MN MO NE ND SD |

> **Note on Alaska:** AK is excluded from all groups when using
> `--horizon 1-Year`.  The Census Bureau does not publish 1-Year estimates
> for geographic areas with fewer than 65,000 inhabitants, which causes
> folktables to download a malformed file and raises a `pandas.ParserError`.
> Use `--horizon 5-Year` if Alaska coverage is required.

---

## Available feature columns

| Column | ACS variable | Description |
|---|---|---|
| `AGEP` | Age | Binned into six career-stage bands: `Young`, `Young-Adult`, `Mid-Career`, `Experienced`, `Late-Career`, `Retirement-Age`. |
| `COW` | Class of worker | 9 categories (e.g. `Employee-Private-For-Profit`, `Self-Employed-Incorporated`). |
| `SCHL` | Educational attainment | 24 levels from `No-Schooling-Completed` to `Doctorate-Degree`. |
| `MAR` | Marital status | 5 categories (e.g. `Married`, `Divorced`, `Never-Married-Or-Under-15`). |
| `OCCP` | Occupation | SOC-based codes mapped to ~80 readable labels; unmapped codes → `Other-Occupation`. |
| `POBP` | Place of birth | U.S. state FIPS codes and country codes mapped to readable names; unmapped → `Other-NEC`. |
| `RELP` | Relationship to household reference person | Unified map covering both the pre-2019 RELP scheme (codes 0–17) and the 2019+ RELSHIPP scheme (codes 20–38). |
| `WKHP` | Usual hours worked per week | Binned into six bands: `Part-Time-Low`, `Part-Time`, `Near-Full-Time`, `Full-Time`, `Over-Full-Time`, `Extended-Hours`. |
| `SEX` | Sex | `Male` / `Female`. |
| `RAC1P` | Race | 9 single-race categories (also used as the sensitive-attribute group). |

The binary **target column** is always appended regardless of `--columns`:
`income_over_<threshold>` (e.g. `income_over_100000`).

---

## Technical notes

### RELP / RELSHIPP column compatibility

The Census Bureau renamed the household-relationship variable between survey
years: raw PUMS files use `RELP` for years ≤ 2018 and `RELSHIPP` for years
≥ 2019.  The official folktables `BasicProblem` always references `RELP` but
does not rename the column automatically.  This pipeline calls
`normalize_raw_columns()` before `df_to_pandas()`, which renames
`RELSHIPP → RELP` on the raw DataFrame when necessary, making the pipeline
transparent across all supported years.

### Parallelization

| Level | Mechanism | Rationale |
|---|---|---|
| Per-state download | `ThreadPoolExecutor` | I/O-bound; each state is a separate HTTP file download. |
| Column categorization | `ThreadPoolExecutor` | Independent per-column transforms; NumPy-backed pandas operations release the GIL. |
| CSV write (split) | 2 threads | Train and test files written concurrently. |
| Multi-year execution | `ProcessPoolExecutor` | CPU-bound; each year is a fully independent workload that bypasses the GIL via separate processes. |
