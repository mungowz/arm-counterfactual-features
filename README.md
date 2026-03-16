# ACS Income Dataset Pipeline

A two-stage command-line pipeline for building binary-classification datasets
from the U.S. Census Bureau's American Community Survey (ACS) Public Use
Microdata Sample (PUMS), using the
[folktables](https://github.com/socialfoundations/folktables) library, and
computing global feature importance via the **BoCSoR** algorithm.

The pipeline predicts whether an individual's annual personal income (`PINCP`)
exceeds a configurable threshold.  All categorical features are decoded from
numeric ACS codes to human-readable string labels, and continuous features are
discretised into meaningful bands.  Occupation codes (`OCCP`) are aggregated
into the 23 major groups defined by the
[BLS Occupational Employment and Wage Statistics (OEWS)](https://www.bls.gov/oes/2023/may/oes_stru.htm)
program.

---

## Project structure

```
project/
├── src/
│   ├── __init__.py
│   ├── constants.py           # State groups, feature set, bin configs, code maps, OCCP ranges
│   ├── create_dataset.py      # Stage 1: download → encode → split → save
│   ├── feature_importance.py  # Stage 2: BoCSoR XAI (rank encoding, CatBoost, itemsets)
│   └── main.py                # CLI entry point — orchestrates both stages
└── data/
    ├── raw/                   # Cached raw PUMS files (auto-created)
    └── *.csv                  # Processed output datasets
```

---

## Requirements

```
folktables
pandas
numpy
scikit-learn
catboost
```

Install with:

```bash
pip install folktables pandas numpy scikit-learn catboost
```

---

## Quick start

```bash
# Stage 1 only — create the dataset with an 80/20 split
python -m src.main --steps 1 --states NY --years 2024 --test-size 0.2

# Stage 2 only — BoCSoR on an existing dataset (paths inferred automatically)
python -m src.main --steps 2 --states NY --years 2024

# Stage 2 only — explicit file paths
python -m src.main --steps 2 \
    --train data/train_2024_NY_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv \
    --test  data/test_2024_NY_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv

# Both stages end-to-end
python -m src.main --steps 1 2 --states NY --years 2024 --test-size 0.2

# Both stages, all feature columns, custom BoCSoR settings
python -m src.main --steps 1 2 \
    --states CA NY TX --columns ALL --test-size 0.2 \
    --k 11 --percentile 20 --workers 14

# Four years in parallel, then BoCSoR for each year
python -m src.main --steps 1 2 \
    --years 2021 2022 2023 2024 --states midwest --test-size 0.2
```

---

## Pipeline modes (`--steps`)

| Flag | Behaviour |
|---|---|
| `--steps 1` | **Stage 1 only** — download, encode, and save the dataset CSV(s). |
| `--steps 2` | **Stage 2 only** — run BoCSoR on existing files. If `--train`/`--test` are not given, paths are inferred from the ACS parameters. Requires `--years` to be a single value when inferring paths. |
| `--steps 1 2` | **Both stages** — create the dataset then immediately run BoCSoR. Requires `--test-size > 0`. |

---

## Output files

### Stage 1

The filename encodes every parameter that affects the dataset content, so runs
with different configurations never overwrite each other:

```
train_2024_NY_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv
test_2024_northeast_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv
train_2024_midwest_1Y_person_thr100000_colsALL.csv
dataset_2024_ALL_1Y_person_thr100000_colsALL.csv
```

When `--states` is a group/region/division name the group name is used
in the filename instead of the full list of state codes
(e.g. `northeast` instead of `CT_MA_ME_NH_NJ_NY_PA_RI_VT`).

| `--test-size` | Files produced |
|---|---|
| `0.0` (default) | `data/dataset_<year>_<states>_<horizon>_<survey>_thr<threshold>_cols<cols>.csv` |
| `> 0.0` | `data/train_…csv` + `data/test_…csv` |

- `<states>` — state codes joined by `_` and sorted, or `ALL`
- `<horizon>` — `1Y` or `5Y`
- `<cols>` — feature columns joined by `-` (e.g. `COW-SCHL-WKHP`), or `ALL`

The train/test split is **stratified** on the binary target column.

### Stage 2

All stage-2 outputs land in a subdirectory of `--output-dir` (default
`results/`) that encodes the **state scope** and the **year range**:

```
<output-dir>/<states_tag>/<years_tag>/
```

| Scenario | Example path |
|---|---|
| `--states northeast --years 2024` | `results/northeast/2024/` |
| `--states ALL --years 2021 2022 2023 2024` | `results/ALL/2021-2024/<year>/` |
| `--states CA NY TX --years 2024` | `results/CA_NY_TX/2024/` |
| `--states midwest --years 2021 2023` | `results/midwest/2021_2023/` |

Years tag rules: single year → the year itself; contiguous range →
`<first>-<last>`; non-contiguous → years joined by `_`.
When multiple years are processed with `--steps 1 2`, each year also gets
its own sub-directory inside the years tag folder: `results/ALL/2021-2024/2022/`.

| File | Description |
|---|---|
| `feature_importance.csv` | BoCSoR importance scores. Rows: features. Columns: `feature`, `k_1`, `k_3`, …, `k_N`. |
| `feature_importance_itemsets.csv` | All k values merged. Columns: `k_value`, `instance_index`, `features`, `itemset`. One row per boundary instance per k. |
| `feature_importance_itemsets_k<N>.csv` | Same format, one file per k value (e.g. `_k1.csv`, `_k3.csv`, …). |
| `bocsor_summary.md` | Human-readable run summary with importance tables, stability notes, and timing. |

#### Itemset format

Each row of the itemset files represents one boundary instance for one k value:

| Column | Example | Description |
|---|---|---|
| `k_value` | `5` | Neighbourhood size used. |
| `instance_index` | `1423` | Row index in the training set. |
| `features` | `SCHL WKHP` | Space-separated names of relevant features. |
| `itemset` | `SCHL=Bachelors-Degree WKHP=Full-Time` | ARM-ready string — split on spaces to get individual items. |

To build transaction sets for Apriori/FP-Growth:

```python
import pandas as pd

df = pd.read_csv("results/feature_importance_itemsets_k11.csv")
transactions = df["itemset"].str.split(" ").tolist()
```

---

---

## Skip-if-exists behaviour

Both pipeline stages check for existing output files before doing any work.

**Stage 1** (`create_dataset.py`): if the expected `train_*.csv` + `test_*.csv`
(or `dataset_*.csv`) already exist in `--data-dir`, the download and encoding
steps are skipped entirely and the existing files are loaded and returned.
This makes re-runs after a partial failure or parameter tweak instant.

**Stage 2** (`feature_importance.py`): if `feature_importance.csv` (or
`feature_importance_class0.csv` / `feature_importance_class1.csv` when
`--original-class 0 1` is used) already exists in `--output-dir`, that class
is skipped.  Classes whose output is missing are still processed normally.

In both cases the skip is logged at INFO level so it is always visible.

> To force a re-run, delete the relevant output files first.

---

## Runtime estimates

Estimates based on measured performance on a single-state (NY, ~108K rows) run
with `--k 15 --percentile 20 --workers 14`.

| Scope | Approx. rows | Stage 1 | Stage 2 (BoCSoR) | Notes |
|---|---|---|---|---|
| 1 state (NY) | 108K | ~3s | ~18s | Baseline measurement |
| Northeast (9 states) | ~650K | ~35s | ~8–10 min | Boundary selection dominates |
| South (16 states) | ~1.8M | ~90s | ~90 min | O(N²) distance matrix |
| All 49 states | ~10M | ~5 min | hours–days | Use `--percentile 5` to reduce boundary instances |

**Scaling note:** stage 2 boundary selection uses a BallTree with Manhattan
distance, scaling as O(N log N) instead of the previous O(N²).
The BallTree is built once in the main process and inherited by worker
processes via `fork` — no serialisation overhead.  The same tree is reused
for the k-NN step (Algorithm 1) inside each worker.
For very large datasets, reducing `--percentile` (e.g. `--percentile 5`)
further cuts the number of boundary instances to process.

## CLI reference

### Pipeline mode

| Option | Type | Default | Description |
|---|---|---|---|
| `--steps` | int (one or more) | `1` | Pipeline steps to run: `1` = dataset creation, `2` = BoCSoR, `1 2` = both. |

### ACS parameters *(stage 1)*

| Option | Type | Default | Description |
|---|---|---|---|
| `--years` | int (one or more) | `2024` | Survey year(s). Range: 2014–2024. |
| `--horizon` | `1-Year` \| `5-Year` | `1-Year` | Survey horizon. `1-Year` excludes Alaska. |
| `--survey` | `person` \| `household` | `person` | Unit of analysis. |
| `--states` | str (one or more) | `ALL` | State codes (`CA NY TX`), a group name (`northeast`), or `ALL`. |

### Task parameters *(stage 1)*

| Option | Type | Default | Description |
|---|---|---|---|
| `--threshold` | float | `100000` | Income threshold in USD. Target is **1** if `PINCP > threshold`. |

### Output column selection *(stage 1)*

| Option | Type | Default | Description |
|---|---|---|---|
| `--columns` | str (one or more) | `COW SCHL WKHP` | Feature columns to retain. Pass `ALL` to keep every feature. |

### Train / test split *(stage 1)*

| Option | Type | Default | Description |
|---|---|---|---|
| `--test-size` | float | `0.0` | Fraction reserved for the test set. Must be `> 0` with `--steps 1 2`. |
| `--seed` | int | `42` | Random seed for the stratified split and CatBoost. |

### BoCSoR hyperparameters *(stage 2)*

| Option | Type | Default | Description |
|---|---|---|---|
| `--k` | int (one or more) | `11` | Neighbourhood size(s). A single value K is auto-expanded to all odd integers 1…K (e.g. `--k 11` → 1 3 5 7 9 11). Multiple values used as-is (e.g. `--k 1 5 11`). |
| `--percentile` | float | `20.0` | Percentile threshold for boundary instance selection (0–100). |

### CatBoost hyperparameters *(stage 2)*

| Option | Type | Default | Description |
|---|---|---|---|
| `--cb-iterations` | int | `500` | Boosting rounds (epochs). |
| `--cb-lr` | float | `0.05` | Learning rate. |
| `--cb-depth` | int | `6` | Tree depth. |
| `--cb-early-stopping` | int | `0` | Stop if eval loss does not improve for N rounds. `0` = disabled. When enabled, 20% of training data is held out as validation. |
| `--cb-verbose` | flag | off | Print CatBoost training progress. |

### Input / output

| Option | Type | Default | Description |
|---|---|---|---|
| `--data-dir` | path | `data/` | Root directory for stage-1 output. |
| `--output-dir` | path | `results/` | Directory for stage-2 output files. |
| `--train` | path | — | Existing train CSV. Optional with `--steps 2` (inferred if omitted). |
| `--test` | path | — | Existing test CSV. Optional with `--steps 2` (inferred if omitted). |

### Performance *(stage 1)*

| Option | Type | Default | Description |
|---|---|---|---|
| `--workers` | int | `min(4, cpu_count)` | Parallel worker processes for multi-year runs and BoCSoR boundary processing. |

### Logging

| Option | Values | Default | Description |
|---|---|---|---|
| `--log-level` | `DEBUG` `INFO` `WARNING` `ERROR` | `INFO` | Logging verbosity. |

---

## State groups

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

> **Note on Alaska:** AK is excluded from all groups when using `--horizon 1-Year`.

---

## Available feature columns

| Column | ACS variable | Description |
|---|---|---|
| `AGEP` | Age | Binned: `Young`, `Young-Adult`, `Mid-Career`, `Experienced`, `Late-Career`, `Retirement-Age`. |
| `COW` | Class of worker | 9 categories (e.g. `Employee-Private-For-Profit`, `Self-Employed-Incorporated`). |
| `SCHL` | Educational attainment | 24 levels from `No-Schooling-Completed` to `Doctorate-Degree`. |
| `MAR` | Marital status | 5 categories. |
| `OCCP` | Occupation | 23 BLS OEWS major groups (e.g. `Management`, `Computer-And-Mathematical`). |
| `POBP` | Place of birth | State FIPS and country codes mapped to readable names. |
| `RELP` | Relationship to household reference person | Unified pre-2019 and 2019+ scheme. |
| `WKHP` | Usual hours worked per week | Binned: `Part-Time-Low`, `Part-Time`, `Near-Full-Time`, `Full-Time`, `Over-Full-Time`, `Extended-Hours`. |
| `SEX` | Sex | `Male` / `Female`. |
| `RAC1P` | Race | 9 single-race categories. |

The binary **target column** is always appended: `income_over_<threshold>`.

---

## Stage 2 — BoCSoR feature importance

Stage 2 implements the **Boundary Crossing Solo Ratio (BoCSoR)** algorithm
(Alfeo et al., 2023) adapted for fully-categorical data.

### Algorithm

BoCSoR explains *which features matter most* for instances that sit close to
the classifier's decision boundary.  The analysis runs on the **training set**
— the classifier has full knowledge of training instances, making their
boundary behaviour the most informative signal.

For each boundary instance:

1. Find the k nearest neighbours from the opposite class in rank-encoded
   Manhattan space (**Algorithm 1** — no interpolation, categorical data).
2. For each counterfactual, substitute each differing feature value back to
   the original instance's value one at a time; if the model prediction flips
   back to the original class, that feature is **relevant** (**Algorithm 2**).
   Each feature is tested independently (restored before the next is tested).
3. Take the **union** of relevant features across all k counterfactuals and
   record one itemset row for this boundary instance.

The BoCSoR score for a feature is the fraction of boundary instances for
which it appears in the relevant union:

```
BoCSoR(feature_i) = count of boundary instances where feature_i is relevant
                  ÷ n_boundary_instances_with_counterfactual
```

### Rank-based encoding

All features are categorical, so Manhattan distance is computed on
**rank-encoded** representations:

- **Ordinal columns** (`AGEP`, `SCHL`, `WKHP`) — ranks follow the declared
  semantic order (e.g. `Young=1` < `Young-Adult=2` < … < `Retirement-Age=6`).
- **All other columns** — ranks assigned in **lexicographic (alphabetical)
  order**. This is consistent but carries no semantic meaning for nominal
  features — it only provides unique integers for distance computation.

Rank maps are built **from the training set only** to prevent data leakage.

Distance formula:

```
dist(a, b) = 2 × Σ|rank_i(a) − rank_i(b)| / Σ max_rank_i
```

Result is in [0, 2].

### Multi-k evaluation

A single value `K` passed to `--k` is auto-expanded to all odd integers from
1 to K:

| `--k` | Evaluated k values |
|---|---|
| `11` *(default)* | 1, 3, 5, 7, 9, 11 |
| `7` | 1, 3, 5, 7 |
| `15` | 1, 3, 5, 7, 9, 11, 13, 15 |

Multiple explicit values are used as-is: `--k 1 5 11` → [1, 5, 11].

Running across a ladder of neighbourhood sizes lets you assess how stable the
importance ranking is as the counterfactual search becomes broader.

### Performance optimisations

- **BallTree boundary selection**: O(N log N) instead of O(N²) pairwise
  matrix.  Built once on the cf-class instances, reused for all k values.
  For 650K rows this reduces boundary selection from ~8 min to ~2 min.
- **Boundary instance selection**: computed once, reused for all k values.
  If no boundary instances exist, all k values are skipped immediately.
- **Batched predict**: for each boundary instance, all modified instances
  across all k counterfactuals are collected into a single NumPy array and
  passed to `model.predict()` in one call, eliminating Python round-trip
  overhead.
- **Parallel processing**: boundary instances are split into chunks and
  processed with `ProcessPoolExecutor` using `fork` start method, so workers
  inherit the loaded model, encoded arrays, and BallTree without reimporting.
- **CatBoost thread control**: each worker uses `thread_count=1` to avoid
  competing internal thread pools.

### Classifier

**CatBoost** is used because it accepts raw string-valued categorical columns
with no manual encoding, handles high-cardinality columns (`OCCP`, `POBP`)
robustly via ordered target statistics, and achieves state-of-the-art
accuracy on tabular categorical data (Alfeo et al., 2023).

---

## Technical notes

### RELP / RELSHIPP column compatibility

The Census Bureau renamed the household-relationship variable between survey
years. This pipeline calls `normalize_raw_columns()` before `df_to_pandas()`,
which renames `RELSHIPP → RELP` on the raw DataFrame when necessary.

### OCCP range-based classification

`OCCP` codes are contiguous integer blocks aligned with SOC major groups.
The mapping uses `(lower, upper, label)` tuples — more concise and complete
than enumerating every individual code.

### Parallelisation

| Level | Mechanism | Rationale |
|---|---|---|
| Per-state download | `ThreadPoolExecutor` | I/O-bound. |
| Column categorisation | `ThreadPoolExecutor` | Independent per-column transforms. |
| CSV write (split) | 2 threads | Train and test written concurrently. |
| Multi-year execution | `ProcessPoolExecutor` | CPU-bound, bypasses GIL. |
| BoCSoR boundary chunks | `ProcessPoolExecutor` (fork) | CPU-bound, inherits model via fork. |

---

## Reference

Alfeo, A.L., Zippo, A.G., Catrambone, V., Cimino, M.G.C.A., Toschi, N.,
Valenza, G. (2023). *From local counterfactuals to global feature importance:
efficient, robust, and model-agnostic explanations for brain connectivity
networks.* Computer Methods and Programs in Biomedicine, 236, 107550.