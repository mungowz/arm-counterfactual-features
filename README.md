# ACS Income Dataset Pipeline

A three-stage command-line pipeline for building binary-classification datasets
from the U.S. Census Bureau's American Community Survey (ACS) Public Use
Microdata Sample (PUMS), computing global feature importance via the **BoCSoR**
algorithm, and mining macroscopic association rules via **FP-Growth**.

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
│   ├── constants.py                # State groups, feature set, bin configs, code maps, OCCP ranges
│   ├── create_dataset.py           # Stage 1: download → encode → split → save
│   ├── feature_importance.py       # Stage 2: BoCSoR XAI (rank encoding, CatBoost, itemsets)
│   ├── macroscopic_data_mining.py  # Stage 3: FP-Growth ARM on macroscopic feature labels
│   └── main.py                     # CLI entry point — orchestrates all three stages
└── data/
    ├── raw/                        # Cached raw PUMS files (auto-created)
    └── *.csv                       # Processed output datasets
```

---

## Requirements

```
folktables
pandas
numpy
scikit-learn
catboost
mlxtend
matplotlib
seaborn
```

Install with:

```bash
pip install folktables pandas numpy scikit-learn catboost mlxtend matplotlib seaborn
```

---

## Quick start

```bash
# Single state, default columns (COW SCHL WKHP), default 80/20 split
python -m src.main --states NY --years 2024

# Custom BoCSoR settings
python -m src.main --states CA NY TX --columns ALL --k 11 --percentile 20

# Explain both class 0 and class 1 boundaries
python -m src.main --states northeast --years 2024 --original-class 0 1

# Multiple years (each year produces its own output sub-directory)
python -m src.main --states midwest --years 2021 2022 2023 2024

# All 49 states
python -m src.main --states ALL --years 2024
```

Each run always produces three stage-1 CSV files (`dataset_*.csv`,
`train_*.csv`, `test_*.csv`) plus the stage-2 BoCSoR output.  If the
files already exist from a previous run they are loaded directly — no
re-download or re-encoding.

---

## Pipeline behaviour

The pipeline always runs all three stages. Re-runs are safe:

- **Stage 1** is skipped if the expected `train_*.csv` / `test_*.csv` files
  already exist in `--data-dir`.
- **Stage 2** skips any class whose `feature_importance[_classN].csv` already
  exists in the output directory.
- **Stage 3** skips any class whose `arm_rules[_classN].csv` already exists
  in the output directory.

---

## Output files

### Stage 1

The filename encodes every parameter that affects the dataset content, so runs
with different configurations never overwrite each other:

Every run always produces all three files together:

```
dataset_2024_NY_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv
train_2024_NY_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv
test_2024_NY_1Y_person_thr100000_colsCOW-SCHL-WKHP.csv
```

When `--states` is a group/region/division name the group name is used
in the filename instead of the full list of state codes
(e.g. `northeast` instead of `CT_MA_ME_NH_NJ_NY_PA_RI_VT`).

Filename pattern:
```
{prefix}_{year}_{states}_{horizon}_{survey}_thr{threshold}_cols{cols}.csv
```

- `{prefix}` — `dataset`, `train`, or `test`
- `{states}` — group name, sorted state codes joined by `_`, or `ALL`
- `{horizon}` — `1Y` or `5Y`
- `{cols}` — feature columns joined by `-` (e.g. `COW-SCHL-WKHP`), or `ALL`

The train/test split is **stratified** on the binary target column (default 80/20).

### Stage 2

All stage-2 outputs land in a subdirectory of `--output-dir` (default
`results/`) that encodes the **state scope** and the **year range**:

```
<output-dir>/<states_tag>/<years_tag>/cols<cols_tag>/pct<N>/
```

| Scenario | Example path |
|---|---|
| `--states northeast --years 2024` | `results/northeast/2024/colsCOW-SCHL-WKHP/pct20/` |
| `--states northeast --years 2024 --columns COW OCCP SCHL WKHP` | `results/northeast/2024/colsCOW-OCCP-SCHL-WKHP/pct20/` |
| `--states northeast --years 2024 --percentile 10` | `results/northeast/2024/colsCOW-SCHL-WKHP/pct10/` |
| `--states ALL --years 2021 2022 2023 2024` | `results/ALL/2021-2024/colsCOW-SCHL-WKHP/pct20/<year>/` |
| `--states midwest --years 2021 2023 --columns ALL` | `results/midwest/2021_2023/colsALL/pct20/` |

Years tag rules: single year → the year itself; contiguous range →
`<first>-<last>`; non-contiguous → years joined by `_`.
Columns tag: feature columns sorted and joined by `-`, prefixed with `cols`
(e.g. `colsCOW-SCHL-WKHP`, `colsALL` for all columns).
Percentile tag: boundary selection percentile as integer, prefixed with `pct`
(e.g. `pct20`, `pct10`).  Together these two tags ensure that runs with any
combination of `--columns` and `--percentile` on the same states and year
never overwrite each other.
When multiple years are processed, each year also gets its own
sub-directory inside the cols/pct folder: `results/ALL/2021-2024/colsCOW-SCHL-WKHP/pct20/2022/`.

| File | Description |
|---|---|
| `feature_importance.csv` | BoCSoR importance scores. Rows: features. Columns: `feature`, `k_1`, `k_3`, …, `k_N`. |
| `feature_importance_itemsets.csv` | All k values merged. Columns: `k_value`, `instance_index`, `features`, `itemset`. One row per boundary instance per k. |
| `feature_importance_itemsets_k<N>.csv` | Same format, one file per k value (e.g. `_k1.csv`, `_k3.csv`, …). |
| `bocsor_summary.md` | Human-readable run summary with importance tables, stability notes, and timing. |
| `arm_rules.csv` | All unique association rules surviving the grid search (support, confidence, lift, lift filter). |
| `arm_grid_summary.csv` | Grid search summary: one row per `(min_support, min_confidence)` cell with the rule count. |

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

**Stage 1** (`create_dataset.py`): if all three expected files
(`dataset_*.csv`, `train_*.csv`, `test_*.csv`) already exist in `--data-dir`,
the download and encoding steps are skipped entirely and the files are loaded
directly.  If any one of the three is missing the full pipeline reruns.

**Stage 2** (`feature_importance.py`): if `feature_importance.csv` (or
`feature_importance_class0.csv` / `feature_importance_class1.csv` when
`--original-class 0 1` is used) already exists in `--output-dir`, that class
is skipped.  Classes whose output is missing are still processed normally.

**Stage 3** (`macroscopic_data_mining.py`): if `association_rules/all_k/arm[suffix]_all_k_rules.csv`
already exists in the output directory, that class is skipped entirely.  Individual
per-k runs (`association_rules/k<N>/`) are also skipped if their output already exists.
k values that were processed but yielded zero rules leave a sentinel file
(`.arm[suffix]_done`) so they are not re-executed on subsequent runs.

In all cases the skip is logged at INFO level so it is always visible.

> To force a re-run, delete the relevant output files first.

---

## Runtime estimates

Estimates based on measured performance on a single-state (NY, ~108K rows) run
with `--k 15 --percentile 20 --workers 14`.

| Scope | Approx. rows | Stage 1 | Stage 2 (BoCSoR) | Notes |
|---|---|---|---|---|
| 1 state (NY) | 108K | ~3s | ~18s | Baseline measurement |
| Northeast (9 states) | ~650K | ~35s | ~8–10 min | Boundary selection dominates |
| South (16 states) | ~1.8M | ~90s | ~8–12 min | BallTree scales well |
| All 49 states (filtered) | ~1.4M | ~5 min | ~6 min | adult_filter reduces rows significantly |

**Scaling note:** stage 2 boundary selection uses a BallTree with Manhattan
distance, scaling as O(N log N) instead of the previous O(N²).
The BallTree is built once in the main process and inherited by worker
processes via `fork` — no serialisation overhead.  The same tree is reused
for the k-NN step (Algorithm 1) inside each worker.
For very large datasets, reducing `--percentile` (e.g. `--percentile 5`)
further cuts the number of boundary instances to process.

## CLI reference

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
| `--test-size` | float | `0.2` | Fraction reserved for the test split (0.0–1.0). Default: 0.2. |
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

### Performance *(stage 1)*

| Option | Type | Default | Description |
|---|---|---|---|
| `--workers` | int | auto-detected | Worker processes for stage-1 multi-year parallelism and stage-2 BoCSoR boundary processing.  Auto-detected as `max(1, min(14, cpu_count - 2))`.  Can be overridden manually. |

### ARM hyperparameters *(stage 3)*

| Option | Type | Default | Description |
|---|---|---|---|
| `--arm-min-support` | float | `0.05` | Minimum support threshold for FP-Growth (lower bound of the grid). |
| `--arm-max-support` | float | `1.00` | Maximum support upper-bound filter (discards rules above this). |
| `--arm-support-step` | float | auto | Step size for the support grid. Auto-computed: targets ~40 levels, rounded to nearest human-readable value (0.005, 0.01, 0.02, 0.025, 0.04, 0.05, 0.1). Override with an explicit value if needed. |
| `--arm-min-confidence` | float | `0.50` | Minimum confidence threshold for rule generation. |
| `--arm-max-confidence` | float | `1.00` | Maximum confidence upper-bound filter. |
| `--arm-confidence-step` | float | auto | Step size for the confidence grid. Same auto-detection algorithm as `--arm-support-step`. |
| `--arm-lift-low` | float | `0.75` | Lower boundary of the lift independence interval. Rules with lift ≥ this AND ≤ `--arm-lift-high` are discarded. |
| `--arm-lift-high` | float | `1.25` | Upper boundary of the lift independence interval. |
| `--arm-k` | int | `None` | If set, process only that k value (`feature_importance_itemsets_k<K>.csv`). Default: process all k values found automatically. |
| `--arm-workers` | int | auto-detected | Thread-pool size for the parallel confidence-axis sweep. Auto-detected as `max(1, min(16, cpu_count - 2))`. |

### Logging

| Option | Values | Default | Description |
|---|---|---|---|
| `--log-level` | `DEBUG` `INFO` `WARNING` `ERROR` | `INFO` | Logging verbosity. |

---

## Stage 3 — Macroscopic Association Rule Mining

Stage 3 implements a **macroscopic** view of the feature co-occurrence patterns
discovered by BoCSoR in stage 2.

### Macroscopic analysis

The itemset files produced by stage 2 contain tokens of the form
`FEATURE=value` (e.g. `SCHL=Bachelors-Degree WKHP=Full-Time`).  Stage 3
discards the *values* and retains only the **feature labels**, so each boundary
instance becomes a transaction over the set of feature *names* that were
relevant for it (e.g. `{SCHL, WKHP}`).

This collapses the fine-grained value-level itemsets into a coarser, more
interpretable signal: which *features* tend to co-occur as relevant at the
decision boundary, independent of the specific value combinations.

### Output directory structure

All stage-3 outputs are written under two sub-folders of the stage-2 output
directory:

```
<output_dir>/
├── feature_importance/          ← stage-2 files live here
└── association_rules/
    ├── k<N>/                    ← one sub-folder per k value
    │   ├── arm[suffix]_rules.csv
    │   ├── arm[suffix]_grid_summary.csv
    │   └── heatmaps/
    │       ├── heatmap_support_confidence[suffix].png
    │       ├── heatmap_support_lift[suffix].png
    │       └── heatmap_confidence_lift[suffix].png
    └── all_k/                   ← rules aggregated across all k values
        ├── arm[suffix]_all_k_rules.csv
        ├── arm[suffix]_all_k_grid_summary.csv
        └── heatmaps/
            ├── heatmap_support_confidence[suffix].png
            ├── heatmap_support_lift[suffix].png
            └── heatmap_confidence_lift[suffix].png
```

When `--original-class 0 1` is used all files carry a `_class0` / `_class1`
suffix, mirroring the stage-2 naming convention.

### FP-Growth

Frequent itemsets are mined from the label-level transactions using the
FP-Growth algorithm (`mlxtend`).  Association rules are then generated and
evaluated with three metrics:

| Metric | Description | Filter |
|---|---|---|
| **support** | Fraction of transactions containing both antecedent and consequent. | `[min_support, max_support]` |
| **confidence** | P(consequent \| antecedent). | `[min_confidence, max_confidence]` |
| **lift** | Ratio of observed to expected co-occurrence under independence. | Keep: lift < `arm_lift_low` (contrast) OR lift > `arm_lift_high` (positive correlation). **Discard**: lift ∈ [`arm_lift_low`, `arm_lift_high`] (near-independence). |

### Grid search

A grid search over all `(min_support, min_confidence)` pairs is run per k value.
The confidence axis is swept in parallel across a thread pool.  FP-Growth is
executed once per distinct support threshold and the result is cached across
all confidence values at that support level.

### Per-k runs and combined output

Stage 3 automatically discovers all per-k itemset files
(`feature_importance_itemsets_k<N>.csv`) produced by stage 2 and runs the full
grid search independently for each k.  Results are written to
`association_rules/k<N>/`.  After all k runs complete, the rules are aggregated
(with deduplication) into `association_rules/all_k/`.

Use `--arm-k K` to restrict processing to a single k value.

### Heatmaps

Three heatmaps are generated for each k value **and** for the aggregated all-k
run.  All heatmaps use a blue colour scale where **darker = more rules**.

| Heatmap | Rows | Columns | Notes |
|---|---|---|---|
| Support × Confidence | min_support values | min_confidence values | Counts from the grid-search summary. |
| Support × Lift | min_support values | lift bins | Counts of surviving rules per bin. |
| Confidence × Lift | min_confidence values | lift bins | Counts of surviving rules per bin. |

On the lift-based heatmaps the **lift independence window**
(`[arm_lift_low, arm_lift_high]`) is annotated with a red hatched band, making
it immediately visible that no rules are generated in that region — only
positive-correlation rules (lift > `arm_lift_high`) and contrast /
negative-correlation rules (lift < `arm_lift_low`) are retained.

### Performance optimisations

The following optimisations are applied in stage 3:

- **Vectorised transaction parsing** — the `itemset` column is parsed with
  pandas `str.split()` + `explode()` + `groupby()` instead of a Python row loop.
- **Boolean matrix pre-computation** — the one-hot boolean matrix for FP-Growth
  is built once with numpy and reused across all support thresholds.
- **FP-Growth cache** — FP-Growth runs exactly once per distinct `min_support`
  value; results are reused for all confidence values at that support level.
- **Adaptive rule generation strategy** — the implementation is chosen at
  runtime by probing the actual rule count at the lowest support threshold:
  with few columns (e.g. 3) `association_rules()` is called once per support
  level and per-cell filtering is a vectorised numpy mask (thread overhead
  would dominate); with many columns (e.g. `--columns ALL`) rule generation
  is expensive and a `ThreadPoolExecutor` evaluates grid cells concurrently.
  The crossover threshold is 500 rules per support level.
- **Vectorised deduplication** — rules are deduplicated via serialised frozenset
  keys and `drop_duplicates()`, replacing row-level loops.
- **Single-pass filter** — support, confidence and lift filters are combined
  into one boolean mask with no intermediate DataFrame materialisation.

### Output files

| File | Description |
|---|---|
| `association_rules/k<N>/arm[suffix]_rules.csv` | Unique rules for k=N surviving all filters. Columns: `k_value`, `antecedents`, `consequents`, `support`, `confidence`, `lift`, `leverage`, `conviction`, `grid_support`, `grid_confidence`. |
| `association_rules/k<N>/arm[suffix]_grid_summary.csv` | Grid summary for k=N: one row per `(min_support, min_confidence)` cell with `n_rules`. |
| `association_rules/k<N>/heatmaps/*.png` | Three heatmaps (support×confidence, support×lift, confidence×lift) for k=N. |
| `association_rules/all_k/arm[suffix]_all_k_rules.csv` | All unique rules aggregated across every k value. |
| `association_rules/all_k/arm[suffix]_all_k_grid_summary.csv` | Aggregated grid summary (rule counts summed across k values). |
| `association_rules/all_k/heatmaps/*.png` | Three heatmaps for the aggregated all-k dataset. |

When `--original-class 0 1` is used all files carry a `_class0` / `_class1`
suffix, mirroring the stage-2 naming convention.

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
- **Worker auto-detection**: `max(1, min(14, cpu_count - 2))` — reserves
  2 logical CPUs for the OS and the main process, caps at 14 to avoid
  diminishing returns from CatBoost's internal thread pools.  The same
  formula is used for both stage-1 multi-year workers and stage-2 BoCSoR
  boundary processing.  Override with `--workers N` if needed.

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