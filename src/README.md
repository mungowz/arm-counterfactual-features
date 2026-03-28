# ACS Income Dataset Pipeline

A four-stage command-line pipeline for building binary-classification datasets
from the U.S. Census Bureau's American Community Survey (ACS) Public Use
Microdata Sample (PUMS), computing global feature importance via the **BoCSoR**
algorithm, mining macroscopic association rules via **FP-Growth**, and drilling
down into value-level microscopic rules anchored to the macroscopic findings.

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
│   ├── feature_importance.py       # Stage 2: BoCSoR XAI (hybrid encoding, CatBoost/MLP, itemsets)
│   ├── macroscopic_data_mining.py  # Stage 3: FP-Growth ARM on macroscopic feature labels
│   ├── microscopic_data_mining.py  # Stage 4: FP-Growth ARM on full LABEL=value tokens
│   └── main.py                     # CLI entry point — orchestrates all four stages
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
# Single state — threshold and dead zone auto-selected
python -m src.main --states NY --years 2024

# Group — threshold auto-selected ($100,700 for northeast)
python -m src.main --states northeast --years 2024

# Custom BoCSoR settings (all columns is already the default)
python -m src.main --states CA NY TX --k 11 --percentile 20

# Use only a subset of columns
python -m src.main --states NY --years 2024 --columns COW SCHL WKHP

# Override the auto-selected threshold explicitly
python -m src.main --states NY --years 2024 --threshold 109500

# Disable the dead zone entirely
python -m src.main --states NY --years 2024 --margin 0

# Custom dead zone half-width
python -m src.main --states northeast --years 2024 --margin 5000

# Multiple years (each year produces its own output sub-directory)
python -m src.main --states midwest --years 2021 2022 2023 2024

# All 49 states — national fallback threshold ($94,200)
python -m src.main --states ALL --years 2024
```

Each run always executes all four stages in sequence. Stage 1 produces three
CSV files (`dataset_*.csv`, `train_*.csv`, `test_*.csv`). Stage 2 produces
BoCSoR output for **both** boundary directions (class 0→1 and class 1→0, in
separate `_class0` / `_class1` files). Stage 3 and 4 produce macroscopic and
microscopic association rules respectively. If any output already exists from
a previous run it is loaded or skipped directly — no re-computation.

The income threshold is **auto-selected** when `--threshold` is omitted:
single state → per-state value; group name → group value; multiple states or
ALL → national fallback. All values follow the Pew Research Center formula
(T = 2 × M_fam ÷ √3, ACS 2024 family medians).

The **dead zone** is enabled by default: individuals whose income falls
within ±margin of the threshold are excluded from the dataset.  The margin
is auto-computed from the ACS Margin of Error (MOE) for median family
income, propagated through the Pew formula (margin = 2 × MOE / √3).
Pass `--margin 0` to disable.

---

## Pipeline behaviour

The pipeline always runs all four stages. Re-runs are safe:

- **Stage 1** is skipped if the expected `train_*.csv` / `test_*.csv` files
  already exist in `--data-dir`.
- **Stage 2** always explains both boundary directions (class 0→1 and class
  1→0).  Each direction produces its own `_class0` / `_class1` suffixed output
  files.  A direction is skipped only if its `feature_importance_class<N>.csv`
  already exists in the output directory.
- **Stage 3** skips any class whose `association_rules/all_k/arm[suffix]_all_k_rules.csv`
  already exists.  Individual per-k runs are also skipped if their output exists.
  k values with zero rules leave a sentinel file (`.arm[suffix]_done`) so they
  are not re-executed.
- **Stage 4** skips any k value whose `micro[suffix]_rules.csv` already exists
  in the corresponding `micro/` sub-folder.  The all_k aggregation is always
  refreshed when any per-k result is new.

---

## Output files

### Stage 1

The filename encodes every parameter that affects the dataset content, so runs
with different configurations never overwrite each other:

Every run always produces all three files together:

```
dataset_2024_NY_1Y_person_thr100000_colsALL.csv
train_2024_NY_1Y_person_thr100000_colsALL.csv
test_2024_NY_1Y_person_thr100000_colsALL.csv
```

When `--states` is a group/region/division name the group name is used
in the filename instead of the full list of state codes
(e.g. `northeast` instead of `CT_MA_ME_NH_NJ_NY_PA_RI_VT`).

Filename pattern:
```
{prefix}_{year}_{states}_{horizon}_{survey}_thr{threshold}[_dz{margin}]_cols{cols}.csv
```

- `{prefix}` — `dataset`, `train`, or `test`
- `{states}` — group name, sorted state codes joined by `_`, or `ALL`
- `{horizon}` — `1Y` or `5Y`
- `_dz{margin}` — dead zone half-width (only present when margin > 0)
- `{cols}` — feature columns joined by `-` (e.g. `COW-SCHL-WKHP`), or `ALL`

The train/test split is **fixed at 80/20**, stratified on the binary target column.

### Stage 2

All stage-2 outputs land in a subdirectory of `--output-dir` (default
`results/`) that encodes the **state scope**, **year range**, **columns**,
**threshold**, **percentile**, and **classifier** so that different
configurations never overwrite each other:

```
<output-dir>/<states_tag>/<years_tag>/cols<cols_tag>/thr<N>/pct<N>/<classifier>/
```

| Scenario | Example path |
|---|---|
| `--states northeast --years 2024` | `results/northeast/2024/colsALL/thr100700/pct20/catboost/` |
| `--states northeast --years 2024 --columns COW OCCP SCHL WKHP` | `results/northeast/2024/colsCOW-OCCP-SCHL-WKHP/thr100700/pct20/catboost/` |
| `--states northeast --years 2024 --percentile 10` | `results/northeast/2024/colsALL/thr100700/pct10/catboost/` |
| `--states northeast --years 2024 --classifier mlp` | `results/northeast/2024/colsALL/thr100700/pct20/mlp/` |
| `--states ALL --years 2021 2022 2023 2024` | `results/ALL/2021-2024/colsALL/thr94200/pct20/catboost/<year>/` |
| `--states midwest --years 2021 2023 --columns COW SCHL WKHP` | `results/midwest/2021_2023/colsCOW-SCHL-WKHP/thr91100/pct20/catboost/` |
| `--states NY --threshold 50000` | `results/NY/2024/colsALL/thr50000/pct20/catboost/` |

Years tag rules: single year → the year itself; contiguous range →
`<first>-<last>`; non-contiguous → years joined by `_`.
Columns tag: feature columns sorted and joined by `-`, prefixed with `cols`
(e.g. `colsCOW-SCHL-WKHP` when using `--columns COW SCHL WKHP`, or `colsALL`
when using the default — all columns).
Threshold tag: income threshold as integer, prefixed with `thr`
(e.g. `thr94200`, `thr50000`).
Percentile tag: boundary selection percentile as integer, prefixed with `pct`
(e.g. `pct20`, `pct10`).  Together these tags ensure that runs with any
combination of `--columns`, `--threshold`, and `--percentile` on the same
states and year never overwrite each other.
When multiple years are processed, each year also gets its own
sub-directory inside the cols/thr/pct folder: `results/ALL/2021-2024/colsALL/thr94200/pct20/catboost/2022/`.

| File | Description |
|---|---|
| `feature_importance.csv` | BoCSoR importance scores. Rows: features. Columns: `feature`, `k_1`, `k_3`, …, `k_N`. |
| `feature_importance_itemsets.csv` | All k values merged. Columns: `k_value`, `instance_index`, `features`, `itemset`. One row per boundary instance per k. |
| `feature_importance_itemsets_k<N>.csv` | Same format, one file per k value (e.g. `_k1.csv`, `_k3.csv`, …). |
| `bocsor_distances.csv` | One row per `(boundary_instance, k_neighbour)` pair. Columns: `k_value`, `instance_index`, `cf_index`, `k_neighbour_rank` (1 = closest), `distance` (hybrid Manhattan, raw sum, guaranteed > 0; 1.0 = one nominal feature change), `n_diff_features` (number of original features that differ).  Sorted by `k_value`, `instance_index`, `k_neighbour_rank` ascending. |
| `bocsor_filter_stats.csv` | One row per k value. Columns: `k`, `boundary_instances`, `instances_with_cf`, `instances_filtered_dist0`, `pct_filtered`, `instances_with_relevant_features`. Shows how many boundary instances were discarded by the distance > 0 filter at each k. |
| `plots/bocsor_distance_histogram_rank<N>.png` | Histogram of distances to the N-th nearest counterfactual, stacked by number of differing features.  X-axis: hybrid Manhattan distance (1.0 = one nominal change). Y-axis: number of instances. Red dashed line: median. |
| `plots/bocsor_distance_histograms_per_rank.png` | Combined figure with one subplot per neighbour rank (1st through k-th), stacked by differing features, for side-by-side comparison of how distance grows with rank. |
| `plots/bocsor_diff_features_pct.png` | Stacked percentage bars showing, for each neighbour rank, what fraction of counterfactuals differ in 1, 2, 3, … features. Percentages > 5% are labelled inside the bars. |
| `bocsor_summary.md` | Human-readable run summary with importance tables, stability notes, and timing. |
| `pipeline.log` | Full pipeline log (all stages, append mode across re-runs). |
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

**Stage 2** (`feature_importance.py`): both boundary directions (class 0→1
and class 1→0) are always computed, producing `feature_importance_class0.csv`
and `feature_importance_class1.csv`.  A direction is skipped only if its
output file already exists in `--output-dir`.

**Stage 3** (`macroscopic_data_mining.py`): if `association_rules/all_k/arm[suffix]_all_k_rules.csv`
already exists in the output directory, that class is skipped entirely.  Individual
per-k runs (`association_rules/k<N>/`) are also skipped if their output already exists.
k values that were processed but yielded zero rules leave a sentinel file
(`.arm[suffix]_done`) so they are not re-executed on subsequent runs.

**Stage 4** (`microscopic_data_mining.py`): if `micro[suffix]_rules.csv` already
exists in a `association_rules/k<N>/micro/` folder, that k is skipped.  The
all_k microscopic aggregation is always regenerated from the available per-k
results when at least one new k was processed.

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
| `--threshold` | float | *auto* | Income threshold in USD. Target is **1** if `PINCP > threshold`. If omitted, the threshold is selected automatically from pre-computed Pew Research Center upper-income values (T = 2 × M_fam ÷ √3, ACS 2024). Resolution order: single state → state-level value; group name → group value; multiple states or ALL → national fallback ($94,200). Pass an explicit value to override. |
| `--margin` | float | *auto* | Dead zone half-width in dollars. Individuals with PINCP in [threshold − margin, threshold + margin] are excluded. Default: auto-computed from ACS MOE for median family income (margin = 2 × MOE / √3). Pass `0` to disable. |

### Output column selection *(stage 1)*

| Option | Type | Default | Description |
|---|---|---|---|
| `--columns` | str (one or more) | `ALL` | Feature columns to retain. Default: all feature columns. Pass specific names to filter (e.g. `--columns COW SCHL WKHP`). |

### Train / test split *(stage 1)*

The split is fixed at **80 % train / 20 % test**, stratified on the binary
target.  Use `--seed` to control reproducibility.

| Option | Type | Default | Description |
|---|---|---|---|
| `--seed` | int | `42` | Random seed for the stratified split and classifier. |

### BoCSoR hyperparameters *(stage 2)*

| Option | Type | Default | Description |
|---|---|---|---|
| `--k` | int (one or more) | `11` | Neighbourhood size(s). A single value K is auto-expanded to all odd integers 1…K (e.g. `--k 11` → 1 3 5 7 9 11). Multiple values used as-is (e.g. `--k 1 5 11`). |
| `--percentile` | float | `20.0` | Percentile threshold for boundary instance selection (0–100). |
| `--classifier` | choice | `catboost` | Classifier. `catboost` accepts raw string categoricals natively. `mlp` (Multi-Layer Perceptron) provides a fundamentally different decision boundary geometry, useful for verifying model-agnosticity. |

### Classifier hyperparameters *(stage 2 — shared by CatBoost and MLP)*

| Option | Type | Default | Description |
|---|---|---|---|
| `--cb-iterations` | int | `500` | Boosting rounds / training epochs. |
| `--cb-lr` | float | `0.05` | Learning rate. |
| `--cb-depth` | int | `6` | Tree depth (CatBoost) / hidden layer size exponent (MLP: two layers of width `2^depth`). |
| `--cb-early-stopping` | int | `0` | Stop if eval loss does not improve for N rounds. `0` = disabled. When enabled, 20% of training data is held out as validation (CatBoost) or 10% (MLP). |
| `--cb-verbose` | flag | off | Print training progress. |

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

### Micro ARM hyperparameters *(stage 4)*

| Option | Type | Default | Description |
|---|---|---|---|
| `--micro-min-support` | float | `0.01` | Minimum support for microscopic FP-Growth. Lower than macro (`0.05`) because filtered transaction sets are smaller (only transactions containing all labels of the anchor rule). |
| `--micro-max-support` | float | `1.00` | Maximum support upper-bound filter. |
| `--micro-support-step` | float | auto | Step size for the microscopic support grid (same auto-detection as `--arm-support-step`). |
| `--micro-min-confidence` | float | `0.30` | Minimum confidence for microscopic rule generation. Lower than macro (`0.50`) to capture a wider range of value-level patterns in smaller subsets. |
| `--micro-max-confidence` | float | `1.00` | Maximum confidence upper-bound filter. |
| `--micro-confidence-step` | float | auto | Step size for the microscopic confidence grid. |
| `--micro-lift-low` | float | `0.75` | Lower boundary of the lift independence interval — same as macro. Rules with lift ∈ [`--micro-lift-low`, `--micro-lift-high`] are discarded. |
| `--micro-lift-high` | float | `1.25` | Upper boundary of the lift independence interval — same as macro. |
| `--micro-k` | int | `None` | If set, run microscopic ARM only for this k value. Default: all k values for which macroscopic rules exist. |
| `--micro-workers` | int | auto-detected | Thread-pool size for the microscopic grid search (parallel path only). |

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
    │   ├── arm[suffix]_frequent_itemsets.csv
    │   ├── heatmaps/
    │   │   ├── heatmap_support_confidence[suffix].png
    │   │   ├── heatmap_support_lift[suffix].png
    │   │   └── heatmap_confidence_lift[suffix].png
    │   └── micro/               ← stage-4 microscopic outputs for this k
    │       ├── micro[suffix]_rules.csv
    │       ├── micro[suffix]_grid_summary.csv
    │       └── heatmaps/
    │           ├── heatmap_support_confidence[suffix].png
    │           ├── heatmap_support_lift[suffix].png
    │           └── heatmap_confidence_lift[suffix].png
    └── all_k/                   ← rules aggregated across all k values
        ├── arm[suffix]_all_k_rules.csv
        ├── arm[suffix]_all_k_grid_summary.csv
        ├── heatmaps/
        │   ├── heatmap_support_confidence[suffix].png
        │   ├── heatmap_support_lift[suffix].png
        │   └── heatmap_confidence_lift[suffix].png
        └── micro/               ← stage-4 microscopic outputs aggregated
            ├── micro[suffix]_all_k_rules.csv
            ├── micro[suffix]_all_k_grid_summary.csv
            └── heatmaps/
                ├── heatmap_support_confidence[suffix].png
                ├── heatmap_support_lift[suffix].png
                └── heatmap_confidence_lift[suffix].png
```

All files carry a `_class0` / `_class1` suffix since both boundary directions
are always computed, mirroring the stage-2 naming convention.

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
FP-Growth is executed once per distinct support threshold and the result is
cached.  Rule generation and filtering use an **adaptive strategy** chosen at
runtime by probing the actual rule volume at the lowest support threshold (see
Performance optimisations below).

### Per-k runs and combined output

Stage 3 automatically discovers all per-k itemset files
(`feature_importance_itemsets_k<N>.csv`) produced by stage 2 — searching both
the root output directory and the `feature_importance/` sub-folder — and runs
the full grid search independently for each k.  Results are written to
`association_rules/k<N>/`.  After all k runs complete, the rules are aggregated
(with deduplication) into `association_rules/all_k/`.

Use `--arm-k K` to restrict processing to a single k value.

Stage-2 files (`feature_importance*.csv`, `bocsor_summary*.md`) are moved into
the `feature_importance/` sub-folder automatically at the end of stage 3, after
all reads are complete.

### Heatmaps

Three heatmaps are generated for each k value **and** for the aggregated all-k
run.  All heatmaps use a blue colour scale where **darker = more rules**.  A
subtitle on each PNG shows the exact active filter thresholds, making every
image fully self-documenting.

| Heatmap | Rows | Columns | Cell value | Notes |
|---|---|---|---|---|
| Support × Confidence | `min_support` thresholds (grid) | `min_confidence` thresholds (grid) | n_rules at that threshold pair | Direct view of where the grid produces rules |
| Support × Lift | actual `support` values (binned) | actual `lift` values (binned) | n_rules in that bin pair | Shows where rules concentrate in metric space |
| Confidence × Lift | actual `confidence` values (binned) | actual `lift` values (binned) | n_rules in that bin pair | Shows where rules concentrate in metric space |

On the lift-based heatmaps the **lift independence window**
(`[arm_lift_low, arm_lift_high]`) is annotated with a red hatched band, making
it immediately visible that no rules fall in that region — only
positive-correlation rules (lift > `arm_lift_high`) and contrast /
negative-correlation rules (lift < `arm_lift_low`) are retained.

### Performance optimisations

- **Transaction parsing via numpy lexsort + split** — after pandas `str.split()`
  + `explode()` produces the flat token frame, transactions are reconstructed
  using `np.lexsort((labels, txn_ids))` to sort the flat arrays, `np.diff` to
  locate transaction boundaries, and `np.split` to partition the label array.
  This replaces `groupby + apply(lambda s: sorted(s.tolist()))` and is **~2×
  faster** on large per-k files (e.g. 44 k rows at k=15 for ALL states).
- **Boolean matrix pre-computation** — the one-hot boolean matrix for FP-Growth
  is built once with numpy advanced indexing (`arr[row_idx, col_idx] = True`)
  and reused across all support thresholds.
- **FP-Growth cache** — FP-Growth runs exactly once per distinct `min_support`
  value; results are reused for all confidence values at that support level.
- **Adaptive rule generation strategy** — the implementation is chosen at
  runtime by probing the actual rule count at the lowest support threshold:
  with few columns (e.g. `--columns COW SCHL WKHP`) `association_rules()` is
  called once per support level and per-cell filtering is a vectorised numpy
  mask (thread overhead would dominate); with many columns (the default — all
  10 features) rule generation is expensive and a `ThreadPoolExecutor`
  evaluates grid cells concurrently.
  The crossover threshold is 500 rules per support level.
- **Vectorised deduplication** — rules are deduplicated via serialised frozenset
  keys and `drop_duplicates()`, replacing row-level loops.
- **Single-pass filter** — support, confidence and lift filters are combined
  into one boolean mask with no intermediate DataFrame materialisation.

### Output files

#### Rules CSV (`arm[suffix]_rules.csv`)

Each row is a unique association rule that survived all filters.  Every row is
fully self-describing — no external file is needed to interpret the thresholds.

| Column | Description |
|---|---|
| `k_value` | k value (per-k files only) |
| `antecedents` | Feature label(s) in the antecedent, joined by ` & ` |
| `consequents` | Feature label(s) in the consequent |
| `antecedent support` | P(antecedent) |
| `consequent support` | P(consequent) |
| `support` | P(antecedent ∪ consequent) — actual rule metric |
| `confidence` | P(consequent \| antecedent) — actual rule metric |
| `lift` | Observed / expected co-occurrence — actual rule metric |
| `leverage` | P(A∪C) − P(A)·P(C) |
| `conviction` | (1 − P(C)) / (1 − confidence) — written as empty cell when `confidence = 1.0` (mathematically ∞) |
| `lift_type` | `positive_correlation` (lift > `arm_lift_high`) or `negative_correlation` (lift < `arm_lift_low`) |
| `grid_min_support` | `min_support` threshold at which this rule was found |
| `grid_min_confidence` | `min_confidence` threshold at which this rule was found |
| `filter_min_support` | Global lower bound of the support grid |
| `filter_max_support` | Global upper bound of the support grid |
| `filter_min_confidence` | Global lower bound of the confidence grid |
| `filter_max_confidence` | Global upper bound of the confidence grid |
| `filter_lift_kept_below` | Rules with lift below this are kept (negative correlation) |
| `filter_lift_kept_above` | Rules with lift above this are kept (positive correlation) |
| `filter_lift_discarded` | Discarded lift interval, e.g. `[0.75, 1.25]` |

All floating-point values are written with `%.6f` format (6 fixed decimal
places).  This guarantees that values like `1.0` appear as `1.000000` and
`0.625` as `0.625000` — never truncated to bare integers or missing the
leading zero.  Infinities (`conviction` when `confidence = 1.0`) are replaced
with an empty cell before writing.

Rows are sorted **alphabetically** by `antecedents` then `consequents`.
Within each cell (antecedents, consequents) tokens are joined in alphabetical
order (e.g. `COW & SCHL`, never `SCHL & COW`).  This makes symmetric rule
pairs (e.g. `SCHL → WKHP` and `WKHP → SCHL`) easy to identify by eye.

Columns removed (not informative for this analysis): `zhangs_metric`, `jaccard`,
`certainty`, `kulczynski`, `representativity`.

#### Frequent itemsets CSV (`arm[suffix]_frequent_itemsets.csv`)

One file per k value containing the FP-Growth frequent itemsets at the lowest
`min_support` threshold of the grid (i.e. the most permissive run, producing
the maximum number of itemsets).  Sorted descending by support.

| Column | Description |
|---|---|
| `k_value` | k value for this run |
| `itemsets` | Frozenset serialised as tokens joined by ` & `, e.g. `SCHL & WKHP` |
| `support` | Fraction of transactions containing this itemset |

#### Grid summary CSV (`arm[suffix]_grid_summary.csv`)

One row per `(min_support, min_confidence)` grid cell.  Columns: `min_support`,
`min_confidence`, `n_rules`, plus the same `filter_*` columns as the rules CSV.

#### Heatmap PNGs (`heatmaps/*.png`)

| File | Description |
|---|---|
| `heatmap_support_confidence[suffix].png` | Support (rows) × Confidence (cols), threshold view |
| `heatmap_support_lift[suffix].png` | Support (rows) × Lift (cols), actual-value binned view |
| `heatmap_confidence_lift[suffix].png` | Confidence (rows) × Lift (cols), actual-value binned view |

All files carry a `_class0` / `_class1` suffix since both boundary directions
are always computed, mirroring the stage-2 naming convention.

---

## Stage 4 — Microscopic Association Rule Mining

Stage 4 is the **microscopic** companion to stage 3.  While stage 3 discards
feature values and works only with feature *labels* (e.g. `{SCHL, WKHP}`),
stage 4 retains the full `LABEL=value` tokens (e.g. `{SCHL=Bachelors-Degree,
WKHP=Full-Time}`) to produce value-level association rules anchored to the
macroscopic findings.

### Relation to macroscopic rules

For each macroscopic rule `antecedent_labels → consequent_labels` produced by
stage 3, the microscopic analysis:

1. Extracts the set of feature labels that appear in the rule's antecedent
   **and** consequent (their union).
2. Filters the itemset CSV: retains only boundary instances whose itemset
   contains **at least one token for every label** in that set — i.e. the
   transaction must have both a `SCHL=*` token and a `WKHP=*` token for a
   macro rule `WKHP → SCHL`.  Transactions that contain only one of the two
   labels are excluded.  Extra tokens whose label is not in the set are kept,
   preserving the full value context of each instance.
3. Runs an independent FP-Growth grid search on the filtered microscopic
   transactions, using its own lower thresholds (`min_support=0.01`,
   `min_confidence=0.30`) independent of the macroscopic grid parameters.
4. Annotates every surviving microscopic rule with `macro_rule_id`,
   `macro_antecedents`, and `macro_consequents` so the value-level finding
   can always be traced back to its macroscopic anchor.

**Why require all labels**: requiring all labels of the macro rule ensures that
only instances where every feature in the rule is simultaneously discriminant
are analysed.  With the old "at least one" filter, a transaction containing
only `SCHL=Bachelors-Degree` (but no `WKHP` token) would be included in the
`WKHP → SCHL` analysis — diluting the signal and producing itemsets unrelated
to the joint WKHP/SCHL relationship identified at the macroscopic level.

**Default thresholds vs macroscopic**

| Parameter | Macro default | Micro default | Reason |
|---|---|---|---|
| `min_support` | `0.05` | `0.01` | Filtered subsets are smaller — lower threshold captures rare but real co-occurrences |
| `min_confidence` | `0.50` | `0.30` | Smaller subsets need a wider confidence window to surface patterns |
| lift window | `[0.75, 1.25]` | `[0.75, 1.25]` | Same independence criterion — discards near-independent rules |

### Output files

#### Microscopic rules CSV (`micro[suffix]_rules.csv`)

Each row is a unique value-level association rule.  Columns prepend the
macroscopic provenance before the standard metric columns:

| Column | Description |
|---|---|
| `macro_rule_id` | 0-based index of the macroscopic anchor rule |
| `macro_antecedents` | Macroscopic antecedent label(s), e.g. `SCHL` |
| `macro_consequents` | Macroscopic consequent label(s), e.g. `WKHP` |
| `k_value` | k value (per-k files only) |
| `antecedents` | Microscopic antecedent `LABEL=value` item(s) |
| `consequents` | Microscopic consequent `LABEL=value` item(s) |
| `antecedent support` | P(antecedent) |
| `consequent support` | P(consequent) |
| `support` | P(antecedent ∪ consequent) |
| `confidence` | P(consequent \| antecedent) |
| `lift` | Observed / expected co-occurrence |
| `leverage` | P(A∪C) − P(A)·P(C) |
| `conviction` | (1 − P(C)) / (1 − confidence) — empty cell when `confidence = 1.0` (mathematically ∞) |
| `lift_type` | `positive_correlation` or `negative_correlation` |
| `grid_min_support` | min_support threshold that produced this rule |
| `grid_min_confidence` | min_confidence threshold that produced this rule |
| `filter_*` | Active filter thresholds (self-documenting) |

All floating-point values are written with `%.6f` format (6 fixed decimal
places — same as stage 3).  `conviction = inf` (when `confidence = 1.0`) is
replaced with an empty cell.

Rows are sorted **alphabetically** by `macro_antecedents`, `macro_consequents`,
`antecedents`, `consequents` — rules from the same macroscopic anchor are
grouped together, and within each group symmetric pairs (`A→B` / `B→A`) are
adjacent for easy identification.

#### Microscopic grid summary CSV (`micro[suffix]_grid_summary.csv`)

One row per `(macro_rule_id, min_support, min_confidence)` triplet with
`n_rules` and the same `filter_*` columns.

#### Heatmap PNGs

Same three heatmaps as stage 3 (Support×Confidence, Support×Lift,
Confidence×Lift), generated per k and for all_k, saved under
`association_rules/k<N>/micro/heatmaps/` and `association_rules/all_k/micro/heatmaps/`.

### Performance

Stage 4 shares all performance infrastructure with stage 3:

- **Parallel per-macro-rule processing** — each macroscopic rule generates
  an independent filtered transaction set and grid search with no shared
  mutable state.  The per-macro-rule loop uses `ProcessPoolExecutor` with
  fork (exploded tokens inherited in shared memory).  Inner grid search
  worker count is reduced to avoid oversubscription:
  `total_threads ≈ n_parallel × inner_workers ≤ cpu_count`.
  With 20 macro rules and 14 cores this gives ~14× speedup vs sequential.
  Falls back to sequential for a single rule or single worker.
- The exploded token DataFrame is loaded **once per k** and reused for all
  macroscopic rules — no repeated CSV reads.
- **Transaction filtering via numpy lexsort + split** — `_filter_transactions_for_rule`
  uses `pandas.isin()` to identify matching transaction IDs, then reconstructs
  the token lists with `np.lexsort + np.split` instead of `groupby + apply`.
  This is **~4× faster** on large token frames (e.g. 57 k tokens at k=15 for
  ALL states) and is called once per macroscopic rule per k.
- The grid search uses the same **adaptive strategy** (vectorised path for
  few items, threaded path for many items).
- All floating-point values in CSV output use **`%.6f`** format (6 fixed
  decimal places): `1.0` → `1.000000`, `0.625` → `0.625000`.  Infinities
  (`conviction` when `confidence = 1.0`) are replaced with empty cells.

---

## Income thresholds (auto-selected)

When `--threshold` is omitted the pipeline selects the threshold automatically
using the **Pew Research Center upper-income formula** (Kochhar, 2022):

```
T = 2 × M_fam ÷ √3
```

where M_fam is the ACS 1-year 2024 family median income for the state or
group.  Values are rounded to the nearest $100.

### National fallback

| Scope | Threshold |
|---|---|
| USA (49 states, no AK) | **$94,200** |

### Per state-group thresholds

| Group | Threshold | vs USA |
|---|---|---|
| `northeast` | $100,700 | +7% |
| `midwest` | $91,100 | −3% |
| `south` | $86,000 | −9% |
| `west` | $101,400 | +8% |
| `new_england` | $111,300 | +18% |
| `middle_atlantic` | $96,500 | +2% |
| `east_north_central` | $90,400 | −4% |
| `west_north_central` | $92,400 | −2% |
| `south_atlantic` | $89,800 | −5% |
| `east_south_central` | $73,700 | −22% |
| `west_south_central` | $86,300 | −8% |
| `mountain` | $93,300 | −1% |
| `pacific` | $106,000 | +13% |
| `sunbelt` | $92,600 | −2% |
| `rust_belt` | $90,900 | −4% |
| `great_plains` | $92,400 | −2% |

### Per-state thresholds (sorted by value)

| # | State | Threshold | # | State | Threshold |
|---|---|---|---|---|---|
| 1 | DC | $126,700 | 26 | WI | $91,800 |
| 2 | MA | $122,000 | 27 | MT | $91,700 |
| 3 | MD | $117,900 | 28 | TX | $90,800 |
| 4 | UT | $116,600 | 29 | MO | $90,100 |
| 5 | NH | $114,200 | 30 | PA | $89,500 |
| 6 | VA | $112,000 | 31 | WY | $89,000 |
| 7 | CO | $111,700 | 32 | MI | $88,900 |
| 8 | HI | $111,200 | 33 | IN | $87,800 |
| 9 | WA | $108,500 | 34 | ND | $87,300 |
| 10 | CT | $106,400 | 35 | ME | $86,600 |
| 11 | NJ | $105,800 | 36 | OH | $85,700 |
| 12 | MN | $104,000 | 37 | ID | $85,300 |
| 13 | NE | $103,300 | 38 | TN | $84,300 |
| 14 | CA | $103,100 | 39 | GA | $83,100 |
| 15 | OR | $102,200 | 40 | FL | $83,100 |
| 16 | IL | $101,200 | 41 | SC | $79,800 |
| 17 | DE | $100,500 | 42 | NC | $79,700 |
| 18 | VT | $98,100 | 43 | OK | $77,700 |
| 19 | KS | $97,100 | 44 | AR | $73,000 |
| 20 | AZ | $95,800 | 45 | KY | $71,600 |
| 21 | SD | $94,700 | 46 | NM | $69,400 |
| 22 | RI | $94,700 | 47 | AL | $69,300 |
| 23 | NY | $94,400 | 48 | WV | $69,300 |
| 24 | NV | $93,500 | 49 | LA | $66,400 |
| 25 | IA | $92,900 | 50 | MS | $63,500 |

Alaska (AK) is excluded from the ACS 1-year survey and has no threshold entry.
Puerto Rico (PR) falls back to the national value ($94,200).
Source: ACS 1-year 2024, Census Bureau (ACSBR-025).

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

### How the original paper uses data

The paper states that BoCSoR operates on the training set.  In practice the
training set serves as the **starting pool** — the actual counterfactuals used
for feature importance are **synthetic**:

1. Boundary instances are selected from the training set (percentile filter
   on inter-class distance) — these are real training rows.
2. For each boundary instance, k nearest neighbours of the opposite class are
   retrieved from the training set — also real.
3. Intermediate points are generated via `np.linspace` between the boundary
   instance and each neighbour — these are **synthetic**, not in any dataset.
4. The first synthetic midpoint classified as the opposite class becomes the
   `closestCF` — a **synthetic** point near the decision boundary.
5. Feature substitution on `closestCF` creates yet another **synthetic**
   instance, on which `model.predict()` is called — the model is probed on
   an instance it has never seen during training.

The model is therefore queried on synthetic, unseen instances to explore its
decision surface.  This is functionally equivalent to probing with test data,
except the probe points are constructed strategically near the boundary rather
than sampled randomly.

### Categorical adaptation

With fully-categorical data, interpolation is not meaningful — there is no
continuous path between `"Bachelors-Degree"` and `"Doctorate-Degree"`.  Our
adaptation preserves the spirit of the algorithm:

For each boundary instance:

1. Find the k nearest neighbours from the opposite class in hybrid-encoded
   Manhattan space that **differ in at least one feature** (distance > 0).
   Cross-class duplicates (identical feature vectors, different label) are
   skipped by over-querying the BallTree and filtering.  This guarantees
   every counterfactual has ≥ 1 feature to substitute, producing clean
   itemsets for downstream ARM (**adapted Algorithm 1**).
2. For each counterfactual, substitute each differing feature value back to
   the original instance's value one at a time; if the model prediction flips
   back to the original class, that feature is **relevant** (**Algorithm 2**).
   Each feature is tested independently (restored before the next is tested).
   The modified counterfactual is a **synthetic instance** that likely does
   not exist in the training set — the model is probed on unseen data, just
   as in the original paper.
3. Take the **union** of relevant features across all k counterfactuals and
   record one itemset row for this boundary instance.

The BoCSoR score for a feature is the fraction of boundary instances for
which it appears in the relevant union:

```
BoCSoR(feature_i) = count of boundary instances where feature_i is relevant
                  ÷ n_boundary_instances_with_counterfactual
```

### Hybrid distance encoding

All features are categorical.  Manhattan distance is computed on a
**hybrid-encoded** representation that makes ordinal and nominal columns
commensurable — each column contributes values in [0, 1]:

- **Ordinal columns** (`AGEP`, `SCHL`, `WKHP`) — rank-based encoding
  normalised per-column with **min-max** to [0, 1].  Ranks follow the
  declared semantic order (e.g. `Young=1` < `Young-Adult=2` < … <
  `Retirement-Age=6`).  Unknown values fall back to 0.5 (mid-range neutral).

- **Nominal columns** (all others) — **one-hot encoding**, with each bit
  divided by 2 (values in {0.0, 0.5}).  Within a single nominal column, two
  samples either share the same category (Manhattan distance = 0) or differ
  (distance = 0.5 + 0.5 = 1.0, equivalent to Hamming distance = 1).  This
  ensures nominal columns also contribute values in [0, 1] per original
  column, identical to ordinal columns.

Encoding maps are built **from the training set only** to prevent data leakage.

Distance formula (raw hybrid Manhattan, no global normalisation):

```
dist(a, b) = Σ_i |enc_i(a) - enc_i(b)|
```

where `enc_i ∈ [0, 1]` for every encoded column `i`.  A single nominal
feature change contributes exactly **1.0**; a single ordinal step in SCHL
(23 levels) contributes **1/22 ≈ 0.045**.  The distance is directly
interpretable as "how many feature-equivalent changes apart".

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

- **Vectorised hybrid encoding**: ordinal columns use pre-normalised dict
  lookup via `pd.Series.map(dict)` (C-optimised in pandas) instead of a
  per-row Python lambda.  Nominal columns use numpy boolean vectorisation.
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
- **Classifier thread control**: each worker uses `thread_count=1` (CatBoost) to avoid competing internal thread pools. The MLP wrapper silently drops this keyword argument.
- **Worker auto-detection**: `max(1, min(14, cpu_count - 2))` — reserves
  2 logical CPUs for the OS and the main process, caps at 14 to avoid
  diminishing returns from the classifier's internal thread pools.  The same
  formula is used for both stage-1 multi-year workers and stage-2 BoCSoR
  boundary processing.  Override with `--workers N` if needed.

### Classifier

Two classifiers are available via `--classifier`:

**CatBoost** (default) accepts raw string-valued categorical columns with no manual encoding, handles high-cardinality columns (`OCCP`, `POBP`) robustly via ordered target statistics, and achieves state-of-the-art accuracy on tabular categorical data (Alfeo et al., 2023).

**MLP** (Multi-Layer Perceptron, via sklearn `MLPClassifier`) provides a fundamentally different decision boundary geometry compared to tree-based classifiers — smooth non-linear surfaces vs. axis-aligned splits — making it an ideal complement to CatBoost for verifying that BoCSoR results are model-agnostic. MLP requires numeric inputs; categorical features are integer-encoded (ordinal columns use their semantic rank, nominal columns use lexicographic rank) and standardised (zero-mean, unit-variance). The `--cb-depth` parameter maps to two hidden layers of width `2^depth` (e.g. `--cb-depth 6` → `(64, 64)`); `--cb-iterations` maps to `max_iter`.

The choice of classifier does not affect the BoCSoR algorithm itself — only the decision boundary being explored changes. Both classifiers work only on correctly classified instances, consistent with the original BoCSoR approach.

---

## Technical notes

### RELP / RELSHIPP column compatibility

The Census Bureau renamed the household-relationship variable between survey
years. This pipeline calls `normalize_raw_columns()` before `df_to_pandas()`,
which renames `RELSHIPP → RELP` on the raw DataFrame when necessary.

### OCCP range-based classification

`OCCP` codes are contiguous integer blocks aligned with SOC major groups.
The mapping uses `(lower, upper, label)` tuples — more concise and complete
than enumerating every individual code.  Categorisation is vectorised via
`np.searchsorted` + fancy indexing (~10× faster than per-row Python lookups
on 1M+ rows).

### Parallelisation

| Level | Mechanism | Rationale |
|---|---|---|
| Per-state download | `ThreadPoolExecutor` | I/O-bound. |
| Column categorisation | `ThreadPoolExecutor` | Independent per-column transforms; NumPy ops release GIL. |
| CSV write (split) | 3 threads | Dataset, train and test written concurrently. |
| Multi-year stage 1 | `ProcessPoolExecutor` | CPU-bound, bypasses GIL. |
| BoCSoR boundary chunks | `ProcessPoolExecutor` (fork) | CPU-bound, inherits model/BallTree via fork. |
| ARM grid search | `ThreadPoolExecutor` (adaptive) | Only activated when rule volume exceeds 500/support level. |
| Micro ARM per-macro-rule | `ProcessPoolExecutor` (fork) | Each macro rule is independent; inherits exploded tokens via fork. Inner grid workers reduced to avoid oversubscription. |

---

## Dead zone (margin-based sample exclusion)

With categorical features and a continuous income target binarised at a
threshold, individuals whose income falls near the threshold boundary are
inherently ambiguous — two people with identical feature profiles can end up
in different classes simply because one earns $93K and the other $95K.  This
creates cross-class collisions at distance 0 in the BoCSoR encoding space,
reducing the informativeness of the analysis.

The **dead zone** addresses this by excluding individuals whose income
`PINCP` falls within ±margin of the threshold:

- **Class 1**: `PINCP > threshold + margin` (clearly above)
- **Class 0**: `PINCP ≤ threshold − margin` (clearly below)
- **Excluded**: `PINCP ∈ (threshold − margin, threshold + margin]`

### Default margin (auto from ACS MOE)

When `--margin` is omitted, the margin is computed from the **ACS 1-Year
Margin of Error** for median family income (Census Bureau Table B19113),
propagated through the Pew upper-income formula:

```
margin = 2 × MOE_median_family_income / √3
```

For groups of states the MOE decreases with sample size:
`group_MOE = mean(member_MOEs) / √n_members`.

| Scope | Approx. MOE | Approx. margin |
|---|---|---|
| Large state (CA, TX, FL, NY) | $900–1,100 | $1,000–1,300 |
| Medium state (WA, AZ, MN) | $1,800–2,200 | $2,100–2,500 |
| Small state (VT, WY) | $4,500–5,200 | $5,200–6,000 |
| Northeast (9 states) | ~$700 (pooled) | ~$800 |
| National (49 states) | $500 | $600 |

### Disabling the dead zone

Pass `--margin 0` to disable the dead zone entirely.  The pipeline then
behaves identically to a standard threshold binarisation with no exclusions.
Filenames without dead zone omit the `_dz` tag, so runs with and without
dead zone never overwrite each other.

---

## Logging

All pipeline output is logged both to **stdout** and to a **log file**
saved in the output directory:

```
results/<states>/<years>/cols<cols>/thr<N>/pct<N>/<classifier>/pipeline.log
```

The file uses append mode — re-runs add to the existing log rather than
overwriting it.  The timestamp format includes the full date
(`YYYY-MM-DD HH:MM:SS`) for traceability across runs.

---

## Reference

Alfeo, A.L., Zippo, A.G., Catrambone, V., Cimino, M.G.C.A., Toschi, N.,
Valenza, G. (2023). *From local counterfactuals to global feature importance:
efficient, robust, and model-agnostic explanations for brain connectivity
networks.* Computer Methods and Programs in Biomedicine, 236, 107550.