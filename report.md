# Parameter Rationale — FP-Growth Association Rules

This document explains the reasoning behind the `min`, `max`, and `delta` values
chosen for `support`, `confidence`, and `lift` in `explore_association_rules()`,
and the rationale for the neutral lift window filter.
All choices are calibrated on the actual item-frequency distribution of
`labels_only_unique.csv` (6,108 transactions, 8 unique items).

---

## Item-Frequency Reference

These are the empirical support values of each individual item in the dataset,
which serve as the anchors for every parameter decision below.

| Item  | Frequency | Support |
|-------|-----------|---------|
| SCHL  | 2,915     | 47.72%  |
| WKHP  | 2,684     | 43.94%  |
| COW   | 2,027     | 33.19%  |
| RELP  | 2,026     | 33.17%  |
| RAC1P | 1,783     | 29.19%  |
| AGEP  | 1,762     | 28.85%  |
| MAR   | 1,009     | 16.52%  |
| SEX   |   845     | 13.83%  |

---

## Support

The support of an itemset is the fraction of transactions that contain it.

```psuedocode
sup_min=0.02,  sup_max=0.50,  sup_delta=0.02
```

**`sup_min = 0.02`**
The rarest item is SEX at 13.83%. For a 2-item rule involving the two rarest
items (SEX and MAR), the expected joint support under statistical independence is:

```pseudocode
P(SEX) × P(MAR) = 0.1383 × 0.1652 ≈ 0.023
```

Setting `sup_min = 0.02` places the lower bound just below this threshold,
ensuring that even the rarest meaningful pairwise combinations are reachable.
Going lower would only surface itemsets with fewer than ~120 occurrences out of
6,108, which is statistical noise at this dataset size.

**`sup_max = 0.50`**
The most frequent item is SCHL at 47.72%. A support threshold above 0.50 would
filter out even the single most common item, making frequent-itemset mining
trivially empty. 0.50 is therefore the natural ceiling imposed by the data.

**`sup_delta = 0.02`**
With only 8 unique items, the number of possible itemsets is small and the
support landscape changes meaningfully at each 2-percentage-point step. A coarser
delta (e.g. 0.05) would skip over thresholds where the set of frequent itemsets
changes composition; a finer one (e.g. 0.01) would double the computation without
revealing new structure, since the dataset has ~6,100 rows and a 1% step
corresponds to a difference of roughly 61 transactions.

---

## Confidence

The confidence of a rule `A → B` is the fraction of transactions containing `A`
that also contain `B`, i.e. `P(B | A)`.

```pseudocode
conf_min=0.10,  conf_max=1.00,  conf_delta=0.05
```

**`conf_min = 0.10`**
A rule with confidence below 10% holds in fewer than 1 out of 10 transactions
containing the antecedent. Such rules have little predictive value in practice,
but since the goal is to explore the full parameter space, 0.10 is the lowest
threshold that still corresponds to a detectable conditional relationship.
Setting it to 0.0 would include trivially weak rules that carry no signal.

**`conf_max = 1.00`**
Confidence is bounded at 1.0 by definition. A rule with confidence = 1.0 is
deterministic: every transaction containing the antecedent also contains the
consequent. These are among the most informative rules, so capping at 0.90
(as in the original code) would exclude them entirely.

**`conf_delta = 0.05`**
A 5-percentage-point step gives 19 evenly spaced thresholds across [0.10, 1.00].
This is fine enough to distinguish rules at neighbouring confidence levels
(e.g. 0.60 vs 0.65) without inflating the number of combinations unnecessarily.
The confidence landscape for a small-item dataset like this one does not
justify a finer resolution.

---

## Lift

The lift of a rule `A → B` measures how much more (or less) often `A` and `B`
co-occur compared to what would be expected if they were statistically independent:

```pseudocode
lift(A → B) = P(A ∪ B) / (P(A) × P(B))
```

```pseudocode
lift_min=0.0,  lift_max=7.0,  lift_delta=0.1
```

The three lift regions and their interpretation:

| Lift value  | Meaning                                                                                 |
| ----------- | --------------------------------------------------------------------------------------- |
| `lift < 1`  | Negative correlation — A and B co-occur *less* than by chance (they "avoid" each other) |
| `lift = 1`  | Statistical independence — knowing A tells us nothing about B                           |
| `lift > 1`  | Positive correlation — A and B co-occur *more* than by chance                           |

**`lift_min = 0.0`**
The absolute theoretical minimum of lift is 0, which occurs when the antecedent
and consequent never co-occur in the same transaction. Rules with lift < 1 indicate
negative correlations: pairs of features that tend *not* to appear together in
counterfactual transactions. These are just as informative as positive correlations
and are fully included by setting `lift_min = 0.0`.
Previous values of 1.0 (original code) and 0.1 (first revision) were both
discarding part of the negative-correlation region without justification.
0.0 is the only value that provides truly complete coverage of the lift spectrum.

**`lift_max = 7.0`**
The theoretical maximum lift for a rule involving the rarest item (SEX, support =
0.1383) is:

```pseudoce
lift_max_theoretical = 1 / support(SEX) = 1 / 0.1383 ≈ 7.23
```

Setting `lift_max = 7.0` covers virtually the entire reachable lift range for
this dataset. No rule can realistically exceed this value, so extending the grid
further would only add empty summary rows.

**`lift_delta = 0.1`**
With a range of [0.0, 7.0] (71 steps), a step of 0.1 is fine enough to
distinguish rules with similar but different lift values (e.g. 2.3 vs 2.4),
while keeping the total number of summary rows manageable. A finer delta would
not add interpretive value given that lift estimates themselves carry sampling
variance at this dataset size.

---

## Neutral Lift Window Filter

```pseudocode
lift_neutral_half_window=0.25  →  excluded window: [0.75, 1.25]
```

Rules with lift ≈ 1 indicate that the antecedent and consequent are
nearly statistically independent: knowing one tells us almost nothing
about the other. Such rules are not actionable and inflate the output
without contributing insight.

The window is parameterised as a half-width around 1, so:

```pseudocode
excluded: lift ∈ [1 - half_window, 1 + half_window] = [0.75, 1.25]
retained: lift < 0.75  (meaningful negative correlation)
          lift > 1.25  (meaningful positive correlation)
```

The choice of 0.25 as half-width is a practical threshold: a lift of
0.75 or 1.25 already represents a 25% deviation from independence, which
is detectable and interpretable with 6,108 transactions.
The `lift_neutral_half_window` parameter can be tightened (e.g. 0.1 for
a stricter filter) or widened (e.g. 0.5) depending on how conservative
the analysis needs to be.

After the filter is applied, any `conf_*` folder left with no surviving
rules is automatically deleted by `cleanup_empty_folders()`, along with
its parent `sup_*` folder if all confidence subfolders are removed.

---

## Parameter Update — v2 (based on observed output)

After running the first full exploration, the actual output revealed two important
constraints that justify tightening the parameter ranges.

### Support ceiling: 0.50 → 0.16

From support = 0.18 onward, FP-Growth found only itemsets of length 1. A single-item
itemset cannot produce any association rule of the form `A → B` (both antecedent and
consequent must be non-empty and disjoint), so all `conf_*` folders for those support
values were empty and removed by `cleanup_empty_folders()`. The 17 deleted `sup_*`
folders confirmed that the effective upper bound for rule generation is 0.16.

Setting `sup_max = 0.16` removes dead computation without losing any results.

### Lift ceiling: 7.0 → 2.5 / delta: 0.1 → 0.05

The highest lift observed across all generated rules was **1.97**. The theoretical
maximum of 7.2 (= 1 / support(SEX)) is only reachable if SEX and another item
co-occur in every single transaction that contains SEX — an extreme condition not
present in the data. Keeping `lift_max = 7.0` was filling the summary with hundreds
of rows where `Number_of_Rules = 0`, adding no information.

Setting `lift_max = 2.5` covers the observed maximum with a 27% margin (2.5 vs 1.97)
and accommodates potential variation if the dataset is updated.

With the narrower range [0.0, 2.5] a finer step of `lift_delta = 0.05` (51 values
instead of 71) adds meaningful resolution — distinguishing rules at lift 1.30 vs 1.35,
for example — at virtually no extra computational cost.

### Updated parameter call

```python
summary = explore_association_rules(
    df         = df_encoded,
    output_dir = ar_output_dir,
    sup_min=0.02,  sup_max=0.16,  sup_delta=0.02,
    conf_min=0.10, conf_max=1.00, conf_delta=0.05,
    lift_min=0.0,  lift_max=2.5,  lift_delta=0.05,
    lift_neutral_half_window=0.25,
)
```

---

## Heatmaps

After the exploration and cleanup, `plot_heatmaps()` generates three PNG files
saved under `output_dir/heatmaps/`:

| File                             | Axes                        | Aggregation                          |
| -------------------------------- | --------------------------- | ------------------------------------ |
| `heatmap_support_confidence.png` | x = support, y = confidence | max rules over all lift thresholds   |
| `heatmap_support_lift.png`       | x = support, y = lift       | max rules over all confidence values |
| `heatmap_confidence_lift.png`    | x = confidence, y = lift    | max rules over all support values    |

Each cell shows the **maximum** `Number_of_Rules` achievable for that 2D combination,
regardless of the third parameter. This aggregation choice answers the question
"what is the best this (sup, conf) pair can do?" rather than averaging over all
lift thresholds, which would dilute the signal from the most productive configurations.

The colormap is `YlOrBr` (yellow → orange → brown): lighter cells correspond to
fewer rules (or zero), darker cells to more rules. Cell values are annotated
directly on the heatmap; zeros are omitted to reduce clutter. Text color switches
from black to white automatically when the cell is dark enough to ensure readability.

---

## Symmetric Rule Deduplication

After the neutral-window lift filter, `deduplicate_symmetric_rules()` is applied
before saving each `rules.csv` / `rules_detailed.csv`.

**Why symmetric rules are redundant**
For any pair of items A and B, the lift is mathematically identical in both
directions:

```pseudocode
lift(A → B) = P(A ∪ B) / (P(A) · P(B)) = lift(B → A)
```

Keeping both A→B and B→A doubles the number of rows without adding any
information about the strength or existence of the association. In a dataset
with only 8 items, 39 of the 102 raw rules (38%) were symmetric duplicates.

**Which direction is kept**
The direction with higher confidence is retained, because confidence is
asymmetric and measures the conditional probability `P(B | A)` — it expresses
the actual predictive power of the specific direction. In case of a tie,
the rule with the shorter antecedent is preferred (simpler premise).

**Negative-correlation rules**
Rules with lift < 1 (all involving RELP in this dataset) are subject to the
same criterion and are never filtered out solely because of their lift value.
Both directions of a negative-correlation pair are evaluated and the one with
higher confidence is kept, preserving the full signal about which features tend
not to co-occur in counterfactual transactions.

---

## K-Variation Experiment

### Why k matters

`k_neighbors` controls how many candidates from the opposite class are considered
when searching for counterfactuals. A larger k means more neighbors are examined,
which increases the chances of finding a 1-sparse counterfactual (one that differs
by exactly one feature) for each test sample. This directly affects the
`labels_only_unique.csv` that feeds into the association rule mining:

| k small                                                   | k large                                              |
| --------------------------------------------------------- | ---------------------------------------------------- |
| Fewer transactions (less chance of finding a 1-sparse CF) | More transactions                                    |
| Items appear less frequently → lower support values       | Items appear more frequently → higher support values |
| Sparser co-occurrences → fewer 2-itemsets                 | Denser co-occurrences → more 2-itemsets and rules    |

### How the experiment is structured

`feature_importance.py / run_for_k_values(k_values, ...)` runs the full
counterfactual extraction pipeline for each k in the list, using the same
train/test split for all k values (same `random_state=42`) so comparisons
are fair. Output per k:

```pseudocode
important_features_dir/
├── k_3/
│   ├── transactions_values.csv
│   ├── labels_only.csv
│   └── labels_only_unique.csv
├── k_5/
│   └── ...
└── ...
```

The function returns a dict `{k: path_to_labels_only_unique.csv}` which is
passed directly to `run_k_comparison()` in `association_rules.py`.

`association_rules.py / run_k_comparison(k_labels_map, ...)` runs
`explore_association_rules()` for each k under `output_dir/k_{k}/`, then
aggregates results in `output_dir/k_comparison/`:

```tree
association_rules/
├── k_3/          ← full exploration output for k=3
├── k_5/
├── ...
└── k_comparison/
    ├── k_comparison_summary.csv
    ├── k_comparison_summary.txt
    ├── heatmap_k_support.png      x=support,    y=k
    ├── heatmap_k_confidence.png   x=confidence, y=k
    └── heatmap_k_lift.png         x=lift,       y=k
```

### Why parameters must be recalibrated for each k

The three parameters that depend on item frequencies change with k and must
not be kept fixed across experiments:

`sup_min` — the pairwise floor `P(rarest) × P(second_rarest)` changes because
item frequencies shift when more or fewer transactions are extracted.

`sup_max` — the threshold above which only length-1 itemsets survive depends
on the co-occurrence density, which grows with k.

`lift_max` — the theoretical ceiling `1 / support(rarest_item)` changes
directly with the rarest item's frequency.

`calibrate_parameters(encoded_df)` in `association_rules.py` recomputes all
three automatically from the actual item frequencies of each k's dataset.
`conf_min/max/delta` and the neutral window are never auto-tuned since they
cover the full `[0, 1]` range by definition and are independent of k.

---

## Binning Corrections — create_dataset.py

### AGEP

The `ACSIncome` task in folktables automatically filters to `AGEP >= 16`
(working-age population only). The original lower bin boundary of 0 included
ages 0–15 which are never present in the filtered dataset. Corrected to 15
so the first bin effectively starts at 16:

```python
# before
bins=[0,  29, 44, 59, 150]
# after
bins=[15, 29, 44, 59, 150]
```

### WKHP

The original thresholds did not match standard labor definitions. The BLS
defines part-time as fewer than 35 hours per week, and the FLSA standard
full-time workweek is 40 hours. The previous bins placed 30–34h in
"Full-Time" and called 40–49h "Overtime" despite 40h being the standard:

```python
# before — misaligned with BLS/FLSA definitions
bins=[-1, 29, 39, 49, 150]   # Part-Time 0-29, Full-Time 30-39, Overtime 40-49

# after — aligned with BLS (< 35h = part-time) and FLSA (40h standard)
bins=[-1, 34, 40, 49, 150]   # Part-Time 0-34, Full-Time 35-40, Overtime 41-49
```

| Label      | Before    | After     | Standard          |
|------------|-----------|-----------|-------------------|
| Part-Time  | 0–29 h    | 0–34 h    | BLS: < 35 h       |
| Full-Time  | 30–39 h   | 35–40 h   | FLSA: 40 h/week   |
| Overtime   | 40–49 h   | 41–49 h   | Beyond FLSA std.  |
| Intensive  | 50+ h     | 50+ h     | unchanged         |

---

## Pipeline Orchestration — main.py

`main.py` runs all three steps in sequence by importing functions directly
from the sibling scripts. Each script remains fully executable standalone
via its own `__main__` block — `main.py` only adds the orchestration layer.

```pseudocode
main.py
 ├── STEP 1  create_dataset.py
 │           create_ny_2018_dataset() + categorize_dataset()
 ├── STEP 2  feature_importance.py
 │           run_for_k_values([1,3,5,7,9,11,13,15,17,19])
 │           → k_labels_map {k: path_to_labels_only_unique.csv}
 └── STEP 3  association_rules.py
             run_k_comparison(k_labels_map, auto_calibrate=True)
             → k_{k}/ per-k exploration + k_comparison/ cross-k summary
```

The train/test split in `run_for_k_values` uses `random_state=42` and is
performed once before the k loop, so all k values see the same data.

---

## Empirical Results — K-Variation Experiment (ACSIncome NY 2018)

Run on 20,605 test samples (80/20 split, `random_state=42`). All k values
produced exactly 6,025 transactions — the number of samples with at least
one 1-sparse counterfactual does not change with k (the 1-sparsity constraint
is strict), but the *content* of each transaction grows with k as more
neighbors are examined.

### Calibrated parameters per k

| k  | sup_max | lift_max | lift ceiling (raw) | combos w/ rules |
|----|---------|----------|--------------------|-----------------|
| 1  | —       | —        | —                  | skipped (sparse)|
| 3  | 0.08    | 10.0     | 17.41              | 90              |
| 5  | 0.10    | 10.0     | 10.97              | 208             |
| 7  | 0.14    | 8.5      | 8.16               | 895             |
| 9  | 0.16    | 7.5      | 7.06               | 2,020           |
| 11 | 0.18    | 6.5      | 6.25               | 3,058           |
| 13 | 0.20    | 6.0      | 5.83               | 3,806           |
| 15 | 0.20    | 5.5      | 5.50               | 3,880           |
| 17 | 0.22    | 5.5      | 5.23               | 4,404           |
| 19 | 0.22    | 5.5      | 5.08               | 4,711           |

### Key observations

**k=1 is always skipped.** With only one neighbor candidate per sample, the
probability that it is 1-sparse is low enough that the resulting transactions
contain no co-occurrences above any support threshold. `calibrate_parameters`
detects this (sup_max == sup_min after the scan) and returns `None`.

**sup_max grows monotonically with k.** More neighbors → denser transactions
→ feature pairs co-occur more frequently → 2-itemsets survive at higher
support thresholds. Grows from 0.08 (k=3) to 0.22 (k=17,19).

**lift_max decreases with k.** As k grows, even the rarest item (SEX) appears
more often: support goes from 0.024 (k=1) to 0.197 (k=19), pushing the
theoretical lift ceiling from 41 down to ~5. From k=13 onward lift_max
stabilises at 5.5–6.0.

**Rule count grows rapidly up to k≈13, then flattens.** The jump from k=9
(2,020) to k=11 (3,058) to k=13 (3,806) is large; k=15→17→19 show
diminishing returns (+74, +524, +307). This suggests k=13–15 is a reasonable
sweet spot for this dataset if runtime is a concern.

**Item support ordering is stable from k=7 onward:**
SEX < MAR < AGEP ≈ RAC1P < RELP ≈ COW < WKHP < SCHL.
This means the relative importance of features in the counterfactual
transactions is consistent across higher k values.
