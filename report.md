# Report — FP-Growth Association Rules

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

```pseudocode
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

```pseudocode
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
