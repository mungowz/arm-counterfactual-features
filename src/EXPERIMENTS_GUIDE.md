# Association Rules Mining - Experiments Guide

## Available Files

### 1. `association_rules.py` (default version)

Single script to extract association rules with fixed parameters.

**Usage:**

```bash
python src/association_rules.py
```

**Output:**

- `results/association_rules.csv` - Formatted rules
- `results/association_rules_detailed.csv` - Rules with all metrics
- `results/association_rules_summary.txt` - Summary

---

### 2. `experiment_association_rules.py` (new - for experiments)

Script to run multiple experiments with different configurations.

**Usage:**

```bash
python src/experiment_association_rules.py
```

**Output:**

```pseudocode
results/experiments/
├── Conservative_high_thresholds/
│   ├── rules.csv
│   ├── rules_detailed.csv
│   └── summary.txt
├── Moderate_balanced/
│   ├── rules.csv
│   ├── rules_detailed.csv
│   └── summary.txt
├── ... (other experiments)
├── experiments_comparison.csv       # Comparison of all experiments
└── experiments_summary.txt          # Statistical summary
```

---

## How to Customize Experiments

Modify the `src/experiment_configs.json` file:

```json
{
  "experiments": [
    {
      "name": "Experiment Name",
      "min_support": 0.05,
      "min_confidence": 0.50,
      "min_lift": 1.0
    }
  ]
}
```

---

## Parameter Explanation

### **Support**

- Frequency of itemset in dataset
- **Range:** 0 to 1 (0% to 100%)
- **Low (0.01-0.05):** Finds rare itemsets
- **High (0.15+):** Finds frequent itemsets

### **Confidence**

- Conditional probability P(B|A) for rule A→B
- **Range:** 0 to 1
- **Low (0.30):** Weak but numerous rules
- **High (0.80+):** Strong but fewer rules

### **Lift**

- Association measure between A and B
- **Lift = 1:** A and B independent
- **Lift > 1:** Positive association (B more likely with A)
- **Lift < 1:** Negative association (B less likely with A)
- **Recommended range:** 0.8 to 2.0

---

## Default Configurations

| Name                           | Support | Confidence | Lift | Purpose                    |
| ------------------------------ | ------- | ---------- | ---- | -------------------------- |
| Conservative                   | 0.10    | 0.70       | 1.2  | Very reliable rules        |
| Moderate                       | 0.05    | 0.50       | 1.0  | Good balance               |
| Liberal                        | 0.02    | 0.30       | 0.8  | Maximum discovery          |
| High support focus             | 0.15    | 0.40       | 1.0  | Very frequent patterns     |
| High confidence focus          | 0.05    | 0.80       | 1.0  | Very accurate rules        |
| High lift focus                | 0.05    | 0.40       | 1.5  | Strong associations        |
| Very low support               | 0.01    | 0.30       | 0.9  | Rare patterns              |

---

## Exploration Tips

1. **Start with moderate** to understand the data
2. **Increase support** if you find too many rules
3. **Increase confidence** if you seek accurate rules
4. **Increase lift** if you seek strong and non-trivial associations
5. **Decrease support** if you seek interesting rare patterns

---

## Additional Metrics in Output

- **Leverage:** P(A,B) - P(A)×P(B), ranges from -1 to 1
- **Conviction:** Measures the directional implication of A→B

---

## FP-Growth Compatibility

FP-Growth is more efficient than Apriori for:

- Large datasets
- Low support values
- High number of items

Feel free to run experiments with very low parameters (support ≤ 0.01)!
