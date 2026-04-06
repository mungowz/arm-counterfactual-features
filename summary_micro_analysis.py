"""
src/summary_micro_analysis.py
──────────────────────────────
Post-analysis: Microscopic Knowledge Discovery.

Scans all microscopic association rule CSV files produced by stage 4 and
quantifies the *geometric fragmentation* of value-level rules anchored
to each macroscopic "mother" rule.

Two operating modes
───────────────────
  Without macro filter (default):
    All micro rules above the grid thresholds are analysed.

  With macro filter (--use-macro-filter):
    Only micro rules whose macroscopic mother passes a separate strength
    test (macro support, confidence, lift) are analysed.  This implements
    hierarchical support pruning: weak mothers are discarded and all their
    micro-daughters are skipped in one shot.

For each surviving (state, model, percentile, k, class, macro_rule,
threshold_cell) the script counts how many distinct micro rules exist.
This answers: "How many different value-level paths does the model use
to implement this particular bias?"

Comparing CatBoost vs MLP on the same macro rule reveals architectural
differences: tree models produce few axis-aligned micro rules, while
neural networks fragment the space into many smooth micro rules.

Semantic classification
───────────────────────
Micro rules inherit their macro mother's semantic category:
  - Actionable: mother features ⊆ {SCHL, COW, WKHP, OCCP}
  - Fairness:   mother features ⊆ {SCHL, COW, WKHP, OCCP, SEX, RAC1P}
                 with ≥1 actionable AND ≥1 sensitive feature

Rules outside these categories are ignored.

Key fix over the original script
─────────────────────────────────
The micro rules CSV already contains 'macro_antecedents' and
'macro_consequents' columns written by stage 4.  This script uses them
directly instead of reverse-engineering the macro rule from micro-level
antecedents — which was fragile and incorrect when micro rules contained
extra features beyond the anchor macro rule.

The macro filter now correctly locates the macroscopic rules CSV in the
parent k-directory (association_rules/k<N>/arm_*_rules.csv), not inside
the micro/ subdirectory where the original script searched.

Performance
───────────
  - Macro feature extraction uses the existing macro_antecedents /
    macro_consequents columns (no re-parsing of micro tokens).
  - Grid search pre-filters at minimum thresholds and applies per-cell
    masks on pre-extracted numpy arrays.
  - Path parsing uses the same regex as the macro summary script.

Usage
─────
    python -m src.summary_micro_analysis [OPTIONS]
    python -m src.summary_micro_analysis --use-macro-filter --macro-min-supp 0.10
"""

from __future__ import annotations

import argparse
import itertools
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("src.summary_micro_analysis")

# ─────────────────────────────────────────────────────────────────────────────
# Semantic feature categories (same as macro script)
# ─────────────────────────────────────────────────────────────────────────────

ACTIONABLE_FEATURES = frozenset({"SCHL", "COW", "WKHP", "OCCP"})
SENSITIVE_FEATURES  = frozenset({"SEX", "RAC1P"})
ALLOWED_FEATURES    = ACTIONABLE_FEATURES | SENSITIVE_FEATURES

# ─────────────────────────────────────────────────────────────────────────────
# Path parsing (shared regex with macro script)
# ─────────────────────────────────────────────────────────────────────────────

_PATH_RE = re.compile(
    r"(?P<states>[^/]+)"
    r"/(?P<year>[^/]+)"
    r"/cols(?P<cols>[^/]+)"
    r"/thr(?P<thr>\d+)"
    r"/pct(?P<pct>\d+)"
    r"/(?P<classifier>[^/]+)"
    r"(?:/(?P<extra_year>\d{4}))?"
    r"/association_rules"
    r"/(?P<k_folder>k\d+|all_k)"
)


def _parse_path(file_path: Path, base_dir: Path) -> dict | None:
    rel = str(file_path.relative_to(base_dir))
    m = _PATH_RE.search(rel)
    if not m:
        return None
    k_raw = m.group("k_folder")
    name  = file_path.name
    return {
        "state":      m.group("states"),
        "year":       m.group("extra_year") or m.group("year"),
        "columns":    m.group("cols"),
        "threshold":  int(m.group("thr")),
        "percentile": int(m.group("pct")),
        "classifier": m.group("classifier"),
        "k":          "all" if k_raw == "all_k" else k_raw.replace("k", ""),
        "class":      "class1" if "class1" in name else ("class0" if "class0" in name else "both"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction from macro columns
# ─────────────────────────────────────────────────────────────────────────────

def _parse_features(s: str) -> frozenset:
    """Extract feature names from a string like 'SCHL & WKHP' or 'SCHL'."""
    if pd.isna(s):
        return frozenset()
    return frozenset(tok.strip() for tok in str(s).split("&") if tok.strip())


def _set_label(fs: frozenset) -> str:
    """Deterministic string label for a feature set: '{COW, SCHL}'."""
    return "{" + ", ".join(sorted(fs)) + "}"


# ─────────────────────────────────────────────────────────────────────────────
# Macro filter: locate and load macro rules for a micro file
# ─────────────────────────────────────────────────────────────────────────────

def _find_macro_rules_for_micro(micro_path: Path) -> Path | None:
    """
    Given a micro rules file path like
        .../association_rules/k5/micro/micro_class0_rules.csv
    return the corresponding macro rules file:
        .../association_rules/k5/arm_class0_rules.csv

    Falls back to the all_k macro file if the per-k file doesn't exist.
    """
    # micro_path.parent = .../micro/
    # micro_path.parent.parent = .../k5/ or .../all_k/
    k_dir = micro_path.parent.parent
    suffix = "_class1" if "class1" in micro_path.name else "_class0" if "class0" in micro_path.name else ""

    # Try per-k macro rules first.
    candidate = k_dir / f"arm{suffix}_rules.csv"
    if candidate.exists():
        return candidate

    # Fall back to all_k.
    all_k_dir = k_dir.parent / "all_k"
    candidate = all_k_dir / f"arm{suffix}_all_k_rules.csv"
    if candidate.exists():
        return candidate

    return None


def _load_valid_macro_rules(
    macro_path: Path,
    min_support: float,
    min_confidence: float,
    min_lift: float,
) -> set[str]:
    """
    Load macro rules passing the strength filter and return a set of
    canonical rule strings like '{COW, SCHL} -> {WKHP}'.
    """
    df = pd.read_csv(macro_path)
    if df.empty:
        return set()
    mask = (
        (df["support"]    >= min_support)
        & (df["confidence"] >= min_confidence)
        & (df["lift"]       >= min_lift)
    )
    df = df[mask]
    if df.empty:
        return set()
    ant_labels  = df["antecedents"].apply(lambda s: _set_label(_parse_features(s)))
    cons_labels = df["consequents"].apply(lambda s: _set_label(_parse_features(s)))
    return set(ant_labels + " -> " + cons_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis
# ─────────────────────────────────────────────────────────────────────────────

def _analyse_micro_file(
    micro_path: Path,
    meta: dict,
    support_grid: list[float],
    confidence_grid: list[float],
    lift_th: float,
    use_macro_filter: bool,
    valid_macro_rules: set[str] | None,
) -> list[dict]:
    """Analyse one micro rules CSV and return result rows."""
    df = pd.read_csv(micro_path)
    if df.empty:
        return []

    # ── Build the macro rule label from existing columns ──────────────────
    # Stage 4 already writes macro_antecedents / macro_consequents.
    if "macro_antecedents" in df.columns and "macro_consequents" in df.columns:
        ant_feats  = df["macro_antecedents"].apply(_parse_features)
        cons_feats = df["macro_consequents"].apply(_parse_features)
    elif "antecedents" in df.columns and "consequents" in df.columns:
        # Fallback: extract macro-level labels from micro antecedents.
        ant_feats  = df["antecedents"].apply(
            lambda s: frozenset(tok.split("=")[0].strip() for tok in str(s).split("&") if "=" in tok) if pd.notna(s) else frozenset()
        )
        cons_feats = df["consequents"].apply(
            lambda s: frozenset(tok.split("=")[0].strip() for tok in str(s).split("&") if "=" in tok) if pd.notna(s) else frozenset()
        )
    else:
        logger.warning("Skipping %s: no antecedent/consequent columns.", micro_path.name)
        return []

    df["_macro_rule"] = ant_feats.apply(_set_label) + " -> " + cons_feats.apply(_set_label)
    df["_all_feats"]  = ant_feats | cons_feats

    # ── Macro filter ─────────────────────────────────────────────────────
    if use_macro_filter and valid_macro_rules is not None:
        df = df[df["_macro_rule"].isin(valid_macro_rules)]
        if df.empty:
            return []

    # ── Semantic classification ──────────────────────────────────────────
    df["_is_actionable"] = df["_all_feats"].apply(
        lambda f: len(f) > 0 and f <= ACTIONABLE_FEATURES
    )
    df["_is_fairness"] = df["_all_feats"].apply(
        lambda f: f <= ALLOWED_FEATURES and bool(f & ACTIONABLE_FEATURES) and bool(f & SENSITIVE_FEATURES)
    )
    # Keep only actionable or fairness rules.
    df = df[df["_is_actionable"] | df["_is_fairness"]]
    if df.empty:
        return []

    # ── Grid search ──────────────────────────────────────────────────────
    min_s = min(support_grid)
    min_c = min(confidence_grid)
    base = df[
        (df["support"] >= min_s) & (df["confidence"] >= min_c) & (df["lift"] >= lift_th)
    ]
    if base.empty:
        return []

    sup_v   = base["support"].values
    conf_v  = base["confidence"].values
    rules   = base["_macro_rule"].values
    is_act  = base["_is_actionable"].values

    rows: list[dict] = []
    for s, c in itertools.product(support_grid, confidence_grid):
        mask = (sup_v >= s) & (conf_v >= c)
        if not mask.any():
            continue
        # Count micro rules per macro mother.
        sub_rules = rules[mask]
        sub_act   = is_act[mask]
        unique_mothers, counts = np.unique(sub_rules, return_counts=True)
        for mother, count in zip(unique_mothers, counts):
            # Determine type from first occurrence.
            mother_mask = sub_rules == mother
            rule_type = "actionable" if sub_act[mother_mask].any() else "fairness"
            rows.append({
                **meta,
                "macro_rule":      mother,
                "rule_type":       rule_type,
                "min_support":     s,
                "min_confidence":  c,
                "min_lift":        lift_th,
                "micro_rule_count": int(count),
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_micro_summary(
    base_dir: Path, output_dir: Path,
    min_support: float = 0.01, max_support: float = 0.05, step_support: float = 0.01,
    min_confidence: float = 0.30, max_confidence: float = 0.80, step_confidence: float = 0.10,
    lift: float = 1.25,
    use_macro_filter: bool = False,
    macro_min_support: float = 0.05,
    macro_min_confidence: float = 0.50,
    macro_min_lift: float = 1.25,
) -> Path | None:
    """Scan all micro rule CSVs and produce a summary.  Returns output path or None."""
    sup_grid  = np.arange(min_support, max_support + step_support / 2, step_support).round(4).tolist()
    conf_grid = np.arange(min_confidence, max_confidence + step_confidence / 2, step_confidence).round(4).tolist()

    logger.info("Micro summary — support grid: %s  confidence grid: %s  lift >= %.2f",
                sup_grid, conf_grid, lift)
    if use_macro_filter:
        logger.info("Macro filter ACTIVE (sup >= %.2f, conf >= %.2f, lift >= %.2f)",
                     macro_min_support, macro_min_confidence, macro_min_lift)
    else:
        logger.info("Macro filter DISABLED (all micro rules analysed).")

    all_files = list(base_dir.rglob("micro_*rules.csv"))
    logger.info("Found %d microscopic rule files.", len(all_files))

    all_rows: list[dict] = []

    for fp in all_files:
        meta = _parse_path(fp, base_dir)
        if meta is None:
            continue

        # ── Macro filter: locate and load strong macro rules ─────────────
        valid_macro: set[str] | None = None
        if use_macro_filter:
            macro_path = _find_macro_rules_for_micro(fp)
            if macro_path is None:
                logger.debug("No macro rules found for %s — skipping.", fp.name)
                continue
            valid_macro = _load_valid_macro_rules(
                macro_path, macro_min_support, macro_min_confidence, macro_min_lift,
            )
            if not valid_macro:
                logger.debug("No strong macro rules for %s — skipping.", fp.name)
                continue

        rows = _analyse_micro_file(
            fp, meta, sup_grid, conf_grid, lift,
            use_macro_filter, valid_macro,
        )
        all_rows.extend(rows)

    if not all_rows:
        logger.warning("No results for microscopic summary.")
        return None

    summary = pd.DataFrame(all_rows).sort_values(
        ["state", "classifier", "percentile", "k", "class", "macro_rule",
         "min_support", "min_confidence"],
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"summary_micro_sup{min_support}_conf{min_confidence}"
    if use_macro_filter:
        tag += "_filtered"
    out = output_dir / f"{tag}.csv"
    summary.to_csv(out, index=False)
    logger.info("Micro summary → %s  (%d rows)", out, len(summary))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.summary_micro_analysis",
        description="Microscopic Knowledge Discovery — fragmentation analysis "
                    "of value-level rules anchored to macroscopic mothers.",
    )
    parser.add_argument("--base-dir",   type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))

    g = parser.add_argument_group("Micro grid search")
    g.add_argument("--min-supp",  type=float, default=0.01)
    g.add_argument("--max-supp",  type=float, default=0.05)
    g.add_argument("--step-supp", type=float, default=0.01)
    g.add_argument("--min-conf",  type=float, default=0.30)
    g.add_argument("--max-conf",  type=float, default=0.80)
    g.add_argument("--step-conf", type=float, default=0.10)
    g.add_argument("--lift",      type=float, default=1.25)

    mf = parser.add_argument_group("Macro filter (hierarchical pruning)")
    mf.add_argument("--use-macro-filter", action="store_true",
                    help="Only analyse micro rules whose macro mother passes the strength test.")
    mf.add_argument("--macro-min-supp", type=float, default=0.05)
    mf.add_argument("--macro-min-conf", type=float, default=0.50)
    mf.add_argument("--macro-lift",     type=float, default=1.25)

    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s", datefmt="%H:%M:%S")
    run_micro_summary(
        base_dir=args.base_dir, output_dir=args.output_dir,
        min_support=args.min_supp, max_support=args.max_supp, step_support=args.step_supp,
        min_confidence=args.min_conf, max_confidence=args.max_conf, step_confidence=args.step_conf,
        lift=args.lift,
        use_macro_filter=args.use_macro_filter,
        macro_min_support=args.macro_min_supp,
        macro_min_confidence=args.macro_min_conf,
        macro_min_lift=args.macro_lift,
    )


if __name__ == "__main__":
    main()
