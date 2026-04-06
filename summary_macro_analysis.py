"""
src/summary_macro_analysis.py
─────────────────────────────
Post-analysis: Macroscopic Knowledge Discovery.

Scans all macroscopic association rule CSV files produced by stage 3 and
performs a grid-search stress test over (support, confidence, lift) to
classify surviving rules into two semantic categories:

  Actionable rules — patterns based exclusively on modifiable / merit-based
  features (SCHL, COW, WKHP, OCCP).  These represent decision paths an
  individual can, in principle, change.

  Fairness / bias rules — patterns where at least one actionable feature
  co-occurs with at least one sensitive feature (SEX, RAC1P).  These
  represent structurally entangled decision paths where merit cannot be
  separated from protected attributes.

Rules containing neither actionable nor sensitive features (or only
sensitive features without actionable ones) are discarded as noise.

The grid search answers: "As I raise the statistical bar, do the bias
rules disappear (noise) or persist (structural)?"

Output
──────
A single CSV in <output-dir>/analysis/ with one row per (state, year,
classifier, percentile, k, class, support, confidence, lift) cell,
plus a global summary row at the bottom.

Performance
───────────
  - Feature extraction is vectorised via Series.str.findall (no axis=1).
  - Grid search exploits monotonicity: pre-filter at the lowest thresholds,
    then per-cell masks are cheap numpy boolean ANDs on pre-extracted arrays.
  - Path parsing uses regex on the relative path string.

Usage
─────
    python -m src.summary_macro_analysis [OPTIONS]
    python -m src.summary_macro_analysis --base-dir results --lift 1.5
"""

from __future__ import annotations

import argparse
import itertools
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("src.summary_macro_analysis")

# ─────────────────────────────────────────────────────────────────────────────
# Semantic feature categories
# ─────────────────────────────────────────────────────────────────────────────

ACTIONABLE_FEATURES = frozenset({"SCHL", "COW", "WKHP", "OCCP"})
SENSITIVE_FEATURES  = frozenset({"SEX", "RAC1P"})
ALLOWED_FEATURES    = ACTIONABLE_FEATURES | SENSITIVE_FEATURES

# ─────────────────────────────────────────────────────────────────────────────
# Path parsing
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
    """Extract pipeline metadata from a rule CSV path."""
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
# Vectorised classification
# ─────────────────────────────────────────────────────────────────────────────

def _extract_features(series: pd.Series) -> pd.Series:
    """Extract feature-name frozensets from a column of frozenset-like strings."""
    return series.fillna("").astype(str).str.findall(r"[A-Za-z0-9_]+").apply(frozenset)


def _classify_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_actionable and is_fairness boolean columns."""
    all_feats = _extract_features(df["antecedents"]) | _extract_features(df["consequents"])
    df["all_features"]  = all_feats
    df["is_actionable"] = all_feats.apply(lambda f: len(f) > 0 and f <= ACTIONABLE_FEATURES)
    df["is_fairness"]   = all_feats.apply(
        lambda f: f <= ALLOWED_FEATURES and bool(f & ACTIONABLE_FEATURES) and bool(f & SENSITIVE_FEATURES)
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Grid search (monotonicity-aware)
# ─────────────────────────────────────────────────────────────────────────────

def _run_grid(
    df: pd.DataFrame, meta: dict,
    support_grid: list[float], confidence_grid: list[float], lift_th: float,
) -> list[dict]:
    """Grid search on a pre-classified DataFrame.  Pre-filters at minimums."""
    base = df[
        (df["support"] >= min(support_grid))
        & (df["confidence"] >= min(confidence_grid))
        & (df["lift"] >= lift_th)
    ]
    if base.empty:
        return []
    sup_v  = base["support"].values
    conf_v = base["confidence"].values
    act_v  = base["is_actionable"].values
    fair_v = base["is_fairness"].values

    rows: list[dict] = []
    for s, c in itertools.product(support_grid, confidence_grid):
        mask  = (sup_v >= s) & (conf_v >= c)
        total = int(mask.sum())
        if total == 0:
            continue
        act  = int(act_v[mask].sum())
        fair = int(fair_v[mask].sum())
        if act == 0 and fair == 0:
            continue
        rows.append({
            **meta, "min_support": s, "min_confidence": c, "min_lift": lift_th,
            "total_rules": total, "actionable": act, "fairness": fair,
            "pct_actionable": round(act / total * 100, 2),
            "pct_fairness":   round(fair / total * 100, 2),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_macro_summary(
    base_dir: Path, output_dir: Path,
    min_support: float = 0.05, max_support: float = 0.20, step_support: float = 0.05,
    min_confidence: float = 0.50, max_confidence: float = 1.00, step_confidence: float = 0.10,
    lift: float = 1.25,
) -> Path | None:
    """Scan all macro rule CSVs and produce a summary.  Returns output path or None."""
    sup_grid  = np.arange(min_support, max_support + step_support / 2, step_support).round(4).tolist()
    conf_grid = np.arange(min_confidence, max_confidence + step_confidence / 2, step_confidence).round(4).tolist()

    logger.info("Macro summary — support grid: %s  confidence grid: %s  lift >= %.2f",
                sup_grid, conf_grid, lift)

    all_files = [f for f in base_dir.rglob("arm_*rules.csv") if "/micro/" not in str(f)]
    logger.info("Found %d macroscopic rule files.", len(all_files))

    all_rows: list[dict] = []
    g_total = g_act = g_fair = 0

    for fp in all_files:
        meta = _parse_path(fp, base_dir)
        if meta is None:
            continue
        df = pd.read_csv(fp)
        if df.empty or "antecedents" not in df.columns:
            continue
        df = _classify_rules(df)

        bm = (df["support"] >= min_support) & (df["confidence"] >= min_confidence) & (df["lift"] >= lift)
        g_total += int(bm.sum())
        g_act   += int((bm & df["is_actionable"]).sum())
        g_fair  += int((bm & df["is_fairness"]).sum())

        all_rows.extend(_run_grid(df, meta, sup_grid, conf_grid, lift))

    if not all_rows:
        logger.warning("No results for macroscopic summary.")
        return None

    summary = pd.DataFrame(all_rows).sort_values(
        ["state", "classifier", "percentile", "k", "class", "min_support", "min_confidence"]
    ).reset_index(drop=True)

    pct_a = round(g_act / max(g_total, 1) * 100, 2)
    pct_f = round(g_fair / max(g_total, 1) * 100, 2)
    summary = pd.concat([summary, pd.DataFrame([{
        "state": ">> GLOBAL TOTAL <<", "classifier": "-", "percentile": "-",
        "k": "-", "class": "-", "year": "-", "columns": "-", "threshold": "-",
        "min_support": min_support, "min_confidence": min_confidence, "min_lift": lift,
        "total_rules": g_total, "actionable": g_act, "fairness": g_fair,
        "pct_actionable": pct_a, "pct_fairness": pct_f,
    }])], ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"summary_macro_sup{min_support}_conf{min_confidence}.csv"
    summary.to_csv(out, index=False)
    logger.info("Macro summary → %s  (%d rows)", out, len(summary))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.summary_macro_analysis",
        description="Macroscopic Knowledge Discovery — grid-search stress test.",
    )
    parser.add_argument("--base-dir",   type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    g = parser.add_argument_group("Grid search")
    g.add_argument("--min-supp",  type=float, default=0.05)
    g.add_argument("--max-supp",  type=float, default=0.20)
    g.add_argument("--step-supp", type=float, default=0.05)
    g.add_argument("--min-conf",  type=float, default=0.50)
    g.add_argument("--max-conf",  type=float, default=1.00)
    g.add_argument("--step-conf", type=float, default=0.10)
    g.add_argument("--lift",      type=float, default=1.25)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s", datefmt="%H:%M:%S")
    run_macro_summary(
        base_dir=args.base_dir, output_dir=args.output_dir,
        min_support=args.min_supp, max_support=args.max_supp, step_support=args.step_supp,
        min_confidence=args.min_conf, max_confidence=args.max_conf, step_confidence=args.step_conf,
        lift=args.lift,
    )


if __name__ == "__main__":
    main()
