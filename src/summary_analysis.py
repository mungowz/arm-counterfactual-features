"""
src/summary_analysis.py

Post-analysis script that scans all per-k association rule CSVs from
stages 3-4, classifies rules as actionable or biased, and produces
both individual rule listings and grid-sweep summaries.

Classification:
  - Actionable: all features in {SCHL, COW, WKHP, OCCP}
  - Biased: at least one feature in {SEX, RAC1P}, any others allowed

The grid is auto-detected from each CSV's metadata columns
(grid_min_support, grid_min_confidence, filter_lift_kept_above).
Optional floor filters prune the grid to stronger cells.

Output (in <output-dir>/):
  macro_rules.csv    - individual macro rules surviving floor filters
  micro_rules.csv    - individual micro rules surviving floor filters
  summary_macro.csv  - grid-sweep counts per experiment config
  summary_micro.csv  - grid-sweep micro counts per config + macro rule
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("src.summary_analysis")

ACTIONABLE_FEATURES = frozenset({"SCHL", "COW", "WKHP", "OCCP"})
SENSITIVE_FEATURES = frozenset({"SEX", "RAC1P"})

_DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 4) - 2))

# Only read the columns we actually need from the CSVs.
_MACRO_USECOLS = {
    "antecedents", "consequents",
    "support", "confidence", "lift",
    "leverage", "conviction", "lift_type",
    "grid_min_support", "grid_min_confidence", "filter_lift_kept_above",
    "filter_min_support", "filter_max_support",
    "filter_min_confidence", "filter_max_confidence",
}
_MICRO_USECOLS = {
    "macro_antecedents", "macro_consequents",
    "antecedents", "consequents",
    "support", "confidence", "lift",
    "leverage", "conviction", "lift_type",
    "grid_min_support", "grid_min_confidence", "filter_lift_kept_above",
    "filter_min_support", "filter_max_support",
    "filter_min_confidence", "filter_max_confidence",
}

# Columns kept in the individual rule output files (no grid metadata).
_MACRO_OUT_COLS = [
    "antecedents", "consequents",
    "support", "confidence", "lift",
    "leverage", "conviction", "lift_type",
]
_MICRO_OUT_COLS = [
    "macro_antecedents", "macro_consequents",
    "antecedents", "consequents",
    "support", "confidence", "lift",
    "leverage", "conviction", "lift_type",
]

_CONFIG_COLS = [
    "state", "year", "columns", "threshold",
    "percentile", "classifier", "k", "class",
]


def _safe_read_csv(path: Path, usecols: set[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(path, usecols=lambda c: c in usecols)
    except (pd.errors.EmptyDataError, FileNotFoundError, ValueError):
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Path parsing
# ---------------------------------------------------------------------------

_PATH_RE = re.compile(
    r"(?P<states>[^/]+)"
    r"/(?P<year>[^/]+)"
    r"/cols(?P<cols>[^/]+)"
    r"/thr(?P<thr>\d+)"
    r"/pct(?P<pct>\d+)"
    r"/(?P<classifier>[^/]+)"
    r"(?:/(?P<extra_year>\d{4}))?"
    r"/association_rules"
    r"/(?P<k_folder>k\d+)"
)


def _parse_path(file_path: Path, base_dir: Path) -> dict | None:
    """Extract experiment metadata from the file's directory structure."""
    try:
        rel = str(file_path.relative_to(base_dir))
    except ValueError:
        return None
    m = _PATH_RE.search(rel)
    if not m:
        return None
    name = file_path.name
    return {
        "state": m.group("states"),
        "year": m.group("extra_year") or m.group("year"),
        "columns": m.group("cols"),
        "threshold": int(m.group("thr")),
        "percentile": int(m.group("pct")),
        "classifier": m.group("classifier"),
        "k": m.group("k_folder").replace("k", ""),
        "class": ("class1" if "class1" in name
                  else "class0" if "class0" in name
                  else "both"),
    }


# ---------------------------------------------------------------------------
# Feature extraction and classification
# ---------------------------------------------------------------------------

def _parse_features(s: str) -> frozenset[str]:
    """'SCHL & WKHP' -> frozenset({'SCHL', 'WKHP'}).
    Also handles 'SCHL=Bachelors & WKHP=Full-Time' (strips values)."""
    if pd.isna(s):
        return frozenset()
    tokens = [tok.strip() for tok in str(s).split("&") if tok.strip()]
    return frozenset(tok.split("=")[0].strip() for tok in tokens)


def _set_label(fs: frozenset[str]) -> str:
    """Canonical label for a feature set, e.g. '{COW, SCHL}'."""
    return "{" + ", ".join(sorted(fs)) + "}"


def _classify_bulk(ant_series, con_series):
    """Classify all rules in one pass.

    Returns (types, labels) as object arrays.
    types[i] is 'actionable', 'biased', or 'other'.
    If a rule has any sensitive feature it's biased, even if it also
    has actionable ones. Actionable requires *only* actionable features.
    """
    n = len(ant_series)
    types = np.empty(n, dtype=object)
    labels = np.empty(n, dtype=object)
    ant_v, con_v = ant_series.values, con_series.values

    for i in range(n):
        a = _parse_features(ant_v[i])
        c = _parse_features(con_v[i])
        f = a | c
        labels[i] = _set_label(a) + " -> " + _set_label(c)

        if not f:
            types[i] = "other"
        elif f & SENSITIVE_FEATURES:
            types[i] = "biased"
        elif f <= ACTIONABLE_FEATURES:
            types[i] = "actionable"
        else:
            types[i] = "other"

    return types, labels


def _build_macro_labels(ant_series, con_series):
    """Like _classify_bulk but only builds label strings (no classification).
    Used for micro mother-matching where we just need the key, not the type."""
    n = len(ant_series)
    labels = np.empty(n, dtype=object)
    ant_v, con_v = ant_series.values, con_series.values
    for i in range(n):
        a = _parse_features(ant_v[i])
        c = _parse_features(con_v[i])
        labels[i] = _set_label(a) + " -> " + _set_label(c)
    return labels


# ---------------------------------------------------------------------------
# Grid extraction from CSV metadata
# ---------------------------------------------------------------------------

def _extract_grid(df, floor_sup, floor_conf, lift_override):
    """Rebuild the experiment's full grid from CSV metadata.

    The rules CSV after dedup only has grid_min_support values where rules
    were *first found* (typically the lowest threshold). So unique values
    don't cover the full grid. Instead we reconstruct it from:
      - filter_min_support / filter_max_support (grid bounds)
      - grid_min_support unique values (to infer the step)

    Returns (sup_grid, conf_grid, lift) or None if columns are missing.
    """
    needed = {"grid_min_support", "grid_min_confidence", "filter_lift_kept_above"}
    if not needed <= set(df.columns):
        return None

    lift = float(df["filter_lift_kept_above"].dropna().iloc[0])
    if lift_override is not None:
        lift = lift_override

    # Rebuild support grid from bounds + inferred step.
    sup_grid = _rebuild_axis(
        df, "grid_min_support", "filter_min_support", "filter_max_support")
    conf_grid = _rebuild_axis(
        df, "grid_min_confidence", "filter_min_confidence", "filter_max_confidence")

    if sup_grid is None or conf_grid is None:
        return None

    # Apply floor filters.
    if floor_sup is not None:
        sup_grid = [s for s in sup_grid if s >= floor_sup]
    if floor_conf is not None:
        conf_grid = [c for c in conf_grid if c >= floor_conf]

    if not sup_grid or not conf_grid:
        return None
    return sup_grid, conf_grid, lift


def _rebuild_axis(df, grid_col, min_col, max_col):
    """Reconstruct a full grid axis from bounds and inferred step.

    Uses the unique values in grid_col to figure out the step size,
    then generates the full range from min_col to max_col.
    Falls back to unique values if bounds columns are missing.
    """
    unique_vals = sorted(df[grid_col].dropna().unique().tolist())
    if not unique_vals:
        return None

    # If we have bound columns, reconstruct the full range.
    if min_col in df.columns and max_col in df.columns:
        lo = float(df[min_col].dropna().iloc[0])
        hi = float(df[max_col].dropna().iloc[0])

        # Infer step from the unique values (minimum gap between consecutive values).
        if len(unique_vals) >= 2:
            diffs = [round(unique_vals[i+1] - unique_vals[i], 6)
                     for i in range(len(unique_vals) - 1)]
            step = min(d for d in diffs if d > 0)
        else:
            # Single value -- can't infer step, use a reasonable default.
            step = 0.05

        full_grid = np.arange(lo, hi + step / 2, step).round(6).tolist()
        return full_grid

    # No bound columns available, fall back to unique values as-is.
    return unique_vals


# ---------------------------------------------------------------------------
# Grid sweep (for summary files)
# ---------------------------------------------------------------------------

def _grid_sweep_macro(sup_v, conf_v, type_v, meta, sup_grid, conf_grid, lift_th):
    """Count rules at each (support, confidence) cell.
    total includes all rule types (actionable + biased + other)."""
    sup_masks = {s: sup_v >= s for s in sup_grid}
    conf_masks = {c: conf_v >= c for c in conf_grid}
    is_act = type_v == "actionable"
    is_biased = type_v == "biased"

    rows = []
    for s in sup_grid:
        sm = sup_masks[s]
        for c in conf_grid:
            mask = sm & conf_masks[c]
            total = int(mask.sum())
            if total == 0:
                continue
            act = int(is_act[mask].sum())
            biased = int(is_biased[mask].sum())
            if act == 0 and biased == 0:
                continue
            rows.append({
                **meta,
                "min_support": s, "min_confidence": c, "min_lift": lift_th,
                "total_rules": total,
                "actionable": act, "biased": biased,
                "pct_actionable": round(act / total * 100, 2),
                "pct_biased": round(biased / total * 100, 2),
            })
    return rows


def _grid_sweep_micro(sup_v, conf_v, label_v, label_to_type, meta,
                       sup_grid, conf_grid, lift_th):
    """Count micro rules per macro mother at each grid cell."""
    sup_masks = {s: sup_v >= s for s in sup_grid}
    conf_masks = {c: conf_v >= c for c in conf_grid}

    rows = []
    for s in sup_grid:
        sm = sup_masks[s]
        for c in conf_grid:
            mask = sm & conf_masks[c]
            if not mask.any():
                continue
            mothers, counts = np.unique(label_v[mask], return_counts=True)
            for mother, count in zip(mothers, counts):
                rows.append({
                    **meta,
                    "macro_rule": str(mother),
                    "rule_type": label_to_type.get(str(mother), "unknown"),
                    "min_support": s, "min_confidence": c, "min_lift": lift_th,
                    "micro_rule_count": int(count),
                })
    return rows


# ---------------------------------------------------------------------------
# Floor filter (for individual rule files)
# ---------------------------------------------------------------------------

def _apply_floors(df, min_sup, min_conf, min_lift):
    """Keep only rules with actual metrics >= the given floors.
    None means no filtering on that metric."""
    mask = np.ones(len(df), dtype=bool)
    if min_sup is not None:
        mask &= df["support"].values >= min_sup
    if min_conf is not None:
        mask &= df["confidence"].values >= min_conf
    if min_lift is not None:
        mask &= df["lift"].values >= min_lift
    return df[mask]


def _find_micro_file(macro_path):
    """Given .../k5/arm_class0_rules.csv, find .../k5/micro/micro_class0_rules.csv"""
    k_dir = macro_path.parent
    suffix = ("_class1" if "class1" in macro_path.name
              else "_class0" if "class0" in macro_path.name
              else "")
    candidate = k_dir / "micro" / f"micro{suffix}_rules.csv"
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# Per-file worker (thread-safe, no shared state)
# ---------------------------------------------------------------------------

def _process_one_file(macro_path, base_dir,
                       macro_floor_sup, macro_floor_conf, macro_floor_lift,
                       micro_floor_sup, micro_floor_conf, micro_floor_lift):
    """Process one macro CSV: classify, grid-sweep for summaries,
    floor-filter for individual rules, then chain into micro.

    Returns (macro_rules_df, micro_rules_df, macro_summary_rows, micro_summary_rows).
    """
    empty = pd.DataFrame(), pd.DataFrame(), [], []

    meta = _parse_path(macro_path, base_dir)
    if meta is None:
        return empty

    df = _safe_read_csv(macro_path, _MACRO_USECOLS)
    if df.empty or "antecedents" not in df.columns:
        return empty

    # Classify everything (before filtering -- we need all rules for totals).
    types, labels = _classify_bulk(df["antecedents"], df["consequents"])

    # --- Summary: grid sweep on the full rule set ---
    macro_summary_rows = []
    grid = _extract_grid(df, macro_floor_sup, macro_floor_conf, macro_floor_lift)
    if grid is not None:
        m_sup, m_conf, m_lift = grid
        base_mask = (
            (df["support"].values >= min(m_sup))
            & (df["confidence"].values >= min(m_conf))
            & (df["lift"].values >= m_lift)
        )
        if base_mask.any():
            macro_summary_rows = _grid_sweep_macro(
                df["support"].values[base_mask],
                df["confidence"].values[base_mask],
                types[base_mask],
                meta, m_sup, m_conf, m_lift,
            )

    # --- Individual rules: apply floor filters ---
    df_filt = _apply_floors(df, macro_floor_sup, macro_floor_conf, macro_floor_lift)
    if df_filt.empty:
        return pd.DataFrame(), pd.DataFrame(), macro_summary_rows, []

    types_f, labels_f = _classify_bulk(df_filt["antecedents"], df_filt["consequents"])

    out_cols = [c for c in _MACRO_OUT_COLS if c in df_filt.columns]
    macro_out = df_filt[out_cols].copy()
    macro_out.insert(0, "rule_type", types_f)
    for key in reversed(list(meta)):
        macro_out.insert(0, key, meta[key])

    # --- Micro chain: only actionable/biased mothers ---
    micro_out = pd.DataFrame()
    micro_summary_rows = []

    micro_path = _find_micro_file(macro_path)
    if micro_path is None:
        return macro_out, micro_out, macro_summary_rows, micro_summary_rows

    classified = (types_f == "actionable") | (types_f == "biased")
    if not classified.any():
        return macro_out, micro_out, macro_summary_rows, micro_summary_rows

    surviving_labels = set(labels_f[classified])
    label_to_type = {
        str(l): str(t) for l, t in zip(labels_f[classified], types_f[classified])
    }

    mdf = _safe_read_csv(micro_path, _MICRO_USECOLS)
    if mdf.empty or "macro_antecedents" not in mdf.columns:
        return macro_out, micro_out, macro_summary_rows, micro_summary_rows

    # Match micro rules to surviving macro mothers.
    m_labels = _build_macro_labels(mdf["macro_antecedents"], mdf["macro_consequents"])
    mother_ok = np.array([l in surviving_labels for l in m_labels], dtype=bool)
    if not mother_ok.any():
        return macro_out, micro_out, macro_summary_rows, micro_summary_rows

    mdf_mothers = mdf[mother_ok].reset_index(drop=True)
    m_labels_ok = m_labels[mother_ok]

    # Micro summary: grid sweep.
    mi_grid = _extract_grid(mdf_mothers, micro_floor_sup, micro_floor_conf, micro_floor_lift)
    if mi_grid is not None:
        mi_sup, mi_conf, mi_lift = mi_grid
        mi_base = (
            (mdf_mothers["support"].values >= min(mi_sup))
            & (mdf_mothers["confidence"].values >= min(mi_conf))
            & (mdf_mothers["lift"].values >= mi_lift)
        )
        if mi_base.any():
            micro_summary_rows = _grid_sweep_micro(
                mdf_mothers["support"].values[mi_base],
                mdf_mothers["confidence"].values[mi_base],
                m_labels_ok[mi_base],
                label_to_type, meta, mi_sup, mi_conf, mi_lift,
            )

    # Micro individual rules: floor-filtered.
    mdf_filt = _apply_floors(mdf_mothers, micro_floor_sup, micro_floor_conf, micro_floor_lift)
    if not mdf_filt.empty:
        m_labels_filt = m_labels_ok[mdf_filt.index.values]
        out_mi_cols = [c for c in _MICRO_OUT_COLS if c in mdf_filt.columns]
        micro_out = mdf_filt[out_mi_cols].copy()
        micro_out.insert(0, "macro_rule_type", [label_to_type.get(str(l), "unknown") for l in m_labels_filt])
        for key in reversed(list(meta)):
            micro_out.insert(0, key, meta[key])

    return macro_out, micro_out, macro_summary_rows, micro_summary_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis(base_dir, output_dir,
                  macro_floor_sup=None, macro_floor_conf=None, macro_floor_lift=None,
                  micro_floor_sup=None, micro_floor_conf=None, micro_floor_lift=None,
                  n_workers=_DEFAULT_WORKERS):
    """Scan all per-k macro rule files, classify, grid-sweep, and chain to micro.

    Grids are auto-detected from CSV metadata. Floor filters prune both
    the grid (summaries) and the individual rule listings.
    all_k/ is excluded to avoid double-counting.

    Returns a dict with paths: macro_rules, micro_rules, summary_macro, summary_micro.
    """
    logger.info("=" * 60)
    logger.info("  Unified Association Rule Analysis")
    logger.info("=" * 60)
    logger.info("  Base dir            : %s", base_dir)
    logger.info("  Macro floor support : %s", macro_floor_sup or "all")
    logger.info("  Macro floor conf    : %s", macro_floor_conf or "all")
    logger.info("  Macro floor lift    : %s", macro_floor_lift or "auto")
    logger.info("  Micro floor support : %s", micro_floor_sup or "all")
    logger.info("  Micro floor conf    : %s", micro_floor_conf or "all")
    logger.info("  Micro floor lift    : %s", micro_floor_lift or "auto")
    logger.info("  Workers             : %d", n_workers)
    logger.info("=" * 60)

    # Find per-k macro files, skip all_k and micro subdirs.
    all_files = sorted(
        f for f in base_dir.rglob("arm_*rules.csv")
        if "/all_k/" not in str(f) and "/micro/" not in str(f)
    )
    logger.info("Found %d per-k macro rule files.", len(all_files))
    if not all_files:
        logger.warning("No macro rule files found under %s.", base_dir)
        return {k: None for k in ("macro_rules", "micro_rules", "summary_macro", "summary_micro")}

    macro_parts, micro_parts = [], []
    macro_sum_rows, micro_sum_rows = [], []
    eff = min(n_workers, len(all_files))

    def _do(fp):
        return _process_one_file(fp, base_dir,
                                  macro_floor_sup, macro_floor_conf, macro_floor_lift,
                                  micro_floor_sup, micro_floor_conf, micro_floor_lift)

    if eff <= 1:
        for fp in all_files:
            m, mi, ms, mis = _do(fp)
            if not m.empty: macro_parts.append(m)
            if not mi.empty: micro_parts.append(mi)
            macro_sum_rows.extend(ms)
            micro_sum_rows.extend(mis)
    else:
        logger.info("Dispatching %d files across %d workers.", len(all_files), eff)
        with ThreadPoolExecutor(max_workers=eff) as pool:
            futs = {pool.submit(_do, fp): fp for fp in all_files}
            for fut in as_completed(futs):
                try:
                    m, mi, ms, mis = fut.result()
                    if not m.empty: macro_parts.append(m)
                    if not mi.empty: micro_parts.append(mi)
                    macro_sum_rows.extend(ms)
                    micro_sum_rows.extend(mis)
                except Exception as exc:
                    logger.error("Error processing %s: %s", futs[fut].name, exc)

    # --- Build output subdirectory with floor values in the name ---
    # This way different runs don't overwrite each other.
    tag_parts = []
    if macro_floor_sup is not None:
        tag_parts.append(f"macro_sup{macro_floor_sup}")
    if macro_floor_conf is not None:
        tag_parts.append(f"macro_conf{macro_floor_conf}")
    if macro_floor_lift is not None:
        tag_parts.append(f"macro_lift{macro_floor_lift}")
    if micro_floor_sup is not None:
        tag_parts.append(f"micro_sup{micro_floor_sup}")
    if micro_floor_conf is not None:
        tag_parts.append(f"micro_conf{micro_floor_conf}")
    if micro_floor_lift is not None:
        tag_parts.append(f"micro_lift{micro_floor_lift}")
    subdir = "_".join(tag_parts) if tag_parts else "no_filter"
    output_dir = output_dir / subdir

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = dict.fromkeys((
        "macro_rules", "micro_rules",
        "macro_actionable", "macro_biased",
        "micro_actionable", "micro_biased",
        "summary_macro", "summary_micro",
    ))

    if macro_parts:
        df = (pd.concat(macro_parts, ignore_index=True)
              .sort_values(["state", "classifier", "percentile", "k", "class", "rule_type", "support"],
                           ascending=[True, True, True, True, True, True, False])
              .reset_index(drop=True))
        paths["macro_rules"] = output_dir / "macro_rules.csv"
        df.to_csv(paths["macro_rules"], index=False, float_format="%.6f")
        logger.info("Macro rules        -> %s  (%d)", paths["macro_rules"], len(df))

        # Filtered files: only actionable / only biased.
        df_act = df[df["rule_type"] == "actionable"]
        if not df_act.empty:
            paths["macro_actionable"] = output_dir / "macro_actionable.csv"
            df_act.to_csv(paths["macro_actionable"], index=False, float_format="%.6f")
            logger.info("Macro actionable   -> %s  (%d)", paths["macro_actionable"], len(df_act))

        df_bias = df[df["rule_type"] == "biased"]
        if not df_bias.empty:
            paths["macro_biased"] = output_dir / "macro_biased.csv"
            df_bias.to_csv(paths["macro_biased"], index=False, float_format="%.6f")
            logger.info("Macro biased       -> %s  (%d)", paths["macro_biased"], len(df_bias))
    else:
        logger.warning("No macro results found.")

    if micro_parts:
        df = (pd.concat(micro_parts, ignore_index=True)
              .sort_values(["state", "classifier", "percentile", "k", "class", "macro_rule_type", "macro_antecedents", "macro_consequents", "support"], ascending=[True, True, True, True, True, True, True, True, False])
              .reset_index(drop=True))
        paths["micro_rules"] = output_dir / "micro_rules.csv"
        df.to_csv(paths["micro_rules"], index=False, float_format="%.6f")
        logger.info("Micro rules        -> %s  (%d)", paths["micro_rules"], len(df))

        df_act = df[df["macro_rule_type"] == "actionable"]
        if not df_act.empty:
            paths["micro_actionable"] = output_dir / "micro_actionable.csv"
            df_act.to_csv(paths["micro_actionable"], index=False, float_format="%.6f")
            logger.info("Micro actionable   -> %s  (%d)", paths["micro_actionable"], len(df_act))

        df_bias = df[df["macro_rule_type"] == "biased"]
        if not df_bias.empty:
            paths["micro_biased"] = output_dir / "micro_biased.csv"
            df_bias.to_csv(paths["micro_biased"], index=False, float_format="%.6f")
            logger.info("Micro biased       -> %s  (%d)", paths["micro_biased"], len(df_bias))
    else:
        logger.warning("No micro results found.")

    if macro_sum_rows:
        df = (pd.DataFrame(macro_sum_rows)
              .sort_values(_CONFIG_COLS + ["min_support", "min_confidence"])
              .reset_index(drop=True))
        paths["summary_macro"] = output_dir / "summary_macro.csv"
        df.to_csv(paths["summary_macro"], index=False)
        logger.info("Summary macro      -> %s  (%d rows)", paths["summary_macro"], len(df))

    if micro_sum_rows:
        df = (pd.DataFrame(micro_sum_rows)
              .sort_values(_CONFIG_COLS + ["macro_rule", "min_support", "min_confidence"])
              .reset_index(drop=True))
        paths["summary_micro"] = output_dir / "summary_micro.csv"
        df.to_csv(paths["summary_micro"], index=False)
        logger.info("Summary micro      -> %s  (%d rows)", paths["summary_micro"], len(df))

    logger.info("=" * 60)
    logger.info("  Done.")
    logger.info("=" * 60)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        prog="python -m src.summary_analysis",
        description=(
            "Scan per-k association rules, classify as actionable/biased,\n"
            "produce individual rule listings and grid-sweep summaries.\n"
            "Grids auto-detected from CSV metadata. all_k/ excluded."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base-dir", type=Path, default=Path("results"))
    p.add_argument("--output-dir", type=Path, default=Path("results/analysis"))

    mg = p.add_argument_group("Macro floor filters")
    mg.add_argument("--macro-min-supp", type=float, default=None,
                    help="Min support for macro rules/grid.")
    mg.add_argument("--macro-min-conf", type=float, default=None,
                    help="Min confidence for macro rules/grid.")
    mg.add_argument("--macro-min-lift", type=float, default=None,
                    help="Override lift floor (default: from CSV).")

    ug = p.add_argument_group("Micro floor filters")
    ug.add_argument("--micro-min-supp", type=float, default=None,
                    help="Min support for micro rules/grid.")
    ug.add_argument("--micro-min-conf", type=float, default=None,
                    help="Min confidence for micro rules/grid.")
    ug.add_argument("--micro-min-lift", type=float, default=None,
                    help="Override micro lift floor (default: from CSV).")

    pf = p.add_argument_group("Performance")
    pf.add_argument("--workers", type=int, default=_DEFAULT_WORKERS,
                    help=f"Worker threads (default: {_DEFAULT_WORKERS}).")

    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   default="INFO")
    a = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, a.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    run_analysis(
        base_dir=a.base_dir, output_dir=a.output_dir,
        macro_floor_sup=a.macro_min_supp, macro_floor_conf=a.macro_min_conf,
        macro_floor_lift=a.macro_min_lift,
        micro_floor_sup=a.micro_min_supp, micro_floor_conf=a.micro_min_conf,
        micro_floor_lift=a.micro_min_lift,
        n_workers=a.workers,
    )


if __name__ == "__main__":
    main()