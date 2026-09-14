#!/usr/bin/env python3
"""
Historical skewness diagnostics for the PPA FE study.

Run this after extract_match_PJM_data.py. The script reads the per-match CSVs
from "Match files of original data" and quantifies skewness/tail concentration
for generation, demand, seller LMP, buyer LMP, and the model-derived buyer
purchase price.

Outputs are written to "Historical_Skewness" by default.

Key outputs
-----------
- pooled_variable_skewness.csv
- skewness_by_match_variable.csv
- skewness_by_match_quarter_variable.csv
- skewness_by_match_quarter_hour_variable.csv
- pooled_residual_skewness.csv
- residual_skewness_by_match_variable.csv
- historical_tail_stress_by_match.csv
- historical_tail_stress_summary.csv
- fe_proxy_skewness.csv
- fe_proxy_asg_vs_asc_gap.csv
- historical_skewness_report.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Submission dtype helpers
# ============================================================
# These wrappers reduce DataFrame memory use without changing the financial
# calculations: identifiers/counters are downcast, repeated labels become
# categoricals, and continuous numerical columns remain float64.
import numpy as np

_PD_READ_CSV = pd.read_csv
_PD_READ_EXCEL = pd.read_excel

_INTEGER_DTYPE_CANDIDATES = {
    "match_id": np.int32,
    "hour": np.int16,
    "hour_index": np.int16,
    "replication": np.int16,
    "case_order": np.int16,
    "enabled": np.int8,
    "rank": np.int32,
}

_CATEGORY_DTYPE_CANDIDATES = {
    "case_id",
    "case_family",
    "case_label",
    "combined_category",
    "metric",
    "mutation_axis",
    "mutation_direction",
    "mutation_family",
    "mutation_label",
    "ppa_type",
    "profile_type",
    "risk_group",
    "risk_label",
    "scenario_name",
    "scenario_type",
    "solution_type",
    "status",
    "variable",
    "var_i",
    "var_j",
}


def _integer_dtype_fits(values, dtype) -> bool:
    if len(values) == 0:
        return True
    info = np.iinfo(dtype)
    return float(np.nanmin(values)) >= info.min and float(np.nanmax(values)) <= info.max


def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Conservatively compact non-financial columns after file loading."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_integer_dtype(series.dtype):
            df[col] = pd.to_numeric(series, downcast="integer")

    for col, dtype in _INTEGER_DTYPE_CANDIDATES.items():
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.isna().any():
            continue
        values = numeric.to_numpy(dtype="float64", copy=False)
        rounded = np.rint(values)
        if np.array_equal(values, rounded) and _integer_dtype_fits(rounded, dtype):
            df[col] = rounded.astype(dtype, copy=False)

    n_rows = len(df)
    for col in _CATEGORY_DTYPE_CANDIDATES.intersection(df.columns):
        series = df[col]
        if pd.api.types.is_categorical_dtype(series.dtype):
            continue
        if not (pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype)):
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        n_unique = int(non_null.nunique())
        if n_unique <= min(128, max(2, n_rows // 2)):
            df[col] = series.astype("category")

    return df


def read_csv_optimized(*args, **kwargs) -> pd.DataFrame:
    return optimize_dataframe_dtypes(_PD_READ_CSV(*args, **kwargs))


def read_excel_optimized(*args, **kwargs) -> pd.DataFrame:
    return optimize_dataframe_dtypes(_PD_READ_EXCEL(*args, **kwargs))

try:
    from scipy import stats
except Exception:
    stats = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


SCRIPT_DIR = Path(__file__).resolve().parent


def _build_search_roots():
    roots = []
    for anchor in [SCRIPT_DIR, Path.cwd(), Path("/mnt/data")]:
        if anchor is None:
            continue
        path = Path(anchor).expanduser()
        roots.append(path)
        roots.extend(path.parents)
    seen = set()
    unique = []
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


# ============================================================
# 1. Settings
# ============================================================

SEARCH_ROOTS = _build_search_roots()

MATCH_FOLDER_NAME = "Match files of original data"
OUTPUT_FOLDER_NAME = "Historical_Skewness"

SEASON_MAPPING = {
    12: "Q1", 1: "Q1", 2: "Q1",
    3: "Q2", 4: "Q2", 5: "Q2",
    6: "Q3", 7: "Q3", 8: "Q3",
    9: "Q4", 10: "Q4", 11: "Q4",
}

RAW_VARIABLES = [
    "generation",
    "demand",
    "seller_lmp",
    "buyer_lmp_out",
    "buyer_lmp_in",
]

VOLUME_VARIABLES = ["generation", "demand"]
PRICE_VARIABLES = ["seller_lmp", "buyer_lmp_out", "buyer_lmp_in"]

PROFILE_KEYS = ["match_id", "quarter", "hour"]
POSITIVE_EPS = 1.0


# ============================================================
# 2. Path helpers
# ============================================================

def resolve_existing_dir(candidates: Sequence[Path], description: str) -> Path:
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    raise FileNotFoundError(
        f"Could not find {description}. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def resolve_match_dir(explicit: Optional[str] = None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"--match-dir does not exist or is not a directory: {path}")
        return path

    candidates = []
    for root in SEARCH_ROOTS:
        candidates.extend([
            root / "Input data and files" / MATCH_FOLDER_NAME,
            root / "Data and files" / MATCH_FOLDER_NAME,
            root / MATCH_FOLDER_NAME,
        ])
    return resolve_existing_dir(candidates, f"match folder '{MATCH_FOLDER_NAME}'")


def resolve_output_dir(match_dir: Path, explicit: Optional[str] = None) -> Path:
    out = Path(explicit) if explicit else match_dir.parent / OUTPUT_FOLDER_NAME
    out.mkdir(parents=True, exist_ok=True)
    return out


def extract_match_id(path: Path):
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else path.stem


# ============================================================
# 3. Statistics helpers
# ============================================================

def finite_numeric(x: Iterable) -> pd.Series:
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s.astype(float)


def safe_divide(num, den) -> float:
    try:
        if den is None or not np.isfinite(den) or abs(float(den)) < 1e-12:
            return np.nan
        return float(num) / float(den)
    except Exception:
        return np.nan


def safe_corr(x: Iterable, y: Iterable) -> float:
    tmp = pd.DataFrame({
        "x": pd.to_numeric(pd.Series(x), errors="coerce"),
        "y": pd.to_numeric(pd.Series(y), errors="coerce"),
    }).replace([np.inf, -np.inf], np.nan).dropna()

    if len(tmp) < 3:
        return np.nan
    if tmp["x"].nunique(dropna=True) <= 1 or tmp["y"].nunique(dropna=True) <= 1:
        return np.nan
    return float(tmp["x"].corr(tmp["y"]))


def sample_skewness(x: Iterable) -> float:
    s = finite_numeric(x)
    if len(s) < 3 or s.nunique(dropna=True) <= 1:
        return np.nan
    if stats is not None:
        return float(stats.skew(s.to_numpy(), bias=False, nan_policy="omit"))
    return float(s.skew())


def sample_excess_kurtosis(x: Iterable) -> float:
    s = finite_numeric(x)
    if len(s) < 4 or s.nunique(dropna=True) <= 1:
        return np.nan
    if stats is not None:
        return float(stats.kurtosis(s.to_numpy(), fisher=True, bias=False, nan_policy="omit"))
    return float(s.kurt())


def distribution_metrics(x: Iterable, min_n: int = 3) -> Dict[str, float]:
    """Classical and robust skewness/tail diagnostics.

    sample_skew:
        Fisher-Pearson sample skewness. Positive means right-skewed; negative
        means left-skewed.

    bowley_skew:
        Quartile skewness: (Q3 + Q1 - 2Q2)/(Q3-Q1).

    tail_asym_95_05:
        ((P95-P50) - (P50-P05)) / ((P95-P50) + (P50-P05)).
        Positive means wider upper tail; negative means wider lower tail.

    tail_ratio_95_05:
        (P95-P50)/(P50-P05). Values above 1 mean wider upper tail.
    """
    s = finite_numeric(x)
    n = int(len(s))
    out = {"n": n}

    keys = [
        "mean", "std", "min", "p01", "p05", "p10", "p25", "median", "p75", "p90",
        "p95", "p99", "max", "sample_skew", "excess_kurtosis", "bowley_skew",
        "tail_ratio_95_05", "tail_asym_95_05", "mean_median_gap_std", "zero_share",
        "negative_share", "positive_share", "iqr", "range_95_05",
    ]

    if n == 0:
        out.update({k: np.nan for k in keys})
        return out

    q = s.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    p01, p05, p10, p25, p50, p75, p90, p95, p99 = [float(q.loc[v]) for v in q.index]
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if n > 1 else 0.0

    iqr = p75 - p25
    upper = p95 - p50
    lower = p50 - p05

    out.update({
        "mean": mean,
        "std": std,
        "min": float(s.min()),
        "p01": p01,
        "p05": p05,
        "p10": p10,
        "p25": p25,
        "median": p50,
        "p75": p75,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "max": float(s.max()),
        "sample_skew": sample_skewness(s) if n >= min_n else np.nan,
        "excess_kurtosis": sample_excess_kurtosis(s) if n >= max(min_n, 4) else np.nan,
        "bowley_skew": safe_divide(p75 + p25 - 2.0 * p50, iqr),
        "tail_ratio_95_05": safe_divide(upper, lower),
        "tail_asym_95_05": safe_divide(upper - lower, upper + lower),
        "mean_median_gap_std": safe_divide(mean - p50, std),
        "zero_share": float((s == 0).mean()),
        "negative_share": float((s < 0).mean()),
        "positive_share": float((s > 0).mean()),
        "iqr": iqr,
        "range_95_05": p95 - p05,
    })
    return out


def skew_direction(row: pd.Series) -> str:
    skew = row.get("sample_skew", np.nan)
    tail = row.get("tail_asym_95_05", np.nan)

    if np.isfinite(skew):
        if skew >= 0.50:
            return "right-skewed"
        if skew <= -0.50:
            return "left-skewed"

    if np.isfinite(tail):
        if tail >= 0.20:
            return "right-tail wider"
        if tail <= -0.20:
            return "left-tail wider"

    return "near-symmetric/mild"


def add_direction_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["skew_direction"] = out.apply(skew_direction, axis=1)
    return out


def summarize_groups(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
    min_n: int = 3,
) -> pd.DataFrame:
    """Summarize one or more value columns within optional groups.

    If group_cols contains "variable", that column is preserved as the underlying
    historical variable, and the summarized data column is named "value_column".
    Otherwise, the summarized variable is stored in "variable".
    """
    value_cols = [c for c in value_cols if c in df.columns]
    rows: List[Dict] = []

    if len(group_cols) == 0:
        grouped = [((), df)]
    else:
        grouped = df.groupby(list(group_cols), dropna=False, sort=True)

    for key, g in grouped:
        if len(group_cols) == 0:
            base = {"scope": "pooled"}
        else:
            key_tuple = key if isinstance(key, tuple) else (key,)
            base = dict(zip(group_cols, key_tuple))

        for value_col in value_cols:
            row = dict(base)
            if "variable" in group_cols:
                row["value_column"] = value_col
            else:
                row["variable"] = value_col
            row.update(distribution_metrics(g[value_col], min_n=min_n))
            rows.append(row)

    return pd.DataFrame(rows)


def share_of_sum_in_mask(x: pd.Series, mask: pd.Series) -> float:
    x_num = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x_num.notna().sum() == 0:
        return np.nan

    mask = mask.reindex(x_num.index).fillna(False)
    total = x_num.sum(skipna=True)
    if not np.isfinite(total) or abs(total) < 1e-12:
        return np.nan

    return float(x_num[mask].sum(skipna=True) / total)


# ============================================================
# 4. Load match-level data
# ============================================================

def read_match_files(match_dir: Path) -> pd.DataFrame:
    paths = sorted(match_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in match folder: {match_dir}")

    frames = []
    for path in paths:
        df = read_csv_optimized(path)
        if "timestamp" not in df.columns:
            raise KeyError(f"{path.name} is missing required column: timestamp")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["match_id"] = extract_match_id(path)

        if "hour" not in df.columns:
            df["hour"] = df["timestamp"].dt.hour
        if "quarter" not in df.columns:
            df["quarter"] = df["timestamp"].dt.month.map(SEASON_MAPPING)

        df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int64")
        for col in RAW_VARIABLES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        keep_cols = ["match_id", "timestamp", "quarter", "hour"] + [c for c in RAW_VARIABLES if c in df.columns]
        frames.append(df[keep_cols])

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["match_id", "timestamp"]).reset_index(drop=True)
    return panel


# ============================================================
# 5. De-profiled residuals
# ============================================================

def positive_floor(x: Iterable, eps: float) -> np.ndarray:
    arr = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce"), dtype=float)
    return np.where(np.isfinite(arr), np.maximum(arr, -eps + 1e-9), np.nan)


def compute_profiles_and_residuals(panel: pd.DataFrame, eps: float = POSITIVE_EPS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Quarter-hour medians and residuals.

    This follows the same idea as the correlation-extraction workflow:
    - generation and demand use log-ratio residuals around the quarter-hour median;
    - LMP variables use additive residuals around the quarter-hour median.
    """
    variables = [v for v in RAW_VARIABLES if v in panel.columns]
    profile_frames = []
    residual_frames = []

    for variable in variables:
        prof = (
            panel.groupby(PROFILE_KEYS, dropna=False)[variable]
            .median()
            .reset_index(name="profile_value")
        )
        prof["variable"] = variable
        profile_frames.append(prof)

        tmp = panel[["match_id", "timestamp", "quarter", "hour", variable]].copy()
        tmp = tmp.merge(prof, on=PROFILE_KEYS, how="left")
        tmp["variable"] = variable
        tmp["raw_value"] = pd.to_numeric(tmp[variable], errors="coerce")

        if variable in VOLUME_VARIABLES:
            raw = positive_floor(tmp["raw_value"], eps)
            profile = positive_floor(tmp["profile_value"], eps)
            ratio = (raw + eps) / (profile + eps)
            ratio = np.where(np.isfinite(ratio), np.clip(ratio, np.exp(-20.0), np.exp(20.0)), np.nan)
            tmp["residual"] = np.log(ratio)
            tmp["residual_type"] = "log_ratio_to_quarter_hour_median"
        else:
            tmp["residual"] = tmp["raw_value"] - tmp["profile_value"]
            tmp["residual_type"] = "additive_to_quarter_hour_median"

        residual_frames.append(tmp[[
            "match_id", "timestamp", "quarter", "hour", "variable",
            "profile_value", "raw_value", "residual", "residual_type",
        ]])

    profiles = pd.concat(profile_frames, ignore_index=True) if profile_frames else pd.DataFrame()
    residuals = pd.concat(residual_frames, ignore_index=True) if residual_frames else pd.DataFrame()
    return profiles, residuals


# ============================================================
# 6. Tail-stress metrics
# ============================================================

def add_derived_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()

    if {"buyer_lmp_in", "demand"}.issubset(df.columns):
        df["buyer_spot_bill_in"] = df["buyer_lmp_in"] * df["demand"]
    if {"buyer_lmp_out", "demand"}.issubset(df.columns):
        df["buyer_spot_bill_out"] = df["buyer_lmp_out"] * df["demand"]
    if {"seller_lmp", "generation"}.issubset(df.columns):
        df["seller_spot_revenue"] = df["seller_lmp"] * df["generation"]
    if {"generation", "demand"}.issubset(df.columns):
        df["gen_minus_demand"] = df["generation"] - df["demand"]
        df["shortage_volume_d_minus_g"] = (df["demand"] - df["generation"]).clip(lower=0)
        df["surplus_volume_g_minus_d"] = (df["generation"] - df["demand"]).clip(lower=0)
    if {"seller_lmp", "buyer_lmp_out"}.issubset(df.columns):
        df["basis_seller_minus_buyer_out"] = df["seller_lmp"] - df["buyer_lmp_out"]
    if {"seller_lmp", "buyer_lmp_in"}.issubset(df.columns):
        df["basis_seller_minus_buyer_in"] = df["seller_lmp"] - df["buyer_lmp_in"]
    if {"buyer_lmp_in", "buyer_lmp_out"}.issubset(df.columns):
        df["buyer_markup_price_gap"] = df["buyer_lmp_in"] - df["buyer_lmp_out"]

    return df


def summarize_tail_stress_by_match(panel: pd.DataFrame) -> pd.DataFrame:
    df = add_derived_features(panel)
    price_ref = "buyer_lmp_in" if "buyer_lmp_in" in df.columns else "buyer_lmp_out"
    if price_ref not in df.columns:
        price_ref = None

    rows = []
    for match_id, g in df.groupby("match_id", sort=True):
        row: Dict[str, float] = {
            "match_id": match_id,
            "n_hours": int(len(g)),
            "date_min": g["timestamp"].min(),
            "date_max": g["timestamp"].max(),
            "price_reference_variable": price_ref,
        }

        corr_pairs = [
            ("generation", "demand", "corr_generation_demand"),
            ("demand", price_ref, "corr_demand_buyer_price") if price_ref else None,
            ("generation", "seller_lmp", "corr_generation_seller_lmp"),
            ("seller_lmp", "buyer_lmp_out", "corr_seller_buyer_lmp_out"),
            ("seller_lmp", "buyer_lmp_in", "corr_seller_buyer_lmp_in"),
            ("shortage_volume_d_minus_g", price_ref, "corr_shortage_buyer_price") if price_ref else None,
        ]

        for item in corr_pairs:
            if item is None:
                continue
            x, y, name = item
            row[name] = safe_corr(g[x], g[y]) if x in g.columns and y in g.columns else np.nan

        # Skewness of derived economic quantities.
        feature_list = [
            "buyer_spot_bill_in",
            "buyer_spot_bill_out",
            "seller_spot_revenue",
            "shortage_volume_d_minus_g",
            "surplus_volume_g_minus_d",
            "gen_minus_demand",
            "basis_seller_minus_buyer_out",
            "basis_seller_minus_buyer_in",
            "buyer_markup_price_gap",
        ]
        for feature in feature_list:
            if feature not in g.columns:
                continue
            met = distribution_metrics(g[feature])
            for k in ["sample_skew", "tail_asym_95_05", "p05", "median", "p95"]:
                row[f"{feature}__{k}"] = met[k]

        # Tail coincidence: do high buyer-price hours coincide with demand, low generation, or shortage?
        if price_ref and price_ref in g.columns:
            price = pd.to_numeric(g[price_ref], errors="coerce")
            if price.notna().sum() >= 20:
                top5_price = price >= price.quantile(0.95)
                top10_price = price >= price.quantile(0.90)

                row["top5_price_threshold"] = float(price.quantile(0.95))
                row["top10_price_threshold"] = float(price.quantile(0.90))
                row["top5_price_hour_share"] = float(top5_price.mean())
                row["top10_price_hour_share"] = float(top10_price.mean())

                if "demand" in g.columns:
                    demand = pd.to_numeric(g["demand"], errors="coerce")
                    high10_demand = demand >= demand.quantile(0.90)
                    row["mean_demand_top5_price"] = float(demand[top5_price].mean())
                    row["mean_demand_non_top5_price"] = float(demand[~top5_price].mean())
                    row["demand_top5_price_to_other_ratio"] = safe_divide(
                        row["mean_demand_top5_price"],
                        row["mean_demand_non_top5_price"],
                    )
                    row["demand_share_in_top5_price_hours"] = share_of_sum_in_mask(demand, top5_price)
                    row["joint_top10_price_top10_demand_hour_share"] = float((top10_price & high10_demand).mean())
                    row["joint_top10_price_top10_demand_excess_vs_independence"] = (
                        row["joint_top10_price_top10_demand_hour_share"] - 0.01
                    )

                if "generation" in g.columns:
                    generation = pd.to_numeric(g["generation"], errors="coerce")
                    low10_generation = generation <= generation.quantile(0.10)
                    row["mean_generation_top5_price"] = float(generation[top5_price].mean())
                    row["mean_generation_non_top5_price"] = float(generation[~top5_price].mean())
                    row["generation_top5_price_to_other_ratio"] = safe_divide(
                        row["mean_generation_top5_price"],
                        row["mean_generation_non_top5_price"],
                    )
                    row["generation_share_in_top5_price_hours"] = share_of_sum_in_mask(generation, top5_price)
                    row["joint_top10_price_bottom10_generation_hour_share"] = float((top10_price & low10_generation).mean())
                    row["joint_top10_price_bottom10_generation_excess_vs_independence"] = (
                        row["joint_top10_price_bottom10_generation_hour_share"] - 0.01
                    )

                if "shortage_volume_d_minus_g" in g.columns:
                    shortage = pd.to_numeric(g["shortage_volume_d_minus_g"], errors="coerce")
                    high10_shortage = shortage >= shortage.quantile(0.90)
                    row["mean_shortage_top5_price"] = float(shortage[top5_price].mean())
                    row["mean_shortage_non_top5_price"] = float(shortage[~top5_price].mean())
                    row["shortage_top5_price_to_other_ratio"] = safe_divide(
                        row["mean_shortage_top5_price"],
                        row["mean_shortage_non_top5_price"],
                    )
                    row["shortage_share_in_top5_price_hours"] = share_of_sum_in_mask(shortage, top5_price)
                    row["joint_top10_price_top10_shortage_hour_share"] = float((top10_price & high10_shortage).mean())
                    row["joint_top10_price_top10_shortage_excess_vs_independence"] = (
                        row["joint_top10_price_top10_shortage_hour_share"] - 0.01
                    )

                for bill_feature in ["buyer_spot_bill_in", "buyer_spot_bill_out"]:
                    if bill_feature in g.columns:
                        bill = pd.to_numeric(g[bill_feature], errors="coerce")
                        row[f"{bill_feature}_share_in_top5_price_hours"] = share_of_sum_in_mask(bill, top5_price)
                        if bill.notna().sum() >= 20:
                            top5_bill = bill >= bill.quantile(0.95)
                            row[f"{bill_feature}_share_in_top5_bill_hours"] = share_of_sum_in_mask(bill, top5_bill)

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_across_matches(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    skip = {"match_id", "date_min", "date_max", "price_reference_variable"}
    for col in [c for c in df.columns if c not in skip]:
        s = finite_numeric(df[col])
        if len(s) == 0:
            continue
        met = distribution_metrics(s)
        rows.append({
            "metric": col,
            "n_matches": met["n"],
            "mean_across_matches": met["mean"],
            "median_across_matches": met["median"],
            "p05_across_matches": met["p05"],
            "p95_across_matches": met["p95"],
            "sample_skew_across_matches": met["sample_skew"],
        })
    return pd.DataFrame(rows).sort_values("metric")


# ============================================================
# 7. Physical-PPA buyer-FE proxy diagnostics
# ============================================================

def physical_buyer_fe(
    g: pd.DataFrame,
    profile: str,
    strike: float,
    fixed_q: Optional[float] = None,
    penalty_gamma: float = 0.0,
    availability: float = 0.95,
    price_in_col: str = "buyer_lmp_in",
    price_out_col: str = "buyer_lmp_out",
) -> pd.Series:
    """Buyer FE for physical PPA on historical paths.

    FE_b,t = cost_with_PPA_t - N_in,t d_t

    cost_with_PPA_t =
        pi q_del,t + N_in,t(d_t-q_del,t)^+ - N_out,t(q_del,t-d_t)^+ - Pen_t
    """
    d = pd.to_numeric(g["demand"], errors="coerce")
    gen = pd.to_numeric(g["generation"], errors="coerce")
    n_in = pd.to_numeric(g[price_in_col], errors="coerce")
    n_out = pd.to_numeric(g[price_out_col], errors="coerce") if price_out_col in g.columns else n_in

    if profile == "AsC":
        qdel = d.copy()
    elif profile == "AsG":
        qdel = gen.copy()
    elif profile == "Fix":
        if fixed_q is None or not np.isfinite(fixed_q):
            raise ValueError("fixed_q must be supplied for Fix profile.")
        qdel = pd.Series(float(fixed_q), index=g.index)
    else:
        raise ValueError(f"Unknown profile: {profile}")

    purchase_shortfall = (d - qdel).clip(lower=0)
    resale_surplus = (qdel - d).clip(lower=0)

    penalty = pd.Series(0.0, index=g.index)
    if profile == "AsG" and penalty_gamma > 0:
        penalty_rate = penalty_gamma * float(n_in.quantile(0.95))
        guarantee = availability * float(gen.mean())
        penalty = penalty_rate * (guarantee - qdel).clip(lower=0)

    cost_with_ppa = strike * qdel + n_in * purchase_shortfall - n_out * resale_surplus - penalty
    spot_benchmark = n_in * d
    return cost_with_ppa - spot_benchmark


def build_fe_proxy_outputs(panel: pd.DataFrame, availability: float = 0.95) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = {"generation", "demand", "buyer_lmp_out"}
    if not required.issubset(panel.columns):
        return pd.DataFrame(), pd.DataFrame()

    price_in_col = "buyer_lmp_in" if "buyer_lmp_in" in panel.columns else "buyer_lmp_out"
    price_out_col = "buyer_lmp_out"

    skew_rows = []
    gap_rows = []

    for match_id, g0 in panel.groupby("match_id", sort=True):
        g = g0.dropna(subset=["generation", "demand", price_in_col, price_out_col]).copy()
        if len(g) < 20:
            continue

        n_in = pd.to_numeric(g[price_in_col], errors="coerce")
        strike_rules = {
            "buyer_purchase_price_median": float(n_in.median()),
            "buyer_purchase_price_mean": float(n_in.mean()),
            "buyer_purchase_price_p75": float(n_in.quantile(0.75)),
        }

        if "seller_lmp" in g.columns and g["seller_lmp"].notna().sum() > 0:
            strike_rules["seller_lmp_mean"] = float(pd.to_numeric(g["seller_lmp"], errors="coerce").mean())

        q_rules = {
            "median_demand": float(g["demand"].median()),
            "median_generation": float(g["generation"].median()),
            "min_median_generation_demand": float(min(g["generation"].median(), g["demand"].median())),
            "p25_demand": float(g["demand"].quantile(0.25)),
        }

        for strike_rule, strike in strike_rules.items():
            if not np.isfinite(strike):
                continue

            # As-Consumed and As-Generated.
            for profile in ["AsC", "AsG"]:
                penalty_grid = [0.0, 1.0] if profile == "AsG" else [0.0]
                for penalty_gamma in penalty_grid:
                    fe = physical_buyer_fe(
                        g,
                        profile=profile,
                        strike=strike,
                        fixed_q=None,
                        penalty_gamma=penalty_gamma,
                        availability=availability,
                        price_in_col=price_in_col,
                        price_out_col=price_out_col,
                    )
                    row = {
                        "match_id": match_id,
                        "profile": profile,
                        "strike_rule": strike_rule,
                        "strike": strike,
                        "q_rule": "",
                        "fixed_q": np.nan,
                        "penalty_gamma": penalty_gamma,
                    }
                    row.update(distribution_metrics(fe))
                    skew_rows.append(row)

            # Fixed-volume proxies.
            for q_rule, fixed_q in q_rules.items():
                if not np.isfinite(fixed_q):
                    continue
                fe = physical_buyer_fe(
                    g,
                    profile="Fix",
                    strike=strike,
                    fixed_q=fixed_q,
                    penalty_gamma=0.0,
                    availability=availability,
                    price_in_col=price_in_col,
                    price_out_col=price_out_col,
                )
                row = {
                    "match_id": match_id,
                    "profile": "Fix",
                    "strike_rule": strike_rule,
                    "strike": strike,
                    "q_rule": q_rule,
                    "fixed_q": fixed_q,
                    "penalty_gamma": 0.0,
                }
                row.update(distribution_metrics(fe))
                skew_rows.append(row)

            # AsG protection gap relative to AsC, no penalty.
            fe_asc = physical_buyer_fe(
                g, "AsC", strike, None, 0.0, availability, price_in_col, price_out_col
            )
            fe_asg = physical_buyer_fe(
                g, "AsG", strike, None, 0.0, availability, price_in_col, price_out_col
            )
            gap = fe_asg - fe_asc

            shortage = (pd.to_numeric(g["demand"], errors="coerce") - pd.to_numeric(g["generation"], errors="coerce")).clip(lower=0)
            shortage_value = (n_in - strike) * shortage
            top5_price = n_in >= n_in.quantile(0.95)

            gap_met = distribution_metrics(gap)
            shortage_met = distribution_metrics(shortage_value)

            gap_rows.append({
                "match_id": match_id,
                "strike_rule": strike_rule,
                "strike": strike,
                "comparison": "AsG_no_penalty_minus_AsC",
                "gap_mean": gap_met["mean"],
                "gap_median": gap_met["median"],
                "gap_p05": gap_met["p05"],
                "gap_p95": gap_met["p95"],
                "gap_sample_skew": gap_met["sample_skew"],
                "gap_tail_asym_95_05": gap_met["tail_asym_95_05"],
                "shortage_value_mean": shortage_met["mean"],
                "shortage_value_median": shortage_met["median"],
                "shortage_value_p95": shortage_met["p95"],
                "shortage_value_sample_skew": shortage_met["sample_skew"],
                "shortage_value_tail_asym_95_05": shortage_met["tail_asym_95_05"],
                "shortage_value_share_in_top5_price_hours": share_of_sum_in_mask(shortage_value.clip(lower=0), top5_price),
            })

    return add_direction_labels(pd.DataFrame(skew_rows)), pd.DataFrame(gap_rows)


# ============================================================
# 8. Optional plots
# ============================================================

def make_plots(panel: pd.DataFrame, output_dir: Path, sample_rows: int = 100_000) -> None:
    if plt is None:
        print("matplotlib not available; skipping plots.")
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260423)
    if len(panel) > sample_rows:
        sample = panel.loc[rng.choice(panel.index.to_numpy(), size=sample_rows, replace=False)].copy()
    else:
        sample = panel.copy()

    for var in [v for v in RAW_VARIABLES if v in panel.columns]:
        s = finite_numeric(panel[var])
        if len(s) < 10:
            continue
        lo, hi = s.quantile([0.005, 0.995])
        plt.figure(figsize=(7, 4.5))
        plt.hist(s.clip(lower=lo, upper=hi), bins=80)
        plt.xlabel(var)
        plt.ylabel("count")
        plt.title(f"{var}: pooled historical distribution\nclipped to 0.5%--99.5% for display")
        plt.tight_layout()
        plt.savefig(fig_dir / f"hist_{var}.png", dpi=180)
        plt.close()

    derived = add_derived_features(sample)

    scatter_specs = [
        ("buyer_lmp_in", "demand", "scatter_demand_vs_buyer_lmp_in.png", "Demand versus buyer purchase price"),
        ("buyer_lmp_in", "generation", "scatter_generation_vs_buyer_lmp_in.png", "Generation versus buyer purchase price"),
        ("buyer_lmp_in", "shortage_volume_d_minus_g", "scatter_shortage_vs_buyer_lmp_in.png", "Shortage volume versus buyer purchase price"),
        ("buyer_lmp_out", "demand", "scatter_demand_vs_buyer_lmp_out.png", "Demand versus buyer-side LMP"),
    ]

    for xcol, ycol, fname, title in scatter_specs:
        if {xcol, ycol}.issubset(derived.columns):
            plt.figure(figsize=(6, 5))
            plt.scatter(derived[xcol], derived[ycol], s=4, alpha=0.20)
            plt.xlabel(xcol)
            plt.ylabel(ycol)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(fig_dir / fname, dpi=180)
            plt.close()


# ============================================================
# 9. Report
# ============================================================

def md_value(x) -> str:
    if pd.isna(x):
        return "NA"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.4g}"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return str(x)


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int = 30) -> str:
    if df.empty:
        return "_No data._\n"

    cols = [c for c in columns if c in df.columns]
    d = df.loc[:, cols].head(max_rows)

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(md_value(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_summary_report(
    output_dir: Path,
    match_dir: Path,
    panel: pd.DataFrame,
    pooled_raw: pd.DataFrame,
    by_match_raw: pd.DataFrame,
    pooled_resid: pd.DataFrame,
    tail_df: pd.DataFrame,
    fe_proxy_df: pd.DataFrame,
) -> None:
    report_path = output_dir / "historical_skewness_report.md"

    raw_cols = [
        "variable", "n", "mean", "median", "p05", "p95", "sample_skew",
        "tail_ratio_95_05", "tail_asym_95_05", "zero_share",
        "negative_share", "skew_direction",
    ]
    raw_focus = pooled_raw[[c for c in raw_cols if c in pooled_raw.columns]].copy() if not pooled_raw.empty else pd.DataFrame()

    match_summary_rows = []
    if not by_match_raw.empty:
        for variable, g in by_match_raw.groupby("variable"):
            skew = pd.to_numeric(g["sample_skew"], errors="coerce")
            tail = pd.to_numeric(g["tail_asym_95_05"], errors="coerce")
            match_summary_rows.append({
                "variable": variable,
                "n_matches": int(g["match_id"].nunique()) if "match_id" in g.columns else int(len(g)),
                "median_sample_skew_across_matches": finite_numeric(skew).median(),
                "share_right_skewed_matches_skew_gt_0p5": float((skew > 0.5).mean()),
                "share_left_skewed_matches_skew_lt_neg_0p5": float((skew < -0.5).mean()),
                "median_tail_asym_95_05": finite_numeric(tail).median(),
            })
    match_summary = pd.DataFrame(match_summary_rows)

    resid_focus = pd.DataFrame()
    if not pooled_resid.empty:
        resid_cols = [
            "variable", "value_column", "n", "mean", "median", "p05", "p95",
            "sample_skew", "tail_ratio_95_05", "tail_asym_95_05", "skew_direction",
        ]
        resid_focus = pooled_resid[[c for c in resid_cols if c in pooled_resid.columns]].copy()

    tail_focus_cols = [
        "buyer_spot_bill_in__sample_skew",
        "buyer_spot_bill_in_share_in_top5_price_hours",
        "demand_top5_price_to_other_ratio",
        "generation_top5_price_to_other_ratio",
        "shortage_top5_price_to_other_ratio",
        "shortage_share_in_top5_price_hours",
        "joint_top10_price_top10_demand_excess_vs_independence",
        "joint_top10_price_bottom10_generation_excess_vs_independence",
        "joint_top10_price_top10_shortage_excess_vs_independence",
    ]
    tail_medians = []
    for col in [c for c in tail_focus_cols if c in tail_df.columns]:
        s = finite_numeric(tail_df[col])
        if len(s):
            tail_medians.append({
                "metric": col,
                "median_across_matches": float(s.median()),
                "p05": float(s.quantile(0.05)),
                "p95": float(s.quantile(0.95)),
            })
    tail_medians_df = pd.DataFrame(tail_medians)

    fe_focus = pd.DataFrame()
    if not fe_proxy_df.empty and "strike_rule" in fe_proxy_df.columns:
        tmp = fe_proxy_df[fe_proxy_df["strike_rule"] == "buyer_purchase_price_median"].copy()
        if not tmp.empty:
            fe_focus = (
                tmp.groupby(["profile", "penalty_gamma"], dropna=False)
                .agg(
                    median_FE_skew_across_matches=("sample_skew", "median"),
                    median_FE_tail_asym_across_matches=("tail_asym_95_05", "median"),
                    median_FE_mean_across_matches=("mean", "median"),
                    median_FE_std_across_matches=("std", "median"),
                )
                .reset_index()
            )

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Historical skewness and tail-concentration report\n\n")
        f.write(f"Match folder: `{match_dir}`\n\n")
        f.write(f"Rows loaded: {len(panel):,}\n\n")
        f.write(f"Matches loaded: {panel['match_id'].nunique():,}\n\n")
        if len(panel):
            f.write(f"Date range: {panel['timestamp'].min()} to {panel['timestamp'].max()}\n\n")

        f.write("## 1. Pooled raw-variable skewness\n\n")
        f.write(markdown_table(raw_focus, raw_focus.columns))

        f.write("\n## 2. Match-level skewness summary\n\n")
        f.write(markdown_table(match_summary, match_summary.columns))

        f.write("\n## 3. Pooled de-profiled residual skewness\n\n")
        f.write(markdown_table(resid_focus, resid_focus.columns))

        f.write("\n## 4. Tail-stress metrics across matches\n\n")
        f.write(
            "These metrics indicate whether high buyer-price hours are also high-load, "
            "low-generation, or shortage hours.\n\n"
        )
        f.write(markdown_table(tail_medians_df, tail_medians_df.columns))

        f.write("\n## 5. FE proxy summary, median buyer-price strike only\n\n")
        f.write(
            "These proxy diagnostics use historical paths and simple physical-PPA formulas. "
            "They are not optimizer outputs.\n\n"
        )
        f.write(markdown_table(fe_focus, fe_focus.columns))

        f.write("\n## 6. Interpretation guide\n\n")
        f.write(
            "- Positive `sample_skew` or positive `tail_asym_95_05` means a wider upper tail. "
            "For buyer prices and buyer spot bills, this raw right tail maps into a left tail "
            "in buyer FE because buyer FE subtracts the spot benchmark.\n"
        )
        f.write(
            "- `buyer_spot_bill_in_share_in_top5_price_hours` measures how much of the buyer's "
            "spot bill occurs during the top 5% buyer-price hours. Large values indicate tail "
            "concentration, not merely variance.\n"
        )
        f.write(
            "- `shortage_share_in_top5_price_hours` and "
            "`joint_top10_price_top10_shortage_excess_vs_independence` are relevant for "
            "As-Generated PPAs. If price spikes coincide with `demand > generation`, AsG "
            "delivers less hedge volume when buyer hedge value is highest.\n"
        )


def write_run_summary(output_dir: Path, match_dir: Path, panel: pd.DataFrame, args: argparse.Namespace) -> None:
    summary = {
        "match_folder": str(match_dir),
        "output_folder": str(output_dir),
        "n_rows": int(len(panel)),
        "n_matches": int(panel["match_id"].nunique()) if "match_id" in panel.columns else None,
        "date_min": str(panel["timestamp"].min()) if len(panel) else None,
        "date_max": str(panel["timestamp"].max()) if len(panel) else None,
        "variables_present": [v for v in RAW_VARIABLES if v in panel.columns],
        "make_plots": bool(args.make_plots),
        "include_fe_proxies": not bool(args.no_fe_proxies),
        "notes": [
            "Raw skewness is computed on the matched historical panel.",
            "Residual skewness uses quarter-hour medians: log-ratio residuals for volumes and additive residuals for prices.",
            "buyer_lmp_in is derived by the existing retail-markup rule, not directly observed in PJM raw files.",
            "FE proxy outputs are diagnostic transformations, not selected contracts from the optimizer.",
        ],
    }
    with (output_dir / "skewness_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# ============================================================
# 10. Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze historical skewness and tail concentration in PJM match files."
    )
    parser.add_argument("--match-dir", default=None, help="Path to 'Match files of original data'.")
    parser.add_argument("--output-dir", default=None, help="Output folder. Default: sibling Historical_Skewness.")
    parser.add_argument("--min-cell-n", type=int, default=20, help="Minimum n for quarter-hour cell skewness.")
    parser.add_argument("--profile-eps", type=float, default=POSITIVE_EPS, help="Epsilon for volume log-ratio residuals.")
    parser.add_argument("--make-plots", action="store_true", help="Save histogram and scatter plots.")
    parser.add_argument("--plot-sample-rows", type=int, default=100_000, help="Max rows sampled for scatter plots.")
    parser.add_argument("--no-fe-proxies", action="store_true", help="Skip FE proxy diagnostics.")
    parser.add_argument("--availability", type=float, default=0.95, help="Availability factor for AsG proxy penalty.")
    args = parser.parse_args()

    match_dir = resolve_match_dir(args.match_dir)
    output_dir = resolve_output_dir(match_dir, args.output_dir)

    print(f"Reading match files from: {match_dir}")
    panel = read_match_files(match_dir)
    variables_present = [v for v in RAW_VARIABLES if v in panel.columns]

    if not variables_present:
        raise KeyError("None of the expected variables were found in the match files.")

    print(f"Rows loaded: {len(panel):,}")
    print(f"Matches loaded: {panel['match_id'].nunique():,}")
    print(f"Variables present: {variables_present}")

    # Raw skewness.
    pooled_raw = add_direction_labels(summarize_groups(panel, [], variables_present))
    by_match_raw = add_direction_labels(summarize_groups(panel, ["match_id"], variables_present))
    by_match_quarter_raw = add_direction_labels(summarize_groups(panel, ["match_id", "quarter"], variables_present))
    by_match_qh_raw = add_direction_labels(
        summarize_groups(panel, ["match_id", "quarter", "hour"], variables_present, min_n=args.min_cell_n)
    )

    pooled_raw.to_csv(output_dir / "pooled_variable_skewness.csv", index=False)
    by_match_raw.to_csv(output_dir / "skewness_by_match_variable.csv", index=False)
    by_match_quarter_raw.to_csv(output_dir / "skewness_by_match_quarter_variable.csv", index=False)
    by_match_qh_raw.to_csv(output_dir / "skewness_by_match_quarter_hour_variable.csv", index=False)

    # Residual skewness.
    profiles, residuals = compute_profiles_and_residuals(panel, eps=args.profile_eps)
    profiles.to_csv(output_dir / "quarter_hour_profile_values_long.csv", index=False)
    residuals.to_csv(output_dir / "deprofiled_residuals_long.csv", index=False)

    if residuals.empty:
        pooled_resid = pd.DataFrame()
        resid_by_match = pd.DataFrame()
    else:
        pooled_resid = add_direction_labels(summarize_groups(residuals, ["variable"], ["residual"]))
        resid_by_match = add_direction_labels(summarize_groups(residuals, ["match_id", "variable"], ["residual"]))
        pooled_resid.to_csv(output_dir / "pooled_residual_skewness.csv", index=False)
        resid_by_match.to_csv(output_dir / "residual_skewness_by_match_variable.csv", index=False)

    if not profiles.empty:
        profile_value_skew = add_direction_labels(
            summarize_groups(profiles, ["match_id", "variable"], ["profile_value"])
        )
        profile_value_skew.to_csv(output_dir / "profile_value_skewness_by_match_variable.csv", index=False)

    # Tail-stress and FE-relevant diagnostics.
    tail_df = summarize_tail_stress_by_match(panel)
    tail_summary = summarize_across_matches(tail_df)
    tail_df.to_csv(output_dir / "historical_tail_stress_by_match.csv", index=False)
    tail_summary.to_csv(output_dir / "historical_tail_stress_summary.csv", index=False)

    if args.no_fe_proxies:
        fe_proxy_df, fe_gap_df = pd.DataFrame(), pd.DataFrame()
    else:
        fe_proxy_df, fe_gap_df = build_fe_proxy_outputs(panel, availability=args.availability)
        if not fe_proxy_df.empty:
            fe_proxy_df.to_csv(output_dir / "fe_proxy_skewness.csv", index=False)
        if not fe_gap_df.empty:
            fe_gap_df.to_csv(output_dir / "fe_proxy_asg_vs_asc_gap.csv", index=False)

    if args.make_plots:
        make_plots(panel, output_dir, sample_rows=args.plot_sample_rows)

    write_summary_report(
        output_dir=output_dir,
        match_dir=match_dir,
        panel=panel,
        pooled_raw=pooled_raw,
        by_match_raw=by_match_raw,
        pooled_resid=pooled_resid,
        tail_df=tail_df,
        fe_proxy_df=fe_proxy_df,
    )
    write_run_summary(output_dir, match_dir, panel, args)

    print("\nSaved outputs to:", output_dir)
    for name in [
        "historical_skewness_report.md",
        "pooled_variable_skewness.csv",
        "skewness_by_match_variable.csv",
        "skewness_by_match_quarter_variable.csv",
        "skewness_by_match_quarter_hour_variable.csv",
        "pooled_residual_skewness.csv",
        "residual_skewness_by_match_variable.csv",
        "historical_tail_stress_by_match.csv",
        "historical_tail_stress_summary.csv",
        "fe_proxy_skewness.csv",
        "fe_proxy_asg_vs_asc_gap.csv",
        "skewness_run_summary.json",
    ]:
        p = output_dir / name
        if p.exists():
            print(" -", p)


if __name__ == "__main__":
    main()
