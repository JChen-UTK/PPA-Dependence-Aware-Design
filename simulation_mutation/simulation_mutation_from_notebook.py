#!/usr/bin/env python
# coding: utf-8

# # Calibrated mutation simulation from generated samples
# 
# Run `mutation_sample_generation.ipynb` first. This notebook reads:
# 
# ```text
# Code_Submission/simulation_mutation/mutation_samples/Sample_File_Index.csv
# ```
# 
# and writes PPA simulation outputs to:
# 
# ```text
# Code_Submission/simulation_mutation/Output files (Risk Neutral, Mutation, Verified)
# ```
# 
# It reuses the verified contract-evaluation logic from the uploaded simulation notebook, but the final execution loop reads saved calibrated samples instead of mutating samples on the fly.

# In[1]:


from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from datetime import datetime
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

_original_print = print


def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return _original_print(f"[{timestamp}]", *args, **kwargs)


# In[2]:


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


# In[3]:


# ============================================================
# Main folders
# ============================================================
# Folder that contains the historical match files created by the extraction step.
match_folder_name = "Match files of original data"

# Folder that contains the simplified no-mutation sample bank from Code 3.
sample_root_dir = "Simplified baseline realized samples"

# Optional match-table workbook. The notebook still runs if this file is absent.
match_table_file = "Seller_Buyer_Match_Table.xlsx"
match_table_sheet = "All_Matches"

# Optional Code 2 correlation bank. If present, the shrunk correlation targets
# for (generation, demand) and (seller_lmp, buyer_lmp_out) are merged into the
# plotting-ready summary table.
code2_correlation_file = Path("Input data and files") / "Code2_Correlation" / "correlation_parameter_bank_long.csv"

# Output folder for all per-match CSV files and all-match summary CSV files.
output_dir = "result/Output files (Risk Neutral, No Mutation, Verified)"

# ============================================================
# Risk preferences / contract settings
# ============================================================
# Use 0 for risk-neutral. Positive values make the party more risk-averse
# under the contract-period financial-exposure convention.
# 0.253~60%, 0.524~70%, 0.842~80%, 1.282~90%, 1.645~95%
lambda_s = 0.0
lambda_b = 0.0

# As-Generated guaranteed fraction. The penalty price is calibrated as the
# mean buyer_lmp_in over the retained scenario pool; no scale factor is used.
availability_factor = 0.95

# ============================================================
# Optimization grids
# ============================================================
strike_prices = np.arange(0.0, 600.0, 5.0)
fixed_volumes = np.arange(0.0, 20000.0, 20.0)

# Batch size used for Physical-Fix buyer calculations.
fixed_volume_chunk_size = 50

# ============================================================
# Verification / targeted repair controls
# ============================================================
# Keep the fast formula search for the full grid, then verify the selected
# candidates with exact FE paths before writing outputs.
verification_enabled = True

# Exact check tolerances for the formula winner.
verification_abs_tolerance = 1e-6
verification_rel_tolerance = 1e-6

# Evaluate all AsG / AsC / No-Contract candidates exactly, because that search
# space is small. Fix contracts are repaired with local exact search starting
# from the top formula seeds.
verification_exact_all_nonfix = True
verification_top_k_fix_seeds = 8
verification_fix_price_radius_steps = 1
verification_fix_volume_radius_steps = 1
verification_fix_max_iterations = 20
verification_fix_allow_cross_ppa_type = False

# ============================================================
# Batch controls
# ============================================================
# Use None to run every match found in the historical-match folder.
selected_match_ids = None

# If True, skip a match when its per-match outputs already exist.
skip_existing_output = False

# If True, save the full contract grid for each match.
save_per_match_grid = False

# If True, save the selected FE source for each match.
# The existing Hourly_FE output path is retained for compatibility, but rows
# are now contract-period financial exposures by replication.
save_hourly_fe = True

# If the final decision is No Contract, still save contract-period FE for the best PPA
# candidate (feasible if available, otherwise best infeasible PPA).
save_best_fe_even_if_infeasible = True

# Legacy option retained for path/output compatibility. Contract-period FE
# exports do not use timestamps.
save_timestamp_in_hourly_fe = False

# Scenario label used in the outputs.
scenario_name = "Baseline__No_Mutation__Verified"

# ============================================================
# Category defaults used for the plotting-ready summary table
# ============================================================
shape_corr_low = -0.30
shape_corr_high = 0.30
basis_corr_low = -0.30
basis_corr_high = 0.30
level_balance_tolerance = 0.05

# ============================================================
# Optional timestamp fallback for FE reconstruction later
# ============================================================
simulation_start = "2025-01-01 00:00:00"

CONFIG = {
    "match_folder_name": match_folder_name,
    "sample_root_dir": sample_root_dir,
    "match_table_file": match_table_file,
    "match_table_sheet": match_table_sheet,
    "code2_correlation_file": str(code2_correlation_file),
    "output_dir": output_dir,
    "lambda_s": float(lambda_s),
    "lambda_b": float(lambda_b),
    "availability_factor": float(availability_factor),
    "strike_prices": np.asarray(strike_prices, dtype=float),
    "fixed_volumes": np.asarray(fixed_volumes, dtype=float),
    "fixed_volume_chunk_size": int(fixed_volume_chunk_size),
    "verification_enabled": bool(verification_enabled),
    "verification_abs_tolerance": float(verification_abs_tolerance),
    "verification_rel_tolerance": float(verification_rel_tolerance),
    "verification_exact_all_nonfix": bool(verification_exact_all_nonfix),
    "verification_top_k_fix_seeds": int(verification_top_k_fix_seeds),
    "verification_fix_price_radius_steps": int(verification_fix_price_radius_steps),
    "verification_fix_volume_radius_steps": int(verification_fix_volume_radius_steps),
    "verification_fix_max_iterations": int(verification_fix_max_iterations),
    "verification_fix_allow_cross_ppa_type": bool(verification_fix_allow_cross_ppa_type),
    "selected_match_ids": None if selected_match_ids is None else [int(x) for x in selected_match_ids],
    "skip_existing_output": bool(skip_existing_output),
    "save_per_match_grid": bool(save_per_match_grid),
    "save_hourly_fe": bool(save_hourly_fe),
    "save_best_fe_even_if_infeasible": bool(save_best_fe_even_if_infeasible),
    "save_timestamp_in_hourly_fe": bool(save_timestamp_in_hourly_fe),
    "scenario_name": str(scenario_name),
    "shape_corr_low": float(shape_corr_low),
    "shape_corr_high": float(shape_corr_high),
    "basis_corr_low": float(basis_corr_low),
    "basis_corr_high": float(basis_corr_high),
    "level_balance_tolerance": float(level_balance_tolerance),
    "simulation_start": str(simulation_start),
}
CONFIG


# In[4]:


# ============================================================
# Path resolution and input loading
# ============================================================
def _clean_code(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    return text


def resolve_existing_dir(*candidates) -> Path:
    checked = []
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser()
        checked.append(path)
        if path.exists() and path.is_dir():
            return path.resolve()
    raise FileNotFoundError(
        "Could not resolve an existing directory. Tried:\n" +
        "\n".join(str(p) for p in checked)
    )


def resolve_existing_file(*candidates) -> Optional[Path]:
    checked = []
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser()
        checked.append(path)
        if path.exists() and path.is_file():
            return path.resolve()
    return None


NOTEBOOK_CWD = Path.cwd().resolve()
BUNDLE_ROOT = NOTEBOOK_CWD.parent if NOTEBOOK_CWD.name == "simulation_mutation" else NOTEBOOK_CWD


def candidate_roots(*anchors, max_parent_depth: int = 4):
    roots = []
    seen = set()
    for anchor in anchors:
        if anchor is None:
            continue
        p = Path(anchor).expanduser()
        if p.suffix:
            p = p.parent
        for root in [p, *list(p.parents)[:max_parent_depth]]:
            key = str(root)
            if key not in seen:
                seen.add(key)
                roots.append(root)
    return roots


def resolve_match_dir(match_folder_name: str) -> Path:
    roots = candidate_roots(BUNDLE_ROOT, NOTEBOOK_CWD, Path("/mnt/data"))
    candidates = []
    for root in roots:
        candidates.extend([
            root / match_folder_name,
            root / "Input data and files" / match_folder_name,
            root / "Data and files" / match_folder_name,
        ])
    return resolve_existing_dir(*candidates)


def resolve_sample_root(sample_root_dir: str, match_dir: Path) -> Path:
    roots = candidate_roots(BUNDLE_ROOT, NOTEBOOK_CWD, match_dir.parent, Path("/mnt/data"))
    candidates = []
    for root in roots:
        candidates.extend([
            root / sample_root_dir,
            root / "Input data and files" / sample_root_dir,
            root / "Data and files" / sample_root_dir,
        ])
    return resolve_existing_dir(*candidates)


def resolve_match_table(match_table_file: str) -> Optional[Path]:
    roots = candidate_roots(BUNDLE_ROOT, NOTEBOOK_CWD, Path("/mnt/data"))
    candidates = []
    for root in roots:
        candidates.extend([
            root / match_table_file,
            root / "Input data and files" / match_table_file,
            root / "Data and files" / match_table_file,
        ])
    return resolve_existing_file(*candidates)


def resolve_code2_corr_file(raw_path: str, match_dir: Path) -> Optional[Path]:
    candidate = Path(raw_path)
    roots = candidate_roots(BUNDLE_ROOT, NOTEBOOK_CWD, match_dir.parent, Path("/mnt/data"))
    candidates = []
    for root in roots:
        candidates.append(root / candidate)
    return resolve_existing_file(*candidates)


def resolve_output_root_dir(raw_path: str | Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (BUNDLE_ROOT / candidate).resolve()


def extract_match_id(path_like) -> int:
    path = Path(path_like)
    matched = re.search(r"(\d+)", path.stem)
    if matched is None:
        raise ValueError(f"Could not parse match_id from file name: {path.name}")
    return int(matched.group(1))


def load_match_table(match_table_path: Optional[Path], sheet_name: str = "All_Matches") -> pd.DataFrame:
    if match_table_path is None or not Path(match_table_path).exists():
        return pd.DataFrame()

    xls = pd.ExcelFile(match_table_path)
    if sheet_name in xls.sheet_names:
        df = read_excel_optimized(match_table_path, sheet_name=sheet_name)
    else:
        df = None
        for sh in xls.sheet_names:
            candidate = read_excel_optimized(match_table_path, sheet_name=sh)
            candidate.columns = [str(c).strip() for c in candidate.columns]
            if "match_id" in candidate.columns:
                df = candidate
                break
        if df is None:
            raise KeyError(f"Could not find a sheet with a 'match_id' column in {match_table_path}.")
    df.columns = [str(c).strip() for c in df.columns]
    if "match_id" in df.columns:
        df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["match_id"]).copy()
        df["match_id"] = df["match_id"].astype(int)
    return df


def discover_match_ids(match_dir: Path, selected_match_ids: Optional[Iterable[int]] = None) -> List[int]:
    match_ids = sorted({extract_match_id(path) for path in match_dir.glob("*.csv")})
    if selected_match_ids is not None:
        selected = {int(x) for x in selected_match_ids}
        match_ids = [mid for mid in match_ids if mid in selected]
    if not match_ids:
        raise ValueError("No matches were found under the historical match folder.")
    return match_ids


def load_original_match_df(match_id: int, match_dir: Path) -> pd.DataFrame:
    path = match_dir / f"{int(match_id):03d}.csv"
    if not path.exists():
        alt = list(match_dir.glob(f"*{int(match_id):03d}*.csv"))
        if alt:
            path = alt[0]
    if not path.exists():
        raise FileNotFoundError(f"Historical match file was not found for match_id={match_id}.")
    df = read_csv_optimized(path)
    required = ["timestamp", "hour", "quarter", "generation", "demand", "seller_lmp", "buyer_lmp_out", "buyer_lmp_in"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} is missing required columns: {missing}")
    return df


def discover_replication_dirs(sample_root: Path) -> List[Tuple[int, Path]]:
    rep_dirs = []
    for path in sample_root.iterdir():
        if path.is_dir():
            matched = re.search(r"rep_(\d+)", path.name, flags=re.IGNORECASE)
            if matched:
                rep_dirs.append((int(matched.group(1)), path))
    rep_dirs = sorted(rep_dirs, key=lambda x: x[0])
    if not rep_dirs:
        raise FileNotFoundError(
            f"No replication subfolders like rep_01, rep_02, ... were found under {sample_root}."
        )
    return rep_dirs


def load_baseline_sample_bank(match_id: int, sample_root: Path) -> pd.DataFrame:
    frames = []
    rep_dirs = discover_replication_dirs(sample_root)

    for rep_no, rep_dir in rep_dirs:
        candidate = rep_dir / f"{int(match_id):03d}.csv"
        if not candidate.exists():
            alternatives = list(rep_dir.glob(f"*{int(match_id):03d}*.csv"))
            if alternatives:
                candidate = alternatives[0]
        if not candidate.exists():
            continue

        df = read_csv_optimized(candidate)
        required = ["timestamp", "hour", "quarter", "generation", "demand", "seller_lmp", "buyer_lmp_out", "buyer_lmp_in"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"{candidate} is missing required columns: {missing}")

        df = df.copy()
        df["replication"] = int(rep_no)
        df["hour_index"] = np.arange(1, len(df) + 1, dtype=int)
        df["sample_file"] = str(candidate)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No sample files were found for match_id={match_id} under {sample_root}."
        )

    bank = pd.concat(frames, ignore_index=True)
    bank["replication"] = pd.to_numeric(bank["replication"], errors="coerce").astype(int)
    bank["hour_index"] = pd.to_numeric(bank["hour_index"], errors="coerce").astype(int)
    return bank


def load_code2_corr_targets(code2_corr_path: Optional[Path]) -> pd.DataFrame:
    if code2_corr_path is None or not Path(code2_corr_path).exists():
        return pd.DataFrame(columns=["match_id", "var_i", "var_j", "corr_target"])

    df = read_csv_optimized(code2_corr_path)
    required = {"match_id", "var_i", "var_j"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["match_id", "var_i", "var_j", "corr_target"])

    if "corr_target" not in df.columns:
        if "corr_shrunk" in df.columns:
            df = df.rename(columns={"corr_shrunk": "corr_target"})
        elif "corr" in df.columns:
            df = df.rename(columns={"corr": "corr_target"})
        else:
            return pd.DataFrame(columns=["match_id", "var_i", "var_j", "corr_target"])

    df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["match_id"]).copy()
    df["match_id"] = df["match_id"].astype(int)
    return df[["match_id", "var_i", "var_j", "corr_target"]].copy()


# In[5]:


# ============================================================
# Summary statistics, categories, and contract logic
# ============================================================
def safe_numeric(series) -> pd.Series:
    return pd.to_numeric(pd.Series(series), errors="coerce")


def safe_corr(a, b) -> float:
    a = safe_numeric(a)
    b = safe_numeric(b)
    valid = a.notna() & b.notna()
    if int(valid.sum()) < 2:
        return np.nan
    a_valid = a.loc[valid]
    b_valid = b.loc[valid]
    if a_valid.nunique() <= 1 or b_valid.nunique() <= 1:
        return np.nan
    return float(a_valid.corr(b_valid))


def summarize_series(series, prefix: str) -> Dict[str, float]:
    s = safe_numeric(series).dropna()
    if s.empty:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_q05": np.nan,
            f"{prefix}_q95": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_mean": float(s.mean()),
        f"{prefix}_median": float(s.median()),
        f"{prefix}_std": float(s.std(ddof=0)),
        f"{prefix}_q05": float(s.quantile(0.05)),
        f"{prefix}_q95": float(s.quantile(0.95)),
        f"{prefix}_min": float(s.min()),
        f"{prefix}_max": float(s.max()),
    }


def scalar_signed_index(left, right, use_abs_denominator: bool = False) -> float:
    left = float(left) if pd.notna(left) else np.nan
    right = float(right) if pd.notna(right) else np.nan
    if pd.isna(left) or pd.isna(right):
        return np.nan
    denominator = abs(left) + abs(right) if use_abs_denominator else left + right
    if denominator == 0:
        return np.nan
    return float((left - right) / denominator)


def classify_corr_regime(value: float, low: float, high: float,
                         label_high: str, label_mid: str, label_low: str) -> Tuple[float, Optional[str]]:
    if pd.isna(value):
        return np.nan, None
    if value >= high:
        return 1, label_high
    if value >= low:
        return 2, label_mid
    return 3, label_low


def classify_level_category(value: float, tol: float,
                            label_low: str, label_mid: str, label_high: str) -> Tuple[float, Optional[str]]:
    if pd.isna(value):
        return np.nan, None
    if value < -tol:
        return 1, label_low
    if value <= tol:
        return 2, label_mid
    return 3, label_high


def choose_preferred_value(original_value, simulated_value):
    if pd.notna(original_value):
        return original_value, "original"
    if pd.notna(simulated_value):
        return simulated_value, "simulated"
    return np.nan, None


def unusual_contracted_volume_flag(solution_row: pd.Series, sample_df: pd.DataFrame) -> str:
    q_opt = pd.to_numeric(pd.Series([solution_row.get("volume_mw", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(q_opt) or q_opt <= 0:
        return "No"
    g95 = float(pd.to_numeric(sample_df["generation"], errors="coerce").quantile(0.95))
    d95 = float(pd.to_numeric(sample_df["demand"], errors="coerce").quantile(0.95))
    return "Yes" if (q_opt > 2.0 * g95 and q_opt > 2.0 * d95) else "No"


def build_basic_stats(match_id: int,
                      original_df: pd.DataFrame,
                      sample_df: pd.DataFrame,
                      corr_targets: pd.DataFrame,
                      config: Dict) -> Dict[str, object]:
    out: Dict[str, object] = {
        "match_id": int(match_id),
        "n_original_rows": int(len(original_df)),
        "n_sample_rows": int(len(sample_df)),
        "n_replications": int(sample_df["replication"].nunique()),
        "hours_per_replication": int(sample_df.groupby("replication")["hour_index"].max().median()),
    }

    original_map = {
        "generation": "generation",
        "demand": "demand",
        "seller_lmp": "seller_lmp",
        "buyer_lmp_out": "buyer_lmp_out",
        "buyer_lmp_in": "buyer_lmp_in",
    }
    simulated_map = original_map.copy()

    for var, col in original_map.items():
        out.update(summarize_series(original_df[col], f"{var}_original"))
    for var, col in simulated_map.items():
        out.update(summarize_series(sample_df[col], f"{var}_simulated"))

    out["shape_corr_original"] = safe_corr(original_df["generation"], original_df["demand"])
    out["basis_corr_original"] = safe_corr(original_df["seller_lmp"], original_df["buyer_lmp_out"])
    out["shape_corr_simulated"] = safe_corr(sample_df["generation"], sample_df["demand"])
    out["basis_corr_simulated"] = safe_corr(sample_df["seller_lmp"], sample_df["buyer_lmp_out"])

    def _lookup_target(match_id: int, v1: str, v2: str) -> float:
        if corr_targets.empty:
            return np.nan
        mask = (
            (corr_targets["match_id"].astype(int) == int(match_id)) &
            (
                ((corr_targets["var_i"] == v1) & (corr_targets["var_j"] == v2)) |
                ((corr_targets["var_i"] == v2) & (corr_targets["var_j"] == v1))
            )
        )
        sub = corr_targets.loc[mask, "corr_target"]
        if sub.empty:
            return np.nan
        return float(pd.to_numeric(sub, errors="coerce").dropna().iloc[0])

    out["shape_corr_target"] = _lookup_target(match_id, "generation", "demand")
    out["basis_corr_target"] = _lookup_target(match_id, "seller_lmp", "buyer_lmp_out")

    for metric in ["shape_corr", "basis_corr"]:
        val, src = choose_preferred_value(out.get(f"{metric}_original", np.nan), out.get(f"{metric}_simulated", np.nan))
        out[f"{metric}_ref"] = val
        out[f"{metric}_ref_source"] = src

    for metric in ["generation", "demand", "seller_lmp", "buyer_lmp_out"]:
        for stat in ["mean", "median"]:
            val, src = choose_preferred_value(out.get(f"{metric}_original_{stat}", np.nan), out.get(f"{metric}_simulated_{stat}", np.nan))
            out[f"{metric}_{stat}_ref"] = val
            out[f"{metric}_{stat}_ref_source"] = src

    out["volume_mismatch_mean_original"] = scalar_signed_index(
        out["generation_original_mean"], out["demand_original_mean"], use_abs_denominator=False
    )
    out["price_spread_mean_original"] = scalar_signed_index(
        out["seller_lmp_original_mean"], out["buyer_lmp_out_original_mean"], use_abs_denominator=True
    )
    out["volume_mismatch_median_original"] = scalar_signed_index(
        out["generation_original_median"], out["demand_original_median"], use_abs_denominator=False
    )
    out["price_spread_median_original"] = scalar_signed_index(
        out["seller_lmp_original_median"], out["buyer_lmp_out_original_median"], use_abs_denominator=True
    )

    out["volume_mismatch_mean_simulated"] = scalar_signed_index(
        out["generation_simulated_mean"], out["demand_simulated_mean"], use_abs_denominator=False
    )
    out["price_spread_mean_simulated"] = scalar_signed_index(
        out["seller_lmp_simulated_mean"], out["buyer_lmp_out_simulated_mean"], use_abs_denominator=True
    )
    out["volume_mismatch_median_simulated"] = scalar_signed_index(
        out["generation_simulated_median"], out["demand_simulated_median"], use_abs_denominator=False
    )
    out["price_spread_median_simulated"] = scalar_signed_index(
        out["seller_lmp_simulated_median"], out["buyer_lmp_out_simulated_median"], use_abs_denominator=True
    )

    for metric in ["volume_mismatch_mean", "price_spread_mean", "volume_mismatch_median", "price_spread_median"]:
        val, src = choose_preferred_value(out.get(f"{metric}_original", np.nan), out.get(f"{metric}_simulated", np.nan))
        out[f"{metric}_ref"] = val
        out[f"{metric}_ref_source"] = src

    out["shape_regime"], out["shape_regime_label"] = classify_corr_regime(
        out["shape_corr_ref"],
        config["shape_corr_low"],
        config["shape_corr_high"],
        "Coincident profile",
        "Weakly coupled profile",
        "Mismatched profile",
    )
    out["basis_regime"], out["basis_regime_label"] = classify_corr_regime(
        out["basis_corr_ref"],
        config["basis_corr_low"],
        config["basis_corr_high"],
        "Convergent market",
        "Moderately divergent market",
        "Highly divergent market",
    )
    out["volume_mean_cat"], out["volume_mean_cat_label"] = classify_level_category(
        out["volume_mismatch_mean_ref"],
        config["level_balance_tolerance"],
        "Seller < Buyer",
        "Balanced",
        "Seller > Buyer",
    )
    out["price_mean_cat"], out["price_mean_cat_label"] = classify_level_category(
        out["price_spread_mean_ref"],
        config["level_balance_tolerance"],
        "Seller < Buyer",
        "Balanced",
        "Seller > Buyer",
    )
    out["volume_median_cat"], out["volume_median_cat_label"] = classify_level_category(
        out["volume_mismatch_median_ref"],
        config["level_balance_tolerance"],
        "Seller < Buyer",
        "Balanced",
        "Seller > Buyer",
    )
    out["price_median_cat"], out["price_median_cat_label"] = classify_level_category(
        out["price_spread_median_ref"],
        config["level_balance_tolerance"],
        "Seller < Buyer",
        "Balanced",
        "Seller > Buyer",
    )

    out["combined_category"] = (
        f"S{int(out['shape_regime']) if pd.notna(out['shape_regime']) else 'NA'}_"
        f"B{int(out['basis_regime']) if pd.notna(out['basis_regime']) else 'NA'}_"
        f"Vm{int(out['volume_mean_cat']) if pd.notna(out['volume_mean_cat']) else 'NA'}_"
        f"Pm{int(out['price_mean_cat']) if pd.notna(out['price_mean_cat']) else 'NA'}_"
        f"Vd{int(out['volume_median_cat']) if pd.notna(out['volume_median_cat']) else 'NA'}_"
        f"Pd{int(out['price_median_cat']) if pd.notna(out['price_median_cat']) else 'NA'}"
    )

    return out


def compute_penalty_rate(sample_df: pd.DataFrame) -> Tuple[float, float]:
    """Calibrate the As-Generated contract-period penalty price.

    The revised formulation uses a fixed penalty price equal to the estimated
    contract-period average buyer purchase price, not a scaled tail
    quantile. The first returned value is the penalty price used in settlements;
    the second is retained as a named calibration diagnostic.
    """
    buyer_purchase_mean = float(pd.to_numeric(sample_df["buyer_lmp_in"], errors="coerce").mean())
    return buyer_purchase_mean, buyer_purchase_mean


def _replication_starts_from_ids(replication_ids) -> np.ndarray:
    ids = np.asarray(replication_ids)
    if ids.size == 0:
        raise ValueError("No replication observations are available.")
    changes = np.ones(ids.size, dtype=bool)
    changes[1:] = ids[1:] != ids[:-1]
    return np.flatnonzero(changes).astype(int)


def _sum_by_replication(values, replication_starts: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    starts = np.asarray(replication_starts, dtype=int)
    if arr.shape[0] == 0:
        raise ValueError("Cannot aggregate an empty array by replication.")
    if starts.size == 0:
        raise ValueError("replication_starts is empty.")
    return np.add.reduceat(arr, starts, axis=0)


def _replication_ddof(n_replications: int) -> int:
    return 1 if int(n_replications) > 1 else 0


def _contract_period_asg_penalty_allocation(g, replication_starts: np.ndarray, availability_factor: float, penalty_rate: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return row-level allocation of the once-per-replication AsG penalty.

    The penalty is computed on contract-period generation for each replication.
    For compatibility with existing row-level exports, the realized penalty is
    allocated to the first row of each replication; summing rows by replication
    exactly recovers the contract-period exposure.
    """
    g = np.asarray(g, dtype=float)
    generation_totals = _sum_by_replication(g, replication_starts)
    expected_generation_total = float(np.mean(generation_totals))
    penalty_by_replication = float(penalty_rate) * np.maximum(
        0.0,
        float(availability_factor) * expected_generation_total - generation_totals,
    )
    allocation = np.zeros_like(g, dtype=float)
    allocation[np.asarray(replication_starts, dtype=int)] = penalty_by_replication
    return allocation, penalty_by_replication, expected_generation_total


def _period_block_stats(fe_s_block, fe_b_block, rev_s_block, lambda_s: float, lambda_b: float, replication_starts: np.ndarray) -> Dict[str, np.ndarray]:
    seller_rep = _sum_by_replication(fe_s_block, replication_starts)
    buyer_rep = _sum_by_replication(fe_b_block, replication_starts)
    revenue_rep = _sum_by_replication(rev_s_block, replication_starts)
    ddof = _replication_ddof(seller_rep.shape[0])

    seller_mean = seller_rep.mean(axis=0)
    buyer_mean = buyer_rep.mean(axis=0)
    seller_var = seller_rep.var(axis=0, ddof=ddof)
    buyer_var = buyer_rep.var(axis=0, ddof=ddof)
    seller_std = np.sqrt(seller_var)
    buyer_std = np.sqrt(buyer_var)

    return {
        "seller_utility": seller_mean + float(lambda_s) * seller_std,
        "buyer_utility": buyer_mean + float(lambda_b) * buyer_std,
        "seller_mean_exposure": seller_mean,
        "buyer_mean_exposure": buyer_mean,
        "seller_exposure_variance": seller_var,
        "buyer_exposure_variance": buyer_var,
        "expected_seller_revenue": revenue_rep.mean(axis=0),
    }


def evaluate_contract(g, d, Ns, Nb_out, Nb_in, mu, nu, pi, q_fixed,
                      lambda_s, lambda_b, availability_factor, penalty_rate, replication_starts):
    T = len(g)

    if mu != "Physical":
        raise ValueError("Virtual PPA candidates are excluded by the revised physical-PPA formulation.")

    if nu == "Fix":
        q_del = np.full(T, q_fixed, dtype=float)
    elif nu == "AsG":
        q_del = np.asarray(g, dtype=float).copy()
    elif nu == "AsC":
        q_del = np.asarray(d, dtype=float).copy()
    else:
        raise ValueError(f"Unknown volume structure: {nu}")

    if nu == "AsG":
        Pen_t, _, _ = _contract_period_asg_penalty_allocation(
            g=g,
            replication_starts=replication_starts,
            availability_factor=availability_factor,
            penalty_rate=penalty_rate,
        )
    else:
        Pen_t = np.zeros(T, dtype=float)

    PPA_t = float(pi) * q_del
    f_s = Ns * (q_del - g) if nu in ["Fix", "AsC"] else np.zeros(T, dtype=float)
    f_b = Nb_in * np.maximum(0.0, d - q_del) - Nb_out * np.maximum(0.0, q_del - d)

    Rev_s = PPA_t - f_s - Pen_t
    FE_s = Ns * q_del - PPA_t + Pen_t
    FE_b = PPA_t + f_b - Nb_in * d - Pen_t

    stats = _summary_from_arrays(
        fe_s=FE_s,
        fe_b=FE_b,
        rev_s=Rev_s,
        lambda_s=lambda_s,
        lambda_b=lambda_b,
        replication_starts=replication_starts,
    )

    return {
        "q_del": q_del,
        "Pen_t": Pen_t,
        "PPA_t": PPA_t,
        "f_s": f_s,
        "f_b": f_b,
        "Rev_s": Rev_s,
        "FE_s": FE_s,
        "FE_b": FE_b,
        "U_s": stats["seller_utility"],
        "U_b": stats["buyer_utility"],
    }


def evaluate_no_contract(g, d, Ns, Nb_out, Nb_in, lambda_s, lambda_b, replication_starts):
    T = len(g)
    out = {
        "q_del": np.zeros(T, dtype=float),
        "Pen_t": np.zeros(T, dtype=float),
        "PPA_t": np.zeros(T, dtype=float),
        "f_s": -Ns * g,
        "f_b": Nb_in * d,
        "Rev_s": Ns * g,
        "FE_s": np.zeros(T, dtype=float),
        "FE_b": np.zeros(T, dtype=float),
    }
    stats = _summary_from_arrays(
        out["FE_s"], out["FE_b"], out["Rev_s"],
        lambda_s=lambda_s, lambda_b=lambda_b, replication_starts=replication_starts,
    )
    out["U_s"] = stats["seller_utility"]
    out["U_b"] = stats["buyer_utility"]
    return out


def _summary_from_arrays(fe_s, fe_b, rev_s, lambda_s, lambda_b, replication_starts: Optional[np.ndarray] = None) -> Dict[str, float]:
    if replication_starts is None:
        replication_starts = np.asarray([0], dtype=int)
    stats = _period_block_stats(
        fe_s_block=np.asarray(fe_s, dtype=float),
        fe_b_block=np.asarray(fe_b, dtype=float),
        rev_s_block=np.asarray(rev_s, dtype=float),
        lambda_s=lambda_s,
        lambda_b=lambda_b,
        replication_starts=replication_starts,
    )
    return {key: float(np.asarray(value)) for key, value in stats.items()}


def _build_block_df_nonfix(ppa_type: str,
                           profile_type: str,
                           strike_prices_block: np.ndarray,
                           fe_s_block: np.ndarray,
                           fe_b_block: np.ndarray,
                           rev_s_block: np.ndarray,
                           lambda_s: float,
                           lambda_b: float,
                           replication_starts: np.ndarray) -> pd.DataFrame:
    stats = _period_block_stats(fe_s_block, fe_b_block, rev_s_block, lambda_s, lambda_b, replication_starts)
    return pd.DataFrame({
        "ppa_type": ppa_type,
        "profile_type": profile_type,
        "volume_mw": np.nan,
        "strike_price_mwh": strike_prices_block.astype(float),
        "seller_utility": np.asarray(stats["seller_utility"]).astype(float),
        "buyer_utility": np.asarray(stats["buyer_utility"]).astype(float),
        "seller_mean_exposure": np.asarray(stats["seller_mean_exposure"]).astype(float),
        "buyer_mean_exposure": np.asarray(stats["buyer_mean_exposure"]).astype(float),
        "seller_exposure_variance": np.asarray(stats["seller_exposure_variance"]).astype(float),
        "buyer_exposure_variance": np.asarray(stats["buyer_exposure_variance"]).astype(float),
        "expected_seller_revenue": np.asarray(stats["expected_seller_revenue"]).astype(float),
        "feasible": (np.asarray(stats["buyer_utility"]) <= 0.0),
        "decision_metric": "contract_period_grid",
    })


def _build_block_df_fix(ppa_type: str,
                        volume_block: np.ndarray,
                        strike_prices_block: np.ndarray,
                        fe_s_block: np.ndarray,
                        fe_b_block: np.ndarray,
                        rev_s_block: np.ndarray,
                        lambda_s: float,
                        lambda_b: float,
                        replication_starts: np.ndarray) -> pd.DataFrame:
    stats = _period_block_stats(fe_s_block, fe_b_block, rev_s_block, lambda_s, lambda_b, replication_starts)
    vol_grid, strike_grid = np.meshgrid(volume_block.astype(float), strike_prices_block.astype(float), indexing="ij")
    return pd.DataFrame({
        "ppa_type": ppa_type,
        "profile_type": "Fix",
        "volume_mw": vol_grid.reshape(-1),
        "strike_price_mwh": strike_grid.reshape(-1),
        "seller_utility": np.asarray(stats["seller_utility"]).reshape(-1).astype(float),
        "buyer_utility": np.asarray(stats["buyer_utility"]).reshape(-1).astype(float),
        "seller_mean_exposure": np.asarray(stats["seller_mean_exposure"]).reshape(-1).astype(float),
        "buyer_mean_exposure": np.asarray(stats["buyer_mean_exposure"]).reshape(-1).astype(float),
        "seller_exposure_variance": np.asarray(stats["seller_exposure_variance"]).reshape(-1).astype(float),
        "buyer_exposure_variance": np.asarray(stats["buyer_exposure_variance"]).reshape(-1).astype(float),
        "expected_seller_revenue": np.asarray(stats["expected_seller_revenue"]).reshape(-1).astype(float),
        "feasible": (np.asarray(stats["buyer_utility"]).reshape(-1) <= 0.0),
        "decision_metric": "contract_period_grid",
    })


def optimize_ppa_contracts(sample_df: pd.DataFrame, config: Dict, penalty_rate: float) -> pd.DataFrame:
    arrays = prepare_sample_arrays(sample_df)
    g = arrays["g"]
    d = arrays["d"]
    Ns = arrays["Ns"]
    Nb_out = arrays["Nb_out"]
    Nb_in = arrays["Nb_in"]
    replication_starts = arrays["replication_starts"]

    strike_prices = np.asarray(config["strike_prices"], dtype=float)
    fixed_volumes = np.asarray([float(q) for q in config["fixed_volumes"] if np.isfinite(q) and float(q) > 0.0], dtype=float)
    lambda_s = float(config["lambda_s"])
    lambda_b = float(config["lambda_b"])
    availability_factor = float(config["availability_factor"])
    chunk_size = max(1, int(config["fixed_volume_chunk_size"]))

    frames: List[pd.DataFrame] = []

    no_contract = evaluate_no_contract(
        g, d, Ns, Nb_out, Nb_in,
        lambda_s=lambda_s,
        lambda_b=lambda_b,
        replication_starts=replication_starts,
    )
    no_contract_stats = _summary_from_arrays(
        no_contract["FE_s"], no_contract["FE_b"], no_contract["Rev_s"],
        lambda_s=lambda_s, lambda_b=lambda_b, replication_starts=replication_starts,
    )
    frames.append(pd.DataFrame([{
        "ppa_type": "No Contract",
        "profile_type": "N/A",
        "volume_mw": np.nan,
        "strike_price_mwh": np.nan,
        **no_contract_stats,
        "feasible": True,
        "decision_metric": "contract_period_grid",
    }]))

    penalty_asg, _, _ = _contract_period_asg_penalty_allocation(
        g=g,
        replication_starts=replication_starts,
        availability_factor=availability_factor,
        penalty_rate=penalty_rate,
    )

    Ns_col = Ns[:, None]
    g_col = g[:, None]
    d_col = d[:, None]
    nb_in_col = Nb_in[:, None]
    nb_out_col = Nb_out[:, None]
    penalty_col = penalty_asg[:, None]
    revenue_no_contract = (Ns * g)[:, None]

    for profile_type in ["AsG", "AsC"]:
        pi_row = strike_prices[None, :]
        if profile_type == "AsG":
            fe_s = g_col * (Ns_col - pi_row) + penalty_col
            fe_b = (
                pi_row * g_col
                + nb_in_col * np.maximum(d - g, 0.0)[:, None]
                - nb_out_col * np.maximum(g - d, 0.0)[:, None]
                - nb_in_col * d_col
                - penalty_col
            )
            rev_s = pi_row * g_col - penalty_col
        else:
            fe_s = d_col * (Ns_col - pi_row)
            fe_b = d_col * (pi_row - nb_in_col)
            rev_s = revenue_no_contract + d_col * (pi_row - Ns_col)

        frames.append(
            _build_block_df_nonfix(
                ppa_type="Physical",
                profile_type=profile_type,
                strike_prices_block=strike_prices,
                fe_s_block=fe_s,
                fe_b_block=fe_b,
                rev_s_block=rev_s,
                lambda_s=lambda_s,
                lambda_b=lambda_b,
                replication_starts=replication_starts,
            )
        )

    Ns_3d = Ns[:, None, None]
    d_3d = d[:, None, None]
    nb_in_3d = Nb_in[:, None, None]
    nb_out_3d = Nb_out[:, None, None]
    revenue_no_contract_3d = (Ns * g)[:, None, None]

    for start in range(0, len(fixed_volumes), chunk_size):
        volume_block = fixed_volumes[start:start + chunk_size]
        q_3d = volume_block[None, :, None]
        pi_3d = strike_prices[None, None, :]

        fe_s = q_3d * (Ns_3d - pi_3d)
        fe_b = (
            q_3d * pi_3d
            + nb_in_3d * np.maximum(d_3d - q_3d, 0.0)
            - nb_out_3d * np.maximum(q_3d - d_3d, 0.0)
            - nb_in_3d * d_3d
        )
        rev_s = revenue_no_contract_3d + q_3d * (pi_3d - Ns_3d)

        frames.append(
            _build_block_df_fix(
                ppa_type="Physical",
                volume_block=volume_block,
                strike_prices_block=strike_prices,
                fe_s_block=fe_s,
                fe_b_block=fe_b,
                rev_s_block=rev_s,
                lambda_s=lambda_s,
                lambda_b=lambda_b,
                replication_starts=replication_starts,
            )
        )

    return pd.concat(frames, ignore_index=True)


def get_optimal_solution(df_results: pd.DataFrame) -> pd.Series:
    feasible_df = df_results.loc[df_results["feasible"] == True].copy()
    if feasible_df.empty:
        idx = df_results["buyer_utility"].idxmin()
    else:
        idx = feasible_df["seller_utility"].idxmin()
    out = df_results.loc[idx].copy()
    out["best_solution"] = True
    return out


def get_best_ppa_solution(df_results: pd.DataFrame) -> Optional[pd.Series]:
    ppa_df = df_results.loc[df_results["ppa_type"].astype(str) != "No Contract"].copy()
    if ppa_df.empty:
        return None
    feasible_ppa_df = ppa_df.loc[ppa_df["feasible"] == True].copy()
    if feasible_ppa_df.empty:
        idx = ppa_df["buyer_utility"].idxmin()
    else:
        idx = feasible_ppa_df["seller_utility"].idxmin()
    out = ppa_df.loc[idx].copy()
    out["best_ppa_solution"] = True
    return out



# ============================================================
# Exact verification and targeted repair
# ============================================================
def prepare_sample_arrays(sample_df: pd.DataFrame) -> Dict[str, object]:
    flat_df = sample_df.sort_values(["replication", "hour_index"]).reset_index(drop=True)
    replication_ids = pd.to_numeric(flat_df["replication"], errors="coerce").to_numpy()
    replication_starts = _replication_starts_from_ids(replication_ids)
    return {
        "flat_df": flat_df,
        "replication_ids": replication_ids,
        "replication_starts": replication_starts,
        "n_replications": int(len(replication_starts)),
        "g": pd.to_numeric(flat_df["generation"], errors="coerce").to_numpy(dtype=float),
        "d": pd.to_numeric(flat_df["demand"], errors="coerce").to_numpy(dtype=float),
        "Ns": pd.to_numeric(flat_df["seller_lmp"], errors="coerce").to_numpy(dtype=float),
        "Nb_out": pd.to_numeric(flat_df["buyer_lmp_out"], errors="coerce").to_numpy(dtype=float),
        "Nb_in": pd.to_numeric(flat_df["buyer_lmp_in"], errors="coerce").to_numpy(dtype=float),
    }


def _candidate_signature(row: pd.Series) -> Tuple[str, str, Optional[float], Optional[float]]:
    ppa_type = str(row.get("ppa_type", ""))
    profile_type = str(row.get("profile_type", ""))
    volume = pd.to_numeric(pd.Series([row.get("volume_mw", np.nan)]), errors="coerce").iloc[0]
    strike = pd.to_numeric(pd.Series([row.get("strike_price_mwh", np.nan)]), errors="coerce").iloc[0]
    volume_key = None if pd.isna(volume) else round(float(volume), 10)
    strike_key = None if pd.isna(strike) else round(float(strike), 10)
    return (ppa_type, profile_type, volume_key, strike_key)


def _candidate_signature_str(row: pd.Series) -> str:
    ppa_type, profile_type, volume_key, strike_key = _candidate_signature(row)
    return f"{ppa_type}|{profile_type}|{volume_key}|{strike_key}"


def _candidate_signature_from_parts(ppa_type: str,
                                    profile_type: str,
                                    volume_mw,
                                    strike_price_mwh) -> Tuple[str, str, Optional[float], Optional[float]]:
    volume = pd.to_numeric(pd.Series([volume_mw]), errors="coerce").iloc[0]
    strike = pd.to_numeric(pd.Series([strike_price_mwh]), errors="coerce").iloc[0]
    volume_key = None if pd.isna(volume) else round(float(volume), 10)
    strike_key = None if pd.isna(strike) else round(float(strike), 10)
    return (str(ppa_type), str(profile_type), volume_key, strike_key)


def _tolerance_ok(formula_value, exact_value, abs_tol: float, rel_tol: float) -> bool:
    formula_value = float(formula_value) if pd.notna(formula_value) else np.nan
    exact_value = float(exact_value) if pd.notna(exact_value) else np.nan
    if pd.isna(formula_value) and pd.isna(exact_value):
        return True
    if pd.isna(formula_value) or pd.isna(exact_value):
        return False
    bound = abs_tol + rel_tol * max(1.0, abs(formula_value))
    return bool(abs(exact_value - formula_value) <= bound)


def compare_formula_vs_exact(formula_row: pd.Series,
                             exact_row: pd.Series,
                             abs_tol: float,
                             rel_tol: float) -> Dict[str, object]:
    metrics = [
        "seller_utility",
        "buyer_utility",
        "seller_mean_exposure",
        "buyer_mean_exposure",
        "seller_exposure_variance",
        "buyer_exposure_variance",
        "expected_seller_revenue",
    ]
    out: Dict[str, object] = {}
    metric_passes = []
    for metric in metrics:
        formula_value = pd.to_numeric(pd.Series([formula_row.get(metric, np.nan)]), errors="coerce").iloc[0]
        exact_value = pd.to_numeric(pd.Series([exact_row.get(metric, np.nan)]), errors="coerce").iloc[0]
        abs_diff = np.nan if (pd.isna(formula_value) or pd.isna(exact_value)) else float(abs(exact_value - formula_value))
        rel_diff = np.nan
        if pd.notna(abs_diff):
            rel_diff = float(abs_diff / max(1.0, abs(formula_value)))
        out[f"{metric}_formula"] = formula_value
        out[f"{metric}_exact"] = exact_value
        out[f"{metric}_abs_diff"] = abs_diff
        out[f"{metric}_rel_diff"] = rel_diff
        metric_passes.append(_tolerance_ok(formula_value, exact_value, abs_tol=abs_tol, rel_tol=rel_tol))

    formula_feasible = bool(formula_row.get("feasible", False))
    exact_feasible = bool(exact_row.get("feasible", False))
    out["formula_feasible"] = formula_feasible
    out["exact_feasible"] = exact_feasible
    out["feasible_status_match"] = bool(formula_feasible == exact_feasible)
    out["verification_passed"] = bool(all(metric_passes) and out["feasible_status_match"])
    return out


def evaluate_candidate_exact_from_arrays(arrays: Dict[str, object],
                                         candidate_row: pd.Series,
                                         config: Dict,
                                         penalty_rate: float) -> Dict[str, object]:
    g = arrays["g"]
    d = arrays["d"]
    Ns = arrays["Ns"]
    Nb_out = arrays["Nb_out"]
    Nb_in = arrays["Nb_in"]

    ppa_type = str(candidate_row.get("ppa_type", "No Contract"))
    profile_type = str(candidate_row.get("profile_type", "N/A"))
    strike_price = pd.to_numeric(pd.Series([candidate_row.get("strike_price_mwh", np.nan)]), errors="coerce").iloc[0]
    volume_mw = pd.to_numeric(pd.Series([candidate_row.get("volume_mw", np.nan)]), errors="coerce").iloc[0]

    if ppa_type == "No Contract":
        out = evaluate_no_contract(
            g=g,
            d=d,
            Ns=Ns,
            Nb_out=Nb_out,
            Nb_in=Nb_in,
            lambda_s=config["lambda_s"],
            lambda_b=config["lambda_b"],
            replication_starts=arrays["replication_starts"],
        )
    else:
        out = evaluate_contract(
            g=g,
            d=d,
            Ns=Ns,
            Nb_out=Nb_out,
            Nb_in=Nb_in,
            mu=ppa_type,
            nu=profile_type,
            pi=0.0 if pd.isna(strike_price) else float(strike_price),
            q_fixed=0.0 if pd.isna(volume_mw) else float(volume_mw),
            lambda_s=config["lambda_s"],
            lambda_b=config["lambda_b"],
            availability_factor=config["availability_factor"],
            penalty_rate=penalty_rate,
            replication_starts=arrays["replication_starts"],
        )

    stats = _summary_from_arrays(
        fe_s=out["FE_s"],
        fe_b=out["FE_b"],
        rev_s=out["Rev_s"],
        lambda_s=config["lambda_s"],
        lambda_b=config["lambda_b"],
        replication_starts=arrays["replication_starts"],
    )
    exact = {
        "ppa_type": ppa_type,
        "profile_type": profile_type,
        "volume_mw": np.nan if pd.isna(volume_mw) else float(volume_mw),
        "strike_price_mwh": np.nan if pd.isna(strike_price) else float(strike_price),
        **stats,
        "feasible": bool(stats["buyer_utility"] <= 0.0),
        "decision_metric": "exact",
    }
    return exact


def evaluate_candidates_exact(candidate_df: pd.DataFrame,
                              arrays: Dict[str, object],
                              config: Dict,
                              penalty_rate: float,
                              exact_cache: Dict[Tuple[str, str, Optional[float], Optional[float]], Dict[str, object]]) -> pd.DataFrame:
    rows = []
    if candidate_df is None or candidate_df.empty:
        return pd.DataFrame()
    for _, row in candidate_df.iterrows():
        signature = _candidate_signature(row)
        if signature not in exact_cache:
            exact_cache[signature] = evaluate_candidate_exact_from_arrays(
                arrays=arrays,
                candidate_row=row,
                config=config,
                penalty_rate=penalty_rate,
            )
        rows.append(exact_cache[signature])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _grid_increment(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = np.unique(np.sort(arr))
    diffs = np.diff(arr)
    diffs = diffs[diffs > 1e-12]
    if diffs.size == 0:
        return 0.0
    return float(np.min(diffs))


def _find_formula_row_by_signature(grid_df: pd.DataFrame,
                                   signature: Tuple[str, str, Optional[float], Optional[float]]) -> Optional[pd.Series]:
    ppa_type, profile_type, volume_key, strike_key = signature
    mask = (
        grid_df["ppa_type"].astype(str).eq(str(ppa_type)) &
        grid_df["profile_type"].astype(str).eq(str(profile_type))
    )

    if volume_key is None:
        mask = mask & pd.to_numeric(grid_df["volume_mw"], errors="coerce").isna()
    else:
        mask = mask & np.isclose(pd.to_numeric(grid_df["volume_mw"], errors="coerce"), float(volume_key), atol=1e-9, rtol=0.0)

    if strike_key is None:
        mask = mask & pd.to_numeric(grid_df["strike_price_mwh"], errors="coerce").isna()
    else:
        mask = mask & np.isclose(pd.to_numeric(grid_df["strike_price_mwh"], errors="coerce"), float(strike_key), atol=1e-9, rtol=0.0)

    sub = grid_df.loc[mask].copy()
    if sub.empty:
        return None
    return sub.iloc[0].copy()


def _select_fix_seed_rows(grid_df: pd.DataFrame,
                          formula_best_solution: pd.Series,
                          formula_best_ppa_solution: Optional[pd.Series],
                          config: Dict) -> List[pd.Series]:
    fix_df = grid_df.loc[
        (grid_df["ppa_type"].astype(str) != "No Contract") &
        (grid_df["profile_type"].astype(str) == "Fix")
    ].copy()
    if fix_df.empty:
        return []

    k = max(1, int(config["verification_top_k_fix_seeds"]))
    seed_parts: List[pd.DataFrame] = []

    feasible_fix = fix_df.loc[fix_df["feasible"] == True].copy()
    if not feasible_fix.empty:
        seed_parts.append(feasible_fix.nsmallest(k, "seller_utility"))

    seed_parts.append(fix_df.nsmallest(k, "buyer_utility"))
    seed_parts.append(fix_df.nsmallest(k, "seller_utility"))

    for candidate in [formula_best_solution, formula_best_ppa_solution]:
        if candidate is None:
            continue
        if str(candidate.get("profile_type", "")) == "Fix":
            seed_parts.append(pd.DataFrame([candidate]))

    for ppa_type in ["Physical"]:
        sub = fix_df.loc[fix_df["ppa_type"].astype(str) == ppa_type].copy()
        if sub.empty:
            continue
        feasible_sub = sub.loc[sub["feasible"] == True].copy()
        if not feasible_sub.empty:
            seed_parts.append(feasible_sub.nsmallest(1, "seller_utility"))
        else:
            seed_parts.append(sub.nsmallest(1, "buyer_utility"))

    if not seed_parts:
        return []

    seed_df = pd.concat(seed_parts, ignore_index=True)
    seen = set()
    out: List[pd.Series] = []
    for _, row in seed_df.iterrows():
        signature = _candidate_signature(row)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(row.copy())
    return out


def _fix_neighborhood(grid_df: pd.DataFrame,
                      center_row: pd.Series,
                      config: Dict) -> pd.DataFrame:
    if str(center_row.get("profile_type", "")) != "Fix":
        return pd.DataFrame()

    strike_step = _grid_increment(config["strike_prices"])
    volume_step = _grid_increment([q for q in config["fixed_volumes"] if np.isfinite(q) and float(q) > 0.0])

    price_radius = strike_step * max(1, int(config["verification_fix_price_radius_steps"]))
    volume_radius = volume_step * max(1, int(config["verification_fix_volume_radius_steps"]))

    center_strike = pd.to_numeric(pd.Series([center_row.get("strike_price_mwh", np.nan)]), errors="coerce").iloc[0]
    center_volume = pd.to_numeric(pd.Series([center_row.get("volume_mw", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(center_strike) or pd.isna(center_volume):
        return pd.DataFrame()

    allowed_ppa = [str(center_row.get("ppa_type", ""))]
    if bool(config["verification_fix_allow_cross_ppa_type"]):
        allowed_ppa = ["Physical"]

    strike_col = pd.to_numeric(grid_df["strike_price_mwh"], errors="coerce")
    volume_col = pd.to_numeric(grid_df["volume_mw"], errors="coerce")

    mask = (
        grid_df["profile_type"].astype(str).eq("Fix") &
        grid_df["ppa_type"].astype(str).isin(allowed_ppa) &
        np.isfinite(strike_col) &
        np.isfinite(volume_col) &
        (np.abs(strike_col - float(center_strike)) <= price_radius + 1e-12) &
        (np.abs(volume_col - float(center_volume)) <= volume_radius + 1e-12)
    )
    return grid_df.loc[mask].copy()


def _is_same_candidate(left: pd.Series, right: pd.Series) -> bool:
    return _candidate_signature(left) == _candidate_signature(right)


def _choose_best_fix_local(exact_fix_df: pd.DataFrame) -> Optional[pd.Series]:
    if exact_fix_df is None or exact_fix_df.empty:
        return None
    return get_best_ppa_solution(exact_fix_df)


def repair_fix_from_seed(grid_df: pd.DataFrame,
                         seed_row: pd.Series,
                         arrays: Dict[str, object],
                         config: Dict,
                         penalty_rate: float,
                         exact_cache: Dict[Tuple[str, str, Optional[float], Optional[float]], Dict[str, object]]) -> Dict[str, object]:
    if str(seed_row.get("profile_type", "")) != "Fix":
        return {
            "seed_signature": _candidate_signature(seed_row),
            "final_signature": _candidate_signature(seed_row),
            "iterations": 0,
            "visited_signatures": [],
        }

    current_formula_row = seed_row.copy()
    path_signatures: List[Tuple[str, str, Optional[float], Optional[float]]] = []
    seen = set()

    max_iterations = max(1, int(config["verification_fix_max_iterations"]))

    for iteration in range(max_iterations):
        current_signature = _candidate_signature(current_formula_row)
        path_signatures.append(current_signature)
        seen.add(current_signature)

        neighborhood_df = _fix_neighborhood(grid_df=grid_df, center_row=current_formula_row, config=config)
        if neighborhood_df.empty:
            break

        exact_neighborhood_df = evaluate_candidates_exact(
            candidate_df=neighborhood_df,
            arrays=arrays,
            config=config,
            penalty_rate=penalty_rate,
            exact_cache=exact_cache,
        )
        best_local = _choose_best_fix_local(exact_neighborhood_df)
        if best_local is None:
            break

        best_signature = _candidate_signature(best_local)
        if best_signature == current_signature:
            break
        if best_signature in seen:
            break

        next_formula_row = _find_formula_row_by_signature(grid_df=grid_df, signature=best_signature)
        if next_formula_row is None:
            break

        current_formula_row = next_formula_row

    return {
        "seed_signature": _candidate_signature(seed_row),
        "final_signature": _candidate_signature(current_formula_row),
        "iterations": len(path_signatures),
        "visited_signatures": [str(sig) for sig in path_signatures],
    }


def annotate_grid_with_exact_results(grid_df: pd.DataFrame,
                                     exact_cache: Dict[Tuple[str, str, Optional[float], Optional[float]], Dict[str, object]],
                                     formula_best_solution: pd.Series,
                                     verified_best_solution: pd.Series,
                                     verified_best_ppa_solution: Optional[pd.Series]) -> pd.DataFrame:
    out = grid_df.copy()
    out["candidate_id"] = out.apply(_candidate_signature_str, axis=1)

    if exact_cache:
        exact_records = []
        for signature, record in exact_cache.items():
            record_copy = dict(record)
            record_copy["candidate_id"] = _candidate_signature_str(pd.Series(record))
            exact_records.append(record_copy)
        exact_df = pd.DataFrame(exact_records)
        if not exact_df.empty:
            rename_map = {
                "seller_utility": "exact_seller_utility",
                "buyer_utility": "exact_buyer_utility",
                "seller_mean_exposure": "exact_seller_mean_exposure",
                "buyer_mean_exposure": "exact_buyer_mean_exposure",
                "seller_exposure_variance": "exact_seller_exposure_variance",
                "buyer_exposure_variance": "exact_buyer_exposure_variance",
                "expected_seller_revenue": "exact_expected_seller_revenue",
                "feasible": "exact_feasible",
                "decision_metric": "exact_decision_metric",
            }
            exact_df = exact_df.rename(columns=rename_map)
            keep_cols = ["candidate_id"] + list(rename_map.values())
            out = out.merge(exact_df[keep_cols], on="candidate_id", how="left")

    formula_best_id = _candidate_signature_str(formula_best_solution)
    verified_best_id = _candidate_signature_str(verified_best_solution)
    verified_best_ppa_id = None if verified_best_ppa_solution is None else _candidate_signature_str(verified_best_ppa_solution)

    out["formula_selected"] = out["candidate_id"].eq(formula_best_id)
    out["verified_selected"] = out["candidate_id"].eq(verified_best_id)
    out["verified_best_ppa"] = False if verified_best_ppa_id is None else out["candidate_id"].eq(verified_best_ppa_id)
    out["exact_evaluated"] = out.get("exact_seller_utility", pd.Series(index=out.index, dtype=float)).notna()

    return out


def select_verified_solutions(sample_df: pd.DataFrame,
                              grid_df: pd.DataFrame,
                              config: Dict,
                              penalty_rate: float) -> Dict[str, object]:
    arrays = prepare_sample_arrays(sample_df)
    exact_cache: Dict[Tuple[str, str, Optional[float], Optional[float]], Dict[str, object]] = {}

    formula_best_solution = get_optimal_solution(grid_df)
    formula_best_ppa_solution = get_best_ppa_solution(grid_df)

    if not bool(config.get("verification_enabled", True)):
        exact_cache[_candidate_signature(formula_best_solution)] = evaluate_candidate_exact_from_arrays(
            arrays=arrays,
            candidate_row=formula_best_solution,
            config=config,
            penalty_rate=penalty_rate,
        )
        if formula_best_ppa_solution is not None:
            exact_cache[_candidate_signature(formula_best_ppa_solution)] = evaluate_candidate_exact_from_arrays(
                arrays=arrays,
                candidate_row=formula_best_ppa_solution,
                config=config,
                penalty_rate=penalty_rate,
            )
        verified_best_solution = pd.Series(exact_cache[_candidate_signature(formula_best_solution)]).copy()
        verified_best_ppa_solution = None if formula_best_ppa_solution is None else pd.Series(
            exact_cache[_candidate_signature(formula_best_ppa_solution)]
        ).copy()
        verification_report = {
            "verification_enabled": False,
            "formula_best_verification_passed": True,
            "formula_best_feasible_formula": bool(formula_best_solution.get("feasible", False)),
            "formula_best_feasible_exact": bool(verified_best_solution.get("feasible", False)),
            "formula_best_changed_after_verification": False,
            "formula_best_ppa_changed_after_verification": False,
            "n_exact_candidates_evaluated": int(len(exact_cache)),
            "n_fix_seed_rows": 0,
            "fix_repair_reports_json": json.dumps([]),
        }
        verified_grid_df = annotate_grid_with_exact_results(
            grid_df=grid_df,
            exact_cache=exact_cache,
            formula_best_solution=formula_best_solution,
            verified_best_solution=verified_best_solution,
            verified_best_ppa_solution=verified_best_ppa_solution,
        )
        exact_candidate_df = pd.DataFrame(list(exact_cache.values()))
        return {
            "formula_best_solution": formula_best_solution,
            "formula_best_ppa_solution": formula_best_ppa_solution,
            "verified_best_solution": verified_best_solution,
            "verified_best_ppa_solution": verified_best_ppa_solution,
            "verification_report": verification_report,
            "verified_grid_df": verified_grid_df,
            "exact_candidate_df": exact_candidate_df,
        }

    exact_cache[_candidate_signature(formula_best_solution)] = evaluate_candidate_exact_from_arrays(
        arrays=arrays,
        candidate_row=formula_best_solution,
        config=config,
        penalty_rate=penalty_rate,
    )
    if formula_best_ppa_solution is not None:
        exact_cache[_candidate_signature(formula_best_ppa_solution)] = evaluate_candidate_exact_from_arrays(
            arrays=arrays,
            candidate_row=formula_best_ppa_solution,
            config=config,
            penalty_rate=penalty_rate,
        )

    exact_candidate_parts: List[pd.DataFrame] = []

    if bool(config["verification_exact_all_nonfix"]):
        nonfix_df = grid_df.loc[grid_df["profile_type"].astype(str) != "Fix"].copy()
        exact_nonfix_df = evaluate_candidates_exact(
            candidate_df=nonfix_df,
            arrays=arrays,
            config=config,
            penalty_rate=penalty_rate,
            exact_cache=exact_cache,
        )
        if not exact_nonfix_df.empty:
            exact_candidate_parts.append(exact_nonfix_df)

    fix_seed_rows = _select_fix_seed_rows(
        grid_df=grid_df,
        formula_best_solution=formula_best_solution,
        formula_best_ppa_solution=formula_best_ppa_solution,
        config=config,
    )
    fix_repair_reports = []
    for seed_row in fix_seed_rows:
        fix_repair_reports.append(
            repair_fix_from_seed(
                grid_df=grid_df,
                seed_row=seed_row,
                arrays=arrays,
                config=config,
                penalty_rate=penalty_rate,
                exact_cache=exact_cache,
            )
        )

    exact_candidate_df = pd.DataFrame(list(exact_cache.values()))
    if exact_candidate_df.empty:
        exact_candidate_df = evaluate_candidates_exact(
            candidate_df=pd.DataFrame([formula_best_solution]),
            arrays=arrays,
            config=config,
            penalty_rate=penalty_rate,
            exact_cache=exact_cache,
        )
    else:
        exact_candidate_df = exact_candidate_df.copy()

    exact_candidate_df = exact_candidate_df.drop_duplicates(
        subset=["ppa_type", "profile_type", "volume_mw", "strike_price_mwh"]
    ).reset_index(drop=True)

    verified_best_solution = get_optimal_solution(exact_candidate_df)
    verified_best_ppa_solution = get_best_ppa_solution(exact_candidate_df)

    formula_best_exact = pd.Series(
        exact_cache[_candidate_signature(formula_best_solution)]
    ).copy()
    if formula_best_ppa_solution is not None:
        formula_best_ppa_exact = pd.Series(
            exact_cache[_candidate_signature(formula_best_ppa_solution)]
        ).copy()
    else:
        formula_best_ppa_exact = None

    formula_vs_exact_best = compare_formula_vs_exact(
        formula_row=formula_best_solution,
        exact_row=formula_best_exact,
        abs_tol=float(config["verification_abs_tolerance"]),
        rel_tol=float(config["verification_rel_tolerance"]),
    )
    formula_vs_exact_best_ppa = (
        compare_formula_vs_exact(
            formula_row=formula_best_ppa_solution,
            exact_row=formula_best_ppa_exact,
            abs_tol=float(config["verification_abs_tolerance"]),
            rel_tol=float(config["verification_rel_tolerance"]),
        )
        if (formula_best_ppa_solution is not None and formula_best_ppa_exact is not None)
        else {}
    )

    verified_grid_df = annotate_grid_with_exact_results(
        grid_df=grid_df,
        exact_cache=exact_cache,
        formula_best_solution=formula_best_solution,
        verified_best_solution=verified_best_solution,
        verified_best_ppa_solution=verified_best_ppa_solution,
    )

    verification_report = {
        "verification_enabled": bool(config["verification_enabled"]),
        "formula_best_verification_passed": bool(formula_vs_exact_best.get("verification_passed", False)),
        "formula_best_feasible_formula": formula_vs_exact_best.get("formula_feasible"),
        "formula_best_feasible_exact": formula_vs_exact_best.get("exact_feasible"),
        "formula_best_changed_after_verification": bool(
            _candidate_signature(formula_best_solution) != _candidate_signature(verified_best_solution)
        ),
        "formula_best_ppa_changed_after_verification": bool(
            (formula_best_ppa_solution is not None and verified_best_ppa_solution is not None) and
            (_candidate_signature(formula_best_ppa_solution) != _candidate_signature(verified_best_ppa_solution))
        ),
        "n_exact_candidates_evaluated": int(len(exact_cache)),
        "n_fix_seed_rows": int(len(fix_seed_rows)),
        "fix_repair_reports_json": json.dumps(fix_repair_reports),
    }
    verification_report.update({f"formula_best__{k}": v for k, v in formula_vs_exact_best.items()})
    verification_report.update({f"formula_best_ppa__{k}": v for k, v in formula_vs_exact_best_ppa.items()})

    return {
        "formula_best_solution": formula_best_solution,
        "formula_best_ppa_solution": formula_best_ppa_solution,
        "verified_best_solution": verified_best_solution,
        "verified_best_ppa_solution": verified_best_ppa_solution,
        "verification_report": verification_report,
        "verified_grid_df": verified_grid_df,
        "exact_candidate_df": exact_candidate_df,
    }


# In[6]:


# ============================================================
# Contract-period FE selection, verification-aware output writing, and batch run
# ============================================================
def select_hourly_fe_export_solution(best_solution: pd.Series,
                                     best_ppa_solution: Optional[pd.Series],
                                     save_best_fe_even_if_infeasible: bool) -> Tuple[pd.Series, str]:
    if str(best_solution.get("ppa_type", "No Contract")) != "No Contract":
        return best_solution, "Selected_Best_Solution"

    if not save_best_fe_even_if_infeasible:
        return best_solution, "Selected_Best_Solution"

    if best_ppa_solution is None:
        return best_solution, "Selected_Best_Solution"

    if str(best_ppa_solution.get("ppa_type", "No Contract")) == "No Contract":
        return best_solution, "Selected_Best_Solution"

    if bool(best_ppa_solution.get("feasible", False)):
        return best_ppa_solution, "Best_Feasible_PPA_While_Decision_Is_No_Contract"

    return best_ppa_solution, "Best_Infeasible_PPA"


def evaluate_hourly_fe_for_solution(sample_df: pd.DataFrame,
                                    solution: pd.Series,
                                    config: Dict,
                                    penalty_rate: float) -> pd.DataFrame:
    """Return one contract-period financial-exposure row per replication.

    The function name and the existing Hourly_FE output path are retained for
    compatibility with the current mutation workflow. The contents now follow
    the revised document: exposure is aggregated over the contract period for
    each replication, not reported as hourly exposure observations.
    """
    arrays = prepare_sample_arrays(sample_df)
    flat_df = arrays["flat_df"]
    replication_starts = arrays["replication_starts"]
    replication_labels = flat_df.iloc[replication_starts]["replication"].astype(int).to_numpy()
    n_hours = np.diff(np.r_[replication_starts, len(flat_df)]).astype(int)
    ppa_type = str(solution.get("ppa_type", "No Contract"))
    profile_type = str(solution.get("profile_type", "N/A"))
    strike_price = pd.to_numeric(pd.Series([solution.get("strike_price_mwh", np.nan)]), errors="coerce").iloc[0]
    volume_mw = pd.to_numeric(pd.Series([solution.get("volume_mw", np.nan)]), errors="coerce").iloc[0]

    if ppa_type == "No Contract":
        out = evaluate_no_contract(
            arrays["g"], arrays["d"], arrays["Ns"], arrays["Nb_out"], arrays["Nb_in"],
            lambda_s=config["lambda_s"],
            lambda_b=config["lambda_b"],
            replication_starts=replication_starts,
        )
    else:
        out = evaluate_contract(
            g=arrays["g"],
            d=arrays["d"],
            Ns=arrays["Ns"],
            Nb_out=arrays["Nb_out"],
            Nb_in=arrays["Nb_in"],
            mu=ppa_type,
            nu=profile_type,
            pi=0.0 if pd.isna(strike_price) else float(strike_price),
            q_fixed=0.0 if pd.isna(volume_mw) else float(volume_mw),
            lambda_s=config["lambda_s"],
            lambda_b=config["lambda_b"],
            availability_factor=config["availability_factor"],
            penalty_rate=penalty_rate,
            replication_starts=replication_starts,
        )

    fe_df = pd.DataFrame({
        "replication": replication_labels,
        "n_hours": n_hours,
        "seller_fe": _sum_by_replication(out["FE_s"], replication_starts),
        "buyer_fe": _sum_by_replication(out["FE_b"], replication_starts),
        "seller_revenue": _sum_by_replication(out["Rev_s"], replication_starts),
        "asg_shortfall_penalty": _sum_by_replication(out["Pen_t"], replication_starts),
        "exposure_level": "contract_period_replication",
    })

    if config.get("save_timestamp_in_hourly_fe", False) and "timestamp" in flat_df.columns:
        last_positions = np.r_[replication_starts[1:], len(flat_df)] - 1
        fe_df["period_start_timestamp"] = flat_df.iloc[replication_starts]["timestamp"].to_numpy()
        fe_df["period_end_timestamp"] = flat_df.iloc[last_positions]["timestamp"].to_numpy()

    return fe_df


def build_case_info_long(match_id: int,
                         match_row: Dict[str, object],
                         basic_stats: Dict[str, object],
                         penalty_rate: float,
                         buyer_purchase_mean: float,
                         best_solution: pd.Series,
                         best_ppa_solution: Optional[pd.Series],
                         formula_best_solution: pd.Series,
                         formula_best_ppa_solution: Optional[pd.Series],
                         verification_report: Dict[str, object],
                         scenario_name: str,
                         config: Dict) -> pd.DataFrame:
    records = []

    def add(section: str, item: str, value):
        records.append({"section": section, "item": item, "value": value})

    add("Scenario_Bank", "scenario_name", scenario_name)
    add("Scenario_Bank", "sample_root_dir", config["sample_root_dir"])
    add("Scenario_Bank", "n_replications", basic_stats.get("n_replications"))
    add("Scenario_Bank", "hours_per_replication", basic_stats.get("hours_per_replication"))

    add("Risk_Setting", "lambda_s", config["lambda_s"])
    add("Risk_Setting", "lambda_b", config["lambda_b"])
    add("Risk_Setting", "availability_factor", config["availability_factor"])
    add("Risk_Setting", "penalty_rate", penalty_rate)
    add("Risk_Setting", "buyer_lmp_in_mean", buyer_purchase_mean)

    add("Decision", "ppa_type", best_solution.get("ppa_type"))
    add("Decision", "profile_type", best_solution.get("profile_type"))
    add("Decision", "strike_price_mwh", best_solution.get("strike_price_mwh"))
    add("Decision", "volume_mw", best_solution.get("volume_mw"))
    add("Decision", "seller_utility", best_solution.get("seller_utility"))
    add("Decision", "buyer_utility", best_solution.get("buyer_utility"))
    add("Decision", "feasible", best_solution.get("feasible"))

    if best_ppa_solution is not None:
        add("Best_PPA", "ppa_type", best_ppa_solution.get("ppa_type"))
        add("Best_PPA", "profile_type", best_ppa_solution.get("profile_type"))
        add("Best_PPA", "strike_price_mwh", best_ppa_solution.get("strike_price_mwh"))
        add("Best_PPA", "volume_mw", best_ppa_solution.get("volume_mw"))
        add("Best_PPA", "seller_utility", best_ppa_solution.get("seller_utility"))
        add("Best_PPA", "buyer_utility", best_ppa_solution.get("buyer_utility"))
        add("Best_PPA", "feasible", best_ppa_solution.get("feasible"))

    add("Formula_Decision", "ppa_type", formula_best_solution.get("ppa_type"))
    add("Formula_Decision", "profile_type", formula_best_solution.get("profile_type"))
    add("Formula_Decision", "strike_price_mwh", formula_best_solution.get("strike_price_mwh"))
    add("Formula_Decision", "volume_mw", formula_best_solution.get("volume_mw"))
    add("Formula_Decision", "seller_utility", formula_best_solution.get("seller_utility"))
    add("Formula_Decision", "buyer_utility", formula_best_solution.get("buyer_utility"))
    add("Formula_Decision", "feasible", formula_best_solution.get("feasible"))

    if formula_best_ppa_solution is not None:
        add("Formula_Best_PPA", "ppa_type", formula_best_ppa_solution.get("ppa_type"))
        add("Formula_Best_PPA", "profile_type", formula_best_ppa_solution.get("profile_type"))
        add("Formula_Best_PPA", "strike_price_mwh", formula_best_ppa_solution.get("strike_price_mwh"))
        add("Formula_Best_PPA", "volume_mw", formula_best_ppa_solution.get("volume_mw"))
        add("Formula_Best_PPA", "seller_utility", formula_best_ppa_solution.get("seller_utility"))
        add("Formula_Best_PPA", "buyer_utility", formula_best_ppa_solution.get("buyer_utility"))
        add("Formula_Best_PPA", "feasible", formula_best_ppa_solution.get("feasible"))

    for key, value in verification_report.items():
        add("Verification", key, value)

    for key, value in match_row.items():
        add("Match_Metadata", key, value)

    stat_sections = {
        "Correlations": [
            "shape_corr_original", "basis_corr_original", "shape_corr_simulated", "basis_corr_simulated",
            "shape_corr_target", "basis_corr_target", "shape_corr_ref", "basis_corr_ref",
        ],
        "Means": [
            "generation_original_mean", "demand_original_mean", "seller_lmp_original_mean", "buyer_lmp_out_original_mean", "buyer_lmp_in_original_mean",
            "generation_simulated_mean", "demand_simulated_mean", "seller_lmp_simulated_mean", "buyer_lmp_out_simulated_mean", "buyer_lmp_in_simulated_mean",
        ],
        "Medians": [
            "generation_original_median", "demand_original_median", "seller_lmp_original_median", "buyer_lmp_out_original_median", "buyer_lmp_in_original_median",
            "generation_simulated_median", "demand_simulated_median", "seller_lmp_simulated_median", "buyer_lmp_out_simulated_median", "buyer_lmp_in_simulated_median",
        ],
        "Plot_Indices": [
            "volume_mismatch_mean_ref", "price_spread_mean_ref", "volume_mismatch_median_ref", "price_spread_median_ref",
        ],
        "Categories": [
            "shape_regime", "shape_regime_label", "basis_regime", "basis_regime_label",
            "volume_mean_cat", "volume_mean_cat_label", "price_mean_cat", "price_mean_cat_label",
            "volume_median_cat", "volume_median_cat_label", "price_median_cat", "price_median_cat_label",
            "combined_category",
        ],
    }

    for section, items in stat_sections.items():
        for item in items:
            add(section, item, basic_stats.get(item))

    return pd.DataFrame(records)


def to_plain_dict(series: Optional[pd.Series]) -> Dict[str, object]:
    if series is None:
        return {}
    data = {}
    for key, value in series.to_dict().items():
        if isinstance(value, (np.floating, np.integer)):
            data[key] = value.item()
        else:
            data[key] = value
    return data


def run_one_match(match_id: int,
                  match_dir: Path,
                  sample_root: Path,
                  output_root: Path,
                  match_table_df: pd.DataFrame,
                  corr_targets: pd.DataFrame,
                  config: Dict) -> Dict[str, object]:
    original_df = load_original_match_df(match_id, match_dir)
    sample_df = load_baseline_sample_bank(match_id, sample_root)
    penalty_rate, buyer_purchase_mean = compute_penalty_rate(sample_df)

    grid_df_formula = optimize_ppa_contracts(sample_df=sample_df, config=config, penalty_rate=penalty_rate)
    verification_bundle = select_verified_solutions(
        sample_df=sample_df,
        grid_df=grid_df_formula,
        config=config,
        penalty_rate=penalty_rate,
    )

    formula_best_solution = verification_bundle["formula_best_solution"]
    formula_best_ppa_solution = verification_bundle["formula_best_ppa_solution"]
    best_solution = verification_bundle["verified_best_solution"]
    best_ppa_solution = verification_bundle["verified_best_ppa_solution"]
    verification_report = verification_bundle["verification_report"]
    grid_df = verification_bundle["verified_grid_df"]

    basic_stats = build_basic_stats(
        match_id=match_id,
        original_df=original_df,
        sample_df=sample_df,
        corr_targets=corr_targets,
        config=config,
    )

    if not match_table_df.empty and "match_id" in match_table_df.columns:
        selected = match_table_df.loc[match_table_df["match_id"] == int(match_id)].copy()
        match_row = selected.iloc[0].to_dict() if not selected.empty else {}
    else:
        match_row = {}

    selected_fe_solution, fe_source = select_hourly_fe_export_solution(
        best_solution=best_solution,
        best_ppa_solution=best_ppa_solution,
        save_best_fe_even_if_infeasible=config["save_best_fe_even_if_infeasible"],
    )

    match_output_dir = output_root / "Per_Match" / f"Match_{int(match_id):04d}"
    match_output_dir.mkdir(parents=True, exist_ok=True)

    case_info_df = build_case_info_long(
        match_id=match_id,
        match_row=match_row,
        basic_stats=basic_stats,
        penalty_rate=penalty_rate,
        buyer_purchase_mean=buyer_purchase_mean,
        best_solution=best_solution,
        best_ppa_solution=best_ppa_solution,
        formula_best_solution=formula_best_solution,
        formula_best_ppa_solution=formula_best_ppa_solution,
        verification_report=verification_report,
        scenario_name=config["scenario_name"],
        config=config,
    )
    case_info_path = match_output_dir / "Case_Info.csv"
    _write_csv_compact(case_info_df, case_info_path)

    best_solution_dict = to_plain_dict(best_solution)
    best_solution_dict["match_id"] = int(match_id)
    best_solution_dict["scenario_name"] = config["scenario_name"]
    best_solution_dict["penalty_rate"] = penalty_rate
    best_solution_dict["buyer_lmp_in_mean"] = buyer_purchase_mean
    best_solution_dict["unusual_contracted_volume"] = unusual_contracted_volume_flag(best_solution, sample_df)
    best_solution_dict["formula_selected_ppa_type"] = formula_best_solution.get("ppa_type")
    best_solution_dict["formula_selected_profile_type"] = formula_best_solution.get("profile_type")
    best_solution_dict["formula_selected_volume_mw"] = formula_best_solution.get("volume_mw")
    best_solution_dict["formula_selected_strike_price_mwh"] = formula_best_solution.get("strike_price_mwh")
    best_solution_dict.update(verification_report)
    best_solution_path = match_output_dir / "Best_Solution.csv"
    _write_csv_compact(pd.DataFrame([best_solution_dict]), best_solution_path)

    best_ppa_solution_dict = to_plain_dict(best_ppa_solution) if best_ppa_solution is not None else {}
    if best_ppa_solution_dict:
        best_ppa_solution_dict["match_id"] = int(match_id)
        best_ppa_solution_dict["scenario_name"] = config["scenario_name"]
        best_ppa_solution_dict["penalty_rate"] = penalty_rate
        best_ppa_solution_dict["buyer_lmp_in_mean"] = buyer_purchase_mean
        best_ppa_solution_dict["unusual_contracted_volume"] = unusual_contracted_volume_flag(best_ppa_solution, sample_df)
        if formula_best_ppa_solution is not None:
            best_ppa_solution_dict["formula_best_ppa_type"] = formula_best_ppa_solution.get("ppa_type")
            best_ppa_solution_dict["formula_best_ppa_profile_type"] = formula_best_ppa_solution.get("profile_type")
            best_ppa_solution_dict["formula_best_ppa_volume_mw"] = formula_best_ppa_solution.get("volume_mw")
            best_ppa_solution_dict["formula_best_ppa_strike_price_mwh"] = formula_best_ppa_solution.get("strike_price_mwh")
        best_ppa_solution_dict.update(verification_report)
        best_ppa_solution_path = match_output_dir / "Best_PPA_Solution.csv"
        _write_csv_compact(pd.DataFrame([best_ppa_solution_dict]), best_ppa_solution_path)
    else:
        best_ppa_solution_path = None

    if config["save_per_match_grid"]:
        grid_path = match_output_dir / "Contract_Grid.csv"
        _write_csv_compact(grid_df, grid_path)
    else:
        grid_path = None

    hourly_fe_path = None
    hourly_fe_index_row = None
    if config["save_hourly_fe"]:
        fe_df = evaluate_hourly_fe_for_solution(
            sample_df=sample_df,
            solution=selected_fe_solution,
            config=config,
            penalty_rate=penalty_rate,
        )
        fe_subdir = output_root / "Hourly_FE"
        fe_subdir.mkdir(parents=True, exist_ok=True)
        hourly_fe_path = fe_subdir / f"Hourly_FE_Match_{int(match_id):04d}__{config['scenario_name']}.csv"
        _write_csv_compact(fe_df, hourly_fe_path, decimal_places=int(globals().get("hourly_fe_decimal_places", 2)))

        hourly_fe_index_row = {
            "match_id": int(match_id),
            "scenario_name": config["scenario_name"],
            "scenario_type": "baseline",
            "hourly_fe_file": str(hourly_fe_path),
            "fe_source": fe_source,
            "exposure_level": "contract_period_replication",
            "decision_ppa_type": best_solution.get("ppa_type"),
            "decision_profile_type": best_solution.get("profile_type"),
            "decision_strike_price_mwh": best_solution.get("strike_price_mwh"),
            "decision_volume_mw": best_solution.get("volume_mw"),
            "formula_decision_ppa_type": formula_best_solution.get("ppa_type"),
            "formula_decision_profile_type": formula_best_solution.get("profile_type"),
            "formula_decision_strike_price_mwh": formula_best_solution.get("strike_price_mwh"),
            "formula_decision_volume_mw": formula_best_solution.get("volume_mw"),
            "ppa_type": selected_fe_solution.get("ppa_type"),
            "profile_type": selected_fe_solution.get("profile_type"),
            "strike_price_mwh": selected_fe_solution.get("strike_price_mwh"),
            "volume_mw": selected_fe_solution.get("volume_mw"),
            "feasible": selected_fe_solution.get("feasible"),
            "n_replications": basic_stats["n_replications"],
            "hours_per_replication": basic_stats["hours_per_replication"],
            **verification_report,
        }

    summary_row = dict(basic_stats)
    summary_row.update({
        "scenario_name": config["scenario_name"],
        "seller_utility": best_solution.get("seller_utility"),
        "buyer_utility": best_solution.get("buyer_utility"),
        "seller_mean_exposure": best_solution.get("seller_mean_exposure"),
        "buyer_mean_exposure": best_solution.get("buyer_mean_exposure"),
        "seller_exposure_variance": best_solution.get("seller_exposure_variance"),
        "buyer_exposure_variance": best_solution.get("buyer_exposure_variance"),
        "expected_seller_revenue": best_solution.get("expected_seller_revenue"),
        "ppa_type": best_solution.get("ppa_type"),
        "profile_type": best_solution.get("profile_type"),
        "volume_mw": best_solution.get("volume_mw"),
        "strike_price_mwh": best_solution.get("strike_price_mwh"),
        "feasible": best_solution.get("feasible"),
        "unusual_contracted_volume": unusual_contracted_volume_flag(best_solution, sample_df),
        "best_ppa_type": best_ppa_solution.get("ppa_type") if best_ppa_solution is not None else None,
        "best_ppa_profile_type": best_ppa_solution.get("profile_type") if best_ppa_solution is not None else None,
        "best_ppa_volume_mw": best_ppa_solution.get("volume_mw") if best_ppa_solution is not None else np.nan,
        "best_ppa_strike_price_mwh": best_ppa_solution.get("strike_price_mwh") if best_ppa_solution is not None else np.nan,
        "best_ppa_feasible": best_ppa_solution.get("feasible") if best_ppa_solution is not None else None,
        "formula_selected_ppa_type": formula_best_solution.get("ppa_type"),
        "formula_selected_profile_type": formula_best_solution.get("profile_type"),
        "formula_selected_volume_mw": formula_best_solution.get("volume_mw"),
        "formula_selected_strike_price_mwh": formula_best_solution.get("strike_price_mwh"),
        "formula_best_ppa_type": formula_best_ppa_solution.get("ppa_type") if formula_best_ppa_solution is not None else None,
        "formula_best_ppa_profile_type": formula_best_ppa_solution.get("profile_type") if formula_best_ppa_solution is not None else None,
        "formula_best_ppa_volume_mw": formula_best_ppa_solution.get("volume_mw") if formula_best_ppa_solution is not None else np.nan,
        "formula_best_ppa_strike_price_mwh": formula_best_ppa_solution.get("strike_price_mwh") if formula_best_ppa_solution is not None else np.nan,
        "penalty_rate": penalty_rate,
        "buyer_lmp_in_mean": buyer_purchase_mean,
        "fe_source": fe_source,
        "exposure_level": "contract_period_replication",
        "case_info_file": str(case_info_path),
        "best_solution_file": str(best_solution_path),
        "best_ppa_solution_file": str(best_ppa_solution_path) if best_ppa_solution_path is not None else "",
        "contract_grid_file": str(grid_path) if grid_path is not None else "",
        "hourly_fe_file": str(hourly_fe_path) if hourly_fe_path is not None else "",
        **verification_report,
    })

    for key, value in match_row.items():
        summary_row[f"match__{key}"] = value

    return {
        "summary_row": summary_row,
        "best_solution_row": best_solution_dict,
        "best_ppa_solution_row": best_ppa_solution_dict,
        "hourly_fe_index_row": hourly_fe_index_row,
        "verification_report": verification_report,
    }


# In[7]:


# ============================================================
# Mutation simulation settings
# ============================================================
# This script reuses the verified no-mutation contract-evaluation functions above.
# The new part is the scenario loop: baseline + three mutation levels per enabled
# mutation family.

from scipy.stats import norm, rankdata

# IMPORTANT:
# The uploaded verified simulation notebook loads realized baseline samples rather
# than the latent Z random-bank files. Therefore this mutation implementation uses
# deterministic Gaussian-rank recoloring of the realized baseline bank. It preserves
# the realized baseline bank for the no-mutation case and preserves within
# quarter-hour marginal values for the mutated cases, while changing the dependence
# structure by targeted Gaussian-copula correlation perturbations.

# Output folder for the baseline + mutation run.
mutation_output_dir = str(Path("simulation_mutation") / "Output files (Risk Neutral, Mutation, Verified)")

# Run all four economically interpretable mutation families by default.
# To run only one family, for example basis deterioration, set:
# enabled_mutation_families = ["basis"]
enabled_mutation_families = ["shape", "basis", "load_price", "cannibalization"]

# Three mutation levels required for Section 5.
mutation_levels_abs = [0.10, 0.30, 0.50]

# If True, each mutated sample preserves the exact empirical marginal values
# within each (quarter, hour) cell. This is the safest option for the current
# code because the marginal distributions are already embedded in the realized
# baseline sample folders.
preserve_quarter_hour_marginals = True

# If True, save the mutated sample pool for each match/scenario. This is useful
# for diagnostics but can create large files.
save_mutated_sample_pools = False

# Keep baseline scenario name identical to the verified no-mutation notebook
# so that baseline outputs remain directly comparable.
baseline_scenario_name = "Baseline__No_Mutation__Verified"

# Add mutation metadata columns to the all-match summary and contract-period FE index.
# Compact CSV writer used by per-match outputs in this notebook.
# The global variables output_decimal_places and aggregate_output_drop_columns are
# defined in the sample-based execution cell below. Defaults are used if that cell
# has not been executed yet.
def _compact_table_for_export(df: pd.DataFrame,
                              *,
                              drop_columns: Optional[List[str]] = None,
                              decimal_places: Optional[int] = None) -> pd.DataFrame:
    out = df.copy()
    if drop_columns:
        out = out.drop(columns=[c for c in drop_columns if c in out.columns], errors="ignore")
    if decimal_places is not None:
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            out[numeric_cols] = out[numeric_cols].round(int(decimal_places))
    return out


def _write_csv_compact(df: pd.DataFrame,
                       path: Path,
                       *,
                       drop_columns: Optional[List[str]] = None,
                       decimal_places: Optional[int] = None) -> None:
    dp = int(globals().get("output_decimal_places", 2)) if decimal_places is None else int(decimal_places)
    drop = drop_columns
    if drop is None and bool(globals().get("compact_aggregate_outputs", True)):
        drop = list(globals().get("aggregate_output_drop_columns", []))
    out = _compact_table_for_export(df, drop_columns=drop, decimal_places=dp)
    float_format = f"%.{dp}f"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format=float_format)


mutation_method = "gaussian_rank_recoloring_realized_bank"

# Override selected CONFIG values for the mutation run.
CONFIG.update({
    "output_dir": mutation_output_dir,
    "scenario_name": baseline_scenario_name,
    "skip_existing_output": False,
    "save_per_match_grid": False,
    "save_hourly_fe": True,
    "save_best_fe_even_if_infeasible": True,
    "save_timestamp_in_hourly_fe": False,
    "mutation_method": mutation_method,
    "preserve_quarter_hour_marginals": bool(preserve_quarter_hour_marginals),
})

VECTOR_COLUMNS = ["generation", "demand", "seller_lmp", "buyer_lmp_out"]
VECTOR_LABELS = {
    "generation": "g",
    "demand": "d",
    "seller_lmp": "Ns",
    "buyer_lmp_out": "Nb_out",
}

MUTATION_FAMILY_SPECS = {
    "shape": {
        "mutation_family_label": "Shape deterioration",
        "target_pair": ("generation", "demand"),
        "target_pair_label": "rho(g,d)",
        "direction": -1.0,
        "scenario_prefix": "ShapeDeterioration",
    },
    "basis": {
        "mutation_family_label": "Basis deterioration",
        "target_pair": ("seller_lmp", "buyer_lmp_out"),
        "target_pair_label": "rho(Ns,Nb_out)",
        "direction": -1.0,
        "scenario_prefix": "BasisDeterioration",
    },
    "load_price": {
        "mutation_family_label": "Buyer load-price intensification",
        "target_pair": ("demand", "buyer_lmp_out"),
        "target_pair_label": "rho(d,Nb_out)",
        "direction": +1.0,
        "scenario_prefix": "LoadPriceIntensification",
    },
    "cannibalization": {
        "mutation_family_label": "Seller-side cannibalization intensification",
        "target_pair": ("generation", "seller_lmp"),
        "target_pair_label": "rho(g,Ns)",
        "direction": -1.0,
        "scenario_prefix": "SellerCannibalization",
    },
}


def _delta_token(delta: float) -> str:
    sign = "p" if float(delta) >= 0 else "m"
    return f"{sign}{abs(float(delta)):.2f}".replace(".", "p")


def build_mutation_scenario_specs(enabled_families: Iterable[str],
                                  mutation_levels: Iterable[float]) -> List[Dict[str, object]]:
    scenario_specs: List[Dict[str, object]] = [{
        "scenario_name": baseline_scenario_name,
        "scenario_type": "baseline",
        "scenario_order": 0,
        "mutation_family": "baseline",
        "mutation_family_label": "Baseline",
        "target_pair": None,
        "target_pair_label": "none",
        "target_delta": 0.0,
        "mutation_level_abs": 0.0,
        "mutation_level_label": "Baseline",
        "mutation_method": "none",
    }]

    scenario_order = 1
    for family in enabled_families:
        if family not in MUTATION_FAMILY_SPECS:
            raise KeyError(f"Unknown mutation family: {family!r}. Valid keys: {list(MUTATION_FAMILY_SPECS)}")
        spec = MUTATION_FAMILY_SPECS[family]
        for level_abs in mutation_levels:
            delta = float(spec["direction"]) * float(level_abs)
            scenario_name = f"Mutation__{spec['scenario_prefix']}__delta_{_delta_token(delta)}"
            scenario_specs.append({
                "scenario_name": scenario_name,
                "scenario_type": "mutation",
                "scenario_order": scenario_order,
                "mutation_family": family,
                "mutation_family_label": spec["mutation_family_label"],
                "target_pair": tuple(spec["target_pair"]),
                "target_pair_label": spec["target_pair_label"],
                "target_delta": float(delta),
                "mutation_level_abs": float(level_abs),
                "mutation_level_label": f"Delta {delta:+.2f}",
                "mutation_method": mutation_method,
            })
            scenario_order += 1
    return scenario_specs


def _nearest_correlation_by_eigen_clip(matrix: np.ndarray,
                                       min_eigenvalue: float = 1e-8,
                                       max_iterations: int = 5) -> np.ndarray:
    """Return a symmetric PSD correlation matrix using eigenvalue clipping.

    This is a compact nearest-valid-correlation repair. It is deterministic and
    sufficient for Cholesky/correlation use in this notebook.
    """
    C = np.asarray(matrix, dtype=float)
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)

    for _ in range(int(max_iterations)):
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, float(min_eigenvalue))
        C = (eigvecs * eigvals) @ eigvecs.T
        C = 0.5 * (C + C.T)
        diag = np.sqrt(np.maximum(np.diag(C), float(min_eigenvalue)))
        C = C / np.outer(diag, diag)
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 1.0)

    return C


def _matrix_sqrt_psd(matrix: np.ndarray, inverse: bool = False, eps: float = 1e-8) -> np.ndarray:
    C = _nearest_correlation_by_eigen_clip(matrix, min_eigenvalue=eps)
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, eps)
    if inverse:
        weights = 1.0 / np.sqrt(eigvals)
    else:
        weights = np.sqrt(eigvals)
    return (eigvecs * weights) @ eigvecs.T


def _normal_scores(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.zeros_like(x, dtype=float)
    valid = np.isfinite(x)
    if valid.sum() <= 1 or np.nanstd(x[valid]) <= 0:
        return out
    ranks = rankdata(x[valid], method="average")
    u = ranks / (valid.sum() + 1.0)
    out[valid] = norm.ppf(u)
    return out


def _latent_matrix_from_values(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    return np.column_stack([_normal_scores(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)) for col in columns])


def _safe_corr_matrix_from_latent(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    n_cols = Z.shape[1]
    if Z.shape[0] <= 2:
        return np.eye(n_cols)
    C = np.corrcoef(Z, rowvar=False)
    if not np.all(np.isfinite(C)):
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(C, 1.0)
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return _nearest_correlation_by_eigen_clip(C)


def _safe_pair_corr(df: pd.DataFrame, col_i: str, col_j: str) -> float:
    return safe_corr(df[col_i], df[col_j])


def _assign_sorted_values_by_score(original_values: np.ndarray,
                                   target_scores: np.ndarray) -> np.ndarray:
    original_values = np.asarray(original_values, dtype=float)
    target_scores = np.asarray(target_scores, dtype=float)
    if len(original_values) <= 1:
        return original_values.copy()

    # Stable sorting makes the transformation deterministic even when ties occur.
    score_order = np.argsort(target_scores, kind="mergesort")
    sorted_values = np.sort(original_values, kind="mergesort")
    out = np.empty_like(original_values, dtype=float)
    out[score_order] = sorted_values
    return out


def _buyer_purchase_multiplier(sample_df: pd.DataFrame) -> np.ndarray:
    out = pd.to_numeric(sample_df["buyer_lmp_out"], errors="coerce").to_numpy(dtype=float)
    inn = pd.to_numeric(sample_df["buyer_lmp_in"], errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        multiplier = inn / out
    finite = np.isfinite(multiplier) & (np.abs(out) > 1e-12)
    fill_value = float(np.nanmedian(multiplier[finite])) if finite.any() else 1.0
    multiplier = np.where(finite, multiplier, fill_value)
    multiplier = np.where(np.isfinite(multiplier), multiplier, fill_value)
    return multiplier


def mutate_sample_bank_by_gaussian_rank_recoloring(sample_df: pd.DataFrame,
                                                   scenario_spec: Dict[str, object],
                                                   preserve_qh_marginals: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create one mutated scenario pool from the realized baseline pool.

    The no-mutation baseline is returned unchanged. For mutation cases, the
    four base variables [g, d, Ns, Nb_out] are Gaussianized by ranks within each
    quarter, recolored to the mutated latent correlation matrix, and then mapped
    back to the original empirical values. If preserve_qh_marginals=True, the
    rank-to-value mapping is done separately within each (quarter, hour) cell.
    """
    flat = sample_df.sort_values(["replication", "hour_index"]).reset_index(drop=True).copy()
    flat["_buyer_purchase_multiplier"] = _buyer_purchase_multiplier(flat)

    if str(scenario_spec.get("scenario_type", "baseline")) == "baseline":
        diag_rows = []
        for quarter_value, q_df in flat.groupby("quarter", sort=True):
            Z_q = _latent_matrix_from_values(q_df, VECTOR_COLUMNS)
            C_q_latent = _safe_corr_matrix_from_latent(Z_q)
            for pair in [
                ("generation", "demand", "rho(g,d)"),
                ("seller_lmp", "buyer_lmp_out", "rho(Ns,Nb_out)"),
                ("demand", "buyer_lmp_out", "rho(d,Nb_out)"),
                ("generation", "seller_lmp", "rho(g,Ns)"),
            ]:
                ii = VECTOR_COLUMNS.index(pair[0])
                jj = VECTOR_COLUMNS.index(pair[1])
                physical_corr = _safe_pair_corr(q_df, pair[0], pair[1])
                latent_corr = float(C_q_latent[ii, jj])
                diag_rows.append({
                    "scenario_name": scenario_spec["scenario_name"],
                    "scenario_type": "baseline",
                    "scenario_order": int(scenario_spec["scenario_order"]),
                    "mutation_family": scenario_spec["mutation_family"],
                    "mutation_family_label": scenario_spec["mutation_family_label"],
                    "target_pair_label": pair[2],
                    "target_var_i": pair[0],
                    "target_var_j": pair[1],
                    "target_delta": 0.0,
                    "quarter": quarter_value,
                    "n_rows_quarter": int(len(q_df)),
                    "base_latent_corr": latent_corr,
                    "pre_correction_target_corr": latent_corr,
                    "post_correction_target_corr": latent_corr,
                    "realized_physical_corr": physical_corr,
                    "realized_latent_corr": latent_corr,
                    "psd_correction_frobenius": 0.0,
                    "min_eigen_pre_correction": np.nan,
                    "min_eigen_post_correction": np.nan,
                    "mutation_method": "none",
                    "preserve_quarter_hour_marginals": bool(preserve_qh_marginals),
                })
        flat = flat.drop(columns=["_buyer_purchase_multiplier"])
        return flat, pd.DataFrame(diag_rows)

    target_pair = tuple(scenario_spec["target_pair"])
    target_delta = float(scenario_spec["target_delta"])
    i = VECTOR_COLUMNS.index(target_pair[0])
    j = VECTOR_COLUMNS.index(target_pair[1])

    mutated_values = {col: pd.to_numeric(flat[col], errors="coerce").to_numpy(dtype=float).copy() for col in VECTOR_COLUMNS}
    diag_rows = []

    for quarter_value, q_positions_index in flat.groupby("quarter", sort=True).groups.items():
        q_pos = np.asarray(list(q_positions_index), dtype=int)
        q_df = flat.iloc[q_pos].copy()
        Z = _latent_matrix_from_values(q_df, VECTOR_COLUMNS)
        C_base = _safe_corr_matrix_from_latent(Z)

        C_pre = C_base.copy()
        C_pre[i, j] = np.clip(C_base[i, j] + target_delta, -1.0, 1.0)
        C_pre[j, i] = C_pre[i, j]
        C_pre = 0.5 * (C_pre + C_pre.T)
        np.fill_diagonal(C_pre, 1.0)

        eig_pre = np.linalg.eigvalsh(C_pre)
        C_post = _nearest_correlation_by_eigen_clip(C_pre)
        eig_post = np.linalg.eigvalsh(C_post)
        correction_frob = float(np.linalg.norm(C_post - C_pre, ord="fro"))

        Z_centered = Z - np.nanmean(Z, axis=0, keepdims=True)
        S_base = _safe_corr_matrix_from_latent(Z_centered)
        W = _matrix_sqrt_psd(S_base, inverse=True)
        T = _matrix_sqrt_psd(C_post, inverse=False)
        Z_target = Z_centered @ W @ T

        # Map recolored scores back to original marginal values.
        if preserve_qh_marginals and "hour" in flat.columns:
            q_local = q_df.reset_index(drop=True)
            for col_idx, col in enumerate(VECTOR_COLUMNS):
                col_values_q = q_local[col].to_numpy(dtype=float)
                new_values_q = col_values_q.copy()
                for _, local_positions in q_local.groupby(["quarter", "hour"], sort=False).groups.items():
                    local_pos = np.asarray(list(local_positions), dtype=int)
                    new_values_q[local_pos] = _assign_sorted_values_by_score(
                        original_values=col_values_q[local_pos],
                        target_scores=Z_target[local_pos, col_idx],
                    )
                mutated_values[col][q_pos] = new_values_q
        else:
            for col_idx, col in enumerate(VECTOR_COLUMNS):
                mutated_values[col][q_pos] = _assign_sorted_values_by_score(
                    original_values=q_df[col].to_numpy(dtype=float),
                    target_scores=Z_target[:, col_idx],
                )

        # Diagnostics after the mutated values are assigned for the target pair.
        q_diag_base_df = q_df
        # Build a temporary mutated quarter frame for diagnostics.
        q_diag_mut_df = q_df.copy()
        for col in VECTOR_COLUMNS:
            q_diag_mut_df[col] = mutated_values[col][q_pos]
        Z_mut = _latent_matrix_from_values(q_diag_mut_df, VECTOR_COLUMNS)
        C_mut_latent = _safe_corr_matrix_from_latent(Z_mut)

        diag_rows.append({
            "scenario_name": scenario_spec["scenario_name"],
            "scenario_type": scenario_spec["scenario_type"],
            "scenario_order": int(scenario_spec["scenario_order"]),
            "mutation_family": scenario_spec["mutation_family"],
            "mutation_family_label": scenario_spec["mutation_family_label"],
            "target_pair_label": scenario_spec["target_pair_label"],
            "target_var_i": target_pair[0],
            "target_var_j": target_pair[1],
            "target_delta": target_delta,
            "quarter": quarter_value,
            "n_rows_quarter": int(len(q_df)),
            "base_latent_corr": float(C_base[i, j]),
            "pre_correction_target_corr": float(C_pre[i, j]),
            "post_correction_target_corr": float(C_post[i, j]),
            "realized_physical_corr": _safe_pair_corr(q_diag_mut_df, target_pair[0], target_pair[1]),
            "realized_latent_corr": float(C_mut_latent[i, j]),
            "psd_correction_frobenius": correction_frob,
            "min_eigen_pre_correction": float(np.min(eig_pre)),
            "min_eigen_post_correction": float(np.min(eig_post)),
            "mutation_method": mutation_method,
            "preserve_quarter_hour_marginals": bool(preserve_qh_marginals),
        })

    for col in VECTOR_COLUMNS:
        flat[col] = mutated_values[col]

    flat["buyer_lmp_in"] = (
        flat["_buyer_purchase_multiplier"].to_numpy(dtype=float)
        * flat["buyer_lmp_out"].to_numpy(dtype=float)
    )

    flat = flat.drop(columns=["_buyer_purchase_multiplier"])
    return flat, pd.DataFrame(diag_rows)


def _scenario_meta_for_row(scenario_spec: Dict[str, object]) -> Dict[str, object]:
    return {
        "scenario_name": scenario_spec["scenario_name"],
        "scenario_type": scenario_spec["scenario_type"],
        "scenario_order": int(scenario_spec["scenario_order"]),
        "mutation_family": scenario_spec["mutation_family"],
        "mutation_family_label": scenario_spec["mutation_family_label"],
        "target_pair_label": scenario_spec["target_pair_label"],
        "target_delta": float(scenario_spec["target_delta"]),
        "mutation_level_abs": float(scenario_spec["mutation_level_abs"]),
        "mutation_level_label": scenario_spec["mutation_level_label"],
        "mutation_method": scenario_spec["mutation_method"],
    }


def run_one_match_for_scenario(match_id: int,
                               original_df: pd.DataFrame,
                               sample_df: pd.DataFrame,
                               output_root: Path,
                               match_table_df: pd.DataFrame,
                               corr_targets: pd.DataFrame,
                               config: Dict,
                               scenario_spec: Dict[str, object],
                               mutation_diag_df: pd.DataFrame) -> Dict[str, object]:
    scenario_name = str(scenario_spec["scenario_name"])
    scenario_config = dict(config)
    scenario_config["scenario_name"] = scenario_name

    penalty_rate, buyer_purchase_mean = compute_penalty_rate(sample_df)

    grid_df_formula = optimize_ppa_contracts(sample_df=sample_df, config=scenario_config, penalty_rate=penalty_rate)
    verification_bundle = select_verified_solutions(
        sample_df=sample_df,
        grid_df=grid_df_formula,
        config=scenario_config,
        penalty_rate=penalty_rate,
    )

    formula_best_solution = verification_bundle["formula_best_solution"]
    formula_best_ppa_solution = verification_bundle["formula_best_ppa_solution"]
    best_solution = verification_bundle["verified_best_solution"]
    best_ppa_solution = verification_bundle["verified_best_ppa_solution"]
    verification_report = verification_bundle["verification_report"]
    grid_df = verification_bundle["verified_grid_df"]

    basic_stats = build_basic_stats(
        match_id=match_id,
        original_df=original_df,
        sample_df=sample_df,
        corr_targets=corr_targets,
        config=scenario_config,
    )

    if not mutation_diag_df.empty:
        # Summarize the target-pair diagnostics over quarters for convenient plotting/reporting.
        target_diag = mutation_diag_df.copy()
        for col in [
            "base_latent_corr", "pre_correction_target_corr", "post_correction_target_corr",
            "realized_physical_corr", "realized_latent_corr", "psd_correction_frobenius",
        ]:
            if col in target_diag.columns:
                basic_stats[f"mutation_{col}_mean_over_quarters"] = float(pd.to_numeric(target_diag[col], errors="coerce").mean())
                basic_stats[f"mutation_{col}_max_abs_over_quarters"] = float(pd.to_numeric(target_diag[col], errors="coerce").abs().max())
        if "min_eigen_pre_correction" in target_diag.columns:
            basic_stats["mutation_min_eigen_pre_correction_min_over_quarters"] = float(pd.to_numeric(target_diag["min_eigen_pre_correction"], errors="coerce").min())
        if "min_eigen_post_correction" in target_diag.columns:
            basic_stats["mutation_min_eigen_post_correction_min_over_quarters"] = float(pd.to_numeric(target_diag["min_eigen_post_correction"], errors="coerce").min())

    if not match_table_df.empty and "match_id" in match_table_df.columns:
        selected = match_table_df.loc[match_table_df["match_id"] == int(match_id)].copy()
        match_row = selected.iloc[0].to_dict() if not selected.empty else {}
    else:
        match_row = {}

    selected_fe_solution, fe_source = select_hourly_fe_export_solution(
        best_solution=best_solution,
        best_ppa_solution=best_ppa_solution,
        save_best_fe_even_if_infeasible=scenario_config["save_best_fe_even_if_infeasible"],
    )

    match_output_dir = output_root / "Per_Match" / f"Match_{int(match_id):04d}"
    match_output_dir.mkdir(parents=True, exist_ok=True)

    case_info_df = build_case_info_long(
        match_id=match_id,
        match_row=match_row,
        basic_stats=basic_stats,
        penalty_rate=penalty_rate,
        buyer_purchase_mean=buyer_purchase_mean,
        best_solution=best_solution,
        best_ppa_solution=best_ppa_solution,
        formula_best_solution=formula_best_solution,
        formula_best_ppa_solution=formula_best_ppa_solution,
        verification_report=verification_report,
        scenario_name=scenario_name,
        config=scenario_config,
    )
    scenario_meta_records = [
        {"section": "Mutation", "item": key, "value": value}
        for key, value in _scenario_meta_for_row(scenario_spec).items()
    ]
    if scenario_meta_records:
        case_info_df = pd.concat([case_info_df, pd.DataFrame(scenario_meta_records)], ignore_index=True)

    case_info_path = match_output_dir / f"Case_Info__{scenario_name}.csv"
    _write_csv_compact(case_info_df, case_info_path)

    best_solution_dict = to_plain_dict(best_solution)
    best_solution_dict["match_id"] = int(match_id)
    best_solution_dict.update(_scenario_meta_for_row(scenario_spec))
    best_solution_dict["penalty_rate"] = penalty_rate
    best_solution_dict["buyer_lmp_in_mean"] = buyer_purchase_mean
    best_solution_dict["unusual_contracted_volume"] = unusual_contracted_volume_flag(best_solution, sample_df)
    best_solution_dict["formula_selected_ppa_type"] = formula_best_solution.get("ppa_type")
    best_solution_dict["formula_selected_profile_type"] = formula_best_solution.get("profile_type")
    best_solution_dict["formula_selected_volume_mw"] = formula_best_solution.get("volume_mw")
    best_solution_dict["formula_selected_strike_price_mwh"] = formula_best_solution.get("strike_price_mwh")
    best_solution_dict.update(verification_report)
    best_solution_path = match_output_dir / f"Best_Solution__{scenario_name}.csv"
    _write_csv_compact(pd.DataFrame([best_solution_dict]), best_solution_path)

    best_ppa_solution_dict = to_plain_dict(best_ppa_solution) if best_ppa_solution is not None else {}
    best_ppa_solution_path = None
    if best_ppa_solution_dict:
        best_ppa_solution_dict["match_id"] = int(match_id)
        best_ppa_solution_dict.update(_scenario_meta_for_row(scenario_spec))
        best_ppa_solution_dict["penalty_rate"] = penalty_rate
        best_ppa_solution_dict["buyer_lmp_in_mean"] = buyer_purchase_mean
        best_ppa_solution_dict["unusual_contracted_volume"] = unusual_contracted_volume_flag(best_ppa_solution, sample_df)
        if formula_best_ppa_solution is not None:
            best_ppa_solution_dict["formula_best_ppa_type"] = formula_best_ppa_solution.get("ppa_type")
            best_ppa_solution_dict["formula_best_ppa_profile_type"] = formula_best_ppa_solution.get("profile_type")
            best_ppa_solution_dict["formula_best_ppa_volume_mw"] = formula_best_ppa_solution.get("volume_mw")
            best_ppa_solution_dict["formula_best_ppa_strike_price_mwh"] = formula_best_ppa_solution.get("strike_price_mwh")
        best_ppa_solution_dict.update(verification_report)
        best_ppa_solution_path = match_output_dir / f"Best_PPA_Solution__{scenario_name}.csv"
        _write_csv_compact(pd.DataFrame([best_ppa_solution_dict]), best_ppa_solution_path)

    grid_path = None
    if scenario_config["save_per_match_grid"]:
        grid_path = match_output_dir / f"Contract_Grid__{scenario_name}.csv"
        _write_csv_compact(grid_df, grid_path)

    sample_pool_path = None
    if save_mutated_sample_pools:
        sample_pool_dir = output_root / "Scenario_Pools"
        sample_pool_dir.mkdir(parents=True, exist_ok=True)
        sample_pool_path = sample_pool_dir / f"ScenarioPool_Match_{int(match_id):04d}__{scenario_name}.csv"
        _write_csv_compact(sample_df, sample_pool_path)

    hourly_fe_path = None
    hourly_fe_index_row = None
    if scenario_config["save_hourly_fe"]:
        fe_df = evaluate_hourly_fe_for_solution(
            sample_df=sample_df,
            solution=selected_fe_solution,
            config=scenario_config,
            penalty_rate=penalty_rate,
        )
        fe_subdir = output_root / "Hourly_FE"
        fe_subdir.mkdir(parents=True, exist_ok=True)
        hourly_fe_path = fe_subdir / f"Hourly_FE_Match_{int(match_id):04d}__{scenario_name}.csv"
        _write_csv_compact(fe_df, hourly_fe_path, decimal_places=int(globals().get("hourly_fe_decimal_places", 2)))

        hourly_fe_index_row = {
            "match_id": int(match_id),
            **_scenario_meta_for_row(scenario_spec),
            "hourly_fe_file": str(hourly_fe_path),
            "fe_source": fe_source,
            "exposure_level": "contract_period_replication",
            "decision_ppa_type": best_solution.get("ppa_type"),
            "decision_profile_type": best_solution.get("profile_type"),
            "decision_strike_price_mwh": best_solution.get("strike_price_mwh"),
            "decision_volume_mw": best_solution.get("volume_mw"),
            "formula_decision_ppa_type": formula_best_solution.get("ppa_type"),
            "formula_decision_profile_type": formula_best_solution.get("profile_type"),
            "formula_decision_strike_price_mwh": formula_best_solution.get("strike_price_mwh"),
            "formula_decision_volume_mw": formula_best_solution.get("volume_mw"),
            "ppa_type": selected_fe_solution.get("ppa_type"),
            "profile_type": selected_fe_solution.get("profile_type"),
            "strike_price_mwh": selected_fe_solution.get("strike_price_mwh"),
            "volume_mw": selected_fe_solution.get("volume_mw"),
            "feasible": selected_fe_solution.get("feasible"),
            "n_replications": basic_stats["n_replications"],
            "hours_per_replication": basic_stats["hours_per_replication"],
            **verification_report,
        }

    summary_row = dict(basic_stats)
    summary_row.update({
        **_scenario_meta_for_row(scenario_spec),
        "seller_utility": best_solution.get("seller_utility"),
        "buyer_utility": best_solution.get("buyer_utility"),
        "buyer_participation_slack": -float(best_solution.get("buyer_utility", np.nan)),
        "seller_mean_exposure": best_solution.get("seller_mean_exposure"),
        "buyer_mean_exposure": best_solution.get("buyer_mean_exposure"),
        "seller_exposure_variance": best_solution.get("seller_exposure_variance"),
        "buyer_exposure_variance": best_solution.get("buyer_exposure_variance"),
        "expected_seller_revenue": best_solution.get("expected_seller_revenue"),
        "ppa_type": best_solution.get("ppa_type"),
        "profile_type": best_solution.get("profile_type"),
        "selected_profile_for_switch": (
            "No Contract" if str(best_solution.get("ppa_type")) == "No Contract" else str(best_solution.get("profile_type"))
        ),
        "volume_mw": best_solution.get("volume_mw"),
        "strike_price_mwh": best_solution.get("strike_price_mwh"),
        "feasible": best_solution.get("feasible"),
        "unusual_contracted_volume": unusual_contracted_volume_flag(best_solution, sample_df),
        "best_ppa_type": best_ppa_solution.get("ppa_type") if best_ppa_solution is not None else None,
        "best_ppa_profile_type": best_ppa_solution.get("profile_type") if best_ppa_solution is not None else None,
        "best_ppa_volume_mw": best_ppa_solution.get("volume_mw") if best_ppa_solution is not None else np.nan,
        "best_ppa_strike_price_mwh": best_ppa_solution.get("strike_price_mwh") if best_ppa_solution is not None else np.nan,
        "best_ppa_feasible": best_ppa_solution.get("feasible") if best_ppa_solution is not None else None,
        "formula_selected_ppa_type": formula_best_solution.get("ppa_type"),
        "formula_selected_profile_type": formula_best_solution.get("profile_type"),
        "formula_selected_volume_mw": formula_best_solution.get("volume_mw"),
        "formula_selected_strike_price_mwh": formula_best_solution.get("strike_price_mwh"),
        "formula_best_ppa_type": formula_best_ppa_solution.get("ppa_type") if formula_best_ppa_solution is not None else None,
        "formula_best_ppa_profile_type": formula_best_ppa_solution.get("profile_type") if formula_best_ppa_solution is not None else None,
        "formula_best_ppa_volume_mw": formula_best_ppa_solution.get("volume_mw") if formula_best_ppa_solution is not None else np.nan,
        "formula_best_ppa_strike_price_mwh": formula_best_ppa_solution.get("strike_price_mwh") if formula_best_ppa_solution is not None else np.nan,
        "penalty_rate": penalty_rate,
        "buyer_lmp_in_mean": buyer_purchase_mean,
        "fe_source": fe_source,
        "exposure_level": "contract_period_replication",
        "case_info_file": str(case_info_path),
        "best_solution_file": str(best_solution_path),
        "best_ppa_solution_file": str(best_ppa_solution_path) if best_ppa_solution_path is not None else "",
        "contract_grid_file": str(grid_path) if grid_path is not None else "",
        "hourly_fe_file": str(hourly_fe_path) if hourly_fe_path is not None else "",
        "scenario_pool_file": str(sample_pool_path) if sample_pool_path is not None else "",
        **verification_report,
    })

    for key, value in match_row.items():
        summary_row[f"match__{key}"] = value

    return {
        "summary_row": summary_row,
        "best_solution_row": best_solution_dict,
        "best_ppa_solution_row": best_ppa_solution_dict,
        "hourly_fe_index_row": hourly_fe_index_row,
        "verification_report": verification_report,
    }


def _decision_signature_columns(df: pd.DataFrame) -> pd.Series:
    ppa = df.get("ppa_type", pd.Series("", index=df.index)).fillna("").astype(str)
    profile = df.get("profile_type", pd.Series("", index=df.index)).fillna("").astype(str)
    strike = pd.to_numeric(df.get("strike_price_mwh", pd.Series(np.nan, index=df.index)), errors="coerce").round(6).astype(str)
    volume = pd.to_numeric(df.get("volume_mw", pd.Series(np.nan, index=df.index)), errors="coerce").round(6).astype(str)
    return ppa + "|" + profile + "|" + strike + "|" + volume


def build_mutation_comparison_outputs(summary_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if summary_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = summary_df.copy()
    df["selected_profile_for_switch"] = np.where(
        df["ppa_type"].astype(str).eq("No Contract"),
        "No Contract",
        df["profile_type"].fillna("Unknown").astype(str),
    )
    df["decision_signature"] = _decision_signature_columns(df)

    baseline = df.loc[df["scenario_type"].astype(str).eq("baseline")].copy()
    baseline = baseline[[
        "match_id", "ppa_type", "profile_type", "selected_profile_for_switch",
        "decision_signature", "strike_price_mwh", "volume_mw",
        "seller_utility", "buyer_utility", "buyer_participation_slack",
    ]].rename(columns={
        "ppa_type": "baseline_ppa_type",
        "profile_type": "baseline_profile_type",
        "selected_profile_for_switch": "baseline_profile_for_switch",
        "decision_signature": "baseline_decision_signature",
        "strike_price_mwh": "baseline_strike_price_mwh",
        "volume_mw": "baseline_volume_mw",
        "seller_utility": "baseline_seller_utility",
        "buyer_utility": "baseline_buyer_utility",
        "buyer_participation_slack": "baseline_buyer_participation_slack",
    })

    mut = df.loc[df["scenario_type"].astype(str).eq("mutation")].copy()
    if mut.empty or baseline.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    long_df = mut.merge(baseline, on="match_id", how="left")
    long_df["full_decision_identical_to_baseline"] = (
        long_df["decision_signature"].astype(str) == long_df["baseline_decision_signature"].astype(str)
    )
    long_df["profile_identical_to_baseline"] = (
        long_df["selected_profile_for_switch"].astype(str) == long_df["baseline_profile_for_switch"].astype(str)
    )

    for col in ["strike_price_mwh", "volume_mw", "seller_utility", "buyer_utility", "buyer_participation_slack"]:
        base_col = f"baseline_{col}"
        if base_col in long_df.columns and col in long_df.columns:
            long_df[f"delta_{col}"] = pd.to_numeric(long_df[col], errors="coerce") - pd.to_numeric(long_df[base_col], errors="coerce")

    matrix_rows = []
    group_cols = [
        "scenario_name", "scenario_order", "mutation_family", "mutation_family_label",
        "target_pair_label", "target_delta", "mutation_level_abs", "mutation_level_label",
        "baseline_profile_for_switch", "selected_profile_for_switch",
    ]
    for keys, sub in long_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["count"] = int(len(sub))
        base_count = int((long_df[
            (long_df["scenario_name"].astype(str) == str(row["scenario_name"])) &
            (long_df["baseline_profile_for_switch"].astype(str) == str(row["baseline_profile_for_switch"]))
        ]).shape[0])
        row["share_within_baseline_profile"] = row["count"] / base_count if base_count else np.nan
        matrix_rows.append(row)
    switch_matrix_df = pd.DataFrame(matrix_rows)

    case_rows = []
    for keys, sub in df.groupby([
        "scenario_name", "scenario_type", "scenario_order", "mutation_family",
        "mutation_family_label", "target_pair_label", "target_delta",
        "mutation_level_abs", "mutation_level_label",
    ], dropna=False):
        row = dict(zip([
            "scenario_name", "scenario_type", "scenario_order", "mutation_family",
            "mutation_family_label", "target_pair_label", "target_delta",
            "mutation_level_abs", "mutation_level_label",
        ], keys))
        n = len(sub)
        row["n_matches"] = int(n)
        row["ppa_formation_rate"] = float((sub["ppa_type"].astype(str) != "No Contract").mean()) if n else np.nan
        for profile in ["Fix", "AsC", "AsG", "No Contract"]:
            row[f"profile_share_{profile.replace(' ', '_')}"] = float((sub["selected_profile_for_switch"].astype(str) == profile).mean()) if n else np.nan

        if str(row["scenario_type"]) == "mutation":
            cmp_sub = long_df.loc[long_df["scenario_name"].astype(str).eq(str(row["scenario_name"]))].copy()
            row["full_decision_identical_rate"] = float(cmp_sub["full_decision_identical_to_baseline"].mean()) if not cmp_sub.empty else np.nan
            row["profile_identical_rate"] = float(cmp_sub["profile_identical_to_baseline"].mean()) if not cmp_sub.empty else np.nan
            for col in ["strike_price_mwh", "volume_mw", "seller_utility", "buyer_utility", "buyer_participation_slack"]:
                dcol = f"delta_{col}"
                row[f"mean_{dcol}"] = float(pd.to_numeric(cmp_sub[dcol], errors="coerce").mean()) if dcol in cmp_sub.columns else np.nan
        else:
            row["full_decision_identical_rate"] = 1.0
            row["profile_identical_rate"] = 1.0
            for col in ["strike_price_mwh", "volume_mw", "seller_utility", "buyer_utility", "buyer_participation_slack"]:
                row[f"mean_delta_{col}"] = 0.0

        case_rows.append(row)

    case_summary_df = pd.DataFrame(case_rows).sort_values(["scenario_order", "scenario_name"]).reset_index(drop=True)
    long_df = long_df.sort_values(["mutation_family", "scenario_order", "match_id"]).reset_index(drop=True)
    switch_matrix_df = switch_matrix_df.sort_values([
        "mutation_family", "scenario_order", "baseline_profile_for_switch", "selected_profile_for_switch"
    ]).reset_index(drop=True)

    return case_summary_df, switch_matrix_df, long_df


# ## Execute calibrated sample-based simulation

# In[ ]:


# ============================================================
# Calibrated sample-based mutation simulation
# ============================================================
# This cell replaces the on-the-fly mutation loop. It reads pre-generated
# calibrated mutation samples from:
#   Code_Submission/simulation_mutation/mutation_samples
#
# It saves PPA simulation outputs to:
#   Code_Submission/simulation_mutation/Output files (Risk Neutral, Mutation, Verified)

from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# SECTION 1. USER CONTROLS
# ============================================================

# Optional absolute Code_Submission override. Leave as None unless Jupyter resolves paths incorrectly.
# Example:
# code_root_override = "/absolute/path/to/Code for submission"
code_root_override = None

mutation_samples_dir = str(Path("simulation_mutation") / "mutation_samples")
simulation_output_dir = str(Path("simulation_mutation") / "Output files (Risk Neutral, Mutation, Verified)")

# Scenario filters. Leave as None to run all scenarios in Sample_File_Index.csv.
run_match_ids = None                         # e.g. [1, 2, 3] or None
run_mutation_families = None                 # e.g. ["shape", "basis"] or None
run_target_physical_shifts = None            # e.g. [-0.10, -0.30, -0.50] or None
run_baseline = True

# If True, skip scenario output when Case_Info__<scenario>.csv already exists.
# Leave False when reproducing a full result set.
skip_existing_scenario_outputs = False

# Compact output controls.
# Numeric output tables and contract-period FE files are rounded to two decimals by default.
# Path/debug columns are removed from aggregate outputs unless needed for downstream plotting.
output_decimal_places = 2
compact_aggregate_outputs = True
aggregate_output_drop_columns = ["sample_file", "sample_source", "sample_path_rel", "code_root", "scenario_pool_file"]
hourly_fe_decimal_places = 2

# These override CONFIG values from the verified simulation functions.
CONFIG.update({
    "output_dir": simulation_output_dir,
    "skip_existing_output": bool(skip_existing_scenario_outputs),
    "save_per_match_grid": False,
    "save_hourly_fe": True,
    "save_best_fe_even_if_infeasible": True,
    "save_timestamp_in_hourly_fe": False,
    "mutation_method": "calibrated_gaussian_rank_recoloring_realized_bank_from_saved_samples",
})


# ============================================================
# SECTION 2. PATH AND SAMPLE-INDEX HELPERS
# ============================================================

NOTEBOOK_CWD = Path.cwd().resolve()

if code_root_override is not None:
    CODE_ROOT = Path(code_root_override).expanduser().resolve()
elif NOTEBOOK_CWD.name == "simulation_mutation":
    CODE_ROOT = NOTEBOOK_CWD.parent.resolve()
else:
    CODE_ROOT = NOTEBOOK_CWD.resolve()

SIM_MUTATION_ROOT = CODE_ROOT / "simulation_mutation"


def _resolve_under_code_root(path_like: str | Path) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p.resolve()
    candidates = [
        CODE_ROOT / p,
        NOTEBOOK_CWD / p,
        SIM_MUTATION_ROOT / p,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (CODE_ROOT / p).resolve()


def _sample_index_path(samples_dir: str | Path) -> Path:
    samples_path = _resolve_under_code_root(samples_dir)
    index_path = samples_path / "Sample_File_Index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Sample_File_Index.csv was not found: {index_path}")
    return index_path


def _sample_diag_path(samples_dir: str | Path) -> Path:
    samples_path = _resolve_under_code_root(samples_dir)
    diag_path = samples_path / "Mutation_Sample_Diagnostics.csv"
    return diag_path


def _resolve_sample_csv(sample_file_value) -> Path:
    text = str(sample_file_value).strip()
    if text == "" or text.lower() in {"nan", "none", "<na>"}:
        raise ValueError("Sample file value is empty.")
    p = Path(text).expanduser()
    if p.is_absolute() and p.exists():
        return p.resolve()
    candidates = [
        CODE_ROOT / p,
        NOTEBOOK_CWD / p,
        SIM_MUTATION_ROOT / p,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Sample CSV was not found for value: {sample_file_value}")


def _is_baseline_row(row: pd.Series) -> bool:
    return str(row.get("scenario_type", "")).strip().lower() == "baseline"


def _as_float_filter(values):
    if values is None:
        return None
    if isinstance(values, (int, float, np.integer, np.floating)):
        return [float(values)]
    return [float(x) for x in values]


def _numeric_filter_mask(series: pd.Series, allowed_values, atol: float = 1e-9) -> pd.Series:
    allowed = _as_float_filter(allowed_values)
    if allowed is None:
        return pd.Series(True, index=series.index)
    numeric = pd.to_numeric(series, errors="coerce")
    keep = pd.Series(False, index=series.index)
    for value in allowed:
        keep = keep | np.isclose(numeric, float(value), atol=atol)
    return keep


def load_and_filter_sample_index(samples_dir: str | Path) -> pd.DataFrame:
    index_df = read_csv_optimized(_sample_index_path(samples_dir))
    index_df.columns = [str(c).strip() for c in index_df.columns]

    required = {"match_id", "scenario_name", "scenario_type", "mutation_family"}
    missing = sorted(required - set(index_df.columns))
    if missing:
        raise KeyError(f"Sample_File_Index.csv is missing required columns: {missing}")

    index_df["match_id"] = pd.to_numeric(index_df["match_id"], errors="coerce").astype("Int64")
    index_df = index_df.dropna(subset=["match_id"]).copy()
    index_df["match_id"] = index_df["match_id"].astype(int)

    if "scenario_order" not in index_df.columns:
        index_df["scenario_order"] = 0
    index_df["scenario_order"] = pd.to_numeric(index_df["scenario_order"], errors="coerce").fillna(9999).astype(int)

    if run_match_ids is not None:
        allowed = {int(x) for x in run_match_ids}
        index_df = index_df.loc[index_df["match_id"].isin(allowed)].copy()

    baseline_mask = index_df["scenario_type"].astype(str).eq("baseline")
    if not bool(run_baseline):
        index_df = index_df.loc[~baseline_mask].copy()
        baseline_mask = index_df["scenario_type"].astype(str).eq("baseline")

    if run_mutation_families is not None:
        allowed_fam = {str(x) for x in run_mutation_families}
        index_df = index_df.loc[baseline_mask | index_df["mutation_family"].astype(str).isin(allowed_fam)].copy()
        baseline_mask = index_df["scenario_type"].astype(str).eq("baseline")

    if run_target_physical_shifts is not None and "target_physical_shift" in index_df.columns:
        keep_shift = baseline_mask | _numeric_filter_mask(index_df["target_physical_shift"], run_target_physical_shifts)
        index_df = index_df.loc[keep_shift].copy()

    if index_df.empty:
        raise ValueError("No scenarios remain after filtering Sample_File_Index.csv.")

    return index_df.sort_values(["match_id", "scenario_order", "scenario_name"]).reset_index(drop=True)


def load_sample_diagnostics(samples_dir: str | Path) -> pd.DataFrame:
    path = _sample_diag_path(samples_dir)
    if not path.exists():
        return pd.DataFrame()
    diag = read_csv_optimized(path)
    if "match_id" in diag.columns:
        diag["match_id"] = pd.to_numeric(diag["match_id"], errors="coerce").astype("Int64")
        diag = diag.dropna(subset=["match_id"]).copy()
        diag["match_id"] = diag["match_id"].astype(int)
    return diag


def _scenario_spec_from_index_row(row: pd.Series) -> dict:
    scenario_type = str(row.get("scenario_type", "mutation"))
    target_var_i = str(row.get("target_var_i", "")).strip()
    target_var_j = str(row.get("target_var_j", "")).strip()
    target_pair = None if scenario_type == "baseline" or target_var_i == "" or target_var_j == "" else (target_var_i, target_var_j)

    def _num(name, default=np.nan):
        return float(pd.to_numeric(pd.Series([row.get(name, default)]), errors="coerce").iloc[0])

    spec = {
        "scenario_name": str(row.get("scenario_name")),
        "scenario_type": scenario_type,
        "scenario_order": int(pd.to_numeric(pd.Series([row.get("scenario_order", 9999)]), errors="coerce").fillna(9999).iloc[0]),
        "mutation_family": str(row.get("mutation_family", "baseline")),
        "mutation_family_label": str(row.get("mutation_family_label", row.get("mutation_family", ""))),
        "target_pair": target_pair,
        "target_pair_label": str(row.get("target_pair_label", "none")),
        "target_delta": _num("target_delta", 0.0),
        "mutation_level_abs": _num("mutation_level_abs", 0.0),
        "mutation_level_label": str(row.get("mutation_level_label", "")),
        "mutation_method": str(row.get("mutation_method", "")),
    }

    # Preserve calibrated physical-shift metadata for downstream plotting/tables.
    extra_cols = [
        "target_var_i", "target_var_j", "target_physical_shift", "desired_physical_shift_abs",
        "calibrated_latent_delta", "calibration_error", "calibration_feasible",
        "preserve_marginals_mode",
    ]
    for col in extra_cols:
        if col in row.index:
            value = row.get(col)
            if col in {
                "target_physical_shift", "desired_physical_shift_abs", "calibrated_latent_delta",
                "calibration_error"
            }:
                spec[col] = _num(col, np.nan)
            else:
                spec[col] = value
    return spec


def build_scenario_specs_from_index(index_df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, sub in index_df.sort_values(["scenario_order", "scenario_name"]).groupby("scenario_name", sort=False):
        first = sub.iloc[0]
        rows.append(_scenario_spec_from_index_row(first))
    return sorted(rows, key=lambda s: (int(s.get("scenario_order", 9999)), str(s.get("scenario_name", ""))))


# Override the earlier mutation metadata helper so calibrated physical-shift columns
# are carried into Simulation_Plot_Data_All_Matches.csv, Hourly_FE_Index.csv, and summaries.
def _scenario_meta_for_row(scenario_spec: Dict[str, object]) -> Dict[str, object]:
    base = {
        "scenario_name": scenario_spec.get("scenario_name"),
        "scenario_type": scenario_spec.get("scenario_type"),
        "scenario_order": int(scenario_spec.get("scenario_order", 9999)),
        "mutation_family": scenario_spec.get("mutation_family"),
        "mutation_family_label": scenario_spec.get("mutation_family_label"),
        "target_pair_label": scenario_spec.get("target_pair_label"),
        "target_delta": float(scenario_spec.get("target_delta", 0.0)),  # latent delta actually applied
        "mutation_level_abs": float(scenario_spec.get("mutation_level_abs", 0.0)),
        "mutation_level_label": scenario_spec.get("mutation_level_label"),
        "mutation_method": scenario_spec.get("mutation_method"),
    }
    for col in [
        "target_var_i", "target_var_j", "target_physical_shift", "desired_physical_shift_abs",
        "calibrated_latent_delta", "calibration_error", "calibration_feasible",
        "preserve_marginals_mode",
    ]:
        if col in scenario_spec:
            base[col] = scenario_spec.get(col)
    return base


# ============================================================
# SECTION 3. EXECUTE SAMPLE-BASED SIMULATION
# ============================================================

sample_index_df = load_and_filter_sample_index(mutation_samples_dir)
sample_diag_df = load_sample_diagnostics(mutation_samples_dir)
scenario_specs = build_scenario_specs_from_index(sample_index_df)

match_dir = resolve_match_dir(CONFIG["match_folder_name"])
sample_root = resolve_sample_root(CONFIG["sample_root_dir"], match_dir=match_dir)
match_table_path = resolve_match_table(CONFIG["match_table_file"])
match_table_df = load_match_table(match_table_path, sheet_name=CONFIG["match_table_sheet"])
corr_targets_path = resolve_code2_corr_file(CONFIG["code2_correlation_file"], match_dir=match_dir)
corr_targets_df = load_code2_corr_targets(corr_targets_path)

output_root = resolve_output_root_dir(CONFIG["output_dir"])
output_root.mkdir(parents=True, exist_ok=True)
(output_root / "Per_Match").mkdir(parents=True, exist_ok=True)
(output_root / "Hourly_FE").mkdir(parents=True, exist_ok=True)
(output_root / "Diagnostics").mkdir(parents=True, exist_ok=True)

print("Resolved paths:")
print("  Code root           :", CODE_ROOT)
print("  Sample index        :", _sample_index_path(mutation_samples_dir))
print("  Output root         :", output_root)
print("  Historical matches  :", match_dir)
print("  Baseline sample root:", sample_root)
print()
print(f"Matches selected      : {sample_index_df['match_id'].nunique()}")
print(f"Scenarios selected    : {len(scenario_specs)}")
print("Scenario list:")
for s in scenario_specs:
    print(
        f"  {int(s['scenario_order']):02d} | {s['scenario_name']} | "
        f"family={s.get('mutation_family')} | "
        f"target_physical_shift={s.get('target_physical_shift', np.nan)} | "
        f"latent_delta={s.get('calibrated_latent_delta', s.get('target_delta', np.nan))}"
    )

summary_rows = []
best_solution_rows = []
best_ppa_solution_rows = []
hourly_fe_index_rows = []
run_status_rows = []
mutation_diag_rows = []

for counter, match_id in enumerate(sorted(sample_index_df["match_id"].unique().tolist()), start=1):
    print(f"[{counter}/{sample_index_df['match_id'].nunique()}] match_id={int(match_id):04d}")
    try:
        original_df = load_original_match_df(int(match_id), match_dir)
    except Exception as exc:
        run_status_rows.append({
            "match_id": int(match_id),
            "scenario_name": "",
            "scenario_type": "",
            "status": "failed_loading_original_match",
            "error_message": str(exc),
        })
        print(f"    failed loading original match: {exc}")
        continue

    match_rows = sample_index_df.loc[sample_index_df["match_id"].eq(int(match_id))].copy()
    match_rows = match_rows.sort_values(["scenario_order", "scenario_name"]).reset_index(drop=True)

    for scenario_counter, (_, index_row) in enumerate(match_rows.iterrows(), start=1):
        scenario_spec = _scenario_spec_from_index_row(index_row)
        scenario_name = str(scenario_spec["scenario_name"])
        print(f"    [{scenario_counter}/{len(match_rows)}] {scenario_name}")

        try:
            if _is_baseline_row(index_row) and str(index_row.get("sample_source", "")).strip() == "baseline_sample_root":
                scenario_sample_df = load_baseline_sample_bank(int(match_id), sample_root)
            else:
                sample_path_value = index_row.get("sample_path_rel", index_row.get("sample_file", ""))
                sample_csv = _resolve_sample_csv(sample_path_value)
                scenario_sample_df = read_csv_optimized(sample_csv)

            if not sample_diag_df.empty:
                diag_df = sample_diag_df.loc[
                    (sample_diag_df["match_id"].eq(int(match_id))) &
                    (sample_diag_df["scenario_name"].astype(str).eq(scenario_name))
                ].copy()
            else:
                diag_df = pd.DataFrame()

            if not diag_df.empty:
                mutation_diag_rows.extend(diag_df.to_dict("records"))

            out = run_one_match_for_scenario(
                match_id=int(match_id),
                original_df=original_df,
                sample_df=scenario_sample_df,
                output_root=output_root,
                match_table_df=match_table_df,
                corr_targets=corr_targets_df,
                config=CONFIG,
                scenario_spec=scenario_spec,
                mutation_diag_df=diag_df,
            )

            summary_rows.append(out["summary_row"])
            best_solution_rows.append(out["best_solution_row"])
            if out["best_ppa_solution_row"]:
                best_ppa_solution_rows.append(out["best_ppa_solution_row"])
            if out["hourly_fe_index_row"] is not None:
                hourly_fe_index_rows.append(out["hourly_fe_index_row"])

            run_status_rows.append({
                "match_id": int(match_id),
                **_scenario_meta_for_row(scenario_spec),
                "status": "success",
                "error_message": "",
                "decision_changed_after_verification": out["verification_report"].get("formula_best_changed_after_verification"),
                "formula_best_verification_passed": out["verification_report"].get("formula_best_verification_passed"),
                "n_exact_candidates_evaluated": out["verification_report"].get("n_exact_candidates_evaluated"),
            })

            print(
                "       selected="
                f"{out['best_solution_row'].get('ppa_type')} / {out['best_solution_row'].get('profile_type')} | "
                f"seller={out['best_solution_row'].get('seller_utility')} | "
                f"buyer={out['best_solution_row'].get('buyer_utility')}"
            )

        except Exception as exc:
            run_status_rows.append({
                "match_id": int(match_id),
                **_scenario_meta_for_row(scenario_spec),
                "status": "failed_scenario",
                "error_message": str(exc),
            })
            print(f"       failed scenario: {exc}")

summary_df = pd.DataFrame(summary_rows)
best_solution_df = pd.DataFrame(best_solution_rows)
best_ppa_solution_df = pd.DataFrame(best_ppa_solution_rows)
hourly_fe_index_df = pd.DataFrame(hourly_fe_index_rows)
run_status_df = pd.DataFrame(run_status_rows)
mutation_diag_df = pd.DataFrame(mutation_diag_rows)

case_summary_df, switch_matrix_df, switch_long_df = build_mutation_comparison_outputs(summary_df)

# Merge calibrated scenario metadata into aggregate switch/case tables when available.
scenario_meta_cols = [
    "scenario_name", "target_physical_shift", "desired_physical_shift_abs",
    "calibrated_latent_delta", "calibration_error", "calibration_feasible",
    "preserve_marginals_mode",
]
available_meta_cols = [c for c in scenario_meta_cols if c in sample_index_df.columns]
scenario_meta_df = sample_index_df[available_meta_cols].drop_duplicates("scenario_name") if available_meta_cols else pd.DataFrame()

if not scenario_meta_df.empty:
    if not case_summary_df.empty and "scenario_name" in case_summary_df.columns:
        existing_extra = [c for c in scenario_meta_df.columns if c in case_summary_df.columns and c != "scenario_name"]
        case_summary_df = case_summary_df.drop(columns=existing_extra, errors="ignore").merge(
            scenario_meta_df, on="scenario_name", how="left"
        )
    if not switch_matrix_df.empty and "scenario_name" in switch_matrix_df.columns:
        existing_extra = [c for c in scenario_meta_df.columns if c in switch_matrix_df.columns and c != "scenario_name"]
        switch_matrix_df = switch_matrix_df.drop(columns=existing_extra, errors="ignore").merge(
            scenario_meta_df, on="scenario_name", how="left"
        )
    if not switch_long_df.empty and "scenario_name" in switch_long_df.columns:
        existing_extra = [c for c in scenario_meta_df.columns if c in switch_long_df.columns and c != "scenario_name"]
        switch_long_df = switch_long_df.drop(columns=existing_extra, errors="ignore").merge(
            scenario_meta_df, on="scenario_name", how="left"
        )

def _compact_table_for_export(df: pd.DataFrame,
                              *,
                              drop_columns: Optional[List[str]] = None,
                              decimal_places: Optional[int] = None) -> pd.DataFrame:
    out = df.copy()
    if drop_columns:
        out = out.drop(columns=[c for c in drop_columns if c in out.columns], errors="ignore")
    if decimal_places is not None:
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            out[numeric_cols] = out[numeric_cols].round(int(decimal_places))
    return out


def _write_csv_compact(df: pd.DataFrame,
                       path: Path,
                       *,
                       drop_columns: Optional[List[str]] = None,
                       decimal_places: Optional[int] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = _compact_table_for_export(
        df,
        drop_columns=drop_columns if compact_aggregate_outputs else None,
        decimal_places=decimal_places,
    )
    float_format = None if decimal_places is None else f"%.{int(decimal_places)}f"
    out.to_csv(path, index=False, float_format=float_format)


summary_path = output_root / "Simulation_Plot_Data_All_Matches.csv"
best_solution_path = output_root / "Simulation_Best_Solutions_All_Matches.csv"
best_ppa_solution_path = output_root / "Simulation_Best_PPA_Solutions_All_Matches.csv"
hourly_fe_index_path = output_root / "Hourly_FE_Index.csv"
run_status_path = output_root / "Simulation_Run_Status.csv"
mutation_diag_path = output_root / "Mutation_Correlation_Diagnostics.csv"
case_summary_path = output_root / "Mutation_Case_Summary.csv"
switch_matrix_path = output_root / "Mutation_Profile_Switch_Matrix.csv"
switch_long_path = output_root / "Mutation_Profile_Switch_Long.csv"
config_path = output_root / "Simulation_Config.json"

_write_csv_compact(summary_df, summary_path, drop_columns=aggregate_output_drop_columns, decimal_places=output_decimal_places)
_write_csv_compact(best_solution_df, best_solution_path, drop_columns=aggregate_output_drop_columns, decimal_places=output_decimal_places)
_write_csv_compact(best_ppa_solution_df, best_ppa_solution_path, drop_columns=aggregate_output_drop_columns, decimal_places=output_decimal_places)
_write_csv_compact(hourly_fe_index_df, hourly_fe_index_path, drop_columns=aggregate_output_drop_columns, decimal_places=output_decimal_places)
_write_csv_compact(run_status_df, run_status_path, drop_columns=aggregate_output_drop_columns, decimal_places=output_decimal_places)
_write_csv_compact(mutation_diag_df, mutation_diag_path, drop_columns=aggregate_output_drop_columns, decimal_places=output_decimal_places)
_write_csv_compact(case_summary_df, case_summary_path, drop_columns=aggregate_output_drop_columns, decimal_places=output_decimal_places)
_write_csv_compact(switch_matrix_df, switch_matrix_path, drop_columns=aggregate_output_drop_columns, decimal_places=output_decimal_places)
_write_csv_compact(switch_long_df, switch_long_path, drop_columns=aggregate_output_drop_columns, decimal_places=output_decimal_places)

def _to_json_safe(value):
    """Recursively convert notebook config objects to standard JSON types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_to_json_safe(x) for x in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA or pd.isna(value) is True:
        return None
    if isinstance(value, dict):
        return {str(_to_json_safe(k)): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(x) for x in value]
    return str(value)


json_ready_config = dict(CONFIG)
json_ready_config["mutation_samples_dir"] = str(_resolve_under_code_root(mutation_samples_dir))
json_ready_config["run_match_ids"] = run_match_ids
json_ready_config["run_mutation_families"] = run_mutation_families
json_ready_config["run_target_physical_shifts"] = run_target_physical_shifts
json_ready_config["output_decimal_places"] = output_decimal_places
json_ready_config["compact_aggregate_outputs"] = compact_aggregate_outputs
json_ready_config["aggregate_output_drop_columns"] = aggregate_output_drop_columns
json_ready_config["scenario_specs"] = scenario_specs

json_ready_config = _to_json_safe(json_ready_config)
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(json_ready_config, f, indent=2)

print()
print("Saved sample-based calibrated mutation simulation outputs:")
print("  Plot-ready summary       :", summary_path)
print("  Best solutions           :", best_solution_path)
print("  Best PPA solutions       :", best_ppa_solution_path)
print("  Contract-period FE index          :", hourly_fe_index_path)
print("  Run status               :", run_status_path)
print("  Correlation diagnostics  :", mutation_diag_path)
print("  Case summary             :", case_summary_path)
print("  Profile switch matrix    :", switch_matrix_path)
print("  Profile switch long      :", switch_long_path)
print("  Config                   :", config_path)

if not run_status_df.empty:
    print()
    print("Run-status counts:")
    print(run_status_df["status"].value_counts(dropna=False).to_string())

if not case_summary_df.empty:
    print()
    print("Mutation case summary:")
    display_cols = [
        "scenario_name", "target_physical_shift", "calibrated_latent_delta",
        "n_matches", "ppa_formation_rate", "profile_share_Fix", "profile_share_AsC",
        "profile_share_AsG", "profile_share_No_Contract",
        "full_decision_identical_rate", "profile_identical_rate",
    ]
    display_cols = [c for c in display_cols if c in case_summary_df.columns]
    print(case_summary_df[display_cols].to_string(index=False))

summary_df.head()
