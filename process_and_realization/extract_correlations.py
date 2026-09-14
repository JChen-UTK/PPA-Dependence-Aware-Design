from pathlib import Path
import json
import re
import warnings
from itertools import combinations

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
from scipy import stats

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


"""
Code 2 (stable patch)
---------------------
Robust correlation extraction for the simplified baseline.

Stability patch:
  - guards against constant-series and all-tie issues in correlation estimation
  - suppresses correlation warnings from degenerate columns and replaces those
    entries with neutral values before shrinkage
  - keeps the same output files and folder structure as the original code

Output folder:
    <Input data and files>/Code2_Correlation
"""


# ============================================================
# 1) Settings
# ============================================================
SEARCH_ROOTS = _build_search_roots()

MATCH_FOLDER_NAME = "Match files of original data"
CODE1_OUTPUT_SUBFOLDER = "Marginal_Fit"
CODE2_OUTPUT_SUBFOLDER = "Correlation"

VARIABLES = ["generation", "demand", "seller_lmp", "buyer_lmp_out"]
PROFILE_KEYS = ["quarter", "hour"]

POSITIVE_EPS = 1.0
WINSOR_LOWER = 0.005
WINSOR_UPPER = 0.995
CORR_SHRINK_K = 200.0
MIN_MATCH_OBS_FOR_RAW = 30
CORR_EIG_FLOOR = 1e-8


# ============================================================
# 2) Path helpers
# ============================================================
def resolve_existing_dir(candidates, description):
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    raise FileNotFoundError(
        f"Could not find {description}. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def resolve_match_dir():
    candidates = []
    for root in SEARCH_ROOTS:
        candidates.extend([
            root / "Input data and files" / MATCH_FOLDER_NAME,
            root / "Data and files" / MATCH_FOLDER_NAME,
            root / MATCH_FOLDER_NAME,
        ])
    return resolve_existing_dir(candidates, f"match folder '{MATCH_FOLDER_NAME}'")


def resolve_output_root(match_dir):
    preferred = [
        match_dir.parent,
    ]
    for root in SEARCH_ROOTS:
        preferred.extend([
            root / "Input data and files",
            root / "Data and files",
            root,
        ])
    for path in preferred:
        if path.exists() and path.is_dir():
            return path
    return match_dir.parent


def extract_match_id(path: Path):
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else path.stem


# ============================================================
# 3) Data preparation helpers
# ============================================================
def finite_series(x):
    s = pd.Series(x, dtype="float64")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s


def winsorize_series(x, lower_q=WINSOR_LOWER, upper_q=WINSOR_UPPER):
    s = finite_series(x)
    out = s.copy()
    valid = s.dropna()
    if valid.empty:
        return out
    lo = float(valid.quantile(lower_q))
    hi = float(valid.quantile(upper_q))
    out.loc[valid.index] = valid.clip(lower=lo, upper=hi)
    return out


def gaussianize_series(x):
    s = finite_series(x)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    valid = s.dropna()
    if valid.empty:
        return out
    ranks = valid.rank(method="average")
    u = ranks / (len(valid) + 1.0)
    u = np.clip(u, 1e-6, 1.0 - 1e-6)
    out.loc[valid.index] = stats.norm.ppf(u)
    return out


def nearest_correlation_matrix(matrix):
    A = np.asarray(matrix, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Correlation matrix must be square.")
    A = np.where(np.isfinite(A), A, 0.0)
    A = 0.5 * (A + A.T)
    try:
        vals, vecs = np.linalg.eigh(A)
    except np.linalg.LinAlgError:
        return np.eye(A.shape[0], dtype=float)
    vals = np.clip(vals, CORR_EIG_FLOOR, None)
    B = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(B), CORR_EIG_FLOOR, None))
    B = B / np.outer(d, d)
    B = np.where(np.isfinite(B), B, 0.0)
    np.fill_diagonal(B, 1.0)
    return B


def safe_pair_corr(x, y, method="pearson"):
    tmp = pd.DataFrame({"x": finite_series(x), "y": finite_series(y)})
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(tmp))
    if n < 2:
        return np.nan, n
    if tmp["x"].nunique(dropna=True) <= 1 or tmp["y"].nunique(dropna=True) <= 1:
        return np.nan, n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        val = tmp["x"].corr(tmp["y"], method=method)
    return (float(val) if pd.notna(val) else np.nan), n


def safe_corr(df):
    clean = df.copy()
    for col in clean.columns:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan)
    if clean.shape[0] < 2:
        return np.eye(clean.shape[1], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        arr = clean.corr().to_numpy(dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    np.fill_diagonal(arr, 1.0)
    return nearest_correlation_matrix(arr)


def safe_positive_part(x):
    arr = np.asarray(pd.to_numeric(x, errors="coerce"), dtype=float)
    floor = -POSITIVE_EPS + 1e-9
    mask = np.isfinite(arr)
    arr = np.where(mask, np.maximum(arr, floor), np.nan)
    return arr


def compute_profiles_and_residuals_from_match_dir(match_dir):
    def make_profile(x):
        s = finite_series(x).dropna()
        return float(s.median()) if not s.empty else np.nan

    def compute_residual(raw_value, profile_value, transform):
        x = np.asarray(pd.to_numeric(raw_value, errors="coerce"), dtype=float)
        p = np.asarray(pd.to_numeric(profile_value, errors="coerce"), dtype=float)
        if transform == "log_ratio":
            x = safe_positive_part(x)
            p = safe_positive_part(p)
            ratio = (x + POSITIVE_EPS) / (p + POSITIVE_EPS)
            ratio = np.where(np.isfinite(ratio), np.clip(ratio, np.exp(-20.0), np.exp(20.0)), np.nan)
            return np.log(ratio)
        if transform == "additive":
            return x - p
        raise ValueError(f"Unsupported transform: {transform}")

    variable_transforms = {
        "generation": "log_ratio",
        "demand": "log_ratio",
        "seller_lmp": "additive",
        "buyer_lmp_out": "additive",
    }

    profile_rows = []
    residual_rows = []

    for path in sorted(match_dir.glob("*.csv")):
        df = read_csv_optimized(path)
        required = ["timestamp", "hour", "quarter"] + VARIABLES
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"{path.name} missing required columns: {missing}")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
        df = df.dropna(subset=["timestamp", "hour", "quarter"]).copy()
        df["hour"] = df["hour"].astype(int)
        df["match_id"] = extract_match_id(path)

        match_id = df["match_id"].iloc[0]
        for variable in VARIABLES:
            p = (
                df.groupby(PROFILE_KEYS)[variable]
                .apply(make_profile)
                .reset_index(name="profile_value")
            )
            p["match_id"] = match_id
            p["variable"] = variable
            profile_rows.append(p)

            tmp = df[["match_id", "timestamp", "quarter", "hour", variable]].copy()
            tmp = tmp.merge(p, on=["match_id", "quarter", "hour"], how="left")
            tmp["variable"] = variable
            tmp["raw_value"] = pd.to_numeric(tmp[variable], errors="coerce")
            tmp["residual"] = compute_residual(
                tmp["raw_value"], tmp["profile_value"], variable_transforms[variable]
            )
            residual_rows.append(
                tmp[["match_id", "timestamp", "quarter", "hour", "variable", "profile_value", "raw_value", "residual"]]
            )

    profiles_df = pd.concat(profile_rows, ignore_index=True)
    residual_df = pd.concat(residual_rows, ignore_index=True)
    residual_df["residual"] = pd.to_numeric(residual_df["residual"], errors="coerce")
    residual_df.loc[~np.isfinite(residual_df["residual"]), "residual"] = np.nan
    return profiles_df, residual_df


# ============================================================
# 4) Main
# ============================================================
def main():
    match_dir = resolve_match_dir()
    output_root = resolve_output_root(match_dir)

    code1_dir = output_root / CODE1_OUTPUT_SUBFOLDER
    output_dir = output_root / CODE2_OUTPUT_SUBFOLDER
    output_dir.mkdir(parents=True, exist_ok=True)

    if (code1_dir / "residual_panel.csv").exists() and (code1_dir / "representative_profiles.csv").exists():
        residual_df = read_csv_optimized(code1_dir / "residual_panel.csv")
        profiles_df = read_csv_optimized(code1_dir / "representative_profiles.csv")
    else:
        profiles_df, residual_df = compute_profiles_and_residuals_from_match_dir(match_dir)

    residual_df["timestamp"] = pd.to_datetime(residual_df["timestamp"], errors="coerce")
    residual_df["residual"] = pd.to_numeric(residual_df["residual"], errors="coerce")
    residual_df.loc[~np.isfinite(residual_df["residual"]), "residual"] = np.nan

    # ------------------------------------------------------------
    # 4.1 Profile correlations (structural, based on representative profiles)
    # ------------------------------------------------------------
    profile_pair_rows = []
    for match_id, g in profiles_df.groupby("match_id"):
        wide = (
            g.pivot_table(
                index=["quarter", "hour"],
                columns="variable",
                values="profile_value",
                aggfunc="first",
            )
            .reset_index()
        )
        for vi, vj in combinations(VARIABLES, 2):
            pearson, n = safe_pair_corr(wide[vi], wide[vj], method="pearson")
            spearman, _ = safe_pair_corr(wide[vi], wide[vj], method="spearman")
            profile_pair_rows.append(
                {
                    "match_id": match_id,
                    "var_i": vi,
                    "var_j": vj,
                    "n_profile_cells": n,
                    "pearson_profile": pearson,
                    "spearman_profile": spearman,
                }
            )

    profile_pair_df = pd.DataFrame(profile_pair_rows)
    profile_pair_df.to_csv(output_dir / "profile_pairwise_correlations.csv", index=False)

    # ------------------------------------------------------------
    # 4.2 Residual correlations (robust, for simulation)
    # ------------------------------------------------------------
    wide_resid = (
        residual_df.pivot_table(
            index=["match_id", "timestamp"],
            columns="variable",
            values="residual",
            aggfunc="first",
        )
        .reset_index()
    )

    for var in VARIABLES:
        wide_resid[var] = wide_resid.groupby("match_id")[var].transform(winsorize_series)

    wide_gauss = wide_resid.copy()
    for var in VARIABLES:
        wide_gauss[var] = wide_gauss.groupby("match_id")[var].transform(gaussianize_series)

    pooled_complete = wide_gauss.dropna(subset=VARIABLES)
    if len(pooled_complete) >= 2:
        pooled_matrix = safe_corr(pooled_complete[VARIABLES])
    else:
        pooled_matrix = np.eye(len(VARIABLES), dtype=float)

    pooled_long = []
    for i, vi in enumerate(VARIABLES):
        for j, vj in enumerate(VARIABLES):
            pooled_long.append(
                {
                    "var_i": vi,
                    "var_j": vj,
                    "corr_pooled": float(pooled_matrix[i, j]),
                }
            )
    pooled_long_df = pd.DataFrame(pooled_long)
    pooled_long_df.to_csv(output_dir / "pooled_residual_correlation_matrix.csv", index=False)

    pair_rows = []
    target_long_rows = []
    target_wide_rows = []

    for match_id, g_raw in wide_resid.groupby("match_id"):
        g_gauss = wide_gauss.loc[g_raw.index].copy()

        raw_complete = g_gauss.dropna(subset=VARIABLES)
        n_complete = int(len(raw_complete))

        if n_complete >= 2:
            raw_matrix = safe_corr(raw_complete[VARIABLES])
        else:
            raw_matrix = pooled_matrix.copy()

        shrink_weight = float(n_complete / (n_complete + CORR_SHRINK_K)) if n_complete > 0 else 0.0
        if n_complete < MIN_MATCH_OBS_FOR_RAW:
            shrink_weight = min(shrink_weight, 0.25)

        target_matrix = nearest_correlation_matrix(
            shrink_weight * raw_matrix + (1.0 - shrink_weight) * pooled_matrix
        )

        wide_row = {"match_id": match_id, "n_complete_obs": n_complete, "shrink_weight": shrink_weight}
        for i, vi in enumerate(VARIABLES):
            for j, vj in enumerate(VARIABLES):
                target_long_rows.append(
                    {
                        "match_id": match_id,
                        "var_i": vi,
                        "var_j": vj,
                        "corr_target": float(target_matrix[i, j]),
                    }
                )
            for j in range(i + 1, len(VARIABLES)):
                vj = VARIABLES[j]
                pair_raw = g_raw[[vi, vj]].dropna()
                pearson, n_pair = safe_pair_corr(pair_raw[vi], pair_raw[vj], method="pearson")
                spearman, _ = safe_pair_corr(pair_raw[vi], pair_raw[vj], method="spearman")

                pair_rows.append(
                    {
                        "match_id": match_id,
                        "var_i": vi,
                        "var_j": vj,
                        "n_pair_obs": n_pair,
                        "n_complete_obs": n_complete,
                        "pearson_residual": pearson,
                        "spearman_residual": spearman,
                        "gaussian_corr_raw": float(raw_matrix[i, j]),
                        "gaussian_corr_shrunk": float(target_matrix[i, j]),
                        "pooled_gaussian_corr": float(pooled_matrix[i, j]),
                        "shrink_weight": shrink_weight,
                    }
                )
                wide_row[f"{vi}__{vj}"] = float(target_matrix[i, j])

        target_wide_rows.append(wide_row)

    pair_df = pd.DataFrame(pair_rows).sort_values(["match_id", "var_i", "var_j"])
    target_long_df = pd.DataFrame(target_long_rows).sort_values(["match_id", "var_i", "var_j"])
    target_wide_df = pd.DataFrame(target_wide_rows).sort_values(["match_id"])

    pair_df.to_csv(output_dir / "residual_pairwise_correlations.csv", index=False)
    target_long_df.to_csv(output_dir / "correlation_parameter_bank_long.csv", index=False)
    target_wide_df.to_csv(output_dir / "correlation_parameter_bank_wide.csv", index=False)

    summary = {
        "match_folder": str(match_dir),
        "code1_folder": str(code1_dir),
        "output_folder": str(output_dir),
        "variables": VARIABLES,
        "winsor_limits": [WINSOR_LOWER, WINSOR_UPPER],
        "corr_shrink_k": CORR_SHRINK_K,
        "min_match_obs_for_raw": MIN_MATCH_OBS_FOR_RAW,
        "corr_eig_floor": CORR_EIG_FLOOR,
        "n_matches": int(target_wide_df["match_id"].nunique()) if not target_wide_df.empty else 0,
        "notes": [
            "Profile correlations are computed on representative quarter x hour profiles and describe structural co-shape.",
            "Residual correlations are computed on de-profiled residuals after rank-to-Gaussian transformation.",
            "Match-level residual correlation matrices are shrunk toward the pooled residual matrix to reduce noise.",
            "Degenerate or constant columns are handled explicitly so that correlation warnings do not break the pipeline.",
            "The shrunk residual correlation matrix is the parameter bank intended for Code 3 sampling.",
        ],
    }
    with open(output_dir / "code2_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Output folder: {output_dir}")
    print("\nSaved files:")
    for name in [
        "profile_pairwise_correlations.csv",
        "pooled_residual_correlation_matrix.csv",
        "residual_pairwise_correlations.csv",
        "correlation_parameter_bank_long.csv",
        "correlation_parameter_bank_wide.csv",
        "code2_run_summary.json",
    ]:
        print(f" - {output_dir / name}")


if __name__ == "__main__":
    main()
