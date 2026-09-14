from pathlib import Path
import json
import re
import warnings

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
Code 1 (stable patch)
---------------------
Residual-based marginal distribution fitting and parameter derivation for:
  - generation
  - demand
  - nodal price group (seller_lmp + buyer_lmp_out only; buyer_lmp_in excluded)

Stability patch:
  - replaces the fragile direct Cramér–von Mises usage with a manual statistic
    plus guarded SciPy p-value evaluation
  - clips pathological fitted shape parameters for t and skewnorm
  - falls back to robust median/MAD-based parameters when MLE fitting is unstable
  - suppresses SciPy hyp-test runtime warnings by handling them explicitly

Output folder:
    <Input data and files>/Code1_Marginal_Fit
"""


# ============================================================
# 1) Settings
# ============================================================
SEARCH_ROOTS = _build_search_roots()

MATCH_FOLDER_NAME = "Match files of original data"
CODE1_OUTPUT_SUBFOLDER = "Marginal_Fit"

PROFILE_KEYS = ["quarter", "hour"]
PROFILE_STAT = "median"   # "median" or "trimmed_mean"
TRIM_PROP = 0.10

POSITIVE_EPS = 1.0
WINSOR_LOWER = 0.005
WINSOR_UPPER = 0.995
MIN_CELL_OBS_DIRECT = 30
MIN_CLIP_OBS_CELL = 5
SHRINK_K = 200.0
SIMPLICITY_BIC_DELTA = 10.0
MIN_SCALE = 1e-6
CDF_EPS = 1e-12
LOGRATIO_RESIDUAL_CLIP = 20.0
CVM_PVALUE_MAX_N = 20000
T_DF_MIN = 2.10
T_DF_MAX = 200.0
SKEWNORM_A_MAX = 30.0

VARIABLE_SPECS = {
    "generation": {
        "sampling_group": "generation",
        "transform": "log_ratio",
        "allow_negative": False,
    },
    "demand": {
        "sampling_group": "demand",
        "transform": "log_ratio",
        "allow_negative": False,
    },
    "seller_lmp": {
        "sampling_group": "nodal_price",
        "transform": "additive",
        "allow_negative": True,
    },
    "buyer_lmp_out": {
        "sampling_group": "nodal_price",
        "transform": "additive",
        "allow_negative": True,
    },
}

SAMPLING_GROUPS = {
    "generation": {
        "variables": ["generation"],
        "candidates": ["norm", "t", "skewnorm", "logistic"],
    },
    "demand": {
        "variables": ["demand"],
        "candidates": ["norm", "t", "skewnorm", "logistic"],
    },
    "nodal_price": {
        "variables": ["seller_lmp", "buyer_lmp_out"],
        "candidates": ["norm", "t", "skewnorm", "logistic", "laplace"],
    },
}

FAMILY_OBJECTS = {
    "norm": stats.norm,
    "t": stats.t,
    "skewnorm": stats.skewnorm,
    "logistic": stats.logistic,
    "laplace": stats.laplace,
}

FAMILY_NPARAMS = {
    "norm": 2,
    "t": 3,
    "skewnorm": 3,
    "logistic": 2,
    "laplace": 2,
}

FAMILY_COMPLEXITY = {
    "norm": 1,
    "logistic": 1,
    "laplace": 1,
    "t": 2,
    "skewnorm": 2,
}


# ============================================================
# 2) Helpers
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


def finite_series(x):
    s = pd.Series(x, dtype="float64")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s.dropna()


def trimmed_mean(x, prop=TRIM_PROP):
    arr = np.asarray(finite_series(x), dtype=float)
    if arr.size == 0:
        return np.nan
    arr.sort()
    k = int(np.floor(prop * arr.size))
    if 2 * k >= arr.size:
        return float(arr.mean())
    return float(arr[k: arr.size - k].mean())


def make_profile(x):
    s = finite_series(x)
    if s.empty:
        return np.nan
    if PROFILE_STAT == "median":
        return float(s.median())
    if PROFILE_STAT == "trimmed_mean":
        return trimmed_mean(s, prop=TRIM_PROP)
    raise ValueError(f"Unsupported PROFILE_STAT: {PROFILE_STAT}")


def winsorize_series(x, lower_q=WINSOR_LOWER, upper_q=WINSOR_UPPER):
    s = finite_series(x)
    if s.empty:
        return s
    lo = float(s.quantile(lower_q))
    hi = float(s.quantile(upper_q))
    return s.clip(lower=lo, upper=hi)


def robust_loc_scale(x):
    arr = np.asarray(finite_series(x), dtype=float)
    if arr.size == 0:
        return 0.0, 1.0
    loc = float(np.median(arr))
    if arr.size == 1:
        return loc, MIN_SCALE
    mad = float(np.median(np.abs(arr - loc)))
    scale = 1.4826 * mad if mad > 0 else float(np.std(arr, ddof=1))
    if (not np.isfinite(scale)) or scale < MIN_SCALE:
        scale = float(np.std(arr, ddof=1)) if arr.size > 1 else MIN_SCALE
    if (not np.isfinite(scale)) or scale < MIN_SCALE:
        scale = MIN_SCALE
    return loc, float(scale)


def safe_positive_part(x):
    arr = np.asarray(pd.to_numeric(x, errors="coerce"), dtype=float)
    floor = -POSITIVE_EPS + 1e-9
    mask = np.isfinite(arr)
    arr = np.where(mask, np.maximum(arr, floor), np.nan)
    return arr


def compute_residual(raw_value, profile_value, transform):
    x = np.asarray(pd.to_numeric(raw_value, errors="coerce"), dtype=float)
    p = np.asarray(pd.to_numeric(profile_value, errors="coerce"), dtype=float)
    if transform == "log_ratio":
        x = safe_positive_part(x)
        p = safe_positive_part(p)
        ratio = (x + POSITIVE_EPS) / (p + POSITIVE_EPS)
        ratio = np.where(np.isfinite(ratio), np.clip(ratio, np.exp(-LOGRATIO_RESIDUAL_CLIP), np.exp(LOGRATIO_RESIDUAL_CLIP)), np.nan)
        return np.log(ratio)
    if transform == "additive":
        return x - p
    raise ValueError(f"Unsupported transform: {transform}")


def build_profiles_for_match(df):
    rows = []
    match_id = df["match_id"].iloc[0]
    for variable in VARIABLE_SPECS:
        tmp = (
            df.groupby(PROFILE_KEYS)[variable]
            .apply(make_profile)
            .reset_index(name="profile_value")
        )
        tmp["match_id"] = match_id
        tmp["variable"] = variable
        tmp["sampling_group"] = VARIABLE_SPECS[variable]["sampling_group"]
        tmp["transform"] = VARIABLE_SPECS[variable]["transform"]
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def add_profile_columns(df, profiles_df):
    out = df.copy()
    for variable in VARIABLE_SPECS:
        p = (
            profiles_df.loc[profiles_df["variable"] == variable, PROFILE_KEYS + ["profile_value"]]
            .rename(columns={"profile_value": f"{variable}_profile"})
            .copy()
        )
        out = out.merge(p, on=PROFILE_KEYS, how="left")
    return out


def prepare_match_df(path):
    df = read_csv_optimized(path)
    required = [
        "timestamp",
        "hour",
        "quarter",
        "generation",
        "demand",
        "seller_lmp",
        "buyer_lmp_out",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} is missing required columns: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df = df.dropna(subset=["timestamp", "hour", "quarter"]).copy()
    df["hour"] = df["hour"].astype(int)
    df["match_id"] = extract_match_id(path)
    return df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return float(default)


def default_params_for_family(family_name, x):
    loc, scale = robust_loc_scale(x)
    if family_name == "t":
        return (8.0, loc, scale)
    if family_name == "skewnorm":
        return (0.0, loc, scale)
    return (loc, scale)


def sanitize_fitted_params(family_name, params, x=None):
    fallback_loc, fallback_scale = robust_loc_scale(x if x is not None else [0.0])

    if family_name in ["norm", "logistic", "laplace"]:
        try:
            loc, scale = params[-2], params[-1]
        except Exception:
            loc, scale = fallback_loc, fallback_scale
        loc = safe_float(loc, fallback_loc)
        scale = abs(safe_float(scale, fallback_scale))
        scale = max(scale, MIN_SCALE)
        return (loc, scale)

    if family_name == "t":
        try:
            df, loc, scale = params[0], params[-2], params[-1]
        except Exception:
            df, loc, scale = 8.0, fallback_loc, fallback_scale
        df = safe_float(df, 8.0)
        if not np.isfinite(df):
            df = 8.0
        df = float(np.clip(df, T_DF_MIN, T_DF_MAX))
        loc = safe_float(loc, fallback_loc)
        scale = abs(safe_float(scale, fallback_scale))
        scale = max(scale, MIN_SCALE)
        return (df, loc, scale)

    if family_name == "skewnorm":
        try:
            a, loc, scale = params[0], params[-2], params[-1]
        except Exception:
            a, loc, scale = 0.0, fallback_loc, fallback_scale
        a = safe_float(a, 0.0)
        if not np.isfinite(a):
            a = 0.0
        a = float(np.clip(a, -SKEWNORM_A_MAX, SKEWNORM_A_MAX))
        loc = safe_float(loc, fallback_loc)
        scale = abs(safe_float(scale, fallback_scale))
        scale = max(scale, MIN_SCALE)
        return (a, loc, scale)

    raise ValueError(f"Unsupported family: {family_name}")


def safe_logpdf_values(x, family_name, params):
    arr = np.asarray(x, dtype=float)
    dist = FAMILY_OBJECTS[family_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logpdf = dist.logpdf(arr, *params)
    logpdf = np.asarray(logpdf, dtype=float)
    return logpdf


def safe_cdf_values(x, family_name, params):
    arr = np.asarray(x, dtype=float)
    dist = FAMILY_OBJECTS[family_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cdf = dist.cdf(arr, *params)
    cdf = np.asarray(cdf, dtype=float)
    if np.any(~np.isfinite(cdf)):
        raise FloatingPointError("Non-finite CDF values encountered.")
    return np.clip(cdf, CDF_EPS, 1.0 - CDF_EPS)


def stable_ks_test(x, family_name, params):
    arr = np.sort(np.asarray(finite_series(x), dtype=float))
    n = arr.size
    if n < 2:
        return np.nan, np.nan, "too_few_obs"

    u = safe_cdf_values(arr, family_name, params)
    i = np.arange(1, n + 1, dtype=float)
    d_plus = np.max(i / n - u)
    d_minus = np.max(u - (i - 1.0) / n)
    stat = float(max(d_plus, d_minus))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            pvalue = float(stats.kstwo.sf(stat, n))
        if not np.isfinite(pvalue):
            pvalue = np.nan
            status = "nonfinite_pvalue"
        else:
            status = "kstwo_approx"
    except Exception:
        pvalue = np.nan
        status = "pvalue_unavailable"

    return stat, pvalue, status


def stable_cvm_test(x, family_name, params):
    arr = np.sort(np.asarray(finite_series(x), dtype=float))
    n = arr.size
    if n < 2:
        return np.nan, np.nan, "too_few_obs"

    u = safe_cdf_values(arr, family_name, params)
    i = np.arange(1, n + 1, dtype=float)
    stat = float(1.0 / (12.0 * n) + np.sum((u - (2.0 * i - 1.0) / (2.0 * n)) ** 2))

    pvalue = np.nan
    status = "manual_stat_only"
    if n <= CVM_PVALUE_MAX_N:
        dist = FAMILY_OBJECTS[family_name]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                res = stats.cramervonmises(arr, lambda v: dist.cdf(v, *params))
            pvalue = safe_float(res.pvalue)
            if np.isfinite(pvalue):
                status = "scipy_asymptotic"
            else:
                pvalue = np.nan
                status = "nonfinite_pvalue"
        except Exception:
            pvalue = np.nan
            status = "asymptotic_unstable"
    else:
        status = "manual_stat_only_large_n"

    return stat, pvalue, status


def fit_candidate_distribution(x, family_name):
    arr = np.asarray(winsorize_series(x), dtype=float)
    if arr.size < 20:
        raise ValueError("Not enough observations to fit candidate distribution.")

    dist = FAMILY_OBJECTS[family_name]
    fit_method = "mle"
    raw_params = None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            raw_params = dist.fit(arr)
    except Exception:
        fit_method = "robust_fallback"
        raw_params = default_params_for_family(family_name, arr)

    params = sanitize_fitted_params(family_name, raw_params, x=arr)
    logpdf = safe_logpdf_values(arr, family_name, params)
    if np.any(~np.isfinite(logpdf)):
        raise ValueError("Non-finite log-likelihood encountered after parameter sanitization.")

    loglik = float(np.sum(logpdf))
    k = FAMILY_NPARAMS[family_name]
    aic = float(2.0 * k - 2.0 * loglik)
    bic = float(k * np.log(len(arr)) - 2.0 * loglik)

    ks_stat, ks_pvalue, ks_status = stable_ks_test(arr, family_name, params)
    cvm_stat, cvm_pvalue, cvm_status = stable_cvm_test(arr, family_name, params)

    return {
        "family": family_name,
        "params": [float(v) for v in params],
        "n": int(len(arr)),
        "loglik": loglik,
        "aic": aic,
        "bic": bic,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "ks_status": ks_status,
        "cvm_stat": cvm_stat,
        "cvm_pvalue": cvm_pvalue,
        "cvm_status": cvm_status,
        "fit_method": fit_method,
    }


def choose_family(fit_rows):
    valid = [r for r in fit_rows if np.isfinite(r.get("bic", np.inf))]
    if not valid:
        raise ValueError("No valid candidate fit is available for family selection.")

    best_bic = min(r["bic"] for r in valid)
    eligible = [r for r in valid if (r["bic"] - best_bic) <= SIMPLICITY_BIC_DELTA]
    eligible.sort(
        key=lambda r: (
            FAMILY_COMPLEXITY.get(r["family"], 99),
            r["bic"],
            safe_float(r.get("ks_stat", np.inf), np.inf),
            safe_float(r.get("cvm_stat", np.inf), np.inf),
        )
    )
    winner = eligible[0].copy()
    winner["selection_reason"] = (
        "lowest_BIC"
        if winner["bic"] == best_bic
        else f"simpler_within_{SIMPLICITY_BIC_DELTA}_BIC"
    )
    return winner


def shape_from_params(family_name, params):
    if family_name == "t":
        return {"df": float(params[0])}
    if family_name == "skewnorm":
        return {"a": float(params[0])}
    return {}


def fit_loc_scale_given_family(x, family_name, shape_param=None):
    arr = np.asarray(winsorize_series(x), dtype=float)
    if arr.size == 0:
        return {"loc": 0.0, "scale": np.nan, "fit_method": "empty"}

    if arr.size < 3:
        loc, scale = robust_loc_scale(arr)
        return {"loc": loc, "scale": scale, "fit_method": "robust_fallback_small_n"}

    dist = FAMILY_OBJECTS[family_name]
    fit_method = "mle"
    params = None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            if family_name == "t" and shape_param and "df" in shape_param:
                params = dist.fit(arr, f0=float(np.clip(shape_param["df"], T_DF_MIN, T_DF_MAX)))
            elif family_name == "skewnorm" and shape_param and "a" in shape_param:
                params = dist.fit(arr, f0=float(np.clip(shape_param["a"], -SKEWNORM_A_MAX, SKEWNORM_A_MAX)))
            else:
                params = dist.fit(arr)
    except Exception:
        fit_method = "robust_fallback"
        params = default_params_for_family(family_name, arr)

    params = sanitize_fitted_params(family_name, params, x=arr)
    if family_name in ["norm", "logistic", "laplace"]:
        loc, scale = params
    else:
        _, loc, scale = params

    if (not np.isfinite(loc)) or (not np.isfinite(scale)):
        loc, scale = robust_loc_scale(arr)
        fit_method = "robust_fallback_nonfinite"

    scale = max(float(scale), MIN_SCALE)
    return {"loc": float(loc), "scale": float(scale), "fit_method": fit_method}


def clip_bounds(x, allow_negative):
    s = finite_series(x)
    if s.empty:
        return np.nan, np.nan
    lo = float(s.quantile(0.005))
    hi = float(s.quantile(0.995))
    if not allow_negative:
        lo = max(0.0, lo)
    return lo, hi


def nearest_non_missing(value, fallback):
    return value if pd.notna(value) else fallback


# ============================================================
# 3) Main
# ============================================================
def main():
    match_dir = resolve_match_dir()
    output_root = resolve_output_root(match_dir)
    output_dir = output_root / CODE1_OUTPUT_SUBFOLDER
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(match_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No match CSV files were found in {match_dir}")

    print(f"Match folder:  {match_dir}")
    print(f"Output folder: {output_dir}")
    print(f"CSV files:     {len(csv_paths)}")

    profile_rows = []
    residual_rows = []
    global_clip_rows = []

    for path in csv_paths:
        df = prepare_match_df(path)
        profiles = build_profiles_for_match(df)
        df_with_profiles = add_profile_columns(df, profiles)

        profile_rows.append(profiles)

        match_id = df["match_id"].iloc[0]
        for variable, spec in VARIABLE_SPECS.items():
            prof_col = f"{variable}_profile"

            tmp = df_with_profiles[
                ["match_id", "timestamp", "quarter", "hour", variable, prof_col]
            ].copy()
            tmp["variable"] = variable
            tmp["sampling_group"] = spec["sampling_group"]
            tmp["transform"] = spec["transform"]
            tmp["raw_value"] = pd.to_numeric(tmp[variable], errors="coerce")
            tmp["profile_value"] = pd.to_numeric(tmp[prof_col], errors="coerce")
            tmp["residual"] = compute_residual(
                tmp["raw_value"], tmp["profile_value"], spec["transform"]
            )
            residual_rows.append(
                tmp[
                    [
                        "match_id",
                        "timestamp",
                        "quarter",
                        "hour",
                        "variable",
                        "sampling_group",
                        "transform",
                        "raw_value",
                        "profile_value",
                        "residual",
                    ]
                ]
            )

            lo, hi = clip_bounds(tmp["raw_value"], allow_negative=spec["allow_negative"])
            global_clip_rows.append(
                {
                    "match_id": match_id,
                    "variable": variable,
                    "global_lower_clip": lo,
                    "global_upper_clip": hi,
                }
            )

    profiles_df = pd.concat(profile_rows, ignore_index=True)
    residual_df = pd.concat(residual_rows, ignore_index=True)
    residual_df["residual"] = pd.to_numeric(residual_df["residual"], errors="coerce")
    residual_df.loc[~np.isfinite(residual_df["residual"]), "residual"] = np.nan
    global_clip_df = pd.DataFrame(global_clip_rows).drop_duplicates()

    profiles_df.to_csv(output_dir / "representative_profiles.csv", index=False)
    residual_df.to_csv(output_dir / "residual_panel.csv", index=False)
    global_clip_df.to_csv(output_dir / "global_clip_bounds.csv", index=False)

    # ------------------------------------------------------------
    # 3.1 Variable-level diagnostics
    # ------------------------------------------------------------
    variable_fit_rows = []
    variable_selected_rows = []

    for variable, spec in VARIABLE_SPECS.items():
        group_name = spec["sampling_group"]
        candidates = SAMPLING_GROUPS[group_name]["candidates"]
        x = winsorize_series(
            residual_df.loc[residual_df["variable"] == variable, "residual"]
        )

        fit_rows = []
        for family in candidates:
            try:
                fit = fit_candidate_distribution(x, family)
                fit["error"] = ""
            except Exception as exc:
                fit = {
                    "family": family,
                    "params": [],
                    "n": int(len(x)),
                    "loglik": np.nan,
                    "aic": np.inf,
                    "bic": np.inf,
                    "ks_stat": np.nan,
                    "ks_pvalue": np.nan,
                    "ks_status": "failed",
                    "cvm_stat": np.nan,
                    "cvm_pvalue": np.nan,
                    "cvm_status": "failed",
                    "fit_method": "failed",
                    "error": str(exc),
                }
            fit["level"] = "variable"
            fit["name"] = variable
            fit["sampling_group"] = group_name
            fit_rows.append(fit)
            variable_fit_rows.append(fit)

        winner = choose_family(fit_rows)
        variable_selected_rows.append(
            {
                "variable": variable,
                "sampling_group": group_name,
                "selected_family": winner["family"],
                "selection_reason": winner["selection_reason"],
                "shape_json": json.dumps(shape_from_params(winner["family"], winner["params"]), sort_keys=True),
                "fit_method": winner.get("fit_method", ""),
                "n_used": winner["n"],
                "loglik": winner["loglik"],
                "aic": winner["aic"],
                "bic": winner["bic"],
                "ks_stat": winner["ks_stat"],
                "ks_pvalue": winner["ks_pvalue"],
                "ks_status": winner.get("ks_status", ""),
                "cvm_stat": winner["cvm_stat"],
                "cvm_pvalue": winner["cvm_pvalue"],
                "cvm_status": winner.get("cvm_status", ""),
            }
        )

    variable_fit_df = pd.DataFrame(variable_fit_rows)
    variable_selected_df = pd.DataFrame(variable_selected_rows)

    variable_fit_df["selected"] = False
    for _, row in variable_selected_df.iterrows():
        mask = (
            (variable_fit_df["level"] == "variable")
            & (variable_fit_df["name"] == row["variable"])
            & (variable_fit_df["family"] == row["selected_family"])
        )
        variable_fit_df.loc[mask, "selected"] = True

    variable_fit_df.to_csv(output_dir / "candidate_fit_metrics_variable_level.csv", index=False)
    variable_selected_df.to_csv(output_dir / "selected_families_variable_level.csv", index=False)

    # ------------------------------------------------------------
    # 3.2 Sampling-group family selection (actual family used later)
    # ------------------------------------------------------------
    group_fit_rows = []
    selected_group_rows = []

    for group_name, group_spec in SAMPLING_GROUPS.items():
        candidates = group_spec["candidates"]
        x = winsorize_series(
            residual_df.loc[residual_df["sampling_group"] == group_name, "residual"]
        )

        fit_rows = []
        for family in candidates:
            try:
                fit = fit_candidate_distribution(x, family)
                fit["error"] = ""
            except Exception as exc:
                fit = {
                    "family": family,
                    "params": [],
                    "n": int(len(x)),
                    "loglik": np.nan,
                    "aic": np.inf,
                    "bic": np.inf,
                    "ks_stat": np.nan,
                    "ks_pvalue": np.nan,
                    "ks_status": "failed",
                    "cvm_stat": np.nan,
                    "cvm_pvalue": np.nan,
                    "cvm_status": "failed",
                    "fit_method": "failed",
                    "error": str(exc),
                }
            fit["level"] = "sampling_group"
            fit["name"] = group_name
            fit_rows.append(fit)
            group_fit_rows.append(fit)

        winner = choose_family(fit_rows)
        selected_group_rows.append(
            {
                "sampling_group": group_name,
                "selected_family": winner["family"],
                "selection_reason": winner["selection_reason"],
                "shape_json": json.dumps(shape_from_params(winner["family"], winner["params"]), sort_keys=True),
                "fit_method": winner.get("fit_method", ""),
                "n_used": winner["n"],
                "loglik": winner["loglik"],
                "aic": winner["aic"],
                "bic": winner["bic"],
                "ks_stat": winner["ks_stat"],
                "ks_pvalue": winner["ks_pvalue"],
                "ks_status": winner.get("ks_status", ""),
                "cvm_stat": winner["cvm_stat"],
                "cvm_pvalue": winner["cvm_pvalue"],
                "cvm_status": winner.get("cvm_status", ""),
            }
        )

    group_fit_df = pd.DataFrame(group_fit_rows)
    selected_group_df = pd.DataFrame(selected_group_rows)

    group_fit_df["selected"] = False
    for _, row in selected_group_df.iterrows():
        mask = (
            (group_fit_df["level"] == "sampling_group")
            & (group_fit_df["name"] == row["sampling_group"])
            & (group_fit_df["family"] == row["selected_family"])
        )
        group_fit_df.loc[mask, "selected"] = True

    group_fit_df.to_csv(output_dir / "candidate_fit_metrics_sampling_groups.csv", index=False)
    selected_group_df.to_csv(output_dir / "selected_families_for_sampling.csv", index=False)

    # ------------------------------------------------------------
    # 3.3 Pooled quarter x hour benchmark scales (variable-specific)
    # ------------------------------------------------------------
    selected_group_map = {
        row["sampling_group"]: {
            "family": row["selected_family"],
            "shape": json.loads(row["shape_json"]) if isinstance(row["shape_json"], str) and str(row["shape_json"]).strip() else {},
        }
        for _, row in selected_group_df.iterrows()
    }

    pooled_benchmark_rows = []
    global_variable_scale_rows = []

    for variable, spec in VARIABLE_SPECS.items():
        group_name = spec["sampling_group"]
        family = selected_group_map[group_name]["family"]
        shape = selected_group_map[group_name]["shape"]

        var_resid = residual_df.loc[residual_df["variable"] == variable].copy()
        full_x = winsorize_series(var_resid["residual"])
        full_est = fit_loc_scale_given_family(full_x, family, shape)
        global_variable_scale_rows.append(
            {
                "variable": variable,
                "sampling_group": group_name,
                "global_benchmark_scale": full_est["scale"],
                "global_fit_method": full_est["fit_method"],
            }
        )

        for (quarter, hour), g in var_resid.groupby(PROFILE_KEYS):
            x = winsorize_series(g["residual"])
            est = fit_loc_scale_given_family(x, family, shape)
            pooled_benchmark_rows.append(
                {
                    "variable": variable,
                    "sampling_group": group_name,
                    "quarter": quarter,
                    "hour": int(hour),
                    "benchmark_loc": 0.0,
                    "benchmark_scale": est["scale"],
                    "n_benchmark": int(len(x)),
                    "fit_method": est["fit_method"],
                }
            )

    pooled_benchmark_df = pd.DataFrame(pooled_benchmark_rows)
    global_variable_scale_df = pd.DataFrame(global_variable_scale_rows)

    pooled_benchmark_df.to_csv(output_dir / "pooled_qh_benchmarks.csv", index=False)
    global_variable_scale_df.to_csv(output_dir / "global_variable_benchmark_scales.csv", index=False)

    # ------------------------------------------------------------
    # 3.4 Parameter bank for generation
    # ------------------------------------------------------------
    param_rows = []

    for variable, spec in VARIABLE_SPECS.items():
        group_name = spec["sampling_group"]
        family = selected_group_map[group_name]["family"]
        shape = selected_group_map[group_name]["shape"]
        allow_negative = spec["allow_negative"]

        var_resid = residual_df.loc[residual_df["variable"] == variable].copy()
        bench_var = pooled_benchmark_df.loc[pooled_benchmark_df["variable"] == variable].copy()
        global_scale = float(
            global_variable_scale_df.loc[
                global_variable_scale_df["variable"] == variable, "global_benchmark_scale"
            ].iloc[0]
        )

        for (match_id, quarter, hour), g in var_resid.groupby(["match_id", "quarter", "hour"]):
            x = winsorize_series(g["residual"])
            n = int(len(x))

            if n > 0:
                est = fit_loc_scale_given_family(x, family, shape)
                fit_source = "direct" if n >= MIN_CELL_OBS_DIRECT else "sparse_direct"
            else:
                est = {"loc": 0.0, "scale": np.nan, "fit_method": "empty"}
                fit_source = "pooled_only"

            bench_row = bench_var.loc[
                (bench_var["quarter"] == quarter) & (bench_var["hour"] == int(hour))
            ]
            if bench_row.empty:
                benchmark_loc = 0.0
                benchmark_scale = global_scale
            else:
                benchmark_loc = float(bench_row["benchmark_loc"].iloc[0])
                benchmark_scale = float(bench_row["benchmark_scale"].iloc[0])

            raw_loc = nearest_non_missing(est["loc"], 0.0)
            raw_scale = nearest_non_missing(est["scale"], benchmark_scale)
            raw_scale = max(float(raw_scale), MIN_SCALE)

            shrink_weight = float(n / (n + SHRINK_K)) if n > 0 else 0.0
            loc_shrunk = shrink_weight * raw_loc + (1.0 - shrink_weight) * benchmark_loc
            scale_shrunk = shrink_weight * raw_scale + (1.0 - shrink_weight) * benchmark_scale
            scale_shrunk = max(float(scale_shrunk), MIN_SCALE)

            profile_value = float(finite_series(g["profile_value"]).median()) if not finite_series(g["profile_value"]).empty else np.nan

            if n >= MIN_CLIP_OBS_CELL:
                lower_clip, upper_clip = clip_bounds(g["raw_value"], allow_negative=allow_negative)
                clip_source = "cell"
            else:
                row_global = global_clip_df.loc[
                    (global_clip_df["match_id"] == match_id) & (global_clip_df["variable"] == variable)
                ]
                if row_global.empty:
                    lower_clip, upper_clip = np.nan, np.nan
                else:
                    lower_clip = row_global["global_lower_clip"].iloc[0]
                    upper_clip = row_global["global_upper_clip"].iloc[0]
                clip_source = "match_variable_global"

            param_rows.append(
                {
                    "match_id": match_id,
                    "variable": variable,
                    "sampling_group": group_name,
                    "transform": spec["transform"],
                    "family": family,
                    "shape_json": json.dumps(shape, sort_keys=True),
                    "quarter": quarter,
                    "hour": int(hour),
                    "n_obs": n,
                    "fit_source": fit_source,
                    "fit_method": est.get("fit_method", ""),
                    "profile_value": profile_value,
                    "loc_raw": raw_loc,
                    "scale_raw": raw_scale,
                    "benchmark_loc": benchmark_loc,
                    "benchmark_scale": benchmark_scale,
                    "shrink_weight": shrink_weight,
                    "loc_shrunk": loc_shrunk,
                    "scale_shrunk": scale_shrunk,
                    "lower_clip": lower_clip,
                    "upper_clip": upper_clip,
                    "clip_source": clip_source,
                }
            )

    params_df = pd.DataFrame(param_rows).sort_values(
        ["match_id", "variable", "quarter", "hour"]
    )
    params_df.to_csv(output_dir / "marginal_parameter_bank.csv", index=False)

    # ------------------------------------------------------------
    # 3.5 Summary file
    # ------------------------------------------------------------
    summary = {
        "match_folder": str(match_dir),
        "output_folder": str(output_dir),
        "n_match_files": int(len(csv_paths)),
        "profile_stat": PROFILE_STAT,
        "profile_keys": PROFILE_KEYS,
        "positive_eps": POSITIVE_EPS,
        "winsor_limits": [WINSOR_LOWER, WINSOR_UPPER],
        "min_cell_obs_direct": MIN_CELL_OBS_DIRECT,
        "shrink_k": SHRINK_K,
        "cdf_eps": CDF_EPS,
        "cvm_pvalue_max_n": CVM_PVALUE_MAX_N,
        "shape_bounds": {
            "t_df_min": T_DF_MIN,
            "t_df_max": T_DF_MAX,
            "skewnorm_abs_a_max": SKEWNORM_A_MAX,
        },
        "sampling_group_selection": selected_group_rows,
        "notes": [
            "buyer_lmp_in is intentionally excluded from distribution fitting.",
            "Family diagnostics are written both at variable level and at sampling-group level.",
            "The sampling family actually used later is selected at the sampling-group level: generation, demand, nodal_price.",
            "Representative profiles are quarter x hour medians by default, which is a simplification relative to full-history reproduction.",
            "Cell parameters are shrunk toward pooled quarter x hour benchmarks to reduce historical noise.",
            "Cramér-von Mises statistics are computed manually; SciPy p-values are used only when numerically stable.",
        ],
    }
    with open(output_dir / "code1_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSelected families for sampling:")
    print(selected_group_df.to_string(index=False))
    print("\nKey outputs written:")
    for name in [
        "representative_profiles.csv",
        "residual_panel.csv",
        "candidate_fit_metrics_variable_level.csv",
        "selected_families_variable_level.csv",
        "candidate_fit_metrics_sampling_groups.csv",
        "selected_families_for_sampling.csv",
        "pooled_qh_benchmarks.csv",
        "marginal_parameter_bank.csv",
        "code1_run_summary.json",
    ]:
        print(f" - {output_dir / name}")


if __name__ == "__main__":
    main()
