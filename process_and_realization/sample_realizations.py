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
Code 3 (stable patch)
---------------------
Generate simplified baseline realizations using:
  - Code 1 marginal parameter bank
  - Code 2 shrunk residual correlation bank

Stability patch:
  - guards against non-finite ppf outputs and extreme log-ratio back-transforms
  - repairs nearly singular correlation matrices before sampling
  - handles malformed or missing shape parameters robustly

Output:
    <parent of Match files of original data>/Simplified baseline realized samples/
        rep_01/<match>.csv
        rep_02/<match>.csv
        ...
        rep_05/<match>.csv
"""


# ============================================================
# 1) Settings
# ============================================================
SEARCH_ROOTS = _build_search_roots()

MATCH_FOLDER_NAME = "Match files of original data"
CODE1_OUTPUT_SUBFOLDER = "Marginal_Fit"
CODE2_OUTPUT_SUBFOLDER = "Correlation"
OUTPUT_FOLDER_NAME = "Simplified baseline realized samples"

VARIABLES = ["generation", "demand", "seller_lmp", "buyer_lmp_out"]
VARIABLE_TRANSFORMS = {
    "generation": "log_ratio",
    "demand": "log_ratio",
    "seller_lmp": "additive",
    "buyer_lmp_out": "additive",
}
ALLOW_NEGATIVE = {
    "generation": False,
    "demand": False,
    "seller_lmp": True,
    "buyer_lmp_out": True,
}

POSITIVE_EPS = 1.0
N_REPLICATIONS = 5
GLOBAL_SEED = 20260424
MIN_SCALE = 1e-6
PPF_EPS = 1e-9
LOGRATIO_CLIP = 20.0
CORR_EIG_FLOOR = 1e-8
T_DF_MIN = 2.10
T_DF_MAX = 200.0
SKEWNORM_A_MAX = 30.0
GAUSS_Z_CLIP = 8.0

# buyer_lmp_in rule from the extraction script
BETA_A = 2.0
BETA_B = 3.0
MARKUP_LOW = 0.5
MARKUP_HIGH = 2.0


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


def resolve_input_root(match_dir):
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


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return float(default)


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


def draw_gaussian_copula_uniforms(n, corr_matrix, rng):
    corr_matrix = nearest_correlation_matrix(corr_matrix)
    d = corr_matrix.shape[0]
    try:
        chol = np.linalg.cholesky(corr_matrix)
        z = rng.standard_normal((n, d)) @ chol.T
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(corr_matrix)
        vals = np.clip(vals, CORR_EIG_FLOOR, None)
        sqrt_cov = vecs @ np.diag(np.sqrt(vals))
        z = rng.standard_normal((n, d)) @ sqrt_cov.T
    z = np.clip(z, -GAUSS_Z_CLIP, GAUSS_Z_CLIP)
    return stats.norm.cdf(z)


def parse_shape_json(x):
    if pd.isna(x):
        return {}
    if isinstance(x, dict):
        return x
    txt = str(x).strip()
    if not txt or txt.lower() in {"nan", "none", "null"}:
        return {}
    try:
        obj = json.loads(txt)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def sanitize_shape(family, shape):
    shape = shape if isinstance(shape, dict) else {}
    if family == "t":
        df = safe_float(shape.get("df", 8.0), 8.0)
        if not np.isfinite(df):
            df = 8.0
        return {"df": float(np.clip(df, T_DF_MIN, T_DF_MAX))}
    if family == "skewnorm":
        a = safe_float(shape.get("a", 0.0), 0.0)
        if not np.isfinite(a):
            a = 0.0
        return {"a": float(np.clip(a, -SKEWNORM_A_MAX, SKEWNORM_A_MAX))}
    return {}


def safe_scale_array(x):
    arr = np.asarray(x, dtype=float)
    arr = np.where(np.isfinite(arr), np.abs(arr), np.nan)
    arr = np.where((~np.isfinite(arr)) | (arr < MIN_SCALE), MIN_SCALE, arr)
    return arr


def family_ppf(u, family, shape, loc, scale):
    u = np.clip(np.asarray(u, dtype=float), PPF_EPS, 1.0 - PPF_EPS)
    loc = np.asarray(loc, dtype=float)
    loc = np.where(np.isfinite(loc), loc, 0.0)
    scale = safe_scale_array(scale)
    shape = sanitize_shape(family, shape)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if family == "norm":
            x = stats.norm.ppf(u, loc=loc, scale=scale)
        elif family == "logistic":
            x = stats.logistic.ppf(u, loc=loc, scale=scale)
        elif family == "laplace":
            x = stats.laplace.ppf(u, loc=loc, scale=scale)
        elif family == "t":
            df = float(shape.get("df", 8.0))
            x = stats.t.ppf(u, df, loc=loc, scale=scale)
        elif family == "skewnorm":
            a = float(shape.get("a", 0.0))
            x = stats.skewnorm.ppf(u, a, loc=loc, scale=scale)
        else:
            raise ValueError(f"Unsupported family: {family}")

    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, loc)
    return x


def inverse_transform(residual, profile_value, transform):
    residual = np.asarray(residual, dtype=float)
    profile_value = np.asarray(profile_value, dtype=float)
    profile_value = np.where(np.isfinite(profile_value), profile_value, 0.0)

    if transform == "log_ratio":
        residual = np.where(np.isfinite(residual), np.clip(residual, -LOGRATIO_CLIP, LOGRATIO_CLIP), 0.0)
        return (profile_value + POSITIVE_EPS) * np.exp(residual) - POSITIVE_EPS
    if transform == "additive":
        residual = np.where(np.isfinite(residual), residual, 0.0)
        return profile_value + residual
    raise ValueError(f"Unsupported transform: {transform}")


def apply_bounds(x, lower, upper, allow_negative, fallback=None):
    x = np.asarray(x, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    fallback = np.zeros_like(x) if fallback is None else np.asarray(fallback, dtype=float)
    fallback = np.where(np.isfinite(fallback), fallback, 0.0)

    x = np.where(np.isfinite(x), x, fallback)
    if not allow_negative:
        x = np.maximum(x, 0.0)

    valid_lower = np.isfinite(lower)
    valid_upper = np.isfinite(upper)
    bad_bounds = valid_lower & valid_upper & (upper < lower)
    if np.any(bad_bounds):
        upper = np.where(bad_bounds, lower, upper)

    x = np.where(valid_lower, np.maximum(x, lower), x)
    x = np.where(valid_upper, np.minimum(x, upper), x)
    x = np.where(np.isfinite(x), x, fallback)
    return x


def generate_buyer_lmp_in(buyer_lmp_out, rng):
    base = np.asarray(buyer_lmp_out, dtype=float)
    base = np.where(np.isfinite(base), base, 0.0)
    x = MARKUP_LOW + (MARKUP_HIGH - MARKUP_LOW) * rng.beta(
        BETA_A, BETA_B, size=len(base)
    )
    out = base + x * np.abs(base)
    return np.where(np.isfinite(out), out, base)


def build_correlation_matrix(match_id, corr_long_df, pooled_corr_df):
    variables = VARIABLES
    matrix = np.eye(len(variables), dtype=float)

    sub = corr_long_df.loc[corr_long_df["match_id"] == match_id].copy()
    if sub.empty:
        sub = pooled_corr_df.copy().rename(columns={"corr_pooled": "corr_target"})

    for _, row in sub.iterrows():
        if row["var_i"] not in variables or row["var_j"] not in variables:
            continue
        i = variables.index(row["var_i"])
        j = variables.index(row["var_j"])
        val = safe_float(row.get("corr_target", np.nan), np.nan)
        if not np.isfinite(val):
            continue
        val = float(np.clip(val, -0.999, 0.999))
        matrix[i, j] = val
        matrix[j, i] = val

    np.fill_diagonal(matrix, 1.0)
    return nearest_correlation_matrix(matrix)


def build_match_parameter_frame(skeleton, match_id, params_df):
    out = skeleton.copy()
    out["match_id"] = match_id

    needed_cols = [
        "match_id",
        "variable",
        "quarter",
        "hour",
        "transform",
        "family",
        "shape_json",
        "profile_value",
        "loc_shrunk",
        "scale_shrunk",
        "lower_clip",
        "upper_clip",
    ]

    param_sub = params_df.loc[params_df["match_id"] == match_id, needed_cols].copy()
    if param_sub.empty:
        raise ValueError(f"No marginal parameters were found for match_id={match_id}.")

    for variable in VARIABLES:
        pv = param_sub.loc[param_sub["variable"] == variable].copy()
        if pv.empty:
            raise ValueError(f"No parameters were found for match_id={match_id}, variable={variable}.")

        fallback = (
            pv.groupby(["match_id", "variable"], as_index=False)
            .agg(
                transform=("transform", "first"),
                family=("family", "first"),
                shape_json=("shape_json", "first"),
                profile_value=("profile_value", "median"),
                loc_shrunk=("loc_shrunk", "median"),
                scale_shrunk=("scale_shrunk", "median"),
                lower_clip=("lower_clip", "min"),
                upper_clip=("upper_clip", "max"),
            )
            .iloc[0]
        )

        keep = [
            "quarter",
            "hour",
            "transform",
            "family",
            "shape_json",
            "profile_value",
            "loc_shrunk",
            "scale_shrunk",
            "lower_clip",
            "upper_clip",
        ]
        pv = pv[keep].drop_duplicates(subset=["quarter", "hour"]).copy()
        pv = pv.rename(columns={c: f"{variable}__{c}" for c in keep if c not in ["quarter", "hour"]})
        out = out.merge(pv, on=["quarter", "hour"], how="left")

        for c in [
            "transform",
            "family",
            "shape_json",
            "profile_value",
            "loc_shrunk",
            "scale_shrunk",
            "lower_clip",
            "upper_clip",
        ]:
            col = f"{variable}__{c}"
            fallback_value = fallback[c]
            out[col] = out[col].fillna(fallback_value)

    return out


# ============================================================
# 3) Main
# ============================================================
def main():
    match_dir = resolve_match_dir()
    input_root = resolve_input_root(match_dir)

    code1_dir = input_root / CODE1_OUTPUT_SUBFOLDER
    code2_dir = input_root / CODE2_OUTPUT_SUBFOLDER
    output_dir = match_dir.parent / OUTPUT_FOLDER_NAME

    params_path = code1_dir / "marginal_parameter_bank.csv"
    corr_long_path = code2_dir / "correlation_parameter_bank_long.csv"
    pooled_corr_path = code2_dir / "pooled_residual_correlation_matrix.csv"

    if not params_path.exists():
        raise FileNotFoundError(f"Could not find Code 1 parameter bank: {params_path}")
    if not corr_long_path.exists():
        raise FileNotFoundError(f"Could not find Code 2 correlation bank: {corr_long_path}")
    if not pooled_corr_path.exists():
        raise FileNotFoundError(f"Could not find pooled correlation file: {pooled_corr_path}")

    params_df = read_csv_optimized(params_path)
    corr_long_df = read_csv_optimized(corr_long_path)
    pooled_corr_df = read_csv_optimized(pooled_corr_path)

    for col in ["profile_value", "loc_shrunk", "scale_shrunk", "lower_clip", "upper_clip"]:
        if col in params_df.columns:
            params_df[col] = pd.to_numeric(params_df[col], errors="coerce")

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(match_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No match CSV files were found in {match_dir}")

    summary_rows = []

    for rep in range(1, N_REPLICATIONS + 1):
        rep_dir = output_dir / f"rep_{rep:02d}"
        rep_dir.mkdir(parents=True, exist_ok=True)

        for path in csv_paths:
            df = read_csv_optimized(path)
            required = ["timestamp", "hour", "quarter"] + VARIABLES
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise KeyError(f"{path.name} missing required columns: {missing}")

            skeleton = df[["timestamp", "hour", "quarter"]].copy()
            skeleton["hour"] = pd.to_numeric(skeleton["hour"], errors="coerce")
            skeleton = skeleton.dropna(subset=["hour"]).copy()
            skeleton["hour"] = skeleton["hour"].astype(int)
            match_id = extract_match_id(path)

            merged = build_match_parameter_frame(skeleton, match_id, params_df)
            corr_matrix = build_correlation_matrix(match_id, corr_long_df, pooled_corr_df)

            rng = np.random.default_rng(GLOBAL_SEED + rep * 100000 + int(match_id))
            u = draw_gaussian_copula_uniforms(len(merged), corr_matrix, rng)

            simulated = {}
            for j, variable in enumerate(VARIABLES):
                family = str(merged[f"{variable}__family"].iloc[0])
                shape = parse_shape_json(merged[f"{variable}__shape_json"].iloc[0])
                transform = str(merged[f"{variable}__transform"].iloc[0])

                profile_value = merged[f"{variable}__profile_value"].to_numpy(dtype=float)
                loc = merged[f"{variable}__loc_shrunk"].to_numpy(dtype=float)
                scale = merged[f"{variable}__scale_shrunk"].to_numpy(dtype=float)
                lower = merged[f"{variable}__lower_clip"].to_numpy(dtype=float)
                upper = merged[f"{variable}__upper_clip"].to_numpy(dtype=float)

                residual = family_ppf(u[:, j], family, shape, loc, scale)
                values = inverse_transform(residual, profile_value, transform)
                values = apply_bounds(values, lower, upper, ALLOW_NEGATIVE[variable], fallback=profile_value)

                simulated[variable] = values

            buyer_lmp_in = generate_buyer_lmp_in(simulated["buyer_lmp_out"], rng)

            out_df = pd.DataFrame(
                {
                    "timestamp": skeleton["timestamp"],
                    "hour": skeleton["hour"].astype(int),
                    "quarter": skeleton["quarter"],
                    "generation": simulated["generation"],
                    "demand": simulated["demand"],
                    "seller_lmp": simulated["seller_lmp"],
                    "buyer_lmp_out": simulated["buyer_lmp_out"],
                    "buyer_lmp_in": buyer_lmp_in,
                }
            )

            out_path = rep_dir / path.name
            out_df.to_csv(out_path, index=False)

            summary_rows.append(
                {
                    "replication": rep,
                    "match_id": match_id,
                    "n_rows": len(out_df),
                    "output_file": str(out_path),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["replication", "match_id"])
    summary_df.to_csv(output_dir / "realization_summary.csv", index=False)

    run_summary = {
        "match_folder": str(match_dir),
        "code1_folder": str(code1_dir),
        "code2_folder": str(code2_dir),
        "output_folder": str(output_dir),
        "n_replications": N_REPLICATIONS,
        "variables": VARIABLES,
        "stability_controls": {
            "ppf_eps": PPF_EPS,
            "logratio_clip": LOGRATIO_CLIP,
            "corr_eig_floor": CORR_EIG_FLOOR,
            "t_df_min": T_DF_MIN,
            "t_df_max": T_DF_MAX,
            "skewnorm_abs_a_max": SKEWNORM_A_MAX,
        },
        "buyer_lmp_in_rule": {
            "type": "scaled_beta_absolute_spread",
            "formula": "buyer_lmp_in = buyer_lmp_out + X * abs(buyer_lmp_out)",
            "beta_a": BETA_A,
            "beta_b": BETA_B,
            "markup_low": MARKUP_LOW,
            "markup_high": MARKUP_HIGH,
        },
        "notes": [
            "Each replication preserves the original timestamp / hour / quarter skeleton.",
            "Marginal uncertainty comes from Code 1 shrunk residual parameters.",
            "Cross-variable dependence comes from Code 2 shrunk Gaussian-score correlation matrices.",
            "Non-finite ppf draws and nearly singular correlation matrices are handled explicitly in this stable patch.",
            "No serial autocorrelation is imposed in this simplified baseline; this is intentional.",
        ],
    }
    with open(output_dir / "code3_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print(f"Output folder: {output_dir}")
    print(f"Replications:  {N_REPLICATIONS}")
    print("\nSaved files:")
    print(f" - {output_dir / 'realization_summary.csv'}")
    print(f" - {output_dir / 'code3_run_summary.json'}")
    for rep in range(1, N_REPLICATIONS + 1):
        print(f" - {output_dir / f'rep_{rep:02d}'}")


if __name__ == "__main__":
    main()
