# Exported from Simulation_no_mutation_verified_decision.ipynb

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd



MODULE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = MODULE_DIR.parent


def _unique_paths(paths):
    out = []
    seen = set()
    for path in paths:
        if path is None:
            continue
        p = Path(path).expanduser()
        try:
            key = str(p.resolve()) if p.exists() else str(p)
        except Exception:
            key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _candidate_roots(*anchors, max_parent_depth: int = 4):
    roots = []
    for anchor in anchors:
        if anchor is None:
            continue
        p = Path(anchor).expanduser()
        if p.suffix:
            p = p.parent
        roots.append(p)
        roots.extend(list(p.parents)[:max_parent_depth])
    return _unique_paths(roots)


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
output_dir = "Output files (Risk Neutral, No Mutation, Verified)"

# ============================================================
# Risk preferences / contract settings
# ============================================================
# Use 0 for risk-neutral. Positive values make the party more risk-averse
# under the notebook's existing FE convention.
# 0.253~60%, 0.524~70%, 0.842~80%, 1.282~90%, 1.645~95%
lambda_s = 0.0
lambda_b = 0.0

# Fixed penalty rate = gamma × Q95(buyer_lmp_in) from the pooled baseline sample bank.
gamma = 1.0
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
verification_fix_allow_cross_ppa_type = True

# ============================================================
# Batch controls
# ============================================================
# Use None to run every match found in the historical-match folder.
selected_match_ids = None

# If True, skip a match when its per-match outputs already exist.
skip_existing_output = False

# If True, save the full contract grid for each match.
save_per_match_grid = False

# If True, save the hourly FE of the selected FE source for each match.
save_hourly_fe = True

# If the final decision is No Contract, still save the hourly FE for the best PPA
# candidate (feasible if available, otherwise best infeasible PPA).
save_best_fe_even_if_infeasible = True

# Save timestamps directly inside the hourly FE CSV files.
# If False, timestamps can still be reconstructed later by the FE plotting notebook.
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
    "gamma": float(gamma),
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



def resolve_match_dir(match_folder_name: str) -> Path:
    roots = _candidate_roots(MODULE_DIR, WORKSPACE_DIR, Path.cwd(), Path('/mnt/data'))
    candidates = [root / match_folder_name for root in roots]
    candidates.extend([
        Path("Input data and files") / match_folder_name,
        Path("Data and files") / match_folder_name,
    ])
    return resolve_existing_dir(*candidates)



def resolve_sample_root(sample_root_dir: str, match_dir: Path) -> Path:
    roots = _candidate_roots(MODULE_DIR, WORKSPACE_DIR, match_dir.parent, Path.cwd(), Path('/mnt/data'))
    candidates = [root / sample_root_dir for root in roots]
    candidates.extend([
        Path("Input data and files") / sample_root_dir,
        Path("Data and files") / sample_root_dir,
    ])
    return resolve_existing_dir(*candidates)



def resolve_match_table(match_table_file: str, match_dir: Optional[Path] = None) -> Optional[Path]:
    roots = _candidate_roots(MODULE_DIR, WORKSPACE_DIR, match_dir.parent if match_dir is not None else None, Path.cwd(), Path('/mnt/data'))
    candidates = [root / match_table_file for root in roots]
    candidates.extend([
        Path("Input data and files") / match_table_file,
        Path("Data and files") / match_table_file,
    ])
    return resolve_existing_file(*candidates)



def resolve_code2_corr_file(raw_path: str, match_dir: Path) -> Optional[Path]:
    candidate = Path(raw_path)
    roots = _candidate_roots(MODULE_DIR, WORKSPACE_DIR, match_dir.parent, Path.cwd(), Path('/mnt/data'))
    candidates = [root / candidate for root in roots]
    return resolve_existing_file(*candidates)


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
        df = pd.read_excel(match_table_path, sheet_name=sheet_name)
    else:
        df = None
        for sh in xls.sheet_names:
            candidate = pd.read_excel(match_table_path, sheet_name=sh)
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
    df = pd.read_csv(path)
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

        df = pd.read_csv(candidate)
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

    df = pd.read_csv(code2_corr_path)
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


def compute_penalty_rate(sample_df: pd.DataFrame, gamma: float) -> Tuple[float, float]:
    buyer_purchase_q95 = float(pd.to_numeric(sample_df["buyer_lmp_in"], errors="coerce").quantile(0.95))
    return float(gamma) * buyer_purchase_q95, buyer_purchase_q95


def evaluate_contract(g, d, Ns, Nb_out, Nb_in, mu, nu, pi, q_fixed,
                      lambda_s, lambda_b, availability_factor, penalty_rate):
    T = len(g)
    G_avg = np.mean(g)

    if nu == "Fix":
        q_del = np.full(T, q_fixed, dtype=float)
    elif nu == "AsG":
        q_del = g.copy()
    elif nu == "AsC":
        q_del = d.copy()
    else:
        raise ValueError(f"Unknown volume structure: {nu}")

    if nu == "AsG":
        Pen_t = penalty_rate * np.maximum(0.0, availability_factor * G_avg - q_del)
    else:
        Pen_t = np.zeros(T)

    if mu == "Physical":
        PPA_t = pi * q_del
    elif mu == "Virtual":
        PPA_t = (pi - Ns) * q_del
    else:
        raise ValueError(f"Unknown PPA type: {mu}")

    if mu == "Physical" and nu in ["Fix", "AsC"]:
        f_s = Ns * (q_del - g)
    elif mu == "Physical" and nu == "AsG":
        f_s = np.zeros(T)
    elif mu == "Virtual":
        f_s = -Ns * g
    else:
        raise ValueError("Invalid seller balancing case")

    if mu == "Physical":
        f_b = Nb_in * np.maximum(0.0, d - q_del) - Nb_out * np.maximum(0.0, q_del - d)
    elif mu == "Virtual":
        f_b = Nb_in * d
    else:
        raise ValueError("Invalid buyer balancing case")

    Rev_s = PPA_t - f_s - Pen_t
    FE_s = Ns * q_del - (PPA_t - Pen_t)
    FE_b = (PPA_t + f_b - Pen_t) - Nb_in * d

    U_s = float(np.mean(FE_s) + lambda_s * np.std(FE_s, ddof=0))
    U_b = float(np.mean(FE_b) + lambda_b * np.std(FE_b, ddof=0))

    return {
        "q_del": q_del,
        "Pen_t": Pen_t,
        "PPA_t": PPA_t,
        "f_s": f_s,
        "f_b": f_b,
        "Rev_s": Rev_s,
        "FE_s": FE_s,
        "FE_b": FE_b,
        "U_s": U_s,
        "U_b": U_b,
    }


def evaluate_no_contract(g, d, Ns, Nb_out, Nb_in, lambda_s, lambda_b):
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
    out["U_s"] = float(np.mean(out["FE_s"]) + lambda_s * np.std(out["FE_s"], ddof=0))
    out["U_b"] = float(np.mean(out["FE_b"]) + lambda_b * np.std(out["FE_b"], ddof=0))
    return out


def _summary_from_arrays(fe_s, fe_b, rev_s, lambda_s, lambda_b) -> Dict[str, float]:
    seller_mean = float(np.mean(fe_s))
    buyer_mean = float(np.mean(fe_b))
    seller_std = float(np.std(fe_s, ddof=0))
    buyer_std = float(np.std(fe_b, ddof=0))
    return {
        "seller_utility": float(seller_mean + lambda_s * seller_std),
        "buyer_utility": float(buyer_mean + lambda_b * buyer_std),
        "seller_mean_exposure": seller_mean,
        "buyer_mean_exposure": buyer_mean,
        "seller_exposure_variance": float(np.var(fe_s, ddof=0)),
        "buyer_exposure_variance": float(np.var(fe_b, ddof=0)),
        "expected_seller_revenue": float(np.mean(rev_s)),
    }


def _linear_stats(base: np.ndarray, coeff: np.ndarray, strike_prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = base[:, None] + coeff[:, None] * strike_prices[None, :]
    return values.mean(axis=0), values.std(axis=0, ddof=0)


def optimize_ppa_contracts(sample_df: pd.DataFrame, config: Dict, penalty_rate: float) -> pd.DataFrame:
    g = pd.to_numeric(sample_df["generation"], errors="coerce").to_numpy(dtype=float)
    d = pd.to_numeric(sample_df["demand"], errors="coerce").to_numpy(dtype=float)
    Ns = pd.to_numeric(sample_df["seller_lmp"], errors="coerce").to_numpy(dtype=float)
    Nb_out = pd.to_numeric(sample_df["buyer_lmp_out"], errors="coerce").to_numpy(dtype=float)
    Nb_in = pd.to_numeric(sample_df["buyer_lmp_in"], errors="coerce").to_numpy(dtype=float)

    strike_prices = np.asarray(config["strike_prices"], dtype=float)
    fixed_volumes = np.asarray(config["fixed_volumes"], dtype=float)
    lambda_s = float(config["lambda_s"])
    lambda_b = float(config["lambda_b"])
    availability_factor = float(config["availability_factor"])
    chunk_size = int(config["fixed_volume_chunk_size"])

    results: List[Dict[str, object]] = []

    # Outside option
    no_contract = evaluate_no_contract(g, d, Ns, Nb_out, Nb_in, lambda_s=lambda_s, lambda_b=lambda_b)
    no_contract_stats = _summary_from_arrays(
        no_contract["FE_s"], no_contract["FE_b"], no_contract["Rev_s"], lambda_s=lambda_s, lambda_b=lambda_b
    )
    results.append({
        "ppa_type": "No Contract",
        "profile_type": "N/A",
        "volume_mw": np.nan,
        "strike_price_mwh": np.nan,
        **no_contract_stats,
        "feasible": True,
    })

    G_avg = float(np.mean(g))
    mean_Ns = float(np.mean(Ns))
    std_Ns = float(np.std(Ns, ddof=0))
    expected_no_contract_revenue = float(np.mean(Ns * g))
    penalty_asg = penalty_rate * np.maximum(0.0, availability_factor * G_avg - g)

    # ---------- Physical AsG ----------
    base_s = g * Ns + penalty_asg
    coeff_s = -g
    base_b = Nb_in * np.maximum(0.0, d - g) - Nb_out * np.maximum(0.0, g - d) - penalty_asg - Nb_in * d
    coeff_b = g
    base_r = -penalty_asg
    coeff_r = g

    seller_mean, seller_std = _linear_stats(base_s, coeff_s, strike_prices)
    buyer_mean, buyer_std = _linear_stats(base_b, coeff_b, strike_prices)
    revenue_mean = (base_r[:, None] + coeff_r[:, None] * strike_prices[None, :]).mean(axis=0)

    for j, pi in enumerate(strike_prices):
        seller_utility = float(seller_mean[j] + lambda_s * seller_std[j])
        buyer_utility = float(buyer_mean[j] + lambda_b * buyer_std[j])
        results.append({
            "ppa_type": "Physical",
            "profile_type": "AsG",
            "volume_mw": np.nan,
            "strike_price_mwh": float(pi),
            "seller_utility": seller_utility,
            "buyer_utility": buyer_utility,
            "seller_mean_exposure": float(seller_mean[j]),
            "buyer_mean_exposure": float(buyer_mean[j]),
            "seller_exposure_variance": float(seller_std[j] ** 2),
            "buyer_exposure_variance": float(buyer_std[j] ** 2),
            "expected_seller_revenue": float(revenue_mean[j]),
            "feasible": bool(buyer_utility <= 0.0),
        })

    # ---------- Virtual AsG ----------
    base_s = 2.0 * Ns * g + penalty_asg
    coeff_s = -g
    base_b = -Ns * g - penalty_asg
    coeff_b = g
    seller_mean, seller_std = _linear_stats(base_s, coeff_s, strike_prices)
    buyer_mean, buyer_std = _linear_stats(base_b, coeff_b, strike_prices)

    for j, pi in enumerate(strike_prices):
        seller_utility = float(seller_mean[j] + lambda_s * seller_std[j])
        buyer_utility = float(buyer_mean[j] + lambda_b * buyer_std[j])
        results.append({
            "ppa_type": "Virtual",
            "profile_type": "AsG",
            "volume_mw": np.nan,
            "strike_price_mwh": float(pi),
            "seller_utility": seller_utility,
            "buyer_utility": buyer_utility,
            "seller_mean_exposure": float(seller_mean[j]),
            "buyer_mean_exposure": float(buyer_mean[j]),
            "seller_exposure_variance": float(seller_std[j] ** 2),
            "buyer_exposure_variance": float(buyer_std[j] ** 2),
            "expected_seller_revenue": float(np.mean(pi * g - penalty_asg)),
            "feasible": bool(buyer_utility <= 0.0),
        })

    # ---------- Physical AsC ----------
    base_s = d * Ns
    coeff_s = -d
    base_b = -Nb_in * d
    coeff_b = d
    seller_mean, seller_std = _linear_stats(base_s, coeff_s, strike_prices)
    buyer_mean, buyer_std = _linear_stats(base_b, coeff_b, strike_prices)
    for j, pi in enumerate(strike_prices):
        seller_utility = float(seller_mean[j] + lambda_s * seller_std[j])
        buyer_utility = float(buyer_mean[j] + lambda_b * buyer_std[j])
        results.append({
            "ppa_type": "Physical",
            "profile_type": "AsC",
            "volume_mw": np.nan,
            "strike_price_mwh": float(pi),
            "seller_utility": seller_utility,
            "buyer_utility": buyer_utility,
            "seller_mean_exposure": float(seller_mean[j]),
            "buyer_mean_exposure": float(buyer_mean[j]),
            "seller_exposure_variance": float(seller_std[j] ** 2),
            "buyer_exposure_variance": float(buyer_std[j] ** 2),
            "expected_seller_revenue": float(expected_no_contract_revenue + np.mean(d * (pi - Ns))),
            "feasible": bool(buyer_utility <= 0.0),
        })

    # ---------- Virtual AsC ----------
    base_s = 2.0 * d * Ns
    coeff_s = -d
    base_b = -Ns * d
    coeff_b = d
    seller_mean, seller_std = _linear_stats(base_s, coeff_s, strike_prices)
    buyer_mean, buyer_std = _linear_stats(base_b, coeff_b, strike_prices)
    for j, pi in enumerate(strike_prices):
        seller_utility = float(seller_mean[j] + lambda_s * seller_std[j])
        buyer_utility = float(buyer_mean[j] + lambda_b * buyer_std[j])
        results.append({
            "ppa_type": "Virtual",
            "profile_type": "AsC",
            "volume_mw": np.nan,
            "strike_price_mwh": float(pi),
            "seller_utility": seller_utility,
            "buyer_utility": buyer_utility,
            "seller_mean_exposure": float(seller_mean[j]),
            "buyer_mean_exposure": float(buyer_mean[j]),
            "seller_exposure_variance": float(seller_std[j] ** 2),
            "buyer_exposure_variance": float(buyer_std[j] ** 2),
            "expected_seller_revenue": float(expected_no_contract_revenue + np.mean((pi - Ns) * d)),
            "feasible": bool(buyer_utility <= 0.0),
        })

    # ---------- Fixed Volume ----------
    valid_fixed_volumes = np.asarray([float(q) for q in fixed_volumes if np.isfinite(q) and float(q) > 0.0], dtype=float)

    for start in range(0, len(valid_fixed_volumes), chunk_size):
        q_chunk = valid_fixed_volumes[start:start + chunk_size]
        q_matrix = q_chunk[None, :]
        d_matrix = d[:, None]
        shortage = np.maximum(d_matrix - q_matrix, 0.0)
        surplus = np.maximum(q_matrix - d_matrix, 0.0)
        buyer_base = Nb_in[:, None] * shortage - Nb_out[:, None] * surplus - (Nb_in * d)[:, None]

        buyer_base_mean = buyer_base.mean(axis=0)
        buyer_base_std = buyer_base.std(axis=0, ddof=0)

        for q_idx, q in enumerate(q_chunk):
            # Physical Fix
            seller_mean_arr = q * (mean_Ns - strike_prices)
            seller_std_arr = q * std_Ns
            buyer_mean_arr = buyer_base_mean[q_idx] + strike_prices * q
            buyer_std_arr = np.full_like(strike_prices, buyer_base_std[q_idx], dtype=float)
            expected_seller_revenue_arr = expected_no_contract_revenue + q * (strike_prices - mean_Ns)

            for j, pi in enumerate(strike_prices):
                seller_utility = float(seller_mean_arr[j] + lambda_s * seller_std_arr)
                buyer_utility = float(buyer_mean_arr[j] + lambda_b * buyer_std_arr[j])
                results.append({
                    "ppa_type": "Physical",
                    "profile_type": "Fix",
                    "volume_mw": float(q),
                    "strike_price_mwh": float(pi),
                    "seller_utility": seller_utility,
                    "buyer_utility": buyer_utility,
                    "seller_mean_exposure": float(seller_mean_arr[j]),
                    "buyer_mean_exposure": float(buyer_mean_arr[j]),
                    "seller_exposure_variance": float(seller_std_arr ** 2),
                    "buyer_exposure_variance": float(buyer_std_arr[j] ** 2),
                    "expected_seller_revenue": float(expected_seller_revenue_arr[j]),
                    "feasible": bool(buyer_utility <= 0.0),
                })

            # Virtual Fix
            seller_mean_arr = q * (2.0 * mean_Ns - strike_prices)
            seller_std_arr = 2.0 * q * std_Ns
            buyer_mean_arr = q * (strike_prices - mean_Ns)
            buyer_std_arr = np.full_like(strike_prices, q * std_Ns, dtype=float)
            expected_seller_revenue_arr = expected_no_contract_revenue + q * (strike_prices - mean_Ns)

            for j, pi in enumerate(strike_prices):
                seller_utility = float(seller_mean_arr[j] + lambda_s * seller_std_arr)
                buyer_utility = float(buyer_mean_arr[j] + lambda_b * buyer_std_arr[j])
                results.append({
                    "ppa_type": "Virtual",
                    "profile_type": "Fix",
                    "volume_mw": float(q),
                    "strike_price_mwh": float(pi),
                    "seller_utility": seller_utility,
                    "buyer_utility": buyer_utility,
                    "seller_mean_exposure": float(seller_mean_arr[j]),
                    "buyer_mean_exposure": float(buyer_mean_arr[j]),
                    "seller_exposure_variance": float(seller_std_arr ** 2),
                    "buyer_exposure_variance": float(buyer_std_arr[j] ** 2),
                    "expected_seller_revenue": float(expected_seller_revenue_arr[j]),
                    "feasible": bool(buyer_utility <= 0.0),
                })

    df_results = pd.DataFrame(results)
    return df_results


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
    return {
        "flat_df": flat_df,
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
        )

    stats = _summary_from_arrays(
        fe_s=out["FE_s"],
        fe_b=out["FE_b"],
        rev_s=out["Rev_s"],
        lambda_s=config["lambda_s"],
        lambda_b=config["lambda_b"],
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

    for ppa_type in ["Physical", "Virtual"]:
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
        allowed_ppa = ["Physical", "Virtual"]

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



# ============================================================
# Hourly FE selection, verification-aware output writing, and batch run
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
    arrays = prepare_sample_arrays(sample_df)
    flat_df = arrays["flat_df"]
    ppa_type = str(solution.get("ppa_type", "No Contract"))
    profile_type = str(solution.get("profile_type", "N/A"))
    strike_price = pd.to_numeric(pd.Series([solution.get("strike_price_mwh", np.nan)]), errors="coerce").iloc[0]
    volume_mw = pd.to_numeric(pd.Series([solution.get("volume_mw", np.nan)]), errors="coerce").iloc[0]

    if ppa_type == "No Contract":
        out = evaluate_no_contract(
            arrays["g"], arrays["d"], arrays["Ns"], arrays["Nb_out"], arrays["Nb_in"],
            lambda_s=config["lambda_s"], lambda_b=config["lambda_b"]
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
            pi=float(strike_price),
            q_fixed=0.0 if pd.isna(volume_mw) else float(volume_mw),
            lambda_s=config["lambda_s"],
            lambda_b=config["lambda_b"],
            availability_factor=config["availability_factor"],
            penalty_rate=penalty_rate,
        )

    fe_df = pd.DataFrame({
        "replication": flat_df["replication"].astype(int),
        "hour_index": flat_df["hour_index"].astype(int),
        "seller_fe": np.asarray(out["FE_s"], dtype=float),
        "buyer_fe": np.asarray(out["FE_b"], dtype=float),
    })

    if config["save_timestamp_in_hourly_fe"]:
        fe_df["timestamp"] = flat_df["timestamp"]

    return fe_df


def build_case_info_long(match_id: int,
                         match_row: Dict[str, object],
                         basic_stats: Dict[str, object],
                         penalty_rate: float,
                         buyer_purchase_q95: float,
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
    add("Risk_Setting", "gamma", config["gamma"])
    add("Risk_Setting", "availability_factor", config["availability_factor"])
    add("Risk_Setting", "penalty_rate", penalty_rate)
    add("Risk_Setting", "buyer_lmp_in_q95", buyer_purchase_q95)

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
    penalty_rate, buyer_purchase_q95 = compute_penalty_rate(sample_df, gamma=config["gamma"])

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
        buyer_purchase_q95=buyer_purchase_q95,
        best_solution=best_solution,
        best_ppa_solution=best_ppa_solution,
        formula_best_solution=formula_best_solution,
        formula_best_ppa_solution=formula_best_ppa_solution,
        verification_report=verification_report,
        scenario_name=config["scenario_name"],
        config=config,
    )
    case_info_path = match_output_dir / "Case_Info.csv"
    case_info_df.to_csv(case_info_path, index=False)

    best_solution_dict = to_plain_dict(best_solution)
    best_solution_dict["match_id"] = int(match_id)
    best_solution_dict["scenario_name"] = config["scenario_name"]
    best_solution_dict["penalty_rate"] = penalty_rate
    best_solution_dict["buyer_lmp_in_q95"] = buyer_purchase_q95
    best_solution_dict["unusual_contracted_volume"] = unusual_contracted_volume_flag(best_solution, sample_df)
    best_solution_dict["formula_selected_ppa_type"] = formula_best_solution.get("ppa_type")
    best_solution_dict["formula_selected_profile_type"] = formula_best_solution.get("profile_type")
    best_solution_dict["formula_selected_volume_mw"] = formula_best_solution.get("volume_mw")
    best_solution_dict["formula_selected_strike_price_mwh"] = formula_best_solution.get("strike_price_mwh")
    best_solution_dict.update(verification_report)
    best_solution_path = match_output_dir / "Best_Solution.csv"
    pd.DataFrame([best_solution_dict]).to_csv(best_solution_path, index=False)

    best_ppa_solution_dict = to_plain_dict(best_ppa_solution) if best_ppa_solution is not None else {}
    if best_ppa_solution_dict:
        best_ppa_solution_dict["match_id"] = int(match_id)
        best_ppa_solution_dict["scenario_name"] = config["scenario_name"]
        best_ppa_solution_dict["penalty_rate"] = penalty_rate
        best_ppa_solution_dict["buyer_lmp_in_q95"] = buyer_purchase_q95
        best_ppa_solution_dict["unusual_contracted_volume"] = unusual_contracted_volume_flag(best_ppa_solution, sample_df)
        if formula_best_ppa_solution is not None:
            best_ppa_solution_dict["formula_best_ppa_type"] = formula_best_ppa_solution.get("ppa_type")
            best_ppa_solution_dict["formula_best_ppa_profile_type"] = formula_best_ppa_solution.get("profile_type")
            best_ppa_solution_dict["formula_best_ppa_volume_mw"] = formula_best_ppa_solution.get("volume_mw")
            best_ppa_solution_dict["formula_best_ppa_strike_price_mwh"] = formula_best_ppa_solution.get("strike_price_mwh")
        best_ppa_solution_dict.update(verification_report)
        best_ppa_solution_path = match_output_dir / "Best_PPA_Solution.csv"
        pd.DataFrame([best_ppa_solution_dict]).to_csv(best_ppa_solution_path, index=False)
    else:
        best_ppa_solution_path = None

    if config["save_per_match_grid"]:
        grid_path = match_output_dir / "Contract_Grid.csv"
        grid_df.to_csv(grid_path, index=False)
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
        fe_df.to_csv(hourly_fe_path, index=False)

        hourly_fe_index_row = {
            "match_id": int(match_id),
            "scenario_name": config["scenario_name"],
            "scenario_type": "baseline",
            "hourly_fe_file": str(hourly_fe_path),
            "fe_source": fe_source,
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
        "buyer_lmp_in_q95": buyer_purchase_q95,
        "fe_source": fe_source,
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



# ============================================================
# Library entry point
# ============================================================
DEFAULT_CONFIG = dict(CONFIG)

def _normalize_config(config_overrides=None):
    cfg = dict(DEFAULT_CONFIG)
    if config_overrides:
        for key, value in dict(config_overrides).items():
            cfg[key] = value
    # normalize arrays and selected match ids
    cfg["strike_prices"] = np.asarray(cfg["strike_prices"], dtype=float)
    cfg["fixed_volumes"] = np.asarray(cfg["fixed_volumes"], dtype=float)
    if cfg.get("selected_match_ids", None) is not None:
        cfg["selected_match_ids"] = [int(x) for x in cfg["selected_match_ids"]]
    cfg["fixed_volume_chunk_size"] = int(cfg["fixed_volume_chunk_size"])
    cfg["verification_top_k_fix_seeds"] = int(cfg["verification_top_k_fix_seeds"])
    cfg["verification_fix_price_radius_steps"] = int(cfg["verification_fix_price_radius_steps"])
    cfg["verification_fix_volume_radius_steps"] = int(cfg["verification_fix_volume_radius_steps"])
    cfg["verification_fix_max_iterations"] = int(cfg["verification_fix_max_iterations"])
    cfg["skip_existing_output"] = bool(cfg["skip_existing_output"])
    cfg["save_per_match_grid"] = bool(cfg["save_per_match_grid"])
    cfg["save_hourly_fe"] = bool(cfg["save_hourly_fe"])
    cfg["save_best_fe_even_if_infeasible"] = bool(cfg["save_best_fe_even_if_infeasible"])
    cfg["save_timestamp_in_hourly_fe"] = bool(cfg["save_timestamp_in_hourly_fe"])
    cfg["verification_enabled"] = bool(cfg["verification_enabled"])
    cfg["verification_exact_all_nonfix"] = bool(cfg["verification_exact_all_nonfix"])
    cfg["verification_fix_allow_cross_ppa_type"] = bool(cfg["verification_fix_allow_cross_ppa_type"])
    return cfg

def run_simulation(config_overrides=None):
    CONFIG = _normalize_config(config_overrides)

    match_dir = resolve_match_dir(CONFIG["match_folder_name"])
    sample_root = resolve_sample_root(CONFIG["sample_root_dir"], match_dir=match_dir)
    match_table_path = resolve_match_table(CONFIG["match_table_file"], match_dir=match_dir)
    match_table_df = load_match_table(match_table_path, sheet_name=CONFIG["match_table_sheet"])
    corr_targets_path = resolve_code2_corr_file(CONFIG["code2_correlation_file"], match_dir=match_dir)
    corr_targets_df = load_code2_corr_targets(corr_targets_path)

    output_root = Path(CONFIG["output_dir"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "Per_Match").mkdir(parents=True, exist_ok=True)
    if CONFIG["save_hourly_fe"]:
        (output_root / "Hourly_FE").mkdir(parents=True, exist_ok=True)

    match_ids = discover_match_ids(match_dir, selected_match_ids=CONFIG["selected_match_ids"])

    print("Resolved paths:")
    print("  Historical match folder :", match_dir)
    print("  Sample root             :", sample_root)
    print("  Match table             :", match_table_path if match_table_path is not None else "Not found (optional)")
    print("  Code 2 corr file        :", corr_targets_path if corr_targets_path is not None else "Not found (optional)")
    print("  Output root             :", output_root)
    print()
    print(f"Matches selected: {len(match_ids)}")
    print(f"Risk preferences: lambda_s={CONFIG['lambda_s']}, lambda_b={CONFIG['lambda_b']}")
    print(f"Scenario name    : {CONFIG['scenario_name']}")
    print(f"Save hourly FE   : {CONFIG['save_hourly_fe']}")
    print(f"Save best PPA FE : {CONFIG['save_best_fe_even_if_infeasible']}")
    print(f"Verification on  : {CONFIG['verification_enabled']}")

    summary_rows = []
    best_solution_rows = []
    best_ppa_solution_rows = []
    hourly_fe_index_rows = []
    run_status_rows = []

    for counter, match_id in enumerate(match_ids, start=1):
        match_output_dir = output_root / "Per_Match" / f"Match_{int(match_id):04d}"
        expected_best_solution = match_output_dir / "Best_Solution.csv"
        expected_case_info = match_output_dir / "Case_Info.csv"
        expected_hourly_fe = output_root / "Hourly_FE" / f"Hourly_FE_Match_{int(match_id):04d}__{CONFIG['scenario_name']}.csv"

        can_skip = expected_best_solution.exists() and expected_case_info.exists()
        if CONFIG["save_hourly_fe"]:
            can_skip = can_skip and expected_hourly_fe.exists()

        if CONFIG["skip_existing_output"] and can_skip:
            print(f"[{counter}/{len(match_ids)}] match_id={match_id}: skipped_existing_output")
            run_status_rows.append({
                "match_id": int(match_id),
                "status": "skipped_existing_output",
                "error_message": "",
            })
            continue

        print(f"[{counter}/{len(match_ids)}] Running match_id={match_id} ...")
        try:
            out = run_one_match(
                match_id=int(match_id),
                match_dir=match_dir,
                sample_root=sample_root,
                output_root=output_root,
                match_table_df=match_table_df,
                corr_targets=corr_targets_df,
                config=CONFIG,
            )

            summary_rows.append(out["summary_row"])
            best_solution_rows.append(out["best_solution_row"])
            if out["best_ppa_solution_row"]:
                best_ppa_solution_rows.append(out["best_ppa_solution_row"])
            if out["hourly_fe_index_row"] is not None:
                hourly_fe_index_rows.append(out["hourly_fe_index_row"])

            run_status_rows.append({
                "match_id": int(match_id),
                "status": "success",
                "error_message": "",
                "decision_changed_after_verification": out["verification_report"].get("formula_best_changed_after_verification"),
                "formula_best_verification_passed": out["verification_report"].get("formula_best_verification_passed"),
                "n_exact_candidates_evaluated": out["verification_report"].get("n_exact_candidates_evaluated"),
            })

            print(
                "    done | "
                f"Formula={out['summary_row']['formula_selected_ppa_type']}/{out['summary_row']['formula_selected_profile_type']} | "
                f"Verified={out['summary_row']['ppa_type']}/{out['summary_row']['profile_type']} | "
                f"Strike={out['summary_row']['strike_price_mwh']} | "
                f"Volume={out['summary_row']['volume_mw']}"
            )

        except Exception as exc:
            run_status_rows.append({
                "match_id": int(match_id),
                "status": "failed",
                "error_message": str(exc),
            })
            print(f"    failed: {exc}")

    summary_df = pd.DataFrame(summary_rows).sort_values("match_id").reset_index(drop=True) if summary_rows else pd.DataFrame()
    best_solution_df = pd.DataFrame(best_solution_rows).sort_values("match_id").reset_index(drop=True) if best_solution_rows else pd.DataFrame()
    best_ppa_solution_df = pd.DataFrame(best_ppa_solution_rows).sort_values("match_id").reset_index(drop=True) if best_ppa_solution_rows else pd.DataFrame()
    hourly_fe_index_df = pd.DataFrame(hourly_fe_index_rows).sort_values(["match_id", "scenario_name"]).reset_index(drop=True) if hourly_fe_index_rows else pd.DataFrame()
    run_status_df = pd.DataFrame(run_status_rows).sort_values("match_id").reset_index(drop=True) if run_status_rows else pd.DataFrame()

    summary_path = output_root / "Simulation_Plot_Data_All_Matches.csv"
    best_solution_path = output_root / "Simulation_Best_Solutions_All_Matches.csv"
    best_ppa_solution_path = output_root / "Simulation_Best_PPA_Solutions_All_Matches.csv"
    hourly_fe_index_path = output_root / "Hourly_FE_Index.csv"
    run_status_path = output_root / "Simulation_Run_Status.csv"
    config_path = output_root / "Simulation_Config.json"

    summary_df.to_csv(summary_path, index=False)
    best_solution_df.to_csv(best_solution_path, index=False)
    best_ppa_solution_df.to_csv(best_ppa_solution_path, index=False)
    hourly_fe_index_df.to_csv(hourly_fe_index_path, index=False)
    run_status_df.to_csv(run_status_path, index=False)

    json_ready_config = dict(CONFIG)
    json_ready_config["strike_prices"] = [float(x) for x in CONFIG["strike_prices"]]
    json_ready_config["fixed_volumes"] = [float(x) for x in CONFIG["fixed_volumes"]]
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(json_ready_config, f, indent=2)

    print()
    print("Saved output files:")
    print("  Plot-ready summary :", summary_path)
    print("  Best solutions     :", best_solution_path)
    print("  Best PPA solutions :", best_ppa_solution_path)
    print("  Hourly FE index    :", hourly_fe_index_path)
    print("  Run status         :", run_status_path)
    print("  Config             :", config_path)

    if not run_status_df.empty:
        print()
        print("Run-status counts:")
        print(run_status_df["status"].value_counts(dropna=False))

    summary_df.head()


    return {
        "summary_df": summary_df,
        "best_solution_df": best_solution_df,
        "best_ppa_solution_df": best_ppa_solution_df,
        "hourly_fe_index_df": hourly_fe_index_df,
        "run_status_df": run_status_df,
        "output_root": output_root,
        "summary_path": summary_path,
        "best_solution_path": best_solution_path,
        "best_ppa_solution_path": best_ppa_solution_path,
        "hourly_fe_index_path": hourly_fe_index_path,
        "run_status_path": run_status_path,
        "config_path": config_path,
    }

if __name__ == "__main__":
    run_simulation()
