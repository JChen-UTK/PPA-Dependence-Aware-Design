from pathlib import Path
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
import numpy as np
import re

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
# 1) Settings
# ============================================================
SEARCH_ROOTS = _build_search_roots()

OUTPUT_FOLDER_NAME = "Match files of original data"

SEASON_MAPPING = {
    12: "Q1", 1: "Q1", 2: "Q1",
    3: "Q2", 4: "Q2", 5: "Q2",
    6: "Q3", 7: "Q3", 8: "Q3",
    9: "Q4", 10: "Q4", 11: "Q4",
}

CLIP_NEGATIVE_GENERATION = True
PURCHASE_IN_RULE = "scaled_beta_absolute_spread"
BETA_A = 2.0
BETA_B = 3.0
MARKUP_LOW = 0.5
MARKUP_HIGH = 2.0
GLOBAL_SEED = 20260423

# ============================================================
# 2) Helpers
# ============================================================
def resolve_existing_path(candidates, description):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {description}. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def resolve_pjm_root():
    candidates = []
    for root in SEARCH_ROOTS:
        candidates.extend([
            root / "Input data and files" / "PJM Data",
            root / "Data and files" / "PJM Data",
            root / "PJM Data",
        ])
    return resolve_existing_path(candidates, "PJM raw-data root (PJM Data)")


def resolve_workbook_candidates(filename):
    candidates = []
    for root in SEARCH_ROOTS:
        candidates.extend([
            root / "Input data and files" / filename,
            root / "Data and files" / filename,
            root / filename,
        ])
    return candidates



def canon(x):
    if pd.isna(x):
        return None
    x = str(x).strip().upper()
    x = x.replace("-", "")
    x = x.replace("&", "AND")
    x = re.sub(r"[^A-Z0-9]", "", x)
    return x or None



def find_first_existing(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None



def assign_time_parts(df, dt_col="datetime"):
    df = df.copy()
    df["hour"] = df[dt_col].dt.hour
    df["quarter"] = df[dt_col].dt.month.map(SEASON_MAPPING)
    return df



def build_locational_mappers(locational_info_path):
    loc_df = read_excel_optimized(locational_info_path).copy()

    for col in loc_df.columns:
        loc_df[col] = loc_df[col].astype(str).str.strip()

    mr_col = find_first_existing(
        loc_df,
        ["Market Region (mkt_region)", "Market_Region", "Market Region"],
    )
    lmp_zone_col = find_first_existing(
        loc_df,
        ["LMP Transmission Zone", "LMP_Zone", "LMP Zone", "Zone"],
    )
    load_zone_col = find_first_existing(
        loc_df,
        ["Load Data zone", "Load_Data_Zone", "Load Zone", "Load_Zone"],
    )
    load_area_list_col = find_first_existing(
        loc_df,
        [
            "Load Data load_area (Matches to LMP Zone)",
            "Load Data load_area",
            "Load_Area_List",
            "Load Area",
            "Load_Area",
            "load_area",
        ],
    )

    if lmp_zone_col is None:
        raise KeyError("Could not find the LMP zone column in Locational Information.xlsx")

    lmpzone_to_region = {}
    loadzone_to_lmpzone = {}
    loadarea_to_lmpzone = {}

    for _, row in loc_df.iterrows():
        lmp_zone = canon(row[lmp_zone_col]) if lmp_zone_col else None
        market_region = canon(row[mr_col]) if mr_col else None
        load_zone = canon(row[load_zone_col]) if load_zone_col else None

        if lmp_zone and market_region:
            lmpzone_to_region[lmp_zone] = market_region
        if load_zone and lmp_zone:
            loadzone_to_lmpzone[load_zone] = lmp_zone

        raw_areas = str(row[load_area_list_col]) if load_area_list_col else ""
        for area in re.split(r"[,;]", raw_areas):
            area = area.strip()
            if area and area.upper() != "NAN":
                loadarea_to_lmpzone[canon(area)] = lmp_zone

    return {
        "lmpzone_to_region": lmpzone_to_region,
        "loadzone_to_lmpzone": loadzone_to_lmpzone,
        "loadarea_to_lmpzone": loadarea_to_lmpzone,
    }



def load_match_table(match_table_path):
    xls = pd.ExcelFile(match_table_path)
    if "All_Matches" in xls.sheet_names:
        df = read_excel_optimized(match_table_path, sheet_name="All_Matches")
    else:
        candidate = None
        for sh in xls.sheet_names:
            tmp = read_excel_optimized(match_table_path, sheet_name=sh)
            if "match_id" in tmp.columns:
                candidate = tmp
                break
        if candidate is None:
            raise KeyError("Could not find a sheet with a 'match_id' column in Seller_Buyer_Match_Table.xlsx")
        df = candidate

    required = [
        "match_id",
        "seller_market_region",
        "buyer_zone",
        "seller_lmp_file",
        "buyer_lmp_file",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Match table is missing required columns: {missing}")

    df = df.copy()
    df["seller_market_region_norm"] = df["seller_market_region"].map(canon)
    df["buyer_zone_norm"] = df["buyer_zone"].map(canon)
    df["seller_lmp_file_norm"] = df["seller_lmp_file"].astype(str).str.strip().str.upper()
    df["buyer_lmp_file_norm"] = df["buyer_lmp_file"].astype(str).str.strip().str.upper()

    return df



def load_generation_raw(pjm_root):
    path = pjm_root / "Generation" / "hourly_total_gen_all_area.csv"
    df = read_csv_optimized(path)

    time_col = find_first_existing(df, ["datetime_beginning_ept", "datetime", "timestamp"])
    area_col = find_first_existing(df, ["area", "Area"])
    value_col = find_first_existing(df, ["total_generation_mw", "mw", "value"])

    if not all([time_col, area_col, value_col]):
        raise KeyError("Generation file is missing one or more required columns.")

    df["datetime"] = pd.to_datetime(df[time_col], errors="coerce")
    df["MARKET_REGION"] = df[area_col].map(canon)
    df["generation"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["datetime", "MARKET_REGION", "generation"]).copy()

    if CLIP_NEGATIVE_GENERATION:
        df["generation"] = df["generation"].clip(lower=0)

    return df[["datetime", "MARKET_REGION", "generation"]]



def load_demand_raw_to_lmp_zone(pjm_root, mappers):
    path = pjm_root / "Demand" / "hourly_metered_load_all.csv"
    df_raw = read_csv_optimized(path)

    time_col = find_first_existing(df_raw, ["datetime_beginning_ept", "datetime", "timestamp"])
    load_area_col = find_first_existing(df_raw, ["load_area", "Load_Area", "Load Area"])
    zone_col = find_first_existing(df_raw, ["zone", "Zone"])
    value_col = find_first_existing(df_raw, ["mw", "MW", "value"])

    if not all([time_col, load_area_col, value_col]):
        raise KeyError("Demand file is missing one or more required columns.")

    df = df_raw.copy()
    df["datetime"] = pd.to_datetime(df[time_col], errors="coerce")
    df["LOAD_AREA"] = df[load_area_col].map(canon)
    df["demand"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["datetime", "LOAD_AREA", "demand"]).copy()

    df["LMP_ZONE"] = df["LOAD_AREA"].map(mappers["loadarea_to_lmpzone"])

    if zone_col is not None:
        zone_lookup = (
            df_raw[[load_area_col, zone_col]]
            .drop_duplicates()
            .copy()
            .rename(columns={load_area_col: "load_area_raw", zone_col: "load_zone_raw"})
        )
        zone_lookup["LOAD_AREA"] = zone_lookup["load_area_raw"].map(canon)
        zone_lookup["LOAD_ZONE"] = zone_lookup["load_zone_raw"].map(canon)
        loadarea_to_loadzone = dict(zip(zone_lookup["LOAD_AREA"], zone_lookup["LOAD_ZONE"]))

        missing = df["LMP_ZONE"].isna()
        df.loc[missing, "LOAD_ZONE"] = df.loc[missing, "LOAD_AREA"].map(loadarea_to_loadzone)
        df.loc[missing, "LMP_ZONE"] = df.loc[missing, "LOAD_ZONE"].map(mappers["loadzone_to_lmpzone"])

    df = df.dropna(subset=["LMP_ZONE"]).copy()

    df = df.groupby(["datetime", "LMP_ZONE"], as_index=False)["demand"].sum()
    return df[["datetime", "LMP_ZONE", "demand"]]



def index_folder_csvs(folder_path):
    return {p.name.upper(): p for p in folder_path.glob("*.csv")}



def load_exact_seller_lmp(path):
    df = read_csv_optimized(path)
    time_col = find_first_existing(df, ["datetime_beginning_ept", "datetime", "timestamp"])
    value_col = find_first_existing(df, ["total_lmp_rt", "lmp", "value"])
    if not all([time_col, value_col]):
        raise KeyError(f"Seller LMP file is missing one or more required columns: {path}")

    df["datetime"] = pd.to_datetime(df[time_col], errors="coerce")
    df["seller_lmp"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["datetime", "seller_lmp"]).copy()
    return df[["datetime", "seller_lmp"]]



def load_exact_buyer_lmp(path):
    df = read_csv_optimized(path)
    time_col = find_first_existing(df, ["datetime_beginning_ept", "datetime", "timestamp"])
    value_col = find_first_existing(df, ["total_lmp_rt", "lmp", "value"])
    if not all([time_col, value_col]):
        raise KeyError(f"Buyer LMP file is missing one or more required columns: {path}")

    df["datetime"] = pd.to_datetime(df[time_col], errors="coerce")
    df["buyer_lmp_out"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["datetime", "buyer_lmp_out"]).copy()
    return df[["datetime", "buyer_lmp_out"]]



def generate_buyer_lmp_in(nb_out_series, seed):
    if PURCHASE_IN_RULE != "scaled_beta_absolute_spread":
        raise ValueError(f"Unsupported PURCHASE_IN_RULE: {PURCHASE_IN_RULE}")

    rng = np.random.default_rng(seed)
    x = MARKUP_LOW + (MARKUP_HIGH - MARKUP_LOW) * rng.beta(BETA_A, BETA_B, size=len(nb_out_series))
    nb_out = nb_out_series.to_numpy(dtype=float)
    return nb_out + x * np.abs(nb_out)


# ============================================================
# 3) Resolve files and folders
# ============================================================
pjm_root = resolve_pjm_root()
match_table_path = resolve_existing_path(
    resolve_workbook_candidates("Seller_Buyer_Match_Table.xlsx"),
    "Seller_Buyer_Match_Table.xlsx",
)
locational_info_path = resolve_existing_path(
    resolve_workbook_candidates("Locational Information.xlsx"),
    "Locational Information.xlsx",
)

seller_lmp_folder = pjm_root / "Gen_LMP"
buyer_lmp_folder = pjm_root / "Demand_LMP"

output_dir = pjm_root.parent / OUTPUT_FOLDER_NAME
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# 4) Load source tables once
# ============================================================
mappers = build_locational_mappers(locational_info_path)
match_df = load_match_table(match_table_path)
generation_df = load_generation_raw(pjm_root)
demand_df = load_demand_raw_to_lmp_zone(pjm_root, mappers)

seller_file_index = index_folder_csvs(seller_lmp_folder)
buyer_file_index = index_folder_csvs(buyer_lmp_folder)

seller_cache = {}
buyer_cache = {}

# ============================================================
# 5) Extract and save one CSV per match
# ============================================================
summary_rows = []
errors = []

for _, row in match_df.sort_values("match_id").iterrows():
    match_id = int(row["match_id"])
    out_name = f"{match_id:03d}.csv"

    seller_region = row["seller_market_region_norm"]
    buyer_zone = row["buyer_zone_norm"]
    seller_file_key = row["seller_lmp_file_norm"]
    buyer_file_key = row["buyer_lmp_file_norm"]

    try:
        if seller_file_key not in seller_file_index:
            raise FileNotFoundError(f"Seller LMP file not found in Gen_LMP folder: {seller_file_key}")
        if buyer_file_key not in buyer_file_index:
            raise FileNotFoundError(f"Buyer LMP file not found in Demand_LMP folder: {buyer_file_key}")

        if seller_file_key not in seller_cache:
            seller_cache[seller_file_key] = load_exact_seller_lmp(seller_file_index[seller_file_key])
        if buyer_file_key not in buyer_cache:
            buyer_cache[buyer_file_key] = load_exact_buyer_lmp(buyer_file_index[buyer_file_key])

        seller_lmp_df = seller_cache[seller_file_key]
        buyer_lmp_df = buyer_cache[buyer_file_key]

        gen_part = generation_df.loc[
            generation_df["MARKET_REGION"] == seller_region,
            ["datetime", "generation"],
        ].copy()

        dem_part = demand_df.loc[
            demand_df["LMP_ZONE"] == buyer_zone,
            ["datetime", "demand"],
        ].copy()

        merged = (
            gen_part
            .merge(dem_part, on="datetime", how="inner")
            .merge(seller_lmp_df, on="datetime", how="inner")
            .merge(buyer_lmp_df, on="datetime", how="inner")
        )

        if merged.empty:
            raise ValueError("Merged match-level data is empty after joining generation, demand, seller LMP, and buyer LMP.")

        merged = merged.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)
        merged = assign_time_parts(merged, dt_col="datetime")
        merged["buyer_lmp_in"] = generate_buyer_lmp_in(
            merged["buyer_lmp_out"],
            seed=GLOBAL_SEED + match_id,
        )

        final_df = merged[[
            "datetime",
            "hour",
            "quarter",
            "generation",
            "demand",
            "seller_lmp",
            "buyer_lmp_out",
            "buyer_lmp_in",
        ]].rename(columns={"datetime": "timestamp"})

        final_df.to_csv(output_dir / out_name, index=False)

        summary_rows.append(
            {
                "match_id": match_id,
                "output_file": out_name,
                "n_rows": len(final_df),
            }
        )

    except Exception as exc:
        errors.append({"match_id": match_id, "error": str(exc)})

summary_df = pd.DataFrame(summary_rows)
error_df = pd.DataFrame(errors)

print(f"PJM root resolved to:      {pjm_root}")
print(f"Match table resolved to:   {match_table_path}")
print(f"Locational info resolved:  {locational_info_path}")
print(f"Output folder:             {output_dir}")
print(f"Expected matches:          {match_df['match_id'].nunique()}")
print(f"Files written:             {len(summary_df)}")
print(f"Files failed:              {len(error_df)}")

if not summary_df.empty:
    print("\nSample of created files:")
    print(summary_df.head(10).to_string(index=False))

if not error_df.empty:
    print("\nFailures:")
    print(error_df.to_string(index=False))
