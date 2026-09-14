#!/usr/bin/env python3
"""
Simple wrapper to add match_id and scenario columns to scenario path files.
This modifies files in-place efficiently.
"""

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
from pathlib import Path
import re
import sys

def get_file_batch_info(filepath):
    """Extract match_id and scenario from file path."""
    parts = filepath.parts
    
    # For mutation files: ...mutation_samples/{family}/{shift}/Match_XXXX__...csv
    match_id = None
    scenario = None
    family = None
    shift = None
    
    # Extract match ID from filename
    filename = filepath.name
    match = re.search(r'Match_(\d+)', filename)
    if match:
        match_id = int(match.group(1))
    elif filename.isdigit() or (Path(filename).stem.isdigit()):
        # For baseline files like '001.csv'
        match_id = int(Path(filename).stem)
    
    # Try to infer scenario/family from path structure
    for i, part in enumerate(parts):
        if part in ('basis', 'cannibalization', 'load_price', 'shape'):
            family = part
            # Next part should be the shift like 'target_physical_shift_m0p10'
            if i + 1 < len(parts) and parts[i + 1].startswith('target_physical_shift'):
                shift = parts[i + 1]
                scenario = f'Mutation__{"BasisDeterioration" if family=="basis" else "SellerCannibalization" if family=="cannibalization" else "LoadPriceIntensification" if family=="load_price" else "ShapeDeterioration"}__{shift}'
                break
    
    # Fallback to filename parsing
    if scenario is None:
        if 'Baseline' in filename:
            scenario = 'Baseline__No_Mutation__Verified'
            family = 'baseline'
        else:
            # Try to parse from filename
            family_match = re.search(r'__([\w]+)__target_physical', filename)
            if family_match:
                family = family_match.group(1)
                shift_match = re.search(r'target_physical_shift_([mp\d.p]+)', filename)
                if shift_match:
                    shift = 'target_physical_shift_' + shift_match.group(1)
                    scenario = f'Mutation__{family}__{shift}'
                    family_map = {
                        'BasisDeterioration': 'basis',
                        'SellerCannibalization': 'cannibalization',
                        'LoadPriceIntensification': 'load_price',
                        'ShapeDeterioration': 'shape',
                    }
                    family = family_map.get(family, family.lower())
    
    return match_id, scenario, family

def process_file(filepath):
    """Add metadata columns to a single file."""
    try:
        match_id, scenario, family = get_file_batch_info(filepath)
        
        if match_id is None or scenario is None:
            print(f"Skipping {filepath.name} - could not extract metadata")
            return False
        
        df = read_csv_optimized(filepath)
        
        # Add columns
        df['match_id'] = match_id
        df['scenario'] = scenario
        if family:
            df['mutation_family'] = family
        
        # Save
        df.to_csv(filepath, index=False)
        return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    base_dir = Path(__file__).resolve().parents[1]
    
    # Process mutation files
    mutation_base = base_dir / 'simulation_mutation' / 'mutation_samples'
    processed = 0
    for csv_file in sorted(mutation_base.rglob('*.csv')):
        if 'Match_' not in csv_file.name:
            continue  # Skip index/diagnostic files
        
        if process_file(csv_file):
            processed += 1
            if processed % 200 == 0:
                print(f"Processed {processed} mutation files...")
                sys.stdout.flush()
    
    print(f"Mutation files: {processed}")
    
    # Process baseline files
    baseline_base = base_dir / 'Input data and files' / 'Simplified baseline realized samples'
    processed = 0
    for csv_file in sorted(baseline_base.rglob('*.csv')):
        if process_file(csv_file):
            processed += 1
            if processed % 10 == 0:
                print(f"Processed {processed} baseline files...")
                sys.stdout.flush()
    
    print(f"Baseline files: {processed}")

if __name__ == '__main__':
    main()
