#!/usr/bin/env python3
"""
Quick preprocessing script to add match_id and scenario metadata to mutation samples.
Processes files in-place to avoid excessive storage.
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

def extract_match_id(filename):
    """Extract match ID from filename like 'Match_0001__Mutation__BasisDeterioration__target_physical_shift_m0p10.csv'"""
    match = re.search(r'Match_(\d+)', filename)
    if match:
        return int(match.group(1))
    # Fallback for other formats
    parts = filename.split('_')
    for part in parts:
        if part.isdigit():
            return int(part)
    raise ValueError(f"Could not extract match ID from {filename}")

def extract_scenario_info(filename):
    """Extract scenario information from filename."""
    # Format: Match_XXXX__Mutation__FamilyName__target_physical_shift_VALUE.csv
    if 'Baseline' in filename:
        return 'Baseline__No_Mutation__Verified', 'baseline', 0.0
    
    # Extract family name
    family_match = re.search(r'__([\w]+)__target_physical', filename)
    if not family_match:
        return None, None, None
    family = family_match.group(1)
    
    # Extract shift value
    shift_match = re.search(r'target_physical_shift_([mp])(\d+)p(\d+)', filename)
    if shift_match:
        sign = -1 if shift_match.group(1) == 'm' else 1
        shift_value = sign * float(f"{shift_match.group(2)}.{shift_match.group(3)}")
    else:
        shift_value = 0.0
    
    scenario = f'Mutation__{family}__target_physical_shift_{"m" if shift_value < 0 else "p"}{abs(shift_value):.2f}'.replace('.', 'p')
    return scenario, family, shift_value

def process_mutation_files(root_dir):
    """Add metadata columns to all mutation sample files."""
    root = Path(root_dir)
    processed = 0
    
    for csv_file in sorted(root.rglob('*.csv')):
        try:
            # Extract metadata from filename
            filename = csv_file.name
            match_id = extract_match_id(filename)
            scenario, family, shift_value = extract_scenario_info(filename)
            
            if scenario is None:
                print(f"Skipping {csv_file.name} - could not extract metadata")
                continue
            
            # Read and add metadata
            df = read_csv_optimized(csv_file)
            df['match_id'] = match_id
            df['scenario'] = scenario
            df['mutation_family'] = family
            df['target_shift'] = shift_value
            
            # Save back
            df.to_csv(csv_file, index=False)
            processed += 1
            
            if processed % 100 == 0:
                print(f"Processed {processed} files...")
                sys.stdout.flush()
        
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    print(f"Total processed: {processed}")
    return processed

def process_baseline_files(root_dir):
    """Add metadata columns to baseline files."""
    root = Path(root_dir)
    processed = 0
    
    for rep_dir in sorted(root.glob('rep_*')):
        if not rep_dir.is_dir():
            continue
        
        rep_num = int(rep_dir.name.split('_')[1])
        
        for csv_file in sorted(rep_dir.glob('*.csv')):
            try:
                match_id = int(csv_file.stem)
                
                df = read_csv_optimized(csv_file)
                df['match_id'] = match_id
                df['replication'] = rep_num
                df['scenario'] = 'Baseline__No_Mutation__Verified'
                df['mutation_family'] = 'baseline'
                df['target_shift'] = 0.0
                
                df.to_csv(csv_file, index=False)
                processed += 1
                
                if processed % 10 == 0:
                    print(f"Processed {processed} baseline files...")
                    sys.stdout.flush()
            
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue
    
    print(f"Total baseline processed: {processed}")
    return processed

if __name__ == '__main__':
    import sys
    base_dir = Path(__file__).resolve().parents[1]
    
    print("Processing mutation sample files...")
    mutation_dir = base_dir / 'simulation_mutation' / 'mutation_samples'
    mutation_count = process_mutation_files(mutation_dir)
    
    print("\nProcessing baseline files...")
    baseline_dir = base_dir / 'Input data and files' / 'Simplified baseline realized samples'
    baseline_count = process_baseline_files(baseline_dir)
    
    print(f"\nTotal files processed: {mutation_count + baseline_count}")
