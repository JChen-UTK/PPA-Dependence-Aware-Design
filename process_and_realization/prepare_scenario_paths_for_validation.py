#!/usr/bin/env python3
"""
Prepare scenario paths in a consolidated location for validation.

This script creates a unified directory with properly labeled files for the validation script.
Processes files one at a time to avoid excessive memory usage.
"""

import os
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

def extract_shift_value(label):
    """Extract shift value from label like 'target_physical_shift_m0p10'."""
    match = re.search(r'([pm])(\d+)p(\d+)', label)
    if match:
        sign = -1 if match.group(1) == 'm' else 1
        value = float(f"{match.group(2)}.{match.group(3)}") * sign
        return value
    return 0.0

def prepare_baseline_files(base_dir, output_dir):
    """Copy and rename baseline files."""
    baseline_dir = Path(base_dir) / 'Input data and files' / 'Simplified baseline realized samples'
    output_baseline = output_dir / 'baseline'
    output_baseline.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for rep_num in range(1, 6):
        rep_dir = baseline_dir / f'rep_{rep_num:02d}'
        if not rep_dir.exists():
            continue
        
        for match_file in sorted(rep_dir.glob('*.csv')):
            match_num = int(match_file.stem)
            
            # Read the file and add scenario columns
            df = read_csv_optimized(match_file)
            df['replication'] = rep_num
            df['match_id'] = match_num
            df['scenario_name'] = 'Baseline__No_Mutation__Verified'
            df['scenario'] = 'Baseline__No_Mutation__Verified'
            df['mutation_family'] = 'baseline'
            df['target_shift'] = 0.0
            
            # Rename columns
            rename_map = {
                'hour_index': 'hour',
                'generation': 'G',
                'demand': 'D',
                'seller_lmp': 'P_S',
                'buyer_lmp_in': 'P_B_in',
                'buyer_lmp_out': 'P_B_out',
            }
            for old, new in rename_map.items():
                if old in df.columns:
                    df = df.rename(columns={old: new})
            
            # Save file
            output_file = output_baseline / f'Match_{match_num:04d}__Baseline__No_Mutation__Verified.csv'
            df.to_csv(output_file, index=False)
            count += 1
            if count % 10 == 0:
                print(f"  Processed {count} baseline files...")
    
    print(f"Total baseline files: {count}")
    return count

def prepare_mutation_files(base_dir, output_dir):
    """Copy and rename mutation files."""
    mutation_dir = Path(base_dir) / 'simulation_mutation' / 'mutation_samples'
    
    family_map = {
        'basis': 'BasisDeterioration',
        'cannibalization': 'SellerCannibalization',
        'load_price': 'LoadPriceIntensification',
        'shape': 'ShapeDeterioration',
    }
    
    count = 0
    for family_folder, family_name in family_map.items():
        family_dir = mutation_dir / family_folder
        if not family_dir.exists():
            continue
        
        for shift_dir in sorted(family_dir.glob('target_physical_shift_*')):
            shift_label = shift_dir.name
            shift_value = extract_shift_value(shift_label)
            
            output_mutation = output_dir / f'{family_folder}_{shift_label}'
            output_mutation.mkdir(parents=True, exist_ok=True)
            
            for match_file in sorted(shift_dir.glob('*.csv')):
                filename = match_file.stem
                match_id = int(filename.split('_')[1])
                
                # Read the file and add scenario columns
                df = read_csv_optimized(match_file)
                df['replication'] = df['replication'].astype(int)
                df['match_id'] = match_id
                scenario_name = f'Mutation__{family_name}__{shift_label}'
                df['scenario_name'] = scenario_name
                df['scenario'] = scenario_name
                df['mutation_family'] = family_name
                df['target_shift'] = shift_value
                
                # Rename columns
                rename_map = {
                    'hour_index': 'hour',
                    'generation': 'G',
                    'demand': 'D',
                    'seller_lmp': 'P_S',
                    'buyer_lmp_in': 'P_B_in',
                    'buyer_lmp_out': 'P_B_out',
                }
                for old, new in rename_map.items():
                    if old in df.columns:
                        df = df.rename(columns={old: new})
                
                # Save file
                output_file = output_mutation / match_file.name
                df.to_csv(output_file, index=False)
                count += 1
                if count % 50 == 0:
                    print(f"  Processed {count} mutation files...")
    
    print(f"Total mutation files: {count}")
    return count

def main():
    base_dir = Path(__file__).resolve().parents[1]
    output_dir = base_dir / 'process_and_realization' / 'scenario_paths_for_validation'
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Preparing baseline files...")
    baseline_count = prepare_baseline_files(base_dir, output_dir)
    
    print("\nPreparing mutation files...")
    mutation_count = prepare_mutation_files(base_dir, output_dir)
    
    print(f"\nTotal files prepared: {baseline_count + mutation_count}")
    print(f"Output directory: {output_dir}")
    print(f"Ready for validation with glob: {output_dir}/**/*.csv")

if __name__ == '__main__':
    main()
