#!/usr/bin/env python3
"""Parallel sensitivity runner for the signed-price regeneration pass.

This helper reuses the control cells from
``1_Sensitivity_Run_revised_contract_period.ipynb`` and the local
``verified_simulation_engine.py``. It preserves the notebook's case definitions
and output filenames, but parallelizes the per-match engine calls.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = SCRIPT_DIR / "1_Sensitivity_Run_revised_contract_period.ipynb"
ENGINE_PATH = SCRIPT_DIR / "verified_simulation_engine.py"


def _load_engine_module():
    spec = importlib.util.spec_from_file_location("verified_simulation_engine", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import engine from {ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_notebook_context() -> Dict[str, Any]:
    namespace: Dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(NOTEBOOK_PATH),
    }
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    # Execute code cells through the input-loading/profile-cache stage, but not
    # the original sequential case loop.
    for cell_index in range(1, 8):
        cell = notebook["cells"][cell_index]
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", ""))
        exec(compile(source, f"{NOTEBOOK_PATH.name}:cell{cell_index}", "exec"), namespace)
    return namespace


def _worker_run_one_match(args: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    match_id, overrides = args
    engine = _load_engine_module()
    cfg = engine._normalize_config(overrides)

    match_dir = engine.resolve_match_dir(cfg["match_folder_name"])
    sample_root = engine.resolve_sample_root(cfg["sample_root_dir"], match_dir=match_dir)
    match_table_path = engine.resolve_match_table(cfg["match_table_file"], match_dir=match_dir)
    match_table_df = engine.load_match_table(match_table_path, sheet_name=cfg["match_table_sheet"])
    corr_targets_path = engine.resolve_code2_corr_file(cfg["code2_correlation_file"], match_dir=match_dir)
    corr_targets_df = engine.load_code2_corr_targets(corr_targets_path)

    output_root = Path(cfg["output_dir"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "Per_Match").mkdir(parents=True, exist_ok=True)
    if cfg["save_hourly_fe"]:
        (output_root / "Hourly_FE").mkdir(parents=True, exist_ok=True)

    try:
        out = engine.run_one_match(
            match_id=int(match_id),
            match_dir=match_dir,
            sample_root=sample_root,
            output_root=output_root,
            match_table_df=match_table_df,
            corr_targets=corr_targets_df,
            config=cfg,
        )
        return {
            "match_id": int(match_id),
            "status": "success",
            "error_message": "",
            "summary_row": out["summary_row"],
            "best_solution_row": out["best_solution_row"],
            "best_ppa_solution_row": out["best_ppa_solution_row"],
            "hourly_fe_index_row": out["hourly_fe_index_row"],
            "verification_report": out["verification_report"],
        }
    except Exception as exc:
        return {
            "match_id": int(match_id),
            "status": "failed",
            "error_message": str(exc),
            "traceback": traceback.format_exc(limit=20),
        }


def _write_case_outputs(case_output_dir: Path,
                        cfg: Dict[str, Any],
                        results: List[Dict[str, Any]]) -> Dict[str, Path]:
    success = [r for r in results if r.get("status") == "success"]

    summary_rows = [r["summary_row"] for r in success]
    best_rows = [r["best_solution_row"] for r in success]
    best_ppa_rows = [r["best_ppa_solution_row"] for r in success if r.get("best_ppa_solution_row")]
    hourly_rows = [r["hourly_fe_index_row"] for r in success if r.get("hourly_fe_index_row") is not None]

    run_status_rows = []
    for r in results:
        row = {
            "match_id": int(r["match_id"]),
            "status": r.get("status", ""),
            "error_message": r.get("error_message", ""),
        }
        report = r.get("verification_report") or {}
        row.update({
            "decision_changed_after_verification": report.get("formula_best_changed_after_verification"),
            "formula_best_verification_passed": report.get("formula_best_verification_passed"),
            "n_exact_candidates_evaluated": report.get("n_exact_candidates_evaluated"),
        })
        if r.get("traceback"):
            row["traceback"] = r["traceback"]
        run_status_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("match_id").reset_index(drop=True) if summary_rows else pd.DataFrame()
    best_df = pd.DataFrame(best_rows).sort_values("match_id").reset_index(drop=True) if best_rows else pd.DataFrame()
    best_ppa_df = pd.DataFrame(best_ppa_rows).sort_values("match_id").reset_index(drop=True) if best_ppa_rows else pd.DataFrame()
    hourly_df = pd.DataFrame(hourly_rows).sort_values(["match_id", "scenario_name"]).reset_index(drop=True) if hourly_rows else pd.DataFrame()
    status_df = pd.DataFrame(run_status_rows).sort_values("match_id").reset_index(drop=True) if run_status_rows else pd.DataFrame()

    paths = {
        "summary": case_output_dir / "Simulation_Plot_Data_All_Matches.csv",
        "best": case_output_dir / "Simulation_Best_Solutions_All_Matches.csv",
        "best_ppa": case_output_dir / "Simulation_Best_PPA_Solutions_All_Matches.csv",
        "hourly": case_output_dir / "Hourly_FE_Index.csv",
        "status": case_output_dir / "Simulation_Run_Status.csv",
        "config": case_output_dir / "Simulation_Config.json",
    }
    summary_df.to_csv(paths["summary"], index=False)
    best_df.to_csv(paths["best"], index=False)
    best_ppa_df.to_csv(paths["best_ppa"], index=False)
    hourly_df.to_csv(paths["hourly"], index=False)
    status_df.to_csv(paths["status"], index=False)

    json_ready_config = dict(cfg)
    json_ready_config["strike_prices"] = [float(x) for x in np.asarray(cfg["strike_prices"], dtype=float)]
    json_ready_config["fixed_volumes"] = [float(x) for x in np.asarray(cfg["fixed_volumes"], dtype=float)]
    with open(paths["config"], "w", encoding="utf-8") as f:
        json.dump(json_ready_config, f, indent=2)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()

    os.chdir(SCRIPT_DIR)
    ns = _load_notebook_context()
    engine = ns["engine"]
    config = ns["CONFIG"]
    case_table_df = ns["case_table_df"]
    baseline_sample_root = ns["baseline_sample_root"]
    match_folder_path = ns["match_folder_path"]
    sensitivity_output_root = ns["sensitivity_output_root"]
    profile_cache = ns["profile_cache"]
    build_case_sample_bank = ns["build_case_sample_bank"]
    slugify = ns["slugify"]

    match_ids = engine.discover_match_ids(match_folder_path, selected_match_ids=config["selected_match_ids"])
    manifest_rows = []
    run_status_rows = []

    print(f"Parallel workers: {args.workers}", flush=True)
    print(f"Cases selected: {len(case_table_df)}", flush=True)
    print(f"Matches selected: {len(match_ids)}", flush=True)

    for _, case_row in case_table_df.iterrows():
        case_id = str(case_row["case_id"])
        case_label = str(case_row.get("case_label", case_id))
        case_slug = f"{slugify(case_id)}__{slugify(case_label)}"
        case_output_dir = sensitivity_output_root / "Cases" / case_slug
        case_sample_root = sensitivity_output_root / "Case_Sample_Banks" / case_slug
        case_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Case {case_id}: {case_label} ===", flush=True)
        case_uses_transformed_samples = not (
            np.isclose(float(case_row["volume_resid_multiplier"]), 1.0)
            and np.isclose(float(case_row["price_resid_multiplier"]), 1.0)
        )

        try:
            active_sample_root = build_case_sample_bank(
                case_row=case_row,
                baseline_sample_root=baseline_sample_root,
                case_sample_root=case_sample_root,
                profile_cache=profile_cache,
                rebuild=config["rebuild_case_sample_banks"],
            )

            overrides = {
                "sample_root_dir": str(active_sample_root),
                "match_folder_name": str(match_folder_path),
                "output_dir": str(case_output_dir),
                "lambda_s": float(case_row["lambda_s"]),
                "lambda_b": float(case_row["lambda_b"]),
                "gamma": float(case_row["gamma"]),
                "selected_match_ids": None,
                "save_hourly_fe": bool(config["save_hourly_fe"]),
                "save_best_fe_even_if_infeasible": bool(config["save_best_fe_even_if_infeasible"]),
                "save_per_match_grid": bool(config["save_per_match_grid"]),
                "save_timestamp_in_hourly_fe": bool(config["save_timestamp_in_hourly_fe"]),
                "scenario_name": f"Sensitivity__{case_slug}",
            }

            results: List[Dict[str, Any]] = []
            with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
                futures = {
                    executor.submit(_worker_run_one_match, (int(match_id), overrides)): int(match_id)
                    for match_id in match_ids
                }
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    result = future.result()
                    results.append(result)
                    if completed == 1 or completed % 10 == 0 or completed == len(match_ids):
                        ok = sum(1 for r in results if r.get("status") == "success")
                        failed = sum(1 for r in results if r.get("status") == "failed")
                        print(f"  completed {completed}/{len(match_ids)} | success={ok} failed={failed}", flush=True)

            _write_case_outputs(case_output_dir, engine._normalize_config(overrides), results)
            failed = sum(1 for r in results if r.get("status") == "failed")
            run_status = "success" if failed == 0 else "partial_failed"
            err_msg = "" if failed == 0 else f"{failed} match-level failures"

        except Exception as exc:
            active_sample_root = case_sample_root if case_uses_transformed_samples else baseline_sample_root
            run_status = "failed"
            err_msg = str(exc)
            print(traceback.format_exc(limit=20), flush=True)
            if config["stop_on_error"]:
                raise

        manifest_rows.append({
            "case_id": case_id,
            "case_label": case_label,
            "case_slug": case_slug,
            "case_family": case_row.get("case_family", ""),
            "case_order": case_row.get("case_order", np.nan),
            "case_value": case_row.get("case_value", ""),
            "lambda_s": case_row.get("lambda_s", np.nan),
            "lambda_b": case_row.get("lambda_b", np.nan),
            "gamma": case_row.get("gamma", np.nan),
            "volume_resid_multiplier": case_row.get("volume_resid_multiplier", np.nan),
            "price_resid_multiplier": case_row.get("price_resid_multiplier", np.nan),
            "uses_transformed_samples": case_uses_transformed_samples,
            "sample_root_dir": str(active_sample_root),
            "case_output_dir": str(case_output_dir),
            "status": run_status,
            "notes": case_row.get("notes", ""),
        })
        run_status_rows.append({
            "case_id": case_id,
            "case_label": case_label,
            "status": run_status,
            "error_message": err_msg,
        })

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["case_order", "case_id"]).reset_index(drop=True)
    run_status_df = pd.DataFrame(run_status_rows).sort_values(["case_label", "case_id"]).reset_index(drop=True)
    manifest_df.to_csv(sensitivity_output_root / "Sensitivity_Case_Manifest.csv", index=False)
    run_status_df.to_csv(sensitivity_output_root / "Sensitivity_Case_Run_Status.csv", index=False)
    print("\nSaved sensitivity manifest and run status.", flush=True)
    print(run_status_df.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
