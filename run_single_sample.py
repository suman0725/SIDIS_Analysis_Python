#!/usr/bin/env python3
"""
run_single_sample.py - Non-Interactive Batch Version
Optimized for RG-D Targets: LD2, Carbon, Copper, and Tin.
"""

import os
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import uproot

# Physics & Bank Module Imports
from physics_constants import E_BEAM
from common_cuts import detect_polarity
from physics import (
    get_Q2, get_W, get_y, get_xB, get_nu,
    get_zh, get_pt2, get_phih, get_theta, get_phi,
)
from electron_cuts import electron_cutflow
from bank_builders import REC_BRANCHES, MC_BRANCHES, build_per_particle_arrays
from pids import PID

FORWARD_STATUS_MIN = 2000
FORWARD_STATUS_MAX = 4000

def forward_status_mask(status):
    status = np.asarray(status)
    abs_status = np.abs(status)
    return (abs_status >= FORWARD_STATUS_MIN) & (abs_status < FORWARD_STATUS_MAX)

# ==========================
# 1. Process one ROOT file
# ==========================

def process_file(path, target, sample_type="data",
                 max_events=None, apply_tm=False, tm_min=None,
                 start_event_idx=0, diagnostic_mode=True):
    
    pol = detect_polarity(path)
    branch_list = list(REC_BRANCHES)
    if sample_type == "sim" and apply_tm:
        branch_list += MC_BRANCHES

    with uproot.open(path) as f:
        tree = f["data"]
        arrs = tree.arrays(branch_list, library="ak", entry_stop=max_events)

    df_all = build_per_particle_arrays(arrs, target_group=target)

    if sample_type == "sim" and apply_tm:
        from truth_matching import add_truth_matching
        df_all = add_truth_matching(df_all, arrs, quality_min=tm_min)

    # --- THE ACCOUNTANT (Inventory Report) ---
    if diagnostic_mode:
        f_mask = forward_status_mask(df_all["status"].to_numpy())
        df_fd = df_all[f_mask]
        print(f"\n{'='*60}\n ACCOUNTANT REPORT: {os.path.basename(path)}\n{'-'*60}")
        print(f" CATEGORY            |  TOTAL (All)  |  FORWARD (FD)")
        print(f"{'-'*60}")
        # Note: Added fallback for 'charge' check if not in df_all
        c_pos = df_all['charge'] > 0 if 'charge' in df_all.columns else (df_all['pid'] == 211)
        c_neg = df_all['charge'] < 0 if 'charge' in df_all.columns else (df_all['pid'] == 11)
        
        counts = {
            "Total Tracks": (len(df_all), len(df_fd)),
            "Positives (+)": (len(df_all[c_pos]), len(df_fd[df_fd['pid'].isin([211, 2212, 321])])),
            "Negatives (-)": (len(df_all[c_neg]), len(df_fd[df_fd['pid'].isin([11, -211])])),
            "PID 0": (len(df_all[df_all['pid']==0]), len(df_fd[df_fd['pid']==0])),
            "Electrons (11)": (len(df_all[df_all['pid']==11]), len(df_fd[df_fd['pid']==11])),
            "Pions (211)": (len(df_all[df_all['pid']==211]), len(df_fd[df_fd['pid']==211])),
        }
        for lbl, (ac, fc) in counts.items():
            print(f" {lbl:<19} | {ac:<13} | {fc:<12}")
        print(f"{'='*60}\n")

    # --- ELECTRON SELECTION ---
    final_mask, cf_steps, masks = electron_cutflow(
    df_all, 
    target=target, 
    polarity=pol, 
    sample_type=sample_type  # <--- Make sure this is here!
)
    df_e_all = df_all[final_mask].copy()

    if not df_e_all.empty:
        px, py, pz = df_e_all["px"].to_numpy(), df_e_all["py"].to_numpy(), df_e_all["pz"].to_numpy()
        df_e_all["Q2"], df_e_all["xB"], df_e_all["W"], df_e_all["y"] = \
            get_Q2(E_BEAM, px, py, pz), get_xB(E_BEAM, px, py, pz), get_W(E_BEAM, px, py, pz), get_y(E_BEAM, px, py, pz)
        df_e_all["theta"], df_e_all["phi"] = get_theta(px, py, pz, True), get_phi(px, py, True)

    # --- PION SELECTION & DIAGNOSTICS ---
    f_mask_all = forward_status_mask(df_all["status"].to_numpy())
    pip_mask = (df_all["pid"] == PID.PION_PLUS) & f_mask_all
    df_pip_diag = df_all[pip_mask].copy()
    
    golden_ids = df_e_all["event_idx_local"].unique() if not df_e_all.empty else []
    df_pip_diag["is_after_electron_cut"] = df_pip_diag["event_idx_local"].isin(golden_ids)

    # --- DIAGNOSTIC SAVING BLOCK (FIXED) ---
    if diagnostic_mode:
        # Define base first so it can be used in filenames
        base = os.path.splitext(os.path.basename(path))[0]
        
        # Save Pion Diagnostics
        df_pip_diag.to_parquet(f"diag_pip_{target}_{base}.parquet")
        
        # Save Electron Diagnostics (Filtering to match lengths)
        e_only_mask = (df_all['pid'] == 11)
        df_e_diag = df_all[e_only_mask].copy()
        for step, m_arr in masks.items():
            df_e_diag[f"pass_{step}"] = m_arr[e_only_mask]
        
        df_e_diag.to_parquet(f"diag_e_{target}_{base}.parquet")

    if df_e_all.empty:
        return pd.DataFrame(), cf_steps, len(arrs), 0

    # --- PAIRING & MERGE ---
    df_e_top = df_e_all.sort_values(["event_idx_local", "p"], ascending=[True, False]).groupby("event_idx_local", as_index=False).head(1)
    df_pip_diag["theta"], df_pip_diag["phi"] = get_theta(df_pip_diag["px"], df_pip_diag["py"], df_pip_diag["pz"], True), get_phi(df_pip_diag["px"], df_pip_diag["py"], True)
    
    df_e_m = df_e_top.rename(columns={"px":"e_px", "py":"e_py", "pz":"e_pz", "p":"e_p", "theta":"e_theta", "phi":"e_phi", "event_id":"rc_event"})
    df_pip_m = df_pip_diag[["event_idx_local", "px", "py", "pz", "p", "theta", "phi"]].rename(columns={"px":"pip_px", "py":"pip_py", "pz":"pip_pz", "p":"pip_p", "theta":"pip_theta", "phi":"pip_phi"})

    df_sidis = pd.merge(df_e_m, df_pip_m, on="event_idx_local", how="left")
    
    if not df_sidis.empty:
        df_sidis["zh"] = get_zh(E_BEAM, df_sidis["e_px"], df_sidis["e_py"], df_sidis["e_pz"], df_sidis["pip_px"], df_sidis["pip_py"], df_sidis["pip_pz"])
        df_sidis["pT2"] = get_pt2(E_BEAM, df_sidis["e_px"], df_sidis["e_py"], df_sidis["e_pz"], df_sidis["pip_px"], df_sidis["pip_py"], df_sidis["pip_pz"])
        df_sidis["phi_h"] = get_phih(E_BEAM, df_sidis["e_px"], df_sidis["e_py"], df_sidis["e_pz"], df_sidis["pip_px"], df_sidis["pip_py"], df_sidis["pip_pz"], True)
        df_sidis["w_pip"] = np.where(df_sidis["pip_p"].notna(), 1, 0)
        df_sidis["sel_event_idx"] = df_sidis["event_idx_local"] + start_event_idx

    out_cols = ["run", "rc_event", "sel_event_idx", "e_px", "e_py", "e_pz", "e_p", "Q2", "xB", "W", "pip_px", "pip_py", "pip_pz", "pip_p", "zh", "pT2", "phi_h", "w_pip"]
    
    # Ensure all out_cols exist (fill with NaN if missing)
    for c in out_cols:
        if c not in df_sidis.columns:
            df_sidis[c] = np.nan
            
    return df_sidis[out_cols], cf_steps, len(arrs), len(df_e_top)

# ==========================
# 2. Process many files (Wrapper)
# ==========================

def process_target(files, target, sample_type="data", max_events=None, diag=True, 
                   apply_tm=False, tm_min=0.4): 
    all_rows = []       # <--- THIS WAS MISSING
    total_cutflow = None
    next_idx = 0
    
    # Get polarity from common_cuts.py
    pol = detect_polarity(files[0])

    for path in files:
        print(f"  - processing {path}")
        try:
            df, cf, n_read, n_e = process_file(
                path, 
                target, 
                sample_type, 
                max_events=max_events, 
                apply_tm=apply_tm, 
                tm_min=tm_min, 
                start_event_idx=next_idx, 
                diagnostic_mode=diag 
            )
            next_idx += n_e
            if not df.empty: 
                all_rows.append(df)
            
            if total_cutflow is None: 
                total_cutflow = {k: 0 for k in cf.keys()}
            for k in total_cutflow: 
                total_cutflow[k] += cf[k]["N"]
        except Exception as e:
            print(f"    !!! Error: {e}")
            continue

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(), total_cutflow, pol

# ==========================
# 3. Batch Main Execution
# ==========================

def main():
    t_start = time.time()
    parser = argparse.ArgumentParser(description="Non-interactive CLAS12 RG-D Analyzer")
    parser.add_argument("--target", required=True, help="LD2, CxC, Cu, Sn")
    parser.add_argument("--root-file", required=True, help="Path to ROOT file")
    parser.add_argument("--sim", action="store_true", help="Set if simulation")
    parser.add_argument("--entry-stop", type=int, help="Max events to process")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    args = parser.parse_args()

    sample_type = "sim" if args.sim else "data"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> BATCH MODE: {args.target} ({sample_type.upper()})")
    # In main() - This single line handles BOTH Data and Sim perfectly
    df_sidis, cutflow, pol = process_target(
        files=[args.root_file], 
        target=args.target, 
        sample_type=sample_type, # 'data' or 'sim' based on flags
        max_events=args.entry_stop, 
        apply_tm=args.sim,       # Only True if --sim is typed
        tm_min=0.4, 
        diag=True
    )

    if not df_sidis.empty:
        out_name = f"sidis_{args.target}_{pol}_{os.path.basename(args.root_file)}.parquet"
        df_sidis.to_parquet(out_dir / out_name)
        print(f"Successfully Saved: {out_name}")

    if cutflow:
        base = cutflow.get("nphe", 1)
        print(f"\n[FINAL CUTFLOW: {args.target}]")
        print(f"{'-'*45}\n  {'Step':<12} | {'Count':<10} | {'Survival %':<12}\n{'-'*45}")
        for k in ["pid", "status", "nphe", "p", "vz", "pcal", "dc", "sf"]:
            val = cutflow.get(k, 0)
            print(f"  {k:<12} | {val:<10} | {100*val/base if base>0 else 0:>10.2f}%")
        print(f"{'-'*45}\n")

    print(f"Done in {time.time()-t_start:.1f}s")

if __name__ == "__main__":
    main()