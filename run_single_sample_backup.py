#!/usr/bin/env python3
"""
run_single_sample.py

High-level driver to analyze ONE sample (data or simulation)
for one or more targets (LD2, CxC, Cu, Sn, All).

Optimized: Uses vectorized Pandas merging instead of row-loops for pairing.
"""

import os
import glob
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import uproot

from physics_constants import E_BEAM
from common_cuts import 
from physics import (
    get_Q2, get_W, get_y, get_xB, get_nu,
    get_zh, get_pt2, get_phih, get_theta, get_phi,
)
from electron_cuts import electron_cutflow
from bank_builders import REC_BRANCHES, MC_BRANCHES, build_per_particle_arrays
from path_utils import ask_files_for_target, enable_path_completion
from truth_matching import add_truth_matching, ask_truth_match_options
from pids import PID

FORWARD_STATUS_MIN = 2000
FORWARD_STATUS_MAX = 4000

def forward_status_mask(status):
    status = np.asarray(status)
    abs_status = np.abs(status)
    return (abs_status >= FORWARD_STATUS_MIN) & (abs_status < FORWARD_STATUS_MAX)


# ==========================
# 3. Process one ROOT file
# ==========================

def process_file(path, target, sample_type="data",
                 max_events=None, apply_tm=False, tm_min=None,
                 start_event_idx=0, diagnostic_mode=False):
    """
    Process a single ROOT file.
    """
    # Decide which branches to read
    branch_list = list(REC_BRANCHES)
    if sample_type == "sim" and apply_tm:
        branch_list = REC_BRANCHES + MC_BRANCHES

    with uproot.open(path) as f:
        # Check available tree names if standard fails
        if "data" in f:
            tree = f["data"]
        elif "rec" in f:
            tree = f["rec"]
        else:
            # Fallback for some DSTs
            tree = f[f.keys()[0]]
            
        arrs = tree.arrays(branch_list, library="ak", entry_stop=max_events)

    n_events_read = len(arrs["REC_Particle_pid"])
    
    # 1. Build per-particle DataFrame (all REC tracks)
    df_all = build_per_particle_arrays(arrs, target_group=target)

    # 2. (SIM) attach truth matching & quality cut
    if sample_type == "sim" and apply_tm:
        df_all = add_truth_matching(df_all, arrs, quality_min=tm_min)

    # 3. Step-by-step electron selection
    final_mask, cf_steps, masks = electron_cutflow(
        df_all,
        target=target,
        polarity="OB",
        target_group_for_edge=None,
        use_avg_sf=False,
        verbose=False,
    )

    # 4. Filter Electrons
    df_e_all = df_all[final_mask].copy()

    # 5. Compute Electron Kinematics (Vectorized)
    px = df_e_all["px"].to_numpy()
    py = df_e_all["py"].to_numpy()
    pz = df_e_all["pz"].to_numpy()

    df_e_all["Q2"] = get_Q2(E_BEAM, px, py, pz)
    df_e_all["W"]  = get_W(E_BEAM, px, py, pz)
    df_e_all["y"]  = get_y(E_BEAM, px, py, pz)
    df_e_all["xB"] = get_xB(E_BEAM, px, py, pz)
    df_e_all["nu"] = get_nu(E_BEAM, px, py, pz)
    df_e_all["theta"] = get_theta(px, py, pz, degrees=True)
    df_e_all["phi"]   = get_phi(px, py, degrees=True)

    # 6. Filter Pions (PID 211 + Forward)
    # Note: We use the raw 'df_all' to find pions, not the electron subset
    forward_mask = forward_status_mask(df_all["status"].to_numpy())
    pip_mask = (df_all["pid"] == PID.PION_PLUS) & forward_mask
    df_pip_all = df_all[pip_mask].copy()

    # Calculate Pion Kinematics (Vectorized)
    df_pip_all["theta"] = get_theta(df_pip_all["px"], df_pip_all["py"], df_pip_all["pz"], degrees=True)
    df_pip_all["phi"]   = get_phi(df_pip_all["px"], df_pip_all["py"], degrees=True)

    # 7. Cutflow Counts
    # BASE is 'nphe' step
    N_all    = int(len(df_all))
    N_nphe   = int(cf_steps["nphe"]["N"]) # Base
    cutflow_out = {
        "N_all":    N_all,
        "N_pid":    int(cf_steps["pid"]["N"]),
        "N_status": int(cf_steps["status"]["N"]),
        "N_nphe":   N_nphe,
        "N_p":      int(cf_steps["p"]["N"]),
        "N_vz":     int(cf_steps["vz"]["N"]),
        "N_pcal":   int(cf_steps["pcal"]["N"]),
        "N_dc":     int(cf_steps["dc"]["N"]),
        "N_sf":     int(cf_steps["sf"]["N"]),
        "N_final":  int(cf_steps["final"]["N"]),
        "N_base":   N_nphe,
    }

    if df_e_all.empty:
        return pd.DataFrame(), cutflow_out, n_events_read, 0

    # 8. Select Top Electron per Event
    # Sort by Momentum (Highest P first)
    df_e_sorted = df_e_all.sort_values(["event_idx_local", "p"], ascending=[True, False])
    # Keep top 1
    df_e_top = df_e_sorted.groupby("event_idx_local", as_index=False).head(1)

    # 9. Pairing Strategy: Left Merge
    # We want to keep ALL electrons, even if they have no pion (inclusive DIS).
    # We join Pions to Electrons based on 'event_idx_local'.
    
    # Prepare columns to rename to avoid collision
    e_cols = ["event_idx_local", "run", "event_id", "px", "py", "pz", "p", "theta", "phi", "Q2", "xB", "y", "W", "nu"]
    # Add helicity if it exists
    if "REC_Event_helicity" in arrs.fields:
         # Note: bank_builders doesn't always put helicity in df_all unless requested,
         # but usually run/event_id are there. 
         # Assuming helicity logic is handled in bank_builders or via a join.
         # For now, let's proceed with what's in df.
         pass
         
    pip_cols = ["event_idx_local", "px", "py", "pz", "p", "theta", "phi"]

    # Subset DF for cleaner merge
    df_e_merge = df_e_top.copy()
    df_pip_merge = df_pip_all[pip_cols].copy()

    # Rename columns for clarity (e_ vs pip_)
    df_e_merge = df_e_merge.rename(columns={
        "px": "e_px", "py": "e_py", "pz": "e_pz", "p": "e_p", 
        "theta": "e_theta", "phi": "e_phi",
        "event_id": "rc_event"
    })
    
    df_pip_merge = df_pip_merge.rename(columns={
        "px": "pip_px", "py": "pip_py", "pz": "pip_pz", "p": "pip_p",
        "theta": "pip_theta", "phi": "pip_phi"
    })

    # PERFORM THE MERGE
    # Left join ensures we keep the electron even if no pion matches
    df_sidis = pd.merge(df_e_merge, df_pip_merge, on="event_idx_local", how="left")

    # 10. Calculate Hadron Physics (Vectorized)
    # Rows with no pion will result in NaN, which is exactly what we want.
    
    df_sidis["zh"] = get_zh(E_BEAM, 
                            df_sidis["e_px"], df_sidis["e_py"], df_sidis["e_pz"],
                            df_sidis["pip_px"], df_sidis["pip_py"], df_sidis["pip_pz"])
    
    df_sidis["pT2"] = get_pt2(E_BEAM, 
                              df_sidis["e_px"], df_sidis["e_py"], df_sidis["e_pz"],
                              df_sidis["pip_px"], df_sidis["pip_py"], df_sidis["pip_pz"])
    
    df_sidis["phi_h"] = get_phih(E_BEAM, 
                                 df_sidis["e_px"], df_sidis["e_py"], df_sidis["e_pz"],
                                 df_sidis["pip_px"], df_sidis["pip_py"], df_sidis["pip_pz"],
                                 degrees=True)

    # 11. Add Weights and Metadata
    # Adjust event index to be global for this process run
    df_sidis["sel_event_idx"] = df_sidis["event_idx_local"] + start_event_idx
    df_sidis["ttree_entry_idx"] = df_sidis["event_idx_local"] # same in single file context
    
    # Weights: Electron is always 1 (since we filtered for it). 
    # Pion is 1 if it exists (not NaN), else 0.
    df_sidis["w_e"] = 1
    df_sidis["w_pip"] = np.where(df_sidis["pip_p"].notna(), 1, 0)
    
    # Add helicity placeholder if missing (usually handled in bank_builders)
    if "helicity" not in df_sidis.columns:
        df_sidis["helicity"] = 0 

    # 12. Final Column Selection
    out_cols = [
        "run", "sel_event_idx", "ttree_entry_idx", "rc_event", "helicity",
        "e_px", "e_py", "e_pz", "e_p", "e_theta", "e_phi",
        "Q2", "xB", "y", "W", "nu",
        "pip_px", "pip_py", "pip_pz", "pip_p", "pip_theta", "pip_phi",
        "zh", "pT2", "phi_h", "w_e", "w_pip"
    ]
    
    # Ensure all columns exist (fill NaNs where appropriate)
    for c in out_cols:
        if c not in df_sidis.columns:
            df_sidis[c] = np.nan

    return df_sidis[out_cols], cutflow_out, n_events_read, len(df_e_top)

# ==========================
# 4. Process many files (Wrapper)
# ==========================

def process_target(files, target, sample_type="data",
                   max_events=None, apply_tm=False, tm_min=None):
    all_rows = []
    total_cutflow = None
    events_used   = 0 
    next_event_idx = 0

    pol = detect_polarity(files[0])

    for i, path in enumerate(files):
        if max_events is not None:
            remaining = max_events - events_used
            if remaining <= 0: break
            entry_stop = remaining
        else:
            entry_stop = None

        print(f"  - processing {path}")
        try:
            df_rows_file, cutflow_file, n_ev_file, n_events_with_e = process_file(
                path, target, sample_type, entry_stop, apply_tm, tm_min, next_event_idx
            )
        except Exception as e:
            print(f"    !!! Error processing file {path}: {e}")
            continue

        if max_events is not None:
            events_used += n_ev_file

        next_event_idx += n_events_with_e

        # Optimization: Don't append empty DFs
        if not df_rows_file.empty:
            all_rows.append(df_rows_file)

        if total_cutflow is None:
            total_cutflow = {k: 0 for k in cutflow_file.keys()}

        for k, v in cutflow_file.items():
            total_cutflow[k] += v

    if all_rows:
        df_all_rows = pd.concat(all_rows, ignore_index=True)
    else:
        df_all_rows = pd.DataFrame()
        if total_cutflow is None: total_cutflow = {}

    return df_all_rows, total_cutflow, pol

# ==========================
# 5. Interactive UI (Unchanged logic, just simplified)
# ==========================

def ask_sample_type():
    while True:
        ans = input("Sample type? [data/sim]: ").strip().lower()
        if ans in ("data", "sim"): return ans

def ask_targets():
    print("\nTargets: 1) LD2, 2) CxC, 3) Cu, 4) Sn, 5) All")
    ans = input("Choose target(s) [1-5]: ").strip()
    mapping = {"1":["LD2"], "2":["CxC"], "3":["Cu"], "4":["Sn"], "5":["LD2", "CxC", "Cu", "Sn"]}
    return mapping.get(ans, ["LD2"])

def ask_event_limit():
    print("\nEvent mode: 1) Full, 2) Test (5k), 3) Custom")
    ans = input("Choose: ").strip()
    if ans == "2": return 5000
    if ans == "3": return int(input("Enter total events: "))
    return None

def ask_output_dir():
    enable_path_completion()
    txt = input("\nOutput dir [default: cwd]: ").strip()
    return Path(txt).expanduser() if txt else Path(".")

def main():
    t_start = time.time()
    print("=== Optimized Single-sample RG-D Analysis ===")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="store_true")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--target")
    parser.add_argument("--root-file")
    parser.add_argument("--entry-stop", type=int)
    parser.add_argument("--out-dir")
    args, _ = parser.parse_known_args()

    non_interactive = args.root_file is not None

    if non_interactive:
        sample_type = "data" if args.data else "sim"
        targets = [args.target]
        max_events_total = args.entry_stop
        out_dir = Path(args.out_dir) if args.out_dir else Path(".")
        files_by_target = {args.target: [args.root_file]}
        apply_tm, tm_min = False, None
    else:
        sample_type = ask_sample_type()
        targets     = ask_targets()
        max_events_total = ask_event_limit()
        out_dir = None
        files_by_target = {}
        apply_tm, tm_min = ask_truth_match_options(sample_type)

    cusn_files = None

    for target in targets:
        if target in files_by_target:
            files = files_by_target[target]
        else:
            if sample_type == "data" and target in ("Cu", "Sn"):
                if cusn_files is None: cusn_files = ask_files_for_target("CuSn")
                files = cusn_files
            else:
                files = ask_files_for_target(target)

        if not files: continue
        if out_dir is None:
            out_dir = ask_output_dir()
            out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Processing {target} ===")
        df_sidis, cutflow, pol = process_target(
            files, target, sample_type, max_events_total, apply_tm, tm_min
        )

        if df_sidis.empty:
            print("  No good electrons found.")
            continue

        src_tag = os.path.splitext(os.path.basename(files[0]))[0]
        out_name = f"sidis_{target}_{sample_type}_{pol}_{src_tag}.parquet"
        df_sidis.to_parquet(out_dir / out_name)
        print(f"  -> Saved to {out_dir / out_name}")

        # Simple Cutflow Print
        if cutflow:
            base = cutflow.get("N_nphe", 1)
            print(f"\n  Final Cutflow (Base: Nphe={base})")
            for k in ["N_all", "N_pid", "N_nphe", "N_vz", "N_pcal", "N_dc", "N_sf", "N_final"]:
                val = cutflow.get(k, 0)
                print(f"  {k:<10}: {val:8d} ({100*val/base if base>0 else 0:.1f}%)")

    print(f"\nDone. {time.time()-t_start:.1f}s")

if __name__ == "__main__":
    main()