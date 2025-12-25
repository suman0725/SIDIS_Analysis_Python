"""
bank_builders.py (Vectorized)

High-performance replacement for constructing flat DataFrames from CLAS12 ROOT files.
Replaces slow Python loops with NumPy global indexing.
"""

import numpy as np
import pandas as pd
import awkward as ak

from fc_functions import get_sf
from physics import get_p, get_phi, get_sector

# ---------------------------
# 1) Branch list
# ---------------------------
REC_BRANCHES = [
    # Event info
    "RUN_config_run",
    "RUN_config_event",
    "REC_Event_helicity",
    # Particle info
    "REC_Particle_pid",
    "REC_Particle_px",
    "REC_Particle_py",
    "REC_Particle_pz",
    "REC_Particle_status",
    "REC_Particle_vz",
    "REC_Particle_vx",
    "REC_Particle_vy",
    "REC_Particle_beta",
    "REC_Particle_chi2pid",
    # Cherenkov
    "REC_Cherenkov_pindex",
    "REC_Cherenkov_nphe",
    "REC_Cherenkov_detector",
    # Calorimeter
    "REC_Calorimeter_pindex",
    "REC_Calorimeter_detector",
    "REC_Calorimeter_layer",
    "REC_Calorimeter_energy",
    "REC_Calorimeter_lv",
    "REC_Calorimeter_lw",
    # DC Trajectory
    "REC_Traj_detector",
    "REC_Traj_layer",
    "REC_Traj_pindex",
    "REC_Traj_edge",
    "REC_Traj_x",
    "REC_Traj_y",
]

MC_BRANCHES = [
    "MC_Particle_pid",
    "MC_RecMatch_pindex",
    "MC_RecMatch_mcindex",
    "MC_RecMatch_quality",
]

# ---------------------------
# 2) Helper: Map Hits to Particles
# ---------------------------
def map_hits_to_particles_vectorized(arrs, bank_prefix, hit_mask, value_branch, total_particles, particle_offsets):
    """
    Vectorized map of satellite bank values (Traj, Calo, Cher) to main Particle bank.
    """
    pindex_jagged = arrs[f"{bank_prefix}_pindex"]
    value_jagged  = arrs[f"{bank_prefix}_{value_branch}"]
    
    # Get Event Index for every hit
    # axis=0 gives row (event) index, broadcast ensures shape matches pindex
    event_indices = ak.flatten(
        ak.broadcast_arrays(ak.local_index(pindex_jagged, axis=0), pindex_jagged)[0][hit_mask]
    ).to_numpy()
    
    flat_pindex = ak.flatten(pindex_jagged[hit_mask]).to_numpy()
    flat_value  = ak.flatten(value_jagged[hit_mask]).to_numpy()
    
    # Global Index = (Start Index of this Event) + (pindex of the hit)
    global_indices = particle_offsets[event_indices] + flat_pindex
    
    # Safety check
    valid_idx = (global_indices >= 0) & (global_indices < total_particles)
    
    # Fill result array (Default is 0.0)
    result = np.zeros(total_particles, dtype=np.float32)
    result[global_indices[valid_idx]] = flat_value[valid_idx]
    
    return result

# ---------------------------
# 3) Main Builder
# ---------------------------

def build_per_particle_arrays(arrs, target_group="LD2"):
    """
    Vectorized construction of the flat DataFrame.
    """
    # --- A. Setup Backbone (REC::Particle) ---
    pid_jagged = arrs["REC_Particle_pid"]
    counts = ak.num(pid_jagged)
    total_particles = np.sum(counts)
    
    # Offset array: Index where event i starts
    offsets = np.concatenate(([0], np.cumsum(counts.to_numpy())[:-1]))
    
    # Flatten Basic Kinematics
    pid    = ak.flatten(pid_jagged).to_numpy()
    px     = ak.flatten(arrs["REC_Particle_px"]).to_numpy()
    py     = ak.flatten(arrs["REC_Particle_py"]).to_numpy()
    pz     = ak.flatten(arrs["REC_Particle_pz"]).to_numpy()
    status = ak.flatten(arrs["REC_Particle_status"]).to_numpy()
    vz     = ak.flatten(arrs["REC_Particle_vz"]).to_numpy()
    
    # Expand Event Info
    run_num = np.repeat(arrs["RUN_config_run"].to_numpy(), counts.to_numpy())
    # hel     = np.repeat(arrs["REC_Event_helicity"].to_numpy(), counts.to_numpy())
    
    # --- B. Map HTCC (Nphe) ---
    cher_det = arrs["REC_Cherenkov_detector"]
    mask_htcc = (cher_det == 15)
    nphe = map_hits_to_particles_vectorized(
        arrs, "REC_Cherenkov", mask_htcc, "nphe", total_particles, offsets
    )

    # --- C. Map PCAL (Det 7, Layer 1) ---
    calo_det = arrs["REC_Calorimeter_detector"]
    calo_lay = arrs["REC_Calorimeter_layer"]
    mask_pcal = (calo_det == 7) & (calo_lay == 1)
    
    v_pcal = map_hits_to_particles_vectorized(arrs, "REC_Calorimeter", mask_pcal, "lv", total_particles, offsets)
    w_pcal = map_hits_to_particles_vectorized(arrs, "REC_Calorimeter", mask_pcal, "lw", total_particles, offsets)
    e_pcal = map_hits_to_particles_vectorized(arrs, "REC_Calorimeter", mask_pcal, "energy", total_particles, offsets)
    
    pcal_fid = (v_pcal > 9.0) & (w_pcal > 9.0)
    pcal_E_ok = (e_pcal > 0.06)

    # --- D. Map DC (Det 6) Regions ---
    traj_det = arrs["REC_Traj_detector"]
    traj_lay = arrs["REC_Traj_layer"]
    
    mask_dc = (traj_det == 6)
    mask_r1 = mask_dc & (traj_lay >= 6)  & (traj_lay <= 12)
    mask_r2 = mask_dc & (traj_lay >= 18) & (traj_lay <= 24)
    mask_r3 = mask_dc & (traj_lay >= 30) & (traj_lay <= 36)
    
    edge_r1 = map_hits_to_particles_vectorized(arrs, "REC_Traj", mask_r1, "edge", total_particles, offsets)
    edge_r2 = map_hits_to_particles_vectorized(arrs, "REC_Traj", mask_r2, "edge", total_particles, offsets)
    edge_r3 = map_hits_to_particles_vectorized(arrs, "REC_Traj", mask_r3, "edge", total_particles, offsets)
    
    x_r1 = map_hits_to_particles_vectorized(arrs, "REC_Traj", mask_r1, "x", total_particles, offsets)
    y_r1 = map_hits_to_particles_vectorized(arrs, "REC_Traj", mask_r1, "y", total_particles, offsets)
    
    x_r2 = map_hits_to_particles_vectorized(arrs, "REC_Traj", mask_r2, "x", total_particles, offsets)
    y_r2 = map_hits_to_particles_vectorized(arrs, "REC_Traj", mask_r2, "y", total_particles, offsets)
    
    x_r3 = map_hits_to_particles_vectorized(arrs, "REC_Traj", mask_r3, "x", total_particles, offsets)
    y_r3 = map_hits_to_particles_vectorized(arrs, "REC_Traj", mask_r3, "y", total_particles, offsets)

    # Logic: Prioritize R3 -> R2 -> R1 for the single 'edge' column
    final_edge = np.zeros(total_particles, dtype=np.float32)
    final_region = np.zeros(total_particles, dtype=np.int32)
    
    has_r3 = (edge_r3 != 0)
    final_edge[has_r3] = edge_r3[has_r3]
    final_region[has_r3] = 3
    
    has_r2 = (edge_r2 != 0)
    final_edge[has_r2] = edge_r2[has_r2]
    final_region[has_r2] = 2
    
    has_r1 = (edge_r1 != 0)
    final_edge[has_r1] = edge_r1[has_r1]
    final_region[has_r1] = 1

    # --- E. Sampling Fraction ---
    sf_flat = get_sf(arrs)

    # --- F. Calculate Kinematics ---
    p_flat = get_p(px, py, pz)
    phi_flat = get_phi(px, py, degrees=True)
    sec_flat = get_sector(phi_flat)

    # --- G. Build DataFrame ---
    df = pd.DataFrame({
        "event_id": np.repeat(arrs["RUN_config_event"].to_numpy(), counts.to_numpy()),
        "run":      run_num,
        "event_idx_local": np.repeat(np.arange(len(counts)), counts),
        
        "pid":    pid,
        "px":     px, "py": py, "pz": pz,
        "status": status,
        "vz":     vz,
        
        "p":      p_flat,
        "phi":    phi_flat,
        "sector": sec_flat,
        "sf":     sf_flat,
        "Nphe":   nphe,
        
        # PCAL
        "v_pcal": v_pcal, "w_pcal": w_pcal, "E_pcal": e_pcal,
        "pcal_E_ok": pcal_E_ok, "pcal_fid":  pcal_fid,
        
        # DC
        "edge":   final_edge,
        "region": final_region,
        
        "dc_edge_r1": edge_r1, "dc_edge_r2": edge_r2, "dc_edge_r3": edge_r3,
        "dc_x_r1": x_r1, "dc_y_r1": y_r1,
        "dc_x_r2": x_r2, "dc_y_r2": y_r2,
        "dc_x_r3": x_r3, "dc_y_r3": y_r3,
    })

    return df