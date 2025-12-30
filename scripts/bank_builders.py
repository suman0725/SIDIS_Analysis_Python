"""
bank_builders.py (Vectorized)

High-performance builder for CLAS12 DataFrames.
Includes internal Sampling Fraction calculation to avoid redundant array flattening.
"""

import numpy as np
import pandas as pd
import awkward as ak

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
    "REC_Particle_charge",
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
    Vectorized map of satellite bank values to main Particle bank.
    Uses scatter-add to ensure hits within the same mask are summed, not erased.
    """
    pindex_jagged = arrs[f"{bank_prefix}_pindex"]
    value_jagged  = arrs[f"{bank_prefix}_{value_branch}"]
    
    # 1. Get Event Index for every hit
    event_indices = ak.flatten(
        ak.broadcast_arrays(ak.local_index(pindex_jagged, axis=0), pindex_jagged)[0][hit_mask]
    ).to_numpy()
    
    # 2. Flatten data
    flat_pindex = ak.flatten(pindex_jagged[hit_mask]).to_numpy()
    flat_value  = ak.flatten(value_jagged[hit_mask]).to_numpy()
    
    # 3. Global Index calculation
    global_indices = particle_offsets[event_indices] + flat_pindex
    
    # 4. Create empty array
    result = np.zeros(total_particles, dtype=np.float32)
    valid_idx = (global_indices >= 0) & (global_indices < total_particles)
    
    # --- THE MODIFICATION ---
    # We use np.add.at instead of = so we don't erase data if a particle 
    # hits two sensors in the same detector layer.
    np.add.at(result, global_indices[valid_idx], flat_value[valid_idx])
    
    return result
# ---------------------------
# 3) Main Builder
# ---------------------------

def build_per_particle_arrays(arrs, target_group="LD2"):
    """
    Vectorized construction of the flat DataFrame for CLAS12 analysis.
    """
    # --- A. Setup Backbone (REC::Particle) ---
    pid_jagged = arrs["REC_Particle_pid"]
    counts = ak.num(pid_jagged)
    total_particles = np.sum(counts)
    
    # Offset array: Index where event i starts in the flat array
    offsets = np.concatenate(([0], np.cumsum(counts.to_numpy())[:-1]))
    
    # Flatten Basic Kinematics
    pid    = ak.flatten(pid_jagged).to_numpy()
    charge = ak.flatten(arrs["REC_Particle_charge"]).to_numpy()
    px     = ak.flatten(arrs["REC_Particle_px"]).to_numpy()
    py     = ak.flatten(arrs["REC_Particle_py"]).to_numpy()
    pz     = ak.flatten(arrs["REC_Particle_pz"]).to_numpy()
    status = ak.flatten(arrs["REC_Particle_status"]).to_numpy()
    vz     = ak.flatten(arrs["REC_Particle_vz"]).to_numpy()
    beta    = ak.flatten(arrs["REC_Particle_beta"]).to_numpy()  
    chi2pid = ak.flatten(arrs["REC_Particle_chi2pid"]).to_numpy()
    
    run_num = np.repeat(arrs["RUN_config_run"].to_numpy(), counts.to_numpy())

    # --- B. Calculate Kinematics (Needed for SF and Sector) ---
    p_flat = get_p(px, py, pz)
    phi_flat = get_phi(px, py, degrees=True)
    sec_flat = get_sector(phi_flat)

    # --- C. Map HTCC (Nphe) ---
    # Sums multiple hits if an electron hits two mirrors (mirror crack)
    cher_det = arrs["REC_Cherenkov_detector"]
    # HTCC (Detector 15)
    mask_htcc = (cher_det == 15)
    nphe_htcc = map_hits_to_particles_vectorized(arrs, "REC_Cherenkov", mask_htcc, "nphe", total_particles, offsets)

    # LTCC (Detector 16)
    mask_ltcc = (cher_det == 16)
    nphe_ltcc = map_hits_to_particles_vectorized(arrs, "REC_Cherenkov", mask_ltcc, "nphe", total_particles, offsets)

    # --- D. Map Calorimeter Layers (Separately) ---
    calo_det = arrs["REC_Calorimeter_detector"]
    calo_lay = arrs["REC_Calorimeter_layer"]
    
    # Define Masks for PCAL (1), ECIN (4), ECOUT (7)
    # Note: Det 7 is ECAL
    m_pcal  = (calo_det == 7) & (calo_lay == 1)
    m_ecin  = (calo_det == 7) & (calo_lay == 4)
    m_ecout = (calo_det == 7) & (calo_lay == 7)
    
    # Get Energy per layer (Keeps them separate as requested)
    e_pcal  = map_hits_to_particles_vectorized(arrs, "REC_Calorimeter", m_pcal,  "energy", total_particles, offsets)
    e_ecin  = map_hits_to_particles_vectorized(arrs, "REC_Calorimeter", m_ecin,  "energy", total_particles, offsets)
    e_ecout = map_hits_to_particles_vectorized(arrs, "REC_Calorimeter", m_ecout, "energy", total_particles, offsets)
    
    # Get PCAL coordinate for fiducial cuts
    v_pcal = map_hits_to_particles_vectorized(arrs, "REC_Calorimeter", m_pcal, "lv", total_particles, offsets)
    w_pcal = map_hits_to_particles_vectorized(arrs, "REC_Calorimeter", m_pcal, "lw", total_particles, offsets)

    # --- E. Sampling Fraction (SF) Calculation ---
    # SF = (Total Energy in Calorimeter) / Momentum
    # Since we use map_hits_to_particles_vectorized, we can just sum the flat arrays
    Etot_flat = e_pcal + e_ecin + e_ecout
    
    with np.errstate(divide='ignore', invalid='ignore'):
        sf_flat = np.where(p_flat > 0, Etot_flat / p_flat, np.nan)

    # --- F. Map DC Trajectory (Det 6) ---
    traj_det = arrs["REC_Traj_detector"]
    traj_lay = arrs["REC_Traj_layer"]
    mask_dc = (traj_det == 6)

    def map_dc(m, branch):
        return map_hits_to_particles_vectorized(arrs, "REC_Traj", m, branch, total_particles, offsets)
    
    # 1. Create separate masks for 6, 18, 36 (Verified layers)
    mask_r1 = mask_dc & (traj_lay == 6)
    mask_r2 = mask_dc & (traj_lay == 18)
    mask_r3 = mask_dc & (traj_lay == 36)
    
    # 2. Map coordinates and edges for each region specifically
    x_r1, y_r1, edge_r1 = map_dc(mask_r1, "x"), map_dc(mask_r1, "y"), map_dc(mask_r1, "edge")
    x_r2, y_r2, edge_r2 = map_dc(mask_r2, "x"), map_dc(mask_r2, "y"), map_dc(mask_r2, "edge")
    x_r3, y_r3, edge_r3 = map_dc(mask_r3, "x"), map_dc(mask_r3, "y"), map_dc(mask_r3, "edge")

    # --- G. Build Final DataFrame ---
    df = pd.DataFrame({
        # Event Info
        "event_id": np.repeat(arrs["RUN_config_event"].to_numpy(), counts.to_numpy()),
        "run":      run_num,
        "event_idx_local": np.repeat(np.arange(len(counts)), counts),
        
        # Rec Particle
        "pid":    pid,
        "charge": charge,
        "px":     px, "py": py, "pz": pz,
        "status": status,
        "vz":     vz,
        "p":      p_flat,
        "phi":    phi_flat,
        "sector": sec_flat,
        "sf":     sf_flat,
        "beta":   beta,   
        "chi2pid": chi2pid,

        # Cherenkov
        "Nphe_htcc": nphe_htcc,
        "Nphe_ltcc": nphe_ltcc,
        
        # Calorimeter
        "E_pcal":  e_pcal,
        "E_ecin":  e_ecin,
        "E_ecout": e_ecout,
        "v_pcal":  v_pcal, 
        "w_pcal":  w_pcal,
        
        # DC Trajectory (Matching names from Section F)
        "dc_x_r1": x_r1, "dc_y_r1": y_r1, "dc_edge_r1": edge_r1,
        "dc_x_r2": x_r2, "dc_y_r2": y_r2, "dc_edge_r2": edge_r2,
        "dc_x_r3": x_r3, "dc_y_r3": y_r3, "dc_edge_r3": edge_r3,
    })

    return df