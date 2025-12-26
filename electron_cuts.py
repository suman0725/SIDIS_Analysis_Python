import numpy as np
from params_electron_sf_outbending import SF_PARAMS_OB 

# ---------- 1) Vz Windows (RG-D Outbending) ----------
VZ_WINDOWS_OB = {
    "LD2": (-15.0,  5.0),
    "CxC": (-10.6,  5.0),
    "Cu":  (-10.6, -6.5),
    "Sn":  (-5.5,   5.0),
}

# ---------- 2) DC Edge Cuts ----------
EDGE_CUTS_OB = {
    "LD2": {1: 1.68, 2: 2.00, 3: 8.75},
    "CxC": {1: 1.70, 2: 2.02, 3: 8.92},
    "CuSn": {1: 1.69, 2: 2.00, 3: 8.89},
}

# ---------- 3) Helper Functions ----------
def sf_mean(p, coeffs):
    a, b, c, d = coeffs
    return a + b*p + c*p*p + d*p*p*p

def sf_sigma(p, coeffs):
    a, b, c, d = coeffs
    return a + b*p + c*p*p + d*p*p*p

def sf_cut_mask(p, sf, sector, target, polarity="OB"):
    target_key = "CuSn" if target in ["Cu", "Sn"] else target
    params_dict = SF_PARAMS_OB
    
    p, sf, sector = np.asarray(p), np.asarray(sf), np.asarray(sector)
    mask = np.zeros(len(p), dtype=bool)

    for s in range(1, 7):
        idx = (sector == s)
        if not np.any(idx): continue
        pars = params_dict[target_key][s]
        mu  = sf_mean(p[idx], pars["mu"])
        sig = sf_sigma(p[idx], pars["sigma"])
        mask[idx] = (sf[idx] >= mu - 3*sig) & (sf[idx] <= mu + 3*sig)
    return mask

def vz_mask(vz, target, polarity="OB"):
    vz_min, vz_max = VZ_WINDOWS_OB[target]
    return (vz >= vz_min) & (vz <= vz_max)

def dc_edge_mask(edge, region, target_group, polarity="OB"):
    mask = np.zeros(len(edge), dtype=bool)
    for r in (1, 2, 3):
        idx = (region == r)
        if not np.any(idx): continue
        mask[idx] = (edge[idx] > EDGE_CUTS_OB[target_group][r])
    return mask

def calo_fid_mask(v, w):
    return (v > 9.0) & (w > 9.0)

def pcal_energy_cut(epcal, e_min=0.06):
    return epcal > e_min

# ---------- 4) The Master Cutflow Function ----------

def electron_cutflow(df, target, polarity="OB", sample_type="data"):
    """
    Final RG-D Electron Cutflow Logic.
    """
    # Extraction (Ensures variables are available for logic)
    pid    = df["pid"].to_numpy()
    status = df["status"].to_numpy()
    vz     = df["vz"].to_numpy()
    Nphe   = df["Nphe"].to_numpy()
    p      = df["p"].to_numpy()
    sf     = df["sf"].to_numpy()
    sector = df["sector"].to_numpy()
    v_pcal = df["v_pcal"].to_numpy()
    w_pcal = df["w_pcal"].to_numpy()
    E_pcal = df["E_pcal"].to_numpy()
    edge   = df["edge"].to_numpy()
    region = df["region"].to_numpy()

    target_group = "CuSn" if target in ["Cu", "Sn"] else target

    masks = {}
    
    # --- STEP 1: THE BASE ---
    # Reference point: All FD Electron candidates
    masks["base"] = (pid == 11) & (status > -4000) & (status <= -2000)
    N_base = int(np.sum(masks["base"]))

    # --- STEP 2: SEQUENTIAL QUALITY ---
    if sample_type == "data" and target in ("Cu", "Sn"):
        # Experimental solid targets: Separate Vz peak first
        masks["vz"]   = masks["base"] & vz_mask(vz, target, polarity)
        masks["nphe"] = masks["vz"] & (Nphe > 2)
        prev = masks["nphe"]
    else:
        # LD2/CxC/Sim: Check HTCC first
        masks["nphe"] = masks["base"] & (Nphe > 2)
        masks["vz"]   = masks["nphe"] & vz_mask(vz, target, polarity)
        prev = masks["vz"]

    # --- STEP 3: PHYSICS ---
    masks["p"]    = prev & (p > 0.8)
    masks["pcal"] = masks["p"] & pcal_energy_cut(E_pcal) & calo_fid_mask(v_pcal, w_pcal)
    masks["dc"]   = masks["pcal"] & dc_edge_mask(edge, region, target_group, polarity)
    
    sf_ok = sf_cut_mask(p, sf, sector, target, polarity)
    masks["final"] = masks["dc"] & sf_ok

    # --- STEP 4: CUTFLOW GENERATION ---
    # These strings MUST match the keys in the 'masks' dictionary above
    order = ["base", "vz", "nphe", "p", "pcal", "dc", "final"]
    
    cutflow = {}
    for step in order:
        N = int(np.sum(masks[step]))
        eff = 100.0 * N / N_base if N_base > 0 else 0.0
        cutflow[step] = {"N": N, "eff_base": eff}

    return masks["final"], cutflow, masks