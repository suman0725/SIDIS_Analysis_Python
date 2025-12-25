# rgd_cuts/electron.py

import numpy as np

# import the SF parameter dictionaries you defined

from params_electron_sf_outbending import SF_PARAMS_OB  # OB table

# ---------- 1) Simple constants from the note ----------

# Vz windows (outbending only, for now)
VZ_WINDOWS_OB = {
    "LD2": (-15.0,  5.0),
    "CxC": (-10.6,  5.0),
    "Cu":  (-10.6, -6.5),
    "Sn":  (-5.5,   5.0),
}

# DC edge cuts per region (outbending)
EDGE_CUTS_OB = {
    "LD2": {1: 1.68, 2: 2.00, 3: 8.75},
    "CxC": {1: 1.70, 2: 2.02, 3: 8.92},
    "CuSn": {1: 1.69, 2: 2.00, 3: 8.89},
}


# ---------- 2) SF helper functions (OB+IB) ----------

def sf_mean(p, coeffs):
    a, b, c, d = coeffs
    return a + b*p + c*p*p + d*p*p*p

def sf_sigma(p, coeffs):
    a, b, c, d = coeffs
    return a + b*p + c*p*p + d*p*p*p


def sf_cut_mask(p, sf, sector, target, polarity="OB", use_avg=False):
    """
    Vectorized SF cut.

    p, sf   : numpy arrays of same length
    sector  : numpy array of sector numbers (1..6) for each track
    target  : "LD2", "CxC", "Cu", "Sn"
    polarity: "OB" or "IB"
    """

    if polarity == "OB":
        params_dict = SF_PARAMS_OB
    elif polarity == "IB":
        params_dict = SF_PARAMS_IB
    else:
        raise ValueError(f"Unknown polarity {polarity}")

    # map Cu/Sn to CuSn entry in tables
    if use_avg:
        target_key = "Average"
    else:
        if target in ["Cu", "Sn"]:
            target_key = "CuSn"
        else:
            target_key = target

    p = np.asarray(p)
    sf = np.asarray(sf)
    sector = np.asarray(sector)

    mask = np.zeros(len(p), dtype=bool)

    for s in range(1, 7):
        idx = (sector == s)
        if not np.any(idx):
            continue

        pars = params_dict[target_key][s]
        mu  = sf_mean(p[idx],  pars["mu"])
        sig = sf_sigma(p[idx], pars["sigma"])

        mask[idx] = (sf[idx] >= mu - 3*sig) & (sf[idx] <= mu + 3*sig)

    return mask



def sf_bounds_for_plot(p, target="LD2", polarity="OB", sector=1, n_sigma=3):
    """
    For a 1D array p, return (sf_low, sf_high) using the same
    mu(p), sigma(p) polynomials as the SF cut, for one sector.
    Used only to draw the two lines on the SF vs p plot.
    """
    if polarity != "OB":
        raise NotImplementedError("Just OB here; add SF_PARAMS_IB if needed")

    params_dict = SF_PARAMS_OB

    # map Cu/Sn to CuSn like in sf_cut_mask
    if target in ["Cu", "Sn"]:
        target_key = "CuSn"
    else:
        target_key = target

    p = np.asarray(p)
    pars = params_dict[target_key][sector]

    mu   = sf_mean(p,  pars["mu"])
    sig  = sf_sigma(p, pars["sigma"])

    sf_low  = mu - n_sigma * sig
    sf_high = mu + n_sigma * sig

    return sf_low, sf_high



# ---------- 3) Vertex window mask ----------

def vz_mask(vz, target, polarity="OB"):
    if polarity != "OB":
        # you can add IB windows later if needed
        raise NotImplementedError("Only outbending Vz implemented.")
    vz_min, vz_max = VZ_WINDOWS_OB[target]
    vz = np.asarray(vz)
    return (vz >= vz_min) & (vz <= vz_max)


# ---------- 4) DC fiducial mask ----------

def dc_edge_mask(edge, region, target_group="LD2", polarity="OB"):
    """
    target_group: "LD2", "CxC", or "CuSn"
    region      : numpy array of DC region numbers (1, 2, or 3)
    edge        : numpy array of REC::Traj.edge values
    """
    if polarity != "OB":
        raise NotImplementedError("Only outbending DC cuts implemented.")

    edge = np.asarray(edge)
    region = np.asarray(region)

    mask = np.zeros(len(edge), dtype=bool)

    for r in (1, 2, 3):
        idx = (region == r)
        if not np.any(idx):
            continue
        cut_val = EDGE_CUTS_OB[target_group][r]
        mask[idx] = (edge[idx] > cut_val)

    return mask


# ---------- 5) Calorimeter fiducial mask and energy cuts ----------

def calo_fid_mask(v, w):
    v = np.asarray(v)
    w = np.asarray(w)
    return (v > 9.0) & (w > 9.0)


def pcal_energy_cut(epcal, e_min=0.06):
    """
    Simple PCAL energy cut on a per-track E_pcal variable.
    epcal is the reconstructed PCAL energy for that track.
    """
    epcal = np.asarray(epcal)
    return epcal > e_min


# ---------- 6) Full electron mask ----------

def electron_mask(df, target, polarity="OB", target_group_for_edge=None, use_avg_sf=False):
    """
    df: pandas DataFrame with columns:
        pid, status, p, sf, vz, v_pcal, w_pcal, edge, sector, region, Nphe
    target: "LD2", "CxC", "Cu", "Sn"
    polarity: "OB" or "IB"
    """

    if target_group_for_edge is None:
        target_group_for_edge = "CuSn" if target in ["Cu", "Sn"] else target

    base = (
        (df["pid"] == 11) &
        (df["status"] > -4000) & (df["status"] <= -2000) &
        (df["p"] > 0.8) &
        (df["Nphe"] > 2)
    )

    sfmask = sf_cut_mask(
        p=df["p"].to_numpy(),
        sf=df["sf"].to_numpy(),
        sector=df["sector"].to_numpy(),
        target=target,
        polarity=polarity,
        use_avg=use_avg_sf,
    )

    vzmask  = vz_mask(df["vz"].to_numpy(), target, polarity=polarity)
    calofid = calo_fid_mask(df["v_pcal"].to_numpy(), df["w_pcal"].to_numpy())
    epcalmask = pcal_energy_cut(df["E_pcal"].to_numpy(), e_min=0.06)
    
    dcfid   = dc_edge_mask(df["edge"].to_numpy(), df["region"].to_numpy(),
                           target_group_for_edge, polarity=polarity)

    return base.to_numpy() & sfmask & vzmask & calofid & dcfid & epcalmask


# ---------- 7) DIS region mask ----------

def dis_mask(Q2, W, y, Q2_min=1.0, W_min=2.0, y_max=0.85):
    """
    Basic DIS phase-space selection for the scattered electron:
      Q2 > Q2_min  (GeV^2)
      W  > W_min   (GeV)
      y  < y_max

    All inputs are numpy-like arrays of the same length.
    """
    Q2 = np.asarray(Q2)
    W  = np.asarray(W)
    y  = np.asarray(y)

    return (Q2 > Q2_min) & (W > W_min) & (y < y_max)



def electron_cutflow(df, target, polarity="OB",
                     target_group_for_edge=None,
                     use_avg_sf=False,
                     verbose=False):
    """
    Build step-by-step electron masks and cutflow numbers.

    df: pandas.DataFrame with columns:
        pid, status, p, sf, vz, v_pcal, w_pcal, E_pcal,
        edge, region, sector, Nphe

    Returns
    -------
    final_mask : 1D bool np.array  (full electron selection)
    cutflow    : dict { step_name : {"N": int, "eff_base": float} }
                 BASE is always 'nphe' (PID+status+Nphe>2), but for
                 Cu/Sn this is already inside their Vz slice.
    masks      : dict { step_name : bool array }
    """

    import numpy as np

    if target_group_for_edge is None:
        target_group_for_edge = "CuSn" if target in ["Cu", "Sn"] else target

    # pull columns as NumPy for speed
    pid     = df["pid"].to_numpy()
    status  = df["status"].to_numpy()
    p       = df["p"].to_numpy()
    sf      = df["sf"].to_numpy()
    vz      = df["vz"].to_numpy()
    Nphe    = df["Nphe"].to_numpy()
    v_pcal  = df["v_pcal"].to_numpy()
    w_pcal  = df["w_pcal"].to_numpy()
    E_pcal  = df["E_pcal"].to_numpy()
    edge    = df["edge"].to_numpy()
    region  = df["region"].to_numpy()
    sector  = df["sector"].to_numpy()

    masks = {}
    masks["all"] = np.ones_like(pid, dtype=bool)

    # --- Cu/Sn: Vz first to separate targets ---
    if target in ("Cu", "Sn"):
        # 1) Vz window defines which target you are in
        masks["vz"] = vz_mask(vz, target, polarity=polarity)

        # 2) PID == 11, but only inside this Vz slice
        masks["pid"] = masks["vz"] & (pid == 11)

        # 3) forward, negative status (FD electron)
        masks["status"] = masks["pid"] & (status > -4000) & (status <= -2000)

        # 4) Nphe > 2   (BASE for efficiencies)
        masks["nphe"] = masks["status"] & (Nphe > 2)

        # 5) p > 0.8 GeV
        masks["p"] = masks["nphe"] & (p > 0.8)

    # --- LD2 / CxC: old order (Vz after BASE) ---
    else:
        # 1) PID == 11
        masks["pid"] = (pid == 11)

        # 2) forward, negative status
        masks["status"] = masks["pid"] & (status > -4000) & (status <= -2000)

        # 3) Nphe > 2   (BASE)
        masks["nphe"] = masks["status"] & (Nphe > 2)

        # 4) p > 0.8 GeV
        masks["p"] = masks["nphe"] & (p > 0.8)

        # 5) Vz window (applied after BASE for LD2/CxC)
        masks["vz"] = masks["p"] & vz_mask(vz, target, polarity=polarity)

    # --- Common tail: PCAL, DC, SF ---
    mask_epcal    = pcal_energy_cut(E_pcal, e_min=0.06)
    mask_calo_fid = calo_fid_mask(v_pcal, w_pcal)

    # For Cu/Sn, "p" already includes Vz.
    # For LD2/CxC, use "vz" so cutflow is monotonic.
    if target in ("Cu", "Sn"):
        prev_for_pcal = masks["p"]
    else:
        prev_for_pcal = masks["vz"]

    masks["pcal"] = prev_for_pcal & mask_epcal & mask_calo_fid


    mask_dc = dc_edge_mask(edge, region,
                           target_group=target_group_for_edge,
                           polarity=polarity)
    masks["dc"] = masks["pcal"] & mask_dc

    sf_ok = sf_cut_mask(
        p=p,
        sf=sf,
        sector=sector,
        target=target,
        polarity=polarity,
        use_avg=use_avg_sf,
    )
    masks["sf"] = masks["dc"] & sf_ok

    # final electron mask
    final_mask = masks["sf"]
    masks["final"] = final_mask

    # ---------------- cutflow numbers ----------------
    # BASE = masks["nphe"] in both cases
    N_base = int(masks["nphe"].sum()) if masks["nphe"].size > 0 else 0

    # Different step order for printout
    if target in ("Cu", "Sn"):
        order = ["all", "vz", "pid", "status", "nphe", "p",
                 "pcal", "dc", "sf", "final"]
    else:
        order = ["all", "pid", "status", "nphe", "p", "vz",
                 "pcal", "dc", "sf", "final"]

    cutflow = {}
    for step in order:
        N = int(masks[step].sum())
        eff = 100.0 * N / N_base if N_base > 0 else 0.0
        cutflow[step] = {"N": N, "eff_base": eff}

    if verbose:
        print("\nElectron cutflow (target {}, pol {}):".format(target, polarity))
        print("  BASE sample = 'nphe' (PID+status+Nphe>2)")
        print("  {:8s} {:>10s} {:>12s}".format("step", "N", "% of BASE"))
        for step in order:
            N = cutflow[step]["N"]
            eff = cutflow[step]["eff_base"]
            print("  {:8s} {:10d} {:12.2f}".format(step, N, eff))

    return final_mask, cutflow, masks 
