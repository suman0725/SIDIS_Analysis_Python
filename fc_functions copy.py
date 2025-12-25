import awkward as ak
import numpy as np

from physics import get_p  # uses get_p(px, py, pz)

# Layer codes in REC_Calorimeter for detector 7 (ECAL)
PCAL_LAYER = 1
ECIN_LAYER = 4
ECOUT_LAYER = 7

def get_sf(arrs):
    """
    Compute sampling fraction SF = E_tot / p for each REC_Particle.

    E_tot:
      sum of REC_Calorimeter_energy over all hits with
      - REC_Calorimeter_detector == 7  (ECAL)
      - REC_Calorimeter_layer in {1 (PCAL), 4 (ECIN), 7 (ECOUT)}

    Returns
    -------
    SF_flat : 1D numpy array
        Sampling fraction for each flattened REC_Particle entry,
        aligned with flattened REC_Particle_pid / px / py / pz.
    """
    # particle kinematics (awkward, jagged)
    px = arrs["REC_Particle_px"]
    py = arrs["REC_Particle_py"]
    pz = arrs["REC_Particle_pz"]

    # calorimeter info
    calo_pindex = arrs["REC_Calorimeter_pindex"]
    calo_det    = arrs["REC_Calorimeter_detector"]
    calo_layer  = arrs["REC_Calorimeter_layer"]
    calo_energy = arrs["REC_Calorimeter_energy"]

    ETOT_list = []

    # loop over events
    for px_ev, py_ev, pz_ev, pidx_ev, det_ev, layer_ev, e_ev in zip(
        px, py, pz,
        calo_pindex, calo_det, calo_layer, calo_energy
    ):
        n_part = len(px_ev)
        # total ECAL energy per particle in this event
        ETOT_ev = np.zeros(n_part, dtype=np.float32)

        for pidx, det, layer, e in zip(pidx_ev, det_ev, layer_ev, e_ev):
            if pidx < 0 or pidx >= n_part:
                continue
            if det != 7:
                continue  # only ECAL detector
            if layer not in (PCAL_LAYER, ECIN_LAYER, ECOUT_LAYER):
                continue  # only PCAL + ECIN + ECOUT

            ETOT_ev[pidx] += e

        ETOT_list.append(ETOT_ev)

    # flatten ETOT to 1D, aligned with flattened particles
    ETOT_flat = ak.to_numpy(ak.flatten(ak.Array(ETOT_list)))

    # compute p from px,py,pz and flatten
    px_flat = ak.to_numpy(ak.flatten(px))
    py_flat = ak.to_numpy(ak.flatten(py))
    pz_flat = ak.to_numpy(ak.flatten(pz))

    p_flat = get_p(px_flat, py_flat, pz_flat)

    # sampling fraction: SF = E_tot / p
    with np.errstate(divide="ignore", invalid="ignore"):
        SF_flat = np.where(p_flat > 0, ETOT_flat / p_flat, np.nan)

    return SF_flat

