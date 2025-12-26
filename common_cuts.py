import uproot  # <--- THIS IS THE MISSING PIECE
import numpy as np
def detect_polarity(path) -> str:
    """Read RUN_config_torus from the 'data' tree."""
    try:
        with uproot.open(path) as f:
            # Uproot handles 'data;24', 'data;8' etc. automatically
            tree = f["data"]
            torus = tree.arrays("RUN_config_torus", entry_stop=1, library="np")["RUN_config_torus"][0]
            return "OB" if torus > 0 else "IB"
    except Exception as e:
        raise RuntimeError(f"CRITICAL: Could not read 'data' tree or torus from {path}: {e}")


def forward_status_mask(status):
    """
    Checks if a track is in the Forward Carriage (FD).
    Bitwise check: (abs(status) // 1000) & 2 > 0 
    Matches tracks in the 2000-3999 range.
    """
    status = np.asarray(status)
    abs_status = np.abs(status)
    # The bitwise logic is very robust for CLAS12 status codes
    return (abs_status // 1000).astype(int) & 2 > 0
# Or if you prefer ranges:
#is_forward = 2000 <= abs(status) < 4000
