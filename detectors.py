# rgd_cuts/detectors.py
"""
CLAS12 detector / region / layer IDs in one place.

Usage:
    from rgd_cuts.detectors import RegionID, DetectorID, ECAL_LAYERS

    if det_id == DetectorID.FTOF:
        ...

    if layer_id == ECAL_LAYERS["PCAL"]:
        ...
"""

from enum import IntEnum


# -------------------------
# Regions
# -------------------------

class RegionID(IntEnum):
    FT = 1000
    FD = 2000
    CD = 3000
    BD = 4000


# -------------------------
# Detectors
# -------------------------

class DetectorID(IntEnum):
    FTOF   = 12
    CTOF   = 4
    CND    = 3
    CVT    = 5
    DC     = 6
    ECAL   = 7
    FTCAL  = 10
    FTTRK  = 13
    FTHODO = 11
    HTCC   = 15
    LTCC   = 16
    BMT    = 1
    FMT    = 8
    RF     = 17
    RICH   = 18
    RTPC   = 19
    HEL    = 20
    BAND   = 21


# -------------------------
# Layers / sub-layers
# -------------------------

# FTOF layers (within detector FTOF)
FTOF_LAYERS = {
    "1A": 1,
    "1B": 2,
    "2":  3,
}

# CND layers
CND_LAYERS = {
    "OFF": 150,
    "1":   151,
    "2":   152,
    "3":   153,
}

# ECAL layers
ECAL_LAYERS = {
    "PCAL": 1,
    "ECIN": 4,
    "ECOUT": 7,
}

# DC regions (layer numbers)
DC_LAYERS = {
    "R1": 6,
    "R2": 12,
    "R3": 18,
    "R4": 24,
    "R5": 30,
    "R6": 36,
}

# CVT layers
CVT_LAYERS = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
}

# BAND / backward TOF layers
BAND_LAYERS = {
    "OFF": 250,
    "BTOF1": 251,
    "BTOF2": 252,
    "BTOF3": 253,
    "BTOF4": 254,
    "BTOF5": 255,
    "BVETO": 256,
}

