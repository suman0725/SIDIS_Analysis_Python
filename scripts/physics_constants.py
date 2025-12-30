"""
physics_constants.py

All basic constants for CLAS12 DIS analysis.
"""

import numpy as np
# Energy (GeV) 
E_BEAM = 10.54 
# Masses (GeV)
M_PROTON   = 0.9382720813
M_NEUTRON  = 0.9395654133
M_NUCLEON  = M_PROTON       # default target mass for xB, W
M_ELECTRON = 0.0005109989
M_PION_CHARGED = 0.13957 

# Angle conversions
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

