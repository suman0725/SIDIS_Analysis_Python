#!/usr/bin/env python3
"""
electron_snapshot.py

Builds a per-track parquet for electron candidates with all detector variables
and cutflow masks (pid/status/nphe/p/vz/pcal/dc/sf/final). Useful for before/after
plots of DC x/y, p–theta–phi, SF, etc.

Usage:
  python electron_snapshot.py /path/to/file.root --target Cu --tree data;24 --entry-stop 5000
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import uproot

from bank_builders import REC_BRANCHES, build_per_particle_arrays
from electron_cuts import electron_cutflow, EDGE_CUTS_OB
from physics import get_theta


def detect_polarity(path, tree_names=("data;24", "data;8")) -> str:
    """
    Read RUN_config_torus from the first entry to decide polarity.
    Returns "OB" if torus > 0, else "IB".
    Tries a list of tree names.
    """
    for tname in tree_names:
        try:
            with uproot.open(path) as f:
                tree = f[tname]
                torus = tree.arrays("RUN_config_torus", entry_stop=1, library="np")["RUN_config_torus"][0]
                return "OB" if torus > 0 else "IB"
        except Exception:
            continue
    return "OB"


def make_snapshot(root_path, target, tree_name="data;24", entry_stop=None, out_path=None):
    print(f"Reading {root_path} (tree={tree_name}, entry_stop={entry_stop})")
    with uproot.open(root_path) as f:
        tree = f[tree_name]
        arrs = tree.arrays(REC_BRANCHES, entry_stop=entry_stop, library="ak")

    df = build_per_particle_arrays(arrs, target_group=target)

    # basic angle
    df["theta"] = get_theta(df["px"], df["py"], df["pz"], degrees=True)

    # electron cutflow masks
    _, _, masks = electron_cutflow(df, target=target, polarity="OB")
    for name, m in masks.items():
        df[f"mask_{name}"] = m

    # keep everything; just ensure writable path
    pol = detect_polarity(root_path)
    if out_path is None:
        src_tag = Path(root_path).stem
        out_path = f"electron_tracks_{target}_{pol}_{src_tag}.parquet"

    df.to_parquet(out_path)
    print(f"Saved electron snapshot to {out_path} (rows={len(df)})")


def main():
    ap = argparse.ArgumentParser(description="Build per-track parquet with electron cutflow masks")
    ap.add_argument("root_file", help="Input ROOT file path")
    ap.add_argument("--target", default="Cu", help="Target name, default Cu")
    ap.add_argument("--tree", default="data;24", help="TTree name (default data;24)")
    ap.add_argument("--entry-stop", type=int, default=None, help="Optional entry_stop for quick tests")
    ap.add_argument("--out", default=None, help="Output parquet path")
    args = ap.parse_args()
    make_snapshot(args.root_file, args.target, tree_name=args.tree, entry_stop=args.entry_stop, out_path=args.out)


if __name__ == "__main__":
    main()
