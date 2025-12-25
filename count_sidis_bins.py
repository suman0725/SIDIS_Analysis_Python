#!/usr/bin/env python3
"""
count_sidis_bins.py
"""

import argparse
import sys
import glob
import os
from itertools import product
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import boost_histogram as bh
except ImportError:
    sys.stderr.write("boost-histogram is required. Install with: python3 -m pip install --user boost-histogram\n")
    raise

DEFAULT_BINS: Dict[str, List[float]] = {
    "Q2": [1.0, 2.0, 4.0, 8.0],
    "xB": [0.0, 0.2, 0.4, 0.6, 1.1],
    "zh": [0.0, 0.3, 0.5, 0.7, 0.9, 1.1],
    "pT2": [0.0, 0.3, 0.7, 1.2, 2.0, 3.0],
    "phi_h": [0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0],
}

def parse_binspec(spec: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    if not spec:
        return out
    for part in [p.strip() for p in spec.split(";") if p.strip()]:
        if "=" not in part:
            continue
        dim, vals = part.split("=", 1)
        edges = [float(x) for x in vals.split(",") if x.strip()]
        if len(edges) >= 2:
            out[dim.strip()] = sorted(edges)
    return out

def ensure_bins(axis_list: List[str], user_bins: Dict[str, List[float]]) -> Dict[str, List[float]]:
    dims = set(axis_list)
    bins: Dict[str, List[float]] = {}
    for dim in dims:
        bins[dim] = user_bins.get(dim, DEFAULT_BINS.get(dim, [0, 1]))
    return bins

def build_axes(dim_order: List[str], bins: Dict[str, List[float]]):
    axes = []
    for dim in dim_order:
        edges = bins[dim]
        if dim == "phi_h" and len(edges) >= 2 and edges[0] == 0.0 and edges[-1] == 360.0:
            step = edges[1] - edges[0]
            if all(abs((edges[i + 1] - edges[i]) - step) < 1e-6 for i in range(len(edges) - 2)):
                axes.append(bh.axis.Regular(len(edges) - 1, edges[0], edges[-1], circular=True))
                continue
        axes.append(bh.axis.Variable(edges))
    return axes

def main():
    ap = argparse.ArgumentParser(description="Flexible SIDIS bin counter")
    ap.add_argument("parquet", help="Path to parquet files (use quotes if using wildcards like *.parquet)")
    ap.add_argument("--axes", default="Q2,xB,zh,pT2", help="Comma list for pi+ axes")
    ap.add_argument("--bins", help='Bin spec string')
    ap.add_argument("--apply-dis", action="store_true", help="Apply DIS cuts")
    ap.add_argument("--q2-min", type=float, default=1.0)
    ap.add_argument("--w-min", type=float, default=2.0)
    ap.add_argument("--y-max", type=float, default=0.85)
    ap.add_argument("--y-min", type=float, default=0.0)
    ap.add_argument("--zh-min", type=float, default=0.3)
    ap.add_argument("--zh-max", type=float, default=1.1)
    ap.add_argument("--pt2-min", type=float, default=0.0)
    ap.add_argument("--pt2-max", type=float, default=1.5)
    ap.add_argument("--integrate-q2xb", action="store_true", help="Integrate Q2,xB")
    ap.add_argument("--integrate-over", help="Comma-separated dimensions to integrate")
    ap.add_argument("--out-csv", help="Optional CSV output path")
    args = ap.parse_args()

    # --- FIX: Handle Wildcards properly ---
    # This expands the "*" if provided in quotes
    file_list = sorted(glob.glob(args.parquet))
    if not file_list:
        # Fallback: maybe it's a directory?
        if os.path.isdir(args.parquet):
             file_list = sorted(glob.glob(os.path.join(args.parquet, "*.parquet")))
    
    if not file_list:
        print(f"Error: No files found matching: {args.parquet}")
        sys.exit(1)
        
    print(f"Loading {len(file_list)} files...")
    df_list = [pd.read_parquet(f) for f in file_list]
    df = pd.concat(df_list, ignore_index=True)
    # --------------------------------------

    # Axis order for hadron histogram
    axis_list = [a.strip() for a in args.axes.split(",") if a.strip()]
    if "Q2" not in axis_list: axis_list.insert(0, "Q2")
    if "xB" not in axis_list: axis_list.insert(1, "xB")

    integrate_list: List[str] = []
    if args.integrate_over:
        integrate_list = [d.strip() for d in args.integrate_over.split(",") if d.strip()]
    elif args.integrate_q2xb:
        integrate_list = ["Q2", "xB"]
    integrate_set = set(integrate_list)

    dims_for_bins = list(set(axis_list) | integrate_set)
    user_bins = parse_binspec(args.bins) if args.bins else {}
    bins = ensure_bins(dims_for_bins, user_bins)

    for dim in integrate_set:
        if dim in bins:
            bins[dim] = [bins[dim][0], bins[dim][-1]]

    print("Axes:", ", ".join(axis_list))
    for dim in axis_list:
        e = bins[dim]
        print(f"  {dim}: {len(e)-1} bins [{e[0]} .. {e[-1]}]")

    if "sel_event_idx" not in df.columns:
        raise RuntimeError("Expected sel_event_idx in parquet")

    if args.apply_dis:
        e_rows = df[df["w_e"] == 1].copy()
        dis_mask = (
            (e_rows["Q2"] >= args.q2_min)
            & (e_rows["W"] >= args.w_min)
            & (e_rows["y"] <= args.y_max)
            & (e_rows["y"] >= args.y_min)
        )
        passing_events = e_rows.loc[dis_mask, "sel_event_idx"].unique()
        df = df[df["sel_event_idx"].isin(passing_events)]

    df_e = df[df["w_e"] == 1].dropna(subset=["Q2", "xB"])
    df_h = df[df["w_pip"] == 1].copy()
    df_h = df_h.dropna(subset=[col for col in axis_list if col in df_h.columns])
    
    if "zh" in df_h.columns:
        df_h = df_h[(df_h["zh"] >= args.zh_min) & (df_h["zh"] <= args.zh_max)]
    if "pT2" in df_h.columns:
        df_h = df_h[(df_h["pT2"] >= args.pt2_min) & (df_h["pT2"] <= args.pt2_max)]

    hist_e = bh.Histogram(*build_axes(["Q2", "xB"], bins), storage=bh.storage.Weight())
    hist_e.fill(df_e["Q2"].to_numpy(), df_e["xB"].to_numpy(), weight=df_e["w_e"].to_numpy())

    h_axes = build_axes(axis_list, bins)
    hist_h = bh.Histogram(*h_axes, storage=bh.storage.Weight())
    hist_h.fill(*[df_h[dim].to_numpy() for dim in axis_list], weight=df_h["w_pip"].to_numpy())

    view_e, var_e = hist_e.view(), hist_e.view().variance
    view_h, var_h = hist_h.view(), hist_h.view().variance
    axis_bins = [bins[d] for d in axis_list]
    rows = []

    print("\nBin table (skip empty):")
    print_dims = [d for d in axis_list if d not in integrate_set]
    header_parts = [f"{dim}_lo-{dim}_hi" for dim in print_dims]
    header = " | ".join(header_parts) + " | N_e | err_e | N_pi | err_pi | R | err_R"
    print(header)
    print("-" * len(header))

    try:
        q2_axis_pos = axis_list.index("Q2")
        xb_axis_pos = axis_list.index("xB")
    except ValueError:
        raise RuntimeError("Q2 and xB must be present in axes")

    for idx_tuple in product(*[range(len(b) - 1) for b in axis_bins]):
        q_bin = idx_tuple[q2_axis_pos]
        xb_bin = idx_tuple[xb_axis_pos]
        
        n_e = float(view_e[q_bin, xb_bin].value)
        err_e = float(np.sqrt(var_e[q_bin, xb_bin]))

        n_h = float(view_h[idx_tuple].value)
        err_h = float(np.sqrt(var_h[idx_tuple]))
        
        if n_e == 0 and n_h == 0: continue
            
        if n_e > 0:
            R = n_h / n_e
            err_R = R * np.sqrt((err_h / n_h) ** 2 + (err_e / n_e) ** 2) if n_h > 0 else 0.0
        else:
            R = 0.0
            err_R = 0.0

        edges = [(axis_bins[i][idx], axis_bins[i][idx + 1]) for i, idx in enumerate(idx_tuple)]
        
        if integrate_set:
            kept_edges = [edges[i] for i, dim in enumerate(axis_list) if dim not in integrate_set]
            edge_txt = " | ".join([f"{lo:.3f}-{hi:.3f}" for lo, hi in kept_edges])
        else:
            edge_txt = " | ".join([f"{lo:.3f}-{hi:.3f}" for lo, hi in edges])
        
        print(f"{edge_txt} | {n_e:.0f} | {err_e:.1f} | {n_h:.0f} | {err_h:.1f} | {R:.4f} | {err_R:.4f}")

        if integrate_set:
            row = {}
            for i, (lo, hi) in enumerate(edges):
                dim = axis_list[i]
                if dim in integrate_set: continue
                row[f"{dim}_lo"] = lo
                row[f"{dim}_hi"] = hi
        else:
            row = {f"{axis_list[i]}_lo": lo for i, (lo, _) in enumerate(edges)}
            row.update({f"{axis_list[i]}_hi": hi for i, (_, hi) in enumerate(edges)})
        
        row.update({"N_e": n_e, "err_e": err_e, "N_pi": n_h, "err_pi": err_h, "R": R, "err_R": err_R})
        rows.append(row)

    print(f"\nTotals: electrons={float(hist_e.sum().value):.0f}, pi+={float(hist_h.sum().value):.0f}")

    if args.out_csv:
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f"Wrote CSV to {args.out_csv}")

if __name__ == "__main__":
    main()