#!/usr/bin/env python3
"""
electron_geo_plots.py

Make before/after geometry plots for electrons using the snapshot parquet
from electron_snapshot.py. Before = pid & status & Nphe>2. After = mask_final.
Outputs: DC x-y per region, and theta/phi/p/vz histograms.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_dc_xy_side_by_side(df_before, df_after, region, out_path):
    xcol = f"dc_x_r{region}"
    ycol = f"dc_y_r{region}"
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    if not df_before.empty:
        axs[0].scatter(df_before[xcol], df_before[ycol], s=2, alpha=0.3)
    axs[0].set_title(f"Region {region} before")
    if not df_after.empty:
        axs[1].scatter(df_after[xcol], df_after[ycol], s=2, alpha=0.3, color="C1")
    axs[1].set_title(f"Region {region} after")
    for ax in axs:
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_hist_overlay(df_before, df_after, col, bins, out_path, xlabel):
    plt.figure(figsize=(5, 4))
    plt.hist(df_before[col], bins=bins, histtype="step", label="before")
    plt.hist(df_after[col], bins=bins, histtype="step", label="after", color="C1")
    plt.xlabel(xlabel)
    plt.ylabel("Entries")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Before/after electron geometry plots from snapshot parquet")
    ap.add_argument("parquet", help="Input parquet from electron_snapshot.py")
    ap.add_argument("--out-prefix", default="egeo", help="Output prefix for plots")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)

    # Before: pid & status & nphe>2; After: full mask_final
    before_mask = df["mask_pid"] & df["mask_status"] & df["mask_nphe"]
    after_mask = df["mask_final"]

    df_before = df[before_mask]
    df_after = df[after_mask]

    # helper to pick rows that have nonzero coordinates in a given region
    def region_slice(df_in, reg):
        xcol = f"dc_x_r{reg}"
        ycol = f"dc_y_r{reg}"
        return df_in[(df_in[xcol] != 0) | (df_in[ycol] != 0)]

    # DC x-y per region (side-by-side)
    for reg in (1, 2, 3):
        sub_b = region_slice(df_before, reg)
        sub_a = region_slice(df_after, reg)
        plot_dc_xy_side_by_side(sub_b, sub_a, reg, f"{args.out_prefix}_dc_r{reg}.png")

    # theta, phi, p, vz histos (before/after overlay)
    plot_hist_overlay(df_before, df_after, "theta", bins=100, out_path=f"{args.out_prefix}_theta.png", xlabel="theta [deg]")
    plot_hist_overlay(df_before, df_after, "phi",   bins=100, out_path=f"{args.out_prefix}_phi.png",   xlabel="phi [deg]")
    plot_hist_overlay(df_before, df_after, "p",     bins=100, out_path=f"{args.out_prefix}_p.png",     xlabel="p [GeV]")
    plot_hist_overlay(df_before, df_after, "vz",    bins=100, out_path=f"{args.out_prefix}_vz.png",    xlabel="vz [cm]")


if __name__ == "__main__":
    main()
