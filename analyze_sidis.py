#!/usr/bin/env python3
"""
analyze_sidis.py

Quick sanity-check script: read a SIDIS parquet, apply optional DIS cuts,
and plot basic DIS (Q2, xB, y, W, nu) and SIDIS (zh, pT2, phi_h) distributions.
Designed for a small test sample (e.g., 5k events) to validate shapes/ranges.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def add_lines(ax, vlines=None, hlines=None):
    if vlines:
        for xv in vlines:
            ax.axvline(xv, color="red", ls="--", lw=1)
    if hlines:
        for yv in hlines:
            ax.axhline(yv, color="blue", ls="--", lw=1)


def parse_line_map(entries):
    """Parse CLI entries like 'Q2:1.5,2.5' into {'Q2':[1.5,2.5]}."""
    out = {}
    if not entries:
        return out
    for item in entries:
        if ":" not in item:
            continue
        key, vals = item.split(":", 1)
        if not vals:
            continue
        out[key] = [float(v) for v in vals.split(",") if v.strip() != ""]
    return out


def hist1d(series, bins, out_path, xlabel, title=None, rng=None, annotate=False, log=False, vlines=None):
    s = series.dropna()
    if rng is not None:
        s = s[(s >= rng[0]) & (s <= rng[1])]
    plt.figure(figsize=(5, 4))
    plt.hist(s, bins=bins, histtype="step")
    add_lines(plt.gca(), vlines=vlines)
    if title:
        plt.title(title)
    if annotate and not s.empty:
        p1, p99 = s.quantile(0.01), s.quantile(0.99)
        plt.text(0.02, 0.95, f"p1={p1:.3f}\np99={p99:.3f}", transform=plt.gca().transAxes,
                 va="top", ha="left", fontsize=9)
    plt.xlabel(xlabel)
    plt.ylabel("Entries")
    if log:
        plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def hist2d(x, y, bins, out_path, xlabel, ylabel, title=None, xlim=None, ylim=None, log=False, vlines=None, hlines=None):
    from matplotlib.colors import LogNorm
    plt.figure(figsize=(5, 4))
    xx = x.dropna()
    yy = y.dropna()
    if xlim is not None:
        mask = (xx >= xlim[0]) & (xx <= xlim[1])
        xx = xx[mask]; yy = yy[mask]
    if ylim is not None:
        mask = (yy >= ylim[0]) & (yy <= ylim[1])
        xx = xx[mask]; yy = yy[mask]
    norm = LogNorm() if log else None
    plt.hist2d(xx, yy, bins=bins, cmap="viridis", norm=norm)
    add_lines(plt.gca(), vlines=vlines, hlines=hlines)
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label="counts")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot DIS/SIDIS sanity histos from SIDIS parquet")
    ap.add_argument("parquet", help="SIDIS parquet (from run_single_sample.py)")
    ap.add_argument("--out-prefix", default="sidis_test", help="Output prefix for plots")
    ap.add_argument("--apply-dis", action="store_true", help="Apply DIS cuts: Q2>1, W>2, y<0.85")
    ap.add_argument("--vlines", type=float, nargs="*", default=[],
                    help="Default x-axis guide lines (all plots unless overridden)")
    ap.add_argument("--hlines", type=float, nargs="*", default=[],
                    help="Default y-axis guide lines (2D plots unless overridden)")
    ap.add_argument("--vlines-map", action="append", default=[],
                    help="Per-plot vertical lines, e.g. Q2:1.5,2.5 (repeatable)")
    ap.add_argument("--hlines-map", action="append", default=[],
                    help="Per-plot horizontal lines for 2D, e.g. Q2_xB:2.0 (repeatable)")
    args = ap.parse_args()

    vmap = parse_line_map(args.vlines_map)
    hmap = parse_line_map(args.hlines_map)

    df = pd.read_parquet(args.parquet)

    # electrons: one per event w_e==1
    e = df[df["w_e"] == 1]
    if args.apply_dis:
        e = e[(e["Q2"] > 1.0) & (e["W"] > 2.0) & (e["y"] < 0.85)]
        df = df[df["sel_event_idx"].isin(e["sel_event_idx"])]

    # DIS histos (linear + log)
    hist1d(e["Q2"], bins=100, out_path=f"{args.out_prefix}_Q2.png", xlabel="Q2 [GeV^2]", vlines=vmap.get("Q2", args.vlines))
    hist1d(e["Q2"], bins=100, out_path=f"{args.out_prefix}_Q2_log.png", xlabel="Q2 [GeV^2]", log=True, vlines=vmap.get("Q2", args.vlines))

    hist1d(e["xB"], bins=100, out_path=f"{args.out_prefix}_xB.png", xlabel="xB", rng=(0, 1.1), annotate=True, vlines=vmap.get("xB", args.vlines))
    hist1d(e["xB"], bins=100, out_path=f"{args.out_prefix}_xB_log.png", xlabel="xB", rng=(0, 1.1), annotate=True, log=True, vlines=vmap.get("xB", args.vlines))

    hist1d(e["y"],  bins=100, out_path=f"{args.out_prefix}_y.png",  xlabel="y", vlines=vmap.get("y", args.vlines))
    hist1d(e["y"],  bins=100, out_path=f"{args.out_prefix}_y_log.png",  xlabel="y", log=True, vlines=vmap.get("y", args.vlines))

    hist1d(e["W"],  bins=100, out_path=f"{args.out_prefix}_W.png",  xlabel="W [GeV]", vlines=vmap.get("W", args.vlines))
    hist1d(e["W"],  bins=100, out_path=f"{args.out_prefix}_W_log.png",  xlabel="W [GeV]", log=True, vlines=vmap.get("W", args.vlines))

    hist1d(e["nu"], bins=100, out_path=f"{args.out_prefix}_nu.png", xlabel="nu [GeV]", vlines=vmap.get("nu", args.vlines))
    hist1d(e["nu"], bins=100, out_path=f"{args.out_prefix}_nu_log.png", xlabel="nu [GeV]", log=True, vlines=vmap.get("nu", args.vlines))

    hist2d(e["xB"], e["Q2"], bins=100, out_path=f"{args.out_prefix}_Q2_xB.png", xlabel="xB", ylabel="Q2 [GeV^2]", xlim=(0,1.1), vlines=vmap.get("Q2_xB", args.vlines), hlines=hmap.get("Q2_xB", args.hlines))
    hist2d(e["xB"], e["Q2"], bins=100, out_path=f"{args.out_prefix}_Q2_xB_log.png", xlabel="xB", ylabel="Q2 [GeV^2]", xlim=(0,1.1), log=True, vlines=vmap.get("Q2_xB", args.vlines), hlines=hmap.get("Q2_xB", args.hlines))

    # SIDIS histos (pi+ rows)
    pip = df[df["w_pip"] == 1]
    if not pip.empty:
        hist1d(pip["zh"],   bins=100, out_path=f"{args.out_prefix}_zh.png",   xlabel="z_h", rng=(0,1.1), annotate=True, vlines=vmap.get("zh", args.vlines))
        hist1d(pip["zh"],   bins=100, out_path=f"{args.out_prefix}_zh_log.png",   xlabel="z_h", rng=(0,1.1), annotate=True, log=True, vlines=vmap.get("zh", args.vlines))

        hist1d(pip["pT2"],  bins=100, out_path=f"{args.out_prefix}_pT2.png",  xlabel="pT^2 [GeV^2]", rng=(0,3.0), annotate=True, vlines=vmap.get("pT2", args.vlines))
        hist1d(pip["pT2"],  bins=100, out_path=f"{args.out_prefix}_pT2_log.png",  xlabel="pT^2 [GeV^2]", rng=(0,3.0), annotate=True, log=True, vlines=vmap.get("pT2", args.vlines))

        hist1d(pip["phi_h"],bins=100, out_path=f"{args.out_prefix}_phi_h.png",xlabel="phi_h [deg]", vlines=vmap.get("phi_h", args.vlines))
        hist1d(pip["phi_h"],bins=100, out_path=f"{args.out_prefix}_phi_h_log.png",xlabel="phi_h [deg]", log=True, vlines=vmap.get("phi_h", args.vlines))

        hist2d(pip["zh"], pip["pT2"], bins=100, out_path=f"{args.out_prefix}_zh_pT2.png", xlabel="z_h", ylabel="pT^2 [GeV^2]", xlim=(0,1.1), ylim=(0,3.0), vlines=vmap.get("zh_pT2", args.vlines), hlines=hmap.get("zh_pT2", args.hlines))
        hist2d(pip["zh"], pip["pT2"], bins=100, out_path=f"{args.out_prefix}_zh_pT2_log.png", xlabel="z_h", ylabel="pT^2 [GeV^2]", xlim=(0,1.1), ylim=(0,3.0), log=True, vlines=vmap.get("zh_pT2", args.vlines), hlines=hmap.get("zh_pT2", args.hlines))
        hist2d(pip["pT2"], pip["zh"], bins=100, out_path=f"{args.out_prefix}_pT2_zh.png", xlabel="pT^2 [GeV^2]", ylabel="z_h", xlim=(0,3.0), ylim=(0,1.1), vlines=vmap.get("pT2_zh", args.vlines), hlines=hmap.get("pT2_zh", args.hlines))
        hist2d(pip["pT2"], pip["zh"], bins=100, out_path=f"{args.out_prefix}_pT2_zh_log.png", xlabel="pT^2 [GeV^2]", ylabel="z_h", xlim=(0,3.0), ylim=(0,1.1), log=True, vlines=vmap.get("pT2_zh", args.vlines), hlines=hmap.get("pT2_zh", args.hlines))
        for col in ("zh", "pT2"):
            series = pip[col].dropna()
            if series.empty:
                continue
            print(f"{col} min={series.min():.3f} max={series.max():.3f} p1={series.quantile(0.01):.3f} p99={series.quantile(0.99):.3f}")

    print(f"Saved plots with prefix {args.out_prefix}")


if __name__ == "__main__":
    main()
