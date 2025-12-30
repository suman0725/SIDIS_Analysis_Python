#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os
import numpy as np

def run_target_report(target_case, df, output_dir="outputs"):
    """
    Generates a PDF report for a specific target (Cu, Sn, or inclusive CuSn).
    Filters both base and cleaned samples by e_vz to preserve target nature.
    """
    output_pdf = os.path.join(output_dir, f"diagnostic_pip_{target_case}_report.pdf")
    
    # 1. APPLY TARGET FILTER FIRST using e_vz
    if target_case == "Cu":
        df_target = df[df['e_vz'].between(-10.6, -6.5)].copy()
    elif target_case == "Sn":
        df_target = df[df['e_vz'].between(-5.5, 5.0)].copy()
    else: # CuSn (Inclusive)
        df_target = df.copy()

    # 2. Define comparison groups
    df_base = df_target.copy() 
    df_after = df_target[df_target['pass_e_cleanness'] == True].copy()

    label_base = r"$\pi^+$ from base $e'$"
    label_after = r"$\pi^+$ from $e'$ after all cuts"

    # 3. Pre-calculate Matching
    for d in [df_base, df_after]:
        if not d.empty:
            d['pip_dvz'] = d['e_vz'] - d['pip_vz']

    with PdfPages(output_pdf) as pdf:

        # --- PAGE 1: Pion Vertex ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df_base['pip_vz'], bins=300, range=(-25, 20), color='grey', alpha=0.3, label=label_base)
        ax.hist(df_after['pip_vz'], bins=300, range=(-25, 20), color='blue', alpha=0.7, 
                label=label_after, histtype='step', lw=1.5)
        ax.set_title(fr"{target_case} Pion Vertex Distribution: $V_{{z,\pi^+}}$")
        ax.set_xlabel(r"$V_{z,\pi^+}$ [cm]"); ax.set_ylabel("Counts")
        ax.legend(loc='upper right'); ax.grid(False)
        pdf.savefig(); plt.close()
        
        # --- PAGE 2: Beta vs P ---
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        for col, data in enumerate([df_base, df_after]):
            ax = axes[col]
            if not data.empty:
                im = ax.hist2d(data['pip_p'], data['pip_beta'], bins=200, range=[[0, 8], [0.8, 1.2]], cmap='viridis', cmin=1)
                ax.set_title(fr"$\beta$ vs $P_{{\pi^+}}$: " + (label_base if col==0 else label_after))
                ax.set_xlabel(r"$P_{\pi^+}$ [GeV]"); ax.set_ylabel(r"$\beta$")
                p_vals = np.linspace(0.5, 8, 100)
                ax.plot(p_vals, p_vals / np.sqrt(p_vals**2 + 0.13957**2), color='red', linestyle='--', alpha=0.6)
                plt.colorbar(im[3], ax=ax)
        pdf.savefig(); plt.close()

        # --- PAGE 3: Delta Vz ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df_base['pip_dvz'], bins=200, range=(-10, 10), color='grey', alpha=0.3, label=label_base)
        ax.hist(df_after['pip_dvz'], bins=200, range=(-10, 10), color='blue', alpha=0.7, label=label_after, histtype='step', lw=2)
        ax.set_title(fr"Vertex Matching: $\Delta V_z(e', \pi^+)$")
        ax.set_xlabel(r"$\Delta V_z$ [cm]"); ax.legend(); ax.grid(False)
        pdf.savefig(); plt.close()

        # --- PAGE 4: Chi2PID ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df_base['pip_chi2pid'], bins=100, range=(-5, 5), color='grey', alpha=0.3, label=label_base)
        ax.hist(df_after['pip_chi2pid'], bins=100, range=(-5, 5), color='green', alpha=0.7, label=label_after, histtype='step', lw=2)
        ax.set_title(fr"$\chi^2_{{PID, \pi^+}}$ Distribution") 
        ax.set_xlabel(fr"$\chi^2_{{PID, \pi^+}}$"); ax.legend(); pdf.savefig(); plt.close()

        # --- PAGE 5: SIDIS z ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df_base['pip_zh'], bins=100, range=(0.01, 1.2), color='grey', alpha=0.3, label=label_base)
        ax.hist(df_after['pip_zh'], bins=100, range=(0.01, 1.2), color='blue', alpha=0.7, label=label_after, histtype='step', lw=1.5)
        ax.set_title(fr"Energy Fraction $z_{{\pi^+}}$")
        ax.set_xlabel(fr"$z_{{\pi^+}}$"); ax.legend(); pdf.savefig(); plt.close()
        
        # --- PAGE 6: SIDIS PT2 ---
        fig, ax = plt.subplots(figsize=(10, 6))
        pt2_base = df_base[df_base['pip_pt2'] > 0.001]['pip_pt2']
        pt2_after = df_after[df_after['pip_pt2'] > 0.001]['pip_pt2']
        ax.hist(pt2_base, bins=100, range=(0.01, 2), color='grey', alpha=0.3, label=label_base)
        ax.hist(pt2_after, bins=100, range=(0.01, 2), color='purple', alpha=0.7, label=label_after, histtype='step', lw=1.5)
        ax.set_title(fr"Transverse Momentum $P_{{T,\pi^+}}^2$")
        ax.set_xlabel(fr"$P_{{T,\pi^+}}^2$ [GeV$^2$]"); ax.legend(); pdf.savefig(); plt.close()

        # --- PAGE 7: Trento Azimuthal Angle ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df_base['pip_phih'], bins=100, range=(0, 360), color='grey', alpha=0.3, label=label_base)
        ax.hist(df_after['pip_phih'], bins=100, range=(0, 360), color='orange', alpha=0.7, label=label_after, histtype='step', lw=1.5)
        ax.set_title(fr"Trento Azimuthal Angle $\phi_{{trento, \pi^+}}$")
        ax.set_xlabel(fr"$\phi_{{trento, \pi^+}}$ [deg]"); ax.legend(); pdf.savefig(); plt.close()

        # --- PAGES 8-10: DC GEOMETRY ---
        for region in [1, 2, 3]:
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            plt.subplots_adjust(wspace=0.25)
            x_col, y_col = f"pip_dc_x_r{region}", f"pip_dc_y_r{region}"
            r_lim = {1: 200, 2: 300, 3: 400}[region]
            for col, data in enumerate([df_base, df_after]):
                ax = axes[col]
                if not data.empty:
                    im = ax.hist2d(data[x_col], data[y_col], bins=200, range=[[-r_lim, r_lim], [-r_lim, r_lim]], cmap='viridis', cmin=1)
                    ax.set_title(fr"DC R{region}: " + (label_base if col==0 else label_after))
                    ax.set_xlabel("x [cm]"); ax.set_ylabel("y [cm]"); ax.set_aspect('equal')
                    plt.colorbar(im[3], ax=ax, fraction=0.046, pad=0.04)
            pdf.savefig(); plt.close()

        # --- PAGES 11-13: 1D Kinematics ---
        kin_plots = [
            ('pip_p', r"$P_{\pi^+}$ [GeV]", (0, 10)),
            ('pip_theta', r"$\theta_{\pi^+}$ [deg]", (0, 65)),
            ('pip_phi', r"$\phi_{\pi^+}$ [deg]", (-180, 180))
        ]
        for col_name, x_label, x_range in kin_plots:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df_base[col_name], bins=150, range=x_range, color='grey', alpha=0.3, label=label_base)
            ax.hist(df_after[col_name], bins=150, range=x_range, color='blue', alpha=0.7, 
                    label=label_after, histtype='step', lw=1.5)
            ax.set_title(fr"{x_label.split('[')[0].strip()} Distribution")
            ax.set_xlabel(x_label); ax.set_ylabel("Counts"); ax.legend(); ax.grid(False)
            pdf.savefig(); plt.close()

        # --- PAGE 14: Summary ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        n_base, n_after = len(df_base), len(df_after)
        reduction = ((n_base - n_after) / n_base * 100) if n_base > 0 else 0
        summary_text = (
            f"Reduction Summary for Target: {target_case}\n"
            f"{'='*40}\n"
            f"Total Pions (Base e-): {n_base}\n"
            f"Total Pions (Clean e-): {n_after}\n\n"
            f"Rejection Percentage: {reduction:.2f}%\n"
            f"Retention Percentage: {100-reduction:.2f}%"
        )
        ax.text(0.1, 0.5, summary_text, fontsize=14, family='monospace', verticalalignment='center')
        ax.set_title(fr"Numerical Statistics: {target_case}")
        pdf.savefig(); plt.close()

def main():
    path = "/work/clas12/suman/new_RGD_Analysis/data/CuSn/018752/diagnostics/diag_pip_CuSn_rec_clas_018752.evio.00000-00004.parquet"
    if not os.path.exists(path):
        print(f"!!! File not found: {path}"); return
    
    df = pd.read_parquet(path)
    if not os.path.exists("outputs"): os.makedirs("outputs")
    
    for case in ["CuSn", "Cu", "Sn"]:
        print(f">>> Processing Target: {case}")
        run_target_report(case, df)
    print(">>> COMPLETE.")

if __name__ == "__main__":
    main()