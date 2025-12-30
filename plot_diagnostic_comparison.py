#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
import sys
import os
import numpy as np

# Load analysis parameters
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
try:
    from params_electron_sf_outbending import SF_PARAMS_OB
    from electron_cuts import sf_mean, sf_sigma
except ImportError:
    print("!!! ERROR: Could not find scripts/ folder.")
    sys.exit(1)

def run_target_case(target_case, df, output_dir="outputs"):
    output_pdf = os.path.join(output_dir, f"diagnostic_{target_case}_report.pdf")
    
    # Pre-calculate common masks for speed - UPDATED TO USE e_ PREFIX
    cusn_after_mask = (df['pass_dc'] == True)
    cu_vz_mask = (df['pass_final'] == True) & (df['e_vz'].between(-10.6, -6.5))
    sn_vz_mask = (df['pass_final'] == True) & (df['e_vz'].between(-5.5, 5.0))
    
    # Target-specific selection logic
    if target_case == "Cu":
        target_mask = cu_vz_mask
    elif target_case == "Sn":
        target_mask = sn_vz_mask
    else: # Inclusive CuSn
        target_mask = cusn_after_mask

    final_df = df[target_mask].copy()
    intermediate_df = df[cusn_after_mask].copy()

    # Calculation of missing DIS variables for plotting (No cuts applied)
    # Using e_ prefixed kinematics from the new bank_builder
    for d in [df, final_df, intermediate_df]:
        if 'e_W2' not in d.columns: d['e_W2'] = d['e_W']**2

    with PdfPages(output_pdf) as pdf:
        # --- PAGE 1: Vz (Foils Highlighted) ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['e_vz'], bins=300, range=(-25, 20), color='grey', alpha=0.3, label='CuSn base')
        if target_case == "CuSn":
            ax.hist(df[cu_vz_mask]['e_vz'], bins=300, range=(-25, 20), color='red', alpha=0.4, label='Cu', histtype='step', lw=1.5)
            ax.hist(df[sn_vz_mask]['e_vz'], bins=300, range=(-25, 20), color='blue', alpha=0.7, label='Sn', histtype='step', lw=1.2)
        else:
            c = 'blue' if target_case == 'Sn' else 'red'
            ax.hist(final_df['e_vz'], bins=300, range=(-25, 20), color=c, alpha=0.7, label=f'{target_case} After Cut')
        ax.set_title(f"Vertex Distribution: {target_case}"); ax.set_xlabel("$V_{z,e'}$ [cm]"); ax.legend(); ax.grid(False); pdf.savefig(); plt.close()

        # --- PAGES 2-4: SF vs P (2D Histograms) ---
        p_eval = np.linspace(0.8, 10.0, 50) 
        for sector_pair in [(1,2), (3,4), (5,6)]:
            fig, axes = plt.subplots(2, 2, figsize=(20, 14))
            plt.subplots_adjust(wspace=0.35, hspace=0.4)
            for row, s in enumerate(sector_pair):
                s_df, s_final = df[df['sector'] == s], final_df[final_df['sector'] == s]
                pars = SF_PARAMS_OB["my_CuSn"][s]
                mu, sig = sf_mean(p_eval, pars["mu"]), sf_sigma(p_eval, pars["sigma"])
                for col in [0, 1]:
                    ax = axes[row, col]; data = s_df if col == 0 else s_final
                    im = ax.hist2d(data['e_p'], data['e_sf'], bins=[120, 120], range=[[0, 10], [0, 0.45]], cmap='viridis', cmin=1)
                    fig.colorbar(im[3], ax=ax, fraction=0.046, pad=0.04)
                    ax.set_title(f"Sector {s} " + ("Raw (Base)" if col==0 else f"{target_case} After Cut"))
                    if col == 0:
                        ax.plot(p_eval, mu, color='red', lw=1.5)
                        ax.scatter(p_eval, mu+3*sig, color='red', s=4, label=r"$\mu \pm 3\sigma$")
                        ax.scatter(p_eval, mu-3*sig, color='red', s=4)
                    
                    # SF Axis formatting
                    ax.xaxis.set_major_locator(MultipleLocator(1.0)) 
                    ax.xaxis.set_minor_locator(MultipleLocator(0.5)) 
                    ax.yaxis.set_major_locator(MultipleLocator(0.05))
                    ax.yaxis.set_minor_locator(MultipleLocator(0.025)) 
                    
                    ax.set_ylabel("SF ($E_{tot}/P_{e'}$)"); ax.set_xlabel("$P_{e'}$ [GeV]"); ax.grid(False)
            pdf.savefig(); plt.close()

        # --- PAGES 5-7: DC GEOMETRY ---
        for region in [1, 2, 3]:
            fig, axes = plt.subplots(1, 2, figsize=(20, 10)); plt.subplots_adjust(wspace=0.25)
            x_col, y_col = f"e_dc_x_r{region}", f"e_dc_y_r{region}"
            r_lim = {1: 140, 2: 200, 3: 300}[region] 
            for col in [0, 1]:
                ax = axes[col]; data = df if col == 0 else final_df 
                im = ax.hist2d(data[x_col], data[y_col], bins=250, range=[[-r_lim, r_lim], [-r_lim, r_lim]], cmap='viridis', cmin=1)
                fig.colorbar(im[3], ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f"DC R{region} " + ("Base" if col==0 else f"{target_case} After Cut"))
                ax.set_xlabel("DC X [cm]"); ax.set_ylabel("DC Y [cm]")
                ax.set_aspect('equal'); ax.grid(False)
            pdf.savefig(); plt.close()

       # --- PAGES 8-13: 1D KINEMATICS & DIS VARIABLES (Layered Comparison) ---
        plot_list = [
            ('e_p', r"$P_{e'}$ [GeV]", (0, 12)), ('e_theta', r"$\theta_{e'}$ [deg]", (0, 40)), 
            ('e_phi', r"$\phi_{e'}$ [deg]", (-180, 180)), ('e_Q2', r"$Q^2$ [GeV$^2$]", (0, 12)), 
            ('e_xB', r"$x_B$", (0, 1.0)), ('e_W', r"$W$ [GeV]", (0, 6)), 
            ('e_nu', r"$\nu$ [GeV]", (0, 10.6)), ('e_y', r"$y$", (0, 1)), 
            ('e_W2', r"$W^2$ [GeV$^2$]", (0, 20))
        ]
        
        for col, label, rng in plot_list:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 1. Base (Grey filled) - The full raw sample
            ax.hist(df[col], bins=150, range=rng, color='grey', alpha=0.2, label='CuSn base')
            
            # 2. Intermediate (Black Step) - Inclusive sample after detector cuts
            ax.hist(intermediate_df[col], bins=150, range=rng, color='black', alpha=0.8, 
                    label='CuSn After Cut', histtype='step', lw=1.5)
            
            if target_case == "CuSn":
                # 3. Cu Foil (Red Step)
                ax.hist(df[cu_vz_mask][col], bins=150, range=rng, color='red', alpha=0.8, 
                        label='Cu', histtype='step', lw=1.5)
                # 4. Sn Foil (Blue Step)
                ax.hist(df[sn_vz_mask][col], bins=150, range=rng, color='blue', alpha=0.8, 
                        label='Sn', histtype='step', lw=1.5)
            else:
                # 3. Isolated Target (Red for Cu, Blue for Sn)
                final_color = 'red' if target_case == 'Cu' else 'blue'
                ax.hist(final_df[col], bins=150, range=rng, color=final_color, alpha=0.8, 
                        label=f'{target_case} After Cut', histtype='step', lw=1.8)

            ax.set_title(f"{label.split('[')[0]} Distribution")
            ax.set_xlabel(label)
            leg_loc = 'upper left' if col in ['e_nu', 'e_y', 'e_W2'] else 'upper right'
            ax.legend(loc=leg_loc, fontsize=10)
            ax.grid(False)
            pdf.savefig()
            plt.close()

def main():
    # Update to the REAL path from your log
    path = "/work/clas12/suman/new_RGD_Analysis/data/CuSn/018752/diagnostics/diag_e_CuSn_rec_clas_018752.evio.00000-00004.parquet"
    if not os.path.exists(path): 
        print(f"!!! File not found: {path}")
        return
    df = pd.read_parquet(path)
    if not os.path.exists("outputs"): os.makedirs("outputs")
    for case in ["CuSn", "Cu", "Sn"]:
        run_target_case(case, df)
    print(">>> All Electron Target Reports Created Successfully.")

if __name__ == "__main__":
    main()