import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os

# --- CONFIGURATION ---
TARGETS = [
    ("CxC", "Carbon", "red", "o"),
    ("Cu",  "Copper", "blue", "s"),
    ("Sn",  "Tin",    "green", "^")
]

OUTPUT_PDF = "multiplicity_ratios.pdf"

def get_center(row, col_prefix):
    return 0.5 * (row[f"{col_prefix}_lo"] + row[f"{col_prefix}_hi"])

def main():
    # 1. Load LD2 Data (Baseline)
    if not os.path.exists("counts_LD2.csv"):
        print("Error: counts_LD2.csv not found!")
        return
    
    df_ld2 = pd.read_csv("counts_LD2.csv")
    
    # Create matching index based on bins
    index_cols = ['Q2_lo', 'Q2_hi', 'xB_lo', 'xB_hi', 'zh_lo', 'zh_hi', 'pT2_lo', 'pT2_hi']
    df_ld2.set_index(index_cols, inplace=True)

    # 2. Process Data for All Targets
    processed_data = {}
    
    for code, label, color, marker in TARGETS:
        fname = f"counts_{code}.csv"
        if not os.path.exists(fname):
            print(f"Warning: {fname} not found, skipping.")
            continue
            
        df_target = pd.read_csv(fname)
        df_target.set_index(index_cols, inplace=True)
        
        # Merge Nuclear (Left) with Deuterium (Right)
        # Suffixes: _N = Nuclear, _D = LD2
        df_merged = df_target.join(df_ld2, lsuffix='_N', rsuffix='_D')
        
        # --- CALCULATION AS REQUESTED ---
        
        # 1. Calculate Multiplicity Ratio (MR)
        # R_N and R_D are the Multiplicities from the CSV (N_pi / N_e)
        df_merged['MR'] = df_merged['R_N'] / df_merged['R_D']
        
        # 2. Calculate Error using the counts (N) directly
        # Formula: MR * sqrt(1/Ne_D + 1/Ne_N + 1/Npi_D + 1/Npi_N)
        
        # Avoid division by zero
        valid_mask = (df_merged['N_pi_N'] > 0) & (df_merged['N_pi_D'] > 0) & \
                     (df_merged['N_e_N'] > 0) & (df_merged['N_e_D'] > 0)
        
        df_merged['err_MR'] = 0.0
        
        df_merged.loc[valid_mask, 'err_MR'] = df_merged.loc[valid_mask, 'MR'] * np.sqrt(
            (1.0 / df_merged.loc[valid_mask, 'N_e_D']) + 
            (1.0 / df_merged.loc[valid_mask, 'N_e_N']) + 
            (1.0 / df_merged.loc[valid_mask, 'N_pi_D']) + 
            (1.0 / df_merged.loc[valid_mask, 'N_pi_N'])
        )
        
        # -------------------------------
        
        df_merged.fillna(0, inplace=True)
        df_merged.reset_index(inplace=True)
        
        # Calculate Centers for Plotting
        df_merged['zh_center'] = df_merged.apply(get_center, axis=1, args=('zh',))
        df_merged['pT2_center'] = df_merged.apply(get_center, axis=1, args=('pT2',))
        
        processed_data[label] = df_merged

    # Check if we have data
    if not processed_data:
        print("No data found to plot.")
        return

    # Use first dataset to get bin definitions
    first_df = list(processed_data.values())[0]
    unique_pt2 = first_df[['pT2_lo', 'pT2_hi']].drop_duplicates().sort_values('pT2_lo')
    unique_zh  = first_df[['zh_lo', 'zh_hi']].drop_duplicates().sort_values('zh_lo')

    print(f"Creating PDF: {OUTPUT_PDF} ...")

    with PdfPages(OUTPUT_PDF) as pdf:
        
        # ==========================================
        # PART 1: Plot R vs zh (for fixed pT2 bins)
        # ==========================================
        print("Plotting R vs zh...")
        for _, row in unique_pt2.iterrows():
            pt_lo, pt_hi = row['pT2_lo'], row['pT2_hi']
            
            plt.figure(figsize=(10, 7))
            
            for code, label, color, marker in TARGETS:
                if label not in processed_data: continue
                df = processed_data[label]
                
                # Slice Data
                subset = df[(df['pT2_lo'] == pt_lo) & (df['pT2_hi'] == pt_hi)].sort_values('zh_center')
                if subset.empty: continue

                offset = 0
                if code == "CxC": offset = -0.005
                if code == "Sn":  offset = 0.005
                
                plt.errorbar(subset['zh_center'] + offset, subset['MR'], yerr=subset['err_MR'], 
                             label=label, color=color, marker=marker, linestyle='-', capsize=4)

            plt.title(f"Multiplicity Ratio vs $z_{{\pi^+}}$", fontsize=14)
            plt.xlabel("$z_{\pi^+}$", fontsize=14)
            plt.ylabel("$R$", fontsize=14)
            plt.ylim(0.0, 2.0)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(fontsize=12)
            
            # Info Box
            q2_lo, q2_hi = first_df['Q2_lo'][0], first_df['Q2_hi'][0]
            xb_lo, xb_hi = first_df['xB_lo'][0], first_df['xB_hi'][0]
            info = (f"${q2_lo} < Q^2 < {q2_hi}$ GeV$^2$\n"
                    f"${xb_lo} < x_B < {xb_hi}$\n"
                    f"${pt_lo} < P_T^2 < {pt_hi}$ GeV$^2$")
            plt.text(0.05, 0.95, info, transform=plt.gca().transAxes, va='top', 
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            pdf.savefig()
            plt.close()

        # ==========================================
        # PART 2: Plot R vs pT2 (for fixed zh bins)
        # ==========================================
        print("Plotting R vs pT2...")
        for _, row in unique_zh.iterrows():
            zh_lo, zh_hi = row['zh_lo'], row['zh_hi']
            
            plt.figure(figsize=(10, 7))
            
            for code, label, color, marker in TARGETS:
                if label not in processed_data: continue
                df = processed_data[label]
                
                # Slice Data
                subset = df[(df['zh_lo'] == zh_lo) & (df['zh_hi'] == zh_hi)].sort_values('pT2_center')
                if subset.empty: continue

                offset = 0
                if code == "CxC": offset = -0.02
                if code == "Sn":  offset = 0.02
                
                plt.errorbar(subset['pT2_center'] + offset, subset['MR'], yerr=subset['err_MR'], 
                             label=label, color=color, marker=marker, linestyle='-', capsize=4)

            plt.title(f"Multiplicity Ratio vs $P_{{T \pi^+}}^2$", fontsize=14)
            plt.xlabel("$P_{T \pi^+}^2$ (GeV$^2$)", fontsize=14)
            plt.ylabel("$R$", fontsize=14)
            plt.ylim(0.0, 2.0)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(fontsize=12)
            
            # Info Box
            info = (f"${q2_lo} < Q^2 < {q2_hi}$ GeV$^2$\n"
                    f"${xb_lo} < x_B < {xb_hi}$\n"
                    f"${zh_lo} < z_{{\pi^+}} < {zh_hi}$")
            plt.text(0.05, 0.95, info, transform=plt.gca().transAxes, va='top', 
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            pdf.savefig()
            plt.close()

    print(f"Done! All plots saved to {OUTPUT_PDF}")

if __name__ == "__main__":
    main()