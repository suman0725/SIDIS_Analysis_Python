import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import glob
import os
import sys

# --- CONFIGURATION ---
TARGET_PATHS = {
    "LD2":  "/work/clas12/suman/new_RGD_Analysis/data/LD2/018431/parquet/*.parquet",
    "CxC":  "/work/clas12/suman/new_RGD_Analysis/data/CxC/*/parquet/*.parquet",
    "Cu":   "/work/clas12/suman/new_RGD_Analysis/data/CuSn/*/parquet/Cu/*.parquet",
    "Sn":   "/work/clas12/suman/new_RGD_Analysis/data/CuSn/*/parquet/Sn/*.parquet"
}

# Dictionary to map targets to their specific vz column name
# This handles the case where LD2/CxC uses 'vz' but Cu/Sn uses 'vz_e'
VZ_COLUMN_MAP = {
    "LD2": "vz",
    "CxC": "vz",
    "Cu":  "vz_e",  # <--- REPLACE THIS IF DIFFERENT
    "Sn":  "vz_e"   # <--- REPLACE THIS IF DIFFERENT
}

OUTPUT_PDF = "vz_check_all_targets_FINAL.pdf"

def load_and_plot_vz():
    print(f"Starting Vz check. Output PDF: {OUTPUT_PDF}")
    
    with PdfPages(OUTPUT_PDF) as pdf:
        
        for target_name, path_pattern in TARGET_PATHS.items():
            print(f"\n--- Loading {target_name} ---")
            
            files = glob.glob(path_pattern)
            if not files:
                print(f"Warning: No files found for {target_name}.")
                continue
            
            try:
                # Load only the electron rows (w_e == 1) and the necessary column
                cols_to_load = list(set(['w_e', VZ_COLUMN_MAP[target_name]]))
                df_list = [pd.read_parquet(f, columns=cols_to_load) for f in files]
                df = pd.concat(df_list, ignore_index=True)
            except Exception as e:
                print(f"Error reading files for {target_name}: {e}")
                continue

            # Get the correct column name for this specific target
            vz_col_name = VZ_COLUMN_MAP.get(target_name)

            df_e = df[df["w_e"] == 1]
            if vz_col_name not in df_e.columns:
                 print(f"Error: Column '{vz_col_name}' not found in data for {target_name}. Skipping.")
                 continue

            vz_data = df_e[vz_col_name].dropna()
            
            # --- Plotting ---
            plt.figure(figsize=(10, 7))
            
            plt.hist(vz_data, bins=200, range=[-20, 20], color='blue', alpha=0.7, log=True)
            
            plt.title(f"Electron Vertex Z-Position ({vz_col_name}) for {target_name}", fontsize=16)
            plt.xlabel("$v_z$ (cm)", fontsize=14)
            plt.ylabel("Counts (Log Scale)", fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            
            plt.axvline(x=0, color='red', linestyle='-', linewidth=1.5, label='Center of Target Region')
            plt.legend()

            pdf.savefig()
            plt.close()

    print(f"\nSuccessfully created PDF with Vz plots for all targets: {OUTPUT_PDF}")

if __name__ == "__main__":
    load_and_plot_vz()