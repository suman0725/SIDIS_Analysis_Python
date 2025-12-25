import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
import glob
import os

# --- CONFIGURATION (Full Run) ---
# CHANGED: Back to wildcard to grab ALL files
PARQUET_PATH = "/work/clas12/suman/new_RGD_Analysis/data/LD2/018431/parquet/*.parquet"

# Cut Values for Labels (With Units)
TXT_DIS = "$Q^2 > 1.0$ GeV$^2$\n$W > 2.0$ GeV\n$0.25 < y < 0.85$"
TXT_HAD = "$z_h > 0.3$\n$P_T^2 < 1.5$ GeV$^2$"

def check_plots():
    # 1. Load Data
    files = glob.glob(PARQUET_PATH)
    if not files:
        print(f"No parquet files found at: {PARQUET_PATH}")
        return
    
    print(f"Found {len(files)} files. Loading and combining...")
    
    # Load ALL files and combine them
    try:
        df_list = [pd.read_parquet(f) for f in files]
        df = pd.concat(df_list, ignore_index=True)
    except Exception as e:
        print(f"Error reading files: {e}")
        return

    print(f"Total Combined Events: {len(df)}")

    # ---------------------------------------------------------
    # 2. DEFINE SUBSETS
    # ---------------------------------------------------------
    
    # --- A. NO CUTS (Raw) ---
    df_e_raw = df[df["w_e"] == 1]
    df_pi_raw = df[(df["w_pip"] == 1) & df["zh"].notna()]

    # --- B. DIS CUTS ONLY (For Electron Plots) ---
    # Q2 > 1, W > 2, 0.25 < y < 0.85
    dis_mask = (
        (df["Q2"] > 1.0) & 
        (df["W"] > 2.0) & 
        (df["y"] > 0.25) & 
        (df["y"] < 0.85)
    )
    df_dis = df[dis_mask]
    df_e_dis = df_dis[df_dis["w_e"] == 1]

    # --- C. DIS + HADRON CUTS (For Pion Plots) ---
    # Apply DIS cuts first, THEN apply zh > 0.3 and pT2 < 1.5
    had_mask = (
        (df_dis["zh"] > 0.3) & 
        (df_dis["pT2"] < 1.5)
    )
    df_pi_final = df_dis[had_mask & (df_dis["w_pip"] == 1)]

    print(f"Events (Raw e-): {len(df_e_raw)}")
    print(f"Events (DIS e-): {len(df_e_dis)}")
    print(f"Events (Final pi+): {len(df_pi_final)}")

    # ---------------------------------------------------------
    # 3. PLOTTING
    # ---------------------------------------------------------
    pdf_name = 'check_kinematics_full_run.pdf'
    with PdfPages(pdf_name) as pdf:
        
        def plot_page(data_x, data_y, x_name, y_name, rng, title, cut_label, log_scale=False):
            plt.figure(figsize=(8, 6))
            norm = LogNorm() if log_scale else None
            lbl = "Counts (Log)" if log_scale else "Counts (Linear)"
            
            # 2D Hist with rasterized=True for smooth PDF
            plt.hist2d(data_x, data_y, bins=[200, 200], range=rng, cmap="plasma", cmin=1, norm=norm, rasterized=True)
            plt.colorbar(label=lbl)
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            plt.title(title)

            # Add Cut Label Box
            props = dict(boxstyle='round', facecolor='white', alpha=0.9)
            plt.text(0.95, 0.95, cut_label, transform=plt.gca().transAxes, fontsize=10,
                     verticalalignment='top', horizontalalignment='right', bbox=props)

            pdf.savefig()
            plt.close()

        # === PAGE 1-4: RAW DATA (No Cuts) ===
        plot_page(df_e_raw["xB"], df_e_raw["Q2"], "$x_B$", "$Q^2$ (GeV$^2$)", [[0, 1], [0, 12]], 
                  "NO CUTS: Electron Kinematics", "No Cuts")
        
        plot_page(df_pi_raw["zh"], df_pi_raw["pT2"], "$z_h$", "$P_T^2$ (GeV$^2$)", [[0, 1.2], [0, 2.5]], 
                  "NO CUTS: Pion Kinematics", "No Cuts")

        plot_page(df_e_raw["xB"], df_e_raw["Q2"], "$x_B$", "$Q^2$ (GeV$^2$)", [[0, 1], [0, 12]], 
                  "NO CUTS: Electron (Log)", "No Cuts", log_scale=True)
        
        plot_page(df_pi_raw["zh"], df_pi_raw["pT2"], "$z_h$", "$P_T^2$ (GeV$^2$)", [[0, 1.2], [0, 2.5]], 
                  "NO CUTS: Pion (Log)", "No Cuts", log_scale=True)

        # === PAGE 5-8: APPLIED CUTS ===
        
        # 5. Electron Linear (DIS Cuts Only)
        plot_page(df_e_dis["xB"], df_e_dis["Q2"], "$x_B$", "$Q^2$ (GeV$^2$)", [[0, 1], [0, 12]], 
                  "DIS CUTS: Electron Kinematics", TXT_DIS)

        # 6. Pion Linear (DIS + Hadron Cuts)
        plot_page(df_pi_final["zh"], df_pi_final["pT2"], "$z_h$", "$P_T^2$ (GeV$^2$)", [[0, 1.2], [0, 2.5]], 
                  "FINAL CUTS: Pion Kinematics", TXT_DIS + "\n" + TXT_HAD)

        # 7. Electron Log (DIS Cuts Only)
        plot_page(df_e_dis["xB"], df_e_dis["Q2"], "$x_B$", "$Q^2$ (GeV$^2$)", [[0, 1], [0, 12]], 
                  "DIS CUTS: Electron (Log)", TXT_DIS, log_scale=True)

        # 8. Pion Log (DIS + Hadron Cuts)
        plot_page(df_pi_final["zh"], df_pi_final["pT2"], "$z_h$", "$P_T^2$ (GeV$^2$)", [[0, 1.2], [0, 2.5]], 
                  "FINAL CUTS: Pion (Log)", TXT_DIS + "\n" + TXT_HAD, log_scale=True)

    print(f"Done! Saved plots to {pdf_name}")

if __name__ == "__main__":
    check_plots()