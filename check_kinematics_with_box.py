import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import glob
import os

# --- CONFIGURATION ---
PARQUET_PATH = "/work/clas12/suman/new_RGD_Analysis/data/LD2/018431/parquet/*.parquet"

# Cut Values for Labels
TXT_DIS = "$Q^2 > 1.0$ GeV$^2$\n$W > 2.0$ GeV\n$0.25 < y < 0.85$"
TXT_HAD = "$z_h > 0.3$\n$P_T^2 < 1.5$ GeV$^2$"

# --- CENTRAL BIN DEFINITION ---
BOX_XB = [0.3, 0.45]     # xB range
BOX_Q2 = [2.6, 4.6]      # Q2 range

def check_plots():
    # 1. Load Data
    files = glob.glob(PARQUET_PATH)
    if not files:
        print(f"No parquet files found at: {PARQUET_PATH}")
        return
    
    print(f"Found {len(files)} files. Loading and combining...")
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
    df_e_raw = df[df["w_e"] == 1]
    df_pi_raw = df[(df["w_pip"] == 1) & df["zh"].notna()]

    dis_mask = (
        (df["Q2"] > 1.0) & 
        (df["W"] > 2.0) & 
        (df["y"] > 0.25) & 
        (df["y"] < 0.85)
    )
    df_dis = df[dis_mask]
    df_e_dis = df_dis[df_dis["w_e"] == 1]

    # Hadron cuts for pion plots
    had_mask = (df_dis["zh"] > 0.3) & (df_dis["pT2"] < 1.5)
    df_pi_final = df_dis[had_mask & (df_dis["w_pip"] == 1)]

    # ---------------------------------------------------------
    # 3. PLOTTING
    # ---------------------------------------------------------
    pdf_name = 'check_kinematics_with_box.pdf'
    with PdfPages(pdf_name) as pdf:
        
        def plot_page(data_x, data_y, x_name, y_name, rng, title, cut_label, log_scale=False, draw_box=False):
            plt.figure(figsize=(8, 6))
            norm = LogNorm() if log_scale else None
            lbl = "Counts (Log)" if log_scale else "Counts (Linear)"
            
            # Plot Histogram
            plt.hist2d(data_x, data_y, bins=[200, 200], range=rng, cmap="plasma", cmin=1, norm=norm, rasterized=True)
            plt.colorbar(label=lbl)
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            plt.title(title)

            # Draw the Central Bin Rectangle (Only on Q2 vs xB plots)
            if draw_box:
                width = BOX_XB[1] - BOX_XB[0]
                height = BOX_Q2[1] - BOX_Q2[0]
                
                # CHANGED: Box color to BLACK
                rect = patches.Rectangle((BOX_XB[0], BOX_Q2[0]), width, height, 
                                         linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
                plt.gca().add_patch(rect)
                
                # Label text color is also BLACK
                range_text = f"${BOX_XB[0]} < x_B < {BOX_XB[1]}$\n${BOX_Q2[0]} < Q^2 < {BOX_Q2[1]}$"
                plt.text(BOX_XB[0], BOX_Q2[1] + 0.2, range_text, color='black', fontsize=9, fontweight='bold')

            # Info Box
            props = dict(boxstyle='round', facecolor='white', alpha=0.9)
            plt.text(0.95, 0.95, cut_label, transform=plt.gca().transAxes, fontsize=10,
                     verticalalignment='top', horizontalalignment='right', bbox=props)

            pdf.savefig()
            plt.close()

        # === PAGE 1-4: RAW DATA ===
        plot_page(df_e_raw["xB"], df_e_raw["Q2"], "$x_B$", "$Q^2$ (GeV$^2$)", [[0, 1], [0, 12]], 
                  "NO CUTS: Electron Kinematics", "No Cuts", draw_box=True)
        
        plot_page(df_pi_raw["zh"], df_pi_raw["pT2"], "$z_h$", "$P_T^2$ (GeV$^2$)", [[0, 1.2], [0, 2.5]], 
                  "NO CUTS: Pion Kinematics", "No Cuts")

        plot_page(df_e_raw["xB"], df_e_raw["Q2"], "$x_B$", "$Q^2$ (GeV$^2$)", [[0, 1], [0, 12]], 
                  "NO CUTS: Electron (Log)", "No Cuts", log_scale=True, draw_box=True)
        
        plot_page(df_pi_raw["zh"], df_pi_raw["pT2"], "$z_h$", "$P_T^2$ (GeV$^2$)", [[0, 1.2], [0, 2.5]], 
                  "NO CUTS: Pion (Log)", "No Cuts", log_scale=True)

        # === PAGE 5-8: DIS CUTS ===
        plot_page(df_e_dis["xB"], df_e_dis["Q2"], "$x_B$", "$Q^2$ (GeV$^2$)", [[0, 1], [0, 12]], 
                  "DIS CUTS: Electron Kinematics", TXT_DIS, draw_box=True)

        plot_page(df_pi_final["zh"], df_pi_final["pT2"], "$z_h$", "$P_T^2$ (GeV$^2$)", [[0, 1.2], [0, 2.5]], 
                  "FINAL CUTS: Pion Kinematics", TXT_DIS + "\n" + TXT_HAD)

        plot_page(df_e_dis["xB"], df_e_dis["Q2"], "$x_B$", "$Q^2$ (GeV$^2$)", [[0, 1], [0, 12]], 
                  "DIS CUTS: Electron (Log)", TXT_DIS, log_scale=True, draw_box=True)

        plot_page(df_pi_final["zh"], df_pi_final["pT2"], "$z_h$", "$P_T^2$ (GeV$^2$)", [[0, 1.2], [0, 2.5]], 
                  "FINAL CUTS: Pion (Log)", TXT_DIS + "\n" + TXT_HAD, log_scale=True)

    print(f"Done! Saved plots with Range Labels to {pdf_name}")

if __name__ == "__main__":
    check_plots()