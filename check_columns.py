import pandas as pd
import glob
import os
import sys

# Change this path to point to a SINGLE Cu or Sn file
# Use a specific file path, NOT a wildcard, for this check.
TEST_FILE_PATH = "/work/clas12/suman/new_RGD_Analysis/data/CuSn/018752/parquet/Cu/sidis_Cu_data_OB_rec_clas_018752.evio.00000-00004.parquet" 
# NOTE: You must replace 'sample_file.parquet' with an actual file name inside that Cu folder.

def print_columns(path):
    # Find one file matching the pattern (if a wildcard is accidentally used)
    files = glob.glob(path)
    if not files:
        print(f"Error: No files found at {path}")
        return

    # Use the first file found for inspection
    file_to_check = files[0]
    print(f"Inspecting columns in file: {file_to_check}")

    try:
        # Read only the header/schema for speed
        df = pd.read_parquet(file_to_check, columns=[]) 
        
        print("\n--- Available Columns ---")
        for col in sorted(df.columns):
            # Print columns that look like vertex Z columns (vz, z_v, etc.)
            if 'v' in col.lower() or 'z' in col.lower():
                print(f"  {col}")
            else:
                print(f"  {col}") # Print all columns if not too many
        print("-------------------------")
        
        print("Look for: 'vz_e', 'vze', 'vertex_z_e', or 'e_vz'")
        
    except Exception as e:
        print(f"Error reading file {file_to_check}: {e}")

if __name__ == "__main__":
    # IMPORTANT: Update the TEST_FILE_PATH above with a real file name
    print_columns(TEST_FILE_PATH) 
    # If the path above is a folder, change the line below to point to one file:
    # print_columns("/work/clas12/suman/new_RGD_Analysis/data/CuSn/018752/parquet/Cu/018752_C_200.parquet")