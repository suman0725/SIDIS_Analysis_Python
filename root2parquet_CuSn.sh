#!/bin/bash

# --- CONFIGURATION ---
MAXJOBS=20
# Path from your screenshot
BASE_DIR="/w/hallb-scshelf2102/clas12/suman/new_RGD_Analysis/data/CuSn"

# List of Run folders (Add more if you have them, e.g. "018752" "018753")
RUN_LIST=("018752")

echo "Starting Parallel Analysis for Dual Targets (Cu & Sn)..."
start_total=$(date +%s)

# 1. Loop over each Run Folder
for RUN in "${RUN_LIST[@]}"; do
    
    ROOT_DIR="${BASE_DIR}/${RUN}/root"
    
    # 2. Define Separate Output Directories
    OUT_DIR_CU="${BASE_DIR}/${RUN}/parquet/Cu"
    OUT_DIR_SN="${BASE_DIR}/${RUN}/parquet/Sn"
    
    # Create them
    mkdir -p "$OUT_DIR_CU"
    mkdir -p "$OUT_DIR_SN"

    echo "------------------------------------------------"
    echo "Processing Run: $RUN"
    echo "  - Input: $ROOT_DIR"
    echo "  - Output Cu: $OUT_DIR_CU"
    echo "  - Output Sn: $OUT_DIR_SN"
    echo "------------------------------------------------"

    # 3. Loop over ROOT files
    for f in "$ROOT_DIR"/*.root; do
        
        # Safety check
        [ -e "$f" ] || continue
        fname=$(basename "$f")
        
        # --- SUBMIT JOB 1: COPPER (Cu) ---
        # "Hey Python, find the vertices in the Copper region"
        python3 run_single_sample.py --data --target "Cu" \
            --root-file "$f" --out-dir "$OUT_DIR_CU" &
        
        # Check Job Limit
        while (( $(jobs -r | wc -l) >= MAXJOBS )); do sleep 1; done

        # --- SUBMIT JOB 2: TIN (Sn) ---
        # "Hey Python, find the vertices in the Tin region"
        python3 run_single_sample.py --data --target "Sn" \
            --root-file "$f" --out-dir "$OUT_DIR_SN" &

        # Check Job Limit
        while (( $(jobs -r | wc -l) >= MAXJOBS )); do sleep 1; done
        
    done
done

# Wait for everything to finish
echo "Waiting for final jobs..."
wait

elapsed=$(( $(date +%s) - start_total ))
echo "✅ All Cu & Sn files done. Total time: ${elapsed}s"