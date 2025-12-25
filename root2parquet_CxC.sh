#!/bin/bash

# --- CONFIGURATION ---
MAXJOBS=20  # Don't go too high on the interactive node!
TARGET="CxC"
BASE_DIR="/w/hallb-scshelf2102/clas12/suman/new_RGD_Analysis/data/CxC"

# List of Run folders to process
# You can add more here later (e.g. "018341" "018342")
RUN_LIST=("018339" "018340")

echo "Starting Parallel Analysis for target: $TARGET"
start_total=$(date +%s)

# 1. Loop over each Run Folder
for RUN in "${RUN_LIST[@]}"; do
    
    # Define paths for this specific run
    ROOT_DIR="${BASE_DIR}/${RUN}/root"
    OUT_DIR="${BASE_DIR}/${RUN}/parquet"
    
    # Create output directory if it doesn't exist (fixes 'parquest' typo automatically)
    if [ ! -d "$OUT_DIR" ]; then
        echo "Creating output dir: $OUT_DIR"
        mkdir -p "$OUT_DIR"
    fi

    echo "------------------------------------------------"
    echo "Processing Run: $RUN"
    echo "Input:  $ROOT_DIR"
    echo "Output: $OUT_DIR"
    echo "------------------------------------------------"

    # 2. Find all ROOT files in this run folder
    # We use a wildcard *.root to capture all parts (00000-00004, etc.)
    for f in "$ROOT_DIR"/*.root; do
        
        # Safety check: make sure the file exists
        [ -e "$f" ] || continue

        echo "Submitting: $(basename "$f")"

        # 3. Run Python Script in Background
        # Note: We pass the specific output dir for THIS run
        python3 run_single_sample.py --data --target "$TARGET" \
            --root-file "$f" --out-dir "$OUT_DIR" &

        # 4. Job Control (Pause if we hit MAXJOBS)
        while (( $(jobs -r | wc -l) >= MAXJOBS )); do
            sleep 1
        done

    done
done

# Wait for all remaining jobs to finish
echo "Waiting for final jobs to complete..."
wait

elapsed=$(( $(date +%s) - start_total ))
echo "✅ All Runs Done. Total time: ${elapsed}s"