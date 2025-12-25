#!/bin/bash

# --- CONFIGURATION ---
MAXJOBS=30
OUTDIR="/w/hallb-scshelf2102/clas12/suman/new_RGD_Analysis/data/LD2/018431/parquet"
TARGET="LD2"
# Make sure this path exactly matches where your ROOT files are
ROOTGLOB="/w/hallb-scshelf2102/clas12/suman/new_RGD_Analysis/data/LD2/018431/root/rec_clas_018431.evio.*.root"

# Create output directory if it doesn't exist
mkdir -p "$OUTDIR"

echo "Starting Parallel Analysis..."
echo "Input: $ROOTGLOB"
echo "Output: $OUTDIR"

start=$(date +%s)

# Loop over all files matching the pattern
for f in $ROOTGLOB; do
  
  # Check if file actually exists (prevents errors if glob fails)
  [ -e "$f" ] || continue

  echo "Submitting: $f"

  # Run Python script in the background (&)
  # REMOVED: --entry-stop 0 (so it does full file)
  # REMOVED: --yes (not needed)
  python3 run_single_sample.py --data --target "$TARGET" \
    --root-file "$f" --out-dir "$OUTDIR" &

  # Job Control: Pause if we hit MAXJOBS
  while (( $(jobs -r | wc -l) >= MAXJOBS )); do
    wait -n
  done

done

# Wait for the final batch to finish
wait

elapsed=$(( $(date +%s) - start ))
echo "All files done. Elapsed: ${elapsed}s"