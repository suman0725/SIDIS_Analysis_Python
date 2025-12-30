#!/bin/bash
# submit_interactive_diag.sh - Safety Version

# --- CONFIGURATION ---
TARGET="CuSn"
RUN_NUM="018752"
BASE_DIR="/work/clas12/suman/new_RGD_Analysis/data/${TARGET}/${RUN_NUM}"
INPUT_DIR="${BASE_DIR}/root"
OUT_DIR="${BASE_DIR}/diagnostics"
LOG_DIR="${BASE_DIR}/logs"
N_PROCS=30
EVENT_LIMIT=1000000

mkdir -p "$OUT_DIR" "$LOG_DIR"

files=($(ls "${INPUT_DIR}"/*.root))
num_files=${#files[@]}

echo "================================================================"
echo "      CLAS12 RGD INTERACTIVE DIAGNOSTIC TOOL"
echo "================================================================"
for i in "${!files[@]}"; do
    printf "[%3d] %s\n" "$i" "$(basename "${files[$i]}")"
done
echo "----------------------------------------------------------------"
read -p "Select Mode (1=Single, 2=Range, 3=All): " mode_choice

selected_files=()
parallel=true

# Selection Logic
if [ "$mode_choice" == "1" ]; then
    read -p "Index: " idx; selected_files=("${files[$idx]}"); parallel=false
elif [ "$mode_choice" == "2" ]; then
    read -p "Start: " s; read -p "End: " e
    for ((i=s; i<=e; i++)); do selected_files+=("${files[$i]}"); done
elif [ "$mode_choice" == "3" ]; then
    selected_files=("${files[@]}")
else
    exit 1
fi

# --- NEW SAFETY CHECK ---
existing_count=0
for f in "${selected_files[@]}"; do
    base=$(basename "$f" .root)
    if [ -f "${OUT_DIR}/diag_e_${TARGET}_${base}.parquet" ]; then
        ((existing_count++))
    fi
done

if [ $existing_count -gt 0 ]; then
    echo "!!! WARNING: $existing_count files already exist in $OUT_DIR"
    echo "1) Overwrite (Delete old files)"
    echo "2) Skip (Only process missing files)"
    echo "3) Abort"
    read -p "Choose (1, 2, or 3): " safety_choice
    if [ "$safety_choice" == "3" ]; then echo "Aborting."; exit 1; fi
fi

echo "----------------------------------------------------------------"
echo ">>> Processing ${#selected_files[@]} file(s)..."

for root_file in "${selected_files[@]}"; do
    base_name=$(basename "$root_file" .root)
    parquet_file="${OUT_DIR}/diag_e_${TARGET}_${base_name}.parquet"

    # Skip logic if user chose option 2
    if [ "$safety_choice" == "2" ] && [ -f "$parquet_file" ]; then
        echo "Skipping existing: $base_name"
        continue
    fi
    
    if [ "$parallel" = true ]; then
        echo "Launching: $base_name"
        ./run_single_sample.py --target "$TARGET" --root-file "$root_file" \
            --out-dir "$OUT_DIR" --entry-stop $EVENT_LIMIT --diag-only > "${LOG_DIR}/${base_name}.log" 2>&1 &
        
        if [[ $(jobs -r -p | wc -l) -ge $N_PROCS ]]; then wait -n; fi
    else
        ./run_single_sample.py --target "$TARGET" --root-file "$root_file" \
            --out-dir "$OUT_DIR" --entry-stop $EVENT_LIMIT --diag-only
    fi
done

wait
echo ">>> DONE."