#!/bin/bash
# scripts/auto_train_dgx.sh
# Automatically finds GPUs with free memory and launches distributed training!

# 1. Ask nvidia-smi for all GPUs and their free memory (in MB)
# Format output: index, free_memory
# Sort by free memory (column 2) in descending order
FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -n -r)

echo "--- Current DGX GPU Status (Sorted by Free Memory) ---"
echo "$FREE_GPUS" | awk '{print "GPU " $1 " Free Memory: " $2 " MB"}'
echo "------------------------------------------------------"

# 2. Extract GPU IDs that have more than 40,000 MB (40GB) of free memory
# You can change the 40000 threshold if you need more/less memory
AVAILABLE_IDS=()
while IFS=',' read -r index mem; do
    # Trim whitespace
    index=$(echo "$index" | xargs)
    mem=$(echo "$mem" | xargs)
    
    if [ "$mem" -ge 40000 ]; then
        AVAILABLE_IDS+=("$index")
    fi
done <<< "$FREE_GPUS"

# 3. Check if we found any GPUs
NUM_GPUS=${#AVAILABLE_IDS[@]}

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs found with at least 40GB of free memory!"
    echo "The DGX server is completely full right now. Please wait for other jobs to finish."
    exit 1
fi

# 4. Join the IDs with commas (e.g., "1,2,5,7")
GPU_STRING=$(IFS=, ; echo "${AVAILABLE_IDS[*]}")

echo "SUCCESS! Found $NUM_GPUS free GPUs: $GPU_STRING"
echo "Launching training across all $NUM_GPUS GPUs..."

# 5. Launch the training dynamically!
# It will use exactly however many free GPUs it found.
export CUDA_VISIBLE_DEVICES=$GPU_STRING

# Run whatever command the user passed to this script!
python3 -m torch.distributed.run --nproc_per_node=$NUM_GPUS "$@"
