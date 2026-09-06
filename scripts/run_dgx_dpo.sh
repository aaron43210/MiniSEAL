#!/bin/bash
# DPO launch script for DGX A100

echo "=== STARTING DIRECT PREFERENCE OPTIMISATION (DPO) ==="
echo ""
echo "*****************************************"
echo "Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed."
echo "*****************************************"

# Change to the root directory of the project
cd "$(dirname "$0")/.."

# We use both A100 GPUs (3,4) exactly as we did for pretraining and SFT
CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.run \
  --nproc_per_node=2 \
  -m src.train.dpo
