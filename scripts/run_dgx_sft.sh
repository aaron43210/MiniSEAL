#!/bin/bash

# DGX Multi-GPU SFT Launcher
# Usage: ./scripts/run_dgx_sft.sh

echo "=== STARTING SUPERVISED FINE-TUNING (SFT) ==="

# Change to the root directory of the project
cd "$(dirname "$0")/.."

# We use both A100 GPUs (3,4) exactly as we did for pretraining
CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port=29511 \
  -m src.train.sft \
  --config configs/sft_dgx_124M.yaml
