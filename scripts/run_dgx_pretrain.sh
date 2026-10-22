#!/bin/bash
# From-scratch pretraining of the custom ~91M decoder-only LLM on the DGX.
# Auto-detects free GPUs (>=10GB) and launches DDP across all of them.
# Needs: artifacts/tokenizer/* and data/tokenized/{train,val}.bin (already present;
# regenerate with scripts/run_dgx_tokenize.sh if missing).
set -euo pipefail
cd "$(dirname "$0")/.."
echo "=== From-scratch pretraining -> configs/pretrain_dgx_124M.yaml ==="
exec bash scripts/auto_train_dgx.sh \
  --master_port=29501 \
  -m src.train.pretrain \
  --config configs/pretrain_dgx_124M.yaml
