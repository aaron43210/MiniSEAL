#!/bin/bash
# Run a Qwen-track LoRA stage on the single freest GPU.
# Usage: bash scripts/run_dgx_qwen.sh {sft|dpo|seal|hitl}
#   (override interpreter with PYBIN=..., default python3.9)
set -euo pipefail
cd "$(dirname "$0")/.."
PYBIN="${PYBIN:-python3.9}"
STAGE="${1:-}"
case "$STAGE" in
  sft)  MOD=src.train.sft;      CFG=configs/sft_rag_duk.yaml ;;
  dpo)  MOD=src.train.dpo_qwen; CFG=configs/dpo_qwen.yaml ;;
  seal) MOD=src.train.seal;     CFG=configs/seal_qwen.yaml ;;
  hitl) MOD=src.train.hitl;     CFG=configs/hitl_qwen.yaml ;;
  *) echo "Usage: bash scripts/run_dgx_qwen.sh {sft|dpo|seal|hitl}"; exit 1 ;;
esac
# Pick the GPU with the most free memory (these LoRA jobs are single-GPU).
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -n -r | head -1 | cut -d, -f1 | xargs)
echo "[run_dgx_qwen] stage=$STAGE -> $MOD ($CFG) on GPU $GPU"
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES="$GPU" "$PYBIN" -m "$MOD" --config "$CFG"
