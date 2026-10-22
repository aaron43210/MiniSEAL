#!/bin/bash
# Qwen DPO (preference alignment, LoRA) on the freest GPU.
# NOTE: this now runs the Qwen-track DPO (src.train.dpo_qwen), not the old
# from-scratch src.train.dpo. The from-scratch DPO stays available via
#   python3.9 -m src.train.dpo --config configs/dpo_dgx_124M.yaml
exec bash "$(dirname "$0")/run_dgx_qwen.sh" dpo
