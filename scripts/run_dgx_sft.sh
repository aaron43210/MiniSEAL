#!/bin/bash
# Qwen SFT (LoRA) on the freest GPU. Delegates to the Qwen-track dispatcher.
exec bash "$(dirname "$0")/run_dgx_qwen.sh" sft
