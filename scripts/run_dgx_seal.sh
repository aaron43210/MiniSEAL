#!/bin/bash
# Qwen SEAL self-adaptation (held-out eval, LoRA) on the freest GPU.
exec bash "$(dirname "$0")/run_dgx_qwen.sh" seal
