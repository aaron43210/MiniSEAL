#!/bin/bash
# Submit script or local launch script for DGX A100

echo "Starting distributed pretraining on DGX A100..."

# Example assuming 8 GPUs on a single node
python -m torch.distributed.run --nproc_per_node=8 \
  -m src.train.pretrain \
  --config configs/pretrain_dgx_a100.yaml

echo "Training complete!"
