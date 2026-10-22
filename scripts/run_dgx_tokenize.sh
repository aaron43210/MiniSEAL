#!/bin/bash
# Rebuild the self-owned tokenizer + token bins for from-scratch pretraining.
# Only needed if artifacts/tokenizer/ or data/tokenized/ are missing.
set -euo pipefail
cd "$(dirname "$0")/.."
PYBIN="${PYBIN:-python3.9}"
echo "[1/2] Training BPE tokenizer (vocab 8000) on data/raw ..."
"$PYBIN" -m src.tokenizer.train_bpe \
  --input_dir data/raw --output_dir artifacts/tokenizer --vocab_size 8000
echo "[2/2] Encoding corpus -> data/tokenized ..."
"$PYBIN" -m src.tokenizer.encode_corpus \
  --input_dir data/raw \
  --tokenizer_path artifacts/tokenizer/tokenizer.json \
  --tokenizer_config_path artifacts/tokenizer/tokenizer_config.json \
  --output_dir data/tokenized
echo "Done. Token bins written to data/tokenized/."
