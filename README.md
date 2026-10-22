# Self-Owned LLM — From-Scratch Pretraining + Qwen Alignment (SFT · DPO · SEAL · HITL)

A capstone that (1) implements a decoder-only LLM **from scratch** in PyTorch to prove the
fundamentals end-to-end, and (2) builds a practical alignment + continual-learning stack on
top of **Qwen2.5-1.5B-Instruct** using parameter-efficient LoRA.

The repo has **two tracks**:

| Track | Purpose | Model | Status |
|-------|---------|-------|--------|
| **A. From-scratch** (learning track) | Prove the fundamentals: build + pretrain your own LLM | custom LLaMA-style decoder (~91M), own 8k BPE tokenizer | Offline — trained on the DGX; the `.pt` is gitignored, not in the repo |
| **B. Qwen alignment** (live track) | Practical SFT → DPO → SEAL + HITL governance | Qwen2.5-1.5B-Instruct + LoRA/PEFT | Runnable |

> From-scratch weights train on the DGX and are **not** committed (gitignored). The code,
> configs, tokenizer, and training logs are included so the pipeline is reproducible on a GPU.

## Environment
- **Python 3.9** on the DGX (`python3.9`); the code auto-detects CUDA / Apple `mps` / CPU.
- Install: `pip install -r requirements.txt`
- Reproducible lock (authoritative): `bash scripts/freeze_env.sh python3.9` → commit `requirements.lock.txt`.

## Track B — Qwen alignment (the live, runnable pipeline)
Run from the repo root on the DGX. The base model downloads from the HF Hub on first use.

```bash
# 1. SFT  — supervised fine-tune Qwen on instruction data
CUDA_VISIBLE_DEVICES=$FREE_GPU python3.9 -m src.train.sft      --config configs/sft_rag_duk.yaml
# 2. DPO  — preference alignment (Qwen + LoRA)
CUDA_VISIBLE_DEVICES=$FREE_GPU python3.9 -m src.train.dpo_qwen --config configs/dpo_qwen.yaml
# 3. SEAL — reward-gated self-adaptation with HELD-OUT eval
CUDA_VISIBLE_DEVICES=$FREE_GPU python3.9 -m src.train.seal     --config configs/seal_qwen.yaml
# 4. HITL — train a LoRA ONLY on human-APPROVED data
CUDA_VISIBLE_DEVICES=$FREE_GPU python3.9 -m src.train.hitl     --config configs/hitl_qwen.yaml
```
Each writes a LoRA adapter under `model_weights/`; SEAL also writes `runs/self_adapt/results.json`.

### HITL review UI + API
```bash
bash scripts/run_hitl_server.sh      # serves src/hitl/api.py + ui/ dashboard on :8543
```
Flow: Qwen drafts answers → `RiskValidator` risk-screens → quarantine SQLite DB → a human
approves in the UI → `src/train/hitl.py` trains a LoRA only on APPROVED rows (every job is
audited in the `adapter_jobs` table).

### RAG demo
```bash
python3.9 scripts/run_rag.py         # retrieval over data/duk_data.txt + Qwen generation
```

## Track A — From-scratch model (learning track, offline)
Demonstrates building + training an LLM from zero. Requires a GPU and produces a `.pt`
checkpoint that is **not** committed. Run on the DGX:

```bash
# 1. Train the BPE tokenizer (vocab 8000) on the raw corpus directory
python3.9 -m src.tokenizer.train_bpe \
    --input_dir data/raw --output_dir artifacts/tokenizer --vocab_size 8000
# 2. Encode the corpus into token bins
python3.9 -m src.tokenizer.encode_corpus \
    --input_dir data/raw \
    --tokenizer_path artifacts/tokenizer/tokenizer.json \
    --tokenizer_config_path artifacts/tokenizer/tokenizer_config.json \
    --output_dir data/tokenized
# 3. Pretrain the decoder-only transformer
python3.9 -m src.train.pretrain --config configs/pretrain_dgx_124M.yaml
```
Architecture: LLaMA-style decoder (RMSNorm, RoPE, SwiGLU, SDPA, weight tying), ~91M params
at 12L / 768d / 12h, with the 8k-token custom tokenizer. See `src/model/`.

> **Note:** the pretrain configs read token bins from `data/bin/`; the encode step above
> writes to `data/tokenized/`. Point `data.*` in `configs/pretrain_dgx_124M.yaml` at
> `data/tokenized/` (or encode into `data/bin/`) before running.
>
> The custom-model FastAPI services (`src/alignment`, `src/adaptation`,
> `services/model_service`, `services/pretrain_service`) and the gateway pipeline belong to
> this offline track and require a trained `.pt`; they are **not** part of the live Qwen
> runtime (see `docker-compose.yml`).

## Docker (live services only)
```bash
docker-compose up --build -d         # tokenizer-service + HITL service (Qwen)
```

## Tests
```bash
python3.9 -m pytest tests/           # RiskValidator, training math, LoRA round-trip
```

## Repo map
- `src/model/`, `src/train/pretrain.py`, `src/tokenizer/` — **Track A** (from-scratch)
- `src/train/{sft,dpo_qwen,seal,hitl}.py`, `src/hitl/`, `scripts/run_rag.py` — **Track B** (Qwen)
- `configs/*.yaml` — one config per stage
- `PLAN.md` — hardening roadmap (held-out SEAL eval, dependency pinning, architecture unification)
- `src/alignment/`, `src/adaptation/`, `services/`, `src/api/` — custom-model offline track + services
