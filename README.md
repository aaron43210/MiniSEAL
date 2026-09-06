# Self-Owned Language Model: End-to-End Private Pipeline
An industrial-grade, full-stack, decoder-only LLM pipeline built from scratch. This project demonstrates pretraining on raw data, alignment via **SFT + DPO**, and parameter-efficient, continous learning using **SEAL-style Self-Adaptation with LoRA**.

---

##  Compute & Environment Configuration
The architecture runs on an integrated configuration system (`configs/model_config.py`). It dynamically auto-detects hardware resources (Apple Silicon `mps`, single/multi-GPU `CUDA`, or fallback `CPU`) and handles multi-GPU model wrapping via `torch.nn.DataParallel` automatically.

---

##  Execution Steps

### 1. Installation & Environment Setup
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Fetch Real-World Data
Acquire, clean, and format gold-standard evaluation datasets (WikiText-2, Stanford Alpaca, Orca DPO, and SQuAD v1.1) automatically:
```bash
python scripts/download_real_data.py
```

### 3. Step-by-Step Training Order
Run each phase sequentially to train and align your model:

* **Phase 1: Tokenizer Training**
  Train a custom Byte-Pair Encoding (BPE) tokenizer on your corpus:
  ```bash
  python src/tokenizer/tokenizer_service.py
  ```

* **Phase 3: Base Pretraining**
  Pretrain your decoder-only transformer with causal sequence modeling:
  ```bash
  python src/pretrain/pretrain_service.py
  ```

* **Phase 4: SFT & DPO Alignment**
  Finetune and align the pretrained weights on instruction/preference sets:
  ```bash
  python src/alignment/alignment_service.py
  ```

* **Phase 5: SEAL Self-Adaptation**
  Incorporate new knowledge continuously using parameterized LoRA adapters:
  ```bash
  python src/adaptation/adaptation_service.py
  ```

---

##  Docker Orchestration

You can spin up your entire microservices architecture with a single command:
```bash
docker-compose up --build -d
```

### Verify Container Health
Once booted, query the API gateway check endpoint:
```bash
curl http://localhost:8080/health
```

### Trigger the Full Pipeline End-to-End
```bash
curl -X POST http://localhost:8080/pipeline/run \
-H "Content-Type: application/json" \
-d '{
  "pretrain_corpus_paths": ["data/raw/wikitext-2.txt"],
  "sft_data": [
    {"prompt": "What is reinforcement learning?", "response": "A machine learning method based on rewarding desired actions."}
  ],
  "dpo_data": [
    {"prompt": "Describe machine learning.", "chosen": "A field focused on learning patterns from datasets.", "rejected": "It is just basic database filtering."}
  ],
  "adaptation_passage": "The James Webb Space Telescope launched on December 25, 2021.",
  "adaptation_qa_pairs": [
    {"q": "When did JWST launch?", "a": "December 25, 2021"}
  ]
}'
```
