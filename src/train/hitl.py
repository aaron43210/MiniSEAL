# src/train/hitl.py
#
# HITL (Human-In-The-Loop) LoRA trainer.
#
# In this project, HITL training means we ONLY ever fine-tune a LoRA adapter on
# data a human has explicitly APPROVED -- never on raw or unreviewed model
# output. This module is the config-driven entry point and is DB-first: if the
# quarantine SQLite store has APPROVED proposals we reuse the canonical loop in
# src/hitl/approved_lora_train.py; otherwise we fall back to a simple JSONL of
# approved {"prompt","response"} pairs. If no approved data exists anywhere we
# refuse to train and tell the user how to seed it.
#
# Run:  python3 -m src.train.hitl --config configs/hitl_qwen.yaml

import argparse
import os
import sys
import json
import yaml
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Add project root to python path (mirrors src/train/sft.py lines 11-13)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common.utils import ensure_dir, set_seed
from src.hitl.db import QuarantineDatabase
from src.hitl.approved_lora_train import execute_approved_training

# LoRA target modules match the canonical approved_lora_train.py loop
LORA_TARGET_MODULES = ["q_proj", "v_proj"]


def _load_jsonl_pairs(data_path):
    """Read approved pairs from a JSONL file.

    Each line is a JSON object {"prompt": str, "response": str}; the aliases
    {"question": ..., "answer": ...} are also accepted. Returns a list of
    (prompt, response) tuples, skipping blank or malformed lines.
    """
    pairs = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("prompt", row.get("question"))
            response = row.get("response", row.get("answer"))
            if prompt is None or response is None:
                continue
            pairs.append((prompt, response))
    return pairs

def _train_on_pairs(pairs, cfg, out_dir):
    """Inline LoRA loop for a JSONL of approved pairs (DB-less fallback).

    Mirrors the canonical loop in src/hitl/approved_lora_train.py so adapters are
    produced identically whether they originate from the quarantine DB or a file.
    """
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    base_model_path = cfg.get("pretrained_model_path", "Qwen/Qwen2.5-1.5B-Instruct")
    epochs = int(cfg.get("epochs", 5))
    lr = float(cfg.get("learning_rate", 1e-4))

    print(f"[HITL] Loading base model from {base_model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = model.to(device)

    config = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    # Tokenize each approved pair with the chat template (prompt + response as
    # one sequence; labels == input_ids, matching approved_lora_train.py).
    samples = []
    for prompt, response in pairs:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            text = f"User: {prompt}\nAssistant: {response}{tokenizer.eos_token}"
        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc.input_ids.to(device)
        labels = input_ids.clone()
        samples.append((input_ids, labels))

    model.train()
    print(f"[HITL] Optimizing LoRA parameters over {len(samples)} approved pairs...")
    last_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in samples:
            optimizer.zero_grad()
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        last_loss = epoch_loss / max(len(samples), 1)
        print(f"[HITL] Epoch {epoch + 1}/{epochs} | Loss: {last_loss:.4f}")

    version_id = int(time.time())
    adapter_filepath = os.path.join(out_dir, f"lora_adapter_v{version_id}")
    ensure_dir(adapter_filepath)
    model.save_pretrained(adapter_filepath)
    tokenizer.save_pretrained(adapter_filepath)

    meta = {
        "source": "hitl_jsonl",
        "base_model": base_model_path,
        "num_facts": len(samples),
        "epochs": epochs,
        "lr": lr,
        "final_loss": round(last_loss, 4),
        "version_id": version_id,
    }
    with open(os.path.join(adapter_filepath, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return adapter_filepath


def main():
    parser = argparse.ArgumentParser(description="HITL LoRA trainer (train only on human-approved data).")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    out_dir = cfg["out_dir"]
    ensure_dir(out_dir)

    base_model_path = cfg.get("pretrained_model_path", "Qwen/Qwen2.5-1.5B-Instruct")
    db_path = cfg.get("hitl_db_path", "data/quarantine_store.db")
    data_path = cfg.get("hitl_data_path", "data/hitl/approved_feedback.jsonl")
    epochs = int(cfg.get("epochs", 5))
    lr = float(cfg.get("learning_rate", 1e-4))

    # --- Source selection: real HITL (DB-approved proposals) first ---
    approved_in_db = []
    if os.path.exists(db_path):
        try:
            approved_in_db = QuarantineDatabase(db_path).list_proposals("APPROVED")
        except Exception as e:
            print(f"[HITL] Could not read quarantine DB at {db_path}: {e}")

    if approved_in_db:
        print(f"[HITL] Found {len(approved_in_db)} APPROVED proposal(s) in {db_path}. "
              f"Training via the canonical approved_lora_train loop.")
        adapter_path = execute_approved_training(
            db_path=db_path,
            base_model_path=base_model_path,
            output_dir=out_dir,
            epochs=epochs,
            lr=lr,
        )
        print(f"[HITL] Adapter written to {adapter_path}")
        return

    # --- Fallback: a JSONL of human-approved {"prompt","response"} pairs ---
    if os.path.exists(data_path):
        pairs = _load_jsonl_pairs(data_path)
        if not pairs:
            print(f"[HITL] {data_path} exists but has no valid pairs. Nothing to train.")
            sys.exit(1)
        print(f"[HITL] No APPROVED proposals in DB. Training on {len(pairs)} approved "
              f"pair(s) from {data_path}.")
        adapter_path = _train_on_pairs(pairs, cfg, out_dir)
        print(f"[HITL] Adapter written to {adapter_path}")
        return

    # --- Nothing approved anywhere: refuse to train, explain how to seed ---
    print("[HITL] No approved training data found; refusing to train.")
    print(f"       - No APPROVED proposals in quarantine DB: {db_path}")
    print(f"       - No approved JSONL file at: {data_path}")
    print("       Seed data by (a) approving proposals via the HITL review API")
    print("       (src/hitl/api.py), or (b) creating the JSONL with lines like:")
    print('         {"prompt": "...", "response": "..."}')
    sys.exit(1)


if __name__ == "__main__":
    main()
