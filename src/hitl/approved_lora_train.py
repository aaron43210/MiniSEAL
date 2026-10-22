# src/hitl/approved_lora_train.py

import os
import time
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from src.hitl.db import QuarantineDatabase

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

def execute_approved_training(db_path: str, base_model_path: str, output_dir: str, epochs: int, lr: float):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    db = QuarantineDatabase(db_path)
    approved_proposals = db.list_proposals(status="APPROVED")

    if not approved_proposals:
        print("[TRAINING] No APPROVED proposals found. Execution skipped.")
        return None

    job_id = db.record_adapter_job(job_status="RUNNING")

    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Load HuggingFace Tokenizer and Model
        print(f"[TRAINING] Loading base model from {base_model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        if not torch.cuda.is_available():
            model = model.to(device)

        # Inject LoRA using PEFT
        config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        # Tokenize approved samples
        training_samples = []
        for row in approved_proposals:
            messages = [
                {"role": "user", "content": f"Context fact extraction update: Question: {row['question']}"},
                {"role": "assistant", "content": row['proposed_answer']}
            ]
            try:
                prompt_str = tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception:
                prompt_str = f"User: {row['question']}\nAssistant: {row['proposed_answer']}{tokenizer.eos_token}"
            
            enc = tokenizer(prompt_str, return_tensors="pt")
            input_ids = enc.input_ids.to(model.device)
            labels = input_ids.clone()
            training_samples.append((input_ids, labels))

        # Parameter-isolated training loop
        model.train()
        print(f"[TRAINING] Optimizing LoRA parameters over {len(training_samples)} approved facts...")
        last_epoch_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in training_samples:
                optimizer.zero_grad()
                outputs = model(input_ids=x, labels=y)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            last_epoch_loss = epoch_loss / len(training_samples)
            print(f"Epoch {epoch+1}/{epochs} | Run Loss: {last_epoch_loss:.4f}")

        # Save PEFT adapter
        version_id = int(time.time())
        adapter_dirname = f"lora_adapter_v{version_id}"
        adapter_filepath = os.path.join(output_dir, adapter_dirname)
        
        model.save_pretrained(adapter_filepath)
        
        # Save metadata
        meta = {
            "source": "hitl_approved_training",
            "base_model": base_model_path,
            "num_facts": len(training_samples),
            "epochs": epochs,
            "lr": lr,
            "final_loss": round(last_epoch_loss, 4),
            "version_id": version_id,
        }
        with open(os.path.join(adapter_filepath, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[TRAINING] Exported versioned PEFT adapter weights to: {adapter_filepath}")

        # Update status states inside DB
        for row in approved_proposals:
            db.update_status(row["id"], "TRAINED")
        db.update_adapter_job(job_id, "SUCCESS", run_loss=round(last_epoch_loss, 4), adapter_path=adapter_filepath)

        return adapter_filepath

    except Exception as e:
        db.update_adapter_job(job_id, "FAILED", run_loss=None, adapter_path=None)
        print(f"[TRAINING] Job {job_id} FAILED: {e}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a HuggingFace LoRA adapter on APPROVED HITL proposals.")
    parser.add_argument("--db", required=True, help="Path to the quarantine SQLite DB.")
    parser.add_argument("--base-model", required=True, help="HuggingFace Base Model path.")
    parser.add_argument("--output", default=os.path.join("ckpts", "adapters"), help="Directory to write the adapter into.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    path = execute_approved_training(
        db_path=args.db,
        base_model_path=args.base_model,
        output_dir=args.output,
        epochs=args.epochs,
        lr=args.lr,
    )
    if path:
        print(f"[TRAINING] Adapter written to {path}")
    else:
        print("[TRAINING] Nothing trained (no approved proposals).")
