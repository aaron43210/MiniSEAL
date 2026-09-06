# src/hitl/approved_lora_train.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.llm import DecoderOnlyTransformer
from src.lora.lora import inject_lora, get_lora_state_dict
from src.hitl.db import QuarantineDatabase
from tokenizers import Tokenizer
import time

def execute_approved_training(db_path: str, base_model_path: str, tokenizer_path: str, output_dir: str, epochs: int, lr: float):
    # Determine device
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    db = QuarantineDatabase(db_path)
    approved_proposals = db.list_proposals(status="APPROVED")
    
    if not approved_proposals:
        print("[TRAINING] No APPROVED proposals found. Execution skipped.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    checkpoint = torch.load(base_model_path, map_location=device)
    model_args = checkpoint['args'] if 'args' in checkpoint else checkpoint['model_config']
    
    # 1. Load the core model and inject LoRA parameters
    from src.model.config import ModelConfig
    config_obj = ModelConfig(**model_args)
    model = DecoderOnlyTransformer(config_obj)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint['model'])
    inject_lora(model, target_keywords=["q_proj", "v_proj"], rank=16, alpha=32, dropout=0.0)
    model = model.to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    
    # 2. Tokenize approved samples
    training_samples = []
    for row in approved_proposals:
        prompt_str = f"Context fact extraction update: Question: {row['question']} Answer:"
        target_str = f" {row['proposed_answer']} <|eos|>"
        
        input_ids = tokenizer.encode(prompt_str).ids + tokenizer.encode(target_str).ids
        labels = [-100] * len(tokenizer.encode(prompt_str).ids) + tokenizer.encode(target_str).ids
        
        training_samples.append((
            torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0),
            torch.tensor(labels, dtype=torch.long, device=device).unsqueeze(0)
        ))

    # 3. Parameter-isolated training loop
    model.train()
    print(f"[TRAINING] Optimizing {len(trainable_params)} LoRA parameters over {len(training_samples)} approved facts...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in training_samples:
            optimizer.zero_grad()
            logits, _ = model(x)
            
            shift_logits = logits[..., :-1, :].contiguous().view(-1, config_obj.vocab_size)
            shift_labels = y[..., 1:].contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Run Loss: {epoch_loss/len(training_samples):.4f}")

    # 4. Extract and save ONLY the low-rank delta parameters
    lora_deltas = get_lora_state_dict(model)
    version_id = int(time.time())
    adapter_filename = f"lora_adapter_v{version_id}.pt"
    adapter_filepath = os.path.join(output_dir, adapter_filename)
    
    torch.save(lora_deltas, adapter_filepath)
    print(f"[TRAINING] Exported versioned adapter weights directly to: {adapter_filepath}")
    
    # Update status states inside DB
    for row in approved_proposals:
        db.update_status(row["id"], "TRAINED")
        
    return adapter_filepath
