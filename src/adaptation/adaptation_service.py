# src/adaptation/adaptation_service.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import math

from configs.model_config import SystemConfig, PathConfig
from src.model.llm import DecoderOnlyTransformer
from src.model.config import ModelConfig
from tokenizers import Tokenizer

app = FastAPI(
    title="Generative AI Project - SEAL Self-Adaptation",
    description="LoRA Integration & parameter efficient alignment adaptation service",
    version="1.0.0"
)

class AdaptRequest(BaseModel):
    passage: str
    target_qa_pairs: List[Dict[str, str]]
    epochs: int = 5
    lr: float = 1e-4


from src.lora.lora import inject_lora


@app.post("/adapt")
def self_adaptation_pipeline(payload: AdaptRequest):
    if not os.path.exists(PathConfig.DPO_MODEL_CHECKPOINT):
        # Fallback to SFT or Base model if DPO doesn't exist
        fallback_path = PathConfig.SFT_MODEL_CHECKPOINT if os.path.exists(PathConfig.SFT_MODEL_CHECKPOINT) else PathConfig.PRETRAINED_BASE_MODEL
        if not os.path.exists(fallback_path):
            raise HTTPException(status_code=400, detail="Base checkpoint not found. Run SFT/DPO first.")
        checkpoint_path_to_use = fallback_path
    else:
        checkpoint_path_to_use = PathConfig.DPO_MODEL_CHECKPOINT
        
    tokenizer = Tokenizer.from_file(PathConfig.TOKENIZER_PATH)
    checkpoint = torch.load(checkpoint_path_to_use, map_location=SystemConfig.DEVICE)
    model_config_dict = checkpoint.get('model_config', checkpoint.get('args'))
    model_config = ModelConfig(**model_config_dict) if isinstance(model_config_dict, dict) else model_config_dict
    
    model = DecoderOnlyTransformer(model_config)
    model_state = checkpoint.get('model', checkpoint.get('model_state_dict'))
    model.load_state_dict(model_state)
    
    # Apply Parameter Efficient Tuning
    inject_lora(model, target_keywords=["q_proj", "v_proj"], rank=16, alpha=32, dropout=0.0)
    model = model.to(SystemConfig.DEVICE)
    if SystemConfig.DEVICE == "cuda" and SystemConfig.NUM_GPUS > 1:
        model = nn.DataParallel(model)
        
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=payload.lr)
    
    training_samples = []
    for pair in payload.target_qa_pairs:
        prompt_str = f"Context fact extraction update: Question: {pair['q']} Answer:"
        target_str = f" {pair['a']} <|eos|>"
        
        enc_prompt = tokenizer.encode(prompt_str).ids
        enc_target = tokenizer.encode(target_str).ids
        
        input_ids = enc_prompt + enc_target
        labels = [-100] * len(enc_prompt) + enc_target
        
        training_samples.append((
            torch.tensor(input_ids, dtype=torch.long, device=SystemConfig.DEVICE).unsqueeze(0),
            torch.tensor(labels, dtype=torch.long, device=SystemConfig.DEVICE).unsqueeze(0)
        ))
        
    print(f"Adapting LoRA weights on {len(training_samples)} self-edits...")
    model.train()
    for epoch in range(payload.epochs):
        epoch_loss = 0.0
        for x, y in training_samples:
            optimizer.zero_grad()
            logits, _ = model(x)
            
            shift_logits = logits[..., :-1, :].contiguous().view(-1, model_config.vocab_size)
            shift_labels = y[..., 1:].contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Adaptation Epoch {epoch+1}/{payload.epochs} | Loss: {epoch_loss/len(training_samples):.4f}")
        
    state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({"model_state_dict": state_to_save, "args": model_config_dict}, PathConfig.ADAPTED_MODEL_CHECKPOINT)
    
    return {"status": "Self-Adaptation Complete", "saved_path": PathConfig.ADAPTED_MODEL_CHECKPOINT}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
