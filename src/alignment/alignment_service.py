# src/alignment/alignment_service.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Tuple
import uvicorn
import copy

from configs.model_config import SystemConfig, PathConfig, TrainingHyperparameters
from src.model.llm import DecoderOnlyTransformer
from src.model.config import ModelConfig
from tokenizers import Tokenizer

app = FastAPI(
    title="Generative AI Project - Alignment Service",
    description="SFT & DPO Microservice supporting multi-GPU training",
    version="1.0.0"
)

class SFTTrainRequest(BaseModel):
    sft_data: List[Dict[str, str]]
    epochs: int = 3
    lr: float = 2e-5
    batch_size: int = 2

class DPOTrainRequest(BaseModel):
    dpo_data: List[Dict[str, str]]
    epochs: int = 3
    lr: float = 5e-6
    beta: float = 0.1
    batch_size: int = 2


class SFTDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]], tokenizer: Tokenizer, max_len: int = 512):
        self.samples = []
        for item in data:
            prompt = item["prompt"]
            response = item["response"]
            
            enc_prompt = tokenizer.encode(prompt).ids
            enc_response = tokenizer.encode(response).ids + [tokenizer.token_to_id("<|eos|>")]
            
            input_ids = enc_prompt + enc_response
            labels = [-100] * len(enc_prompt) + enc_response
            
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]
            else:
                pad_len = max_len - len(input_ids)
                input_ids = input_ids + [tokenizer.token_to_id("<|pad|>")] * pad_len
                labels = labels + [-100] * pad_len
                
            self.samples.append((torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class DPODataset(Dataset):
    def __init__(self, data: List[Dict[str, str]], tokenizer: Tokenizer, max_len: int = 512):
        self.samples = []
        pad_token = tokenizer.token_to_id("<|pad|>")
        
        for item in data:
            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]
            
            enc_prompt = tokenizer.encode(prompt).ids
            enc_chosen = tokenizer.encode(chosen).ids + [tokenizer.token_to_id("<|eos|>")]
            enc_rejected = tokenizer.encode(rejected).ids + [tokenizer.token_to_id("<|eos|>")]
            
            def pad(ids, labels):
                if len(ids) > max_len:
                    return ids[:max_len], labels[:max_len]
                pad_len = max_len - len(ids)
                return ids + [pad_token] * pad_len, labels + [-100] * pad_len
            
            c_input_ids, c_labels = pad(enc_prompt + enc_chosen, [-100] * len(enc_prompt) + enc_chosen)
            r_input_ids, r_labels = pad(enc_prompt + enc_rejected, [-100] * len(enc_prompt) + enc_rejected)
            
            self.samples.append({
                "c_ids": torch.tensor(c_input_ids, dtype=torch.long),
                "c_labels": torch.tensor(c_labels, dtype=torch.long),
                "r_ids": torch.tensor(r_input_ids, dtype=torch.long),
                "r_labels": torch.tensor(r_labels, dtype=torch.long)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_mask = (shift_labels != -100)
    
    dummy_labels = shift_labels.clone()
    dummy_labels[shift_labels == -100] = 0
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_log_probs = torch.gather(log_probs, dim=-1, index=dummy_labels.unsqueeze(-1)).squeeze(-1)
    return (per_token_log_probs * loss_mask).sum(dim=-1)


def wrap_model_for_device(model: nn.Module) -> nn.Module:
    """Wraps model in DataParallel dynamically if multiple GPUs are detected."""
    model = model.to(SystemConfig.DEVICE)
    if SystemConfig.DEVICE == "cuda" and SystemConfig.NUM_GPUS > 1:
        print(f"[ACCELERATION] Wrapping model in torch.nn.DataParallel across {SystemConfig.NUM_GPUS} GPUs.")
        model = nn.DataParallel(model)
    return model


@app.post("/sft")
def train_sft(payload: SFTTrainRequest):
    if not os.path.exists(PathConfig.PRETRAINED_BASE_MODEL):
        raise HTTPException(status_code=400, detail="Base model checkpoint not found. Run pretraining first.")
    
    tokenizer = Tokenizer.from_file(PathConfig.TOKENIZER_PATH)
    checkpoint = torch.load(PathConfig.PRETRAINED_BASE_MODEL, map_location=SystemConfig.DEVICE)
    model_config_dict = checkpoint.get('model_config', checkpoint.get('args'))
    model_config = ModelConfig(**model_config_dict) if isinstance(model_config_dict, dict) else model_config_dict
    
    raw_model = DecoderOnlyTransformer(model_config)
    model_state = checkpoint.get('model', checkpoint.get('model_state_dict'))
    raw_model.load_state_dict(model_state)
    model = wrap_model_for_device(raw_model)
    model.train()
    
    dataset = SFTDataset(payload.sft_data, tokenizer, max_len=model_config.block_size)
    dataloader = DataLoader(dataset, batch_size=payload.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=payload.lr)
    
    print("Starting SFT training stage...")
    for epoch in range(payload.epochs):
        epoch_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(SystemConfig.DEVICE), labels.to(SystemConfig.DEVICE)
            optimizer.zero_grad()
            
            logits, _ = model(inputs)
            
            shift_logits = logits[..., :-1, :].contiguous().view(-1, model_config.vocab_size)
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"SFT Epoch {epoch+1}/{payload.epochs} | Loss: {epoch_loss/len(dataloader):.4f}")
        
    state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({"model_state_dict": state_to_save, "args": model_config_dict}, PathConfig.SFT_MODEL_CHECKPOINT)
    return {"status": "SFT Complete", "checkpoint_path": PathConfig.SFT_MODEL_CHECKPOINT}


@app.post("/dpo")
def train_dpo(payload: DPOTrainRequest):
    if not os.path.exists(PathConfig.SFT_MODEL_CHECKPOINT):
        raise HTTPException(status_code=400, detail="SFT checkpoint not found. Run SFT first.")
        
    tokenizer = Tokenizer.from_file(PathConfig.TOKENIZER_PATH)
    checkpoint = torch.load(PathConfig.SFT_MODEL_CHECKPOINT, map_location=SystemConfig.DEVICE)
    model_config_dict = checkpoint.get('model_config', checkpoint.get('args'))
    model_config = ModelConfig(**model_config_dict) if isinstance(model_config_dict, dict) else model_config_dict
    
    policy_raw = DecoderOnlyTransformer(model_config)
    model_state = checkpoint.get('model', checkpoint.get('model_state_dict'))
    policy_raw.load_state_dict(model_state)
    policy_model = wrap_model_for_device(policy_raw)
    policy_model.train()
    
    ref_raw = DecoderOnlyTransformer(model_config)
    ref_raw.load_state_dict(copy.deepcopy(model_state))
    ref_model = wrap_model_for_device(ref_raw)
    ref_model.eval()
    
    dataset = DPODataset(payload.dpo_data, tokenizer, max_len=model_config.block_size)
    dataloader = DataLoader(dataset, batch_size=payload.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=payload.lr)
    
    print("Starting DPO training stage...")
    for epoch in range(payload.epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            
            c_ids, c_labels = batch["c_ids"].to(SystemConfig.DEVICE), batch["c_labels"].to(SystemConfig.DEVICE)
            r_ids, r_labels = batch["r_ids"].to(SystemConfig.DEVICE), batch["r_labels"].to(SystemConfig.DEVICE)
            
            p_c_logits, _ = policy_model(c_ids)
            p_r_logits, _ = policy_model(r_ids)
            policy_chosen_logps = get_log_probs(p_c_logits, c_labels)
            policy_rejected_logps = get_log_probs(p_r_logits, r_labels)
            
            with torch.no_grad():
                ref_c_logits, _ = ref_model(c_ids)
                ref_r_logits, _ = ref_model(r_ids)
                ref_chosen_logps = get_log_probs(ref_c_logits, c_labels)
                ref_rejected_logps = get_log_probs(ref_r_logits, r_labels)
                
            pi_log_ratio = policy_chosen_logps - policy_rejected_logps
            ref_log_ratio = ref_chosen_logps - ref_rejected_logps
            
            logits_diff = pi_log_ratio - ref_log_ratio
            loss = -F.logsigmoid(payload.beta * logits_diff).mean()
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"DPO Epoch {epoch+1}/{payload.epochs} | Loss: {epoch_loss/len(dataloader):.4f}")
        
    state_to_save = policy_model.module.state_dict() if isinstance(policy_model, nn.DataParallel) else policy_model.state_dict()
    torch.save({"model_state_dict": state_to_save, "args": model_config_dict}, PathConfig.DPO_MODEL_CHECKPOINT)
    return {"status": "DPO Alignment Complete", "checkpoint_path": PathConfig.DPO_MODEL_CHECKPOINT}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
