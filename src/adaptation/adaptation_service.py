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


@app.get("/health")
def health():
    """Lightweight liveness probe used by the gateway."""
    return {"status": "ok", "service": "adaptation"}

class AdaptRequest(BaseModel):
    passage: str
    target_qa_pairs: List[Dict[str, str]]
    epochs: int = 5
    lr: float = 1e-4
    # Minimum reward improvement (adapted - base) required to accept the self-edit.
    accept_margin: float = 0.0
    # If True, run the HITL RiskValidator over every self-edit and refuse to
    # accept the adapter when any QA pair is HIGH risk, even if reward improves.
    enforce_safety: bool = True


from src.lora.lora import inject_lora, save_lora_adapter

# Canonical LoRA adaptation config, shared with the HITL training path
# (src/hitl/approved_lora_train.py) so adapters are always re-injectable.
LORA_TARGET_KEYWORDS = ["q_proj", "v_proj"]
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0


ADAPT_PROMPT = "Context fact extraction update: Question: {q} Answer:"


def _resolve_eos_id(tokenizer):
    eos_id = tokenizer.token_to_id("<|eos|>")
    if eos_id is None:
        eos_id = tokenizer.token_to_id("<|endoftext|>")
    return eos_id


def _token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 reward. Thin wrapper over the torch-free shared metric."""
    from src.common.metrics import token_f1
    return token_f1(prediction, reference)


@torch.no_grad()
def _generate_answer(raw_model, tokenizer, question: str, device, eos_id, max_new_tokens: int = 32) -> str:
    prompt_str = ADAPT_PROMPT.format(q=question)
    prompt_ids = tokenizer.encode(prompt_str).ids
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    # Greedy decoding (top_k=1) for deterministic, comparable evaluation.
    out = raw_model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=1,
        eos_token_id=eos_id,
    )
    gen_ids = out[0, len(prompt_ids):].tolist()
    if eos_id is not None and eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]
    return tokenizer.decode(gen_ids).strip()


def _evaluate_qa(raw_model, tokenizer, qa_pairs, device, eos_id) -> float:
    """Mean token-F1 of greedy answers over the QA set."""
    if not qa_pairs:
        return 0.0
    raw_model.eval()
    total = 0.0
    for pair in qa_pairs:
        pred = _generate_answer(raw_model, tokenizer, pair["q"], device, eos_id)
        total += _token_f1(pred, pair["a"])
    return total / len(qa_pairs)


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
    model = model.to(SystemConfig.DEVICE)

    eos_id = _resolve_eos_id(tokenizer)

    # --- SEAL safety gate: screen every self-edit BEFORE spending compute ---
    safety_flags = []
    if payload.enforce_safety:
        from src.hitl.validators import RiskValidator
        validator = RiskValidator(
            blocked_keywords=["ignore previous instructions", "system prompt"],
            sensitive_topics=["hack", "illegal", "exploit"],
        )
        for pair in payload.target_qa_pairs:
            level, flags = validator.analyze(pair.get("q", ""), pair.get("a", ""))
            if level == "HIGH":
                safety_flags.append({"q": pair.get("q", ""), "level": level, "flags": flags})
        if safety_flags:
            # Refuse outright: a reward improvement must never override a safety veto.
            raise HTTPException(
                status_code=422,
                detail={"error": "Self-edit rejected by safety gate (HIGH risk).",
                        "flagged": safety_flags},
            )

    # --- SEAL step: baseline reward BEFORE any adaptation ---
    base_reward = _evaluate_qa(model, tokenizer, payload.target_qa_pairs, SystemConfig.DEVICE, eos_id)
    print(f"[SEAL] Base model QA reward (token-F1): {base_reward:.4f}")

    # Apply Parameter Efficient Tuning
    inject_lora(model, target_keywords=LORA_TARGET_KEYWORDS, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
    model = model.to(SystemConfig.DEVICE)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=payload.lr)

    training_samples = []
    for pair in payload.target_qa_pairs:
        prompt_str = ADAPT_PROMPT.format(q=pair['q'])
        target_str = f" {pair['a']}"

        enc_prompt = tokenizer.encode(prompt_str).ids
        enc_target = tokenizer.encode(target_str).ids
        if eos_id is not None:
            enc_target = enc_target + [eos_id]

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

    # --- SEAL step: reward AFTER adaptation, then accept/reject gate ---
    adapted_reward = _evaluate_qa(model, tokenizer, payload.target_qa_pairs, SystemConfig.DEVICE, eos_id)
    improvement = adapted_reward - base_reward
    accepted = improvement >= payload.accept_margin
    print(f"[SEAL] Adapted reward: {adapted_reward:.4f} | base: {base_reward:.4f} | "
          f"improvement: {improvement:+.4f} | accepted: {accepted}")

    saved_path = None
    if accepted:
        # Persist ONLY the low-rank adapter deltas (PEFT storage), with the
        # config + provenance needed to re-inject and serve them later.
        saved_path = PathConfig.ADAPTED_MODEL_CHECKPOINT
        save_lora_adapter(
            model,
            saved_path,
            rank=LORA_RANK,
            alpha=LORA_ALPHA,
            target_keywords=LORA_TARGET_KEYWORDS,
            dropout=LORA_DROPOUT,
            meta={
                "source": "adaptation_service",
                "base_model": checkpoint_path_to_use,
                "base_reward": base_reward,
                "adapted_reward": adapted_reward,
                "improvement": improvement,
                "accept_margin": payload.accept_margin,
            },
        )
        print(f"[SEAL] Self-edit accepted. Adapter saved to {saved_path}")
    else:
        print("[SEAL] Self-edit rejected (no reward improvement). Adapter discarded.")

    return {
        "status": "Self-Adaptation Complete",
        "accepted": accepted,
        "base_reward": round(base_reward, 4),
        "adapted_reward": round(adapted_reward, 4),
        "improvement": round(improvement, 4),
        "saved_path": saved_path,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
