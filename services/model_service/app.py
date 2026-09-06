from typing import Optional
"""
Model Inference Microservice.

This is useful after pretraining checkpoint is available.

Run:
    TOKENIZER_PATH=artifacts/tokenizer/tokenizer.json \
    TOKENIZER_CONFIG=artifacts/tokenizer/tokenizer_config.json \
    CKPT_PATH=runs/pretrain_200m/latest.pt \
    uvicorn services.model_service.app:app --host 0.0.0.0 --port 8002
"""

import os

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from tokenizers import Tokenizer

from src.common.utils import load_json
from src.model.config import ModelConfig
from src.model.llm import DecoderOnlyTransformer


TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "../../artifacts/tokenizer/tokenizer.json")
TOKENIZER_CONFIG = os.environ.get("TOKENIZER_CONFIG", "../../artifacts/tokenizer/tokenizer_config.json")
CKPT_PATH = os.environ.get("CKPT_PATH", "../../runs/pretrain_200m/latest.pt")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = None
tok_cfg = None
model = None

if os.path.exists(TOKENIZER_PATH) and os.path.exists(TOKENIZER_CONFIG):
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    tok_cfg = load_json(TOKENIZER_CONFIG)

if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**ckpt["model_config"])

    model = DecoderOnlyTransformer(model_config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

app = FastAPI(title="Model Service")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: Optional[int] = 50


class GenerateResponse(BaseModel):
    text: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "model",
        "device": device,
        "params": model.get_num_params() if model else 0,
        "model_loaded": model is not None
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """
    Input:
        prompt text

    Output:
        generated text
    """
    if not model or not tokenizer:
        return {"error": "Model or Tokenizer not loaded yet"}
        
    ids = tokenizer.encode(req.prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    )

    text = tokenizer.decode(output_ids[0].tolist())
    return GenerateResponse(text=text)
