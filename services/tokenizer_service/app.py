"""
Tokenizer Microservice.

Input:
    Text through HTTP API.

Output:
    Token IDs or decoded text.

Run:
    TOKENIZER_PATH=artifacts/tokenizer/tokenizer.json \
    TOKENIZER_CONFIG=artifacts/tokenizer/tokenizer_config.json \
    uvicorn services.tokenizer_service.app:app --host 0.0.0.0 --port 8001
"""

import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from tokenizers import Tokenizer

from src.common.utils import load_json


TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "../../artifacts/tokenizer/tokenizer.json")
TOKENIZER_CONFIG = os.environ.get("TOKENIZER_CONFIG", "../../artifacts/tokenizer/tokenizer_config.json")

# Don't fail import if file doesn't exist yet, wait for request
tokenizer = None
tok_cfg = None

if os.path.exists(TOKENIZER_PATH) and os.path.exists(TOKENIZER_CONFIG):
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    tok_cfg = load_json(TOKENIZER_CONFIG)

app = FastAPI(title="Tokenizer Service")


class TokenizeRequest(BaseModel):
    text: str
    add_bos: bool = False
    add_eos: bool = False


class TokenizeResponse(BaseModel):
    ids: List[int]
    n_tokens: int


class DetokenizeRequest(BaseModel):
    ids: List[int]


class DetokenizeResponse(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "tokenizer"}


@app.get("/info")
def info():
    if not tok_cfg:
        return {"error": "Tokenizer not loaded yet"}
    return tok_cfg


@app.post("/tokenize", response_model=TokenizeResponse)
def tokenize(req: TokenizeRequest):
    """
    Input:
        Raw text.
    Output:
        List of token IDs.
    """
    if not tokenizer:
        return {"error": "Tokenizer not loaded yet"}
        
    ids = tokenizer.encode(req.text).ids

    special_ids = tok_cfg["special_token_ids"]

    if req.add_bos:
        ids = [special_ids["bos_token"]] + ids

    if req.add_eos:
        ids = ids + [special_ids["eos_token"]]

    return TokenizeResponse(ids=ids, n_tokens=len(ids))


@app.post("/detokenize", response_model=DetokenizeResponse)
def detokenize(req: DetokenizeRequest):
    """
    Input:
        List of token IDs.
    Output:
        Decoded text.
    """
    if not tokenizer:
        return {"error": "Tokenizer not loaded yet"}
        
    text = tokenizer.decode(req.ids)
    return DetokenizeResponse(text=text)
