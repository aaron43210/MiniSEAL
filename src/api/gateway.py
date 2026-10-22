# src/api/gateway.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from typing import List, Dict
import uvicorn

app = FastAPI(
    title="Generative AI Project - API Gateway",
    description="Unified routing microservice to cleanly execute full-stack LLM processes",
    version="1.0.0"
)

SERVICES = {
    "tokenizer": "http://tokenizer-service:8001",
    "alignment": "http://alignment-service:8002",
    "adaptation": "http://adaptation-service:8003",
    "hitl": "http://hitl-service:8006"
}

class FullPipelineRequest(BaseModel):
    pretrain_corpus_paths: List[str]
    sft_data: List[Dict[str, str]]
    dpo_data: List[Dict[str, str]]
    adaptation_passage: str
    adaptation_qa_pairs: List[Dict[str, str]]


@app.get("/health")
async def health_all():
    """Consolidated health check endpoint for all active microservice containers."""
    status = {}
    async with httpx.AsyncClient() as client:
        for name, url in SERVICES.items():
            try:
                # Prefer the dedicated /health probe; fall back to /docs for any
                # service that doesn't expose one (e.g. the tokenizer service).
                try:
                    response = await client.get(f"{url}/health", timeout=2.0)
                    if response.status_code == 404:
                        response = await client.get(f"{url}/docs", timeout=2.0)
                except httpx.HTTPError:
                    response = await client.get(f"{url}/docs", timeout=2.0)
                status[name] = "ONLINE" if response.status_code == 200 else "ERROR"
            except Exception:
                status[name] = "OFFLINE"
    return {"status": "ACTIVE", "microservices": status}


@app.post("/pipeline/run")
async def execute_full_pipeline(payload: FullPipelineRequest):
    """Orchestrates sequential execution across the microservices."""
    async with httpx.AsyncClient(timeout=600.0) as client:
        
        # 1. Train Tokenizer
        print("Gateway -> Tokenizer service...")
        tok_res = await client.post(
            f"{SERVICES['tokenizer']}/train", 
            json={"corpus_filepaths": payload.pretrain_corpus_paths}
        )
        if tok_res.status_code != 200:
            raise HTTPException(status_code=500, detail="Tokenizer training failed.")
            
        # 2. SFT alignment
        print("Gateway -> Alignment SFT step...")
        sft_res = await client.post(
            f"{SERVICES['alignment']}/sft", 
            json={"sft_data": payload.sft_data}
        )
        if sft_res.status_code != 200:
            raise HTTPException(status_code=500, detail=f"SFT phase failed: {sft_res.text}")
            
        # 3. DPO Alignment
        print("Gateway -> Alignment DPO step...")
        dpo_res = await client.post(
            f"{SERVICES['alignment']}/dpo", 
            json={"dpo_data": payload.dpo_data}
        )
        if dpo_res.status_code != 200:
            raise HTTPException(status_code=500, detail=f"DPO phase failed: {dpo_res.text}")
            
        # 4. SEAL LoRA Self-Adaptation
        print("Gateway -> Self-Adaptation phase...")
        adapt_res = await client.post(
            f"{SERVICES['adaptation']}/adapt", 
            json={
                "passage": payload.adaptation_passage,
                "target_qa_pairs": payload.adaptation_qa_pairs
            }
        )
        if adapt_res.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Adaptation failed: {adapt_res.text}")
            
        return {
            "status": "Pipeline executed successfully",
            "stages": {
                "tokenizer": tok_res.json(),
                "sft": sft_res.json(),
                "dpo": dpo_res.json(),
                "adaptation": adapt_res.json()
            }
        }


@app.post("/pipeline/propose_failure")
async def pipeline_propose_failure(passage: str, questions: List[str]):
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{SERVICES['hitl']}/proposals/generate?passage={passage}", 
            json=questions
        )
        if res.status_code != 200:
            raise HTTPException(status_code=500, detail="Failure staging failed.")
        return res.json()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
