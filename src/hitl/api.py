# src/hitl/api.py
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.hitl.db import QuarantineDatabase
from src.hitl.schemas import ReviewAction, JobTrigger

app = FastAPI(title="MiniSEAL HITL Dashboard")

db_path = os.environ.get("HITL_DB_PATH", "data/quarantine_store.db")
db = QuarantineDatabase(db_path)

# Serve the static UI files
os.makedirs("ui", exist_ok=True)
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

@app.get("/")
def serve_index():
    if not os.path.exists("ui/index.html"):
        return JSONResponse({"error": "UI not built yet"})
    return FileResponse("ui/index.html")


@app.get("/health")
def health():
    """Lightweight liveness probe used by the gateway."""
    return {"status": "ok", "service": "hitl"}

@app.get("/api/proposals")
def get_proposals(status: str = "PENDING"):
    """Returns proposals based on status (PENDING, APPROVED, REJECTED)."""
    rows = db.list_proposals(status)
    return rows

@app.get("/api/metrics")
def get_metrics():
    """Returns placeholder metrics."""
    return {
        "dpo_final_loss": 4.19,
        "dpo_chosen_acc": "74%",
        "proposals_approved": len(db.list_proposals("APPROVED")),
        "proposals_rejected": len(db.list_proposals("REJECTED"))
    }

@app.post("/api/review")
def review_proposal(action: ReviewAction):
    """Approves or rejects a proposal."""
    db.update_status(action.proposal_id, action.action)
    return {"message": f"Proposal {action.proposal_id} updated to {action.action}"}

def run_lora_training(epochs: int, lr: float):
    """Run LoRA training on approved proposals in-process (no fragile subprocess)."""
    from src.hitl.approved_lora_train import execute_approved_training

    base_model = os.environ.get("HITL_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    output_dir = os.environ.get("HITL_ADAPTER_DIR", "ckpts/adapters")

    print(f"Triggering LoRA training on approved proposals (epochs={epochs}, lr={lr})...")
    path = execute_approved_training(
        db_path=db_path,
        base_model_path=base_model,
        output_dir=output_dir,
        epochs=epochs,
        lr=lr,
    )
    print(f"LoRA training complete! Adapter: {path}")

@app.post("/api/train_lora")
def trigger_training(trigger: JobTrigger, background_tasks: BackgroundTasks):
    """Triggers LoRA training on approved proposals in the background."""
    approved_count = len(db.list_proposals("APPROVED"))
    if approved_count == 0:
        raise HTTPException(status_code=400, detail="No approved proposals available for training.")

    epochs = trigger.epochs if trigger.epochs is not None else 5
    lr = trigger.lr if trigger.lr is not None else 1e-4
    background_tasks.add_task(run_lora_training, epochs, lr)
    return {"message": f"Started LoRA training on {approved_count} approved proposals!"}
