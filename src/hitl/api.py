# src/hitl/api.py
import os
import subprocess
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.hitl.db import QuarantineDatabase
from src.hitl.schemas import ReviewAction, JobTrigger

app = FastAPI(title="MiniSEAL HITL Dashboard")

db_path = os.environ.get("HITL_DB_PATH", "quarantine.db")
db = QuarantineDatabase(db_path)

# Serve the static UI files
os.makedirs("ui", exist_ok=True)
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

@app.get("/")
def serve_index():
    if not os.path.exists("ui/index.html"):
        return JSONResponse({"error": "UI not built yet"})
    return FileResponse("ui/index.html")

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

def run_lora_training():
    # Simple subprocess to trigger the LoRA script
    print("Triggering LoRA training on approved proposals...")
    subprocess.run(["python3", "src/hitl/approved_lora_train.py", "--db", db_path])
    print("LoRA training complete!")

@app.post("/api/train_lora")
def trigger_training(trigger: JobTrigger, background_tasks: BackgroundTasks):
    """Triggers LoRA training on approved proposals in the background."""
    approved_count = len(db.list_proposals("APPROVED"))
    if approved_count == 0:
        raise HTTPException(status_code=400, detail="No approved proposals available for training.")
        
    background_tasks.add_task(run_lora_training)
    return {"message": f"Started LoRA training on {approved_count} approved proposals!"}
