# services/hitl_service/app.py

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
import yaml
import uvicorn
import os
import torch

from src.hitl.db import QuarantineDatabase
from src.hitl.validators import RiskValidator
from src.hitl.proposal_engine import ProposalEngine
from src.hitl.approved_lora_train import execute_approved_training
from src.hitl.schemas import ProposalCreate, ReviewAction, JobTrigger

app = FastAPI(
    title="Human-in-the-Loop Microservice Interface",
    version="2.0.0"
)

# 1. Parse YAML Configurations
with open("configs/hitl_debug.yaml", "r") as f:
    config = yaml.safe_load(f)

db = QuarantineDatabase(config["paths"]["db_path"])
validator = RiskValidator(
    blocked_keywords=config["security"]["blocked_keywords"],
    sensitive_topics=config["security"]["flagged_sensitive_topics"]
)

# 2. REST Endpoints

@app.post("/proposals/generate")
def generate_and_stage_proposals(passage: str, questions: list[str]):
    # Determine device
    device_str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    engine = ProposalEngine(
        db, validator, 
        config["paths"]["base_model_path"], 
        config["paths"]["tokenizer_path"], 
        device_str
    )
    count = engine.extract_and_stage(passage, questions)
    return {"status": "success", "proposals_staged": count}

@app.get("/proposals")
def get_proposals(status: str = "PENDING"):
    return db.list_proposals(status)

@app.post("/proposals/review")
def review_proposal(action: ReviewAction):
    status = "APPROVED" if action.action == "APPROVE" else "REJECTED"
    db.update_status(action.proposal_id, status)
    return {"status": "success"}

@app.post("/train")
def run_job(trigger: JobTrigger):
    epochs = trigger.epochs or config["hyperparameters"]["epochs"]
    lr = trigger.lr or config["hyperparameters"]["lr"]
    
    filepath = execute_approved_training(
        config["paths"]["db_path"],
        config["paths"]["base_model_path"],
        config["paths"]["tokenizer_path"],
        config["paths"]["adapters_dir"],
        epochs, lr
    )
    if not filepath:
        return {"status": "skipped", "message": "No approved items to train on."}
    return {"status": "success", "saved_adapter_path": filepath}

# --- 3. HTML Admin Dashboard ---

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    pending_items = db.list_proposals("PENDING")
    approved_items = db.list_proposals("APPROVED")
    
    html_content = """
    <html>
        <head>
            <title>HITL Quarantine Dashboard</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; color: #333; margin: 0; padding: 20px; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                h1 { color: #1a365d; border-bottom: 2px solid #e2e8f0; padding-bottom: 15px; }
                h2 { color: #2c5282; margin-top: 30px; }
                table { width: 100%; border-collapse: collapse; margin-top: 15px; }
                th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #e2e8f0; }
                th { background-color: #ebf8ff; color: #2c5282; }
                .badge { padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
                .badge-low { background-color: #c6f6d5; color: #22543d; }
                .badge-med { background-color: #feebc8; color: #744210; }
                .badge-high { background-color: #fed7d7; color: #742a2a; }
                .btn { padding: 8px 14px; border: none; border-radius: 4px; font-weight: bold; cursor: pointer; }
                .btn-approve { background-color: #48bb78; color: white; margin-right: 5px; }
                .btn-reject { background-color: #e53e3e; color: white; }
                .btn-train { background-color: #3182ce; color: white; font-size: 16px; padding: 12px 24px; margin-top: 20px; }
                .banner-section { display: flex; justify-content: space-between; align-items: center; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="banner-section">
                    <h1>Human-in-the-Loop (HITL) Security Quarantine</h1>
                    <form action="/dashboard/train" method="post">
                        <button class="btn btn-train" type="submit">Deploy Approved Adapters</button>
                    </form>
                </div>
                
                <h2>Pending Review Queue ({len(pending_items)})</h2>
                <table>
                    <tr>
                        <th>ID</th>
                        <th>Passage Context</th>
                        <th>Extracted Query</th>
                        <th>Proposed Self-Edit</th>
                        <th>Risk Assessment</th>
                        <th>Actions</th>
                    </tr>
    """
    for item in pending_items:
        badge_class = f"badge-low" if item["risk_level"] == "LOW" else f"badge-med" if item["risk_level"] == "MEDIUM" else "badge-high"
        html_content += f"""
                    <tr>
                        <td>{item["id"]}</td>
                        <td>{item["passage"][:100]}...</td>
                        <td>{item["question"]}</td>
                        <td><strong>{item["proposed_answer"]}</strong></td>
                        <td>
                            <span class="badge {badge_class}">{item["risk_level"]}</span>
                            <br><small style="color:#e53e3e">{item["risk_flags"] or ""}</small>
                        </td>
                        <td>
                            <form action="/dashboard/review" method="post" style="display:inline;">
                                <input type="hidden" name="proposal_id" value="{item["id"]}">
                                <input type="hidden" name="action" value="APPROVE">
                                <button class="btn btn-approve" type="submit">Approve</button>
                            </form>
                            <form action="/dashboard/review" method="post" style="display:inline;">
                                <input type="hidden" name="proposal_id" value="{item["id"]}">
                                <input type="hidden" name="action" value="REJECT">
                                <button class="btn btn-reject" type="submit">Reject</button>
                            </form>
                        </td>
                    </tr>
        """
    html_content += """
                </table>
                
                <h2>Approved Update Pipeline ({len(approved_items)})</h2>
                <table>
                    <tr>
                        <th>ID</th>
                        <th>Question</th>
                        <th>Vetted Answer</th>
                    </tr>
    """
    for item in approved_items:
        html_content += f"""
                    <tr>
                        <td>{item["id"]}</td>
                        <td>{item["question"]}</td>
                        <td>{item["proposed_answer"]}</td>
                    </tr>
        """
    html_content += """
                </table>
            </div>
        </body>
    </html>
    """
    return html_content

@app.post("/dashboard/review")
def dashboard_action(proposal_id: int = Form(...), action: str = Form(...)):
    status = "APPROVED" if action == "APPROVE" else "REJECTED"
    db.update_status(proposal_id, status)
    return RedirectResponse(url="/dashboard", status_code=303)

@app.post("/dashboard/train")
def dashboard_train():
    execute_approved_training(
        config["paths"]["db_path"],
        config["paths"]["base_model_path"],
        config["paths"]["tokenizer_path"],
        config["paths"]["adapters_dir"],
        config["hyperparameters"]["epochs"],
        config["hyperparameters"]["lr"]
    )
    return RedirectResponse(url="/dashboard", status_code=303)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)
