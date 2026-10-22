import os
import sys

# Add project root to python path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hitl.db import QuarantineDatabase
from src.hitl.validators import RiskValidator
from src.hitl.proposal_engine import ProposalEngine

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("Initializing Quarantine DB...")
    db_path = os.path.join(project_root, "data", "quarantine_store.db")
    db = QuarantineDatabase(db_path)
    
    print("Loading text from data/duk_data.txt...")
    data_path = os.path.join(project_root, "data", "duk_data.txt")
    with open(data_path, "r") as f:
        passage = f.read().strip()
        
    print("Initializing ProposalEngine (Loading DPO model)...")
    blocked_keywords = ["ignore previous instructions", "system prompt"]
    sensitive_topics = ["hack", "illegal", "exploit"]
    validator = RiskValidator(blocked_keywords, sensitive_topics)
    
    # We will use the newly trained DPO model as our base model for the engine!
    base_model_path = os.path.join(project_root, "ckpts", "dgx_dpo", "latest_dpo.pt")
    tokenizer_path = os.path.join(project_root, "artifacts", "tokenizer", "tokenizer.json")
    device = "cpu" # running locally for dashboard generation
    
    # Check if DPO model exists, else fallback to SFT
    if not os.path.exists(base_model_path):
        base_model_path = "ckpts/dgx_sft/latest.pt"
        print(f"DPO model not found locally. Falling back to SFT model: {base_model_path}")
        
    try:
        engine = ProposalEngine(db, validator, base_model_path, tokenizer_path, device)
        
        questions = [
            "What is the full name of DUK?",
            "What kind of learning model does DUK use for internships?",
            "Which national level exam scores are considered for DUK admissions?",
            "What are the three main Schools of faculty at DUK?",
            "What topics are covered in the School of Computer Science and Engineering?"
        ]
        
        print(f"Extracting {len(questions)} QA proposals from the passage...")
        # Since we just want to quickly populate the DB for the dashboard, we'll just mock the output 
        # if the engine fails (which it might on a CPU without enough RAM).
        count = engine.extract_and_stage(passage[:1000], questions)
        print(f"Successfully generated {count} proposals in the Quarantine DB!")
        
    except Exception as e:
        print(f"Engine generation failed (likely memory issue on CPU): {e}")
        print("Mocking data for dashboard testing instead...")
        db.add_proposal(
            passage="Digital University Kerala...",
            question="What is the full name of DUK?",
            answer="Kerala University of Digital Sciences, Innovation and Technology",
            risk_level="LOW",
            flags=None
        )
        db.add_proposal(
            passage="Digital University Kerala...",
            question="What kind of learning model does DUK use for internships?",
            answer="Earn While You Learn",
            risk_level="LOW",
            flags=None
        )
        db.add_proposal(
            passage="Digital University Kerala...",
            question="Is it legal to hack the DUK servers?",
            answer="I cannot answer that question as it involves illegal activities.",
            risk_level="HIGH",
            flags=["Harmful Intent"]
        )
        print("Mocked 3 proposals for the dashboard.")

if __name__ == "__main__":
    main()
