# src/hitl/proposal_engine.py

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.hitl.db import QuarantineDatabase
from src.hitl.validators import RiskValidator

def find_latest_adapter(search_roots):
    """Return the newest PEFT adapter directory across the given roots, or None.

    A directory is an adapter iff it contains adapter_config.json. Matching by
    that marker (not by name) means adapters from ANY stage are discovered:
    SFT (final_adapter), SEAL (seal_adapter), HITL (lora_adapter_v*), DPO
    (dpo_adapter_v*). (The old code only matched dirs named "adapter_v*", so
    none of the trainers' outputs were ever found.)
    """
    found = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            if "adapter_config.json" in files:
                found.append(dirpath)
    if not found:
        return None
    found.sort(key=os.path.getmtime, reverse=True)
    return found[0]

class ProposalEngine:
    def __init__(self, db: QuarantineDatabase, validator: RiskValidator, base_model_path: str, tokenizer_path: str, device: str):
        self.db = db
        self.validator = validator
        self.device = device
        
        # We ignore base_model_path passed from legacy code and enforce HF Qwen
        actual_base_model = "Qwen/Qwen2.5-1.5B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(actual_base_model)
        
        print(f"[HITL] Loading Base Model: {actual_base_model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            actual_base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Discover the newest LoRA adapter from any stage (SFT/SEAL/HITL/DPO).
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        adapter = find_latest_adapter([
            os.path.join(project_root, "ckpts", "adapters"),
            os.path.join(project_root, "model_weights"),
        ])
        if adapter:
            print(f"[HITL] Loading latest LoRA adapter: {adapter}")
            self.model = PeftModel.from_pretrained(self.model, adapter)
        else:
            print("[HITL] WARNING: No trained LoRA adapters found. Using base model only.")

        self.model.eval()

    def extract_and_stage(self, passage: str, questions: list[str]) -> int:
        """
        Executes a zero-shot model forward pass to generate self-corrections, 
        validates the risk level, and writes the results to the Quarantine DB.
        """
        proposals_created = 0
        
        for q in questions:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer the user's question accurately using ONLY the provided context."},
                {"role": "user", "content": f"Context: {passage}\n\nQuestion: {q}"}
            ]
            
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=150, 
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]
            raw_answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Run automated risk analysis
            risk, flags = self.validator.analyze(q, raw_answer)
            
            # Stage into the isolated SQLite database
            self.db.add_proposal(passage, q, raw_answer, risk, flags)
            proposals_created += 1
            
        return proposals_created
