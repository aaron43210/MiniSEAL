# src/hitl/proposal_engine.py

import torch
from tokenizers import Tokenizer
from src.model.llm import DecoderOnlyTransformer
from src.model.config import ModelConfig
from src.hitl.db import QuarantineDatabase
from src.hitl.validators import RiskValidator

class ProposalEngine:
    def __init__(self, db: QuarantineDatabase, validator: RiskValidator, base_model_path: str, tokenizer_path: str, device: str):
        self.db = db
        self.validator = validator
        self.device = device
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Load the base model checkpoint
        checkpoint = torch.load(base_model_path, map_location=device)
        self.model_args = ModelConfig(**checkpoint['args']) if 'args' in checkpoint else ModelConfig(**checkpoint['model_config'])
        self.model = DecoderOnlyTransformer(self.model_args).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint['model'])
        self.model.eval()

    def extract_and_stage(self, passage: str, questions: list[str]) -> int:
        """
        Executes a zero-shot model forward pass to generate self-corrections, 
        validates the risk level, and writes the results to the Quarantine DB.
        """
        proposals_created = 0
        pad_id = self.tokenizer.token_to_id("<|pad|>")
        if pad_id is None:
            pad_id = 0
            
        for q in questions:
            prompt_str = f"Passage Context: {passage}\nQuestion: {q} Answer:"
            input_ids = self.tokenizer.encode(prompt_str).ids
            inputs_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            
            # Simple greedy token generation loop
            generated_ids = []
            for _ in range(50):  # Cap generations at 50 tokens
                with torch.no_grad():
                    logits, _ = self.model(inputs_tensor)
                next_token_id = torch.argmax(logits[0, -1, :]).item()
                if next_token_id == self.tokenizer.token_to_id("<|eos|>") or next_token_id == pad_id:
                    break
                generated_ids.append(next_token_id)
                inputs_tensor = torch.cat([inputs_tensor, torch.tensor([[next_token_id]], device=self.device)], dim=-1)
                
            raw_answer = self.tokenizer.decode(generated_ids).strip()
            
            # Run automated risk analysis
            risk, flags = self.validator.analyze(q, raw_answer)
            
            # Stage into the isolated SQLite database
            self.db.add_proposal(passage, q, raw_answer, risk, flags)
            proposals_created += 1
            
        return proposals_created
