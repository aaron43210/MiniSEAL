# src/hitl/schemas.py

from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class ProposalCreate(BaseModel):
    passage: str = Field(..., description="Raw text context feeding the SEAL extraction")
    question: str = Field(..., description="Target query context")
    proposed_answer: str = Field(..., description="LLM-generated correction")

class ProposalResponse(BaseModel):
    id: int
    passage: str
    question: str
    proposed_answer: str
    status: str
    risk_level: str
    risk_flags: Optional[str]
    created_at: str

class ReviewAction(BaseModel):
    proposal_id: int
    action: str = Field(..., pattern="^(APPROVED|REJECTED)$")

class JobTrigger(BaseModel):
    epochs: Optional[int] = None
    lr: Optional[float] = None
