"""
Single source of truth for the chat/instruction prompt format and special-token
resolution, shared by SFT, DPO, and inference.

Train/inference skew is a silent killer: if SFT, DPO, and serving disagree on the
exact prompt string, the loss mask boundary and the model's expected input drift
apart. Every stage MUST format prompts through `format_prompt` and resolve special
tokens through the helpers here.
"""

from typing import Optional


# Canonical instruction format (Alpaca-style). Change it in exactly one place.
CHAT_PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_prompt(instruction: str) -> str:
    """Format a raw instruction/prompt into the canonical chat prompt."""
    return CHAT_PROMPT_TEMPLATE.format(instruction=instruction.strip())


def resolve_eos_id(tokenizer) -> Optional[int]:
    """Resolve the EOS token id, falling back to <|endoftext|> then None."""
    eos_id = tokenizer.token_to_id("<|eos|>")
    if eos_id is None:
        eos_id = tokenizer.token_to_id("<|endoftext|>")
    return eos_id


def resolve_pad_id(tokenizer) -> int:
    """Resolve the pad token id, falling back to 0 if absent."""
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        pad_id = 0
    return pad_id
