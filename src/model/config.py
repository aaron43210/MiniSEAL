from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ModelConfig:
    """
    Configuration for decoder-only Transformer.

    Inputs:
        vocab_size: Number of tokenizer vocabulary items.
        block_size: Maximum context length.
        n_layer: Number of Transformer blocks.
        n_head: Number of attention heads.
        n_embd: Embedding dimension.
    """

    vocab_size: int
    block_size: int = 1024
    n_layer: int = 16
    n_head: int = 16
    n_embd: int = 1024
    dropout: float = 0.0
    bias: bool = False

    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    # LLaMA-style FFN rounding.
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)
