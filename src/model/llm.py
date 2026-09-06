from typing import Optional
"""
Decoder-only Transformer language model.

Training objective:
    Next-token prediction.

Input:
    input_ids: [batch, seq]

Output:
    logits: [batch, seq, vocab]
    loss: optional cross-entropy loss if targets are given.
"""

import math
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import ModelConfig
from src.model.layers import RMSNorm, TransformerBlock


class DecoderOnlyTransformer(nn.Module):
    """
    GPT/LLaMA-style decoder-only language model.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )

        self.norm = RMSNorm(config.n_embd, config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        # Weight tying: common in GPT models.
        self.lm_head.weight = self.tok_embeddings.weight

        # GPT-style scaled residual initialization.
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer),
                )

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initializes model weights.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        """
        Input:
            input_ids: [batch, seq]
            targets: [batch, seq], optional

        Output:
            logits: [batch, seq, vocab]
            loss: scalar tensor if targets are provided.
        """
        batch, seq_len = input_ids.shape

        if seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block_size {self.config.block_size}"
            )

        x = self.tok_embeddings(input_ids)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Input:
            input_ids: [batch, seq]
            max_new_tokens: number of tokens to generate.
            temperature: sampling temperature.
            top_k: keep only top-k logits if provided.

        Output:
            Generated token IDs: [batch, seq + max_new_tokens]
        """
        self.eval()

        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.block_size:]

            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def get_num_params(self) -> int:
        """
        Output:
            Number of trainable model parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self) -> dict:
        return asdict(self.config)
