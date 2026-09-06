"""
Simple LoRA implementation for the custom Transformer.

LoRA paper:
    Hu et al., 2022
    Low-Rank Adaptation of Large Language Models

Core idea:
    Freeze original linear layer.
    Add trainable low-rank update:
        W' = W + BA

This file supports:
    - injecting LoRA into selected Linear modules
    - saving only LoRA adapter weights
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear layer with LoRA adapters.

    Input:
        base_layer: existing frozen Linear layer.
        rank: LoRA rank.
        alpha: scaling factor.
        dropout: dropout applied before LoRA branch.

    Output:
        base_layer(x) + scaled_lora_update(x)
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Freeze base model weights.
        for p in self.base_layer.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize A with Kaiming, B with zeros.
        This makes initial LoRA update equal to zero.
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)

        lora_hidden = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_hidden, self.lora_B)

        return base_out + self.scaling * lora_out


def freeze_model(model: nn.Module) -> None:
    """
    Freezes all model parameters.
    """
    for p in model.parameters():
        p.requires_grad = False


def inject_lora(
    model: nn.Module,
    target_keywords: List[str],
    rank: int,
    alpha: int,
    dropout: float,
) -> nn.Module:
    """
    Replaces matching Linear layers with LoRALinear.

    Input:
        model: custom Transformer.
        target_keywords:
            Example:
                ["q_proj", "k_proj", "v_proj", "out_proj",
                 "gate_proj", "up_proj", "down_proj"]
        rank: LoRA rank.
        alpha: LoRA alpha.
        dropout: LoRA dropout.

    Output:
        model with LoRA modules injected.
    """
    freeze_model(model)

    def replace(module: nn.Module, prefix: str = ""):
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name

            if isinstance(child, nn.Linear):
                if any(keyword in full_name for keyword in target_keywords):
                    setattr(
                        module,
                        child_name,
                        LoRALinear(
                            base_layer=child,
                            rank=rank,
                            alpha=alpha,
                            dropout=dropout,
                        ),
                    )
            else:
                replace(child, full_name)

    replace(model)
    return model


def get_lora_state_dict(model: nn.Module) -> dict:
    """
    Output:
        State dict containing only LoRA parameters.
    """
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }


def get_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Output:
        trainable parameters, total parameters.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    return trainable, total
