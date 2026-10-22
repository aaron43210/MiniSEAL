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


def load_lora_state_dict(model: nn.Module, lora_state_dict: dict, strict: bool = True) -> None:
    """
    Load LoRA A/B parameters into a model that ALREADY has LoRA injected.

    The model must have been passed through ``inject_lora`` with the SAME
    rank / alpha / target_keywords used when the adapter was trained, otherwise
    the parameter names (and shapes) will not line up. Use ``load_lora_adapter``
    to do that re-injection automatically from a saved adapter file.

    Args:
        model: a model with LoRALinear modules already in place.
        lora_state_dict: mapping of ``*.lora_A`` / ``*.lora_B`` -> tensor.
        strict: if True, require that the adapter keys exactly match the
            model's LoRA parameter names (no missing / unexpected keys).

    Raises:
        KeyError: on a name mismatch when ``strict`` is True.
        RuntimeError: on a shape mismatch (always).
    """
    model_lora = {
        name: param
        for name, param in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }
    if not model_lora:
        raise RuntimeError(
            "No LoRA parameters found on the model. Call inject_lora(...) "
            "(or load_lora_adapter) before load_lora_state_dict."
        )

    adapter_keys = set(lora_state_dict.keys())
    model_keys = set(model_lora.keys())
    if strict:
        missing = model_keys - adapter_keys
        unexpected = adapter_keys - model_keys
        if missing or unexpected:
            raise KeyError(
                "LoRA adapter does not match injected model.\n"
                f"  missing from adapter: {sorted(missing)}\n"
                f"  unexpected in adapter: {sorted(unexpected)}\n"
                "This usually means rank/alpha/target_keywords differ from "
                "when the adapter was trained."
            )

    with torch.no_grad():
        for name, param in model_lora.items():
            if name not in lora_state_dict:
                continue  # non-strict: leave untouched (stays at zero-init)
            incoming = lora_state_dict[name]
            if incoming.shape != param.shape:
                raise RuntimeError(
                    f"Shape mismatch for '{name}': adapter {tuple(incoming.shape)} "
                    f"vs model {tuple(param.shape)}."
                )
            param.copy_(incoming.to(param.device, dtype=param.dtype))


# Canonical on-disk adapter format. Both the automated SEAL gate
# (adaptation_service) and the HITL training path write this exact shape so a
# single loader can serve either. Older raw-dict adapters are still accepted by
# load_lora_adapter (see the backward-compat branch there).
ADAPTER_FORMAT_VERSION = 1


def save_lora_adapter(
    model: nn.Module,
    path: str,
    *,
    rank: int,
    alpha: int,
    target_keywords: List[str],
    dropout: float = 0.0,
    meta: dict | None = None,
) -> str:
    """
    Save ONLY the LoRA deltas plus the config needed to re-inject them.

    Storing rank/alpha/target_keywords alongside the weights is what makes the
    adapter self-describing: the loader can reconstruct the exact LoRALinear
    placement without the caller having to remember the hyperparameters.

    Args:
        model: a model with LoRA injected and (presumably) trained.
        path: destination ``.pt`` file.
        rank, alpha, target_keywords, dropout: the inject_lora config used.
        meta: optional provenance (rewards, source, timestamp, base model, ...).

    Returns:
        The path written.
    """
    import os as _os

    payload = {
        "format_version": ADAPTER_FORMAT_VERSION,
        "lora_state_dict": get_lora_state_dict(model),
        "lora_config": {
            "rank": int(rank),
            "alpha": int(alpha),
            "target_keywords": list(target_keywords),
            "dropout": float(dropout),
        },
        "meta": dict(meta) if meta else {},
    }
    parent = _os.path.dirname(_os.path.abspath(path))
    _os.makedirs(parent, exist_ok=True)
    torch.save(payload, path)
    return path


def load_lora_adapter(model: nn.Module, path: str, device: str = "cpu") -> nn.Module:
    """
    Re-inject LoRA using the adapter's stored config, then load its weights.

    This closes the SEAL loop: an accepted/trained adapter saved with
    ``save_lora_adapter`` can be applied to a fresh base model for serving.

    Backward compatible with legacy files that stored EITHER a bare
    ``{name: tensor}`` state dict OR ``{"lora_state_dict": ..., "args": ...}``
    without a ``lora_config`` block; in that case we fall back to the historical
    defaults (rank=16, alpha=32, targets=[q_proj, v_proj], dropout=0.0) that
    every writer in this repo used.

    Args:
        model: a FRESH base model (no LoRA yet).
        path: adapter file written by ``save_lora_adapter`` (or legacy).
        device: device to map tensors onto.

    Returns:
        The same model, now with LoRA injected and the adapter loaded.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "lora_state_dict" in ckpt:
        lora_sd = ckpt["lora_state_dict"]
        cfg = ckpt.get("lora_config", {})
    else:
        # Legacy: the whole object IS the lora state dict.
        lora_sd = ckpt
        cfg = {}

    rank = int(cfg.get("rank", 16))
    alpha = int(cfg.get("alpha", 32))
    target_keywords = list(cfg.get("target_keywords", ["q_proj", "v_proj"]))
    dropout = float(cfg.get("dropout", 0.0))

    inject_lora(
        model,
        target_keywords=target_keywords,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    model = model.to(device)
    load_lora_state_dict(model, lora_sd, strict=True)
    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Fold each LoRA delta into its frozen base weight and replace the LoRALinear
    with a plain nn.Linear, giving a zero-overhead model for deployment.

    For a LoRALinear with base weight W (out, in) and update (alpha/rank)*B@A:
        W_merged = W + (alpha/rank) * (B @ A)

    After merging the result is numerically equivalent to the LoRALinear's
    forward (minus dropout, which is identity at eval time) but with no extra
    matmul. In-place on ``model``; returns it for convenience.
    """
    def _merge(module: nn.Module):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                base = child.base_layer
                delta = child.scaling * (child.lora_B @ child.lora_A)  # (out, in)
                with torch.no_grad():
                    base.weight.add_(delta.to(base.weight.dtype))
                for p in base.parameters():
                    p.requires_grad_(True)
                setattr(module, child_name, base)
            else:
                _merge(child)

    _merge(model)
    return model


def get_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Output:
        trainable parameters, total parameters.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    return trainable, total
