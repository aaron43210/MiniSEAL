"""
Core Transformer layers.

Implemented:
- RMSNorm
- Rotary Positional Embedding
- Causal Self-Attention
- SwiGLU MLP
- Pre-Norm Transformer block
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import ModelConfig


class RMSNorm(nn.Module):
    """
    RMSNorm from Zhang & Sennrich.

    Difference from LayerNorm:
        - Does not subtract mean.
        - Normalizes by root mean square.
        - Usually cheaper and stable for LLMs.

    Input:
        x: Tensor of shape [batch, seq, hidden]

    Output:
        Normalized tensor of same shape.
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * rms


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding.

    RoPE rotates query/key vectors based on token position.
    It avoids learned absolute positional embeddings.

    Input:
        q/k tensor: [batch, heads, seq, head_dim]

    Output:
        position-rotated q/k tensor.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("RoPE requires even head_dim.")

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            seq_len: Sequence length.
            device: CPU/GPU.
            dtype: Target dtype.

        Output:
            cos, sin tensors shaped [1, 1, seq, head_dim/2]
        """
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq.to(device))
        cos = freqs.cos()[None, None, :, :].to(dtype)
        sin = freqs.sin()[None, None, :, :].to(dtype)
        return cos, sin

    @staticmethod
    def apply_rope(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input:
            x: [batch, heads, seq, head_dim]
            cos/sin: [1, 1, seq, head_dim/2]

        Output:
            RoPE-applied tensor with same shape.
        """
        # Store original dtype
        orig_dtype = x.dtype
        
        # Upcast to float32 for precision stability during rotation
        x = x.float()
        cos = cos.float()
        sin = sin.float()

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        out = torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
        return out.to(orig_dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(2)
        cos, sin = self.get_cos_sin(seq_len, q.device, q.dtype)

        q = self.apply_rope(q, cos, sin)
        k = self.apply_rope(k, cos, sin)

        return q, k


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Uses PyTorch scaled_dot_product_attention.
    This still counts as your own model architecture;
    the CUDA kernel is optimized internally by PyTorch.

    Input:
        x: [batch, seq, hidden]

    Output:
        [batch, seq, hidden]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            theta=config.rope_theta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, T, H] -> [B, heads, T, head_dim]
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # [B, heads, T, head_dim] -> [B, T, H]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, hidden)

        return self.out_proj(attn_output)


class SwiGLUMLP(nn.Module):
    """
    SwiGLU feed-forward network.

    Formula:
        FFN(x) = W_down( SiLU(W_gate x) * W_up x )

    Used in modern LLMs because it gives better quality than plain GELU MLP.

    Input:
        x: [batch, seq, hidden]

    Output:
        [batch, seq, hidden]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        hidden_dim = int(2 * (4 * config.n_embd) / 3)

        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)

        # Round to hardware-friendly multiple.
        hidden_dim = config.multiple_of * math.ceil(hidden_dim / config.multiple_of)

        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class TransformerBlock(nn.Module):
    """
    Pre-Norm Transformer block.

    Structure:
        x = x + Attention(RMSNorm(x))
        x = x + MLP(RMSNorm(x))

    Pre-norm is more stable for deep LMs.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.ffn_norm = RMSNorm(config.n_embd, config.norm_eps)

        self.attn = CausalSelfAttention(config)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x
