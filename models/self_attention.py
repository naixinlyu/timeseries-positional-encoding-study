# attention stuff

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V"""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, value)
        return output, attention_weights


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with residual connection and layer norm."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        residual = x

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # split into heads: (batch, n_heads, seq_len, d_k)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        attn_output, attention_weights = self.attention(q, k, v, mask)

        # concat heads back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        output = self.dropout(self.w_o(attn_output))
        output = self.layer_norm(residual + output)

        return output, attention_weights


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer: MHA + FFN, both with residual + LN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = None,
        dropout: float = 0.1
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.self_attention = MultiHeadSelfAttention(d_model, n_heads, dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_output, attention_weights = self.self_attention(x, mask)
        output = self.layer_norm(attn_output + self.ffn(attn_output))
        return output, attention_weights


if __name__ == "__main__":
    batch_size, seq_len, d_model, n_heads = 2, 10, 64, 4

    x = torch.randn(batch_size, seq_len, d_model)
    mha = MultiHeadSelfAttention(d_model, n_heads)
    output, attn_weights = mha(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")

    # quick permutation equivariance check (no PE)
    perm = torch.randperm(seq_len)
    x_shuffled = x[:, perm, :]

    output_original, _ = mha(x)
    output_shuffled, _ = mha(x_shuffled)
    output_shuffled_reordered = output_shuffled[:, torch.argsort(perm), :]

    diff = (output_original - output_shuffled_reordered).abs().max()
    print(f"\nPermutation equivariance diff: {diff:.6f} (should be ~0)")
