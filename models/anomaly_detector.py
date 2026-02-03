"""
Transformer-based autoencoder for time series anomaly detection.
Reconstruction error (MSE) serves as the anomaly score.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from models.positional_encoding import (
    SinusoidalPositionalEncoding,
    LearnablePositionalEncoding,
    NoPositionalEncoding
)
from models.self_attention import TransformerEncoderLayer


class TimeSeriesAnomalyDetector(nn.Module):
    """
    Autoencoder: input projection -> (optional) PE -> transformer encoder -> output projection.
    Anomaly score = reconstruction MSE.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = None,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        pe_type: str = 'sinusoidal',
        max_len: int = 5000
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding

        self.input_projection = nn.Linear(input_dim, d_model)

        if use_positional_encoding:
            if pe_type == 'sinusoidal':
                self.positional_encoding = SinusoidalPositionalEncoding(
                    d_model, max_len, dropout
                )
            elif pe_type == 'learnable':
                self.positional_encoding = LearnablePositionalEncoding(
                    d_model, max_len, dropout
                )
            else:
                raise ValueError(f"Unknown pe_type: {pe_type}")
        else:
            self.positional_encoding = NoPositionalEncoding(dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, input_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        h = self.input_projection(x)
        h = self.positional_encoding(h)

        attention_weights_list = []
        for layer in self.encoder_layers:
            h, attn_weights = layer(h)
            if return_attention:
                attention_weights_list.append(attn_weights)

        reconstruction = self.output_projection(h)

        if return_attention:
            return reconstruction, attention_weights_list
        return reconstruction, None

    def compute_anomaly_scores(
        self,
        x: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Returns MSE-based anomaly scores.
        reduction: 'none' | 'feature' | 'mean'
        """
        reconstruction, _ = self.forward(x)
        mse = (x - reconstruction) ** 2

        if reduction == 'none':
            return mse
        elif reduction == 'feature':
            return mse.mean(dim=-1)
        elif reduction == 'mean':
            return mse.mean(dim=(1, 2))
        else:
            raise ValueError(f"Unknown reduction: {reduction}")


def create_model_pair(
    input_dim: int,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    **kwargs
) -> Tuple[TimeSeriesAnomalyDetector, TimeSeriesAnomalyDetector]:
    """Create a matched pair of models: one with PE, one without."""
    model_with_pe = TimeSeriesAnomalyDetector(
        input_dim=input_dim, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers,
        use_positional_encoding=True, **kwargs
    )
    model_without_pe = TimeSeriesAnomalyDetector(
        input_dim=input_dim, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers,
        use_positional_encoding=False, **kwargs
    )
    return model_with_pe, model_without_pe


def create_model_trio(
    input_dim: int,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    **kwargs
) -> Tuple[TimeSeriesAnomalyDetector, TimeSeriesAnomalyDetector, TimeSeriesAnomalyDetector]:
    """Create three models: sinusoidal PE, learnable PE, no PE."""
    model_sinusoidal = TimeSeriesAnomalyDetector(
        input_dim=input_dim, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers,
        use_positional_encoding=True, pe_type='sinusoidal', **kwargs
    )
    model_learnable = TimeSeriesAnomalyDetector(
        input_dim=input_dim, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers,
        use_positional_encoding=True, pe_type='learnable', **kwargs
    )
    model_nope = TimeSeriesAnomalyDetector(
        input_dim=input_dim, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers,
        use_positional_encoding=False, **kwargs
    )
    return model_sinusoidal, model_learnable, model_nope


if __name__ == "__main__":
    bs, slen, indim = 4, 100, 10

    m1, m2, m3 = create_model_trio(indim)
    x = torch.randn(bs, slen, indim)

    r1, _ = m1(x, return_attention=True)
    r2, _ = m2(x, return_attention=True)
    r3, _ = m3(x, return_attention=True)

    print(f"in: {x.shape}, out: {r1.shape}")
    for tag, m in [('sin', m1), ('learn', m2), ('nope', m3)]:
        print(f"  {tag}: {sum(p.numel() for p in m.parameters()):,} params")
