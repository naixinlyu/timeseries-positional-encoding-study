# PE implementations

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal PE from Vaswani et al. 2017.
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """Learnable position embeddings, similar to GPT-2 / BERT."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class NoPositionalEncoding(nn.Module):
    """Pass-through (no position info). Used as control in the experiment."""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)


def visualize_positional_encoding(d_model: int = 64, max_len: int = 100):
    """Plot a heatmap of sinusoidal PE values."""
    import matplotlib.pyplot as plt

    pe = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)
    encoding = pe.pe.squeeze(0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im = axes[0].imshow(encoding, cmap='RdBu', aspect='auto')
    axes[0].set_xlabel('Embedding Dimension')
    axes[0].set_ylabel('Position')
    axes[0].set_title('Sinusoidal Positional Encoding')
    plt.colorbar(im, ax=axes[0])

    for dim in [0, 1, 4, 5, 10, 11]:
        axes[1].plot(range(max_len), encoding[:, dim], label=f'dim {dim}')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Encoding Value')
    axes[1].set_title('Selected Dimensions')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = visualize_positional_encoding()
    plt.savefig("positional_encoding_visualization.png", dpi=150)
    plt.show()
