# plots

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anomaly_detector import TimeSeriesAnomalyDetector
from models.positional_encoding import SinusoidalPositionalEncoding
from data.synthetic_data import create_experiment_data


def set_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10


def plot_training_curves(history: dict, save_path: str = None):
    """Loss curves (train & val) for all three models."""
    model_keys = list(history.keys())
    fig, axes = plt.subplots(1, len(model_keys), figsize=(5 * len(model_keys), 4))
    if len(model_keys) == 1:
        axes = [axes]

    titles = {
        'sinusoidal_pe': 'Sinusoidal PE',
        'learnable_pe': 'Learnable PE',
        'no_pe': 'No PE',
        'with_pe': 'With PE',
        'without_pe': 'Without PE'
    }

    for ax, key in zip(axes, model_keys):
        h = history[key]
        ax.plot(h['train_loss'], label='Train Loss', color='blue')
        ax.plot(h['val_loss'], label='Val Loss', color='red')
        ax.axvline(h['best_epoch'] - 1, color='green', linestyle='--',
                   label=f"Best (epoch {h['best_epoch']})")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(titles.get(key, key))
        ax.legend()
        ax.set_yscale('log')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_attention_comparison(models: dict, sample_input: torch.Tensor, save_path: str = None):
    """Attention heatmaps for each model."""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        model.eval()
        with torch.no_grad():
            _, attn = model(sample_input, return_attention=True)

        attn_map = attn[0][0, 0].cpu().numpy()
        im = ax.imshow(attn_map, cmap='Blues', aspect='auto')
        ax.set_title(name)
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_reconstruction_comparison(
    originals: np.ndarray,
    recons_dict: dict,
    labels: np.ndarray,
    sample_idx: int = 0,
    feature_idx: int = 0,
    save_path: str = None
):
    """Original vs reconstruction for one sample, all models."""
    n_plots = 1 + len(recons_dict)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots), sharex=True)

    original = originals[sample_idx, :, feature_idx]
    sample_labels = labels[sample_idx]
    seq_len = len(original)
    x = np.arange(seq_len)

    # original
    axes[0].plot(x, original, 'b-', linewidth=1, label='Original')
    anomaly_mask = sample_labels == 1
    if anomaly_mask.any():
        axes[0].scatter(x[anomaly_mask], original[anomaly_mask], c='red', s=30,
                       label='Anomaly', zorder=5)
    axes[0].set_title('Original')
    axes[0].legend()
    axes[0].set_ylabel('Value')

    colors = ['green', 'purple', 'orange']
    for i, (name, recons) in enumerate(recons_dict.items()):
        ax = axes[i + 1]
        recon = recons[sample_idx, :, feature_idx]
        color = colors[i % len(colors)]
        mse = ((original - recon) ** 2).mean()

        ax.plot(x, original, 'b-', alpha=0.5, linewidth=1, label='Original')
        ax.plot(x, recon, color=color, linewidth=1, label='Reconstruction')
        error = np.abs(original - recon)
        ax.fill_between(x, original - error, original + error, alpha=0.3, color=color)
        ax.set_title(f'{name} (MSE: {mse:.6f})')
        ax.legend()
        ax.set_ylabel('Value')

    axes[-1].set_xlabel('Time Step')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_score_distribution(scores_dict: dict, labels: np.ndarray, save_path: str = None):
    """Histogram of anomaly scores split by normal/anomaly, one per model."""
    n = len(scores_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    labels_flat = labels.flatten()

    for ax, (name, scores) in zip(axes, scores_dict.items()):
        scores_flat = scores.flatten()
        ax.hist(scores_flat[labels_flat == 0], bins=50, alpha=0.7,
                label='Normal', color='blue', density=True)
        ax.hist(scores_flat[labels_flat == 1], bins=50, alpha=0.7,
                label='Anomaly', color='red', density=True)
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Density')
        ax.set_title(name)
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_positional_encoding_visualization(d_model: int = 64, max_len: int = 100, save_path: str = None):
    """Heatmap + per-dimension curves + cosine similarity matrix of the PE."""
    pe = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)
    encoding = pe.pe.squeeze(0).numpy()

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(encoding.T, cmap='RdBu', aspect='auto')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Dimension')
    ax1.set_title('Sinusoidal PE Heatmap')
    plt.colorbar(im, ax=ax1, label='Value')

    ax2 = fig.add_subplot(gs[1, 0])
    for dim in [0, 1, 10, 11, 30, 31]:
        ax2.plot(encoding[:, dim], label=f'dim {dim}')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Value')
    ax2.set_title('Selected Dimensions')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    encoding_norm = encoding / (np.linalg.norm(encoding, axis=1, keepdims=True) + 1e-8)
    similarity = encoding_norm @ encoding_norm.T
    im3 = ax3.imshow(similarity[:50, :50], cmap='viridis', aspect='auto')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Position')
    ax3.set_title('Cosine Similarity')
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_metrics_comparison(eval_results: dict, save_path: str = None):
    """Bar chart of precision / recall / F1 / AUC-ROC for all models."""
    # support both old (2-model) and new (3-model) key formats
    model_configs = []
    if 'metrics_sinusoidal_pe' in eval_results:
        model_configs = [
            ('Sinusoidal PE', eval_results['metrics_sinusoidal_pe']),
            ('Learnable PE', eval_results['metrics_learnable_pe']),
            ('No PE', eval_results['metrics_no_pe']),
        ]
    else:
        model_configs = [
            ('With PE', eval_results['metrics_with_pe']),
            ('Without PE', eval_results['metrics_without_pe']),
        ]

    metric_keys = ['precision', 'recall', 'f1_score', 'auc_roc']
    metric_labels = ['Precision', 'Recall', 'F1 Score', 'AUC-ROC']
    bar_colors = ['steelblue', 'mediumpurple', 'coral']

    x = np.arange(len(metric_labels))
    n = len(model_configs)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (name, metrics) in enumerate(model_configs):
        values = [metrics[m] for m in metric_keys]
        bars = ax.bar(x + (i - n / 2 + 0.5) * width, values, width,
                      label=name, color=bar_colors[i % len(bar_colors)])
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)

    ax.set_ylabel('Score')
    ax.set_title('Anomaly Detection Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def generate_all_visualizations(results_dir: str):
    """Generate all plots from a completed experiment run."""
    set_style()

    viz_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    with open(os.path.join(results_dir, 'training_history.json'), 'r') as f:
        history = json.load(f)
    with open(os.path.join(results_dir, 'evaluation_results.json'), 'r') as f:
        eval_results = json.load(f)

    scores_data = np.load(os.path.join(results_dir, 'scores.npz'))

    print("Generating visualizations...")

    print("  training curves")
    plot_training_curves(history, os.path.join(viz_dir, 'training_curves.png'))

    print("  metrics comparison")
    plot_metrics_comparison(eval_results, os.path.join(viz_dir, 'metrics_comparison.png'))

    # detect 3-model vs 2-model format
    is_trio = 'scores_sinusoidal_pe' in scores_data

    if is_trio:
        scores_dict = {
            'Sinusoidal PE': scores_data['scores_sinusoidal_pe'],
            'Learnable PE': scores_data['scores_learnable_pe'],
            'No PE': scores_data['scores_no_pe'],
        }
        recons_dict = {
            'Sinusoidal PE': scores_data['recons_sinusoidal_pe'],
            'Learnable PE': scores_data['recons_learnable_pe'],
            'No PE': scores_data['recons_no_pe'],
        }
    else:
        scores_dict = {
            'With PE': scores_data['scores_with_pe'],
            'Without PE': scores_data['scores_without_pe'],
        }
        recons_dict = {
            'With PE': scores_data['recons_with_pe'],
            'Without PE': scores_data['recons_without_pe'],
        }

    print("  score distributions")
    plot_score_distribution(
        scores_dict, scores_data['labels'],
        os.path.join(viz_dir, 'score_distribution.png')
    )

    print("  reconstruction comparison")
    plot_reconstruction_comparison(
        scores_data['originals'], recons_dict, scores_data['labels'],
        save_path=os.path.join(viz_dir, 'reconstruction_comparison.png')
    )

    print("  positional encoding")
    plot_positional_encoding_visualization(
        d_model=config['d_model'],
        save_path=os.path.join(viz_dir, 'positional_encoding.png')
    )

    print("  attention patterns")
    checkpoint_dir = os.path.join(results_dir, 'checkpoints')
    models = {}

    if is_trio:
        for name, fname, use_pe, pe_type in [
            ('Sinusoidal PE', 'model_sinusoidal_pe_best.pt', True, 'sinusoidal'),
            ('Learnable PE', 'model_learnable_pe_best.pt', True, 'learnable'),
            ('No PE', 'model_no_pe_best.pt', False, 'sinusoidal'),
        ]:
            m = TimeSeriesAnomalyDetector(
                input_dim=config['n_features'],
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                n_layers=config['n_layers'],
                use_positional_encoding=use_pe,
                pe_type=pe_type
            )
            ckpt = torch.load(os.path.join(checkpoint_dir, fname), map_location='cpu', weights_only=False)
            m.load_state_dict(ckpt['model_state_dict'])
            models[name] = m
    else:
        for name, fname, use_pe in [
            ('With PE', 'model_with_pe_best.pt', True),
            ('Without PE', 'model_without_pe_best.pt', False),
        ]:
            m = TimeSeriesAnomalyDetector(
                input_dim=config['n_features'],
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                n_layers=config['n_layers'],
                use_positional_encoding=use_pe
            )
            ckpt = torch.load(os.path.join(checkpoint_dir, fname), map_location='cpu', weights_only=False)
            m.load_state_dict(ckpt['model_state_dict'])
            models[name] = m

    sample_input = torch.tensor(scores_data['originals'][:1], dtype=torch.float32)
    plot_attention_comparison(models, sample_input, os.path.join(viz_dir, 'attention_comparison.png'))

    print(f"\nAll saved to: {viz_dir}")
    plt.close('all')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()

    generate_all_visualizations(args.results_dir)
