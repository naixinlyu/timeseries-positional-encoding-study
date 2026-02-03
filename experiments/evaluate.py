# eval + permutation test

import os
import sys
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anomaly_detector import TimeSeriesAnomalyDetector
from data.synthetic_data import create_experiment_data


def load_model(checkpoint_path: str, config: dict, use_pe: bool, pe_type: str = 'sinusoidal') -> TimeSeriesAnomalyDetector:
    """Load a trained model from checkpoint."""
    model = TimeSeriesAnomalyDetector(
        input_dim=config['n_features'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        use_positional_encoding=use_pe,
        pe_type=pe_type
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def compute_anomaly_scores(
    model: TimeSeriesAnomalyDetector,
    data_loader,
    device: str = 'cuda'
) -> tuple:
    """Compute per-point reconstruction MSE for every test window."""
    model = model.to(device)
    model.eval()

    all_scores = []
    all_labels = []
    all_recons = []
    all_originals = []

    with torch.no_grad():
        for batch_x, batch_labels in tqdm(data_loader, desc='Computing scores'):
            batch_x = batch_x.to(device)
            reconstruction, _ = model(batch_x)
            mse = ((batch_x - reconstruction) ** 2).mean(dim=-1)

            all_scores.append(mse.cpu().numpy())
            all_labels.append(batch_labels.numpy())
            all_recons.append(reconstruction.cpu().numpy())
            all_originals.append(batch_x.cpu().numpy())

    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    reconstructions = np.concatenate(all_recons, axis=0)
    originals = np.concatenate(all_originals, axis=0)

    return scores, labels, reconstructions, originals


def compute_metrics(scores: np.ndarray, labels: np.ndarray, threshold_percentile: float = 95) -> dict:
    """Precision / Recall / F1 / AUC-ROC from anomaly scores."""
    scores_flat = scores.flatten()
    labels_flat = labels.flatten().astype(int)

    threshold = np.percentile(scores_flat, threshold_percentile)
    predictions = (scores_flat > threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_flat, predictions, average='binary', zero_division=0
    )

    try:
        auc = roc_auc_score(labels_flat, scores_flat)
    except ValueError:
        auc = 0.5

    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc),
        'threshold': float(threshold),
        'mean_score': float(scores_flat.mean()),
        'std_score': float(scores_flat.std()),
        'mean_score_normal': float(scores_flat[labels_flat == 0].mean()),
        'mean_score_anomaly': float(scores_flat[labels_flat == 1].mean()) if labels_flat.sum() > 0 else 0.0
    }

    return metrics


def test_permutation_invariance(
    model: TimeSeriesAnomalyDetector,
    data_loader,
    device: str = 'cuda',
    n_samples: int = 10
) -> dict:
    """
    Shuffle input, run model, reorder output, compare with original.
    Without PE the diff should be ~0 (equivariant); with PE it should be large.
    """
    model = model.to(device)
    model.eval()

    results = {
        'max_diff_after_reorder': [],
        'max_diff_shuffled_vs_original': []
    }

    with torch.no_grad():
        for i, (batch_x, _) in enumerate(data_loader):
            if i >= n_samples:
                break

            batch_x = batch_x.to(device)
            seq_len = batch_x.size(1)

            output_original, _ = model(batch_x)

            perm = torch.randperm(seq_len)
            batch_x_shuffled = batch_x[:, perm, :]
            output_shuffled, _ = model(batch_x_shuffled)

            output_shuffled_reordered = output_shuffled[:, torch.argsort(perm), :]

            diff_after_reorder = (output_original - output_shuffled_reordered).abs().max().item()
            diff_shuffled_vs_original = (output_original - output_shuffled).abs().mean().item()

            results['max_diff_after_reorder'].append(diff_after_reorder)
            results['max_diff_shuffled_vs_original'].append(diff_shuffled_vs_original)

    results['mean_diff_after_reorder'] = float(np.mean(results['max_diff_after_reorder']))
    results['mean_diff_shuffled_vs_original'] = float(np.mean(results['max_diff_shuffled_vs_original']))

    return results


def _eval_one_model(name, model, test_loader, device):
    """Evaluate a single model: anomaly scores, metrics, permutation test."""
    print(f"\n>>> {name}")

    scores, labels, recons, originals = compute_anomaly_scores(model, test_loader, device)
    metrics = compute_metrics(scores, labels)
    perm_test = test_permutation_invariance(model, test_loader, device)

    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    print(f"\nPermutation test:")
    print(f"  diff after reorder: {perm_test['mean_diff_after_reorder']:.6f}")

    return scores, labels, recons, originals, metrics, perm_test


def run_evaluation(results_dir: str, config: dict = None):
    """Evaluate all three models, print comparison, save metrics + scores."""
    if config is None:
        with open(os.path.join(results_dir, 'config.json'), 'r') as f:
            config = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("\nGenerating test data...")
    _, test_loader, _, _ = create_experiment_data(
        train_length=config['train_length'],
        test_length=config['test_length'],
        n_features=config['n_features'],
        window_size=config['window_size'],
        batch_size=config['batch_size'],
        anomaly_type_weights=config.get('anomaly_type_weights'),
        seed=config['data_seed']
    )

    print("\nLoading models...")
    checkpoint_dir = os.path.join(results_dir, 'checkpoints')

    m_sin = load_model(
        os.path.join(checkpoint_dir, 'model_sinusoidal_pe_best.pt'),
        config, use_pe=True, pe_type='sinusoidal'
    )
    m_learn = load_model(
        os.path.join(checkpoint_dir, 'model_learnable_pe_best.pt'),
        config, use_pe=True, pe_type='learnable'
    )
    m_nope = load_model(
        os.path.join(checkpoint_dir, 'model_no_pe_best.pt'),
        config, use_pe=False
    )

    sc_sin, labels, rec_sin, originals, met_sin, perm_sin = \
        _eval_one_model("Sinusoidal PE", m_sin, test_loader, device)

    sc_learn, _, rec_learn, _, met_learn, perm_learn = \
        _eval_one_model("Learnable PE", m_learn, test_loader, device)

    sc_nope, _, rec_nope, _, met_nope, perm_nope = \
        _eval_one_model("No PE", m_nope, test_loader, device)

    print(f"\n--- Comparison ---")
    print(f"F1:  sin={met_sin['f1_score']:.4f}  learn={met_learn['f1_score']:.4f}  nope={met_nope['f1_score']:.4f}")
    print(f"AUC: sin={met_sin['auc_roc']:.4f}  learn={met_learn['auc_roc']:.4f}  nope={met_nope['auc_roc']:.4f}")
    print(f"Perm diff: sin={perm_sin['mean_diff_after_reorder']:.4f}  learn={perm_learn['mean_diff_after_reorder']:.4f}  nope={perm_nope['mean_diff_after_reorder']:.7f}")

    eval_results = {
        'metrics_sinusoidal_pe': met_sin,
        'metrics_learnable_pe': met_learn,
        'metrics_no_pe': met_nope,
        'permutation_test_sinusoidal_pe': perm_sin,
        'permutation_test_learnable_pe': perm_learn,
        'permutation_test_no_pe': perm_nope
    }

    with open(os.path.join(results_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)

    np.savez(
        os.path.join(results_dir, 'scores.npz'),
        scores_sinusoidal_pe=sc_sin,
        scores_learnable_pe=sc_learn,
        scores_no_pe=sc_nope,
        labels=labels,
        originals=originals,
        recons_sinusoidal_pe=rec_sin,
        recons_learnable_pe=rec_learn,
        recons_no_pe=rec_nope
    )

    print(f"\nResults saved to: {results_dir}")

    return eval_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()

    run_evaluation(args.results_dir)
