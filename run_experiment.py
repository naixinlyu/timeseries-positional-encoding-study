"""
Main entry point. Trains three models (sinusoidal PE / learnable PE / no PE),
evaluates them on anomaly detection, and generates comparison plots.

Data seed is fixed across runs; model seed varies.

Usage: python run_experiment.py [--model_seed 42]
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.train import run_experiment as train_models
from experiments.evaluate import run_evaluation
from experiments.visualize import generate_all_visualizations


def main(model_seed: int = 42):
    print("Positional Encoding Experiment")
    print("-"*40)

    config = {
        'train_length': 10000,
        'test_length': 2000,
        'n_features': 5,
        'window_size': 100,
        'batch_size': 32,
        'anomaly_ratio': 0.05,
        'anomaly_type_weights': {
            'point': 0.2,
            'contextual': 0.6,
            'collective': 0.2
        },
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'epochs': 30,
        'lr': 1e-3,
        'data_seed': 42,
        'model_seed': model_seed
    }

    print("\nConfig:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n[Training]")

    results_dir, histories = train_models(config)

    print("\n[Evaluation]")

    eval_results = run_evaluation(results_dir, config)

    print("\n[Generating plots]")
    generate_all_visualizations(results_dir)

    print("\nDone.")

    print(f"\nResults: {results_dir}")

    m_sin = eval_results['metrics_sinusoidal_pe']
    m_learn = eval_results['metrics_learnable_pe']
    m_nope = eval_results['metrics_no_pe']

    print(f"\nF1: sin={m_sin['f1_score']:.4f}, learn={m_learn['f1_score']:.4f}, none={m_nope['f1_score']:.4f}")

    perm_s = eval_results['permutation_test_sinusoidal_pe']['mean_diff_after_reorder']
    perm_l = eval_results['permutation_test_learnable_pe']['mean_diff_after_reorder']
    perm_n = eval_results['permutation_test_no_pe']['mean_diff_after_reorder']

    print(f"Perm test diff: sin={perm_s:.4f}, learn={perm_l:.4f}, none={perm_n:.7f}")

    return results_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_seed', type=int, default=42)
    args = parser.parse_args()
    main(model_seed=args.model_seed)
