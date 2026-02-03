"""
Synthetic time series data generation with anomaly injection.
We use synthetic data so ground truth labels are known exactly.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import matplotlib.pyplot as plt


def generate_seasonal_pattern(
    length: int,
    n_features: int,
    periods: list = [24, 168],
    noise_level: float = 0.1,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate multi-feature time series with seasonal components + slow trend + noise.
    Returns array of shape (length, n_features).
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(length)
    data = np.zeros((length, n_features))

    for i in range(n_features):
        signal = np.zeros(length)

        for period in periods:
            amplitude = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2 * np.pi)
            signal += amplitude * np.sin(2 * np.pi * t / period + phase)

        trend = np.random.uniform(-0.001, 0.001) * t
        signal += trend

        noise = np.random.normal(0, noise_level, length)
        signal += noise

        data[:, i] = signal

    return data


def inject_anomalies(
    data: np.ndarray,
    anomaly_ratio: float = 0.05,
    anomaly_types: list = ['point', 'contextual', 'collective'],
    type_weights: dict = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject anomalies into time series. Three types:
      - point: extreme single values
      - contextual: value copied from a different position (position-dependent)
      - collective: level shift over a short subsequence
    type_weights: optional dict mapping type name to proportion, e.g. {'contextual': 0.6, 'point': 0.2, 'collective': 0.2}.
    Returns (modified_data, binary_labels).
    """
    if seed is not None:
        np.random.seed(seed)

    length, n_features = data.shape
    data_anomaly = data.copy()
    labels = np.zeros(length, dtype=int)
    n_anomalies = int(length * anomaly_ratio)

    for anomaly_type in anomaly_types:
        if type_weights and anomaly_type in type_weights:
            n_this_type = int(n_anomalies * type_weights[anomaly_type])
        else:
            n_this_type = n_anomalies // len(anomaly_types)

        if anomaly_type == 'point':
            indices = np.random.choice(length, n_this_type, replace=False)
            for idx in indices:
                feat = np.random.randint(n_features)
                data_anomaly[idx, feat] += np.random.choice([-1, 1]) * \
                    np.random.uniform(3, 5) * data[:, feat].std()
                labels[idx] = 1

        elif anomaly_type == 'contextual':
            # swap a value from a different temporal position
            indices = np.random.choice(length - 50, n_this_type, replace=False)
            for idx in indices:
                source_idx = (idx + np.random.randint(20, 50)) % length
                feat = np.random.randint(n_features)
                data_anomaly[idx, feat] = data[source_idx, feat]
                labels[idx] = 1

        elif anomaly_type == 'collective':
            start_indices = np.random.choice(length - 10, n_this_type // 5, replace=False)
            for start_idx in start_indices:
                end_idx = min(start_idx + np.random.randint(5, 10), length)
                feat = np.random.randint(n_features)
                shift = np.random.uniform(2, 4) * data[:, feat].std()
                data_anomaly[start_idx:end_idx, feat] += shift
                labels[start_idx:end_idx] = 1

    return data_anomaly, labels


class TimeSeriesDataset(Dataset):
    """Sliding-window dataset over a time series."""

    def __init__(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        window_size: int = 100,
        stride: int = 1
    ):
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.n_windows = (len(data) - window_size) // stride + 1

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        start = idx * self.stride
        end = start + self.window_size

        window = torch.tensor(self.data[start:end], dtype=torch.float32)

        if self.labels is not None:
            window_labels = torch.tensor(self.labels[start:end], dtype=torch.float32)
            return window, window_labels

        return window, torch.zeros(self.window_size)


def create_experiment_data(
    train_length: int = 10000,
    test_length: int = 2000,
    n_features: int = 5,
    window_size: int = 100,
    batch_size: int = 32,
    anomaly_ratio: float = 0.05,
    anomaly_type_weights: dict = None,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """
    Build train/test DataLoaders. Training data is clean; test data has anomalies.
    Returns (train_loader, test_loader, raw_test_data, test_labels).
    """
    train_data = generate_seasonal_pattern(train_length, n_features, seed=seed)

    test_data_normal = generate_seasonal_pattern(test_length, n_features, seed=seed + 1)
    test_data, test_labels = inject_anomalies(
        test_data_normal, anomaly_ratio, type_weights=anomaly_type_weights, seed=seed + 2
    )

    train_dataset = TimeSeriesDataset(
        train_data, labels=None, window_size=window_size, stride=window_size // 2
    )
    test_dataset = TimeSeriesDataset(
        test_data, labels=test_labels, window_size=window_size, stride=1
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader, test_data, test_labels


def visualize_data(
    data: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Time Series Data",
    n_features_to_show: int = 3
) -> plt.Figure:
    """Plot time series features with anomaly points highlighted in red."""
    n_features = min(n_features_to_show, data.shape[1])

    fig, axes = plt.subplots(n_features, 1, figsize=(14, 3 * n_features), sharex=True)
    if n_features == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(data[:, i], 'b-', alpha=0.7, linewidth=0.8, label='Normal')

        if labels is not None:
            anomaly_mask = labels == 1
            ax.scatter(
                np.where(anomaly_mask)[0],
                data[anomaly_mask, i],
                c='red', s=20, label='Anomaly', zorder=5
            )

        ax.set_ylabel(f'Feature {i+1}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time Step')
    axes[0].set_title(title)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Generating synthetic data...")

    train_loader, test_loader, test_data, test_labels = create_experiment_data(
        train_length=5000, test_length=1000,
        n_features=5, window_size=100, batch_size=32
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    print(f"Test shape:    {test_data.shape}")
    print(f"Anomaly ratio: {test_labels.mean():.2%}")

    fig = visualize_data(test_data[:500], test_labels[:500], "Test Data with Anomalies")
    plt.savefig("synthetic_data_visualization.png", dpi=150)
    plt.show()

    for batch_x, batch_labels in train_loader:
        print(f"\nBatch shape:  {batch_x.shape}")
        print(f"Labels shape: {batch_labels.shape}")
        break
