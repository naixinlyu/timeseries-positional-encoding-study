import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anomaly_detector import TimeSeriesAnomalyDetector, create_model_trio
from data.synthetic_data import create_experiment_data


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = 'cuda',
    model_name: str = 'model',
    save_dir: str = 'checkpoints'
) -> dict:
    """Train a single model, save best checkpoint by val loss. Returns history dict."""
    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)

    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }

    print(f"\n-- {model_name} --")
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Epochs: {epochs}, LR: {lr}")
    print()

    for epoch in range(epochs):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_x, _ in pbar:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            reconstruction, _ = model(batch_x)
            loss = criterion(reconstruction, batch_x)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        # validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                reconstruction, _ = model(batch_x)
                loss = criterion(reconstruction, batch_x)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, os.path.join(save_dir, f'{model_name}_best.pt'))

        scheduler.step()

        print(f'Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'history': history
    }, os.path.join(save_dir, f'{model_name}_final.pt'))

    print(f"\nBest epoch {history['best_epoch']}, val_loss: {history['best_val_loss']:.6f}")

    return history


def run_experiment(config: dict = None):
    """Run full training: generate data, train all three models, save results."""
    if config is None:
        config = {
            'train_length': 10000,
            'test_length': 2000,
            'n_features': 5,
            'window_size': 100,
            'batch_size': 32,
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 2,
            'epochs': 30,
            'lr': 1e-3,
            'data_seed': 42,
            'model_seed': 42
        }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # data generation uses fixed data_seed
    print("\nGenerating data...")
    train_loader, test_loader, test_data, test_labels = create_experiment_data(
        train_length=config['train_length'],
        test_length=config['test_length'],
        n_features=config['n_features'],
        window_size=config['window_size'],
        batch_size=config['batch_size'],
        anomaly_type_weights=config.get('anomaly_type_weights'),
        seed=config['data_seed']
    )

    val_loader = test_loader

    print(f"Training samples: {len(train_loader) * config['batch_size']}")
    print(f"Validation samples: {len(test_loader) * config['batch_size']}")

    # model initialization uses model_seed
    torch.manual_seed(config['model_seed'])
    np.random.seed(config['model_seed'])

    print("\nCreating models...")
    m_sin, m_learn, m_nope = create_model_trio(
        input_dim=config['n_features'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers']
    )

    run_name = f"run_seed{config['model_seed']}"
    results_dir = os.path.join('results', run_name)
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    checkpoint_dir = os.path.join(results_dir, 'checkpoints')

    hist_sin = train_model(
        m_sin, train_loader, val_loader,
        epochs=config['epochs'], lr=config['lr'],
        device=device, model_name='model_sinusoidal_pe',
        save_dir=checkpoint_dir
    )

    hist_learn = train_model(
        m_learn, train_loader, val_loader,
        epochs=config['epochs'], lr=config['lr'],
        device=device, model_name='model_learnable_pe',
        save_dir=checkpoint_dir
    )

    hist_nope = train_model(
        m_nope, train_loader, val_loader,
        epochs=config['epochs'], lr=config['lr'],
        device=device, model_name='model_no_pe',
        save_dir=checkpoint_dir
    )

    histories = {
        'sinusoidal_pe': hist_sin,
        'learnable_pe': hist_learn,
        'no_pe': hist_nope
    }

    with open(os.path.join(results_dir, 'training_history.json'), 'w') as f:
        json.dump(histories, f, indent=2)

    print(f"\nTraining done -> {results_dir}")
    print(f"Best val loss: sin={hist_sin['best_val_loss']:.6f}, learn={hist_learn['best_val_loss']:.6f}, nope={hist_nope['best_val_loss']:.6f}")

    return results_dir, histories


if __name__ == "__main__":
    results_dir, histories = run_experiment()
