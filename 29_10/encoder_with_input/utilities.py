import csv
import math
import os
from pathlib import Path

import torch


def check_tensor(tensor, label):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f'Nonfinite {label}')


def gradient_summary(model):
    maximum, parameter, squared_norm = 0.0, None, 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        check_tensor(p.grad, 'gradient ' + name)
        gradient = p.grad.detach()
        value = gradient.abs().max().item()
        squared_norm += gradient.double().square().sum().item()
        if value > maximum:
            maximum, parameter = value, name
    # Float64 statistics only; gradients are neither changed nor clipped.
    return dict(max_abs=maximum, parameter=parameter, l2=math.sqrt(squared_norm))


def check_state(model, optimizer):
    for name, parameter in model.named_parameters():
        check_tensor(parameter, 'parameter ' + name)
    for state in optimizer.state.values():
        for name, tensor in state.items():
            if torch.is_tensor(tensor):
                check_tensor(tensor, 'optimizer ' + name)


def save_metrics(out, rows):
    path = Path(out) / 'metrics.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    columns = ['epoch', 'train_mse', 'validation_mse', 'train_r2', 'validation_r2']
    with open(temporary, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def plot_loss(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = [r['epoch'] for r in rows]
    fig, ax = plt.subplots()
    ax.plot(epochs, [r['train_mse'] for r in rows], label='Training')
    ax.plot(epochs, [r['validation_mse'] for r in rows], label='Validation')
    ax.set(xlabel='Epoch', ylabel='Mean sample MSE')
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out) / 'loss.png')
    plt.close(fig)
