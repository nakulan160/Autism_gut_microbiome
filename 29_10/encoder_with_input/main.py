import argparse
import json
import math
from pathlib import Path

import torch

from data import load_data, make_split, choose_mask
from model import MicrobiomeEncoder
from engine import seed_all, make_optimizer, train_epoch, validate
from utilities import save_metrics, plot_loss


def load_config(path):
    c = json.loads(Path(path).read_text())
    required = {
        'input', 'output', 'device', 'dtype',
        'epochs', 'seed', 'validation_fraction', 'mask_fraction',
        'excluded_features', 'species_dim', 'abundance_dim', 'abundance_max',
        'abundance_bin_width', 'layers', 'heads', 'feedforward',
        'dropout', 'encoder_activation', 'head_dims', 'head_activation',
        'norm_first', 'layer_norm_eps', 'bias', 'accumulation',
        'beta1', 'beta2', 'adam_eps', 'weight_decay',
        'warmup_updates', 'attention_backend', 'training_mask_seed', 'training_mask_epoch_stride',
        'validation_mask_seed', 'log_every_updates', 'load_log_every_runs', 'plot_every',
    }
    if set(c) != required:
        raise ValueError(f'Missing settings: {required-set(c)}; unknown settings: {set(c)-required}')
    for key in ['epochs', 'species_dim', 'abundance_dim', 'layers', 'heads',
                'feedforward', 'accumulation', 'warmup_updates', 'training_mask_epoch_stride',
                'log_every_updates', 'load_log_every_runs', 'plot_every']:
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ['seed', 'training_mask_seed', 'validation_mask_seed']:
        if type(c[key]) is not int or c[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if not 0 < c['validation_fraction'] < 1 or not 0 < c['mask_fraction'] <= 1:
        raise ValueError('Invalid split or masking fraction')
    if (c['species_dim'] + c['abundance_dim']) % c['heads']:
        raise ValueError('Combined embedding dimension must be divisible by heads')
    if c['attention_backend'] not in ['auto', 'math'] or not 0 <= c['dropout'] < 1:
        raise ValueError('Invalid attention backend or dropout')
    if not 0 <= c['beta1'] < 1 or not 0 <= c['beta2'] < 1:
        raise ValueError('Invalid Adam beta')
    if not math.isfinite(c['adam_eps']) or c['adam_eps'] <= 0:
        raise ValueError('Adam epsilon must be finite and positive')
    if not c['head_dims'] or any(type(x) is not int or x < 1 for x in c['head_dims']):
        raise ValueError('head_dims must contain positive integers')
    if not isinstance(c['excluded_features'], list):
        raise ValueError('excluded_features must be a list')
    if c['dtype'] not in ['float32', 'float64']:
        raise ValueError('dtype must be float32 or float64')
    if c['encoder_activation'] not in ['relu', 'gelu'] or c['head_activation'] not in ['relu', 'gelu']:
        raise ValueError('Activation must be relu or gelu')
    if type(c['norm_first']) is not bool or type(c['bias']) is not bool:
        raise ValueError('norm_first and bias must be true or false')
    for key in ['layer_norm_eps', 'abundance_max', 'abundance_bin_width']:
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not math.isfinite(c['weight_decay']) or c['weight_decay'] < 0:
        raise ValueError('weight_decay must be finite and nonnegative')
    ratio = c['abundance_max'] / c['abundance_bin_width']
    if not math.isfinite(ratio) or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-9):
        raise ValueError('abundance_max must be divisible by abundance_bin_width')
    return c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('settings.json')))
    args = parser.parse_args()
    c = load_config(args.config)
    seed_all(c['seed'])
    device = torch.device(c['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; set device to cpu in settings.json')
    out = Path(c['output']).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output folder is not empty; choose a new output in settings.json')
    out.mkdir(parents=True, exist_ok=True)
    runs, vocabulary, digest = load_data(
        c['input'], c['excluded_features'], getattr(torch, c['dtype']),
        c['abundance_max'], c['load_log_every_runs'])
    training, validation = make_split(len(runs), c['validation_fraction'], c['seed'])
    # Validation masks stay identical across epochs.
    masks = {i: choose_mask(len(runs[i]['ids']), c['mask_fraction'],
             torch.Generator().manual_seed(c['validation_mask_seed'] + i)) for i in validation}
    model = MicrobiomeEncoder(c, len(vocabulary)).to(device=device, dtype=getattr(torch, c['dtype']))
    optimizer = make_optimizer(model, c)
    print(f'Runs: {len(runs):,}; species: {len(vocabulary):,}; '
          f'training: {len(training):,}; validation: {len(validation):,}', flush=True)
    print(f'Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}; '
          f'epochs: {c["epochs"]}; {c["dtype"]}; no clipping; no model saving', flush=True)
    print(f'Input SHA256: {digest}', flush=True)
    rows, update = [], 0
    for epoch in range(1, c['epochs'] + 1):
        train, update = train_epoch(model, optimizer, runs, training, c, device, epoch, update)
        val = validate(model, runs, validation, masks, device, c)
        rows.append(dict(epoch=epoch, train_mse=train['mse'], validation_mse=val['mse'],
                         train_r2=train['r2'], validation_r2=val['r2']))
        save_metrics(out, rows)
        if epoch % c['plot_every'] == 0 or epoch == c['epochs']:
            plot_loss(out, rows)
        print(f"Completed epoch {epoch}: train MSE={train['mse']:.7g}; "
              f"validation MSE={val['mse']:.7g}; train R2={train['r2']}; "
              f"validation R2={val['r2']}", flush=True)
    print(f'Completed. Results: {out}', flush=True)


if __name__ == '__main__':
    main()
