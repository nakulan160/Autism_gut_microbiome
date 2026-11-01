import random
from contextlib import nullcontext

import torch
from sklearn.metrics import r2_score

from data import choose_mask, prepare_sample
from model import masked_loss
from utilities import check_tensor, gradient_summary, check_state


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def attention_context(backend):
    if backend == 'auto':
        return nullcontext()
    if backend != 'math':
        raise ValueError('Attention backend must be auto or math')
    from torch.nn.attention import sdpa_kernel, SDPBackend
    return sdpa_kernel(SDPBackend.MATH)


def learning_rate(update, c):
    d = c['species_dim'] + c['abundance_dim']
    warmup = c['warmup_updates']
    return d ** -0.5 * min(update ** -0.5, update * warmup ** -1.5)


def make_optimizer(model, c):
    return torch.optim.Adam(model.parameters(), lr=0.0,
                            betas=(c['beta1'], c['beta2']), eps=c['adam_eps'],
                            weight_decay=c['weight_decay'])


def epoch_r2(targets, predictions):
    # Pooled across all masked entries, not averaged across samples.
    if len(targets) < 2:
        return None
    return float(r2_score(targets, predictions))


def train_epoch(model, optimizer, runs, train_indices, c, device, epoch, update):
    model.train()
    generator = torch.Generator().manual_seed(c['seed'] + epoch)
    order = torch.randperm(len(train_indices), generator=generator).tolist()
    order = [train_indices[j] for j in order]
    total, count = 0.0, 0
    targets, predictions = [], []
    for start in range(0, len(order), c['accumulation']):
        group = order[start:start + c['accumulation']]
        optimizer.zero_grad(set_to_none=True)
        with attention_context(c['attention_backend']):
            for i in group:
                label = f"epoch {epoch}, update {update + 1}, run {runs[i]['run_id']}"
                try:
                    generator = torch.Generator().manual_seed(c['training_mask_seed'] + epoch * c['training_mask_epoch_stride'] + i)
                    mask = choose_mask(len(runs[i]['ids']), c['mask_fraction'], generator)
                    ids, values, target_ids, target = prepare_sample(runs[i], mask, device)
                    output = model(ids, values)
                    check_tensor(output, 'training predictions')
                    loss = masked_loss(output, target_ids, target)
                    check_tensor(loss, 'training loss')
                    # Average over this actual group, including a short last group.
                    (loss / len(group)).backward()
                    stats = gradient_summary(model)
                except (RuntimeError, ValueError) as exc:
                    raise RuntimeError(f'{label}: {exc}') from exc
                total += loss.detach().item()
                count += 1
                targets.extend(target.detach().cpu().tolist())
                predictions.extend(output[target_ids - 1].detach().cpu().tolist())
        lr = learning_rate(update + 1, c)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        try:
            optimizer.step()  # No clipping and no finite-gradient cutoff.
            check_state(model, optimizer)
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(f'After Adam update {update + 1}, epoch {epoch}: {exc}') from exc
        update += 1
        if update % c['log_every_updates'] == 0 or count == len(order):
            print(f"Epoch {epoch:03d}: {count:,}/{len(order):,}; MSE={total/count:.7g}; "
                  f"lr={lr:.7g}; grad_norm={stats['l2']:.7g}", flush=True)
    return dict(mse=total / count, r2=epoch_r2(targets, predictions)), update


@torch.no_grad()
def validate(model, runs, indices, masks, device, c):
    model.eval()
    total = 0.0
    targets, predictions = [], []
    with attention_context(c['attention_backend']):
        for i in indices:
            ids, values, target_ids, target = prepare_sample(runs[i], masks[i], device)
            output = model(ids, values)
            check_tensor(output, 'validation predictions')
            loss = masked_loss(output, target_ids, target)
            check_tensor(loss, 'validation loss')
            total += loss.item()
            targets.extend(target.cpu().tolist())
            predictions.extend(output[target_ids - 1].cpu().tolist())
    return dict(mse=total / len(indices), r2=epoch_r2(targets, predictions))
