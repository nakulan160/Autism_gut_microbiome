import torch
from masking import prepare_sample
from losses import masked_loss
from diagnostics import check_tensor
from runtime import attention_context


@torch.no_grad()
def validate(model, runs, indices, masks, device, c):
    model.eval()
    total = 0.0
    pairs = []
    with attention_context(c["attention_backend"]):
        for i in indices:
            ids, a, target_ids, target = prepare_sample(runs[i], masks[i], device)
            pred = model(ids, a)
            loss = masked_loss(pred, target_ids, target)
            check_tensor(loss, "validation loss")
            total += float(loss)
            # At most 10000 diagnostic masked predictions; all samples contribute to loss.
            if len(pairs) < 10000:
                pairs.extend(
                    zip(target.cpu().tolist(), pred[target_ids - 1].cpu().tolist())
                )
                pairs = pairs[:10000]
    return total / len(indices), pairs
