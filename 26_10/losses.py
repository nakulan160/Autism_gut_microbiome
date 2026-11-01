from torch.nn import functional as F
from checks import check_loss_inputs


def masked_loss(predictions, ids, targets):
    check_loss_inputs(predictions, ids, targets)
    selected = predictions[ids - 1]
    if selected.shape != targets.shape:
        raise ValueError("Loss broadcasting prohibited")
    return F.mse_loss(selected, targets, reduction="mean")
