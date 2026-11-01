"""Shared scientific checks: fail explicitly instead of broadcasting or remapping silently."""

import torch


def check_loss_inputs(predictions, ids, targets):
    if (
        predictions.ndim != 1
        or ids.ndim != 1
        or targets.shape != ids.shape
        or ids.numel() == 0
    ):
        raise ValueError(
            "Expected vocabulary vector and equal nonempty masked-ID/target vectors"
        )
    if ids.dtype != torch.long:
        raise ValueError("Species IDs must be int64")
    if (ids < 1).any() or (ids > predictions.numel()).any():
        raise ValueError("Species ID outside vocabulary")
    if ids.unique().numel() != ids.numel():
        raise ValueError("Duplicate masked species ID")
