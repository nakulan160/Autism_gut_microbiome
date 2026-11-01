import math, torch


def make_split(n, fraction, seed):
    """Fixed random run split; validation count=ceil(n*fraction)."""
    nv = min(n - 1, max(1, math.ceil(n * fraction)))
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return order[nv:], order[:nv]
