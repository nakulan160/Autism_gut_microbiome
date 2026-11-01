import math, torch


def choose_mask(n, fraction, generator):
    return torch.randperm(n, generator=generator)[: max(1, math.floor(n * fraction))]


def prepare_sample(run, masked, device):
    """Remove complete pairs, keeping original FP32 decimal targets and 1-based species IDs."""
    visible = torch.ones(len(run["ids"]), dtype=torch.bool)
    visible[masked] = False
    return (
        run["ids"][visible].to(device),
        run["values"][visible].to(device),
        run["ids"][masked].to(device),
        run["values"][masked].to(device),
    )
