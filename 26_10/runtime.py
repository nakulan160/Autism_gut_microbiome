from contextlib import nullcontext
import random, torch


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def attention_context(backend):
    if backend == "auto":
        return nullcontext()
    from torch.nn.attention import sdpa_kernel, SDPBackend

    return sdpa_kernel(SDPBackend.MATH)


def learning_rate(update, c):
    d = c["species_dim"] + c["abundance_dim"]
    w = c["warmup_updates"]
    return d ** (-0.5) * min(update ** (-0.5), update * w ** (-1.5))
