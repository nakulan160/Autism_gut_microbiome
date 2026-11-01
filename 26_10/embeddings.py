import torch
from torch import nn


def abundance_bins(values):
    """Round nonnegative abundance half-up; row 0 is valid, NOT padding."""
    if not torch.isfinite(values).all() or (values < 0).any() or (values > 100).any():
        raise ValueError("Abundance range")
    # Double only for bin arithmetic: avoid FP32 addition rounding a value just below .5 upward.
    return torch.floor(values.double() + 0.5).long()


class PairEmbedding(nn.Module):
    def __init__(self, vocabulary_size, species_dim, abundance_dim):
        super().__init__()
        self.species = nn.Embedding(vocabulary_size + 1, species_dim)
        self.abundance = nn.Embedding(101, abundance_dim)

    def forward(self, ids, values):
        return torch.cat(
            (self.species(ids), self.abundance(abundance_bins(values))), dim=-1
        )
