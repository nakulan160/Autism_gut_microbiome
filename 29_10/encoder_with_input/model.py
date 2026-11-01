import torch
from torch import nn
from torch.nn import functional as F


def abundance_bins(values, maximum, width):
    if not torch.isfinite(values).all() or (values < 0).any() or (values > maximum).any():
        raise ValueError('Abundance outside configured range')
    # Round half upward. Float64 is used only for bin arithmetic.
    return torch.floor(values.double() / width + 0.5).long()


class PairEmbedding(nn.Module):
    def __init__(self, vocabulary_size, c):
        super().__init__()
        self.species = nn.Embedding(vocabulary_size + 1, c['species_dim'])
        self.maximum, self.width = c['abundance_max'], c['abundance_bin_width']
        self.abundance = nn.Embedding(round(self.maximum / self.width) + 1, c['abundance_dim'])

    def forward(self, ids, values):
        return torch.cat((self.species(ids),
                          self.abundance(abundance_bins(values, self.maximum, self.width))), dim=-1)


class MicrobiomeEncoder(nn.Module):
    def __init__(self, c, vocabulary_size):
        super().__init__()
        d = c['species_dim'] + c['abundance_dim']
        self.embedding = PairEmbedding(vocabulary_size, c)
        self.cls = nn.Embedding(1, d)
        # Construct each encoder independently; use the configured LayerNorm order.
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d, nhead=c['heads'], dim_feedforward=c['feedforward'],
                dropout=c['dropout'], activation=c['encoder_activation'], batch_first=True,
                norm_first=c['norm_first'], layer_norm_eps=c['layer_norm_eps'], bias=c['bias'],
            ) for _ in range(c['layers'])
        ])
        dims = [d] + c['head_dims'] + [vocabulary_size]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=c['bias']))
            if i < len(dims) - 2:
                layers.append(nn.ReLU() if c['head_activation'] == 'relu' else nn.GELU())
        self.head = nn.Sequential(*layers)

    def encode(self, ids, values):
        x = self.embedding(ids, values).unsqueeze(0)
        x = torch.cat((self.cls.weight.unsqueeze(0), x), dim=1)
        for block in self.blocks:
            x = block(x)
        return x[:, 0, :]  # Final CLS embedding: [1, 512] with agreed settings.

    def forward(self, ids, values):
        return self.head(self.encode(ids, values)).squeeze(0)


def masked_loss(predictions, ids, targets):
    if predictions.ndim != 1 or ids.ndim != 1 or targets.shape != ids.shape or ids.numel() == 0:
        raise ValueError('Expected one output vector and matching nonempty ID/target vectors')
    if ids.dtype != torch.long or (ids < 1).any() or (ids > predictions.numel()).any():
        raise ValueError('Masked species IDs must be int64 within 1..vocabulary_size')
    if ids.unique().numel() != ids.numel():
        raise ValueError('Duplicate masked species ID')
    # Species ID 1 maps to output position 0. Targets remain original decimals.
    return F.mse_loss(predictions[ids - 1], targets, reduction='mean')
