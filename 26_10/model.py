import torch
from torch import nn
from embeddings import PairEmbedding


class MicrobiomeEncoder(nn.Module):
    def __init__(self, c, vocabulary_size):
        super().__init__()
        d = c["species_dim"] + c["abundance_dim"]
        self.embedding = PairEmbedding(
            vocabulary_size, c["species_dim"], c["abundance_dim"]
        )
        self.cls = nn.Embedding(1, d)
        # Separate constructors independently initialize each layer.
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d,
                    c["heads"],
                    c["feedforward"],
                    c["dropout"],
                    activation="relu",
                    batch_first=True,
                    norm_first=False,
                    layer_norm_eps=1e-5,
                )
                for _ in range(c["layers"])
            ]
        )
        dims = [d] + c["head_dims"] + [vocabulary_size]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.head = nn.Sequential(*layers)

    def encode(self, ids, values):
        x = self.embedding(ids, values).unsqueeze(0)
        x = torch.cat((self.cls.weight.unsqueeze(0), x), dim=1)
        for block in self.blocks:
            x = block(x)
        return x[:, 0, :]

    def forward(self, ids, values):
        return self.head(self.encode(ids, values)).squeeze(0)
