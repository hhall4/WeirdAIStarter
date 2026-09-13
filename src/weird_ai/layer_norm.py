import torch
import torch.nn as nn


class LayerNorm(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()

        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        normalized = (x - mean) / torch.sqrt(variance + 1e-6)

        return self.scale * normalized + self.shift