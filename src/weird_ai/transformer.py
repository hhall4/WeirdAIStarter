from torch import nn as nn

from .layer_norm import LayerNorm
from .attention import SelfAttention
from .feed_forward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(
        self,
        emb_dim,
        context_length,
        num_heads,
        dropout,
        qkv_bias=False
    ):
        super().__init__()

        self.norm1 = LayerNorm(emb_dim)
        self.attention = SelfAttention(
            emb_dim,
            emb_dim,
            qkv_bias
        )

        self.norm2 = LayerNorm(emb_dim)
        self.feed_forward = FeedForward(emb_dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))

        return x