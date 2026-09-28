import torch
import torch.nn as nn

from .layer_norm import LayerNorm
from .attention import CausalAttention
from .transformer import TransformerBlock


class WeirdAIModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.context_length = 256
        self.emb_dim = 768
        self.num_heads = 12
        self.num_layers = 12
        self.dropout = 0.1
        self.qkv_bias = False

        self.token_embedding = nn.Embedding(
            vocab_size,
            self.emb_dim
        )

        self.position_embedding = nn.Embedding(
            self.context_length,
            self.emb_dim
        )

        self.dropout_layer = nn.Dropout(self.dropout)

        self.transformer_blocks = nn.ModuleList()

        for _ in range(self.num_layers):
            block = TransformerBlock(
                self.emb_dim,
                self.context_length,
                self.num_heads,
                self.dropout,
                self.qkv_bias
            )

            block.attention = CausalAttention(
                self.emb_dim,
                self.emb_dim,
                self.context_length,
                self.dropout,
                self.qkv_bias
            )

            self.transformer_blocks.append(block)

        self.final_norm = LayerNorm(self.emb_dim)

        self.output_head = nn.Linear(
            self.emb_dim,
            vocab_size,
            bias=False
        )

    def forward(self, x):
        batch_size, num_tokens = x.shape

        if num_tokens > self.context_length:
            raise ValueError(
                f"Input contains {num_tokens} tokens, "
                f"but the maximum context length is {self.context_length}."
            )

        positions = torch.arange(
            num_tokens,
            device=x.device
        )

        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings
        x = self.dropout_layer(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.output_head(x)

        return logits