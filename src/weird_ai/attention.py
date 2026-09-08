import torch
import torch.nn as nn


class SimpleSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        scores = inputs @ inputs.T
        weights = torch.softmax(scores, dim=-1)
        context_vectors = weights @ inputs

        return context_vectors, weights
        """
        Args:
            x: Tensor of shape (num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (num_tokens, embedding_dim)
            attention_weights: Tensor of shape (num_tokens, num_tokens)
        """

class SelfAttention(nn.Module):
    """
    Trainable self-attention using query, key, and value projections.
    """

    def __init__(self, embedding_dim, output_dim, qkv_bias=False):
        super().__init__()

        self.query = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.key = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.value = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (num_tokens, output_dim)
            attention_weights: Tensor of shape (num_tokens, num_tokens)
        """

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = queries @ keys.T
        scores = scores / keys.shape[-1] ** 0.5

        weights = torch.softmax(scores, dim=-1)

        context_vectors = weights @ values

        return context_vectors, weights 

        raise NotImplementedError("Implement trainable self-attention.")

class CausalAttention(nn.Module):
    """
    Self-attention with a causal mask so tokens cannot attend to future tokens.
    """

    def __init__(self, embedding_dim, output_dim, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        self.query = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.key = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.value = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (batch_size, num_tokens, output_dim)
        """

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = queries @ keys.transpose(1, 2)
        scores = scores / keys.shape[-1] ** 0.5

        num_tokens = x.shape[1]
        scores = scores.masked_fill(
            self.mask[:num_tokens, :num_tokens].bool(),
            float("-inf")
        )

        weights = torch.softmax(scores, dim=-1)

        weights = self.dropout(weights)

        context_vectors = weights @ values

        return context_vectors 

        raise NotImplementedError("Implement causal attention.")