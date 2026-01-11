"""
Embedding layer for mapping discrete token IDs to dense vector representations.
"""

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from jaxtyping import Float, Int


class Embedding(Module):
    """
    Lookup table for token embeddings.

    Maps integer token IDs to dense vectors by indexing into a learnable weight matrix.

    Attributes:
        weights: Learnable embedding matrix of shape (n_emb, emb_dim).
    """

    def __init__(
        self,
        n_emb: int,
        emb_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            n_emb: Number of embeddings (vocabulary size).
            emb_dim: Dimension of each embedding vector.
            device: Device to place the weights on.
            dtype: Data type for the weights.
        """
        super().__init__()
        self.n_emb = n_emb
        self.emb_dim = emb_dim

        self.weights = Parameter(torch.empty((n_emb, emb_dim), device=device, dtype=dtype))

    def forward(self, token_ids: Int[Tensor, "... seq"]) -> Float[Tensor, "... seq emb_dim"]:
        """Look up embeddings for the given token IDs.

        Args:
            token_ids: (..., seq) integer token IDs.

        Returns:
            (..., seq, emb_dim) embedding vectors.
        """
        return self.weights[token_ids]
