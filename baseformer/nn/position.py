"""
Rotary Positional Embedding (RoPE) for transformer attention.
"""

import torch
from torch import Tensor
from torch.nn import Module
from jaxtyping import Float, Int

from einops import einsum, rearrange


class RotaryPositionalEmbedding(Module):
    """
    Applies Rotary Positional Embedding (RoPE) to input tensors.

    RoPE encodes position information by rotating pairs of dimensions
    in the embedding space. Each position has a unique rotation matrix
    that is applied to consecutive pairs of features.

    Attributes:
        theta: Base frequency for the rotational encoding.
        d_k: Dimension of the input embeddings (must be even).
        max_seq_len: Maximum sequence length supported.
        rotation_matrices: Precomputed rotation matrices of shape
            (max_seq_len, d_k/2, 2, 2).
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ):
        """
        Args:
            theta: Base frequency for rotational encoding (typically 10000).
            d_k: Dimension of input embeddings (must be even).
            max_seq_len: Maximum sequence length to precompute rotations for.
            device: Device to place the rotation matrices on.
        """
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even, got {d_k}")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Compute rotation angles for each position and dimension pair
        positions = torch.arange(max_seq_len, device=device)
        dim_indices = torch.arange(d_k // 2, device=device)

        # θ_k = theta^(-2k/d_k)
        exponents = -2 * dim_indices / d_k
        thetas = theta ** exponents  # (d_k/2,)

        # angles[i, k] = i * θ_k
        angles = torch.outer(positions.float(), thetas)  # (max_seq_len, d_k/2)

        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        # Build 2x2 rotation matrices for each (position, dimension_pair)
        # Shape: (max_seq_len, d_k/2, 2, 2)
        rotation_matrices = torch.stack([
            torch.stack([cos_angles, -sin_angles], dim=-1),
            torch.stack([sin_angles, cos_angles], dim=-1),
        ], dim=-2)

        self.register_buffer("rotation_matrices", rotation_matrices)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len d_k"]:
        """Apply rotary positional embedding to input tensor.
        If positions are not provided we assume they are from 0 to seq_len - 1.

        Args:
            x: (..., seq_len, d_k) input embeddings.
            positions: (..., seq_len) position indices for each token.

        Returns:
            (..., seq_len, d_k) position-encoded embeddings.
        """
        shape = x.shape
        seq_len, d_k = shape[-2], shape[-1]
        
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)
            positions = positions.expand(shape[:-2] + (seq_len,)) # Expand to (... seq_len)

        # Select rotation matrices for given positions: (batch, seq_len, d_k/2, 2, 2)
        rot_mat = self.rotation_matrices[positions]

        # Split embedding into pairs: (batch, seq_len, d_k/2, 2)
        x_pairs = rearrange(x, "... seq_len (theta_idx j) -> ... seq_len theta_idx j", j=2)

        # Apply rotation to each pair
        x_rotated = einsum(
            rot_mat, x_pairs,
            "... seq_len theta_idx i j, ... seq_len theta_idx j -> ... seq_len theta_idx i"
        )

        # Flatten back to original shape: (batch, seq_len, d_k)
        return rearrange(x_rotated, "... seq_len theta_idx i -> ... seq_len (theta_idx i)")
