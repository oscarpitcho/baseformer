"""
Root Mean Square Layer Normalization (RMSNorm).

RMSNorm is a simplification of LayerNorm that removes the mean centering,
using only the RMS for normalization.

Reference: https://arxiv.org/abs/1910.07467
"""

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from jaxtyping import Float


class RMSNorm(Module):
    """
    Root Mean Square Layer Normalization.

    Normalizes inputs by their RMS (root mean square) and applies a learnable
    scale parameter. Unlike LayerNorm, does not subtract the mean.

    Formula: output = gamma * (x / RMS(x))
    where RMS(x) = sqrt(mean(x^2) + eps)

    Attributes:
        gamma: Learnable scale parameter of shape (d_model,).
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            d_model: Dimension of the input features (last dimension).
            eps: Small constant for numerical stability in the denominator.
            device: Device to place the parameters on.
            dtype: Data type for the parameters.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.gamma = Parameter(torch.ones(d_model, dtype=torch.float32, device=device))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """Normalize input by its root mean square and scale by gamma.

        Args:
            x: (..., d_model) input features.

        Returns:
            (..., d_model) normalized features.
        """
        in_type = x.dtype

        # Promote to float32 for stability.
        # Vector reduction accumulate error in fp16.
        # https://arxiv.org/abs/1710.03740
        x = x.to(torch.float32)
        mean_sq = torch.mean(x ** 2, dim=-1, keepdim=True)
        rms = torch.sqrt(mean_sq + self.eps)

        return (self.gamma * (x / rms)).to(in_type)

