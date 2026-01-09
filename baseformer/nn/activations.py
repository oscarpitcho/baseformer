"""
Activation functions and gated linear units for transformer architectures.
"""

import torch
from torch import Tensor
from torch.nn import Module
from jaxtyping import Float

from baseformer.nn.linear import Linear


class SiLU(Module):
    """
    Sigmoid Linear Unit (SiLU / Swish) activation.

    Computes x * sigmoid(x), a smooth approximation to ReLU.
    """

    def forward(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        """Apply SiLU activation element-wise.

        Args:
            x: (..., d) input features.

        Returns:
            (..., d) activated features.
        """
        return x * torch.sigmoid(x)


class GLU(Module):
    """
    Gated Linear Unit.

    Splits the transformation into a gate and a residual path,
    computing sigmoid(W1 @ x) * (W2 @ x).
    """

    def __init__(
        self,
        d_emb: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            d_emb: Embedding dimension.
            device: Device to place the weights on.
            dtype: Data type for the weights.
        """
        super().__init__()
        self.d_emb = d_emb

        self.w1 = Linear(d_emb, d_emb, use_bias=False, device=device, dtype=dtype)
        self.w2 = Linear(d_emb, d_emb, use_bias=False, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d_emb"]) -> Float[Tensor, "... d_emb"]:
        """Apply gated linear unit.

        Args:
            x: (..., d_emb) input features.

        Returns:
            (..., d_emb) gated features.
        """
        gate = torch.sigmoid(self.w1(x))
        return gate * self.w2(x)


class SwiGLU(Module):
    """
    SwiGLU activation (Swish-Gated Linear Unit).

    Variant of GLU using SiLU instead of sigmoid for the gate:
    W2 @ (SiLU(W1 @ x) * (W3 @ x))

    Reference: https://arxiv.org/abs/2002.05202
    """

    def __init__(
        self,
        d_emb: int,
        d_ff: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            d_emb: Embedding dimension.
            d_ff: Feed-forward hidden dimension.
            device: Device to place the weights on.
            dtype: Data type for the weights.
        """
        super().__init__()
        self.d_emb = d_emb
        self.d_ff = d_ff

        self.w1 = Linear(d_emb, d_ff, use_bias=False, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_emb, use_bias=False, device=device, dtype=dtype)
        self.w3 = Linear(d_emb, d_ff, use_bias=False, device=device, dtype=dtype)
        self.silu = SiLU()

    def forward(self, x: Float[Tensor, "... d_emb"]) -> Float[Tensor, "... d_emb"]:
        """Apply SwiGLU: W2 @ (SiLU(W1 @ x) * (W3 @ x)).

        Args:
            x: (..., d_emb) input features.

        Returns:
            (..., d_emb) transformed features.
        """
        gate = self.silu(self.w1(x))
        res = self.w3(x)

        h = self.w2(gate * res)
        return h
