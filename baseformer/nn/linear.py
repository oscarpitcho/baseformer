"""
Linear (fully connected) layer implementing an affine transformation.
"""

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from jaxtyping import Float


class Linear(Module):
    """
    Applies a linear transformation: y = xW^T + b.

    Weights are stored in (d_out, d_in) layout for efficient matrix multiplication.

    Attributes:
        weights: Learnable weight matrix of shape (d_out, d_in).
        bias: Optional learnable bias vector of shape (d_out,).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        use_bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            d_in: Size of input features.
            d_out: Size of output features.
            use_bias: If True, adds a learnable bias to the output.
            device: Device to place the weights on.
            dtype: Data type for the weights.
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.use_bias = use_bias

        # Wx does inner product over dim d_in.
        # Store it in (d_out, d_in) to ensure contiguous memory layout.
        self.weights = Parameter(torch.empty((d_out, d_in), dtype=dtype, device=device))

        if self.use_bias:
            self.bias = Parameter(torch.empty(d_out, dtype=dtype, device=device))

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """Apply the linear transformation y = xW^T + b.

        Args:
            x: (..., d_in) input features.

        Returns:
            (..., d_out) transformed features.
        """
        h = x @ self.weights.T
        if self.use_bias:
            h = h + self.bias
        return h
