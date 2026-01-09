"""
Common neural network utility functions.
"""

import torch
from torch import Tensor
from jaxtyping import Float


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    """
    Compute numerically stable softmax along a dimension.

    Args:
        x: Input tensor of any shape.
        dim: Dimension along which to compute softmax.

    Returns:
        Tensor of same shape with softmax applied along dim.
    """
    max_val = x.max(dim=dim, keepdim=True).values
    exp_x = (x - max_val).exp()
    return exp_x / exp_x.sum(dim=dim, keepdim=True)