"""
Common neural network utility functions.
"""

import torch
from torch import Tensor
from jaxtyping import Float, Int
from collections.abc import Iterable


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


def cross_entropy(inputs: Float[Tensor, "... seq_length vocab_size"],
                  targets: Int[Tensor, "... seq_length"]) -> Float[Tensor, "..."]:
    """
    Compute cross-entropy loss per sequence.

    Args:
        inputs: Logits tensor of shape (..., seq_length, vocab_size).
        targets: Target indices of shape (..., seq_length).

    Returns:
        Cross-entropy loss averaged over seq_length, shape (...).
    """
    log_sum_exp = torch.logsumexp(inputs, dim=-1)
    logit_target = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - logit_target).mean(dim=-1)


def loss_cross_ent(inputs: Float[Tensor, "... seq_length vocab_size"],
                   targets: Int[Tensor, "... seq_length"]) -> Float[Tensor, ""]:
    """
    Compute mean cross-entropy loss across all sequences.

    Args:
        inputs: Logits tensor of shape (..., seq_length, vocab_size).
        targets: Target indices of shape (..., seq_length).

    Returns:
        Scalar mean cross-entropy loss.
    """
    return cross_entropy(inputs, targets).mean()


def perplexity(inputs: Float[Tensor, "... seq_length vocab_size"],
               targets: Int[Tensor, "... seq_length"]) -> Float[Tensor, ""]:
    """
    Compute mean perplexity across sequences.

    Args:
        inputs: Logits tensor of shape (..., seq_length, vocab_size).
        targets: Target indices of shape (..., seq_length).

    Returns:
        Scalar mean perplexity (exp of cross-entropy).
    """
    return torch.exp(cross_entropy(inputs, targets)).mean()


def clip_gradients_(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clip combined gradients of parameters to have l2 norm at most max_l2_norm.

    Args:
        parameters: Collection of trainable parameters.
        max_l2_norm: Maximum allowed l2-norm (must be positive).

    Note:
        Modifies parameter.grad in-place.
    """
    params = [p for p in parameters if p.grad is not None]

    total_norm_sq = sum((p.grad ** 2).sum() for p in params)
    total_norm = total_norm_sq ** 0.5

    if total_norm > max_l2_norm:
        scale = max_l2_norm / total_norm
        for p in params:
            p.grad.mul_(scale)
