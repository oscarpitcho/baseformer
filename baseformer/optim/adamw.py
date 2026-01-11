"""
AdamW optimizer with decoupled weight decay regularization.
"""

from collections.abc import Iterable, Callable
from typing import Dict, Tuple

import torch


class AdamW(torch.optim.Optimizer):
    """
    Implements the AdamW algorithm (Adam with decoupled weight decay).
.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Dict,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        """
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            lr: Learning rate.
            betas: Coefficients for computing running averages of gradient (beta1)
                   and its square (beta2).
            eps: Term added to denominator for numerical stability.
            weight_decay: Weight decay (L2 penalty) coefficient.
        """
        if lr < 0 or eps < 0 or weight_decay < 0:
            raise ValueError(f"Invalid hyperparameters: lr={lr}, eps={eps}, weight_decay={weight_decay}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta values: {betas}")

        defaults = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)
        self.state["step"] = 0

    @torch.no_grad()
    def step(self, lr_schedule: Callable[[int, float], float] = None) -> None:
        """
        Perform a single optimization step.

        Updates all parameters using AdamW update rule:
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g^2
            alpha_t = alpha * sqrt(1 - beta2^t) / (1 - beta1^t)
            theta = theta - alpha_t * m / (sqrt(v) + eps)
            theta = theta - alpha * lambda * theta

        Args:
            lr_schedule: Optional callable that takes step number and returns learning rate.
        """
        self.state["step"] += 1
        t = self.state["step"]

        for group in self.param_groups:
            beta1, beta2 = group["beta1"], group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # Use lr_schedule if provided, otherwise use group lr
            alpha = lr_schedule(t, group["lr"]) if lr_schedule is not None else group["lr"]

            # Compute adjusted learning rate with bias correction
            alpha_t = alpha * (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Initialize momentum buffers on first step
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m, v = state["m"], state["v"]
                g = p.grad

                # Update biased first and second moment estimates
                m_new = beta1 * m + (1 - beta1) * g
                v_new = beta2 * v + (1 - beta2) * (g ** 2)
                state["m"], state["v"] = m_new, v_new

                # Parameter update
                p -= alpha_t * m_new / (torch.sqrt(v_new) + eps)

                # Apply decoupled weight decay
                p -= alpha * weight_decay * p
