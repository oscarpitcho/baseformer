
import math

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int
) -> float:
    """
    Cosine annealing learning rate schedule with linear warmup.

    The schedule has three phases:
    - Warm-up (t < T_w): α_t = (t / T_w) * α_max
    - Cosine annealing (T_w <= t <= T_c): α_t = α_min + 0.5 * (1 + cos((t - T_w) / (T_c - T_w) * π)) * (α_max - α_min)
    - Post-annealing (t > T_c): α_t = α_min

    Args:
        it: Current iteration t.
        max_learning_rate: Maximum (peak) learning rate α_max.
        min_learning_rate: Minimum (final) learning rate α_min.
        warmup_iters: Number of warm-up iterations T_w.
        cosine_cycle_iters: Iteration number at which cosine annealing ends T_c.

    Returns:
        Learning rate at iteration t.
    """
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate