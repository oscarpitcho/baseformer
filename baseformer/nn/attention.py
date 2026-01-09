"""
Multi-head self-attention mechanism for transformer architectures.
"""

import torch
from torch import Tensor
from jaxtyping import Float, Bool, Int
from einops import einsum, rearrange

from baseformer.nn.utils import softmax
from baseformer.nn.linear import Linear


class MultiHeadSelfAttention(torch.nn.Module):
    """
    Multi-head self-attention layer.

    Applies parallel attention heads with RoPE then combines the results
    via a learned output projection W_o.
    """

    def __init__(self, d_model: int, n_heads: int):
        """
        Args:
            d_model: Model dimension. Must be divisible by n_heads.
            n_heads: Number of parallel attention heads.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        if d_model % n_heads != 0:
            raise ValueError(f"Cannot run multi-head attention if the model dimension "
                            + f"is not divisible by the number of heads. Found {d_model} and {n_heads}")
        
        self.head_dim = d_model // n_heads

        self.QKV_proj = Linear(d_model, 3 * d_model)
        self.O_proj = Linear(d_model, d_model)

        self.rope = None

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        positions: Int[Tensor, "... seq_len"] | None = None,
        mask: Bool[Tensor, "... seq_len seq_len"] | None = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        """
        Apply multi-head self-attention.

        Args:
            x: (..., seq_len, d_model) input features.
            positions: Optional (..., seq_len) position indices for RoPE.
            mask: Optional (..., seq_len, seq_len) boolean attention mask.
                  True values indicate positions to attend to.
                  If None, uses a causal (lower triangular) mask.

        Returns:
            (..., seq_len, d_model) attended features.
        """
        
        qkv = self.QKV_proj(x)

        q = qkv[..., :self.d_model]
        k = qkv[..., self.d_model: 2*self.d_model]
        v = qkv[..., 2*self.d_model: 3*self.d_model]

        # Split up and move head dimension to batch
        q_rearranged = rearrange(q,
            "... seq_len (n_heads head_dim) -> ... n_heads seq_len head_dim", head_dim=self.head_dim)
        k_rearranged = rearrange(k,
            "... seq_len (n_heads head_dim) -> ... n_heads seq_len head_dim", head_dim=self.head_dim)
        v_rearranged = rearrange(v,
            "... seq_len (n_heads head_dim) -> ... n_heads seq_len head_dim", head_dim=self.head_dim)

        # Apply RoPE after splitting into heads (operates on head_dim)
        if self.rope:
            rope_positions = positions.expand(q_rearranged.shape[:-2] + (positions.shape[-1],)) if positions is not None else None
            q_rearranged = self.rope(q_rearranged, rope_positions)  # pylint: disable=E1102
            k_rearranged = self.rope(k_rearranged, rope_positions)  # pylint: disable=E1102

        # Apply causal mask for autoregressive attention when no mask is provided
        if mask is None:
            seq_len = q_rearranged.shape[-2]
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q_rearranged.device))
            mask = mask.expand(q_rearranged.shape[:-2] + (seq_len, seq_len))

        # ( ... n_heads seq_len head_dim)
        h = scaled_dot_product_attention(q_rearranged, k_rearranged, v_rearranged, mask)
    
        # Collapse the heads
        h = rearrange(h, "... n_heads seq_len head_dim -> ... seq_len (n_heads head_dim)")
        h = self.O_proj(h)

        return h

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None
) -> Float[Tensor, "... queries d_v"]:
    """
    Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        Q: (..., queries, d_k) query vectors.
        K: (..., keys, d_k) key vectors.
        V: (..., keys, d_v) value vectors.
        mask: Optional (..., queries, keys) boolean mask.
              True values indicate positions to attend to.

    Returns:
        (..., queries, d_v) attended values.
    """
    d_k = Q.size()[-1]
    qk = einsum(Q, K," ... queries d_k,  ... keys d_k -> ... queries keys") / (d_k ** 0.5)
    
    if mask is not None:
        qk[~mask] = torch.finfo(qk.dtype).min
    att_scores = softmax(qk, dim=-1)

    return einsum(att_scores, V, "... q k, ... k d_v -> ... q d_v")







