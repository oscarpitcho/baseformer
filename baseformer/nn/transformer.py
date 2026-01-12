"""
Transformer language model with pre-norm architecture.

This module implements a decoder-only transformer language model using:
- RMSNorm for layer normalization (pre-norm style)
- Rotary Positional Embeddings (RoPE) for position encoding
- SwiGLU activation in the feed-forward network
- Multi-head self-attention with causal masking support
"""

import torch
from torch import Tensor
from torch.nn import Module
from jaxtyping import Float, Bool, Int
from typing import Iterator, Optional

from baseformer.nn.attention import MultiHeadSelfAttention
from baseformer.nn.activations import SwiGLU
from baseformer.nn.norm import RMSNorm
from baseformer.nn.position import RotaryPositionalEmbedding
from baseformer.nn.embedding import Embedding
from baseformer.nn.linear import Linear
from baseformer.nn.sequential import Sequential
from baseformer.nn.utils import softmax


class TransformerLM(Module):
    """
    Decoder-only transformer language model.

    Architecture:
        1. Token embedding lookup
        2. Stack of TransformerBlocks (pre-norm attention + FFN)
        3. Final RMSNorm
        4. Linear projection to vocabulary logits

    Attributes:
        vocab_size: Size of the token vocabulary.
        d_model: Model/embedding dimension.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads per block.
        d_ff: Hidden dimension of the feed-forward network.
        rope: Optional rotary positional embedding module.
        word_embedding: Token embedding layer.
        network: Sequential stack of transformer blocks.
        ln_final: Final layer normalization.
        lm_decoder: Output projection to vocabulary logits.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope: Optional[RotaryPositionalEmbedding] = None,
        device=None,
        dtype=None
    ):
        """
        Args:
            vocab_size: Size of the token vocabulary.
            d_model: Model/embedding dimension.
            num_layers: Number of transformer blocks.
            num_heads: Number of attention heads per block.
            d_ff: Hidden dimension of the feed-forward network.
            rope: Optional RoPE module for rotary positional embeddings.
            device: Device to place weights on.
            dtype: Data type for weights.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = rope

        self.word_embedding = Embedding(vocab_size, d_model)
        self.network = Sequential(
            *[TransformerBlock(d_model, num_heads, d_ff, rope, device=device, dtype=dtype)
              for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_decoder = Linear(d_model, vocab_size)

    def forward(
        self,
        in_indices: Int[Tensor, "... seq_len"],
        mask: Optional[Bool[Tensor, "... seq_len seq_len"]] = None,
        positions: Optional[Int[Tensor, "... seq_len"]] = None
    ) -> Float[Tensor, "... seq_len vocab_size"]:
        """Compute next-token logits for input token sequences.

        Args:
            in_indices: (..., seq_len) input token indices.
            mask: Optional (..., seq_len, seq_len) attention mask.
                  True values are attended to, False values are masked out.
            positions: Optional (..., seq_len) position indices for RoPE.
                       If None, uses sequential positions 0, 1, 2, ...

        Returns:
            (..., seq_len, vocab_size) logits over vocabulary for each position.
        """
        embeddings = self.word_embedding(in_indices)
        hidden = self.network(embeddings, mask, positions)
        hidden = self.ln_final(hidden)
        logits = self.lm_decoder(hidden)
        return logits

    # TODO: Optimize with prefill + decode phases and KV caching.
    #       Store KV cache as preallocated tensor modified in place.
    # TODO: Implement batched generation.

    def _decode_next_token(
        self,
        token_ids: Int[Tensor, "... seq_len"],
        temperature: float = 1.0,
        top_p: Optional[float] = None
    ) -> Int[Tensor, "... 1"]:
        """Sample the next token given a sequence.

        Args:
            token_ids: (batch, seq_len) tensor of token IDs.
            temperature: Sampling temperature. Higher = more random.
            top_p: Optional nucleus sampling threshold.

        Returns:
            Sampled next token ID.
        """
        logits = self.forward(token_ids)
        logits = logits[:, -1, :]  # Select last position
        return _select_token(logits, temperature, top_p)


    # TODO: Implement lookahead generation.
    #       Yielding at every token forces GPU / CPU synchronization.
    def decode(
        self,
        token_ids: Int[Tensor, "... seq_len"],
        temperature: float = 0.8,
        top_p: Optional[float] = 0.9
    ) -> Iterator[Int[Tensor, "... 1"]]:
        """Generate tokens autoregressively.

        Args:
            token_ids: Initial (..., seq_len) tensor of token IDs (the prompt).
            temperature: Sampling temperature. Higher = more random.
            top_p: Optional nucleus sampling threshold.

        Yields:
            Token IDs one at a time, indefinitely.
        """
        while True:
            next_token = self._decode_next_token(token_ids, temperature, top_p)  # shape (batch, 1)
            token_ids = torch.cat([token_ids, next_token], dim=-1)
            yield next_token


class TransformerBlock(Module):
    """
    Single transformer block with pre-norm architecture.

    Structure:
        x -> RMSNorm -> MultiHeadAttention -> + -> RMSNorm -> SwiGLU FFN -> + -> output
        |___________________________________|   |__________________________|
                    residual                            residual

    Attributes:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        rope: Optional RoPE module for positional embeddings.
        norm1: RMSNorm before attention.
        norm2: RMSNorm before feed-forward.
        mha: Multi-head self-attention layer.
        ffn: SwiGLU feed-forward network.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: Optional[RotaryPositionalEmbedding] = None,
        device=None,
        dtype=None
    ):
        """
        Args:
            d_model: Model dimension.
            num_heads: Number of attention heads.
            d_ff: Feed-forward hidden dimension.
            rope: Optional RoPE module for rotary positional embeddings.
            device: Device to place weights on.
            dtype: Data type for weights.
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = rope

        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.mha = MultiHeadSelfAttention(d_model, num_heads, rope, device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        mask: Optional[Bool[Tensor, "... seq_len seq_len"]] = None,
        positions: Optional[Int[Tensor, "... seq_len"]] = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        """Apply the transformer block.

        Args:
            x: (..., seq_len, d_model) input features.
            mask: Optional (..., seq_len, seq_len) attention mask.
            positions: Optional (..., seq_len) position indices for RoPE.

        Returns:
            (..., seq_len, d_model) transformed features.
        """
        # Pre-norm attention with residual
        h = self.norm1(x)
        h = self.mha(h, positions, mask)
        x = x + h

        # Pre-norm FFN with residual
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h

        return x


def _select_token(
    logits: Float[Tensor, "... vocab_size"],
    temperature: float,
    top_p: Optional[float] = None
) -> Int[Tensor, "..."]:
    """Sample a token from logits with temperature and optional nucleus sampling.

    Args:
        logits: (..., vocab_size) unnormalized log-probabilities.
        temperature: Sampling temperature. Higher = more random, lower = more deterministic.
                     If 0, uses greedy decoding (argmax).
        top_p: Optional nucleus sampling threshold. If set, only tokens whose cumulative
               probability mass is within top_p are considered.

    Returns:
        Sampled token ID.
    """
    # Greedy decoding when temperature is 0
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    scaled_logits = logits / temperature
    probs = softmax(scaled_logits, dim=-1)

    if top_p is not None:
        # Sort probabilities descending, tracking original indices
        sorted_probs, orig_indices = probs.sort(dim=-1, descending=True)

        # Compute cumulative probability mass
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        # Exclude tokens where cumulative prob before them exceeds top_p
        cumsum_before = cumsum - sorted_probs
        exclude_mask = cumsum_before >= top_p

        # Zero out excluded tokens and renormalize
        sorted_probs[exclude_mask] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        # Sample and map back to original vocabulary indices
        sampled_position = torch.multinomial(sorted_probs, num_samples=1)
        return orig_indices.gather(-1, sampled_position)

    return torch.multinomial(probs, num_samples=1)
