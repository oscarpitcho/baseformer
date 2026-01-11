"""
Transformer block with pre-norm architecture using RMSNorm.
"""

from torch import Tensor
from torch.nn import Module
from jaxtyping import Float, Bool, Int

from typing import Optional

from baseformer.nn.attention import MultiHeadSelfAttention
from baseformer.nn.activations import SwiGLU
from baseformer.nn.norm import RMSNorm
from baseformer.nn.position import RotaryPositionalEmbedding
from baseformer.nn.embedding import Embedding
from baseformer.nn.linear import Linear
from baseformer.nn.sequential import Sequential

class TransformerLM(Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding | None = None,
        device = None,
        dtype = None
    ): 
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = rope


        self.word_embedding = Embedding(vocab_size, d_model)

        self.network = Sequential(
            *[TransformerBlock(d_model, num_heads, d_ff, rope, device=device, dtype=dtype) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)

        self.lm_decoder = Linear(d_model, vocab_size)

    
    def forward(self,
                in_indices: Int[Tensor, " batch_size sequence_length"],
                mask: Bool[Tensor, " seq_len seq_len"] = None,
                positions: Int[Tensor, "... seq_len"] = None
                ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:


        # (batch_size d_model)
        start_emb = self.word_embedding(in_indices)

        dec_emb = self.network(start_emb, mask, positions)

        norm_emb = self.ln_final(dec_emb)

        logits = self.lm_decoder(norm_emb)

        return logits



class TransformerBlock(Module):
    """
    Single transformer block with pre-norm architecture.

    Applies RMSNorm before attention (pre-norm), followed by a feed-forward
    network, and a residual connection from input to output.

    Attributes:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        rope: Optional RoPE module for positional embeddings.
        norm: RMSNorm layer applied before attention.
        l1: First linear layer of the feed-forward network.
        l2: Second linear layer of the feed-forward network.
        mha: Multi-head self-attention layer.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: Optional[RotaryPositionalEmbedding] = None,
        device = None,
        dtype = None
    ):
        """
        Args:
            d_model: Model dimension.
            num_heads: Number of attention heads.
            d_ff: Feed-forward hidden dimension.
            rope: Optional RoPE module for rotary positional embeddings.
            device: Device to place the weights on.
            dtype: Data type for the weights.
        """
        super().__init__()
        self.d_model = d_model 
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = rope

        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.mha = MultiHeadSelfAttention(d_model, num_heads, rope, device, dtype=dtype)


    def forward(self,
                x: Float[Tensor, "... seq_len d_model"],
                mask: Bool[Tensor, "... seq_len seq_len"] = None,
                positions: Int[Tensor, "... seq_len"] = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        """Apply the transformer block.

        Args:
            x: (..., seq_len, d_model) input features.
            mask: Optional (..., seq_len, seq_len) attention mask.
            positions: Optional (..., seq_len) position indices for RoPE.

        Returns:
            (..., seq_len, d_model) transformed features.
        """
        # Pre norm attention block
        h = self.norm1(x)
        h = self.mha(h, positions, mask)
        h = h + x

        # Pre norm feed forward block
        x = h
        h = self.norm2(h)
        h = self.ffn(h)
        h = h + x

        return h
