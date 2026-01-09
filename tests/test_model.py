from einops import rearrange
import numpy
import torch
import torch.nn.functional as F

from .adapters import (
    run_multihead_self_attention_with_rope,
    run_rope,
    run_silu,
    run_multihead_self_attention,
    run_swiglu,
    run_rmsnorm,
    run_scaled_dot_product_attention,
    run_softmax,
    run_transformer_block,
    run_transformer_lm,
    run_linear,
    run_embedding,
)


def test_linear(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    w1_weight = ts_state_dict[0]["layers.0.ffn.w1.weight"]
    output = run_linear(
        d_in=d_model,
        d_out=d_ff,
        weights=w1_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(output)


def test_embedding(numpy_snapshot, ts_state_dict, in_indices, vocab_size, d_model):
    embedding_weight = ts_state_dict[0]["token_embeddings.weight"]
    output = run_embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=embedding_weight,
        token_ids=in_indices,
    )
    numpy_snapshot.assert_match(output)


def test_swiglu(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    w1_weight, w2_weight, w3_weight = [ts_state_dict[0][f"layers.0.ffn.{k}.weight"] for k in ["w1", "w2", "w3"]]

    actual_output = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=w1_weight,
        w2_weight=w2_weight,
        w3_weight=w3_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-5)


def test_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask):
    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )

def test_scaled_dot_product_attention_identity_QKV():
    """When Q = K = V = I, attention should return softmax-weighted identity."""
    batch_size = 3
    d_k = 4
    I = torch.eye(d_k)
    
    # Q = K = V = I (identity matrix), batched
    Q = I.unsqueeze(0).expand(batch_size, d_k, d_k)  # (batch, d_k, d_k)
    K = I.unsqueeze(0).expand(batch_size, d_k, d_k)
    V = I.unsqueeze(0).expand(batch_size, d_k, d_k)
    
    output = run_scaled_dot_product_attention(Q=Q, K=K, V=V, mask=None)
    
    # QK^T / sqrt(d_k) = I / sqrt(d_k), then softmax per row
    scores = I / (d_k ** 0.5)
    expected_attn = run_softmax(scores, dim=-1)
    
    # output = attn_weights @ V = attn_weights @ I = attn_weights
    expected_output = expected_attn.unsqueeze(0).expand(batch_size, d_k, d_k)
    
    torch.testing.assert_close(output, expected_output, atol=1e-6, rtol=1e-5)


def test_scaled_dot_product_attention_identity_mask():
    """With Q = K = V = I and identity mask, output should be I."""
    batch_size = 3
    d_k = 4
    I = torch.eye(d_k)
    
    Q = I.unsqueeze(0).expand(batch_size, d_k, d_k)  # (batch, d_k, d_k)
    K = I.unsqueeze(0).expand(batch_size, d_k, d_k)
    V = I.unsqueeze(0).expand(batch_size, d_k, d_k)
    
    # Identity mask: only attend to diagonal (each query attends only to its matching key)
    mask = I.bool().unsqueeze(0).expand(batch_size, d_k, d_k)  # (batch, d_k, d_k)
    
    output = run_scaled_dot_product_attention(Q=Q, K=K, V=V, mask=mask)
    
    # With identity mask:
    # - Off-diagonal scores become -inf, then 0 after softmax
    # - Each row has exactly one valid position, so softmax gives 1 there
    # - attention_weights = I
    # - output = I @ V = I @ I = I
    expected_output = I.unsqueeze(0).expand(batch_size, d_k, d_k)
    
    torch.testing.assert_close(output, expected_output, atol=1e-6, rtol=1e-5)


def test_scaled_dot_product_attention_rank1_KV():
    """With rank-1 K and V matrices, output is the V vector replicated for each query."""
    batch_size = 3
    n_queries = 2
    n_keys = 4
    d_k = 8
    
    # Arbitrary query vectors (different from each other), batched
    Q = torch.randn(batch_size, n_queries, d_k)
    
    # K is rank-1: all key vectors are identical
    k_vec = torch.randn(d_k)
    K = k_vec.view(1, 1, d_k).expand(batch_size, n_keys, d_k)  # (batch, n_keys, d_k)
    
    # V is rank-1: all value vectors are identical
    v_vec = torch.randn(d_k)
    V = v_vec.view(1, 1, d_k).expand(batch_size, n_keys, d_k)  # (batch, n_keys, d_k)
    
    output = run_scaled_dot_product_attention(Q=Q, K=K, V=V, mask=None)
    
    # Each query has identical dot products with all keys (since all keys are the same)
    # → softmax gives uniform weights (1/n_keys each)
    # → weighted sum of identical V vectors = v_vec
    # → output is v_vec replicated for each query
    expected_output = v_vec.view(1, 1, d_k).expand(batch_size, n_queries, d_k)
    
    torch.testing.assert_close(output, expected_output, atol=1e-6, rtol=1e-5)


def test_4d_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask):
    # Shape: (batch_size, num_heads, seq_len, d_k)
    q, k, v = (rearrange(x, "(batch head) seq d -> batch head seq d", head=2) for x in (q, k, v))
    mask = rearrange(mask, "(batch head) query key -> batch head query key", head=2)

    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_multihead_self_attention(numpy_snapshot, in_embeddings, d_model, n_heads, ts_state_dict):
    d, _ = ts_state_dict
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
    ]
    actual_output = run_multihead_self_attention(
        d_model=d_model,
        num_heads=n_heads,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_multihead_self_attention_with_rope(
    numpy_snapshot, in_embeddings, d_model, n_heads, ts_state_dict, n_keys, theta, pos_ids
):
    d, _ = ts_state_dict
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
    ]
    pos_ids = rearrange(pos_ids, "seq -> 1 seq")
    actual_output = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=n_heads,
        max_seq_len=n_keys,
        theta=theta,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
        token_positions=pos_ids,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_transformer_lm(
    numpy_snapshot, vocab_size, n_keys, d_model, n_layers, n_heads, d_ff, theta, ts_state_dict, in_indices
):
    state_dict, _ = ts_state_dict

    actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=state_dict,
        in_indices=in_indices,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-4, rtol=1e-2)


def test_transformer_lm_truncated_input(
    numpy_snapshot, vocab_size, n_keys, d_model, n_layers, n_heads, d_ff, theta, ts_state_dict, in_indices
):
    in_indices_truncated = in_indices[..., : in_indices.shape[-1] // 2]
    truncated_actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=ts_state_dict[0],
        in_indices=in_indices_truncated,
    )

    numpy_snapshot.assert_match(
        truncated_actual_output,
        atol=1e-4,
    )


def test_transformer_block(numpy_snapshot, ts_state_dict, in_embeddings, d_model, n_heads, d_ff, n_keys, theta):
    block_weights = {k.replace("layers.0.", ""): v for k, v in ts_state_dict[0].items() if "layers.0." in k}

    actual_output = run_transformer_block(
        d_model=d_model,
        num_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=n_keys,
        theta=theta,
        weights=block_weights,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_rmsnorm(numpy_snapshot, ts_state_dict, in_embeddings):
    state_dict, _ = ts_state_dict
    reference_weights = state_dict["layers.1.ln1.weight"]
    d_model = reference_weights.shape[0]

    actual_output = run_rmsnorm(d_model=d_model, eps=1e-5, weights=reference_weights, in_features=in_embeddings)

    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_rope(numpy_snapshot, in_embeddings, d_model, theta, n_queries, pos_ids):
    output = run_rope(
        d_model, theta=theta, max_seq_len=n_queries, in_query_or_key=in_embeddings, token_positions=pos_ids
    )
    numpy_snapshot.assert_match(output, atol=1e-6)


def test_silu_matches_pytorch():
    x = torch.tensor(
        [
            [0.2352, 0.9259, 0.5189, 0.4725, 0.9730],
            [0.7581, 0.9692, 0.2129, 0.9345, 0.0149],
        ]
    )
    expected_output = F.silu(x)
    actual_output = run_silu(x)
    numpy.testing.assert_allclose(actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6)
