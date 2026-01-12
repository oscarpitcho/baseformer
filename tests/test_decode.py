import torch
from baseformer.nn.transformer import _select_token


class TestSelectToken:
    """Tests for _select_token decoding helper."""

    def test_greedy_with_low_temperature(self):
        """Very low temperature should almost always select the max logit."""
        torch.manual_seed(42)
        # Shape: (seq_len, vocab_size) - function extracts [..., -1, :]
        logits = torch.tensor([[0.1, 0.2, 10.0, 0.3, 0.4]])  # index 2 has max
        
        # With very low temperature, softmax becomes very peaked
        token = _select_token(logits, temperature=0.01)
        assert token == 2, f"Expected token 2 (max logit), got {token}"

    def test_temperature_1_samples_from_distribution(self):
        """Temperature 1.0 should sample according to softmax probabilities."""
        torch.manual_seed(123)
        logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])  # uniform logits
        
        # With uniform logits and temp=1, all tokens should be possible
        tokens = [_select_token(logits.clone(), temperature=1.0) for _ in range(100)]
        unique_tokens = set(tokens)
        
        # Should see multiple different tokens sampled
        assert len(unique_tokens) > 1, "Expected sampling to produce varied tokens"

    def test_reproducible_with_seed(self):
        """Same seed should produce same token."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        
        torch.manual_seed(999)
        token1 = _select_token(logits.clone(), temperature=1.0)
        
        torch.manual_seed(999)
        token2 = _select_token(logits.clone(), temperature=1.0)
        
        assert token1 == token2, f"Expected reproducible sampling, got {token1} vs {token2}"

    def test_returns_int(self):
        """Result should be a Python int, not a tensor."""
        torch.manual_seed(0)
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        token = _select_token(logits, temperature=1.0)
        
        assert isinstance(token, int), f"Expected int, got {type(token)}"

    def test_valid_token_range(self):
        """Returned token should be within vocab size."""
        torch.manual_seed(42)
        vocab_size = 5
        logits = torch.tensor([[1.0] * vocab_size])
        
        for _ in range(50):
            token = _select_token(logits.clone(), temperature=1.0)
            assert 0 <= token < vocab_size, f"Token {token} out of range [0, {vocab_size})"


class TestTopPSampling:
    """Tests for top-p (nucleus) sampling."""

    def test_top_p_1_includes_all_tokens(self):
        """top_p=1.0 should consider all tokens (equivalent to no filtering)."""
        torch.manual_seed(42)
        logits = torch.tensor([1.0, 1.0, 1.0, 1.0])  # uniform
        
        tokens = [_select_token(logits.clone(), temperature=1.0, top_p=1.0) for _ in range(100)]
        unique_tokens = set(tokens)
        
        assert len(unique_tokens) > 1, "top_p=1.0 should sample from all tokens"

    def test_top_p_restricts_to_nucleus(self):
        """top_p should only sample from tokens within the probability nucleus."""
        torch.manual_seed(0)
        # Token 0 has ~73% prob after softmax, token 1 has ~27%
        # tokens 2,3 have negligible prob
        logits = torch.tensor([5.0, 4.0, 0.0, 0.0])
        
        # With top_p=0.8, only token 0 and 1 should be sampled
        tokens = [_select_token(logits.clone(), temperature=1.0, top_p=0.8) for _ in range(100)]
        
        assert all(t in [0, 1] for t in tokens), f"Expected only tokens 0,1 but got {set(tokens)}"

    def test_very_small_top_p_selects_top_token(self):
        """Very small top_p should effectively select only the highest prob token."""
        torch.manual_seed(42)
        logits = torch.tensor([10.0, 1.0, 1.0, 1.0])  # token 0 dominates
        
        # top_p=0.01 means we stop as soon as cumulative prob >= 0.01
        # Token 0 alone exceeds this threshold
        tokens = [_select_token(logits.clone(), temperature=1.0, top_p=0.01) for _ in range(50)]
        
        assert all(t == 0 for t in tokens), f"Expected only token 0, got {set(tokens)}"

    def test_top_p_with_temperature(self):
        """top_p should work correctly combined with temperature scaling."""
        torch.manual_seed(123)
        logits = torch.tensor([3.0, 2.0, 1.0, 0.0])
        
        # Higher temperature flattens distribution, so more tokens enter nucleus
        tokens_high_temp = [_select_token(logits.clone(), temperature=2.0, top_p=0.9) for _ in range(100)]
        
        # Lower temperature sharpens distribution, fewer tokens in nucleus
        tokens_low_temp = [_select_token(logits.clone(), temperature=0.5, top_p=0.9) for _ in range(100)]
        
        # High temp should have more variety
        assert len(set(tokens_high_temp)) >= len(set(tokens_low_temp))

    def test_top_p_returns_int(self):
        """top_p result should be a Python int."""
        torch.manual_seed(0)
        logits = torch.tensor([1.0, 2.0, 3.0])
        token = _select_token(logits, temperature=1.0, top_p=0.9)
        
        assert isinstance(token, int), f"Expected int, got {type(token)}"

