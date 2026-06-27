"""
test_kv_cache.py — Verify KVCacheManager truncation produces correct cache states.

The correctness claim: truncating a length-N cache to length-M must produce
results identical (within float tolerance) to running a fresh forward pass
up to position M.

If this doesn't hold, the engine will feed the wrong context to the target
model after partial rejections, breaking both correctness and losslessness.

Tests use the draft model (Qwen2.5-0.5B) since it loads faster than the
7B target, but the logic is identical for any causal LM with use_cache=True.

NOTE: These tests require a GPU with the draft model loaded.
      Skip with: pytest tests/test_kv_cache.py -v --ignore-glob="*kv*"
      if you don't yet have GPU access.
"""

from __future__ import annotations

import os
import pytest
import torch
import yaml
from pathlib import Path

from src.kv_cache import KVCacheManager

# ── Skip marker for environments without GPU / models ──────────────────────
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU not available"
)

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def config():
    cfg_path = Path("configs/default.yaml")
    if not cfg_path.exists():
        pytest.skip("configs/default.yaml not found — run from project root")
    return yaml.safe_load(cfg_path.read_text())


@pytest.fixture(scope="module")
def draft_model_and_tokenizer(config):
    """Load draft model once for the whole module."""
    pytest.importorskip("transformers")
    from src.models import load_models
    # Override to load only what we need quickly
    pair = load_models(config)
    return pair.draft, pair.tokenizer


# ── Tests ─────────────────────────────────────────────────────────────────

class TestKVCacheManager:
    """Unit tests for KVCacheManager with synthetic tensors (no model)."""

    def _make_fake_cache(self, n_layers: int = 4, seq_len: int = 10,
                         n_heads: int = 8, head_dim: int = 64) -> KVCacheManager:
        cache = tuple(
            (
                torch.randn(1, n_heads, seq_len, head_dim),
                torch.randn(1, n_heads, seq_len, head_dim),
            )
            for _ in range(n_layers)
        )
        return KVCacheManager(cache)

    def test_seq_len(self):
        mgr = self._make_fake_cache(seq_len=10)
        assert mgr.seq_len() == 10

    def test_num_layers(self):
        mgr = self._make_fake_cache(n_layers=6)
        assert mgr.num_layers() == 6

    def test_empty_cache(self):
        mgr = KVCacheManager()
        assert mgr.seq_len() == 0
        assert mgr.num_layers() == 0

    def test_truncate_preserves_prefix(self):
        """Truncated cache's slice must exactly equal original cache[:, :, :n, :]."""
        mgr = self._make_fake_cache(n_layers=4, seq_len=20)
        truncated = mgr.truncate(8)
        assert truncated.seq_len() == 8
        raw = mgr.get()
        trunc_raw = truncated.get()
        for layer_idx in range(4):
            k_orig = raw[layer_idx][0][:, :, :8, :]
            k_trunc = trunc_raw[layer_idx][0]
            assert torch.allclose(k_orig, k_trunc), f"Key mismatch at layer {layer_idx}"
            v_orig = raw[layer_idx][1][:, :, :8, :]
            v_trunc = trunc_raw[layer_idx][1]
            assert torch.allclose(v_orig, v_trunc), f"Value mismatch at layer {layer_idx}"

    def test_truncate_to_full_length_is_identity(self):
        mgr = self._make_fake_cache(seq_len=15)
        same = mgr.truncate(15)
        assert same.seq_len() == 15

    def test_truncate_to_zero(self):
        mgr = self._make_fake_cache(seq_len=10)
        empty = mgr.truncate(0)
        assert empty.seq_len() == 0

    def test_truncate_out_of_range_raises(self):
        mgr = self._make_fake_cache(seq_len=10)
        with pytest.raises(ValueError):
            mgr.truncate(11)
        with pytest.raises(ValueError):
            mgr.truncate(-1)

    def test_truncate_empty_raises(self):
        mgr = KVCacheManager()
        with pytest.raises(ValueError):
            mgr.truncate(0)

    def test_clone_is_independent(self):
        """Modifying a cloned cache must not affect the original."""
        mgr = self._make_fake_cache(seq_len=10)
        cloned = mgr.clone()
        # Mutate clone's first layer key
        cloned.get()[0][0].fill_(999.0)
        # Original should be unchanged
        assert not torch.allclose(
            mgr.get()[0][0],
            cloned.get()[0][0],
        ), "Clone mutation affected original"


@requires_gpu
class TestKVCacheTruncationCorrectness:
    """
    Golden-reference test: truncated cache must produce same logits as a
    fresh forward pass to the same position.

    This is the important correctness test — synthetic truncation tests above
    only verify slicing behavior, not whether the transformer actually produces
    the same output from a truncated vs freshly-computed cache.
    """

    def test_truncated_cache_matches_fresh_forward(self, draft_model_and_tokenizer):
        """
        1. Run model on tokens [0..N-1], capture cache and logit at N-1.
        2. Run model on tokens [0..M-1] (M < N) fresh, capture cache.
        3. Truncate the N-length cache to M tokens.
        4. Run model on token [M] using both caches.
        5. Assert logits are allclose.
        """
        model, tokenizer = draft_model_and_tokenizer
        device = next(model.parameters()).device

        text = "The quick brown fox jumps over the lazy dog. "
        tokens = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        N = tokens.shape[1]
        M = N // 2  # truncation point

        assert M >= 1 and M < N, "Need at least 2 tokens"

        next_token = tokens[:, M : M + 1]  # the token we'll feed after truncation

        with torch.no_grad():
            # Step 1: Full forward pass, cache length N
            out_full = model(input_ids=tokens, use_cache=True)
            cache_N = KVCacheManager(out_full.past_key_values)
            assert cache_N.seq_len() == N

            # Step 2: Fresh forward to M
            out_fresh_M = model(input_ids=tokens[:, :M], use_cache=True)
            cache_M_fresh = KVCacheManager(out_fresh_M.past_key_values)
            assert cache_M_fresh.seq_len() == M

            # Step 3: Truncate full cache to M
            cache_M_truncated = cache_N.truncate(M)
            assert cache_M_truncated.seq_len() == M

            # Step 4a: Continue from fresh M cache
            out_from_fresh = model(
                input_ids=next_token,
                past_key_values=cache_M_fresh.get(),
                use_cache=True,
            )

            # Step 4b: Continue from truncated M cache
            out_from_truncated = model(
                input_ids=next_token,
                past_key_values=cache_M_truncated.get(),
                use_cache=True,
            )

        # Step 5: Compare logits
        logits_fresh = out_from_fresh.logits[0, -1, :]
        logits_trunc = out_from_truncated.logits[0, -1, :]

        assert torch.allclose(logits_fresh, logits_trunc, atol=1e-3), (
            "Truncated cache produces different logits from fresh forward pass!\n"
            f"Max diff: {(logits_fresh - logits_trunc).abs().max().item():.6f}"
        )

    def test_set_and_get_roundtrip(self, draft_model_and_tokenizer):
        """set() then get() must return the same object."""
        model, tokenizer = draft_model_and_tokenizer
        device = next(model.parameters()).device
        tokens = tokenizer("hello world", return_tensors="pt")["input_ids"].to(device)

        with torch.no_grad():
            out = model(input_ids=tokens, use_cache=True)

        mgr = KVCacheManager()
        mgr.set(out.past_key_values)
        assert mgr.get() is out.past_key_values
