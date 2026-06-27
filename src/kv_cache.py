"""
kv_cache.py — Manual KV cache management with truncation/rollback support.

The standard HuggingFace `past_key_values` is a tuple of (key, value) tensor
pairs, one pair per transformer layer:

    past_key_values: Tuple[Tuple[Tensor, Tensor], ...]
                     shape per tensor: [batch, num_heads, seq_len, head_dim]

On a partial rejection (draft tokens 0..m-1 accepted, token m rejected), we
need to roll back the target model's KV cache to position `m` so the next
round of target verification starts from the correct context — without
re-running a fresh forward pass on the entire prefix (which would defeat the
purpose of caching).

KVCacheManager wraps these tensors and provides:
  - truncate(n): slices all layer caches to [:, :, :n, :]
  - clone():     deep-copies the cache (needed to snapshot draft vs target)
  - get():       returns the raw past_key_values tuple

Notes on correctness:
  - Slicing along dim=2 (seq_len dimension) correctly removes the most-recent
    tokens from all keys and values simultaneously.
  - This is equivalent (in exact arithmetic) to running a fresh forward pass
    up to position n, as verified by tests/test_kv_cache.py.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


# Type alias for HF past_key_values
KVTuple = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


class KVCacheManager:
    """
    Wraps a HuggingFace `past_key_values` object and provides rollback support.

    Attributes:
        _cache: The raw past_key_values (immutable reference, replaced on updates).
    """

    def __init__(self, past_key_values: Optional[KVTuple] = None):
        self._cache: Optional[KVTuple] = past_key_values

    # ── Core methods ──────────────────────────────────────────────────────────

    def get(self) -> Optional[KVTuple]:
        """Return the raw past_key_values tuple (may be None if empty)."""
        return self._cache

    def set(self, past_key_values: Optional[KVTuple]) -> None:
        """Replace the stored cache with a new one."""
        self._cache = past_key_values

    def seq_len(self) -> int:
        """
        Return the number of tokens currently stored in the cache.
        Returns 0 if the cache is empty.
        """
        if self._cache is None:
            return 0
        # keys are [batch, heads, seq_len, head_dim]
        return self._cache[0][0].shape[2]

    def truncate(self, n: int) -> "KVCacheManager":
        """
        Return a NEW KVCacheManager with all layer caches sliced to the first
        *n* tokens (i.e. positions 0 … n-1).

        This is the rollback operation: after accepting m tokens out of K draft
        tokens, call truncate(prompt_len + m) to reset the target cache back
        to the state it would have been in after processing exactly those tokens.

        Args:
            n: Number of tokens to keep (must be >= 0 and <= seq_len()).

        Returns:
            A new KVCacheManager containing the truncated cache.

        Raises:
            ValueError: If the cache is empty or n is out of range.
        """
        if self._cache is None:
            raise ValueError("Cannot truncate an empty cache.")
        current_len = self.seq_len()
        if not (0 <= n <= current_len):
            raise ValueError(
                f"truncate({n}) out of range: cache has {current_len} tokens."
            )

        truncated = tuple(
            (
                layer_k[:, :, :n, :].contiguous(),
                layer_v[:, :, :n, :].contiguous(),
            )
            for layer_k, layer_v in self._cache
        )
        return KVCacheManager(truncated)

    def clone(self) -> "KVCacheManager":
        """
        Deep-copy the cache. Used to snapshot the cache state before draft
        generation so we can restore it independently.
        """
        if self._cache is None:
            return KVCacheManager(None)
        cloned = tuple(
            (layer_k.clone(), layer_v.clone())
            for layer_k, layer_v in self._cache
        )
        return KVCacheManager(cloned)

    def num_layers(self) -> int:
        """Return the number of transformer layers represented in the cache."""
        return len(self._cache) if self._cache is not None else 0

    def __repr__(self) -> str:
        if self._cache is None:
            return "KVCacheManager(empty)"
        return (
            f"KVCacheManager("
            f"layers={self.num_layers()}, "
            f"seq_len={self.seq_len()}, "
            f"dtype={self._cache[0][0].dtype}, "
            f"device={self._cache[0][0].device}"
            f")"
        )
