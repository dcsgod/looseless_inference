"""
draft.py — Draft generation loop using the small model.

Runs K autoregressive steps on the draft model, collecting:
  - The sampled token IDs  (what the draft "proposed")
  - The full probability distributions at each step  (needed for rejection sampling)
  - The updated KV cache  (so the draft cache can be rolled back if tokens are rejected)

Design note on caching:
  We pass in a KVCacheManager (or None) so the draft loop can start from wherever
  the last accepted position is, rather than re-processing the full prompt each round.
  This keeps draft generation O(K) per round regardless of context length.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from src.kv_cache import KVCacheManager
from src.baseline import _sample


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_draft(
    draft_model: PreTrainedModel,
    input_ids: torch.Tensor,
    K: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    cache_manager: Optional[KVCacheManager] = None,
) -> Tuple[List[int], torch.Tensor, KVCacheManager]:
    """
    Generate K draft tokens autoregressively from *draft_model*.

    Args:
        draft_model: Small causal LM in eval mode.
        input_ids:   Token ids for the current context. Shape [1, seq_len].
                     If cache_manager is not None, this should be ONLY the new
                     tokens since the last forward pass (i.e. shape [1, 1] for
                     continuation from cache).
        K:           Number of draft tokens to generate.
        temperature: Sampling temperature.
        top_p:       Nucleus sampling cutoff.
        top_k:       Top-k cutoff.
        cache_manager: Optional KVCacheManager carrying the draft model's KV
                       cache from the previous round. If None, starts fresh.

    Returns:
        draft_token_ids  : List[int] of K sampled token ids.
        draft_probs      : Tensor of shape [K, vocab_size] — full probability
                           distribution at each draft step (for rejection sampling).
        new_cache_manager: Updated KVCacheManager after the K draft steps.
    """
    device = _model_device(draft_model)
    current_ids = input_ids.to(device)

    past_kv = cache_manager.get() if cache_manager is not None else None

    draft_token_ids: List[int] = []
    draft_probs_list: List[torch.Tensor] = []

    for _ in range(K):
        outputs = draft_model(
            input_ids=current_ids,
            past_key_values=past_kv,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :]   # [1, vocab_size]
        past_kv = outputs.past_key_values

        # Full probability distribution (needed for rejection sampling math)
        probs = _logits_to_probs(logits, temperature, top_p, top_k)   # [1, vocab_size]
        draft_probs_list.append(probs.squeeze(0))  # [vocab_size]

        # Sample next token from this distribution
        token_id = _sample_from_probs(probs)
        draft_token_ids.append(token_id)

        # Feed sampled token back as next input
        current_ids = torch.tensor([[token_id]], device=device)

    draft_probs = torch.stack(draft_probs_list, dim=0)  # [K, vocab_size]
    new_cache_manager = KVCacheManager(past_kv)

    return draft_token_ids, draft_probs, new_cache_manager


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _logits_to_probs(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    """
    Convert raw logits → probability distribution with temp/top-p/top-k masking.

    This is the *distribution* version — we need the full vector p(x) over vocab
    so rejection sampling can evaluate p(x_draft) / q(x_draft) for any token.

    Returns:
        probs: shape [1, vocab_size], sums to 1, non-negative.
    """
    if temperature == 0.0:
        # Greedy: deterministic one-hot
        idx = logits.argmax(dim=-1, keepdim=True)  # [1, 1]
        probs = torch.zeros_like(logits)
        probs.scatter_(dim=-1, index=idx, value=1.0)
        return probs

    logits = logits / temperature

    # Top-k masking
    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        topk_vals, _ = torch.topk(logits, k, dim=-1)
        threshold = topk_vals[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    # Softmax → probs
    probs = F.softmax(logits, dim=-1)

    # Top-p (nucleus) masking on probs
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = (cumulative_probs - sorted_probs) > top_p
        sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
        # Scatter back and renormalize
        probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)

    return probs


def _sample_from_probs(probs: torch.Tensor) -> int:
    """Sample a single token id from a probability tensor of shape [1, vocab_size]."""
    return int(torch.multinomial(probs, num_samples=1).item())


def _model_device(model: PreTrainedModel) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
