"""
verify.py — Single forward pass of the target model over prompt + draft tokens.

The key insight of speculative decoding is that we can get the target model's
opinion on ALL K draft positions in a single parallel forward pass, rather
than K sequential passes. This is what makes the algorithm fast.

Given a context of length L and K draft tokens:
  - We feed [context_tokens + draft_tokens] into the target model
  - The target model's output logits at positions L-1 .. L+K-1 are the
    distributions we compare against the draft model's distributions
  - Position L-1 is the logit BEFORE the first draft token (used for the
    bonus token if all K drafts are accepted)
  - Positions L .. L+K-1 are the target's logits conditional on each prefix

Shapes:
    input:   [1, L + K]   tokens
    output:  [K+1, V]     target probability distributions
             where V = vocab_size

             index 0  → target dist at position L-1 (bonus if all accepted)
             index i  → target dist at position L-1+i (used to accept/reject draft[i-1])

Wait — let me be precise about the indexing used in engine.py:

    draft_tokens = [t_0, t_1, ..., t_{K-1}]
    full_ids     = [prompt_ids..., t_0, t_1, ..., t_{K-1}]
                   positions 0..L-1 are prompt, L..L+K-1 are draft

    target forward on full_ids yields logits at each position.
    At position L-1: logit for "what comes after the prompt?" → compare with draft[0]
    At position L  : logit for "what comes after prompt+t_0?" → compare with draft[1]
    ...
    At position L+K-2: logit for "what comes after prompt+t_0..t_{K-2}?" → compare with draft[K-1]
    At position L+K-1: logit for "what comes after prompt+all_drafts?" → bonus token

    So we need logits at positions [L-1, L, ..., L+K-1] → K+1 logit vectors.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from src.kv_cache import KVCacheManager
from src.draft import _logits_to_probs


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def verify_draft(
    target_model: PreTrainedModel,
    context_ids: torch.Tensor,
    draft_token_ids: list[int],
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    target_cache: Optional[KVCacheManager] = None,
) -> Tuple[torch.Tensor, KVCacheManager]:
    """
    Run the target model over [context + draft_tokens] in one forward pass.

    If *target_cache* is provided and non-empty, only the draft tokens are fed
    (the prompt is already in the cache). Otherwise the full sequence is fed.

    Args:
        target_model:    Target causal LM in eval mode.
        context_ids:     Token ids for the prompt/context built so far.
                         Shape [1, context_len].
        draft_token_ids: List of K draft token ids.
        temperature:     Sampling temperature (applied to logits → probs).
        top_p:           Nucleus cutoff for probability distributions.
        top_k:           Top-k cutoff.
        target_cache:    KVCacheManager for the target model. If it already
                         covers the context, only draft tokens are fed.

    Returns:
        target_probs: Tensor of shape [K+1, vocab_size].
                      target_probs[i] is the target distribution AFTER seeing
                      the first i draft tokens (i=0 is after the prompt only).
        new_cache:    KVCacheManager updated with the target model's KV cache
                      for the full sequence (context + all K draft tokens).
    """
    device = _model_device(target_model)
    K = len(draft_token_ids)

    draft_ids = torch.tensor(
        [draft_token_ids], dtype=torch.long, device=device
    )  # [1, K]

    # Decide what tokens to feed based on cache state
    cache_len = target_cache.seq_len() if target_cache is not None else 0
    context_len = context_ids.shape[1]

    if cache_len == context_len and cache_len > 0:
        # Cache covers the full context → feed only the K draft tokens
        feed_ids = draft_ids
        past_kv = target_cache.get()
    elif cache_len == 0 or target_cache is None:
        # No cache → feed full sequence: [context | draft]
        full_ids = torch.cat([context_ids.to(device), draft_ids], dim=1)
        feed_ids = full_ids
        past_kv = None
    else:
        # Partial cache: feed remaining context tokens + draft tokens
        remaining = context_ids[:, cache_len:].to(device)
        feed_ids = torch.cat([remaining, draft_ids], dim=1)
        past_kv = target_cache.get()

    outputs = target_model(
        input_ids=feed_ids,
        past_key_values=past_kv,
        use_cache=True,
    )

    # Full sequence logits from target — shape [1, fed_len, vocab_size]
    all_logits = outputs.logits  # [1, fed_len, vocab]

    # We want logits at positions that correspond to:
    #   - after the last context token (index -K-1 in full sequence)
    #   - after each draft token      (indices -K .. -1)
    # These are the last K+1 logit positions in all_logits.
    relevant_logits = all_logits[0, -(K + 1) :, :]  # [K+1, vocab]

    # Convert to probability distributions using same masking as draft
    target_probs = torch.stack(
        [
            _logits_to_probs(relevant_logits[i].unsqueeze(0), temperature, top_p, top_k).squeeze(0)
            for i in range(K + 1)
        ],
        dim=0,
    )  # [K+1, vocab_size]

    new_cache = KVCacheManager(outputs.past_key_values)
    return target_probs, new_cache


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _model_device(model: PreTrainedModel) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
