"""
rejection_sampling.py — Core speculative decoding accept/reject math.

─── The Algorithm (Leviathan et al., 2023) ────────────────────────────────────

For each draft position i (0-indexed), with:
    x_i    = draft token id at position i
    p_i(x) = target model's probability distribution at position i
    q_i(x) = draft model's probability distribution at position i

Step 1 — Accept or Reject x_i:
    Draw u ~ Uniform(0, 1)
    If u ≤ p_i(x_i) / q_i(x_i):  ACCEPT x_i
    Else:                          REJECT x_i, stop scanning

Step 2 — On rejection at position i, sample a corrected token from:
    p'(x) = normalize( max(0, p_i(x) - q_i(x)) )
    This adjusted distribution ensures the overall output matches the target.

Step 3 — Bonus token (always):
    If all K draft tokens were accepted, sample one extra token from
    the target distribution at position K (p_{K}).

─── Guarantee ─────────────────────────────────────────────────────────────────
The combined accept/resample scheme produces tokens whose marginal distribution
is identical to sampling directly from the target. This is provable:

    P(output = x) = P(accept x_draft) * 1[x == x_draft]
                  + P(reject x_draft) * p'(x)
                  = min(p, q)/q * q[x] * 1[x == x_draft]   <- accept path
                  + (1 - sum_x min(p,q)) * p'(x)            <- reject path
                  = p(x)   ✓

─── This file is deliberately written in raw tensor ops ───────────────────────
No model calls happen here. Pure torch math, fully unit-testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AcceptanceResult:
    """
    Result of processing one round of K draft tokens.

    Attributes:
        n_accepted:     Number of draft tokens accepted (0 … K).
        accepted_ids:   The accepted token ids (length = n_accepted).
        bonus_token_id: A single bonus token sampled from the target distribution
                        at the first rejected (or last) position.
        acceptance_rate: n_accepted / K  (float in [0, 1]).
    """
    n_accepted: int
    accepted_ids: List[int]
    bonus_token_id: int
    acceptance_rate: float


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def accept_reject(
    p_probs: torch.Tensor,
    q_probs: torch.Tensor,
    draft_token_ids: List[int],
) -> AcceptanceResult:
    """
    Run the speculative decoding acceptance/rejection loop.

    Args:
        p_probs:         Target probability distributions. Shape [K+1, vocab_size].
                         p_probs[i] = target dist at position i (i=K is bonus).
        q_probs:         Draft probability distributions.  Shape [K, vocab_size].
                         q_probs[i] = draft dist that produced draft_token_ids[i].
        draft_token_ids: List of K token ids sampled by the draft model.

    Returns:
        AcceptanceResult with accepted token ids, bonus token, and acceptance rate.

    Raises:
        ValueError: If tensor shapes are inconsistent.
    """
    K = len(draft_token_ids)
    if p_probs.shape[0] != K + 1:
        raise ValueError(
            f"p_probs must have shape [K+1, V] = [{K+1}, V], got {list(p_probs.shape)}"
        )
    if q_probs.shape[0] != K:
        raise ValueError(
            f"q_probs must have shape [K, V] = [{K}, V], got {list(q_probs.shape)}"
        )

    accepted_ids: List[int] = []
    first_rejection: int = K  # default: all accepted

    for i in range(K):
        x_i = draft_token_ids[i]
        p_i = p_probs[i]   # [vocab_size]
        q_i = q_probs[i]   # [vocab_size]

        # Acceptance probability = min(1, p(x_i) / q(x_i))
        p_xi = p_i[x_i].item()
        q_xi = q_i[x_i].item()

        acceptance_prob = min(1.0, p_xi / (q_xi + 1e-9))

        u = torch.rand(1).item()
        if u <= acceptance_prob:
            accepted_ids.append(x_i)
        else:
            first_rejection = i
            break

    # Sample bonus (or correction) token
    if first_rejection < K:
        # Rejection at position `first_rejection`: resample from adjusted dist
        bonus_token_id = _resample_adjusted(
            p_probs[first_rejection],
            q_probs[first_rejection],
        )
    else:
        # All K accepted: sample bonus from target dist at position K
        bonus_token_id = _sample_from_probs(p_probs[K])

    return AcceptanceResult(
        n_accepted=len(accepted_ids),
        accepted_ids=accepted_ids,
        bonus_token_id=bonus_token_id,
        acceptance_rate=len(accepted_ids) / K,
    )


def compute_acceptance_probs(
    p_probs: torch.Tensor,
    q_probs: torch.Tensor,
    draft_token_ids: List[int],
) -> torch.Tensor:
    """
    Compute acceptance probability min(1, p(x_i)/q(x_i)) for all K positions.

    Useful for analysis and tests. Returns a 1-D tensor of shape [K].
    """
    K = len(draft_token_ids)
    acceptance = torch.zeros(K)
    for i, x_i in enumerate(draft_token_ids):
        p_xi = p_probs[i, x_i].item()
        q_xi = q_probs[i, x_i].item()
        acceptance[i] = min(1.0, p_xi / (q_xi + 1e-9))
    return acceptance


def expected_tokens_per_round(
    p_probs: torch.Tensor,
    q_probs: torch.Tensor,
    draft_token_ids: List[int],
) -> float:
    """
    Theoretical expected number of accepted tokens per round:
        E[n_accepted] = sum_{i=0}^{K-1} prod_{j=0}^{i} min(1, p_j/q_j)
    Plus 1 for the bonus token. Returns float.
    """
    acc_probs = compute_acceptance_probs(p_probs, q_probs, draft_token_ids)
    cumulative = torch.cumprod(acc_probs, dim=0)
    return float(cumulative.sum().item()) + 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resample_adjusted(
    p: torch.Tensor,
    q: torch.Tensor,
) -> int:
    """
    Sample from the corrected distribution: normalize(max(0, p - q)).

    This ensures that the overall output distribution remains identical to p.

    Args:
        p: Target probability vector, shape [vocab_size]. Sums to ~1.
        q: Draft probability vector,  shape [vocab_size]. Sums to ~1.

    Returns:
        Sampled token id (Python int).
    """
    adjusted = (p - q).clamp(min=0.0)
    total = adjusted.sum()
    if total < 1e-9:
        # Edge case: p ≤ q everywhere → fall back to sampling from p directly.
        return _sample_from_probs(p)
    adjusted = adjusted / total
    return _sample_from_probs(adjusted)


def _sample_from_probs(probs: torch.Tensor) -> int:
    """
    Sample one token from a probability vector.

    Args:
        probs: Shape [vocab_size]. Must be non-negative and sum to 1 (or ~1).

    Returns:
        Scalar token id (Python int).
    """
    # Handle any residual numerical issues
    probs = probs.clamp(min=0.0)
    probs = probs / probs.sum().clamp(min=1e-9)
    return int(torch.multinomial(probs.float(), num_samples=1).item())
