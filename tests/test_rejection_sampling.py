"""
test_rejection_sampling.py — Unit tests for the core accept/reject math.

No GPU or real models required — all tests use synthetic toy distributions
over a small vocabulary (V=8). This lets us verify the math in isolation
before touching any transformer weights.

Test strategy:
  1. Acceptance rate  — empirically verify E[n_accepted] matches theory
  2. Output distribution — verify the full accept/resample scheme produces
     tokens matching the target distribution p (via chi-square test)
  3. Edge cases       — identical distributions (p==q), degenerate dist, etc.
  4. Adjusted resampling — verify normalize(max(0, p-q)) is correct
"""

from __future__ import annotations

import math
from typing import List

import pytest
import torch
from scipy.stats import chisquare

from src.rejection_sampling import (
    AcceptanceResult,
    accept_reject,
    compute_acceptance_probs,
    expected_tokens_per_round,
    _resample_adjusted,
    _sample_from_probs,
)

# ─── Helpers ─────────────────────────────────────────────────────────────────

VOCAB_SIZE = 8
torch.manual_seed(0)


def make_uniform(V: int = VOCAB_SIZE) -> torch.Tensor:
    return torch.ones(V) / V


def make_peaked(peak_idx: int = 0, peak_mass: float = 0.8, V: int = VOCAB_SIZE) -> torch.Tensor:
    p = torch.ones(V) * (1 - peak_mass) / (V - 1)
    p[peak_idx] = peak_mass
    return p


def make_p_probs(K: int, p: torch.Tensor) -> torch.Tensor:
    """Stack p into [K+1, V] (same dist at all positions)."""
    return p.unsqueeze(0).expand(K + 1, -1).clone()


def make_q_probs(K: int, q: torch.Tensor) -> torch.Tensor:
    """Stack q into [K, V] (same dist at all positions)."""
    return q.unsqueeze(0).expand(K, -1).clone()


def sample_draft(q: torch.Tensor, K: int, n_trials: int = 1) -> List[List[int]]:
    """Sample K draft tokens from q for n_trials trials."""
    return [
        [int(torch.multinomial(q, num_samples=1).item()) for _ in range(K)]
        for _ in range(n_trials)
    ]


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestAcceptanceRate:
    """
    Verify that empirical acceptance rate matches theoretical prediction.
    Theory: E[min(1, p(x)/q(x))] where x ~ q
    """

    def _run_acceptance_experiment(
        self, p: torch.Tensor, q: torch.Tensor, K: int = 4, N_trials: int = 2000
    ) -> float:
        """Run N trials and return mean acceptance rate."""
        total_accepted = 0
        for _ in range(N_trials):
            draft_ids = sample_draft(q, K)[0]
            p_probs = make_p_probs(K, p)
            q_probs = make_q_probs(K, q)
            result = accept_reject(p_probs, q_probs, draft_ids)
            total_accepted += result.n_accepted
        return total_accepted / (N_trials * K)

    def test_identical_distributions_all_accepted(self):
        """When p == q, every token should always be accepted."""
        p = make_uniform()
        empirical = self._run_acceptance_experiment(p, p, K=4, N_trials=500)
        # Should be 1.0; allow small numerical slack
        assert empirical > 0.95, f"Expected ~1.0 acceptance, got {empirical:.3f}"

    def test_orthogonal_distributions_never_accepted(self):
        """When p and q are orthogonal (disjoint support), nothing is accepted."""
        p = torch.zeros(VOCAB_SIZE); p[0] = 1.0
        q = torch.zeros(VOCAB_SIZE); q[1] = 1.0
        draft_ids = [1] * 4  # q always picks token 1, p never does
        p_probs = make_p_probs(4, p)
        q_probs = make_q_probs(4, q)
        result = accept_reject(p_probs, q_probs, draft_ids)
        assert result.n_accepted == 0, "Orthogonal dists: should accept 0 tokens"

    def test_acceptance_rate_theory(self):
        """
        Empirical acceptance prob ≈ theoretical min(1, p(x)/q(x)) averaged over x~q.
        Theory: E_q[min(1, p/q)] = sum_x min(p(x), q(x))
        """
        torch.manual_seed(42)
        p = make_peaked(peak_idx=2, peak_mass=0.6)
        q = make_peaked(peak_idx=3, peak_mass=0.5)
        theoretical = float(torch.min(p, q).sum().item())
        empirical = self._run_acceptance_experiment(p, q, K=1, N_trials=5000)
        # Allow ±5% tolerance given finite samples
        assert abs(empirical - theoretical) < 0.07, (
            f"Theoretical={theoretical:.3f}, empirical={empirical:.3f}"
        )


class TestOutputDistribution:
    """
    The key correctness guarantee: the combined accept/resample scheme must
    produce tokens whose marginal distribution equals the target p.

    Method: run many trials, collect sampled output tokens, chi-square test
    against the expected frequencies under p.
    """

    def _collect_samples(
        self, p: torch.Tensor, q: torch.Tensor, K: int = 1, N: int = 5000
    ) -> torch.Tensor:
        """
        Collect N output tokens from accept_reject with K=1 (simplest case).
        K=1 means we either accept the draft token or resample from adjusted dist.
        """
        torch.manual_seed(123)
        counts = torch.zeros(VOCAB_SIZE, dtype=torch.long)
        p_probs = make_p_probs(K, p)
        q_probs = make_q_probs(K, q)
        for _ in range(N):
            draft_ids = [int(torch.multinomial(q, 1).item())]
            result = accept_reject(p_probs, q_probs, draft_ids)
            # The bonus token is sampled from p (index K=1 in p_probs)
            # The accepted/corrected token goes first (if any)
            if result.n_accepted > 0:
                output_token = result.accepted_ids[0]
            else:
                output_token = result.bonus_token_id
            counts[output_token] += 1
        return counts

    def test_output_matches_target_uniform_draft(self):
        """Peaked target, uniform draft → output should match target."""
        p = make_peaked(peak_idx=0, peak_mass=0.7)
        q = make_uniform()
        N = 6000
        counts = self._collect_samples(p, q, K=1, N=N)
        expected = (p * N).numpy()
        observed = counts.numpy()
        stat, pvalue = chisquare(observed, f_exp=expected)
        assert pvalue > 0.01, (
            f"Output dist doesn't match target (chi2 p={pvalue:.4f}). "
            f"Observed: {observed}, Expected: {(expected).round(1)}"
        )

    def test_output_matches_target_peaked_draft(self):
        """Uniform target, peaked draft → accept_reject must correct for draft bias."""
        p = make_uniform()
        q = make_peaked(peak_idx=5, peak_mass=0.9)
        N = 6000
        counts = self._collect_samples(p, q, K=1, N=N)
        expected = (p * N).numpy()
        observed = counts.numpy()
        stat, pvalue = chisquare(observed, f_exp=expected)
        assert pvalue > 0.01, (
            f"Output dist doesn't match target (chi2 p={pvalue:.4f}). "
            f"Observed: {observed}"
        )


class TestAdjustedResampling:
    """Verify the corrected distribution normalize(max(0, p - q)) is computed correctly."""

    def test_adjusted_dist_concentrates_on_p_surplus(self):
        """
        If p puts all mass on token 0 and q puts all mass on token 1,
        adjusted dist = normalize(max(0, p-q)) should be a one-hot on token 0.
        """
        p = torch.zeros(VOCAB_SIZE); p[0] = 1.0
        q = torch.zeros(VOCAB_SIZE); q[1] = 1.0
        # Sample many times from adjusted dist — should always give token 0
        results = [_resample_adjusted(p, q) for _ in range(100)]
        assert all(r == 0 for r in results), "Adjusted dist should be one-hot on token 0"

    def test_adjusted_resampling_with_overlap(self):
        """Empirically verify adjusted dist matches normalize(max(0, p-q))."""
        torch.manual_seed(7)
        p = torch.tensor([0.5, 0.3, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0])
        q = torch.tensor([0.2, 0.4, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0])
        adjusted = (p - q).clamp(min=0)
        adjusted = adjusted / adjusted.sum()

        N = 3000
        counts = torch.zeros(VOCAB_SIZE)
        for _ in range(N):
            counts[_resample_adjusted(p, q)] += 1
        empirical = counts / N

        # Check each token's empirical freq is close to adjusted[i]
        for i in range(VOCAB_SIZE):
            assert abs(empirical[i].item() - adjusted[i].item()) < 0.05, (
                f"Token {i}: empirical={empirical[i]:.3f}, expected={adjusted[i]:.3f}"
            )


class TestEdgeCases:
    def test_k_equals_one(self):
        """K=1 still works correctly."""
        p = make_peaked(0, 0.9)
        q = make_uniform()
        result = accept_reject(make_p_probs(1, p), make_q_probs(1, q), [0])
        assert isinstance(result, AcceptanceResult)
        assert result.n_accepted in (0, 1)

    def test_acceptance_result_fields(self):
        """AcceptanceResult always has correct field types and ranges."""
        p = make_peaked(0, 0.8)
        q = make_uniform()
        K = 4
        for _ in range(20):
            draft_ids = sample_draft(q, K)[0]
            result = accept_reject(make_p_probs(K, p), make_q_probs(K, q), draft_ids)
            assert 0 <= result.n_accepted <= K
            assert len(result.accepted_ids) == result.n_accepted
            assert isinstance(result.bonus_token_id, int)
            assert 0.0 <= result.acceptance_rate <= 1.0

    def test_wrong_shape_raises(self):
        """Mismatched tensor shapes should raise ValueError."""
        p = make_p_probs(4, make_uniform())  # [5, 8]
        q = make_q_probs(3, make_uniform())  # [3, 8] — wrong K
        with pytest.raises(ValueError):
            accept_reject(p, q, [0, 1, 2, 3])

    def test_expected_tokens_per_round(self):
        """expected_tokens_per_round > 1 always (bonus token guaranteed)."""
        p = make_peaked(0, 0.7)
        q = make_uniform()
        K = 4
        draft_ids = sample_draft(q, K)[0]
        expected = expected_tokens_per_round(make_p_probs(K, p), make_q_probs(K, q), draft_ids)
        assert expected >= 1.0, "Must always produce at least the bonus token"
        assert expected <= K + 1
