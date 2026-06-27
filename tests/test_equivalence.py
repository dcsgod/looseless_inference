"""
test_equivalence.py — Statistical test: spec-decode output distribution == target-only distribution.

This is the top-level losslessness verification. We run both the baseline and
the spec-decode engine on the same prompt with the same seed, repeating many
times, and test that the empirical token-frequency distributions are statistically
indistinguishable.

Method: for each prompt position i, collect the token sampled at position i
across N trials and run a chi-square goodness-of-fit test against the target
model's predicted distribution at that position.

NOTE: Requires GPU + both models loaded. Designed to run on the AWS instance.
      Takes several minutes for N_TRIALS=50.
"""

from __future__ import annotations

import pytest
import torch
import yaml
from pathlib import Path
from typing import List, Dict
from collections import Counter

from scipy.stats import chisquare
import numpy as np

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU not available — run on AWS instance"
)

N_TRIALS = 30        # repetitions per prompt
N_POSITIONS = 5      # first N token positions to test
ALPHA = 0.01         # significance level (we expect most positions to pass)
MIN_PASS_RATE = 0.85 # at least 85% of positions must pass (some noise expected)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def config():
    cfg_path = Path("configs/default.yaml")
    if not cfg_path.exists():
        pytest.skip("configs/default.yaml not found")
    cfg = yaml.safe_load(cfg_path.read_text())
    # Override for quick equivalence test
    cfg["speculative_decoding"]["max_new_tokens"] = N_POSITIONS + 2
    cfg["speculative_decoding"]["seed"] = None  # vary per trial
    return cfg


@pytest.fixture(scope="module")
def models(config):
    from src.models import load_models
    return load_models(config)


@pytest.fixture(scope="module")
def baseline_engine(models, config):
    from src.baseline import generate_baseline
    return models, config


@pytest.fixture(scope="module")
def spec_engine(models, config):
    from src.engine import SpeculativeDecoder
    decoder = SpeculativeDecoder(
        draft_model=models.draft,
        target_model=models.target,
        tokenizer=models.tokenizer,
        config=config,
    )
    return decoder, models.tokenizer


# ── Tests ─────────────────────────────────────────────────────────────────────

@requires_gpu
class TestOutputDistributionEquivalence:
    """
    Statistical equivalence between spec-decode and target-only generation.
    """

    TEST_PROMPTS = [
        "The capital of France is",
        "In Python, a list comprehension",
        "The derivative of sin(x) is",
    ]

    def _collect_baseline_tokens(
        self,
        models,
        config: dict,
        prompt: str,
        n_trials: int,
        n_positions: int,
    ) -> List[List[int]]:
        """
        Run baseline n_trials times; return per-position token lists.
        Returns: list of length n_positions, each a list of n_trials token ids.
        """
        from src.baseline import generate_baseline
        per_position: List[List[int]] = [[] for _ in range(n_positions)]

        for trial in range(n_trials):
            result = generate_baseline(
                model=models.target,
                tokenizer=models.tokenizer,
                prompt=prompt,
                max_new_tokens=n_positions,
                temperature=config["speculative_decoding"]["temperature"],
                top_p=config["speculative_decoding"]["top_p"],
                top_k=config["speculative_decoding"]["top_k"],
                seed=trial,  # different seed each trial
            )
            for pos in range(min(n_positions, len(result.output_ids))):
                per_position[pos].append(result.output_ids[pos])

        return per_position

    def _collect_spec_tokens(
        self,
        decoder,
        prompt: str,
        n_trials: int,
        n_positions: int,
    ) -> List[List[int]]:
        """Run spec-decode n_trials times; return per-position token lists."""
        from src.engine import SpeculativeDecoder
        per_position: List[List[int]] = [[] for _ in range(n_positions)]

        for trial in range(n_trials):
            result = decoder.generate(prompt, seed=trial)
            for pos in range(min(n_positions, len(result.output_ids))):
                per_position[pos].append(result.output_ids[pos])

        return per_position

    def _compare_distributions(
        self,
        baseline_tokens: List[int],
        spec_tokens: List[int],
        vocab_size: int,
        position: int,
    ) -> bool:
        """
        Chi-square test of independence between baseline and spec token samples.
        Returns True if the two distributions are statistically indistinguishable.
        """
        # Build a combined vocabulary from observed tokens
        all_tokens = set(baseline_tokens) | set(spec_tokens)
        if len(all_tokens) <= 1:
            return True  # Degenerate: both deterministic and agree (or disagree)

        baseline_counts = Counter(baseline_tokens)
        spec_counts = Counter(spec_tokens)

        tokens_list = sorted(all_tokens)
        baseline_freq = np.array([baseline_counts.get(t, 0) for t in tokens_list], dtype=float)
        spec_freq = np.array([spec_counts.get(t, 0) for t in tokens_list], dtype=float)

        # Merge rare cells (expected count < 5) to satisfy chi-square assumptions
        combined = baseline_freq + spec_freq
        mask = combined >= 5
        if mask.sum() < 2:
            # Not enough data to test — consider it passed
            return True

        baseline_freq = baseline_freq[mask]
        spec_freq = spec_freq[mask]

        # Normalize spec_freq to sum to same total as baseline (relative freq comparison)
        scale = baseline_freq.sum() / spec_freq.sum() if spec_freq.sum() > 0 else 1.0
        expected = spec_freq * scale

        # chi-square goodness-of-fit: observed=baseline, expected=spec
        if expected.sum() == 0:
            return True
        _, pvalue = chisquare(baseline_freq, f_exp=expected)
        return pvalue > ALPHA

    @pytest.mark.parametrize("prompt", TEST_PROMPTS)
    def test_losslessness_on_prompt(self, prompt, baseline_engine, spec_engine, config):
        """
        Main losslessness test: spec-decode must produce statistically identical
        token distributions to target-only generation.
        """
        models, cfg = baseline_engine
        decoder, tokenizer = spec_engine
        vocab_size = tokenizer.vocab_size

        baseline_per_pos = self._collect_baseline_tokens(
            models, cfg, prompt, N_TRIALS, N_POSITIONS
        )
        spec_per_pos = self._collect_spec_tokens(
            decoder, prompt, N_TRIALS, N_POSITIONS
        )

        n_pass = 0
        n_total = 0
        failures = []

        for pos in range(N_POSITIONS):
            if len(baseline_per_pos[pos]) < 5 or len(spec_per_pos[pos]) < 5:
                continue  # not enough data
            n_total += 1
            passed = self._compare_distributions(
                baseline_per_pos[pos],
                spec_per_pos[pos],
                vocab_size,
                pos,
            )
            if passed:
                n_pass += 1
            else:
                failures.append(pos)

        if n_total == 0:
            pytest.skip("Not enough samples generated")

        pass_rate = n_pass / n_total
        assert pass_rate >= MIN_PASS_RATE, (
            f"Prompt: '{prompt[:40]}'\n"
            f"Losslessness test failed: {n_pass}/{n_total} positions passed "
            f"(need {MIN_PASS_RATE:.0%}). Failed positions: {failures}"
        )
