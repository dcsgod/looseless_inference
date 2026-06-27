"""
engine.py — Speculative decoding engine: orchestrates the full draft → verify → accept/reject → bonus loop.

Architecture of one generation round:
┌─────────────────────────────────────────────────────────────┐
│  1. Draft K tokens from small model  (K cheap forward passes)│
│  2. Verify with target in ONE forward pass  (1 expensive pass)│
│  3. Accept/reject each draft token via rejection sampling     │
│  4. Always emit a bonus token from target at rejection point  │
│  5. Rollback target KV cache to last accepted position        │
│  6. Repeat until max_new_tokens reached or EOS emitted        │
└─────────────────────────────────────────────────────────────┘

Speedup intuition:
  - Standard generation: each token = 1 target forward pass
  - Spec-decode:  each round = 1 target pass that produces 1…K+1 tokens
  - If draft is accurate (high acceptance), avg tokens/round ≈ K+1
    at the cost of K draft passes (cheap) + 1 target pass
  - Net gain ≈ K * (target_latency / draft_latency)   [simplified]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.draft import generate_draft
from src.verify import verify_draft
from src.rejection_sampling import accept_reject
from src.kv_cache import KVCacheManager
from src.baseline import GenerationResult


# ─────────────────────────────────────────────────────────────────────────────
# Engine result (extends GenerationResult with spec-decode-specific metrics)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpecDecodeResult(GenerationResult):
    n_rounds: int = 0
    mean_acceptance_rate: float = 0.0
    mean_tokens_per_round: float = 0.0

    def summary(self) -> str:
        base = super().summary()
        return (
            f"{base} | "
            f"rounds={self.n_rounds} | "
            f"accept_rate={self.mean_acceptance_rate:.2%} | "
            f"tok/round={self.mean_tokens_per_round:.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main engine class
# ─────────────────────────────────────────────────────────────────────────────

class SpeculativeDecoder:
    """
    Lossless speculative decoding engine.

    Usage::

        decoder = SpeculativeDecoder(draft_model, target_model, tokenizer, config)
        result  = decoder.generate("Your prompt here")
        print(result.summary())
        print(result.output_text)
    """

    def __init__(
        self,
        draft_model: PreTrainedModel,
        target_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: dict,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer

        sd_cfg = config.get("speculative_decoding", {})
        self.K: int = sd_cfg.get("K", 4)
        self.temperature: float = sd_cfg.get("temperature", 1.0)
        self.top_p: float = sd_cfg.get("top_p", 0.9)
        self.top_k: int = sd_cfg.get("top_k", 50)
        self.default_max_new_tokens: int = sd_cfg.get("max_new_tokens", 256)
        self.seed: Optional[int] = sd_cfg.get("seed", None)

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> SpecDecodeResult:
        """
        Generate text from *prompt* using speculative decoding.

        Args:
            prompt:         Raw text prompt.
            max_new_tokens: Override config max_new_tokens.
            seed:           Override config seed.

        Returns:
            SpecDecodeResult with generated text and all timing/acceptance metrics.
        """
        _seed = seed if seed is not None else self.seed
        if _seed is not None:
            torch.manual_seed(_seed)

        max_new_tokens = max_new_tokens or self.default_max_new_tokens
        target_device = _model_device(self.target_model)

        # ── Tokenize prompt ──────────────────────────────────────────────────
        enc = self.tokenizer(prompt, return_tensors="pt")
        prompt_ids: torch.Tensor = enc["input_ids"].to(target_device)
        prompt_len = prompt_ids.shape[1]

        # ── State tracking ───────────────────────────────────────────────────
        # context_ids always includes prompt + all accepted tokens so far
        context_ids = prompt_ids.clone()
        generated_ids: List[int] = []

        # KV caches — target cache tracks accepted context; draft cache is
        # re-used within each round and snapshotted before draft generation
        target_cache = KVCacheManager()  # starts empty; filled after first verify
        draft_cache = KVCacheManager()   # starts empty; filled during draft loop

        # Metrics
        acceptance_rates: List[float] = []
        tokens_per_round_list: List[int] = []
        n_rounds = 0
        ttft: Optional[float] = None

        t_start = time.perf_counter()

        # ── Prefill: run target on prompt to populate its KV cache ───────────
        # We feed the prompt first so subsequent verify calls only need draft tokens.
        with torch.no_grad():
            prefill_out = self.target_model(
                input_ids=context_ids,
                past_key_values=None,
                use_cache=True,
            )
        target_cache = KVCacheManager(prefill_out.past_key_values)
        # Also prefill draft model cache on prompt
        draft_device = _model_device(self.draft_model)
        with torch.no_grad():
            draft_prefill_out = self.draft_model(
                input_ids=context_ids.to(draft_device),
                past_key_values=None,
                use_cache=True,
            )
        draft_cache = KVCacheManager(draft_prefill_out.past_key_values)

        # ── Main speculative decoding loop ───────────────────────────────────
        while len(generated_ids) < max_new_tokens:
            tokens_left = max_new_tokens - len(generated_ids)
            K = min(self.K, tokens_left)  # don't over-generate at the end

            # 1. Draft K tokens ───────────────────────────────────────────────
            # Feed only the LAST generated token (or empty if first round) so
            # the draft model continues from its cache.
            if len(generated_ids) == 0:
                # First round: cache already has full prompt, feed an empty
                # continuation — we need the last-token id to continue.
                # Actually: the draft cache is positioned at end of prompt.
                # We'll feed a dummy single-token to move it forward.
                # Correction: we pass context_ids and rely on draft to feed
                # only the newest token since its cache covers the rest.
                draft_input = context_ids[:, -1:].to(draft_device)
            else:
                draft_input = torch.tensor(
                    [[generated_ids[-1]]], dtype=torch.long, device=draft_device
                )

            draft_token_ids, draft_probs, draft_cache_new = generate_draft(
                draft_model=self.draft_model,
                input_ids=draft_input,
                K=K,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                cache_manager=draft_cache,
            )
            draft_cache = draft_cache_new

            # 2. Verify with target in one forward pass ───────────────────────
            target_probs, target_cache_new = verify_draft(
                target_model=self.target_model,
                context_ids=context_ids,
                draft_token_ids=draft_token_ids,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                target_cache=target_cache,
            )
            # target_cache_new covers context + all K draft tokens

            # 3. Accept / Reject ──────────────────────────────────────────────
            result = accept_reject(
                p_probs=target_probs,
                q_probs=draft_probs,
                draft_token_ids=draft_token_ids,
            )
            acceptance_rates.append(result.acceptance_rate)
            n_rounds += 1

            # 4. Collect accepted tokens + bonus ──────────────────────────────
            new_tokens: List[int] = result.accepted_ids + [result.bonus_token_id]
            tokens_this_round = len(new_tokens)
            tokens_per_round_list.append(tokens_this_round)

            # Record TTFT at first token
            if ttft is None:
                ttft = time.perf_counter() - t_start

            # 5. Update context and generated list ────────────────────────────
            for tok in new_tokens:
                generated_ids.append(tok)
                if tok == self.tokenizer.eos_token_id:
                    break
            else:
                # Update context_ids with all new tokens (no EOS encountered)
                new_token_tensor = torch.tensor(
                    [new_tokens], dtype=torch.long, device=target_device
                )
                context_ids = torch.cat([context_ids, new_token_tensor], dim=1)

                # 6. Rollback target KV cache to last accepted position ────────
                # After accepting n_accepted draft tokens + 1 bonus, the cache
                # should cover: prompt_len + previous_generated + n_accepted + 1
                # That is exactly the length of context_ids now.
                accepted_total = prompt_len + len(generated_ids)
                if result.n_accepted < K:
                    # Partial acceptance: truncate target cache to accepted prefix
                    # target_cache_new covers prompt + all K drafts
                    # We need to cut it back to prompt + n_accepted
                    target_cache = target_cache_new.truncate(
                        prompt_len + len(generated_ids) - 1  # -1 because bonus not in cache yet
                    )
                    # Also reset draft cache to match accepted position
                    # Draft cache covers: original context + K draft tokens
                    # Truncate to: original context + n_accepted
                    draft_context_len = context_ids.shape[1] - tokens_this_round  # before this round
                    draft_cache = draft_cache.truncate(
                        draft_context_len + result.n_accepted
                    )
                else:
                    # All K accepted: target cache covers everything perfectly
                    target_cache = target_cache_new

                continue

            # EOS was found — break outer loop too
            break

        t_end = time.perf_counter()
        total_time = t_end - t_start
        n_gen = len(generated_ids)

        if n_gen > 1:
            itl = (total_time - (ttft or 0.0)) / (n_gen - 1)
        else:
            itl = 0.0

        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return SpecDecodeResult(
            output_ids=generated_ids,
            output_text=output_text,
            prompt_tokens=prompt_len,
            generated_tokens=n_gen,
            ttft=ttft or total_time,
            total_time=total_time,
            tokens_per_sec=n_gen / total_time if total_time > 0 else 0.0,
            itl=itl,
            n_rounds=n_rounds,
            mean_acceptance_rate=sum(acceptance_rates) / len(acceptance_rates) if acceptance_rates else 0.0,
            mean_tokens_per_round=sum(tokens_per_round_list) / len(tokens_per_round_list) if tokens_per_round_list else 0.0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _model_device(model: PreTrainedModel) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml
    from pathlib import Path
    from src.models import load_models

    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    pair = load_models(config)

    decoder = SpeculativeDecoder(
        draft_model=pair.draft,
        target_model=pair.target,
        tokenizer=pair.tokenizer,
        config=config,
    )

    prompts = [
        "Explain the concept of speculative decoding in one paragraph.",
        "Write a Python function to reverse a linked list.",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt[:70]}…")
        result = decoder.generate(prompt)
        print(result.summary())
        print(f"Output: {result.output_text[:300]}…")
