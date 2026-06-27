"""
baseline.py — Standard autoregressive generation using the target model alone.

This is the reference implementation we benchmark against. It uses a manual
token-by-token loop (not model.generate()) so we have tight control over
timing and KV cache — making it a fair comparison to spec-decode's engine.

Metrics recorded:
- TTFT  : Time To First Token (seconds)
- ITL   : Inter-Token Latency (seconds/token, after first token)
- tok/s : Tokens generated per second (total throughput)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    output_ids: List[int]
    output_text: str
    prompt_tokens: int
    generated_tokens: int
    ttft: float          # seconds to first token
    total_time: float    # wall-clock seconds for full generation
    tokens_per_sec: float
    itl: float           # inter-token latency (avg, excluding first token)

    def summary(self) -> str:
        return (
            f"Generated {self.generated_tokens} tokens | "
            f"TTFT={self.ttft*1000:.1f}ms | "
            f"ITL={self.itl*1000:.1f}ms | "
            f"tok/s={self.tokens_per_sec:.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_baseline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    seed: Optional[int] = None,
) -> GenerationResult:
    """
    Run standard autoregressive generation on *model* for one prompt.

    Args:
        model: The target (or any causal LM) in eval mode.
        tokenizer: Shared tokenizer.
        prompt: Raw text prompt.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 → greedy via argmax).
        top_p: Nucleus sampling cutoff.
        top_k: Top-k sampling cutoff (0 = disabled).
        seed: Optional random seed for reproducibility.

    Returns:
        GenerationResult with timing metrics and decoded text.
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = _model_device(model)

    # Tokenize prompt
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids: torch.Tensor = enc["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    generated_ids: List[int] = []
    past_key_values = None
    current_ids = input_ids

    ttft: Optional[float] = None
    t_start = time.perf_counter()

    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(
                input_ids=current_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]  # [1, vocab_size]
            past_key_values = outputs.past_key_values

            next_token_id = _sample(logits, temperature, top_p, top_k)

            t_now = time.perf_counter()
            if ttft is None:
                ttft = t_now - t_start

            generated_ids.append(next_token_id)
            current_ids = torch.tensor([[next_token_id]], device=device)

            if next_token_id == tokenizer.eos_token_id:
                break

    t_end = time.perf_counter()
    total_time = t_end - t_start
    n_gen = len(generated_ids)

    # ITL = average time per token after the first (first token includes prompt prefill)
    if n_gen > 1:
        itl = (total_time - (ttft or 0.0)) / (n_gen - 1)
    else:
        itl = 0.0

    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return GenerationResult(
        output_ids=generated_ids,
        output_text=output_text,
        prompt_tokens=prompt_len,
        generated_tokens=n_gen,
        ttft=ttft or total_time,
        total_time=total_time,
        tokens_per_sec=n_gen / total_time if total_time > 0 else 0.0,
        itl=itl,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helper (shared with engine.py and draft.py)
# ─────────────────────────────────────────────────────────────────────────────

def _sample(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
) -> int:
    """
    Sample one token from *logits* using temperature + top-p + top-k.

    Args:
        logits: Shape [1, vocab_size] — raw (unnormalized) logits.
        temperature: Divide logits by this. 0 → argmax (greedy).
        top_p: Nucleus cutoff (0–1). 1.0 = no cutoff.
        top_k: Keep top-k tokens before softmax. 0 = no cutoff.

    Returns:
        Scalar token id (Python int).
    """
    if temperature == 0.0:
        return int(logits.argmax(dim=-1).item())

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        topk_vals, _ = torch.topk(logits, k, dim=-1)
        threshold = topk_vals[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    # Top-p (nucleus) filtering
    if 0.0 < top_p < 1.0:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # Remove tokens once cumulative probability exceeds top_p
        sorted_filter = (cumulative_probs - sorted_probs) > top_p
        sorted_probs[sorted_filter] = 0.0
        # Scatter back to original order
        probs = torch.zeros_like(probs).scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)  # renormalize
    else:
        probs = torch.softmax(logits, dim=-1)

    return int(torch.multinomial(probs, num_samples=1).item())


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _model_device(model: PreTrainedModel) -> torch.device:
    """Return the device of the first model parameter."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point — quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml
    from pathlib import Path
    from src.models import load_models

    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    pair = load_models(config)

    test_prompts = [
        "Explain the difference between a transformer and an RNN in simple terms.",
        "Write a Python function to compute the Fibonacci sequence iteratively.",
    ]

    sd_cfg = config["speculative_decoding"]
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt[:60]}…")
        result = generate_baseline(
            model=pair.target,
            tokenizer=pair.tokenizer,
            prompt=prompt,
            max_new_tokens=sd_cfg["max_new_tokens"],
            temperature=sd_cfg["temperature"],
            top_p=sd_cfg["top_p"],
            top_k=sd_cfg["top_k"],
            seed=sd_cfg["seed"],
        )
        print(result.summary())
        print(f"Output: {result.output_text[:200]}…")
