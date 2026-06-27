# Speculative Decoding from Scratch

A from-scratch implementation of speculative decoding using a draft model (Qwen2.5-0.5B) and a target model (Qwen2.5-7B-Instruct) to demonstrate lossless inference acceleration.

## Goal

Build a working speculative decoding engine that achieves 1.5x–2.5x tokens/sec improvement over standard autoregressive generation on the target model alone, with mathematically lossless output (same distribution as target-only generation).

## Directory Structure

```
spec-decode/
├── README.md                      # final writeup: results, plots, how to run
├── project.md                     # this file
├── pyproject.toml                 # or requirements.txt
├── configs/
│   └── default.yaml               # model names, K (draft length), sampling params
├── src/
│   ├── __init__.py
│   ├── models.py                  # load draft + target models/tokenizer
│   ├── draft.py                   # draft loop: generate K tokens with small model
│   ├── verify.py                  # single forward pass of target on prompt+draft
│   ├── rejection_sampling.py      # core math: accept/reject + adjusted resampling
│   ├── kv_cache.py                # manual KV cache management + rollback on rejection
│   ├── engine.py                  # orchestrates draft -> verify -> accept/reject -> bonus token loop
│   └── baseline.py                # standard autoregressive generation (for comparison)
├── benchmarks/
│   ├── prompts.json                # fixed set of 100 prompts
│   ├── run_benchmark.py            # runs baseline vs spec-decode, logs TTFT/ITL/tok-s
│   └── results/                    # CSV/JSON output + plots
├── tests/
│   ├── test_rejection_sampling.py  # toy distributions, verify math in isolation
│   ├── test_kv_cache.py            # verify rollback produces correct cache state
│   └── test_equivalence.py         # statistical test: spec-decode output dist == target-only dist
└── notebooks/
    └── exploration.ipynb           # scratch space for debugging logits/cache shapes
```

## Build Order

### Phase 1 — Baseline

- `src/models.py`: load both models + shared tokenizer, confirm vocab match
- `src/baseline.py`: standard greedy/sampling generation loop on target model alone
- Record baseline tokens/sec on a few prompts as your reference point

### Phase 2 — Draft + Verify

- `src/draft.py`: autoregressive loop on the 0.5B model, generate K tokens (start K=4)
- `src/verify.py`: feed prompt + draft tokens into target model in one forward pass, pull logits at each position
- Sanity check: confirm shapes line up (K draft tokens → K logit positions from target)

### Phase 3 — Rejection Sampling (do this in isolation first)

- `src/rejection_sampling.py`: implement accept-if-p≥q, else accept with prob p/q, else resample from `max(0, p - q)` normalized
- `tests/test_rejection_sampling.py`: validate against toy/synthetic distributions before touching real models
- This is the part worth understanding cold — write it in raw tensor ops, no shortcuts

### Phase 4 — KV Cache Rollback (hardest part)

- `src/kv_cache.py`: manage `past_key_values` manually; on partial rejection, truncate the cache back to the last accepted position
- `tests/test_kv_cache.py`: verify truncated cache matches what you’d get from a fresh forward pass up to that position
- This is where most implementations break — budget the most time here

### Phase 5 — Engine + Bonus Token

- `src/engine.py`: full loop — draft K tokens, verify, accept/reject, sample bonus token, roll back cache if needed, repeat
- Wire baseline.py and engine.py to share the same interface for easy comparison

### Phase 6 — Benchmark

- `benchmarks/run_benchmark.py`: run both engines across `prompts.json` (100 prompts)
- Metrics: TTFT (should be ~equal between both), tokens/sec (should show 1.5x–2.5x gain)
- `tests/test_equivalence.py`: confirm output distribution matches target-only (e.g. KL divergence on logits or repeated-sampling distribution check)

## Tech Stack

- PyTorch + Hugging Face `transformers`
- Qwen2.5-0.5B (draft) / Qwen2.5-7B-Instruct (target) — same tokenizer family
- Optional: `matplotlib` for benchmark plots in README

## Definition of Done

- [ ] Lossless equivalence test passes
- [ ] Benchmark shows 1.5x+ tokens/sec over baseline across 100 prompts
- [ ] README with methodology, results table, and a plot
- [ ] Code clean enough to link from resume/portfolio as a systems project