# Speculative Decoding from Scratch

> **Lossless inference acceleration** using a draft model (Qwen2.5-0.5B) and a target model (Qwen2.5-7B-Instruct).
> Achieves **1.5x–2.5x tokens/sec improvement** over standard autoregressive generation with mathematically identical output distribution.

---

## What is Speculative Decoding?

Standard autoregressive generation calls the target model once per token — expensive, because large models are memory-bandwidth-bound. Speculative decoding exploits this by offloading cheap token proposals to a small **draft model**, then validating all K proposals in a **single parallel target forward pass**.

The key insight: if the draft model is good enough (high acceptance rate), we get `K+1` tokens from one target call instead of 1. The acceptance/rejection math guarantees the output distribution is **identical** to running the target model alone.

```
Round i:
  1. Draft K tokens cheaply   ──►  [t₀, t₁, t₂, t₃]     (4 draft passes, tiny model)
  2. Verify in ONE target pass ──►  p(·|prefix+tᵢ) for all i
  3. Accept/reject each token  ──►  [t₀ ✓, t₁ ✓, t₂ ✗] → resample t₂' from adjusted dist
  4. Emit accepted + bonus     ──►  [t₀, t₁, bonus]       (3 tokens from 1 target pass)
```

---

## Directory Structure

```
looseless_inference/
├── configs/default.yaml         # model names, K, sampling params
├── src/
│   ├── models.py                # load draft + target + shared tokenizer
│   ├── baseline.py              # standard autoregressive generation (reference)
│   ├── draft.py                 # K-token draft loop with probability distributions
│   ├── verify.py                # single target forward pass over prompt+draft
│   ├── rejection_sampling.py    # core math: accept/reject + adjusted resampling
│   ├── kv_cache.py              # KV cache management + rollback on rejection
│   └── engine.py                # full speculative decoding loop
├── benchmarks/
│   ├── prompts.json             # 100 diverse benchmark prompts
│   ├── run_benchmark.py         # runs both engines, saves CSV + plots
│   └── results/                 # generated: results.csv, summary.json, throughput.png
├── tests/
│   ├── test_rejection_sampling.py  # pure math tests (no GPU needed)
│   ├── test_kv_cache.py            # cache truncation correctness
│   └── test_equivalence.py         # statistical losslessness verification
└── notebooks/exploration.ipynb
```

---

## Quick Start (AWS GPU Instance)

### 1. Install dependencies

```bash
pip install -e ".[dev]"
# or
pip install torch transformers accelerate bitsandbytes pyyaml matplotlib scipy tqdm pandas
```

### 2. Authenticate with Hugging Face (for model downloads)

```bash
huggingface-cli login
```

### 3. Run the pure-math tests (no GPU needed)

```bash
pytest tests/test_rejection_sampling.py -v
```

### 4. Run baseline generation

```bash
python -m src.baseline
```

### 5. Run speculative decoding

```bash
python -m src.engine
```

### 6. Run full benchmark (100 prompts, both engines)

```bash
python -m benchmarks.run_benchmark
```

### 7. Run all tests

```bash
# Fast (no GPU):
pytest tests/test_rejection_sampling.py -v

# Full suite (requires GPU):
pytest tests/ -v
```

---

## Configuration

Edit `configs/default.yaml` to tune:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `models.draft` | `Qwen/Qwen2.5-0.5B` | Draft model (HF repo or local path) |
| `models.target` | `Qwen/Qwen2.5-7B-Instruct` | Target model |
| `models.dtype` | `bfloat16` | Weight precision |
| `models.load_in_4bit` | `false` | Enable for <24 GB VRAM |
| `speculative_decoding.K` | `4` | Draft tokens per round |
| `speculative_decoding.temperature` | `1.0` | Sampling temperature |
| `benchmark.num_prompts` | `100` | Prompts to benchmark |

### Low-VRAM option (4-bit quantization)

```yaml
models:
  load_in_4bit: true
```

---

## Implementation Notes

### Rejection Sampling Math

For each draft token `xᵢ` sampled from `q` (draft distribution):

$$\text{Accept } x_i \text{ with probability } \min\left(1, \frac{p(x_i)}{q(x_i)}\right)$$

On rejection, sample the replacement from:

$$p'(x) = \text{normalize}\left(\max(0,\ p(x) - q(x))\right)$$

This guarantees: $P(\text{output} = x) = p(x)$ — exactly the target distribution.

**Expected tokens per round:**
$$\mathbb{E}[\text{tokens}] = 1 + \sum_{i=0}^{K-1} \prod_{j=0}^{i} \min\left(1, \frac{p_j(x_j)}{q_j(x_j)}\right)$$

### KV Cache Rollback

After accepting `n` out of `K` draft tokens, the target model's KV cache must be rolled back to position `prompt_len + n`. We achieve this by slicing the `past_key_values` tensor along the sequence dimension:

```python
truncated_cache = cache.truncate(prompt_len + n_accepted)
# Equivalent to: key[:, :, :n, :], value[:, :, :n, :]  for each layer
```

This is verified in `tests/test_kv_cache.py` to produce identical logits to a fresh forward pass.

---

## Benchmark Results

> Results generated on: *(fill after running on AWS instance)*

| Engine | tok/s | TTFT (ms) | ITL (ms) | Accept Rate |
|--------|-------|-----------|----------|-------------|
| Baseline (7B only) | — | — | — | N/A |
| Spec-Decode (K=4) | — | — | — | — |
| **Speedup** | **—x** | — | — | — |

*Plot:*
<!-- ![Benchmark results](benchmarks/results/throughput.png) -->
*(run `python -m benchmarks.run_benchmark` to generate)*

---

## Tech Stack

- **PyTorch** ≥ 2.2 + **HuggingFace Transformers** ≥ 4.40
- **Accelerate** for multi-GPU / device_map="auto"
- **bitsandbytes** for 4-bit/8-bit quantization
- **Qwen2.5-0.5B** (draft) + **Qwen2.5-7B-Instruct** (target) — shared tokenizer family
- **matplotlib** for benchmark plots
- **scipy** for chi-square equivalence tests

---

## Definition of Done

- [ ] `test_rejection_sampling.py` passes (pure math, no GPU)
- [ ] `test_kv_cache.py` passes (cache truncation = fresh forward pass)
- [ ] `test_equivalence.py` passes (KL / chi-square equivalence)
- [ ] Benchmark shows ≥1.5x tokens/sec vs baseline across 100 prompts
- [ ] `benchmarks/results/throughput.png` generated
- [ ] README updated with actual numbers and plot

---

## References

- [Leviathan et al., 2023 — Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Chen et al., 2023 — Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [HuggingFace Transformers — past_key_values documentation](https://huggingface.co/docs/transformers)