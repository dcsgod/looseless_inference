-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Speculative Decoding from Scratch
-- MAGIC ### Runs on Databricks Free Edition Serverless GPU
-- MAGIC
-- MAGIC **Select "Serverless GPU" in the compute dropdown above before running.**
-- MAGIC
-- MAGIC This notebook walks through all 6 phases in order:
-- MAGIC 1. Install dependencies
-- MAGIC 2. Load models
-- MAGIC 3. Run baseline generation
-- MAGIC 4. Run speculative decoding
-- MAGIC 5. Run full benchmark
-- MAGIC 6. View results + plots

-- COMMAND ----------

-- MAGIC %md ## Phase 0 — Install Dependencies

-- COMMAND ----------

-- MAGIC %pip install torch transformers accelerate bitsandbytes pyyaml matplotlib scipy tqdm pandas huggingface-hub sentencepiece
-- MAGIC dbutils.library.restartPython()

-- COMMAND ----------

-- MAGIC %md ## Phase 0b — Clone repo & set working directory

-- COMMAND ----------

-- MAGIC %sh
-- MAGIC # If running from a fresh Databricks environment, clone the repo:
-- MAGIC # git clone https://github.com/YOUR_USERNAME/looseless_inference.git /tmp/looseless_inference
-- MAGIC # For now we assume the files are already uploaded to the workspace or DBFS.
-- MAGIC echo "Working directory setup complete"

-- COMMAND ----------

import os, sys

# Set the project root — adjust this path if you cloned the repo elsewhere
PROJECT_ROOT = "/Workspace/Users/YOUR_EMAIL/looseless_inference"
# Or if using DBFS:
# PROJECT_ROOT = "/dbfs/FileStore/looseless_inference"

sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
print("Project root:", PROJECT_ROOT)
print("Python path OK:", "src" in str(sys.path))

-- COMMAND ----------

-- MAGIC %md ## Phase 0c — HuggingFace Login

-- COMMAND ----------

# Option A: paste your token directly (don't commit this cell!)
# import os
# os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN_HERE"

# Option B: use Databricks Secrets (recommended)
# hf_token = dbutils.secrets.get(scope="hf", key="token")
# os.environ["HF_TOKEN"] = hf_token

# Then login:
from huggingface_hub import login
# login(token=os.environ["HF_TOKEN"])
print("Set HF_TOKEN above and uncomment login() before proceeding.")

-- COMMAND ----------

-- MAGIC %md ## Phase 1 — Load Models (4-bit quantization for T4/16GB)

-- COMMAND ----------

import yaml, torch
from pathlib import Path

config = yaml.safe_load(Path("configs/default.yaml").read_text())
print("Config loaded:")
print(f"  Draft  : {config['models']['draft']}")
print(f"  Target : {config['models']['target']}")
print(f"  4-bit  : {config['models']['load_in_4bit']}")
print(f"  K      : {config['speculative_decoding']['K']}")
print(f"  GPU    : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (!)'}")
print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")

-- COMMAND ----------

from src.models import load_models

pair = load_models(config)
print(pair)

-- COMMAND ----------

-- MAGIC %md ## Phase 2 — Baseline Generation

-- COMMAND ----------

from src.baseline import generate_baseline

test_prompts = [
    "Explain the difference between a transformer and an RNN.",
    "Write a Python function to reverse a linked list.",
    "What is the central limit theorem?",
]

sd_cfg = config["speculative_decoding"]
baseline_results = []

for prompt in test_prompts:
    result = generate_baseline(
        model=pair.target,
        tokenizer=pair.tokenizer,
        prompt=prompt,
        max_new_tokens=128,
        temperature=sd_cfg["temperature"],
        top_p=sd_cfg["top_p"],
        top_k=sd_cfg["top_k"],
        seed=42,
    )
    baseline_results.append(result)
    print(f"\n{'─'*60}")
    print(f"Prompt  : {prompt[:60]}")
    print(f"Metrics : {result.summary()}")
    print(f"Output  : {result.output_text[:200]}…")

-- COMMAND ----------

-- MAGIC %md ## Phase 3 — Speculative Decoding

-- COMMAND ----------

from src.engine import SpeculativeDecoder

decoder = SpeculativeDecoder(
    draft_model=pair.draft,
    target_model=pair.target,
    tokenizer=pair.tokenizer,
    config=config,
)

spec_results = []

for prompt in test_prompts:
    result = decoder.generate(prompt, max_new_tokens=128, seed=42)
    spec_results.append(result)
    print(f"\n{'─'*60}")
    print(f"Prompt  : {prompt[:60]}")
    print(f"Metrics : {result.summary()}")
    print(f"Output  : {result.output_text[:200]}…")

-- COMMAND ----------

-- MAGIC %md ## Phase 4 — Quick Comparison

-- COMMAND ----------

import pandas as pd

rows = []
for bl, sp, prompt in zip(baseline_results, spec_results, test_prompts):
    speedup = sp.tokens_per_sec / bl.tokens_per_sec if bl.tokens_per_sec > 0 else 0
    rows.append({
        "prompt": prompt[:50] + "…",
        "baseline_tok_s": round(bl.tokens_per_sec, 1),
        "spec_tok_s": round(sp.tokens_per_sec, 1),
        "speedup": round(speedup, 2),
        "acceptance_rate": f"{sp.mean_acceptance_rate:.1%}",
        "tokens_per_round": round(sp.mean_tokens_per_round, 2),
    })

df = pd.DataFrame(rows)
display(df)

-- COMMAND ----------

-- MAGIC %md ## Phase 5 — Run Pure-Math Unit Tests (no GPU needed)

-- COMMAND ----------

-- MAGIC %sh
-- MAGIC cd /Workspace/Users/YOUR_EMAIL/looseless_inference && \
-- MAGIC   python -m pytest tests/test_rejection_sampling.py -v 2>&1 | tail -30

-- COMMAND ----------

-- MAGIC %md ## Phase 6 — Full Benchmark (100 prompts)
-- MAGIC ⚠️ This takes ~20–40 minutes depending on GPU quota. Run when you have time.

-- COMMAND ----------

# Run from terminal or uncomment:
# import subprocess
# result = subprocess.run(
#     ["python", "-m", "benchmarks.run_benchmark", "--num-prompts", "20"],
#     capture_output=True, text=True, cwd=PROJECT_ROOT
# )
# print(result.stdout)
# print(result.stderr)
print("Uncomment the block above to run the full benchmark.")
print("For a quick test, use --num-prompts 10")

-- COMMAND ----------

-- MAGIC %md ## Phase 7 — View Results

-- COMMAND ----------

from pathlib import Path
import pandas as pd

results_path = Path("benchmarks/results/results.csv")
if results_path.exists():
    df = pd.read_csv(results_path)
    display(df)
    
    bl = df[df["engine"] == "baseline"]["tokens_per_sec"]
    sp = df[df["engine"] == "spec_decode"]["tokens_per_sec"]
    if len(bl) > 0 and len(sp) > 0:
        speedup = sp.mean() / bl.mean()
        print(f"\n{'='*50}")
        print(f"Baseline:    {bl.mean():.1f} tok/s")
        print(f"Spec-Decode: {sp.mean():.1f} tok/s")
        print(f"Speedup:     {speedup:.2f}x")
        print(f"{'='*50}")
else:
    print("No results yet — run Phase 6 first.")

-- COMMAND ----------

-- MAGIC %md ## View Plot

-- COMMAND ----------

from PIL import Image
import matplotlib.pyplot as plt

plot_path = Path("benchmarks/results/throughput.png")
if plot_path.exists():
    img = Image.open(plot_path)
    plt.figure(figsize=(14, 9))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("No plot yet — run Phase 6 first.")
