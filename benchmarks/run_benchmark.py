"""
run_benchmark.py — Compare baseline vs speculative decoding across 100 prompts.

Measures and records:
  - TTFT  (Time To First Token, ms)
  - ITL   (Inter-Token Latency, ms)
  - tok/s (Tokens per second)
  - acceptance_rate (spec-decode only)
  - tokens_per_round (spec-decode only)

Outputs:
  - benchmarks/results/results.csv       — per-prompt raw metrics
  - benchmarks/results/summary.json      — aggregate statistics
  - benchmarks/results/throughput.png    — side-by-side bar chart + scatter

Usage:
    # From project root on the AWS instance:
    python -m benchmarks.run_benchmark
    python -m benchmarks.run_benchmark --config configs/default.yaml --num-prompts 20
    python -m benchmarks.run_benchmark --baseline-only   # skip spec-decode
    python -m benchmarks.run_benchmark --spec-only       # skip baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import torch
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Speculative Decoding Benchmark")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--num-prompts", type=int, default=None,
                   help="Override number of prompts from config")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Override max_new_tokens")
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--spec-only", action="store_true")
    p.add_argument("--K", type=int, default=None, help="Override draft length K")
    p.add_argument("--output-dir", default=None,
                   help="Override results directory")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(args: argparse.Namespace) -> None:
    from src.models import load_models
    from src.baseline import generate_baseline
    from src.engine import SpeculativeDecoder

    config = yaml.safe_load(Path(args.config).read_text())

    # Apply CLI overrides
    bm_cfg = config["benchmark"]
    sd_cfg = config["speculative_decoding"]

    num_prompts = args.num_prompts or bm_cfg.get("num_prompts", 100)
    max_new_tokens = args.max_new_tokens or bm_cfg.get("max_new_tokens", 128)
    warmup = bm_cfg.get("warmup_prompts", 2)
    results_dir = Path(args.output_dir or bm_cfg.get("results_dir", "benchmarks/results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.K:
        sd_cfg["K"] = args.K

    run_baseline = not args.spec_only
    run_spec = not args.baseline_only

    # ── Load models ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("Loading models …")
    pair = load_models(config)
    decoder = SpeculativeDecoder(pair.draft, pair.target, pair.tokenizer, config) if run_spec else None
    print("Models loaded.\n")

    # ── Load prompts ─────────────────────────────────────────────────────────
    prompts_path = Path(bm_cfg.get("prompts_file", "benchmarks/prompts.json"))
    prompts_data = json.loads(prompts_path.read_text())
    prompts = prompts_data[:num_prompts]
    print(f"Benchmarking {len(prompts)} prompts "
          f"(+{warmup} warmup) | max_new_tokens={max_new_tokens}\n")

    # ── GPU warmup ───────────────────────────────────────────────────────────
    warmup_prompts = prompts[:warmup]
    if warmup_prompts:
        print("Warming up …")
        for wp in warmup_prompts:
            if run_baseline:
                generate_baseline(
                    pair.target, pair.tokenizer, wp["prompt"],
                    max_new_tokens=32, seed=0,
                )
            if run_spec and decoder:
                decoder.generate(wp["prompt"], max_new_tokens=32, seed=0)
        print("Warmup done.\n")

    # ── Main benchmark loop ──────────────────────────────────────────────────
    baseline_rows: List[Dict] = []
    spec_rows: List[Dict] = []

    bench_prompts = prompts[warmup:]

    for i, entry in enumerate(bench_prompts):
        pid = entry["id"]
        cat = entry.get("category", "unknown")
        prompt = entry["prompt"]
        print(f"[{i+1:3d}/{len(bench_prompts)}] id={pid} cat={cat[:12]:12s} ", end="", flush=True)

        seed = sd_cfg.get("seed", 42)

        if run_baseline:
            bl = generate_baseline(
                model=pair.target,
                tokenizer=pair.tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=sd_cfg["temperature"],
                top_p=sd_cfg["top_p"],
                top_k=sd_cfg["top_k"],
                seed=seed,
            )
            baseline_rows.append({
                "prompt_id": pid,
                "category": cat,
                "engine": "baseline",
                "prompt_tokens": bl.prompt_tokens,
                "generated_tokens": bl.generated_tokens,
                "ttft_ms": bl.ttft * 1000,
                "itl_ms": bl.itl * 1000,
                "tokens_per_sec": bl.tokens_per_sec,
            })
            print(f"baseline={bl.tokens_per_sec:6.1f}tok/s  ", end="", flush=True)

        if run_spec and decoder:
            sp = decoder.generate(prompt, max_new_tokens=max_new_tokens, seed=seed)
            spec_rows.append({
                "prompt_id": pid,
                "category": cat,
                "engine": "spec_decode",
                "prompt_tokens": sp.prompt_tokens,
                "generated_tokens": sp.generated_tokens,
                "ttft_ms": sp.ttft * 1000,
                "itl_ms": sp.itl * 1000,
                "tokens_per_sec": sp.tokens_per_sec,
                "n_rounds": sp.n_rounds,
                "acceptance_rate": sp.mean_acceptance_rate,
                "tokens_per_round": sp.mean_tokens_per_round,
            })
            speedup = sp.tokens_per_sec / baseline_rows[-1]["tokens_per_sec"] if baseline_rows else 0
            print(f"spec={sp.tokens_per_sec:6.1f}tok/s  "
                  f"accept={sp.mean_acceptance_rate:.1%}  "
                  f"speedup={speedup:.2f}x")
        else:
            print()

    # ── Save CSV ─────────────────────────────────────────────────────────────
    all_rows = baseline_rows + spec_rows
    df = pd.DataFrame(all_rows)
    csv_path = results_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # ── Summary statistics ────────────────────────────────────────────────────
    summary = {}
    if baseline_rows:
        bl_df = df[df["engine"] == "baseline"]
        summary["baseline"] = {
            "mean_tokens_per_sec": bl_df["tokens_per_sec"].mean(),
            "median_tokens_per_sec": bl_df["tokens_per_sec"].median(),
            "mean_ttft_ms": bl_df["ttft_ms"].mean(),
            "mean_itl_ms": bl_df["itl_ms"].mean(),
        }
    if spec_rows:
        sp_df = df[df["engine"] == "spec_decode"]
        summary["spec_decode"] = {
            "mean_tokens_per_sec": sp_df["tokens_per_sec"].mean(),
            "median_tokens_per_sec": sp_df["tokens_per_sec"].median(),
            "mean_ttft_ms": sp_df["ttft_ms"].mean(),
            "mean_itl_ms": sp_df["itl_ms"].mean(),
            "mean_acceptance_rate": sp_df["acceptance_rate"].mean(),
            "mean_tokens_per_round": sp_df["tokens_per_round"].mean(),
        }
    if baseline_rows and spec_rows:
        speedup = summary["spec_decode"]["mean_tokens_per_sec"] / summary["baseline"]["mean_tokens_per_sec"]
        summary["speedup_x"] = speedup
        print(f"\n{'='*60}")
        print(f"SPEEDUP:  {speedup:.2f}x  ({summary['baseline']['mean_tokens_per_sec']:.1f} → {summary['spec_decode']['mean_tokens_per_sec']:.1f} tok/s)")
        print(f"ACCEPT RATE: {summary['spec_decode']['mean_acceptance_rate']:.1%}")
        print(f"{'='*60}\n")

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    try:
        _make_plots(df, results_dir, summary)
    except ImportError:
        print("matplotlib not installed — skipping plots")


def _make_plots(df: "pd.DataFrame", results_dir: Path, summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    BLUE = "#4C9BE8"
    GREEN = "#45D194"
    GOLD = "#F5A623"
    TEXT = "#E8EBF0"
    GRID = "#2a2d36"

    def style_ax(ax):
        ax.set_facecolor("#181b23")
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(axis="y", color=GRID, linewidth=0.6)

    # ── 1. Throughput distribution (box plot per engine) ───────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    engines = df["engine"].unique()
    colors = {e: BLUE if "baseline" in e else GREEN for e in engines}
    data_by_engine = [df[df["engine"] == e]["tokens_per_sec"].values for e in engines]
    bp = ax1.boxplot(data_by_engine, patch_artist=True, widths=0.5, notch=False)
    for patch, eng in zip(bp["boxes"], engines):
        patch.set_facecolor(colors[eng])
        patch.set_alpha(0.85)
    for element in ["whiskers", "caps", "medians", "fliers"]:
        for item in bp[element]:
            item.set_color(TEXT)
    ax1.set_xticks(range(1, len(engines) + 1))
    ax1.set_xticklabels([e.replace("_", "\n") for e in engines])
    ax1.set_ylabel("Tokens / sec")
    ax1.set_title("Throughput Distribution")
    style_ax(ax1)

    # ── 2. Per-prompt speedup ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if "baseline" in df["engine"].values and "spec_decode" in df["engine"].values:
        bl = df[df["engine"] == "baseline"].set_index("prompt_id")["tokens_per_sec"]
        sp = df[df["engine"] == "spec_decode"].set_index("prompt_id")["tokens_per_sec"]
        common = bl.index.intersection(sp.index)
        speedups = (sp.loc[common] / bl.loc[common]).values
        x = np.arange(len(speedups))
        colors_bar = [GREEN if s >= 1.5 else GOLD if s >= 1.0 else "#E85454" for s in speedups]
        ax2.bar(x, speedups, color=colors_bar, alpha=0.85, width=0.8)
        ax2.axhline(1.0, color=TEXT, linewidth=1, linestyle="--", alpha=0.5, label="1x (no speedup)")
        ax2.axhline(summary.get("speedup_x", 1.0), color=GOLD, linewidth=1.5,
                    linestyle="-", alpha=0.9, label=f"Mean: {summary.get('speedup_x', 1.0):.2f}x")
        ax2.set_xlabel("Prompt (sorted)")
        ax2.set_ylabel("Speedup (x)")
        ax2.set_title("Per-Prompt Speedup (Spec / Baseline)")
        ax2.legend(fontsize=8, labelcolor=TEXT, facecolor="#181b23", edgecolor=GRID)
    style_ax(ax2)

    # ── 3. Acceptance rate by category ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if "spec_decode" in df["engine"].values:
        sp_df = df[df["engine"] == "spec_decode"]
        cat_accept = sp_df.groupby("category")["acceptance_rate"].mean().sort_values(ascending=False)
        ax3.barh(cat_accept.index, cat_accept.values, color=GREEN, alpha=0.85)
        ax3.set_xlabel("Mean Acceptance Rate")
        ax3.set_title("Draft Acceptance Rate by Category")
        ax3.set_xlim(0, 1)
    style_ax(ax3)

    # ── 4. TTFT comparison ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for eng, color in zip(engines, [BLUE, GREEN]):
        vals = df[df["engine"] == eng]["ttft_ms"].values
        ax4.plot(sorted(vals), color=color, label=eng.replace("_", "-"), linewidth=1.5, alpha=0.9)
    ax4.set_xlabel("Prompt rank (sorted by TTFT)")
    ax4.set_ylabel("TTFT (ms)")
    ax4.set_title("Time To First Token")
    ax4.legend(fontsize=8, labelcolor=TEXT, facecolor="#181b23", edgecolor=GRID)
    style_ax(ax4)

    # ── Super-title ────────────────────────────────────────────────────────
    speedup_str = f"{summary.get('speedup_x', 0):.2f}x speedup" if "speedup_x" in summary else ""
    fig.suptitle(
        f"Speculative Decoding Benchmark  |  {speedup_str}",
        color=TEXT, fontsize=14, fontweight="bold", y=0.98,
    )

    plot_path = results_dir / "throughput.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
