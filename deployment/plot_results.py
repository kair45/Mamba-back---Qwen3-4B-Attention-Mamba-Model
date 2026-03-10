"""
Plot Results: Generate comparison charts from benchmark JSON files.

Produces:
  1. Memory vs Context Length (line chart) — shows KV cache O(n) vs O(1) difference
  2. Decode Speed vs Context Length (bar chart)
  3. Memory reduction % vs Context Length

Usage:
    python deployment/plot_results.py \
        --memory ./deployment/results/memory.json \
        --speed  ./deployment/results/speed.json \
        --output ./deployment/results/
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


MAMBA_COLOR = "#2ecc71"        # Green — always Mamba Hybrid
BASELINE_COLORS = [
    "#e74c3c",  # Red    — first baseline  (e.g. Qwen3-4B)
    "#3498db",  # Blue   — second baseline (e.g. Qwen2.5-4B)
    "#9b59b6",  # Purple
    "#f39c12",  # Orange
    "#1abc9c",  # Teal
]
MARKERS = ["o", "^", "D", "v", "P"]


def get_baseline_style(idx: int) -> dict:
    return {
        "color": BASELINE_COLORS[idx % len(BASELINE_COLORS)],
        "marker": MARKERS[idx % len(MARKERS)],
        "linewidth": 2.5,
        "markersize": 8,
    }


MAMBA_STYLE = {"color": MAMBA_COLOR, "marker": "s", "linewidth": 2.5, "markersize": 8}


def set_style():
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor":   "#16213e",
        "axes.edgecolor":   "#0f3460",
        "axes.labelcolor":  "white",
        "xtick.color":      "white",
        "ytick.color":      "white",
        "text.color":       "white",
        "grid.color":       "#0f3460",
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
        "legend.facecolor": "#16213e",
        "legend.edgecolor": "#0f3460",
        "font.size":        12,
    })


def plot_memory(data: dict, output_dir: str):
    baselines = data["baselines"]
    m = data["mamba"]
    ctx = baselines[0]["context_lengths"] if baselines else m["context_lengths"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Memory Usage: Mamba Hybrid vs Transformer Baselines\n"
        "(KV Cache O(n) vs O(1) SSM Fixed State)",
        fontsize=14, fontweight="bold",
    )

    # --- Left: Memory lines for all models ---
    ax1 = axes[0]
    for idx, bl in enumerate(baselines):
        ax1.plot(ctx, bl["peak_memory_mb"], label=bl["label"], **get_baseline_style(idx))
    ax1.plot(ctx, m["peak_memory_mb"], label=m["label"], **MAMBA_STYLE)
    ax1.set_xlabel("Context Length (tokens)")
    ax1.set_ylabel("Peak Memory Delta (MB)")
    ax1.set_title("Peak Memory vs Context Length")
    ax1.legend(fontsize=9)
    ax1.grid(True)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # --- Right: Grouped bar — Mamba reduction vs each baseline ---
    ax2 = axes[1]
    n_bl = max(len(baselines), 1)
    x = np.arange(len(ctx))
    width = 0.8 / n_bl
    for idx, bl in enumerate(baselines):
        reductions = [
            (1 - m["peak_memory_mb"][i] / bl["peak_memory_mb"][i]) * 100
            if bl["peak_memory_mb"][i] > 0 else 0
            for i in range(len(ctx))
        ]
        offset = (idx - n_bl / 2 + 0.5) * width
        color = BASELINE_COLORS[idx % len(BASELINE_COLORS)]
        bars = ax2.bar(x + offset, reductions, width,
                       label=f"vs {bl['label']}", color=color, alpha=0.85, edgecolor="white")
        for bar, r in zip(bars, reductions):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{r:.1f}%", ha="center", va="bottom", fontsize=9,
                     fontweight="bold", color=color)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{c:,}" for c in ctx])
    ax2.set_xlabel("Context Length (tokens)")
    ax2.set_ylabel("Memory Reduction (%)")
    ax2.set_title("Mamba Memory Reduction\nvs Each Baseline")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y")
    ax2.axhline(y=0, color="white", linewidth=0.8)

    plt.tight_layout()
    out = os.path.join(output_dir, "memory_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_cache_memory(data: dict, output_dir: str):
    """Plot KV cache memory (persistent state after prefill).

    Args:
        data: Dict with 'baselines' and 'mamba', each containing 'cache_memory_mb'.
              Can be the standalone cache_memory.json or the full memory.json.
    """
    baselines = data["baselines"]
    m = data["mamba"]

    if "cache_memory_mb" not in m or not m["cache_memory_mb"]:
        print("No cache_memory_mb data found, skipping cache memory plot.")
        return

    ctx = m["context_lengths"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "KV Cache Memory: Mamba Hybrid vs Transformer Baselines\n"
        "(Persistent State After Prefill \u2014 Isolates O(n) KV vs O(1) SSM)",
        fontsize=14, fontweight="bold",
    )

    # Left: Cache memory lines
    ax1 = axes[0]
    for idx, bl in enumerate(baselines):
        if "cache_memory_mb" in bl and bl["cache_memory_mb"]:
            ax1.plot(ctx, bl["cache_memory_mb"], label=bl["label"], **get_baseline_style(idx))
    ax1.plot(ctx, m["cache_memory_mb"], label=m["label"], **MAMBA_STYLE)
    ax1.set_xlabel("Context Length (tokens)")
    ax1.set_ylabel("Cache Memory (MB)")
    ax1.set_title("KV Cache Memory vs Context Length")
    ax1.legend(fontsize=9)
    ax1.grid(True)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Right: Cache reduction %
    ax2 = axes[1]
    valid_baselines = [bl for bl in baselines if "cache_memory_mb" in bl and bl["cache_memory_mb"]]
    n_bl = max(len(valid_baselines), 1)
    x = np.arange(len(ctx))
    width = 0.8 / n_bl
    for bar_idx, bl in enumerate(valid_baselines):
        reductions = [
            (1 - m["cache_memory_mb"][i] / bl["cache_memory_mb"][i]) * 100
            if bl["cache_memory_mb"][i] > 0 else 0
            for i in range(len(ctx))
        ]
        offset = (bar_idx - n_bl / 2 + 0.5) * width
        color = BASELINE_COLORS[bar_idx % len(BASELINE_COLORS)]
        bars = ax2.bar(x + offset, reductions, width,
                       label=f"vs {bl['label']}", color=color, alpha=0.85, edgecolor="white")
        for bar, r in zip(bars, reductions):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{r:.1f}%", ha="center", va="bottom", fontsize=9,
                     fontweight="bold", color=color)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{c:,}" for c in ctx])
    ax2.set_xlabel("Context Length (tokens)")
    ax2.set_ylabel("Cache Memory Reduction (%)")
    ax2.set_title("Mamba Cache Reduction\nvs Each Baseline")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y")
    ax2.axhline(y=0, color="white", linewidth=0.8)

    plt.tight_layout()
    out = os.path.join(output_dir, "cache_memory_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_speed(data: dict, output_dir: str):
    baselines = data["baselines"]
    m = data["mamba"]
    ctx = baselines[0]["context_lengths"] if baselines else m["context_lengths"]
    x = np.arange(len(ctx))
    n_models = len(baselines) + 1
    width = 0.8 / n_models

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Inference Speed: Mamba Hybrid vs Transformer Baselines",
                 fontsize=14, fontweight="bold")

    # --- Left: Grouped decode speed bars ---
    ax1 = axes[0]
    for idx, bl in enumerate(baselines):
        offset = (idx - n_models / 2 + 0.5) * width
        bars = ax1.bar(x + offset, bl["decode_tokens_per_sec"], width,
                       label=bl["label"],
                       color=BASELINE_COLORS[idx % len(BASELINE_COLORS)],
                       alpha=0.85, edgecolor="white")
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
    # Mamba bar (last group)
    offset = (len(baselines) - n_models / 2 + 0.5) * width
    bars_m = ax1.bar(x + offset, m["decode_tokens_per_sec"], width,
                     label=m["label"], color=MAMBA_COLOR, alpha=0.85, edgecolor="white")
    for bar in bars_m:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c:,}" for c in ctx])
    ax1.set_xlabel("Context Length (tokens)")
    ax1.set_ylabel("Decode Speed (tokens/sec)")
    ax1.set_title("Decode Throughput (Higher is Better)")
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y")

    # --- Right: Speedup lines (one per baseline) ---
    ax2 = axes[1]
    for idx, bl in enumerate(baselines):
        speedups = [
            m["decode_tokens_per_sec"][i] / bl["decode_tokens_per_sec"][i]
            if bl["decode_tokens_per_sec"][i] > 0 else 1.0
            for i in range(len(ctx))
        ]
        style = get_baseline_style(idx)
        ax2.plot(ctx, speedups, label=f"vs {bl['label']}", **style)
        for c, s in zip(ctx, speedups):
            ax2.annotate(f"{s:.2f}×", xy=(c, s), xytext=(c, s + 0.05),
                         ha="center", fontsize=9, color=style["color"], fontweight="bold")
    ax2.axhline(y=1.0, color="white", linestyle="--", linewidth=1.5, label="Baseline (1.0×)")
    ax2.set_xlabel("Context Length (tokens)")
    ax2.set_ylabel("Speedup (Mamba / Baseline)")
    ax2.set_title("Decode Speedup vs Context Length\n(>1.0 means Mamba is faster)")
    ax2.legend(fontsize=9)
    ax2.grid(True)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    out = os.path.join(output_dir, "speed_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_combined_summary(memory_data: dict, speed_data: dict, output_dir: str):
    """2×2 summary figure suitable for README / slides."""
    set_style()
    baselines_mem = memory_data["baselines"]
    m_mem = memory_data["mamba"]
    baselines_spd = speed_data["baselines"]
    m_spd = speed_data["mamba"]

    ctx_mem = baselines_mem[0]["context_lengths"] if baselines_mem else m_mem["context_lengths"]
    ctx_spd = baselines_spd[0]["context_lengths"] if baselines_spd else m_spd["context_lengths"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Mamba Hybrid (Knowledge Distillation) vs Transformer Baselines\n"
        "Edge Deployment Benchmark — Memory & Speed",
        fontsize=13, fontweight="bold",
    )

    # Top-left: Cache memory vs context (use cache_memory_mb if available, else peak)
    has_cache = "cache_memory_mb" in m_mem and m_mem["cache_memory_mb"]
    mem_key = "cache_memory_mb" if has_cache else "peak_memory_mb"
    mem_label = "KV Cache Memory" if has_cache else "Peak Memory"
    ax = axes[0][0]
    for idx, bl in enumerate(baselines_mem):
        values = bl.get(mem_key, bl["peak_memory_mb"])
        ax.plot(ctx_mem, values, label=bl["label"], **get_baseline_style(idx))
    ax.plot(ctx_mem, m_mem[mem_key], label="Mamba Hybrid", **MAMBA_STYLE)
    ax.set_title(f"{mem_label} vs Context Length")
    ax.set_xlabel("Context (tokens)")
    ax.set_ylabel(f"{mem_label} (MB)")
    ax.legend(fontsize=9)
    ax.grid(True)

    # Top-right: Memory reduction % (grouped bar vs each baseline)
    ax = axes[0][1]
    n_bl = max(len(baselines_mem), 1)
    x_pos = np.arange(len(ctx_mem))
    bar_w = 0.8 / n_bl
    for idx, bl in enumerate(baselines_mem):
        bl_values = bl.get(mem_key, bl["peak_memory_mb"])
        m_values = m_mem[mem_key]
        reductions = [
            (1 - m_values[i] / bl_values[i]) * 100
            if bl_values[i] > 0 else 0
            for i in range(len(ctx_mem))
        ]
        color = BASELINE_COLORS[idx % len(BASELINE_COLORS)]
        offset = (idx - n_bl / 2 + 0.5) * bar_w
        bars = ax.bar(x_pos + offset, reductions, bar_w,
                      label=f"vs {bl['label']}", color=color, alpha=0.85, edgecolor="white")
        for b, r in zip(bars, reductions):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                    f"{r:.1f}%", ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color=color)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(c) for c in ctx_mem])
    ax.set_title(f"Mamba {mem_label} Reduction")
    ax.set_xlabel("Context (tokens)")
    ax.set_ylabel("Reduction (%)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y")

    # Bottom-left: Decode speed grouped bars
    x = np.arange(len(ctx_spd))
    n_models = len(baselines_spd) + 1
    width = 0.8 / n_models
    ax = axes[1][0]
    for idx, bl in enumerate(baselines_spd):
        offset = (idx - n_models / 2 + 0.5) * width
        ax.bar(x + offset, bl["decode_tokens_per_sec"], width,
               label=bl["label"],
               color=BASELINE_COLORS[idx % len(BASELINE_COLORS)],
               alpha=0.85, edgecolor="white")
    offset = (len(baselines_spd) - n_models / 2 + 0.5) * width
    ax.bar(x + offset, m_spd["decode_tokens_per_sec"], width,
           label="Mamba Hybrid", color=MAMBA_COLOR, alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in ctx_spd])
    ax.set_title("Decode Speed (tokens/sec)")
    ax.set_xlabel("Context (tokens)")
    ax.set_ylabel("tokens/sec")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y")

    # Bottom-right: Speedup lines (one per baseline)
    ax = axes[1][1]
    for idx, bl in enumerate(baselines_spd):
        speedups = [
            m_spd["decode_tokens_per_sec"][i] / bl["decode_tokens_per_sec"][i]
            if bl["decode_tokens_per_sec"][i] > 0 else 1.0
            for i in range(len(ctx_spd))
        ]
        style = get_baseline_style(idx)
        ax.plot(ctx_spd, speedups, label=f"vs {bl['label']}", **style)
        for c, s in zip(ctx_spd, speedups):
            ax.annotate(f"{s:.2f}×", xy=(c, s), xytext=(c, s + 0.04),
                        ha="center", fontsize=9, color=style["color"], fontweight="bold")
    ax.axhline(y=1.0, color="white", linestyle="--", linewidth=1.5, alpha=0.6)
    ax.set_title("Decode Speedup vs Baselines")
    ax.set_xlabel("Context (tokens)")
    ax.set_ylabel("Speedup (×)")
    ax.legend(fontsize=9)
    ax.grid(True)

    plt.tight_layout()
    out = os.path.join(output_dir, "summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved summary figure: {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=str, default="./deployment/results/memory.json")
    parser.add_argument("--cache_memory", type=str, default="./deployment/results/cache_memory.json")
    parser.add_argument("--speed", type=str, default="./deployment/results/speed.json")
    parser.add_argument("--output", type=str, default="./deployment/results/")
    args = parser.parse_args()

    set_style()
    os.makedirs(args.output, exist_ok=True)

    memory_data, cache_data, speed_data = None, None, None

    if os.path.exists(args.memory):
        with open(args.memory) as f:
            memory_data = json.load(f)
        plot_memory(memory_data, args.output)
    else:
        print(f"Memory results not found: {args.memory}")

    if os.path.exists(args.cache_memory):
        with open(args.cache_memory) as f:
            cache_data = json.load(f)
        plot_cache_memory(cache_data, args.output)
    else:
        print(f"Cache memory results not found: {args.cache_memory}")

    if os.path.exists(args.speed):
        with open(args.speed) as f:
            speed_data = json.load(f)
        plot_speed(speed_data, args.output)
    else:
        print(f"Speed results not found: {args.speed}")

    if speed_data and (cache_data or memory_data):
        plot_combined_summary(cache_data or memory_data, speed_data, args.output)


if __name__ == "__main__":
    main()
