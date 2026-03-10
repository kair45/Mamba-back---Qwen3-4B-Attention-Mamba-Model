#!/usr/bin/env python3
"""
Inference Efficiency Benchmark: Hybrid vs Teacher Model.

Measures and compares:
1. Prefill throughput (tokens/sec) at different sequence lengths
2. Decode throughput (tokens/sec) for autoregressive generation
3. Time-to-first-token (TTFT) latency
4. Peak GPU memory usage
5. KV Cache memory footprint

Outputs:
- CSV results file
- Matplotlib comparison charts (throughput, latency, memory vs seq_len)

Usage:
    # Compare hybrid model with teacher
    python scripts/benchmark_inference.py \
        --hybrid_model ./checkpoints/qwen-mamba-hybrid/final \
        --teacher_model Qwen/Qwen3-4B \
        --output_dir ./benchmark_results

    # Quick benchmark (fewer seq lengths, fewer iterations)
    python scripts/benchmark_inference.py \
        --hybrid_model ./checkpoints/qwen-mamba-hybrid/final \
        --teacher_model Qwen/Qwen3-4B \
        --quick

    # Benchmark only the hybrid model
    python scripts/benchmark_inference.py \
        --hybrid_model ./checkpoints/qwen-mamba-hybrid/final \
        --output_dir ./benchmark_results
"""

import argparse
import csv
import gc
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.helpers import setup_logging, set_seed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark Configuration
# ---------------------------------------------------------------------------

FULL_SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
QUICK_SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192]

WARMUP_ITERS = 3
FULL_BENCH_ITERS = 10
QUICK_BENCH_ITERS = 5

DECODE_NEW_TOKENS = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync_cuda():
    """Synchronize CUDA for accurate timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_memory_stats():
    """Reset CUDA memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


def _get_peak_memory_mb() -> float:
    """Get peak GPU memory in MB since last reset."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def _get_allocated_memory_mb() -> float:
    """Get currently allocated GPU memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def load_model(model_path: str, torch_dtype, device: torch.device):
    """Load a model (hybrid or standard HuggingFace)."""
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            raw_config = json.load(f)
        if raw_config.get("model_type") == "qwen_mamba_hybrid":
            from src.models.hybrid_model import QwenMambaHybridConfig, QwenMambaHybridForCausalLM
            logger.info(f"Loading hybrid model from {model_path}")
            model = QwenMambaHybridForCausalLM.from_pretrained_hybrid(
                model_path, torch_dtype=torch_dtype
            )
            model = model.to(device)
            model.eval()
            return model

    logger.info(f"Loading standard model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Benchmark: Prefill (forward pass on full sequence)
# ---------------------------------------------------------------------------

@torch.no_grad()
def benchmark_prefill(
    model: nn.Module,
    seq_lengths: List[int],
    batch_size: int,
    vocab_size: int,
    device: torch.device,
    dtype: torch.dtype,
    num_iters: int = 10,
) -> List[Dict]:
    """
    Benchmark prefill speed: full forward pass on a given sequence length.

    Measures throughput (tok/s), latency (ms), and peak memory (MB).
    """
    results = []

    for seq_len in seq_lengths:
        logger.info(f"  Prefill benchmark: seq_len={seq_len}, batch={batch_size}")

        try:
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

            # Warmup
            for _ in range(WARMUP_ITERS):
                with torch.amp.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
                    _ = model(input_ids=input_ids)
            _sync_cuda()

            # Benchmark
            _reset_memory_stats()
            _sync_cuda()
            start = time.perf_counter()

            for _ in range(num_iters):
                with torch.amp.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
                    _ = model(input_ids=input_ids)

            _sync_cuda()
            elapsed = time.perf_counter() - start
            peak_mem = _get_peak_memory_mb()

            total_tokens = batch_size * seq_len * num_iters
            throughput = total_tokens / elapsed
            latency_ms = (elapsed / num_iters) * 1000

            results.append({
                "seq_len": seq_len,
                "batch_size": batch_size,
                "throughput_tok_s": round(throughput, 1),
                "latency_ms": round(latency_ms, 2),
                "peak_memory_mb": round(peak_mem, 1),
                "num_iters": num_iters,
            })

            logger.info(
                f"    -> {throughput:.0f} tok/s | {latency_ms:.1f} ms | {peak_mem:.0f} MB"
            )

            # Free memory
            del input_ids
            _reset_memory_stats()

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"    -> OOM at seq_len={seq_len}, skipping.")
            results.append({
                "seq_len": seq_len,
                "batch_size": batch_size,
                "throughput_tok_s": 0,
                "latency_ms": -1,
                "peak_memory_mb": -1,
                "num_iters": 0,
                "error": "OOM",
            })
            _reset_memory_stats()

    return results


# ---------------------------------------------------------------------------
# Benchmark: Decode (autoregressive token generation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def benchmark_decode(
    model: nn.Module,
    tokenizer,
    prompt_lengths: List[int],
    new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    num_iters: int = 5,
) -> List[Dict]:
    """
    Benchmark decode speed: autoregressive generation after a prompt.

    Measures decode throughput, time-to-first-token, and total generation time.
    """
    results = []

    for prompt_len in prompt_lengths:
        logger.info(f"  Decode benchmark: prompt_len={prompt_len}, new_tokens={new_tokens}")

        try:
            input_ids = torch.randint(
                0, tokenizer.vocab_size, (1, prompt_len), device=device
            )
            attention_mask = torch.ones_like(input_ids)

            # Warmup
            with torch.amp.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
                _ = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=min(new_tokens, 16),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            _sync_cuda()

            # Benchmark
            _reset_memory_stats()
            ttft_list = []
            total_time_list = []

            for _ in range(num_iters):
                _sync_cuda()

                # Measure TTFT: time for the first forward pass (prefill)
                t0 = time.perf_counter()
                with torch.amp.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _sync_cuda()
                ttft = time.perf_counter() - t0
                ttft_list.append(ttft)

                # Measure full generation
                _sync_cuda()
                t_gen_start = time.perf_counter()
                with torch.amp.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
                    gen_output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                _sync_cuda()
                total_time = time.perf_counter() - t_gen_start
                total_time_list.append(total_time)

            peak_mem = _get_peak_memory_mb()
            actual_new_tokens = gen_output.shape[1] - input_ids.shape[1]

            avg_ttft = sum(ttft_list) / len(ttft_list)
            avg_total = sum(total_time_list) / len(total_time_list)
            decode_time = max(avg_total - avg_ttft, 1e-6)
            decode_throughput = actual_new_tokens / decode_time

            results.append({
                "prompt_len": prompt_len,
                "new_tokens": actual_new_tokens,
                "ttft_ms": round(avg_ttft * 1000, 2),
                "total_time_ms": round(avg_total * 1000, 2),
                "decode_throughput_tok_s": round(decode_throughput, 1),
                "peak_memory_mb": round(peak_mem, 1),
            })

            logger.info(
                f"    -> TTFT: {avg_ttft*1000:.1f} ms | "
                f"Decode: {decode_throughput:.1f} tok/s | "
                f"Total: {avg_total*1000:.0f} ms | "
                f"Mem: {peak_mem:.0f} MB"
            )

            del input_ids, attention_mask
            _reset_memory_stats()

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"    -> OOM at prompt_len={prompt_len}, skipping.")
            results.append({
                "prompt_len": prompt_len,
                "new_tokens": 0,
                "ttft_ms": -1,
                "total_time_ms": -1,
                "decode_throughput_tok_s": 0,
                "peak_memory_mb": -1,
                "error": "OOM",
            })
            _reset_memory_stats()

    return results


# ---------------------------------------------------------------------------
# Benchmark: Memory profiling
# ---------------------------------------------------------------------------

@torch.no_grad()
def benchmark_memory(
    model: nn.Module,
    seq_lengths: List[int],
    batch_size: int,
    vocab_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Dict]:
    """
    Profile peak memory at different sequence lengths.

    This isolates memory measurement from throughput measurement.
    """
    results = []

    # Measure model weight memory
    _reset_memory_stats()
    model_mem = _get_allocated_memory_mb()

    for seq_len in seq_lengths:
        logger.info(f"  Memory profile: seq_len={seq_len}, batch={batch_size}")

        try:
            _reset_memory_stats()
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

            with torch.amp.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
                outputs = model(input_ids=input_ids, use_cache=True)

            _sync_cuda()
            peak_mem = _get_peak_memory_mb()

            # Estimate KV cache size if available
            kv_cache_mb = 0
            if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
                for layer_kv in outputs.past_key_values:
                    if layer_kv is not None:
                        for tensor in layer_kv:
                            if tensor is not None:
                                kv_cache_mb += tensor.nelement() * tensor.element_size() / (1024 * 1024)

            results.append({
                "seq_len": seq_len,
                "batch_size": batch_size,
                "peak_memory_mb": round(peak_mem, 1),
                "model_weight_mb": round(model_mem, 1),
                "kv_cache_mb": round(kv_cache_mb, 1),
                "activation_mb": round(peak_mem - model_mem - kv_cache_mb, 1),
            })

            logger.info(
                f"    -> Peak: {peak_mem:.0f} MB | "
                f"KV Cache: {kv_cache_mb:.0f} MB | "
                f"Activation: {peak_mem - model_mem - kv_cache_mb:.0f} MB"
            )

            del input_ids, outputs
            _reset_memory_stats()

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"    -> OOM at seq_len={seq_len}, skipping.")
            results.append({
                "seq_len": seq_len,
                "batch_size": batch_size,
                "peak_memory_mb": -1,
                "error": "OOM",
            })
            _reset_memory_stats()

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def generate_plots(
    hybrid_results: Dict,
    teacher_results: Optional[Dict],
    output_dir: str,
):
    """Generate comparison plots using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        logger.warning("matplotlib not installed. Skipping plot generation.")
        logger.warning("Install with: pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)
    has_teacher = teacher_results is not None

    # ---- Plot 1: Prefill Throughput vs Sequence Length ----
    fig, ax = plt.subplots(figsize=(10, 6))

    h_prefill = hybrid_results.get("prefill", [])
    h_seq = [r["seq_len"] for r in h_prefill if r.get("throughput_tok_s", 0) > 0]
    h_thr = [r["throughput_tok_s"] for r in h_prefill if r.get("throughput_tok_s", 0) > 0]

    ax.plot(h_seq, h_thr, "o-", linewidth=2, markersize=8, label="Hybrid (Attention+Mamba)")

    if has_teacher:
        t_prefill = teacher_results.get("prefill", [])
        t_seq = [r["seq_len"] for r in t_prefill if r.get("throughput_tok_s", 0) > 0]
        t_thr = [r["throughput_tok_s"] for r in t_prefill if r.get("throughput_tok_s", 0) > 0]
        ax.plot(t_seq, t_thr, "s--", linewidth=2, markersize=8, label="Teacher (Full Attention)")

    ax.set_xlabel("Sequence Length", fontsize=13)
    ax.set_ylabel("Throughput (tokens/sec)", fontsize=13)
    ax.set_title("Prefill Throughput vs Sequence Length", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "prefill_throughput.png"), dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: prefill_throughput.png")

    # ---- Plot 2: Prefill Latency vs Sequence Length ----
    fig, ax = plt.subplots(figsize=(10, 6))

    h_lat = [r["latency_ms"] for r in h_prefill if r.get("latency_ms", -1) > 0]
    ax.plot(h_seq[:len(h_lat)], h_lat, "o-", linewidth=2, markersize=8, label="Hybrid")

    if has_teacher:
        t_lat = [r["latency_ms"] for r in t_prefill if r.get("latency_ms", -1) > 0]
        ax.plot(t_seq[:len(t_lat)], t_lat, "s--", linewidth=2, markersize=8, label="Teacher")

    ax.set_xlabel("Sequence Length", fontsize=13)
    ax.set_ylabel("Latency (ms)", fontsize=13)
    ax.set_title("Prefill Latency vs Sequence Length", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "prefill_latency.png"), dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: prefill_latency.png")

    # ---- Plot 3: Peak Memory vs Sequence Length ----
    fig, ax = plt.subplots(figsize=(10, 6))

    h_mem_data = hybrid_results.get("memory", [])
    h_mem_seq = [r["seq_len"] for r in h_mem_data if r.get("peak_memory_mb", -1) > 0]
    h_mem_val = [r["peak_memory_mb"] / 1024 for r in h_mem_data if r.get("peak_memory_mb", -1) > 0]

    ax.plot(h_mem_seq, h_mem_val, "o-", linewidth=2, markersize=8, label="Hybrid")

    if has_teacher:
        t_mem_data = teacher_results.get("memory", [])
        t_mem_seq = [r["seq_len"] for r in t_mem_data if r.get("peak_memory_mb", -1) > 0]
        t_mem_val = [r["peak_memory_mb"] / 1024 for r in t_mem_data if r.get("peak_memory_mb", -1) > 0]
        ax.plot(t_mem_seq, t_mem_val, "s--", linewidth=2, markersize=8, label="Teacher")

    ax.set_xlabel("Sequence Length", fontsize=13)
    ax.set_ylabel("Peak GPU Memory (GB)", fontsize=13)
    ax.set_title("Peak Memory Usage vs Sequence Length", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "peak_memory.png"), dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: peak_memory.png")

    # ---- Plot 4: KV Cache Comparison (bar chart) ----
    if has_teacher and h_mem_data and t_mem_data:
        fig, ax = plt.subplots(figsize=(10, 6))

        common_seqs = sorted(
            set(r["seq_len"] for r in h_mem_data if "kv_cache_mb" in r)
            & set(r["seq_len"] for r in t_mem_data if "kv_cache_mb" in r)
        )

        if common_seqs:
            h_kv = {r["seq_len"]: r.get("kv_cache_mb", 0) for r in h_mem_data}
            t_kv = {r["seq_len"]: r.get("kv_cache_mb", 0) for r in t_mem_data}

            import numpy as np
            x = np.arange(len(common_seqs))
            width = 0.35

            bars1 = ax.bar(x - width/2, [t_kv.get(s, 0) for s in common_seqs],
                           width, label="Teacher (Full Attention)")
            bars2 = ax.bar(x + width/2, [h_kv.get(s, 0) for s in common_seqs],
                           width, label="Hybrid (9/36 Attention)")

            ax.set_xlabel("Sequence Length", fontsize=13)
            ax.set_ylabel("KV Cache Size (MB)", fontsize=13)
            ax.set_title("KV Cache Memory: Teacher vs Hybrid", fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{s:,}" for s in common_seqs])
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "kv_cache_comparison.png"), dpi=150)
            plt.close(fig)
            logger.info(f"  Saved: kv_cache_comparison.png")

    # ---- Plot 5: TTFT Comparison ----
    h_decode = hybrid_results.get("decode", [])
    if h_decode:
        fig, ax = plt.subplots(figsize=(10, 6))

        h_dec_seq = [r["prompt_len"] for r in h_decode if r.get("ttft_ms", -1) > 0]
        h_ttft = [r["ttft_ms"] for r in h_decode if r.get("ttft_ms", -1) > 0]
        ax.plot(h_dec_seq, h_ttft, "o-", linewidth=2, markersize=8, label="Hybrid")

        if has_teacher:
            t_decode = teacher_results.get("decode", [])
            t_dec_seq = [r["prompt_len"] for r in t_decode if r.get("ttft_ms", -1) > 0]
            t_ttft = [r["ttft_ms"] for r in t_decode if r.get("ttft_ms", -1) > 0]
            ax.plot(t_dec_seq, t_ttft, "s--", linewidth=2, markersize=8, label="Teacher")

        ax.set_xlabel("Prompt Length (tokens)", fontsize=13)
        ax.set_ylabel("Time to First Token (ms)", fontsize=13)
        ax.set_title("TTFT (Time to First Token) vs Prompt Length", fontsize=14)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "ttft_comparison.png"), dpi=150)
        plt.close(fig)
        logger.info(f"  Saved: ttft_comparison.png")

    # ---- Plot 6: Speedup Ratio ----
    if has_teacher and h_prefill and t_prefill:
        fig, ax = plt.subplots(figsize=(10, 6))

        h_thr_dict = {r["seq_len"]: r["throughput_tok_s"] for r in h_prefill if r.get("throughput_tok_s", 0) > 0}
        t_thr_dict = {r["seq_len"]: r["throughput_tok_s"] for r in t_prefill if r.get("throughput_tok_s", 0) > 0}
        common = sorted(set(h_thr_dict.keys()) & set(t_thr_dict.keys()))

        if common:
            speedups = [h_thr_dict[s] / t_thr_dict[s] for s in common]
            ax.bar([f"{s:,}" for s in common], speedups, color="steelblue", edgecolor="black")
            ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1, label="1.0x (no speedup)")
            ax.set_xlabel("Sequence Length", fontsize=13)
            ax.set_ylabel("Speedup (Hybrid / Teacher)", fontsize=13)
            ax.set_title("Prefill Speedup Ratio vs Sequence Length", fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3, axis="y")

            # Add value labels on bars
            for i, (s, v) in enumerate(zip(common, speedups)):
                ax.text(i, v + 0.02, f"{v:.2f}x", ha="center", va="bottom", fontsize=11, fontweight="bold")

            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "speedup_ratio.png"), dpi=150)
            plt.close(fig)
            logger.info(f"  Saved: speedup_ratio.png")


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

def save_results_csv(results: Dict, output_dir: str, model_name: str):
    """Save benchmark results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    prefix = model_name.replace("/", "_").replace("\\", "_")

    for bench_name in ["prefill", "decode", "memory"]:
        data = results.get(bench_name, [])
        if not data:
            continue

        csv_path = os.path.join(output_dir, f"{prefix}_{bench_name}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        logger.info(f"  CSV saved: {csv_path}")


def save_comparison_summary(
    hybrid_results: Dict,
    teacher_results: Optional[Dict],
    output_dir: str,
):
    """Save a human-readable comparison summary."""
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "comparison_summary.txt")

    lines = []
    lines.append("=" * 70)
    lines.append("  INFERENCE BENCHMARK COMPARISON SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    # Prefill comparison
    h_prefill = {r["seq_len"]: r for r in hybrid_results.get("prefill", [])}
    t_prefill = {}
    if teacher_results:
        t_prefill = {r["seq_len"]: r for r in teacher_results.get("prefill", [])}

    lines.append("--- Prefill Throughput (tokens/sec) ---")
    lines.append(f"{'Seq Len':>10} {'Hybrid':>12} {'Teacher':>12} {'Speedup':>10}")
    lines.append("-" * 48)

    for seq_len in sorted(h_prefill.keys()):
        h = h_prefill[seq_len]
        h_thr = h.get("throughput_tok_s", 0)
        if seq_len in t_prefill:
            t_thr = t_prefill[seq_len].get("throughput_tok_s", 0)
            speedup = h_thr / t_thr if t_thr > 0 else 0
            lines.append(f"{seq_len:>10,} {h_thr:>12,.0f} {t_thr:>12,.0f} {speedup:>9.2f}x")
        else:
            lines.append(f"{seq_len:>10,} {h_thr:>12,.0f} {'N/A':>12} {'N/A':>10}")

    lines.append("")

    # Memory comparison
    h_mem = {r["seq_len"]: r for r in hybrid_results.get("memory", [])}
    t_mem = {}
    if teacher_results:
        t_mem = {r["seq_len"]: r for r in teacher_results.get("memory", [])}

    if h_mem:
        lines.append("--- Peak Memory (MB) ---")
        lines.append(f"{'Seq Len':>10} {'Hybrid':>12} {'Teacher':>12} {'Saving':>10}")
        lines.append("-" * 48)

        for seq_len in sorted(h_mem.keys()):
            h = h_mem[seq_len]
            h_m = h.get("peak_memory_mb", -1)
            if seq_len in t_mem:
                t_m = t_mem[seq_len].get("peak_memory_mb", -1)
                if h_m > 0 and t_m > 0:
                    saving = (1 - h_m / t_m) * 100
                    lines.append(f"{seq_len:>10,} {h_m:>11,.0f} {t_m:>11,.0f} {saving:>9.1f}%")
                else:
                    lines.append(f"{seq_len:>10,} {'OOM' if h_m<0 else f'{h_m:,.0f}':>12} {'OOM' if t_m<0 else f'{t_m:,.0f}':>12} {'N/A':>10}")
            else:
                lines.append(f"{seq_len:>10,} {h_m:>11,.0f} {'N/A':>12} {'N/A':>10}")

        lines.append("")

    # KV Cache comparison
    if h_mem and t_mem:
        lines.append("--- KV Cache Size (MB) ---")
        lines.append(f"{'Seq Len':>10} {'Hybrid':>12} {'Teacher':>12} {'Reduction':>10}")
        lines.append("-" * 48)
        for seq_len in sorted(set(h_mem.keys()) & set(t_mem.keys())):
            h_kv = h_mem[seq_len].get("kv_cache_mb", 0)
            t_kv = t_mem[seq_len].get("kv_cache_mb", 0)
            if t_kv > 0:
                reduction = (1 - h_kv / t_kv) * 100
                lines.append(f"{seq_len:>10,} {h_kv:>11,.0f} {t_kv:>11,.0f} {reduction:>9.1f}%")

    lines.append("")
    lines.append("=" * 70)

    summary = "\n".join(lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    # Also print to console
    print(summary)
    logger.info(f"  Summary saved: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark inference efficiency: Hybrid vs Teacher"
    )
    parser.add_argument("--hybrid_model", type=str, required=True, help="Path to hybrid model")
    parser.add_argument("--teacher_model", type=str, default=None, help="Path/name of teacher model")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--decode_tokens", type=int, default=DECODE_NEW_TOKENS)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer seq lengths & iters)")
    parser.add_argument("--skip_decode", action="store_true", help="Skip decode benchmark")
    parser.add_argument("--skip_memory", action="store_true", help="Skip memory profiling")
    parser.add_argument("--skip_plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def run_all_benchmarks(
    model: nn.Module,
    tokenizer,
    model_name: str,
    seq_lengths: List[int],
    args,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict:
    """Run all benchmark suites on a single model."""
    results = {"model_name": model_name}

    # Prefill benchmark
    logger.info(f"\n[{model_name}] Prefill Benchmark")
    num_iters = QUICK_BENCH_ITERS if args.quick else FULL_BENCH_ITERS
    results["prefill"] = benchmark_prefill(
        model=model,
        seq_lengths=seq_lengths,
        batch_size=args.batch_size,
        vocab_size=tokenizer.vocab_size,
        device=device,
        dtype=dtype,
        num_iters=num_iters,
    )

    # Decode benchmark
    if not args.skip_decode:
        logger.info(f"\n[{model_name}] Decode Benchmark")
        decode_prompt_lengths = [s for s in seq_lengths if s <= 8192]
        results["decode"] = benchmark_decode(
            model=model,
            tokenizer=tokenizer,
            prompt_lengths=decode_prompt_lengths,
            new_tokens=args.decode_tokens,
            device=device,
            dtype=dtype,
            num_iters=max(num_iters // 2, 2),
        )

    # Memory profiling
    if not args.skip_memory:
        logger.info(f"\n[{model_name}] Memory Profile")
        results["memory"] = benchmark_memory(
            model=model,
            seq_lengths=seq_lengths,
            batch_size=args.batch_size,
            vocab_size=tokenizer.vocab_size,
            device=device,
            dtype=dtype,
        )

    return results


def main():
    args = parse_args()

    setup_logging(log_level="INFO")
    set_seed(args.seed)

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        logger.warning("No CUDA GPU detected. Benchmarks will not be meaningful on CPU.")

    seq_lengths = QUICK_SEQ_LENGTHS if args.quick else FULL_SEQ_LENGTHS

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model or args.hybrid_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Benchmark Hybrid Model ---
    logger.info("=" * 60)
    logger.info("  Benchmarking Hybrid Model")
    logger.info("=" * 60)

    hybrid_model = load_model(args.hybrid_model, torch_dtype, device)
    hybrid_results = run_all_benchmarks(
        model=hybrid_model,
        tokenizer=tokenizer,
        model_name="hybrid",
        seq_lengths=seq_lengths,
        args=args,
        device=device,
        dtype=torch_dtype,
    )
    save_results_csv(hybrid_results, args.output_dir, "hybrid")

    # Free hybrid model memory before loading teacher
    del hybrid_model
    _reset_memory_stats()

    # --- Benchmark Teacher Model ---
    teacher_results = None
    if args.teacher_model:
        logger.info("\n" + "=" * 60)
        logger.info("  Benchmarking Teacher Model")
        logger.info("=" * 60)

        teacher_model = load_model(args.teacher_model, torch_dtype, device)
        teacher_results = run_all_benchmarks(
            model=teacher_model,
            tokenizer=tokenizer,
            model_name="teacher",
            seq_lengths=seq_lengths,
            args=args,
            device=device,
            dtype=torch_dtype,
        )
        save_results_csv(teacher_results, args.output_dir, "teacher")

        del teacher_model
        _reset_memory_stats()

    # --- Generate Comparison ---
    logger.info("\n" + "=" * 60)
    logger.info("  Generating Results")
    logger.info("=" * 60)

    save_comparison_summary(hybrid_results, teacher_results, args.output_dir)

    if not args.skip_plots:
        logger.info("\nGenerating plots...")
        generate_plots(hybrid_results, teacher_results, args.output_dir)

    # Save raw JSON
    all_data = {"hybrid": hybrid_results}
    if teacher_results:
        all_data["teacher"] = teacher_results
    json_path = os.path.join(args.output_dir, "benchmark_raw.json")
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2)
    logger.info(f"  Raw data saved: {json_path}")

    logger.info("\nBenchmark complete!")
    logger.info(f"Results directory: {args.output_dir}")


if __name__ == "__main__":
    main()
