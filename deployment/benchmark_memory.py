"""
Memory Benchmark: KV Cache Growth vs Context Length

Compares peak memory usage between:
  - Qwen3-4B (Transformer): KV cache grows O(n) with context length
  - Qwen3-4B-Mamba (Hybrid): Mamba layers use O(1) SSM state, only attention layers have KV cache

Usage:
    python deployment/benchmark_memory.py \
        --transformer /path/to/Qwen3-4B \
        --mamba ./checkpoints/qwen3-4b-mamba-phase2/final \
        --output ./deployment/results/memory.json
"""

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List

import psutil
import torch

# Make sure src/ is importable when running from repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_causal_lm(model_path: str, dtype, trust_remote_code: bool, base_model_path: str = None):
    """Load a causal LM, handling the custom qwen_hybrid model type."""
    import json as _json

    cfg_path = os.path.join(model_path, "config.json")
    with open(cfg_path) as f:
        cfg_dict = _json.load(f)

    if cfg_dict.get("model_type") == "qwen_hybrid":
        from src.models.hybrid_model import QwenHybridForCausalLM
        return QwenHybridForCausalLM.from_pretrained_hybrid(
            model_path, torch_dtype=dtype, base_model_override=base_model_path,
        )

    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )


def get_memory_mb() -> float:
    """Get current process RSS memory in MB (CPU RAM, simulates mobile)."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_gpu_memory_mb() -> float:
    """Get current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def get_gpu_peak_memory_mb() -> float:
    """Get peak GPU memory allocated since last reset, in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def clear_memory():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def benchmark_model_memory(
    model_path: str,
    context_lengths: List[int],
    device: str = "cpu",
    trust_remote_code: bool = True,
    num_new_tokens: int = 64,
    label: str = "Model",
    base_model_path: str = None,
) -> Dict:
    """
    Measure peak memory at different context lengths during text generation.

    For each context length:
      1. Load prompt of that length
      2. Generate num_new_tokens new tokens
      3. Record peak memory throughout generation

    Returns dict with context_lengths -> memory_mb mapping.
    """
    from transformers import AutoTokenizer

    print(f"\n{'='*60}")
    print(f"  Benchmarking: {label}")
    print(f"  Model path: {model_path}")
    print(f"  Device: {device}")
    print(f"  Context lengths: {context_lengths}")
    print(f"{'='*60}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    results = {
        "label": label,
        "model_path": model_path,
        "device": device,
        "context_lengths": context_lengths,
        "peak_memory_mb": [],
        "cache_memory_mb": [],
        "memory_per_token_mb": [],
        "generation_time_s": [],
    }

    for ctx_len in context_lengths:
        print(f"\n  Context length: {ctx_len} tokens")
        clear_memory()

        # Load model fresh for each measurement (ensures clean memory state)
        print(f"    Loading model...", end=" ", flush=True)
        dtype = torch.float16 if device != "cpu" else torch.float32
        model = _load_causal_lm(model_path, dtype, trust_remote_code, base_model_path=base_model_path).to(device)
        model.eval()
        print("done")

        base_mem = get_memory_mb() if device == "cpu" else get_gpu_memory_mb()

        # Reset peak memory tracker so we capture only inference peak
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Build prompt of target length
        # Use a repeating sentence to fill up to ctx_len tokens
        # ~10 tokens per sentence, need enough for the longest context
        num_repeats = max(500, ctx_len // 5)
        filler = "The quick brown fox jumps over the lazy dog. " * num_repeats
        input_ids = tokenizer.encode(filler, return_tensors="pt")
        # Trim or pad to exact ctx_len
        if input_ids.shape[1] > ctx_len:
            input_ids = input_ids[:, :ctx_len]
        input_ids = input_ids.to(device)

        actual_ctx = input_ids.shape[1]
        print(f"    Actual context: {actual_ctx} tokens, generating {num_new_tokens} new tokens...")

        # --- Measure KV cache memory (persistent state after prefill) ---
        if device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            mem_model = torch.cuda.memory_allocated() / 1024 / 1024
            with torch.no_grad():
                _prefill = model(input_ids, use_cache=True)
            _cache = _prefill.past_key_values
            del _prefill
            gc.collect()
            torch.cuda.empty_cache()
            cache_mem = torch.cuda.memory_allocated() / 1024 / 1024 - mem_model
            del _cache
            gc.collect()
            torch.cuda.empty_cache()
        else:
            cache_mem = 0.0
        results["cache_memory_mb"].append(round(cache_mem, 2))

        # Reset peak tracker for generate() measurement
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        peak_mem = base_mem
        t0 = time.perf_counter()

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=num_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        elapsed = time.perf_counter() - t0
        peak_mem = get_memory_mb() if device == "cpu" else get_gpu_peak_memory_mb()

        mem_used = peak_mem - base_mem
        mem_per_token = mem_used / actual_ctx if actual_ctx > 0 else 0

        print(f"    Cache: {cache_mem:.1f} MB | Peak: {mem_used:.1f} MB | Per token: {mem_per_token:.3f} MB | Time: {elapsed:.2f}s")

        results["peak_memory_mb"].append(round(mem_used, 2))
        results["memory_per_token_mb"].append(round(mem_per_token, 4))
        results["generation_time_s"].append(round(elapsed, 3))

        # Clean up model to free memory before next run
        del model, output, input_ids
        clear_memory()

    return results


def main():
    parser = argparse.ArgumentParser(description="Memory benchmark: Mamba vs Transformer baselines")
    parser.add_argument("--mamba", type=str, required=True,
                        help="Path to Mamba hybrid model checkpoint")
    parser.add_argument("--baseline_paths", type=str, required=True,
                        help="Comma-separated model paths, e.g. '/path/qwen3,/path/qwen25'")
    parser.add_argument("--baseline_labels", type=str, required=True,
                        help="Comma-separated labels, e.g. 'Qwen3-4B,Qwen2.5-4B'")
    parser.add_argument("--output", type=str, default="./deployment/results/memory.json",
                        help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda"],
                        help="Device to run on")
    parser.add_argument("--context_lengths", type=str,
                        default="128,256,512,1024,2048,4096",
                        help="Comma-separated context lengths to test")
    parser.add_argument("--new_tokens", type=int, default=64,
                        help="Number of new tokens to generate per test")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Path to base Qwen model (needed for hybrid model loading)")
    args = parser.parse_args()

    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    baseline_paths = [p.strip() for p in args.baseline_paths.split(",")]
    baseline_labels = [lb.strip() for lb in args.baseline_labels.split(",")]
    assert len(baseline_paths) == len(baseline_labels), \
        "--baseline_paths and --baseline_labels must have the same number of entries"

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    all_results = {"baselines": [], "mamba": None}

    # Benchmark each baseline
    for path, label in zip(baseline_paths, baseline_labels):
        result = benchmark_model_memory(
            model_path=path,
            context_lengths=context_lengths,
            device=args.device,
            num_new_tokens=args.new_tokens,
            label=label,
        )
        all_results["baselines"].append(result)

    # Benchmark Mamba hybrid
    mamba_results = benchmark_model_memory(
        model_path=args.mamba,
        context_lengths=context_lengths,
        device=args.device,
        trust_remote_code=True,
        num_new_tokens=args.new_tokens,
        label="Mamba Hybrid",
        base_model_path=args.base_model,
    )
    all_results["mamba"] = mamba_results

    # Summary table
    print("\n" + "=" * 80)
    print("  MEMORY COMPARISON SUMMARY (Peak MB Delta)")
    print("=" * 80)
    header = f"{'Context':>10}"
    for lbl in baseline_labels:
        header += f" | {lbl[:16]:>16}"
    header += f" | {'Mamba Hybrid':>14}"
    print(header)
    print("-" * 80)
    for i, ctx in enumerate(context_lengths):
        row = f"{ctx:>10}"
        for bl in all_results["baselines"]:
            row += f" | {bl['peak_memory_mb'][i]:>16.1f}"
        row += f" | {mamba_results['peak_memory_mb'][i]:>14.1f}"
        print(row)
    print("=" * 80)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Save cache memory as separate JSON
    cache_output = os.path.join(
        os.path.dirname(os.path.abspath(args.output)), "cache_memory.json"
    )
    cache_results = {
        "baselines": [
            {
                "label": bl["label"],
                "model_path": bl["model_path"],
                "device": bl["device"],
                "context_lengths": bl["context_lengths"],
                "cache_memory_mb": bl["cache_memory_mb"],
            }
            for bl in all_results["baselines"]
        ],
        "mamba": {
            "label": mamba_results["label"],
            "model_path": mamba_results["model_path"],
            "device": mamba_results["device"],
            "context_lengths": mamba_results["context_lengths"],
            "cache_memory_mb": mamba_results["cache_memory_mb"],
        },
    }
    with open(cache_output, "w") as f:
        json.dump(cache_results, f, indent=2)
    print(f"Cache memory results saved to {cache_output}")


if __name__ == "__main__":
    main()
