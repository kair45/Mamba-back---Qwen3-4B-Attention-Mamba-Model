"""
Speed Benchmark: Inference Throughput (tokens/sec)

Measures:
  - Prefill speed (prompt processing): tokens/sec during prompt encoding
  - Decode speed (generation): tokens/sec during autoregressive generation
  - Time to First Token (TTFT): latency before first generated token

Compares Qwen3-4B (Transformer) vs Qwen3-4B-Mamba (Hybrid).

Usage:
    python deployment/benchmark_speed.py \
        --transformer /path/to/Qwen3-4B \
        --mamba ./checkpoints/qwen3-4b-mamba-phase2/final \
        --output ./deployment/results/speed.json
"""

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List

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


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def warmup(model, tokenizer, device: str, num_warmup: int = 3):
    """Run a few warmup iterations to stabilize timing."""
    dummy = tokenizer("Hello world", return_tensors="pt").to(device)
    for _ in range(num_warmup):
        with torch.no_grad():
            model.generate(**dummy, max_new_tokens=8, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)


def benchmark_speed(
    model_path: str,
    context_lengths: List[int],
    num_new_tokens: int = 128,
    device: str = "cpu",
    trust_remote_code: bool = True,
    label: str = "Model",
    num_runs: int = 3,
    base_model_path: str = None,
) -> Dict:
    """
    Measure prefill speed, decode speed, and TTFT for a model.

    Args:
        num_runs: Number of runs to average (for stable timing)
    """
    from transformers import AutoTokenizer

    print(f"\n{'='*60}")
    print(f"  Speed Benchmark: {label}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    dtype = torch.float16 if device != "cpu" else torch.float32
    model = _load_causal_lm(model_path, dtype, trust_remote_code, base_model_path=base_model_path).to(device)
    model.eval()

    print("  Warming up...", end=" ", flush=True)
    warmup(model, tokenizer, device)
    print("done")

    results = {
        "label": label,
        "model_path": model_path,
        "device": device,
        "context_lengths": context_lengths,
        "prefill_tokens_per_sec": [],
        "decode_tokens_per_sec": [],
        "ttft_ms": [],
        "total_time_s": [],
    }

    # ~10 tokens per sentence, need enough for the longest context
    max_ctx = max(context_lengths)
    num_repeats = max(500, max_ctx // 5)
    filler = "The quick brown fox jumps over the lazy dog. " * num_repeats

    for ctx_len in context_lengths:
        print(f"\n  Context length: {ctx_len} tokens, generating {num_new_tokens} new tokens")

        input_ids = tokenizer.encode(filler, return_tensors="pt")[:, :ctx_len].to(device)
        actual_ctx = input_ids.shape[1]

        prefill_tps_list, decode_tps_list, ttft_list, total_list = [], [], [], []

        for run in range(num_runs):
            clear_memory()
            if device == "cuda":
                torch.cuda.synchronize()

            # --- Prefill timing (single forward pass) ---
            t_prefill_start = time.perf_counter()
            with torch.no_grad():
                outputs_prefill = model(input_ids, use_cache=True)
            if device == "cuda":
                torch.cuda.synchronize()
            t_prefill_end = time.perf_counter()
            prefill_time = t_prefill_end - t_prefill_start
            prefill_tps = actual_ctx / prefill_time if prefill_time > 0 else 0

            # --- Full generation timing ---
            if device == "cuda":
                torch.cuda.synchronize()
            t_gen_start = time.perf_counter()
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=num_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            if device == "cuda":
                torch.cuda.synchronize()
            t_gen_end = time.perf_counter()

            total_time = t_gen_end - t_gen_start
            generated_tokens = output.shape[1] - input_ids.shape[1]
            # TTFT ≈ prefill time (from separate forward pass)
            ttft_ms = prefill_time * 1000
            # Decode speed: generated_tokens / total_generate_time
            # This is end-to-end generation throughput (includes internal prefill),
            # which is the fairest apples-to-apples comparison across architectures.
            decode_tps = generated_tokens / total_time if total_time > 0 else 0

            prefill_tps_list.append(prefill_tps)
            decode_tps_list.append(decode_tps)
            ttft_list.append(ttft_ms)
            total_list.append(total_time)

            del outputs_prefill, output
            clear_memory()

        avg_prefill = sum(prefill_tps_list) / len(prefill_tps_list)
        avg_decode = sum(decode_tps_list) / len(decode_tps_list)
        avg_ttft = sum(ttft_list) / len(ttft_list)
        avg_total = sum(total_list) / len(total_list)

        print(f"    Prefill: {avg_prefill:.1f} tok/s | Decode: {avg_decode:.1f} tok/s | "
              f"TTFT: {avg_ttft:.1f} ms | Total: {avg_total:.2f}s")

        results["prefill_tokens_per_sec"].append(round(avg_prefill, 1))
        results["decode_tokens_per_sec"].append(round(avg_decode, 1))
        results["ttft_ms"].append(round(avg_ttft, 1))
        results["total_time_s"].append(round(avg_total, 3))

    del model
    clear_memory()
    return results


def main():
    parser = argparse.ArgumentParser(description="Speed benchmark: Mamba vs Transformer baselines")
    parser.add_argument("--mamba", type=str, required=True,
                        help="Path to Mamba hybrid model checkpoint")
    parser.add_argument("--baseline_paths", type=str, required=True,
                        help="Comma-separated model paths, e.g. '/path/qwen3,/path/qwen25'")
    parser.add_argument("--baseline_labels", type=str, required=True,
                        help="Comma-separated labels, e.g. 'Qwen3-4B,Qwen2.5-4B'")
    parser.add_argument("--output", type=str, default="./deployment/results/speed.json")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--context_lengths", type=str, default="128,512,1024,2048")
    parser.add_argument("--new_tokens", type=int, default=128)
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of runs to average for stable timing")
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
        result = benchmark_speed(
            path, context_lengths, args.new_tokens, args.device,
            trust_remote_code=True, label=label, num_runs=args.num_runs,
        )
        all_results["baselines"].append(result)

    # Benchmark Mamba hybrid
    m_results = benchmark_speed(
        args.mamba, context_lengths, args.new_tokens, args.device,
        trust_remote_code=True, label="Mamba Hybrid", num_runs=args.num_runs,
        base_model_path=args.base_model,
    )
    all_results["mamba"] = m_results

    # Summary
    print("\n" + "=" * 80)
    print("  SPEED COMPARISON SUMMARY (Decode tokens/sec)")
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
            row += f" | {bl['decode_tokens_per_sec'][i]:>16.1f}"
        row += f" | {m_results['decode_tokens_per_sec'][i]:>14.1f}"
        print(row)
    print("=" * 80)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
