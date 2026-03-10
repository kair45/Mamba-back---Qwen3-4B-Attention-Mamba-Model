"""
One-shot comparison script: runs memory + speed benchmarks then generates all plots.

Usage:
    python deployment/compare_all.py \
        --transformer /root/autodl-tmp/models/Qwen3-4B \
        --mamba ./checkpoints/qwen3-4b-mamba-phase2/final \
        --device cpu \
        --output ./deployment/results
"""

import argparse
import json
import os
import subprocess
import sys


def run_benchmark(script: str, extra_args: list) -> bool:
    cmd = [sys.executable, script] + extra_args
    print(f"\n>>> Running: {' '.join(cmd)}\n")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(cmd, env=env)
    return result.returncode == 0


def print_summary(results_dir: str):
    """Print a compact text summary from saved JSON files."""
    mem_path = os.path.join(results_dir, "memory.json")
    spd_path = os.path.join(results_dir, "speed.json")

    print("\n" + "=" * 80)
    print("  FINAL COMPARISON SUMMARY")
    print("=" * 80)

    # Cache memory (from separate JSON)
    print_cache_summary(results_dir)

    if os.path.exists(mem_path):
        with open(mem_path) as f:
            data = json.load(f)
        baselines = data["baselines"]
        m = data["mamba"]
        print("\n[Peak Memory] Full inference peak (MB)")
        header = f"  {'Context':>8}"
        for bl in baselines:
            header += f"  {bl['label'][:16]:>16}"
        header += f"  {'Mamba Hybrid':>14}"
        print(header)
        print(f"  {'-'*76}")
        for i, ctx in enumerate(m["context_lengths"]):
            row = f"  {ctx:>8,}"
            for bl in baselines:
                row += f"  {bl['peak_memory_mb'][i]:>16.1f}"
            m_mem = m["peak_memory_mb"][i]
            row += f"  {m_mem:>14.1f}"
            print(row)

    if os.path.exists(spd_path):
        with open(spd_path) as f:
            data = json.load(f)
        baselines = data["baselines"]
        m = data["mamba"]
        print("\n[Speed] Decode throughput (tokens/sec)")
        header = f"  {'Context':>8}"
        for bl in baselines:
            header += f"  {bl['label'][:16]:>16}"
        header += f"  {'Mamba Hybrid':>14}"
        print(header)
        print(f"  {'-'*76}")
        for i, ctx in enumerate(m["context_lengths"]):
            row = f"  {ctx:>8,}"
            for bl in baselines:
                row += f"  {bl['decode_tokens_per_sec'][i]:>16.1f}"
            row += f"  {m['decode_tokens_per_sec'][i]:>14.1f}"
            print(row)

    print("\n" + "=" * 80)
    print(f"  Plots saved in: {results_dir}/")
    print("  - memory_comparison.png")
    print("  - cache_memory_comparison.png")
    print("  - speed_comparison.png")
    print("  - summary.png  ← use this for README / slides")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Full benchmark: Mamba Hybrid vs Transformer baselines (Qwen3-4B, Qwen2.5-4B, etc.)"
    )
    parser.add_argument("--mamba", type=str, required=True,
                        help="Path to Mamba hybrid model checkpoint")
    parser.add_argument("--qwen3", type=str, default=None,
                        help="Path to Qwen3-4B model (baseline)")
    parser.add_argument("--qwen35", type=str, default=None,
                        help="Path to Qwen3.5-4B model (baseline, latest)")
    parser.add_argument("--extra_baseline_paths", type=str, default=None,
                        help="Extra comma-separated baseline paths")
    parser.add_argument("--extra_baseline_labels", type=str, default=None,
                        help="Extra comma-separated baseline labels")
    parser.add_argument("--output", type=str, default="./deployment/results",
                        help="Output directory for JSON + PNG files")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--context_lengths", type=str, default="128,256,512,1024,2048,4096",
                        help="Context lengths for memory benchmark")
    parser.add_argument("--speed_context_lengths", type=str, default="128,512,1024,2048",
                        help="Context lengths for speed benchmark")
    parser.add_argument("--new_tokens", type=int, default=64)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--skip_memory", action="store_true")
    parser.add_argument("--skip_speed", action="store_true")
    parser.add_argument("--skip_plot", action="store_true")
    args = parser.parse_args()

    # Build baseline lists from individual flags + extra
    baseline_paths, baseline_labels = [], []
    if args.qwen3:
        baseline_paths.append(args.qwen3)
        baseline_labels.append("Qwen3-4B")
    if args.qwen35:
        baseline_paths.append(args.qwen35)
        baseline_labels.append("Qwen3.5-4B")
    if args.extra_baseline_paths:
        extra_paths = [p.strip() for p in args.extra_baseline_paths.split(",")]
        extra_labels = [lb.strip() for lb in (args.extra_baseline_labels or "").split(",")]
        for i, p in enumerate(extra_paths):
            lbl = extra_labels[i] if i < len(extra_labels) and extra_labels[i] else f"Baseline{len(baseline_paths)+1}"
            baseline_paths.append(p)
            baseline_labels.append(lbl)

    if not baseline_paths:
        print("ERROR: Provide at least one baseline via --qwen3, --qwen25, or --extra_baseline_paths")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mem_json = os.path.join(args.output, "memory.json")
    spd_json = os.path.join(args.output, "speed.json")

    bl_paths_str = ",".join(baseline_paths)
    bl_labels_str = ",".join(baseline_labels)
    # The first baseline (Qwen3-4B) serves as the base model for hybrid loading
    base_model_for_hybrid = baseline_paths[0]

    if not args.skip_memory:
        ok = run_benchmark(
            os.path.join(script_dir, "benchmark_memory.py"),
            [
                "--mamba", args.mamba,
                "--baseline_paths", bl_paths_str,
                "--baseline_labels", bl_labels_str,
                "--output", mem_json,
                "--device", args.device,
                "--context_lengths", args.context_lengths,
                "--new_tokens", str(args.new_tokens),
                "--base_model", base_model_for_hybrid,
            ]
        )
        if not ok:
            print("Memory benchmark failed. Continuing...")

    if not args.skip_speed:
        ok = run_benchmark(
            os.path.join(script_dir, "benchmark_speed.py"),
            [
                "--mamba", args.mamba,
                "--baseline_paths", bl_paths_str,
                "--baseline_labels", bl_labels_str,
                "--output", spd_json,
                "--device", args.device,
                "--context_lengths", args.speed_context_lengths,
                "--new_tokens", str(args.new_tokens),
                "--num_runs", str(args.num_runs),
                "--base_model", base_model_for_hybrid,
            ]
        )
        if not ok:
            print("Speed benchmark failed. Continuing...")

    cache_json = os.path.join(args.output, "cache_memory.json")

    if not args.skip_plot:
        run_benchmark(
            os.path.join(script_dir, "plot_results.py"),
            [
                "--memory", mem_json,
                "--cache_memory", cache_json,
                "--speed", spd_json,
                "--output", args.output,
            ]
        )

    print_summary(args.output)


def print_cache_summary(results_dir: str):
    """Print cache memory summary from cache_memory.json."""
    cache_path = os.path.join(results_dir, "cache_memory.json")
    if not os.path.exists(cache_path):
        return

    with open(cache_path) as f:
        data = json.load(f)
    baselines = data["baselines"]
    m = data["mamba"]

    print("\n[Cache Memory] KV cache after prefill (MB)")
    header = f"  {'Context':>8}"
    for bl in baselines:
        header += f"  {bl['label'][:16]:>16}"
    header += f"  {'Mamba Hybrid':>14}  {'Reduction':>10}"
    print(header)
    print(f"  {'-'*76}")
    for i, ctx in enumerate(m["context_lengths"]):
        row = f"  {ctx:>8,}"
        for bl in baselines:
            row += f"  {bl['cache_memory_mb'][i]:>16.1f}"
        m_cache = m["cache_memory_mb"][i]
        row += f"  {m_cache:>14.1f}"
        if baselines:
            bl_cache = baselines[0]["cache_memory_mb"][i]
            red = (1 - m_cache / bl_cache) * 100 if bl_cache > 0 else 0
            row += f"  {red:>9.1f}%"
        print(row)


if __name__ == "__main__":
    main()
