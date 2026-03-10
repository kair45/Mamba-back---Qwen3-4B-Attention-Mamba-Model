#!/usr/bin/env python3
"""
Run All Experiments: Comprehensive comparison of hybrid architectures.

This script orchestrates the full experiment pipeline:
1. Convert Qwen3-4B to Mamba hybrid
2. Convert Qwen3-4B to GatedDeltaNet hybrid
3. Train both variants with distillation
4. Evaluate all three models (baseline + 2 hybrids)
5. Generate comparison tables and plots

Usage:
    # Full experiment (requires ~2x A100-80GB for local distillation)
    python scripts/run_all_experiments.py --mode full

    # Quick test with dummy data
    python scripts/run_all_experiments.py --mode test --max_steps 50

    # API distillation mode (single 4090/3090)
    python scripts/run_all_experiments.py --mode api --api_key YOUR_KEY

    # Evaluation only (models already trained)
    python scripts/run_all_experiments.py --mode eval_only \
        --mamba_model_path ./checkpoints/qwen3-4b-mamba/final \
        --gdn_model_path ./checkpoints/qwen3-4b-gdn/final

    # MoE expansion (after training)
    python scripts/run_all_experiments.py --mode moe \
        --mamba_model_path ./checkpoints/qwen3-4b-mamba/final
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.helpers import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run All Experiments")
    parser.add_argument("--mode", type=str, default="test",
                        choices=["full", "test", "api", "eval_only", "moe"],
                        help="Experiment mode")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--output_base", type=str, default="./experiments")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_model", type=str, default="qwen-max")
    # Pre-trained model paths (for eval_only / moe modes)
    parser.add_argument("--mamba_model_path", type=str, default=None)
    parser.add_argument("--gdn_model_path", type=str, default=None)
    # MoE params
    parser.add_argument("--moe_num_experts", type=int, default=8)
    parser.add_argument("--moe_top_k", type=int, default=2)

    return parser.parse_args()


def run_command(cmd: str, description: str, env=None):
    """Run a shell command and log output."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  {description}")
    logger.info(f"  Command: {cmd}")
    logger.info(f"{'='*60}")

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    start_time = time.time()
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, env=merged_env,
    )
    elapsed = time.time() - start_time

    if result.stdout:
        for line in result.stdout.strip().split('\n')[-20:]:
            logger.info(f"  {line}")
    if result.returncode != 0:
        logger.error(f"  Command failed (exit {result.returncode})")
        if result.stderr:
            for line in result.stderr.strip().split('\n')[-10:]:
                logger.error(f"  {line}")

    logger.info(f"  Elapsed: {elapsed:.1f}s")
    return result.returncode == 0


def run_test_mode(args):
    """Quick test with dummy data."""
    max_steps = args.max_steps or 50
    output = args.output_base

    experiments = [
        {
            "name": "mamba",
            "convert_cmd": (
                f"python scripts/convert_model.py "
                f"--base_model {args.base_model} "
                f"--linear_type mamba "
                f"--output_dir {output}/qwen3-4b-mamba-init"
            ),
            "train_cmd": (
                f"python scripts/train_distill.py "
                f"--linear_type mamba "
                f"--base_model {args.base_model} "
                f"--output_dir {output}/qwen3-4b-mamba "
                f"--max_steps {max_steps} --dummy "
                f"--max_seq_length 512 --batch_size 2 --grad_accum 4"
            ),
        },
        {
            "name": "gated_deltanet",
            "convert_cmd": (
                f"python scripts/convert_model.py "
                f"--base_model {args.base_model} "
                f"--linear_type gated_deltanet "
                f"--output_dir {output}/qwen3-4b-gdn-init"
            ),
            "train_cmd": (
                f"python scripts/train_distill.py "
                f"--linear_type gated_deltanet "
                f"--base_model {args.base_model} "
                f"--output_dir {output}/qwen3-4b-gdn "
                f"--max_steps {max_steps} --dummy "
                f"--max_seq_length 512 --batch_size 2 --grad_accum 4"
            ),
        },
    ]

    results = {}
    for exp in experiments:
        logger.info(f"\n{'#'*60}")
        logger.info(f"  Experiment: {exp['name']}")
        logger.info(f"{'#'*60}")

        # Convert
        ok = run_command(exp["convert_cmd"], f"Converting to {exp['name']}")
        results[f"{exp['name']}_convert"] = ok

        # Train
        if ok:
            ok = run_command(exp["train_cmd"], f"Training {exp['name']}")
            results[f"{exp['name']}_train"] = ok

    _save_results(results, output)


def run_full_mode(args):
    """Full training with local teacher."""
    max_steps = args.max_steps or 10000
    output = args.output_base

    for linear_type in ["mamba", "gated_deltanet"]:
        tag = "gdn" if linear_type == "gated_deltanet" else linear_type

        # Convert
        run_command(
            f"python scripts/convert_model.py "
            f"--base_model {args.base_model} "
            f"--linear_type {linear_type} "
            f"--output_dir {output}/qwen3-4b-{tag}-init",
            f"Converting to {linear_type}",
        )

        # Train Phase 1 (linear layers only)
        run_command(
            f"python scripts/train_distill.py "
            f"--linear_type {linear_type} "
            f"--base_model {args.base_model} "
            f"--output_dir {output}/qwen3-4b-{tag} "
            f"--max_steps {max_steps} "
            f"--phase1_steps {max_steps // 5} "
            f"--max_seq_length {args.max_seq_length} "
            f"--batch_size 2 --grad_accum 16",
            f"Training {linear_type} (local teacher)",
        )


def run_api_mode(args):
    """API distillation mode for single GPU."""
    max_steps = args.max_steps or 5000
    output = args.output_base

    env = {}
    if args.api_key:
        env["DASHSCOPE_API_KEY"] = args.api_key

    for linear_type in ["mamba", "gated_deltanet"]:
        tag = "gdn" if linear_type == "gated_deltanet" else linear_type

        run_command(
            f"python scripts/train_distill.py "
            f"--linear_type {linear_type} "
            f"--base_model {args.base_model} "
            f"--output_dir {output}/qwen3-4b-{tag}-api "
            f"--max_steps {max_steps} "
            f"--use_api --api_model {args.api_model} "
            f"--max_seq_length {args.max_seq_length} "
            f"--batch_size 2 --grad_accum 8",
            f"API distillation for {linear_type}",
            env=env,
        )


def run_moe_expansion(args):
    """Convert trained hybrid models to MoE variants."""
    import torch
    from src.models.hybrid_model import QwenHybridForCausalLM
    from src.models.moe_expansion import expand_mlp_to_moe

    for model_path, name in [
        (args.mamba_model_path, "mamba-moe"),
        (args.gdn_model_path, "gdn-moe"),
    ]:
        if model_path is None or not os.path.exists(model_path):
            logger.info(f"Skipping MoE for {name}: path not provided or doesn't exist")
            continue

        logger.info(f"\nExpanding {model_path} to MoE ({name})...")

        model = QwenHybridForCausalLM.from_pretrained_hybrid(model_path)

        model = expand_mlp_to_moe(
            model, num_experts=args.moe_num_experts,
            top_k=args.moe_top_k, shared_expert=True,
        )

        save_path = os.path.join(args.output_base, f"qwen3-4b-{name}")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        logger.info(f"MoE model saved to {save_path}")


def _save_results(results, output_dir):
    """Save experiment results."""
    os.makedirs(output_dir, exist_ok=True)
    results["timestamp"] = datetime.now().isoformat()

    path = os.path.join(output_dir, "experiment_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {path}")


def main():
    args = parse_args()
    setup_logging(log_level="INFO")

    logger.info(f"Experiment mode: {args.mode}")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Output: {args.output_base}")

    os.makedirs(args.output_base, exist_ok=True)

    if args.mode == "test":
        run_test_mode(args)
    elif args.mode == "full":
        run_full_mode(args)
    elif args.mode == "api":
        run_api_mode(args)
    elif args.mode == "moe":
        run_moe_expansion(args)
    elif args.mode == "eval_only":
        logger.info("Evaluation-only mode. Use scripts/evaluate.py directly.")
    else:
        logger.error(f"Unknown mode: {args.mode}")

    logger.info("\nAll experiments completed!")


if __name__ == "__main__":
    main()
