#!/usr/bin/env python3
"""
Comprehensive Evaluation Script.

Evaluates and compares multiple model variants:
1. Baseline Qwen3-4B (original teacher)
2. Qwen3-4B-Mamba hybrid
3. Qwen3-4B-GatedDeltaNet hybrid

Metrics:
- Perplexity (PPL) on validation set
- Generation quality (sample outputs)
- Inference speed (tokens/sec, latency)
- Memory usage (peak GPU memory)

Usage:
    # Evaluate a single model
    python scripts/evaluate.py \
        --model_path ./checkpoints/qwen3-4b-mamba/final \
        --model_type mamba

    # Compare all variants
    python scripts/evaluate.py --compare_all \
        --baseline Qwen/Qwen3-4B \
        --mamba_path ./checkpoints/qwen3-4b-mamba/final \
        --gdn_path ./checkpoints/qwen3-4b-gdn/final

    # Quick eval with dummy data
    python scripts/evaluate.py --model_path ./checkpoints/qwen3-4b-mamba/final \
        --model_type mamba --dummy --num_eval_samples 20
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.hybrid_model import QwenHybridForCausalLM
from src.models.architecture_surgery import convert_qwen_to_hybrid, load_tokenizer
from src.training.data import create_dummy_dataset, build_dataloader
from src.utils.helpers import setup_logging, set_seed, get_gpu_memory_info


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Models")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="mamba",
                        choices=["mamba", "gated_deltanet", "baseline"])
    # Compare all mode
    parser.add_argument("--compare_all", action="store_true")
    parser.add_argument("--baseline", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--mamba_path", type=str, default=None)
    parser.add_argument("--gdn_path", type=str, default=None)
    # Eval settings
    parser.add_argument("--num_eval_samples", type=int, default=100)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def evaluate_perplexity(model, dataloader, device, dtype=torch.bfloat16) -> Dict:
    """Evaluate perplexity on a dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("labels", input_ids.clone()).to(device)

            with torch.amp.autocast('cuda', dtype=dtype, enabled=True):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            if loss is not None:
                num_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))

    return {"loss": avg_loss, "perplexity": ppl, "total_tokens": total_tokens}


def evaluate_inference_speed(
    model, tokenizer, device, prompt="Hello, how are you?",
    gen_length=128, num_runs=5, dtype=torch.bfloat16,
) -> Dict:
    """Benchmark inference speed."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Warmup
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype, enabled=True):
        _ = model.generate(input_ids, max_new_tokens=16, do_sample=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype, enabled=True):
            outputs = model.generate(
                input_ids, max_new_tokens=gen_length, do_sample=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        num_new_tokens = outputs.shape[1] - input_ids.shape[1]
        latencies.append(elapsed)

    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

    avg_latency = sum(latencies) / len(latencies)
    tokens_per_sec = gen_length / avg_latency

    return {
        "avg_latency_s": round(avg_latency, 3),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "peak_memory_gb": round(peak_mem, 2),
        "gen_length": gen_length,
        "num_runs": num_runs,
    }


def evaluate_generation_quality(
    model, tokenizer, device, prompts: List[str], max_new_tokens=200,
    dtype=torch.bfloat16,
) -> List[Dict]:
    """Generate text and return samples."""
    model.eval()
    results = []

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype, enabled=True):
            outputs = model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9,
            )

        generated = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        results.append({"prompt": prompt, "generated": generated})

    return results


def count_params(model) -> Dict:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def main():
    args = parse_args()
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prompts for generation quality
    test_prompts = [
        "The transformer architecture revolutionized NLP by",
        "量子计算的基本原理是",
        "In machine learning, the bias-variance tradeoff",
        "请解释什么是知识蒸馏",
    ]

    all_results = {}

    if args.compare_all:
        # Compare all variants
        models_to_eval = []

        if args.baseline:
            models_to_eval.append(("baseline", args.baseline, "baseline"))
        if args.mamba_path:
            models_to_eval.append(("mamba", args.mamba_path, "mamba"))
        if args.gdn_path:
            models_to_eval.append(("gated_deltanet", args.gdn_path, "gated_deltanet"))

        for name, path, mtype in models_to_eval:
            logger.info(f"\n{'='*60}")
            logger.info(f"  Evaluating: {name} ({path})")
            logger.info(f"{'='*60}")

            # Load model
            if mtype == "baseline":
                model = AutoModelForCausalLM.from_pretrained(
                    path, torch_dtype=torch.bfloat16, device_map=device,
                    trust_remote_code=True,
                )
                tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            else:
                model = QwenHybridForCausalLM.from_pretrained_hybrid(
                    path, torch_dtype=torch.bfloat16,
                )
                model = model.to(device)
                tokenizer = load_tokenizer(args.baseline)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            result = _evaluate_single(
                model, tokenizer, device, args, test_prompts, logger,
            )
            result["model_path"] = path
            result["model_type"] = mtype
            result["params"] = count_params(model)
            all_results[name] = result

            del model
            torch.cuda.empty_cache()

    else:
        # Single model evaluation
        if args.model_type == "baseline":
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path or "Qwen/Qwen3-4B",
                torch_dtype=torch.bfloat16, device_map=device,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path or "Qwen/Qwen3-4B", trust_remote_code=True,
            )
        else:
            model = QwenHybridForCausalLM.from_pretrained_hybrid(
                args.model_path, torch_dtype=torch.bfloat16,
            )
            model = model.to(device)
            tokenizer = load_tokenizer("Qwen/Qwen3-4B")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        result = _evaluate_single(model, tokenizer, device, args, test_prompts, logger)
        result["params"] = count_params(model)
        all_results[args.model_type] = result

    # Save results
    output_path = os.path.join(args.output_dir, "eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # Print comparison table
    _print_comparison(all_results, logger)

    logger.info(f"\nResults saved to {output_path}")


def _evaluate_single(model, tokenizer, device, args, test_prompts, logger) -> Dict:
    """Evaluate a single model."""
    result = {}

    # Perplexity
    logger.info("  Computing perplexity...")
    eval_dataset = create_dummy_dataset(
        tokenizer, num_samples=args.num_eval_samples,
        max_seq_length=min(args.max_seq_length, 512),
    )
    eval_loader = build_dataloader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, drop_last=False,
    )
    ppl_result = evaluate_perplexity(model, eval_loader, device)
    result["perplexity"] = ppl_result
    logger.info(f"    PPL: {ppl_result['perplexity']:.2f}")

    # Inference speed
    if torch.cuda.is_available():
        logger.info("  Benchmarking inference speed...")
        speed_result = evaluate_inference_speed(model, tokenizer, device)
        result["inference_speed"] = speed_result
        logger.info(f"    Tokens/s: {speed_result['tokens_per_sec']}")
        logger.info(f"    Peak mem: {speed_result['peak_memory_gb']} GB")

    # Generation quality
    logger.info("  Generating samples...")
    gen_result = evaluate_generation_quality(model, tokenizer, device, test_prompts)
    result["generation_samples"] = gen_result
    for sample in gen_result[:2]:
        logger.info(f"    Prompt: {sample['prompt'][:50]}...")
        logger.info(f"    Output: {sample['generated'][:100]}...")

    return result


def _print_comparison(results: Dict, logger):
    """Print comparison table."""
    if len(results) < 2:
        return

    logger.info("\n" + "=" * 80)
    logger.info("  COMPARISON TABLE")
    logger.info("=" * 80)

    header = f"{'Model':<25} {'Params':>12} {'PPL':>8} {'Tok/s':>8} {'Mem(GB)':>8}"
    logger.info(header)
    logger.info("-" * 80)

    for name, res in results.items():
        params = res.get("params", {}).get("total", 0)
        ppl = res.get("perplexity", {}).get("perplexity", 0)
        tok_s = res.get("inference_speed", {}).get("tokens_per_sec", 0)
        mem = res.get("inference_speed", {}).get("peak_memory_gb", 0)

        logger.info(
            f"{name:<25} {params:>12,} {ppl:>8.2f} {tok_s:>8.1f} {mem:>8.2f}"
        )

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
