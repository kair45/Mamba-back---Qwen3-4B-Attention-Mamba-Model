#!/usr/bin/env python3
"""
Ablation Study: Analyze the impact of different design choices.

Runs multiple configurations and compares results to answer:
1. How many Attention layers should we keep? (1/2, 1/4, 1/6, 1/8)
2. Does two-phase training help vs single-phase?
3. What is the best distillation loss mix (KD vs CE)?

This script automates model creation, short training, and evaluation
for each ablation variant, then generates a summary table.

Usage:
    # Ablation on attention ratio (main experiment)
    python scripts/ablation_study.py \
        --ablation attention_ratio \
        --base_model Qwen/Qwen3-4B \
        --max_steps 2000 \
        --output_dir ./ablation_results

    # Ablation on distillation loss weights
    python scripts/ablation_study.py \
        --ablation loss_weights \
        --base_model Qwen/Qwen3-4B \
        --max_steps 2000 \
        --output_dir ./ablation_results

    # Ablation on two-phase vs single-phase
    python scripts/ablation_study.py \
        --ablation training_phases \
        --base_model Qwen/Qwen3-4B \
        --max_steps 3000 \
        --output_dir ./ablation_results

    # All ablations
    python scripts/ablation_study.py \
        --ablation all \
        --base_model Qwen/Qwen3-4B \
        --max_steps 2000 \
        --output_dir ./ablation_results
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.utils.helpers import setup_logging, set_seed, print_model_summary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

@dataclass
class AblationVariant:
    """A single ablation experiment variant."""
    name: str
    description: str
    attention_interval: int = 4
    attention_layers: Optional[List[int]] = None
    alpha_kd: float = 0.5
    alpha_ce: float = 0.5
    phase1_steps: int = 0
    learning_rate: float = 5e-4
    max_steps: int = 2000


def get_attention_ratio_variants(num_layers: int = 36, max_steps: int = 2000) -> List[AblationVariant]:
    """
    Ablation: What ratio of Attention layers to keep?

    Tests: 1/2, 1/3, 1/4, 1/6, 1/9 of layers as Attention.
    """
    return [
        AblationVariant(
            name="attn_1_of_2",
            description="Keep 1/2 layers as Attention (50%)",
            attention_interval=2,
            max_steps=max_steps,
        ),
        AblationVariant(
            name="attn_1_of_3",
            description="Keep 1/3 layers as Attention (33%)",
            attention_interval=3,
            max_steps=max_steps,
        ),
        AblationVariant(
            name="attn_1_of_4",
            description="Keep 1/4 layers as Attention (25%) [default]",
            attention_interval=4,
            max_steps=max_steps,
        ),
        AblationVariant(
            name="attn_1_of_6",
            description="Keep 1/6 layers as Attention (17%)",
            attention_interval=6,
            max_steps=max_steps,
        ),
        AblationVariant(
            name="attn_1_of_9",
            description="Keep 1/9 layers as Attention (11%)",
            attention_interval=9,
            max_steps=max_steps,
        ),
        AblationVariant(
            name="attn_first_last_only",
            description="Keep only first and last layer as Attention",
            attention_layers=[0, num_layers - 1],
            max_steps=max_steps,
        ),
    ]


def get_loss_weight_variants(max_steps: int = 2000) -> List[AblationVariant]:
    """
    Ablation: What is the best KD vs CE loss ratio?
    """
    return [
        AblationVariant(
            name="loss_kd_only",
            description="Pure KD loss (alpha_kd=1.0, alpha_ce=0.0)",
            alpha_kd=1.0, alpha_ce=0.0,
            max_steps=max_steps,
        ),
        AblationVariant(
            name="loss_ce_only",
            description="Pure CE loss (alpha_kd=0.0, alpha_ce=1.0)",
            alpha_kd=0.0, alpha_ce=1.0,
            max_steps=max_steps,
        ),
        AblationVariant(
            name="loss_kd_0.7_ce_0.3",
            description="KD-heavy mix (alpha_kd=0.7, alpha_ce=0.3)",
            alpha_kd=0.7, alpha_ce=0.3,
            max_steps=max_steps,
        ),
        AblationVariant(
            name="loss_kd_0.5_ce_0.5",
            description="Equal mix (alpha_kd=0.5, alpha_ce=0.5) [default]",
            alpha_kd=0.5, alpha_ce=0.5,
            max_steps=max_steps,
        ),
        AblationVariant(
            name="loss_kd_0.3_ce_0.7",
            description="CE-heavy mix (alpha_kd=0.3, alpha_ce=0.7)",
            alpha_kd=0.3, alpha_ce=0.7,
            max_steps=max_steps,
        ),
    ]


def get_training_phase_variants(max_steps: int = 3000) -> List[AblationVariant]:
    """
    Ablation: Does two-phase training help?
    """
    return [
        AblationVariant(
            name="single_phase",
            description="Single-phase: train all params from start",
            phase1_steps=0,
            max_steps=max_steps,
        ),
        AblationVariant(
            name="two_phase_500",
            description="Two-phase: 500 steps Mamba-only, then full",
            phase1_steps=500,
            max_steps=max_steps,
        ),
        AblationVariant(
            name="two_phase_1000",
            description="Two-phase: 1000 steps Mamba-only, then full",
            phase1_steps=1000,
            max_steps=max_steps,
        ),
        AblationVariant(
            name="two_phase_2000",
            description="Two-phase: 2000 steps Mamba-only, then full",
            phase1_steps=2000,
            max_steps=max_steps,
        ),
    ]


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_single_ablation(
    variant: AblationVariant,
    base_model: str,
    output_dir: str,
    max_seq_length: int,
    batch_size: int,
    dummy: bool,
    dummy_samples: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Dict:
    """
    Run a single ablation variant: create model, train, evaluate.

    Returns a dict with training loss, eval PPL, and timing info.
    """
    from src.models.architecture_surgery import convert_qwen_to_hybrid, load_tokenizer
    from src.training.distillation import DistillationTrainer, DistillationConfig
    from src.training.data import build_dataloader, create_dummy_dataset

    variant_dir = os.path.join(output_dir, variant.name)
    os.makedirs(variant_dir, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Ablation Variant: {variant.name}")
    logger.info(f"  {variant.description}")
    logger.info(f"{'='*60}")

    start_time = time.time()

    # Step 1: Create hybrid model
    logger.info("Creating hybrid model...")
    hybrid_model, teacher_model = convert_qwen_to_hybrid(
        model_name_or_path=base_model,
        attention_interval=variant.attention_interval,
        attention_layers=variant.attention_layers,
        torch_dtype=dtype,
        load_teacher=True,
    )
    hybrid_model = hybrid_model.to(device)
    teacher_model = teacher_model.to(device)

    num_attn = len(hybrid_model.config.attention_layers)
    num_mamba = len(hybrid_model.config.mamba_layers)
    total_params = sum(p.numel() for p in hybrid_model.parameters())

    # Step 2: Prepare data
    tokenizer = load_tokenizer(base_model)
    train_dataset = create_dummy_dataset(tokenizer, num_samples=dummy_samples, max_seq_length=max_seq_length)
    train_dataloader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Step 3: Train
    distill_cfg = DistillationConfig(
        alpha_kd=variant.alpha_kd,
        alpha_ce=variant.alpha_ce,
        learning_rate=variant.learning_rate,
        max_steps=variant.max_steps,
        phase1_steps=variant.phase1_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        logging_steps=50,
        save_steps=999999,  # Don't save intermediate checkpoints
        eval_steps=999999,
        output_dir=variant_dir,
        bf16=(dtype == torch.bfloat16),
        gradient_checkpointing=True,
        warmup_ratio=0.05,
        dataloader_num_workers=0,
    )

    trainer = DistillationTrainer(
        student_model=hybrid_model,
        teacher_model=teacher_model,
        config=distill_cfg,
        train_dataloader=train_dataloader,
        tokenizer=tokenizer,
    )

    logger.info(f"Training for {variant.max_steps} steps...")
    trainer.train()

    training_time = time.time() - start_time

    # Step 4: Quick eval (PPL on dummy data)
    hybrid_model.eval()
    eval_dataset = create_dummy_dataset(tokenizer, num_samples=50, max_seq_length=max_seq_length)
    eval_dataloader = build_dataloader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = hybrid_model(input_ids=input_ids, labels=labels)
            loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else outputs[0]
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    import math
    ppl = math.exp(min(avg_loss, 20))

    result = {
        "variant": variant.name,
        "description": variant.description,
        "attention_interval": variant.attention_interval,
        "attention_layers_count": num_attn,
        "mamba_layers_count": num_mamba,
        "total_params": total_params,
        "alpha_kd": variant.alpha_kd,
        "alpha_ce": variant.alpha_ce,
        "phase1_steps": variant.phase1_steps,
        "max_steps": variant.max_steps,
        "eval_loss": round(avg_loss, 4),
        "eval_ppl": round(ppl, 2),
        "training_time_min": round(training_time / 60, 1),
    }

    logger.info(f"  Result: PPL={ppl:.2f}, Loss={avg_loss:.4f}, Time={training_time/60:.1f} min")

    # Cleanup
    del hybrid_model, teacher_model, trainer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def generate_ablation_summary(
    results: List[Dict],
    ablation_type: str,
    output_dir: str,
):
    """Generate markdown summary table for ablation results."""
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    lines.append(f"# Ablation Study: {ablation_type}")
    lines.append("")

    if ablation_type == "attention_ratio":
        lines.append("| Variant | Attn Layers | Mamba Layers | Params | Eval PPL | Eval Loss | Time (min) |")
        lines.append("|---------|:-----------:|:------------:|:------:|:--------:|:---------:|:----------:|")
        for r in results:
            lines.append(
                f"| {r['description']} | {r['attention_layers_count']} | {r['mamba_layers_count']} | "
                f"{r['total_params']/1e9:.2f}B | {r['eval_ppl']:.2f} | {r['eval_loss']:.4f} | "
                f"{r['training_time_min']:.0f} |"
            )

    elif ablation_type == "loss_weights":
        lines.append("| Variant | alpha_KD | alpha_CE | Eval PPL | Eval Loss | Time (min) |")
        lines.append("|---------|:--------:|:--------:|:--------:|:---------:|:----------:|")
        for r in results:
            lines.append(
                f"| {r['description']} | {r['alpha_kd']} | {r['alpha_ce']} | "
                f"{r['eval_ppl']:.2f} | {r['eval_loss']:.4f} | {r['training_time_min']:.0f} |"
            )

    elif ablation_type == "training_phases":
        lines.append("| Variant | Phase 1 Steps | Total Steps | Eval PPL | Eval Loss | Time (min) |")
        lines.append("|---------|:-------------:|:-----------:|:--------:|:---------:|:----------:|")
        for r in results:
            lines.append(
                f"| {r['description']} | {r['phase1_steps']} | {r['max_steps']} | "
                f"{r['eval_ppl']:.2f} | {r['eval_loss']:.4f} | {r['training_time_min']:.0f} |"
            )

    lines.append("")

    # Find best variant
    valid_results = [r for r in results if r["eval_ppl"] > 0]
    if valid_results:
        best = min(valid_results, key=lambda r: r["eval_ppl"])
        lines.append(f"**Best variant:** {best['description']} (PPL = {best['eval_ppl']:.2f})")

    lines.append("")

    summary = "\n".join(lines)

    # Save
    md_path = os.path.join(output_dir, f"ablation_{ablation_type}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(summary)

    json_path = os.path.join(output_dir, f"ablation_{ablation_type}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(summary)
    logger.info(f"Results saved: {md_path}")
    logger.info(f"Raw data saved: {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Ablation Study for Qwen-Mamba Hybrid")
    parser.add_argument("--ablation", type=str, required=True,
                        choices=["attention_ratio", "loss_weights", "training_phases", "all"],
                        help="Which ablation to run")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--output_dir", type=str, default="./ablation_results")
    parser.add_argument("--max_steps", type=int, default=2000, help="Training steps per variant")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Shorter seq for fast ablation")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--dummy_samples", type=int, default=500, help="Number of dummy training samples")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    setup_logging(log_level="INFO")
    set_seed(args.seed)

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("  Ablation Study")
    logger.info("=" * 60)
    logger.info(f"  Type:       {args.ablation}")
    logger.info(f"  Base model: {args.base_model}")
    logger.info(f"  Max steps:  {args.max_steps}")
    logger.info(f"  Seq length: {args.max_seq_length}")
    logger.info(f"  Output:     {args.output_dir}")
    logger.info("=" * 60)

    # Get base model layer count
    from transformers import AutoConfig
    base_config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    num_layers = base_config.num_hidden_layers

    # Determine which ablations to run
    ablation_sets = {}

    if args.ablation in ["attention_ratio", "all"]:
        ablation_sets["attention_ratio"] = get_attention_ratio_variants(num_layers, args.max_steps)

    if args.ablation in ["loss_weights", "all"]:
        ablation_sets["loss_weights"] = get_loss_weight_variants(args.max_steps)

    if args.ablation in ["training_phases", "all"]:
        ablation_sets["training_phases"] = get_training_phase_variants(args.max_steps)

    # Run each ablation set
    for ablation_type, variants in ablation_sets.items():
        logger.info(f"\n{'#'*60}")
        logger.info(f"  Running ablation: {ablation_type} ({len(variants)} variants)")
        logger.info(f"{'#'*60}")

        results = []
        for i, variant in enumerate(variants):
            logger.info(f"\n--- Variant {i+1}/{len(variants)}: {variant.name} ---")

            try:
                result = run_single_ablation(
                    variant=variant,
                    base_model=args.base_model,
                    output_dir=os.path.join(args.output_dir, ablation_type),
                    max_seq_length=args.max_seq_length,
                    batch_size=args.batch_size,
                    dummy=True,
                    dummy_samples=args.dummy_samples,
                    dtype=dtype,
                    device=device,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Variant {variant.name} failed: {e}")
                results.append({
                    "variant": variant.name,
                    "description": variant.description,
                    "error": str(e),
                    "eval_ppl": -1,
                    "eval_loss": -1,
                })

        # Generate summary
        generate_ablation_summary(results, ablation_type, args.output_dir)

    logger.info("\nAll ablation studies complete!")
    logger.info(f"Results directory: {args.output_dir}")


if __name__ == "__main__":
    main()
