#!/usr/bin/env python3
"""
Step 1: Architecture Surgery - Convert Qwen to Hybrid Model.

Supports two linear layer types:
- mamba:          Selective State Space Model
- gated_deltanet: Gated Delta Rule Network (Qwen3-Next style)

Usage:
    # Convert to Mamba hybrid
    python scripts/convert_model.py \
        --base_model Qwen/Qwen3-4B \
        --linear_type mamba \
        --output_dir ./checkpoints/qwen3-4b-mamba-init

    # Convert to GatedDeltaNet hybrid
    python scripts/convert_model.py \
        --base_model Qwen/Qwen3-4B \
        --linear_type gated_deltanet \
        --output_dir ./checkpoints/qwen3-4b-gdn-init

    # Dry run (show layer assignments only)
    python scripts/convert_model.py --base_model Qwen/Qwen3-4B --dry_run
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.models.architecture_surgery import (
    convert_qwen_to_hybrid,
    load_tokenizer,
    get_attention_layer_indices,
)
from src.utils.helpers import (
    load_config,
    setup_logging,
    print_model_summary,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Qwen to Hybrid Model")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--linear_type", type=str, default=None,
                        choices=["mamba", "gated_deltanet"],
                        help="Type of linear layer to replace attention")
    parser.add_argument("--attention_interval", type=int, default=None)
    parser.add_argument("--attention_layers", type=str, default=None,
                        help="Comma-separated list of attention layer indices")
    # Mamba params
    parser.add_argument("--mamba_d_state", type=int, default=None)
    parser.add_argument("--mamba_d_conv", type=int, default=None)
    parser.add_argument("--mamba_expand", type=int, default=None)
    # GatedDeltaNet params
    parser.add_argument("--gdn_key_head_dim", type=int, default=None)
    parser.add_argument("--gdn_value_head_dim", type=int, default=None)
    parser.add_argument("--gdn_conv_kernel", type=int, default=None)
    # Common
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)
    set_seed(args.seed)

    # Load config
    config = load_config(args.config)
    model_config = config.get("model", {})
    mamba_config = config.get("mamba", {})
    gdn_config = config.get("gated_deltanet", {})

    # Apply overrides
    base_model = args.base_model or model_config.get("base_model_name", "Qwen/Qwen3-4B")
    linear_type = args.linear_type or model_config.get("linear_layer_type", "mamba")
    attention_interval = args.attention_interval or model_config.get("attention_interval", 4)

    # Default output dir includes the linear type
    default_output = f"./checkpoints/qwen3-4b-{linear_type}-init"
    output_dir = args.output_dir or model_config.get("output_dir", default_output)

    # Mamba params
    mamba_d_state = args.mamba_d_state or mamba_config.get("d_state", 16)
    mamba_d_conv = args.mamba_d_conv or mamba_config.get("d_conv", 4)
    mamba_expand = args.mamba_expand or mamba_config.get("expand", 2)

    # GDN params
    gdn_key_head_dim = args.gdn_key_head_dim or gdn_config.get("key_head_dim", 128)
    gdn_value_head_dim = args.gdn_value_head_dim or gdn_config.get("value_head_dim", 128)
    gdn_conv_kernel = args.gdn_conv_kernel or gdn_config.get("conv_kernel", 4)

    # Attention layers
    attention_layers = None
    if args.attention_layers:
        attention_layers = [int(x.strip()) for x in args.attention_layers.split(",")]

    # Dtype
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    # Print config
    logger.info("=" * 60)
    logger.info(f"  Qwen -> Qwen-{linear_type.title()} Hybrid Conversion")
    logger.info("=" * 60)
    logger.info(f"  Base model:          {base_model}")
    logger.info(f"  Linear layer type:   {linear_type}")
    logger.info(f"  Attention interval:  {attention_interval}")
    if linear_type == "mamba":
        logger.info(f"  Mamba d_state:       {mamba_d_state}")
        logger.info(f"  Mamba d_conv:        {mamba_d_conv}")
        logger.info(f"  Mamba expand:        {mamba_expand}")
    else:
        logger.info(f"  GDN key_head_dim:    {gdn_key_head_dim}")
        logger.info(f"  GDN value_head_dim:  {gdn_value_head_dim}")
        logger.info(f"  GDN conv_kernel:     {gdn_conv_kernel}")
    logger.info(f"  Output dir:          {output_dir}")
    logger.info("=" * 60)

    # Dry run
    if args.dry_run:
        from transformers import AutoConfig
        base_cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        num_layers = base_cfg.num_hidden_layers
        attn_idx, lin_idx = get_attention_layer_indices(num_layers, attention_interval, attention_layers)
        logger.info(f"\nModel has {num_layers} layers:")
        for i in range(num_layers):
            ltype = "Attention" if i in attn_idx else f"{linear_type.title()}"
            logger.info(f"  Layer {i:>3d}: {ltype}")
        logger.info(f"\n  Attention: {len(attn_idx)} / {num_layers}")
        logger.info(f"  {linear_type.title()}: {len(lin_idx)} / {num_layers}")
        return

    # Surgery
    hybrid_model, _ = convert_qwen_to_hybrid(
        model_name_or_path=base_model,
        attention_interval=attention_interval,
        attention_layers=attention_layers,
        linear_layer_type=linear_type,
        mamba_d_state=mamba_d_state,
        mamba_d_conv=mamba_d_conv,
        mamba_expand=mamba_expand,
        gdn_key_head_dim=gdn_key_head_dim,
        gdn_value_head_dim=gdn_value_head_dim,
        gdn_conv_kernel=gdn_conv_kernel,
        torch_dtype=torch_dtype,
        load_teacher=False,
    )

    print_model_summary(hybrid_model, title=f"Hybrid Model ({linear_type})")

    # Save
    logger.info(f"Saving hybrid model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    hybrid_model.save_pretrained(output_dir)

    tokenizer = load_tokenizer(base_model)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"\nConversion complete! Model saved to: {output_dir}")
    logger.info("Next: Run distillation training:")
    logger.info(f"  python scripts/train_distill.py --linear_type {linear_type}")


if __name__ == "__main__":
    main()
