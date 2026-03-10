"""
Utility Functions for Qwen-Mamba Hybrid Model.

Provides common helper functions for configuration loading,
logging setup, parameter counting, and model inspection.
"""

import logging
import os
import random
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Dictionary of configuration values
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
):
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    # Reduce noise from some libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Returns:
        Dictionary with:
        - total: Total parameter count
        - trainable: Trainable parameter count
        - frozen: Frozen parameter count
        - by_module: Dict of parameter counts per top-level module
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    # Count by top-level module
    by_module = {}
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        by_module[name] = {
            "total": module_params,
            "trainable": module_trainable,
        }

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "by_module": by_module,
    }


def print_model_summary(model: torch.nn.Module, title: str = "Model Summary"):
    """
    Print a detailed summary of model parameters.

    Args:
        model: The model to summarize
        title: Title for the summary
    """
    stats = count_parameters(model)

    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  Total parameters:     {stats['total']:>15,}")
    print(f"  Trainable:            {stats['trainable']:>15,}")
    print(f"  Frozen:               {stats['frozen']:>15,}")
    print(f"  Trainable ratio:      {stats['trainable']/max(stats['total'],1)*100:>14.1f}%")
    print(f"{'─' * 60}")

    # Per-module breakdown
    print(f"  {'Module':<30} {'Total':>12} {'Trainable':>12}")
    print(f"  {'─' * 54}")
    for name, info in stats["by_module"].items():
        print(f"  {name:<30} {info['total']:>12,} {info['trainable']:>12,}")
    print(f"{'=' * 60}\n")


def print_layer_types(model: torch.nn.Module):
    """
    Print the type of each decoder layer (Attention vs Mamba).

    Args:
        model: The hybrid model
    """
    # Find the decoder layers
    layers = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        print("Cannot find decoder layers in model.")
        return

    print(f"\n{'=' * 50}")
    print("  Layer Type Mapping")
    print(f"{'=' * 50}")

    for i, layer in enumerate(layers):
        attn_module = layer.self_attn
        module_type = type(attn_module).__name__

        if "mamba" in module_type.lower():
            layer_type = "Mamba (SSM)"
            icon = "🔄"
        else:
            layer_type = "Attention"
            icon = "👁️"

        print(f"  Layer {i:>3d}: {icon} {layer_type:<20} ({module_type})")

    print(f"{'=' * 50}\n")


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory usage info (in GB)."""
    if not torch.cuda.is_available():
        return {"available": False}

    info = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        info[f"gpu_{i}"] = {
            "name": torch.cuda.get_device_properties(i).name,
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(total - allocated, 2),
        }

    return info


def estimate_model_memory(
    hidden_size: int,
    num_layers: int,
    vocab_size: int,
    num_attention_layers: int,
    num_mamba_layers: int,
    mamba_expand: int = 2,
    mamba_d_state: int = 16,
    intermediate_size: Optional[int] = None,
    num_attention_heads: int = 16,
    num_kv_heads: int = 2,
    dtype_bytes: int = 2,  # bf16 = 2 bytes
) -> Dict[str, float]:
    """
    Estimate model memory footprint in GB.

    Returns:
        Dictionary with estimated memory for different components.
    """
    if intermediate_size is None:
        intermediate_size = int(hidden_size * 8 / 3)

    d_inner = hidden_size * mamba_expand

    # Embedding + LM head
    embed_params = vocab_size * hidden_size * 2  # embed + lm_head

    # Attention layer params
    # QKV projection + output projection
    head_dim = hidden_size // num_attention_heads
    q_params = hidden_size * hidden_size
    k_params = hidden_size * (num_kv_heads * head_dim)
    v_params = hidden_size * (num_kv_heads * head_dim)
    o_params = hidden_size * hidden_size
    attn_per_layer = q_params + k_params + v_params + o_params

    # Mamba layer params
    in_proj = hidden_size * d_inner * 2  # x and z
    conv = d_inner * 4  # conv1d (grouped)
    x_proj = d_inner * (mamba_d_state * 2 + hidden_size // 16)  # dt_rank + B + C
    dt_proj = (hidden_size // 16) * d_inner
    a_log = d_inner * mamba_d_state
    d_param = d_inner
    out_proj = d_inner * hidden_size
    mamba_per_layer = in_proj + conv + x_proj + dt_proj + a_log + d_param + out_proj

    # MLP params (same for all layers)
    mlp_per_layer = hidden_size * intermediate_size * 3  # gate + up + down

    # Norms
    norm_per_layer = hidden_size * 2

    # Total
    total_params = (
        embed_params
        + num_attention_layers * (attn_per_layer + mlp_per_layer + norm_per_layer)
        + num_mamba_layers * (mamba_per_layer + mlp_per_layer + norm_per_layer)
    )

    total_bytes = total_params * dtype_bytes
    total_gb = total_bytes / 1e9

    return {
        "total_params": total_params,
        "total_gb": round(total_gb, 2),
        "embedding_gb": round(embed_params * dtype_bytes / 1e9, 2),
        "attention_layers_gb": round(
            num_attention_layers * attn_per_layer * dtype_bytes / 1e9, 2
        ),
        "mamba_layers_gb": round(
            num_mamba_layers * mamba_per_layer * dtype_bytes / 1e9, 2
        ),
        "mlp_layers_gb": round(
            num_layers * mlp_per_layer * dtype_bytes / 1e9, 2
        ),
    }
