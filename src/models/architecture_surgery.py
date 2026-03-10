"""
Architecture Surgery: Convert Qwen to Hybrid (Mamba or GatedDeltaNet).

This module performs the "architecture surgery" that:
1. Loads a pre-trained Qwen model
2. Keeps every N-th attention layer (for ICL capability)
3. Replaces the remaining attention layers with linear layers (Mamba or GatedDeltaNet)
4. Preserves embedding, MLP, normalization, and LM head weights

Supports two linear layer types:
- "mamba":          Selective State Space Model
- "gated_deltanet": Gated Delta Rule Network (Qwen3-Next style)
"""

import copy
import gc
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from .mamba_block import MambaBlock
from .gated_deltanet_block import GatedDeltaNetBlock
from .hybrid_model import QwenHybridConfig, QwenHybridForCausalLM

# Backward compatibility
QwenMambaHybridConfig = QwenHybridConfig
QwenMambaHybridForCausalLM = QwenHybridForCausalLM

logger = logging.getLogger(__name__)


def get_attention_layer_indices(
    num_layers: int,
    attention_interval: int = 4,
    attention_layers: Optional[List[int]] = None,
) -> Tuple[List[int], List[int]]:
    """
    Determine which layers keep attention and which get replaced.

    Args:
        num_layers: Total number of decoder layers
        attention_interval: Keep attention every N layers
        attention_layers: Explicit list of attention layer indices

    Returns:
        Tuple of (attention_layer_indices, linear_layer_indices)
    """
    if attention_layers is not None:
        attn_indices = sorted([i for i in attention_layers if 0 <= i < num_layers])
    else:
        attn_indices = [i for i in range(num_layers) if i % attention_interval == 0]

    linear_indices = [i for i in range(num_layers) if i not in attn_indices]

    return attn_indices, linear_indices


def _replace_attention_with_linear(
    model: nn.Module,
    linear_layer_indices: List[int],
    config: QwenHybridConfig,
) -> nn.Module:
    """
    Replace attention modules in specified layers with linear blocks.

    Supports both Mamba and GatedDeltaNet as replacement types.
    Modifies the model in-place.
    """
    # Access decoder layers (Qwen2/Qwen3: model.model.layers[i].self_attn)
    layers = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        raise ValueError("Cannot find decoder layers in model.")

    num_replaced = 0
    for idx in linear_layer_indices:
        if idx >= len(layers):
            logger.warning(f"Layer {idx} out of range (model has {len(layers)} layers), skipping.")
            continue

        layer = layers[idx]

        # Get device and dtype from old attention
        old_attn = layer.self_attn
        try:
            param = next(old_attn.parameters())
            device = param.device
            dtype = param.dtype
        except StopIteration:
            device = torch.device("cpu")
            dtype = torch.float32

        # Create replacement block
        if config.linear_layer_type == "mamba":
            new_block = MambaBlock(
                config=config,
                layer_idx=idx,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
            ).to(device=device, dtype=dtype)

        elif config.linear_layer_type == "gated_deltanet":
            new_block = GatedDeltaNetBlock(
                config=config,
                layer_idx=idx,
                key_head_dim=config.gdn_key_head_dim,
                value_head_dim=config.gdn_value_head_dim,
                conv_kernel=config.gdn_conv_kernel,
                use_output_gate=config.gdn_use_output_gate,
            ).to(device=device, dtype=dtype)

        else:
            raise ValueError(f"Unknown linear_layer_type: {config.linear_layer_type}")

        # Replace
        layer.self_attn = new_block
        num_replaced += 1

        # Free old attention
        del old_attn
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    layer_type_name = config.linear_type_display
    logger.info(f"Replaced {num_replaced} attention layers with {layer_type_name} blocks.")
    return model


def convert_qwen_to_hybrid(
    model_name_or_path: str = "Qwen/Qwen3-4B",
    attention_interval: int = 4,
    attention_layers: Optional[List[int]] = None,
    linear_layer_type: str = "mamba",
    # Mamba parameters
    mamba_d_state: int = 16,
    mamba_d_conv: int = 4,
    mamba_expand: int = 2,
    # GatedDeltaNet parameters
    gdn_key_head_dim: int = 128,
    gdn_value_head_dim: int = 128,
    gdn_conv_kernel: int = 4,
    gdn_use_output_gate: bool = True,
    gdn_num_heads: Optional[int] = None,
    # Common parameters
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    load_teacher: bool = True,
    trust_remote_code: bool = True,
) -> Tuple[QwenHybridForCausalLM, Optional[AutoModelForCausalLM]]:
    """
    Perform architecture surgery to create a Qwen hybrid model.

    Steps:
    1. Load the pre-trained Qwen model
    2. Create the hybrid configuration
    3. Copy the model for surgery (teacher is kept frozen)
    4. Replace specified attention layers with linear blocks
    5. Return both the hybrid student model and the frozen teacher

    Args:
        model_name_or_path: HuggingFace model name or local path
        attention_interval: Keep every N-th attention layer
        attention_layers: Explicit list of layers to keep as attention
        linear_layer_type: Type of linear layer ("mamba" or "gated_deltanet")
        mamba_*: Mamba-specific parameters
        gdn_*: GatedDeltaNet-specific parameters
        torch_dtype: Model dtype
        device_map: Device map for loading
        load_teacher: Whether to also load the teacher model
        trust_remote_code: Trust remote code from HuggingFace

    Returns:
        Tuple of (hybrid_model, teacher_model)
    """
    logger.info(f"Loading base model: {model_name_or_path}")
    logger.info(f"Linear layer type: {linear_layer_type}")

    # --- Early return: already-converted hybrid model ---
    # If the checkpoint has model_type=qwen_hybrid (output of convert_model.py),
    # load it directly instead of re-running architecture surgery.
    _config_path = os.path.join(model_name_or_path, "config.json") if os.path.isdir(model_name_or_path) else None
    if _config_path and os.path.exists(_config_path):
        import json as _json
        with open(_config_path) as _f:
            _raw = _json.load(_f)
        if _raw.get("model_type") == "qwen_hybrid":
            logger.info("Detected already-converted hybrid model. Loading directly...")
            hybrid_config = QwenHybridConfig.from_pretrained(model_name_or_path)

            # Build the hybrid model structure (no Qwen weights needed yet)
            _tmp_base = AutoModelForCausalLM.from_pretrained(
                hybrid_config.base_model_name,
                torch_dtype=torch_dtype,
                device_map="cpu",
                trust_remote_code=trust_remote_code,
            )
            _tmp_student = copy.deepcopy(_tmp_base)
            _, _linear_indices = get_attention_layer_indices(
                num_layers=hybrid_config.num_hidden_layers,
                attention_interval=hybrid_config.attention_interval,
                attention_layers=hybrid_config.attention_layers,
            )
            _replace_attention_with_linear(
                model=_tmp_student,
                linear_layer_indices=_linear_indices,
                config=hybrid_config,
            )
            _backbone = _tmp_student.model
            _lm_head = _tmp_student.lm_head
            hybrid_model = QwenHybridForCausalLM.from_surgery(
                hybrid_model_backbone=_backbone,
                lm_head=_lm_head,
                config=hybrid_config,
            )
            # Load saved weights
            _ckpt = os.path.join(model_name_or_path, "pytorch_model.bin")
            if os.path.exists(_ckpt):
                _sd = torch.load(_ckpt, map_location="cpu")
                hybrid_model.load_state_dict(_sd, strict=False)
                logger.info(f"Loaded hybrid weights from {_ckpt}")
            hybrid_model = hybrid_model.to(torch_dtype)
            # Re-tie embeddings
            if getattr(hybrid_config, 'tie_word_embeddings', False):
                if hybrid_model.lm_head is not None and hybrid_model.model is not None:
                    hybrid_model.lm_head.weight = hybrid_model.model.embed_tokens.weight
            # Teacher
            teacher_model = None
            if load_teacher:
                logger.info(f"Loading teacher from {hybrid_config.base_model_name}...")
                teacher_model = _tmp_base
                teacher_model.eval()
                for param in teacher_model.parameters():
                    param.requires_grad = False
            else:
                del _tmp_base
                gc.collect()
            return hybrid_model, teacher_model

    # Step 1: Load original Qwen model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    base_config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )

    num_layers = base_config.num_hidden_layers
    logger.info(f"Base model has {num_layers} layers")

    # Step 2: Determine layer assignment
    attn_indices, linear_indices = get_attention_layer_indices(
        num_layers=num_layers,
        attention_interval=attention_interval,
        attention_layers=attention_layers,
    )
    logger.info(f"Attention layers ({len(attn_indices)}): {attn_indices}")
    logger.info(f"{linear_layer_type} layers ({len(linear_indices)}): {linear_indices}")

    # Step 3: Create hybrid configuration
    hybrid_config = QwenHybridConfig.from_qwen_config(
        qwen_config=base_config,
        attention_interval=attention_interval,
        attention_layers=attn_indices,
        linear_layer_type=linear_layer_type,
        mamba_d_state=mamba_d_state,
        mamba_d_conv=mamba_d_conv,
        mamba_expand=mamba_expand,
        gdn_key_head_dim=gdn_key_head_dim,
        gdn_value_head_dim=gdn_value_head_dim,
        gdn_conv_kernel=gdn_conv_kernel,
        gdn_use_output_gate=gdn_use_output_gate,
        gdn_num_heads=gdn_num_heads,
        base_model_name=model_name_or_path,
    )

    # Step 4: Surgery
    logger.info("Performing architecture surgery...")
    student_model = copy.deepcopy(base_model)

    _replace_attention_with_linear(
        model=student_model,
        linear_layer_indices=linear_indices,
        config=hybrid_config,
    )

    # Wrap in hybrid model class
    if hasattr(student_model, 'model') and hasattr(student_model, 'lm_head'):
        backbone = student_model.model
        lm_head = student_model.lm_head
    else:
        raise ValueError("Unexpected model structure.")

    hybrid_model = QwenHybridForCausalLM.from_surgery(
        hybrid_model_backbone=backbone,
        lm_head=lm_head,
        config=hybrid_config,
    )

    # Re-tie embedding weights if needed (Qwen3 uses tie_word_embeddings=True)
    if getattr(base_config, 'tie_word_embeddings', False):
        if hybrid_model.lm_head is not None and hybrid_model.model is not None:
            hybrid_model.lm_head.weight = hybrid_model.model.embed_tokens.weight
            logger.info("Re-tied lm_head weights with embedding weights.")

    # Step 5: Teacher model
    teacher_model = None
    if load_teacher:
        logger.info("Preparing teacher model (frozen)...")
        teacher_model = base_model
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
    else:
        del base_model
        gc.collect()

    # Step 6: Summary
    _print_surgery_summary(hybrid_model, hybrid_config, attn_indices, linear_indices)

    return hybrid_model, teacher_model


def _print_surgery_summary(
    model: QwenHybridForCausalLM,
    config: QwenHybridConfig,
    attn_indices: List[int],
    linear_indices: List[int],
):
    """Print a summary of the architecture surgery."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count linear layer parameters
    linear_params = 0
    for name, param in model.named_parameters():
        if "mamba" in name.lower() or "gated_deltanet" in name.lower() or "deltanet" in name.lower():
            linear_params += param.numel()

    layer_type = config.linear_type_display

    print("\n" + "=" * 70)
    print("  ARCHITECTURE SURGERY SUMMARY")
    print("=" * 70)
    print(f"  Base model:          {config.base_model_name}")
    print(f"  Linear layer type:   {layer_type}")
    print(f"  Hidden size:         {config.hidden_size}")
    print(f"  Total layers:        {config.num_hidden_layers}")
    print(f"  Attention layers:    {len(attn_indices)} -> {attn_indices}")
    print(f"  {layer_type} layers: {len(linear_indices)} -> {linear_indices}")

    if config.linear_layer_type == "mamba":
        print(f"  Mamba d_state:       {config.mamba_d_state}")
        print(f"  Mamba d_conv:        {config.mamba_d_conv}")
        print(f"  Mamba expand:        {config.mamba_expand}")
    else:
        print(f"  GDN key_head_dim:    {config.gdn_key_head_dim}")
        print(f"  GDN value_head_dim:  {config.gdn_value_head_dim}")
        print(f"  GDN conv_kernel:     {config.gdn_conv_kernel}")
        print(f"  GDN num_heads:       {config.gdn_num_heads or 'auto'}")

    print("-" * 70)
    print(f"  Total parameters:    {total_params:,}")
    print(f"  Trainable params:    {trainable_params:,}")
    print(f"  Linear layer params: {linear_params:,}")
    print(f"  Ratio (Linear/Total): {linear_params/max(total_params,1)*100:.1f}%")
    print("=" * 70 + "\n")


def load_tokenizer(model_name_or_path: str, trust_remote_code: bool = True):
    """Load the tokenizer for the base model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def freeze_non_linear_parameters(model: QwenHybridForCausalLM, unfreeze_lm_head: bool = False):
    """
    Freeze all parameters except the linear (Mamba/GatedDeltaNet) layers.

    Useful for Phase 1 training to warm up newly initialized linear layers.
    """
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Keywords to identify linear layer parameters
    linear_keywords = ["mamba", "gated_deltanet", "deltanet", "delta_net"]
    linear_count = 0

    for name, param in model.named_parameters():
        if any(kw in name.lower() for kw in linear_keywords):
            param.requires_grad = True
            linear_count += 1

    if unfreeze_lm_head and model.lm_head is not None:
        for param in model.lm_head.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Froze non-linear parameters. Trainable: {trainable:,}/{total:,} "
        f"({trainable/total*100:.1f}%)"
    )


# Backward compatibility alias
freeze_non_mamba_parameters = freeze_non_linear_parameters


def unfreeze_all_parameters(model: QwenHybridForCausalLM):
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    logger.info("All parameters unfrozen for full training.")
