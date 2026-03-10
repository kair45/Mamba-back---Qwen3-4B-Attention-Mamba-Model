from .mamba_block import MambaMixer, MambaBlock, MambaCache
from .gated_deltanet_block import GatedDeltaNetMixer, GatedDeltaNetBlock, GatedDeltaNetCache
from .hybrid_model import (
    QwenHybridConfig, QwenHybridForCausalLM,
    QwenMambaHybridConfig, QwenMambaHybridForCausalLM,  # backward compat
)
from .architecture_surgery import convert_qwen_to_hybrid, get_attention_layer_indices
from .moe_expansion import expand_mlp_to_moe, MoEMLP, TopKRouter

__all__ = [
    "MambaMixer", "MambaBlock", "MambaCache",
    "GatedDeltaNetMixer", "GatedDeltaNetBlock", "GatedDeltaNetCache",
    "QwenHybridConfig", "QwenHybridForCausalLM",
    "QwenMambaHybridConfig", "QwenMambaHybridForCausalLM",
    "convert_qwen_to_hybrid", "get_attention_layer_indices",
    "expand_mlp_to_moe", "MoEMLP", "TopKRouter",
]
