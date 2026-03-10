"""
Qwen Hybrid Model Definition.

This module defines the hybrid model configuration and model class that
combines Attention layers with linear-complexity layers (Mamba or GatedDeltaNet).

Supports two linear layer types:
- "mamba":          Selective State Space Model (Gu & Dao, 2023)
- "gated_deltanet": Gated Delta Rule Network (Qwen3-Next style)

The model maintains compatibility with HuggingFace's PreTrainedModel API,
supporting save_pretrained/from_pretrained, generate(), and other utilities.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModelForCausalLM
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .mamba_block import MambaBlock, MambaCache, HybridCache
from .gated_deltanet_block import GatedDeltaNetBlock, GatedDeltaNetCache


# Supported linear layer types
LINEAR_LAYER_TYPES = ("mamba", "gated_deltanet")


class QwenHybridConfig(PretrainedConfig):
    """
    Configuration for the Qwen Hybrid model.

    Extends Qwen config with parameters for the linear layer replacement.
    Supports both Mamba and GatedDeltaNet as the linear layer type.
    """
    model_type = "qwen_hybrid"

    def __init__(
        self,
        # Base Qwen config parameters
        vocab_size: int = 151936,
        hidden_size: int = 2560,
        intermediate_size: int = 9728,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        max_position_embeddings: int = 40960,
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = True,
        rope_theta: float = 1000000.0,
        use_sliding_window: bool = False,
        sliding_window: Optional[int] = None,
        max_window_layers: int = 36,
        # Hybrid model parameters
        attention_layers: Optional[List[int]] = None,
        attention_interval: int = 4,
        # Linear layer type selection
        linear_layer_type: str = "mamba",  # "mamba" or "gated_deltanet"
        # Mamba parameters
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        # GatedDeltaNet parameters
        gdn_key_head_dim: int = 128,
        gdn_value_head_dim: int = 128,
        gdn_conv_kernel: int = 4,
        gdn_use_output_gate: bool = True,
        gdn_num_heads: Optional[int] = None,  # None = auto from hidden_size/key_head_dim
        # Base model name
        base_model_name: str = "Qwen/Qwen3-4B",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        # Hybrid-specific
        self.attention_interval = attention_interval
        self.linear_layer_type = linear_layer_type
        self.base_model_name = base_model_name

        # Mamba parameters
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand

        # GatedDeltaNet parameters
        self.gdn_key_head_dim = gdn_key_head_dim
        self.gdn_value_head_dim = gdn_value_head_dim
        self.gdn_conv_kernel = gdn_conv_kernel
        self.gdn_use_output_gate = gdn_use_output_gate
        self.gdn_num_heads = gdn_num_heads

        # Validate
        assert linear_layer_type in LINEAR_LAYER_TYPES, \
            f"linear_layer_type must be one of {LINEAR_LAYER_TYPES}, got '{linear_layer_type}'"

        # Compute which layers are attention vs linear
        if attention_layers is not None:
            self.attention_layers = attention_layers
        else:
            self.attention_layers = [
                i for i in range(num_hidden_layers) if i % attention_interval == 0
            ]

        self.linear_layers = [
            i for i in range(num_hidden_layers) if i not in self.attention_layers
        ]

        # Backward compatibility aliases
        self.mamba_layers = self.linear_layers

    @classmethod
    def from_qwen_config(
        cls,
        qwen_config,
        attention_interval=4,
        attention_layers=None,
        linear_layer_type="mamba",
        # Mamba params
        mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
        # GatedDeltaNet params
        gdn_key_head_dim=128, gdn_value_head_dim=128,
        gdn_conv_kernel=4, gdn_use_output_gate=True, gdn_num_heads=None,
        base_model_name="Qwen/Qwen3-4B",
    ):
        """Create a hybrid config from an existing Qwen config."""
        config_dict = qwen_config.to_dict()
        for key in ["model_type", "architectures", "_name_or_path", "transformers_version"]:
            config_dict.pop(key, None)

        return cls(
            attention_interval=attention_interval,
            attention_layers=attention_layers,
            linear_layer_type=linear_layer_type,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            gdn_key_head_dim=gdn_key_head_dim,
            gdn_value_head_dim=gdn_value_head_dim,
            gdn_conv_kernel=gdn_conv_kernel,
            gdn_use_output_gate=gdn_use_output_gate,
            gdn_num_heads=gdn_num_heads,
            base_model_name=base_model_name,
            **config_dict,
        )

    def is_attention_layer(self, layer_idx: int) -> bool:
        """Check if a given layer should use attention."""
        return layer_idx in self.attention_layers

    def is_linear_layer(self, layer_idx: int) -> bool:
        """Check if a given layer should use a linear layer (Mamba or GatedDeltaNet)."""
        return layer_idx in self.linear_layers

    # Backward compatibility
    def is_mamba_layer(self, layer_idx: int) -> bool:
        return self.is_linear_layer(layer_idx)

    @property
    def linear_type_display(self) -> str:
        """Human-readable linear layer type name."""
        return {
            "mamba": "Mamba (S6 SSM)",
            "gated_deltanet": "GatedDeltaNet",
        }.get(self.linear_layer_type, self.linear_layer_type)


# Backward compatibility alias
QwenMambaHybridConfig = QwenHybridConfig


class QwenHybridForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Qwen Hybrid model for causal language modeling.

    This is a wrapper that holds the actual hybrid model. The model is constructed
    by modifying a Qwen model in-place (replacing attention layers with linear blocks).

    Supports both Mamba and GatedDeltaNet as the linear layer type.
    """
    config_class = QwenHybridConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: QwenHybridConfig):
        super().__init__(config)
        self.config = config
        self.model = None
        self.lm_head = None
        self._linear_cache = None

    def _set_inner_model(self, model_backbone, lm_head):
        """Set the inner model components after surgery."""
        self.model = model_backbone
        self.lm_head = lm_head

    def get_input_embeddings(self):
        if self.model is not None:
            return self.model.embed_tokens
        return None

    def set_input_embeddings(self, value):
        if self.model is not None:
            self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_linear_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Initialize or return the linear layer cache for generation."""
        if self._linear_cache is not None:
            return self._linear_cache

        if self.config.linear_layer_type == "mamba":
            self._linear_cache = MambaCache()
            for layer_idx in self.config.linear_layers:
                d_inner = int(self.config.mamba_expand * self.config.hidden_size)
                self._linear_cache.init_layer(
                    layer_idx=layer_idx,
                    batch_size=batch_size,
                    d_inner=d_inner,
                    d_conv=self.config.mamba_d_conv,
                    d_state=self.config.mamba_d_state,
                    device=device,
                    dtype=dtype,
                )
        elif self.config.linear_layer_type == "gated_deltanet":
            self._linear_cache = GatedDeltaNetCache()
            num_heads = self.config.gdn_num_heads or (
                self.config.hidden_size // self.config.gdn_key_head_dim
            )
            q_dim = num_heads * self.config.gdn_key_head_dim
            for layer_idx in self.config.linear_layers:
                self._linear_cache.init_layer(
                    layer_idx=layer_idx,
                    batch_size=batch_size,
                    num_heads=num_heads,
                    key_head_dim=self.config.gdn_key_head_dim,
                    value_head_dim=self.config.gdn_value_head_dim,
                    conv_kernel=self.config.gdn_conv_kernel,
                    proj_dim=q_dim,
                    device=device,
                    dtype=dtype,
                )

        return self._linear_cache

    # Backward compatibility
    def get_mamba_cache(self, batch_size, device, dtype):
        return self.get_linear_cache(batch_size, device, dtype)

    def reset_cache(self):
        """Reset the linear layer cache."""
        self._linear_cache = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[object] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ):
        """Forward pass of the hybrid model."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.model is None:
            raise RuntimeError(
                "Model not initialized. Use `convert_qwen_to_hybrid()` or "
                "`from_pretrained()` to create the model."
            )

        # Wrap cache in HybridCache so Mamba layers get their SSM state
        if use_cache and not isinstance(past_key_values, HybridCache):
            # Determine batch info for MambaCache initialization
            if input_ids is not None:
                batch_size = input_ids.shape[0]
                device = input_ids.device
            elif inputs_embeds is not None:
                batch_size = inputs_embeds.shape[0]
                device = inputs_embeds.device
            else:
                batch_size = 1
                device = next(self.parameters()).device

            dtype = next(self.parameters()).dtype
            hybrid_cache = HybridCache()
            # Initialize Mamba states for all linear layers
            for layer_idx in self.config.linear_layers:
                d_inner = int(self.config.mamba_expand * self.config.hidden_size)
                hybrid_cache.mamba_cache.init_layer(
                    layer_idx=layer_idx,
                    batch_size=batch_size,
                    d_inner=d_inner,
                    d_conv=self.config.mamba_d_conv,
                    d_state=self.config.mamba_d_state,
                    device=device,
                    dtype=dtype,
                )
            past_key_values = hybrid_cache

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]
        # During generation (no labels), only compute logits for the last
        # num_logits_to_keep positions to avoid materializing a huge
        # (B, L, vocab_size) tensor in fp32.  generate() passes
        # num_logits_to_keep=1 automatically; training passes 0 (keep all).
        if num_logits_to_keep > 0:
            hidden_states = hidden_states[:, -num_logits_to_keep:, :]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, "past_key_values") else None,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """Prepare inputs for the generate() method."""
        # In transformers 5.x, past_key_values may be an empty DynamicCache
        # (never None). Only truncate to last token if the cache has content.
        cache_has_content = (
            past_key_values is not None
            and hasattr(past_key_values, 'get_seq_length')
            and past_key_values.get_seq_length() > 0
        )
        if cache_has_content:
            input_ids = input_ids[:, -1:]

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "num_logits_to_keep": 1,
        }

        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds
            model_inputs["input_ids"] = None

        return model_inputs

    @classmethod
    def from_surgery(cls, hybrid_model_backbone, lm_head, config: QwenHybridConfig):
        """Create from surgery results."""
        model = cls(config)
        model._set_inner_model(hybrid_model_backbone, lm_head)
        return model

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save model with hybrid config."""
        os.makedirs(save_directory, exist_ok=True)

        self.config.save_pretrained(save_directory)

        state_dict = {}
        if self.model is not None:
            for name, param in self.model.named_parameters():
                state_dict[f"model.{name}"] = param.data
            for name, buf in self.model.named_buffers():
                state_dict[f"model.{name}"] = buf

        if self.lm_head is not None and not getattr(self.config, 'tie_word_embeddings', False):
            for name, param in self.lm_head.named_parameters():
                state_dict[f"lm_head.{name}"] = param.data

        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))

        layer_map = {
            "attention_layers": self.config.attention_layers,
            "linear_layers": self.config.linear_layers,
            "linear_layer_type": self.config.linear_layer_type,
            "total_layers": self.config.num_hidden_layers,
        }
        with open(os.path.join(save_directory, "layer_map.json"), "w") as f:
            json.dump(layer_map, f, indent=2)

        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained_hybrid(cls, save_directory: str, device_map="auto", torch_dtype=None, base_model_override: str = None):
        """Load a previously saved hybrid model.

        Args:
            save_directory: Path to the saved hybrid checkpoint.
            device_map: Device map for loading.
            torch_dtype: Model dtype.
            base_model_override: If provided, use this path for the base Qwen
                model instead of config.base_model_name.
        """
        from .architecture_surgery import convert_qwen_to_hybrid

        config = QwenHybridConfig.from_pretrained(save_directory)
        base_model_name = base_model_override or config.base_model_name

        hybrid_model, _ = convert_qwen_to_hybrid(
            model_name_or_path=base_model_name,
            attention_interval=config.attention_interval,
            attention_layers=config.attention_layers,
            linear_layer_type=config.linear_layer_type,
            mamba_d_state=config.mamba_d_state,
            mamba_d_conv=config.mamba_d_conv,
            mamba_expand=config.mamba_expand,
            gdn_key_head_dim=config.gdn_key_head_dim,
            gdn_value_head_dim=config.gdn_value_head_dim,
            gdn_conv_kernel=config.gdn_conv_kernel,
            load_teacher=False,
        )

        state_dict_path = os.path.join(save_directory, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location="cpu")
            hybrid_model.load_state_dict(state_dict, strict=False)

        if torch_dtype is not None:
            hybrid_model = hybrid_model.to(torch_dtype)

        return hybrid_model


# Backward compatibility alias
QwenMambaHybridForCausalLM = QwenHybridForCausalLM
