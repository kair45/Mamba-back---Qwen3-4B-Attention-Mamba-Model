"""
GatedDeltaNet (Gated Delta Rule Network) Block Implementation.

This module provides a GatedDeltaNet layer as a drop-in replacement for
attention layers, following the architecture used in Qwen3-Next.

The delta rule enables associative memory with matrix-valued recurrent state:
    S_t = (1 - β_t) * S_{t-1} + β_t * v_t ⊗ k_t   (gated update)
    o_t = S_t @ q_t                                    (linear readout)

Compared to Mamba's vector-valued state, GatedDeltaNet's matrix state
provides stronger associative recall at the cost of higher memory per head.

Features:
- Multi-head with configurable key/value head dimensions
- Short causal convolution on Q and K for local context
- Output gating (SiLU) for expressiveness
- Pure PyTorch reference implementation
- Optional CUDA-optimized kernels via `fla` (flash-linear-attention) package
- Compatible forward signature with HuggingFace attention layers
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import optimized CUDA kernels from flash-linear-attention
try:
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    HAS_FLA_CUDA = True
except ImportError:
    HAS_FLA_CUDA = False

try:
    from causal_conv1d import causal_conv1d_fn
    HAS_CAUSAL_CONV1D = True
except ImportError:
    HAS_CAUSAL_CONV1D = False


# ---------------------------------------------------------------------------
# GatedDeltaNet Cache (for autoregressive generation)
# ---------------------------------------------------------------------------

@dataclass
class GatedDeltaNetCache:
    """
    Cache for GatedDeltaNet layers during autoregressive generation.

    Stores the matrix-valued recurrent state and convolution states
    for each GatedDeltaNet layer.
    """
    # layer_idx -> (B, num_heads, d_v, d_k) matrix state
    recurrent_states: dict = field(default_factory=dict)
    # layer_idx -> (B, proj_dim, conv_kernel) for Q conv
    q_conv_states: dict = field(default_factory=dict)
    # layer_idx -> (B, proj_dim, conv_kernel) for K conv
    k_conv_states: dict = field(default_factory=dict)

    def init_layer(
        self,
        layer_idx: int,
        batch_size: int,
        num_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel: int,
        proj_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Initialize cache for a specific layer."""
        self.recurrent_states[layer_idx] = torch.zeros(
            batch_size, num_heads, value_head_dim, key_head_dim,
            device=device, dtype=dtype,
        )
        self.q_conv_states[layer_idx] = torch.zeros(
            batch_size, proj_dim, conv_kernel,
            device=device, dtype=dtype,
        )
        self.k_conv_states[layer_idx] = torch.zeros(
            batch_size, proj_dim, conv_kernel,
            device=device, dtype=dtype,
        )


# ---------------------------------------------------------------------------
# Delta Rule Recurrence: Pure PyTorch Reference
# ---------------------------------------------------------------------------

def gated_delta_rule_recurrence_ref(
    q: torch.Tensor,      # (B, H, L, d_k)
    k: torch.Tensor,      # (B, H, L, d_k)
    v: torch.Tensor,      # (B, H, L, d_v)
    beta: torch.Tensor,   # (B, H, L) or (B, H, L, 1)
    initial_state: Optional[torch.Tensor] = None,  # (B, H, d_v, d_k)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch reference for the gated delta rule recurrence.

    For each head h, the recurrence is:
        S_0 = initial_state (or zeros)
        S_t = (1 - β_t) * S_{t-1} + β_t * v_t ⊗ k_t
        o_t = S_t @ q_t

    Args:
        q: Query tensor (B, H, L, d_k)
        k: Key tensor (B, H, L, d_k)
        v: Value tensor (B, H, L, d_v)
        beta: Forget/update gate (B, H, L) - values in [0, 1]
        initial_state: Optional initial recurrent state (B, H, d_v, d_k)

    Returns:
        output: (B, H, L, d_v)
        final_state: (B, H, d_v, d_k)
    """
    B, H, L, d_k = q.shape
    d_v = v.shape[-1]

    # Ensure beta has right shape
    if beta.dim() == 3:
        beta = beta.unsqueeze(-1)  # (B, H, L, 1)

    # Initialize state
    if initial_state is not None:
        S = initial_state.clone()
    else:
        S = torch.zeros(B, H, d_v, d_k, device=q.device, dtype=q.dtype)

    outputs = []
    for t in range(L):
        q_t = q[:, :, t, :]              # (B, H, d_k)
        k_t = k[:, :, t, :]              # (B, H, d_k)
        v_t = v[:, :, t, :]              # (B, H, d_v)
        beta_t = beta[:, :, t, :]        # (B, H, 1)

        # Gated state update: S = (1 - β) * S + β * v ⊗ k
        # v_t.unsqueeze(-1): (B, H, d_v, 1)
        # k_t.unsqueeze(-2): (B, H, 1, d_k)
        outer_vk = v_t.unsqueeze(-1) * k_t.unsqueeze(-2)  # (B, H, d_v, d_k)
        S = (1.0 - beta_t.unsqueeze(-1)) * S + beta_t.unsqueeze(-1) * outer_vk

        # Output: o = S @ q
        o_t = torch.einsum('bhvk,bhk->bhv', S, q_t)  # (B, H, d_v)
        outputs.append(o_t)

    output = torch.stack(outputs, dim=2)  # (B, H, L, d_v)
    return output, S


def gated_delta_rule_step(
    q: torch.Tensor,      # (B, H, d_k)
    k: torch.Tensor,      # (B, H, d_k)
    v: torch.Tensor,      # (B, H, d_v)
    beta: torch.Tensor,   # (B, H) or (B, H, 1)
    state: torch.Tensor,  # (B, H, d_v, d_k)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single-step gated delta rule for autoregressive generation.

    Args:
        q: Query (B, H, d_k)
        k: Key (B, H, d_k)
        v: Value (B, H, d_v)
        beta: Gate (B, H) or (B, H, 1)
        state: Current recurrent state (B, H, d_v, d_k)

    Returns:
        output: (B, H, d_v)
        new_state: (B, H, d_v, d_k)
    """
    if beta.dim() == 2:
        beta = beta.unsqueeze(-1)  # (B, H, 1)

    # Outer product v ⊗ k
    outer_vk = v.unsqueeze(-1) * k.unsqueeze(-2)  # (B, H, d_v, d_k)

    # Gated state update
    new_state = (1.0 - beta.unsqueeze(-1)) * state + beta.unsqueeze(-1) * outer_vk

    # Output
    output = torch.einsum('bhvk,bhk->bhv', new_state, q)  # (B, H, d_v)

    return output, new_state


# ---------------------------------------------------------------------------
# GatedDeltaNetMixer: Core GatedDeltaNet Layer
# ---------------------------------------------------------------------------

class GatedDeltaNetMixer(nn.Module):
    """
    GatedDeltaNet mixer that replaces self-attention.

    Architecture:
        Input (B, L, D) -> Linear projections -> Q, K, V, β, gate
        Q, K -> short causal conv1d -> SiLU activation
        β -> sigmoid (update/forget gate per head)
        Delta Rule: S_t = (1-β_t)·S_{t-1} + β_t·(v_t ⊗ k_t)
        Output: o_t = gate_t ⊙ (S_t · q_t)
        -> out_proj -> Output (B, L, D)

    Args:
        d_model: Model dimension (hidden size)
        num_heads: Number of attention heads
        key_head_dim: Dimension per key/query head
        value_head_dim: Dimension per value head
        conv_kernel: Short convolution kernel size
        use_output_gate: Whether to use output gating
        layer_idx: Layer index for caching
    """

    def __init__(
        self,
        d_model: int,
        num_heads: Optional[int] = None,
        key_head_dim: int = 128,
        value_head_dim: int = 128,
        conv_kernel: int = 4,
        use_output_gate: bool = True,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.conv_kernel = conv_kernel
        self.use_output_gate = use_output_gate
        self.layer_idx = layer_idx

        # Calculate number of heads
        if num_heads is not None:
            self.num_heads = num_heads
        else:
            self.num_heads = d_model // key_head_dim

        self.q_dim = self.num_heads * key_head_dim
        self.k_dim = self.num_heads * key_head_dim
        self.v_dim = self.num_heads * value_head_dim

        # --- Projections ---
        # Q, K projections
        self.q_proj = nn.Linear(d_model, self.q_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.k_dim, bias=False)
        # V projection
        self.v_proj = nn.Linear(d_model, self.v_dim, bias=False)
        # Beta (forget gate) - one scalar per head
        self.beta_proj = nn.Linear(d_model, self.num_heads, bias=True)

        # Output gate
        if use_output_gate:
            self.gate_proj = nn.Linear(d_model, self.v_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.v_dim, d_model, bias=False)

        # --- Short Causal Convolutions on Q and K ---
        self.q_conv = nn.Conv1d(
            in_channels=self.q_dim,
            out_channels=self.q_dim,
            kernel_size=conv_kernel,
            groups=self.q_dim,
            padding=conv_kernel - 1,
            bias=True,
        )
        self.k_conv = nn.Conv1d(
            in_channels=self.k_dim,
            out_channels=self.k_dim,
            kernel_size=conv_kernel,
            groups=self.k_dim,
            padding=conv_kernel - 1,
            bias=True,
        )

        # --- Layer norm for Q and K (stabilizes training) ---
        self.q_norm = nn.LayerNorm(key_head_dim)
        self.k_norm = nn.LayerNorm(key_head_dim)

        # Initialize beta bias to encourage moderate gating initially
        nn.init.constant_(self.beta_proj.bias, -2.0)  # sigmoid(-2) ≈ 0.12

    def _apply_conv(self, x: torch.Tensor, conv: nn.Conv1d, seq_len: int) -> torch.Tensor:
        """Apply causal convolution: (B, L, D) -> conv1d -> (B, L, D)."""
        x = x.permute(0, 2, 1)  # (B, D, L)
        if HAS_CAUSAL_CONV1D and x.is_cuda:
            x = causal_conv1d_fn(
                x=x,
                weight=conv.weight.squeeze(1),
                bias=conv.bias,
                activation=None,
            )
        else:
            x = conv(x)[:, :, :seq_len]
        x = x.permute(0, 2, 1)  # (B, L, D)
        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[GatedDeltaNetCache] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the GatedDeltaNet mixer.

        Args:
            hidden_states: (B, L, D) input tensor
            cache: Optional cache for autoregressive generation

        Returns:
            output: (B, L, D) output tensor
        """
        B, L, D = hidden_states.shape

        # --- Projections ---
        q = self.q_proj(hidden_states)       # (B, L, q_dim)
        k = self.k_proj(hidden_states)       # (B, L, k_dim)
        v = self.v_proj(hidden_states)       # (B, L, v_dim)
        beta = torch.sigmoid(self.beta_proj(hidden_states))  # (B, L, H)

        if self.use_output_gate:
            gate = F.silu(self.gate_proj(hidden_states))  # (B, L, v_dim)

        # --- Determine if generation step ---
        is_generation_step = (
            cache is not None
            and L == 1
            and self.layer_idx is not None
            and self.layer_idx in cache.recurrent_states
        )

        if is_generation_step:
            # --- Single-step generation with cached state ---
            # Update conv states
            q_conv_state = cache.q_conv_states[self.layer_idx]
            q_conv_state = torch.roll(q_conv_state, shifts=-1, dims=-1)
            q_conv_state[:, :, -1] = q[:, 0, :]  # (B, q_dim)
            cache.q_conv_states[self.layer_idx] = q_conv_state

            k_conv_state = cache.k_conv_states[self.layer_idx]
            k_conv_state = torch.roll(k_conv_state, shifts=-1, dims=-1)
            k_conv_state[:, :, -1] = k[:, 0, :]
            cache.k_conv_states[self.layer_idx] = k_conv_state

            # Apply convolution via dot product with weights
            q_conv_w = self.q_conv.weight.squeeze(1)  # (q_dim, conv_kernel)
            q = (q_conv_state * q_conv_w).sum(dim=-1)
            if self.q_conv.bias is not None:
                q = q + self.q_conv.bias
            q = q.unsqueeze(1)  # (B, 1, q_dim)

            k_conv_w = self.k_conv.weight.squeeze(1)
            k = (k_conv_state * k_conv_w).sum(dim=-1)
            if self.k_conv.bias is not None:
                k = k + self.k_conv.bias
            k = k.unsqueeze(1)  # (B, 1, k_dim)

        else:
            # --- Full sequence: apply causal convolution ---
            q = self._apply_conv(q, self.q_conv, L)
            k = self._apply_conv(k, self.k_conv, L)

            # Initialize conv cache if needed
            if cache is not None and self.layer_idx is not None:
                q_t = q.permute(0, 2, 1)  # (B, q_dim, L)
                q_pad = F.pad(q_t, (self.conv_kernel - q_t.shape[-1], 0))
                cache.q_conv_states[self.layer_idx] = q_pad[:, :, -self.conv_kernel:]

                k_t = k.permute(0, 2, 1)
                k_pad = F.pad(k_t, (self.conv_kernel - k_t.shape[-1], 0))
                cache.k_conv_states[self.layer_idx] = k_pad[:, :, -self.conv_kernel:]

        # --- Activation on Q, K ---
        q = F.silu(q)
        k = F.silu(k)

        # --- Reshape to multi-head ---
        q = q.view(B, -1, self.num_heads, self.key_head_dim)    # (B, L, H, d_k)
        k = k.view(B, -1, self.num_heads, self.key_head_dim)    # (B, L, H, d_k)
        v = v.view(B, -1, self.num_heads, self.value_head_dim)  # (B, L, H, d_v)

        # --- Apply head-wise layer norm ---
        q = self.q_norm(q)
        k = self.k_norm(k)

        # --- Transpose to (B, H, L, d) ---
        q = q.permute(0, 2, 1, 3)  # (B, H, L, d_k)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        beta_heads = beta.permute(0, 2, 1)  # (B, H, L)

        # --- Recurrence ---
        if is_generation_step:
            # Single step
            o, new_state = gated_delta_rule_step(
                q=q.squeeze(2),      # (B, H, d_k)
                k=k.squeeze(2),
                v=v.squeeze(2),
                beta=beta_heads.squeeze(2),  # (B, H)
                state=cache.recurrent_states[self.layer_idx],
            )
            cache.recurrent_states[self.layer_idx] = new_state
            o = o.unsqueeze(2)  # (B, H, 1, d_v)

        elif HAS_FLA_CUDA and q.is_cuda and not q.requires_grad:
            # Use optimized CUDA kernel (inference only for now)
            try:
                o = fused_recurrent_gated_delta_rule(q, k, v, beta_heads)
            except Exception:
                o, _ = gated_delta_rule_recurrence_ref(q, k, v, beta_heads)
        else:
            # Pure PyTorch reference
            initial_state = None
            if cache is not None and self.layer_idx is not None:
                initial_state = cache.recurrent_states.get(self.layer_idx, None)

            o, final_state = gated_delta_rule_recurrence_ref(
                q, k, v, beta_heads, initial_state=initial_state,
            )

            if cache is not None and self.layer_idx is not None:
                cache.recurrent_states[self.layer_idx] = final_state

        # --- Reshape back: (B, H, L, d_v) -> (B, L, H*d_v) ---
        o = o.permute(0, 2, 1, 3).contiguous().view(B, -1, self.v_dim)

        # --- Output gating ---
        if self.use_output_gate:
            o = gate * o

        # --- Output projection ---
        output = self.out_proj(o)  # (B, L, D)

        return output


# ---------------------------------------------------------------------------
# GatedDeltaNetBlock: Drop-in replacement for Qwen Attention
# ---------------------------------------------------------------------------

class GatedDeltaNetBlock(nn.Module):
    """
    A GatedDeltaNet block that matches the forward signature of HuggingFace
    attention layers.

    This serves as a drop-in replacement for Qwen2/Qwen3 Attention modules.
    The forward method accepts all standard attention arguments but only uses
    hidden_states.

    Args:
        config: Model config object (must have `hidden_size`)
        layer_idx: Layer index for caching
        key_head_dim: Dimension per key/query head
        value_head_dim: Dimension per value head
        conv_kernel: Short convolution kernel size
        use_output_gate: Whether to use output gating
    """

    def __init__(
        self,
        config,
        layer_idx: Optional[int] = None,
        key_head_dim: int = 128,
        value_head_dim: int = 128,
        conv_kernel: int = 4,
        use_output_gate: bool = True,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.gated_deltanet = GatedDeltaNetMixer(
            d_model=config.hidden_size,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            conv_kernel=conv_kernel,
            use_output_gate=use_output_gate,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[object] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[object]]:
        """
        Forward pass matching HuggingFace attention layer signature.

        Args:
            hidden_states: (B, L, D) input tensor
            Other args: Ignored (for compatibility)

        Returns:
            Tuple of (output, None, past_key_value)
        """
        cache = past_key_value if isinstance(past_key_value, GatedDeltaNetCache) else None

        output = self.gated_deltanet(hidden_states, cache=cache)

        # New transformers unpacks only 2 values from self_attn: (hidden_states, attn_weights)
        return output, None
