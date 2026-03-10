"""
Mamba (Selective State Space Model) Block Implementation.

This module provides a Mamba SSM layer that serves as a drop-in replacement
for attention layers in transformer models. It implements the selective scan
algorithm from the Mamba paper (Gu & Dao, 2023).

Features:
- Pure PyTorch reference implementation (works on any device)
- Optional CUDA-optimized kernels via mamba-ssm package
- Compatible forward signature with HuggingFace attention layers
- Supports autoregressive generation with recurrent state caching
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.cache_utils import DynamicCache as _DynamicCacheBase
except ImportError:
    _DynamicCacheBase = None

try:
    from einops import rearrange
except ImportError:
    rearrange = None

# Try to import optimized CUDA kernels
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
    HAS_MAMBA_CUDA = True
except ImportError:
    HAS_MAMBA_CUDA = False

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    HAS_CAUSAL_CONV1D = True
except ImportError:
    HAS_CAUSAL_CONV1D = False


# ---------------------------------------------------------------------------
# Helper: einops-free rearrange for common patterns
# ---------------------------------------------------------------------------

def _rearrange_bld_to_bdl(x: torch.Tensor) -> torch.Tensor:
    """(B, L, D) -> (B, D, L)"""
    return x.permute(0, 2, 1)


def _rearrange_bdl_to_bld(x: torch.Tensor) -> torch.Tensor:
    """(B, D, L) -> (B, L, D)"""
    return x.permute(0, 2, 1)


def _rearrange_bl_d_to_bld(x: torch.Tensor, batch: int, length: int) -> torch.Tensor:
    """(B*L, D) -> (B, L, D)"""
    return x.view(batch, length, -1)


# ---------------------------------------------------------------------------
# Mamba Cache (for autoregressive generation)
# ---------------------------------------------------------------------------

@dataclass
class MambaCache:
    """
    Cache for Mamba layers during autoregressive generation.

    Stores both the convolution state and the SSM recurrent state
    for each Mamba layer in the model.
    """
    conv_states: dict = field(default_factory=dict)  # layer_idx -> (B, d_inner, d_conv)
    ssm_states: dict = field(default_factory=dict)   # layer_idx -> (B, d_inner, d_state)

    def init_layer(self, layer_idx: int, batch_size: int, d_inner: int,
                   d_conv: int, d_state: int, device: torch.device, dtype: torch.dtype):
        """Initialize cache for a specific layer."""
        self.conv_states[layer_idx] = torch.zeros(
            batch_size, d_inner, d_conv, device=device, dtype=dtype
        )
        self.ssm_states[layer_idx] = torch.zeros(
            batch_size, d_inner, d_state, device=device, dtype=dtype
        )

    def get_conv_state(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self.conv_states.get(layer_idx, None)

    def get_ssm_state(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self.ssm_states.get(layer_idx, None)


class HybridCache(_DynamicCacheBase if _DynamicCacheBase is not None else object):
    """
    Cache that extends DynamicCache with MambaCache for hybrid models.

    Inherits from DynamicCache so that isinstance(cache, DynamicCache)
    returns True — the Qwen3 backbone treats it as a normal KV cache,
    while Mamba layers can access .mamba_cache for SSM states.
    """

    def __init__(self, mamba_cache: Optional[MambaCache] = None):
        if _DynamicCacheBase is not None:
            super().__init__()
        self.mamba_cache = mamba_cache or MambaCache()


# ---------------------------------------------------------------------------
# Selective Scan: Pure PyTorch Reference Implementation
# ---------------------------------------------------------------------------

def selective_scan_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
) -> torch.Tensor:
    """
    Pure PyTorch reference implementation of the selective scan algorithm.

    Implements the recurrence:
        h_t = exp(Δ_t * A) * h_{t-1} + Δ_t * B_t * x_t
        y_t = C_t^T * h_t + D * x_t

    Args:
        u:     (B, D, L) - input signal
        delta: (B, D, L) - discretization step sizes
        A:     (D, N)    - state transition matrix (continuous)
        B:     (B, N, L) - input-dependent input matrix
        C:     (B, N, L) - input-dependent output matrix
        D:     (D,)      - skip connection parameter
        z:     (B, D, L) - gate signal (output gating with SiLU)
        delta_bias: (D,) - bias added to delta before softplus
        delta_softplus: bool - apply softplus to delta

    Returns:
        y: (B, D, L) - output signal
    """
    batch_size, d_dim, seq_len = u.shape
    n_dim = A.shape[1]

    # Apply delta bias and softplus
    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)
    if delta_softplus:
        delta = F.softplus(delta)

    # Discretize A and compute deltaB * u
    # deltaA: (B, D, L, N)
    deltaA = torch.exp(
        delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2)
    )  # (B, D, L, N) = (B, D, L, 1) * (1, D, 1, N)

    # deltaB_u: (B, D, L, N) = delta * B * u
    # delta: (B, D, L), B: (B, N, L), u: (B, D, L)
    deltaB_u = (
        delta.unsqueeze(-1)                         # (B, D, L, 1)
        * B.unsqueeze(1).permute(0, 1, 3, 2)        # (B, 1, L, N)
        * u.unsqueeze(-1)                            # (B, D, L, 1)
    )  # -> (B, D, L, N) via broadcasting

    # Sequential scan (recurrence)
    h = torch.zeros(batch_size, d_dim, n_dim, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(seq_len):
        h = deltaA[:, :, t] * h + deltaB_u[:, :, t]       # (B, D, N)
        y_t = (h * C[:, :, t].unsqueeze(1)).sum(dim=-1)    # (B, D)
        ys.append(y_t)

    y = torch.stack(ys, dim=-1)  # (B, D, L)

    # Skip connection
    if D is not None:
        y = y + D.unsqueeze(0).unsqueeze(-1) * u

    # Output gating
    if z is not None:
        y = y * F.silu(z)

    return y


# ---------------------------------------------------------------------------
# Selective Scan: Single Step (for autoregressive generation)
# ---------------------------------------------------------------------------

def selective_scan_step(
    x: torch.Tensor,       # (B, D)
    delta: torch.Tensor,   # (B, D)
    A: torch.Tensor,       # (D, N)
    B: torch.Tensor,       # (B, N)
    C: torch.Tensor,       # (B, N)
    D: Optional[torch.Tensor] = None,  # (D,)
    z: Optional[torch.Tensor] = None,  # (B, D)
    ssm_state: Optional[torch.Tensor] = None,  # (B, D, N)
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single step of selective scan for autoregressive generation.

    Returns:
        y: (B, D) - output for this step
        new_ssm_state: (B, D, N) - updated SSM state
    """
    if delta_bias is not None:
        delta = delta + delta_bias
    if delta_softplus:
        delta = F.softplus(delta)

    # Discretize
    dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))  # (B, D, N)
    dB = delta.unsqueeze(-1) * B.unsqueeze(1)              # (B, D, N)

    # State update
    if ssm_state is None:
        ssm_state = torch.zeros_like(dA)
    new_state = dA * ssm_state + dB * x.unsqueeze(-1)  # (B, D, N)

    # Output
    y = (new_state * C.unsqueeze(1)).sum(dim=-1)  # (B, D)

    if D is not None:
        y = y + D * x
    if z is not None:
        y = y * F.silu(z)

    return y, new_state


# ---------------------------------------------------------------------------
# MambaMixer: Core Mamba Block (replaces self-attention)
# ---------------------------------------------------------------------------

class MambaMixer(nn.Module):
    """
    Mamba mixer (Selective State Space Model) that replaces self-attention.

    Architecture:
        Input -> in_proj -> [x, z] split
        x -> conv1d -> silu -> x_proj -> [dt, B, C]
        SSM(x, dt, A, B, C, D) * silu(z) -> out_proj -> Output

    Args:
        d_model:   Model dimension (hidden size)
        d_state:   SSM state dimension (N in the paper)
        d_conv:    Local convolution width
        expand:    Inner dimension expansion factor
        dt_rank:   Rank of the dt projection ("auto" = ceil(d_model/16))
        dt_min:    Minimum value for dt initialization
        dt_max:    Maximum value for dt initialization
        dt_init:   Initialization method for dt ("random" or "constant")
        dt_scale:  Scale factor for dt initialization
        dt_init_floor: Minimum absolute value for dt initialization
        bias:      Use bias in linear projections
        conv_bias: Use bias in convolution
        layer_idx: Layer index (for caching during generation)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.layer_idx = layer_idx

        # --- Input Projection ---
        # Projects to x (for SSM) and z (for gating), both of dim d_inner
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        # --- Causal Convolution ---
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=conv_bias,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # --- SSM Parameter Projection ---
        # Projects from d_inner to dt_rank (for delta) + 2*d_state (for B and C)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        # --- Delta (dt) Projection ---
        # Projects from dt_rank back to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Mamba-style dt projection init
        dt_init_std = (self.dt_rank ** -0.5) * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise ValueError(f"Unsupported dt_init: {dt_init}. Expected 'random' or 'constant'.")

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: x = log(exp(y) - 1)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Mark dt_proj bias so that optimizer can treat it specially
        self.dt_proj.bias._no_weight_decay = True

        # --- State Matrix A ---
        # S4D real initialization: A_n = -n for n = 1, ..., d_state
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # --- Skip Connection D ---
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # --- Output Projection ---
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[MambaCache] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Mamba mixer.

        Args:
            hidden_states: (B, L, D) input tensor
            cache: Optional MambaCache for autoregressive generation

        Returns:
            output: (B, L, D) output tensor
        """
        batch_size, seq_len, _ = hidden_states.shape

        # --- Input Projection ---
        xz = self.in_proj(hidden_states)           # (B, L, 2*d_inner)
        xz = _rearrange_bld_to_bdl(xz)             # (B, 2*d_inner, L)
        x, z = xz.chunk(2, dim=1)                  # each: (B, d_inner, L)

        # --- Convolution ---
        is_generation_step = (cache is not None and seq_len == 1
                              and self.layer_idx is not None
                              and cache.get_conv_state(self.layer_idx) is not None)

        if is_generation_step:
            # Single-step generation: use cached conv state
            conv_state = cache.get_conv_state(self.layer_idx)
            conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
            conv_state[:, :, -1] = x[:, :, 0]
            cache.conv_states[self.layer_idx] = conv_state

            conv_weight = self.conv1d.weight.squeeze(1)  # (d_inner, d_conv)
            x = (conv_state * conv_weight.to(conv_state.dtype)).sum(dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias.to(x.dtype)
            x = x.unsqueeze(-1)  # (B, d_inner, 1)
        else:
            # Full sequence: standard causal conv
            if HAS_CAUSAL_CONV1D and x.is_cuda:
                x = causal_conv1d_fn(
                    x=x,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation="silu",
                )
            else:
                x = self.conv1d(x)[:, :, :seq_len]
                x = F.silu(x)

            # Initialize conv cache if needed
            if cache is not None and self.layer_idx is not None:
                conv_state = F.pad(x, (self.d_conv - x.shape[-1], 0))  # (B, d_inner, d_conv)
                cache.conv_states[self.layer_idx] = conv_state[:, :, -self.d_conv:]

        if not (HAS_CAUSAL_CONV1D and x.is_cuda and not is_generation_step):
            # silu already applied in causal_conv1d_fn
            if is_generation_step:
                x = F.silu(x)
            # For non-CUDA path, silu is applied above after conv1d

        # --- SSM Parameters ---
        # Project x to get delta (dt), B, C
        x_proj_input = _rearrange_bdl_to_bld(x).reshape(batch_size * seq_len, self.d_inner)
        x_dbl = self.x_proj(x_proj_input)  # (B*L, dt_rank + 2*d_state)

        dt, B_param, C_param = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # --- Delta Projection ---
        # dt: (B*L, dt_rank) -> (d_inner, B*L) via weight matrix multiply
        dt = self.dt_proj.weight @ dt.t()  # (d_inner, B*L)
        dt = dt.view(self.d_inner, batch_size, seq_len).permute(1, 0, 2)  # (B, d_inner, L)

        # Reshape B, C: (B*L, d_state) -> (B, d_state, L)
        B_param = B_param.view(batch_size, seq_len, self.d_state).permute(0, 2, 1)
        C_param = C_param.view(batch_size, seq_len, self.d_state).permute(0, 2, 1)

        # --- State Matrix ---
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # --- Selective Scan ---
        if is_generation_step:
            # Single step
            y, new_ssm_state = selective_scan_step(
                x=x.squeeze(-1),                           # (B, d_inner)
                delta=dt.squeeze(-1),                       # (B, d_inner)
                A=A,                                        # (d_inner, d_state)
                B=B_param.squeeze(-1),                      # (B, d_state)
                C=C_param.squeeze(-1),                      # (B, d_state)
                D=self.D.float(),
                z=z.squeeze(-1),                            # (B, d_inner)
                ssm_state=cache.get_ssm_state(self.layer_idx),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            cache.ssm_states[self.layer_idx] = new_ssm_state
            y = y.unsqueeze(1)  # (B, 1, d_inner)
        elif HAS_MAMBA_CUDA and x.is_cuda:
            # CUDA-optimized selective scan
            y = selective_scan_fn(
                x.contiguous(),
                dt.contiguous(),
                A.contiguous(),
                B_param.contiguous(),
                C_param.contiguous(),
                D=self.D.float(),
                z=z.contiguous(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            y = _rearrange_bdl_to_bld(y)  # (B, L, d_inner)
        else:
            # Pure PyTorch fallback
            y = selective_scan_ref(
                u=x,
                delta=dt,
                A=A,
                B=B_param,
                C=C_param,
                D=self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            y = _rearrange_bdl_to_bld(y)  # (B, L, d_inner)

        # --- Output Projection ---
        output = self.out_proj(y.to(self.out_proj.weight.dtype))  # (B, L, d_model)
        return output


# ---------------------------------------------------------------------------
# MambaBlock: Drop-in replacement for Qwen2Attention
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """
    A Mamba block that matches the forward signature of HuggingFace attention layers.

    This serves as a drop-in replacement for Qwen2Attention (or similar attention
    modules in decoder layers). The forward method accepts all the standard
    attention arguments but only uses hidden_states.

    Args:
        config: Model config object (must have `hidden_size` attribute)
        layer_idx: Layer index for caching
        d_state:   SSM state dimension
        d_conv:    Convolution width
        expand:    Expansion factor for inner dimension
    """

    def __init__(
        self,
        config,
        layer_idx: Optional[int] = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.mamba = MambaMixer(
            d_model=config.hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
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
            attention_mask: Ignored (Mamba doesn't use attention masks)
            position_ids: Ignored (Mamba is position-agnostic)
            past_key_value: Can be a MambaCache for generation
            output_attentions: Ignored (always returns None)
            use_cache: Whether to use/return cache
            **kwargs: Absorbed for compatibility

        Returns:
            Tuple of (output, None, past_key_value):
            - output: (B, L, D) tensor
            - attn_weights: Always None
            - past_key_value: The cache object if use_cache=True
        """
        # Use MambaCache if provided via past_key_value
        cache = None
        if isinstance(past_key_value, MambaCache):
            cache = past_key_value
        elif isinstance(past_key_value, HybridCache):
            cache = past_key_value.mamba_cache

        output = self.mamba(hidden_states, cache=cache)

        # New transformers unpacks only 2 values from self_attn: (hidden_states, attn_weights)
        # past_key_value is handled separately by the decoder layer
        return output, None
