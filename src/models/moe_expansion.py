"""
MoE (Mixture of Experts) Expansion Module.

Converts a trained dense hybrid model's MLP layers into MoE layers,
following the "Sparse Upcycling" approach (Komatsuzaki et al., 2023).

The basic idea:
1. Take the trained MLP weights as "seed" expert
2. Create N copies with small random perturbations for diversity
3. Add a gating/router network
4. Optionally continue training (only router + experts) with load-balancing loss

This enables scaling the model's knowledge capacity without increasing
per-token compute cost.
"""

import copy
import logging
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TopKRouter(nn.Module):
    """
    Top-K expert routing network.

    Routes each token to the top-K experts based on a learned gating function.
    Includes auxiliary load-balancing loss to encourage uniform expert utilization.

    Args:
        hidden_size: Input dimension
        num_experts: Total number of experts
        top_k: Number of experts to activate per token
        noise_std: Standard deviation of noise added during training (for exploration)
        aux_loss_coef: Coefficient for the load-balancing auxiliary loss
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.01,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.aux_loss_coef = aux_loss_coef

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Route tokens to experts.

        Args:
            hidden_states: (B*L, D) flattened token representations

        Returns:
            expert_weights: (B*L, top_k) normalized weights for selected experts
            expert_indices: (B*L, top_k) indices of selected experts
            aux_loss: Load-balancing auxiliary loss (scalar)
        """
        # Compute gating scores
        logits = self.gate(hidden_states)  # (B*L, num_experts)

        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Top-K selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # (B*L, top_k)

        # Auxiliary load-balancing loss
        aux_loss = self._compute_aux_loss(logits, top_k_indices)

        return top_k_weights, top_k_indices, aux_loss

    def _compute_aux_loss(
        self, logits: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Switch Transformer style load-balancing loss.

        Encourages uniform distribution of tokens across experts.
        """
        num_tokens = logits.shape[0]

        # Fraction of tokens routed to each expert
        # Create one-hot and sum
        one_hot = F.one_hot(indices, self.num_experts).float()  # (B*L, top_k, E)
        tokens_per_expert = one_hot.sum(dim=1).sum(dim=0)  # (E,)
        fraction_tokens = tokens_per_expert / max(num_tokens, 1)

        # Mean routing probability for each expert
        routing_probs = F.softmax(logits, dim=-1)  # (B*L, E)
        mean_routing_prob = routing_probs.mean(dim=0)  # (E,)

        # Aux loss: dot product of fraction and mean prob (uniform = 1/E * 1/E * E = 1/E)
        aux_loss = (fraction_tokens * mean_routing_prob).sum() * self.num_experts
        return self.aux_loss_coef * aux_loss


class MoEMLP(nn.Module):
    """
    Mixture of Experts MLP layer.

    Replaces a single dense MLP with N expert MLPs and a router.
    Each expert has the same architecture as the original MLP.

    Args:
        hidden_size: Model hidden dimension
        intermediate_size: MLP intermediate dimension (per expert)
        num_experts: Number of experts
        top_k: Number of active experts per token
        shared_expert: Whether to include a shared expert (always active)
        aux_loss_coef: Load-balancing loss coefficient
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        shared_expert: bool = True,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.has_shared_expert = shared_expert

        # Router
        self.router = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_coef=aux_loss_coef,
        )

        # Expert MLPs (each is a Qwen-style MLP: gate_proj, up_proj, down_proj)
        self.experts = nn.ModuleList([
            QwenMLP(hidden_size, intermediate_size) for _ in range(num_experts)
        ])

        # Optional shared expert (always active, adds to all tokens)
        if shared_expert:
            self.shared_expert = QwenMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE MLP.

        Args:
            hidden_states: (B, L, D) input tensor

        Returns:
            output: (B, L, D) tensor
            aux_loss: Scalar load-balancing loss
        """
        B, L, D = hidden_states.shape
        x_flat = hidden_states.view(-1, D)  # (B*L, D)

        # Route tokens
        expert_weights, expert_indices, aux_loss = self.router(x_flat)

        # Compute expert outputs (token-by-token routing)
        output = torch.zeros_like(x_flat)

        for i in range(self.num_experts):
            # Find tokens assigned to expert i
            # expert_indices: (B*L, top_k), check which positions have expert i
            mask = (expert_indices == i).any(dim=-1)  # (B*L,)
            if not mask.any():
                continue

            # Get the routing weight for expert i
            # For each position that routes to expert i, find its weight
            pos_mask = (expert_indices == i)  # (B*L, top_k)
            weights_for_i = (expert_weights * pos_mask.float()).sum(dim=-1)  # (B*L,)

            # Expert forward
            expert_input = x_flat[mask]  # (num_tokens_for_i, D)
            expert_output = self.experts[i](expert_input)  # (num_tokens_for_i, D)

            # Weighted contribution
            output[mask] += expert_output * weights_for_i[mask].unsqueeze(-1)

        # Add shared expert contribution
        if self.has_shared_expert:
            shared_out = self.shared_expert(x_flat)
            output = output + shared_out

        output = output.view(B, L, D)
        return output, aux_loss


class QwenMLP(nn.Module):
    """Single Qwen-style MLP: gate_proj + up_proj -> SiLU * linear -> down_proj."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# MoE Expansion Functions
# ---------------------------------------------------------------------------

def expand_mlp_to_moe(
    model: nn.Module,
    num_experts: int = 8,
    top_k: int = 2,
    shared_expert: bool = True,
    noise_scale: float = 0.01,
    target_layers: Optional[List[int]] = None,
    aux_loss_coef: float = 0.01,
) -> nn.Module:
    """
    Convert dense MLP layers in a model to MoE layers (Sparse Upcycling).

    Steps for each target MLP layer:
    1. Extract the original MLP weights
    2. Create N expert copies with small perturbations
    3. Replace the original MLP with MoE version
    4. Optionally keep original as shared expert

    Args:
        model: The hybrid model to expand
        num_experts: Number of experts per MoE layer
        top_k: Number of active experts per token
        shared_expert: Include a shared (always-active) expert
        noise_scale: Scale of perturbation noise for expert diversity
        target_layers: Specific layer indices to convert (None = all)
        aux_loss_coef: Load-balancing loss coefficient

    Returns:
        Modified model with MoE layers
    """
    # Find decoder layers
    layers = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        raise ValueError("Cannot find decoder layers in model.")

    if target_layers is None:
        target_layers = list(range(len(layers)))

    num_converted = 0
    for idx in target_layers:
        if idx >= len(layers):
            continue

        layer = layers[idx]

        # Find the MLP module (Qwen: layer.mlp)
        if not hasattr(layer, 'mlp'):
            logger.warning(f"Layer {idx} has no MLP module, skipping.")
            continue

        old_mlp = layer.mlp

        # Extract dimensions from old MLP
        hidden_size = None
        intermediate_size = None

        if hasattr(old_mlp, 'gate_proj'):
            hidden_size = old_mlp.gate_proj.in_features
            intermediate_size = old_mlp.gate_proj.out_features
        elif hasattr(old_mlp, 'up_proj'):
            hidden_size = old_mlp.up_proj.in_features
            intermediate_size = old_mlp.up_proj.out_features
        else:
            logger.warning(f"Cannot determine MLP dimensions for layer {idx}, skipping.")
            continue

        # Determine device and dtype
        try:
            param = next(old_mlp.parameters())
            device = param.device
            dtype = param.dtype
        except StopIteration:
            device = torch.device("cpu")
            dtype = torch.float32

        # Create MoE MLP
        moe_mlp = MoEMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            shared_expert=shared_expert,
            aux_loss_coef=aux_loss_coef,
        ).to(device=device, dtype=dtype)

        # Copy original weights to experts with perturbation
        with torch.no_grad():
            for expert in moe_mlp.experts:
                _copy_mlp_weights(old_mlp, expert, noise_scale)

            # Shared expert gets exact copy (no noise)
            if shared_expert:
                _copy_mlp_weights(old_mlp, moe_mlp.shared_expert, noise_scale=0.0)

        # Replace MLP with MoE
        layer.mlp = moe_mlp
        num_converted += 1

        logger.info(
            f"Layer {idx}: Converted MLP to MoE "
            f"({num_experts} experts, top-{top_k}, "
            f"shared={'yes' if shared_expert else 'no'})"
        )

    # Clean up
    del old_mlp
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Converted {num_converted} MLP layers to MoE.")
    _print_moe_summary(model, num_experts, top_k)

    return model


def _copy_mlp_weights(
    source_mlp: nn.Module,
    target_mlp: nn.Module,
    noise_scale: float = 0.01,
):
    """Copy MLP weights from source to target with optional noise perturbation."""
    for (src_name, src_param), (tgt_name, tgt_param) in zip(
        source_mlp.named_parameters(), target_mlp.named_parameters()
    ):
        tgt_param.data.copy_(src_param.data)
        if noise_scale > 0:
            noise = torch.randn_like(tgt_param.data) * noise_scale * src_param.data.abs().mean()
            tgt_param.data.add_(noise)


def _print_moe_summary(model: nn.Module, num_experts: int, top_k: int):
    """Print summary after MoE expansion."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count MoE parameters
    moe_params = 0
    for name, param in model.named_parameters():
        if "expert" in name.lower() or "router" in name.lower():
            moe_params += param.numel()

    # Estimate active parameters per token
    # Rough: total - moe_params + moe_params * (top_k / num_experts) + shared
    moe_active_ratio = top_k / max(num_experts, 1)

    print("\n" + "=" * 70)
    print("  MoE EXPANSION SUMMARY")
    print("=" * 70)
    print(f"  Total parameters:      {total_params:>15,}")
    print(f"  MoE parameters:        {moe_params:>15,}")
    print(f"  Active params/token:   ~{int(total_params - moe_params + moe_params * moe_active_ratio):>14,}")
    print(f"  Experts per layer:     {num_experts}")
    print(f"  Active experts/token:  {top_k}")
    print(f"  Expansion ratio:       {total_params / max(total_params - moe_params + moe_params / num_experts, 1):.1f}x")
    print("=" * 70 + "\n")


def freeze_for_moe_training(model: nn.Module):
    """
    Freeze everything except MoE components for MoE fine-tuning.

    Only the router and expert parameters remain trainable.
    This is used after sparse upcycling to train the routing.
    """
    # Freeze all
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze MoE components
    moe_count = 0
    for name, param in model.named_parameters():
        if "expert" in name.lower() or "router" in name.lower() or "gate" in name.lower():
            param.requires_grad = True
            moe_count += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Froze non-MoE parameters. Trainable: {trainable:,}/{total:,} "
        f"({trainable/total*100:.1f}%)"
    )
