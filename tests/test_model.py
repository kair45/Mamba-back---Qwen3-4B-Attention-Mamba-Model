#!/usr/bin/env python3
"""
Unit tests for the Qwen-Mamba Hybrid model components.

Tests:
1. Mamba selective scan (reference implementation)
2. MambaMixer forward pass
3. MambaBlock forward pass (attention-compatible interface)
4. Architecture surgery (layer replacement)
5. Distillation loss computation
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn


class TestSelectiveScan(unittest.TestCase):
    """Test the selective scan reference implementation."""

    def test_selective_scan_basic(self):
        """Test basic selective scan computation."""
        from src.models.mamba_block import selective_scan_ref

        B, D, L, N = 2, 4, 8, 3
        u = torch.randn(B, D, L)
        delta = torch.randn(B, D, L)
        A = -torch.rand(D, N)
        B_param = torch.randn(B, N, L)
        C_param = torch.randn(B, N, L)
        D_param = torch.randn(D)

        y = selective_scan_ref(
            u, delta, A, B_param, C_param,
            D=D_param, delta_softplus=True,
        )

        self.assertEqual(y.shape, (B, D, L))
        self.assertFalse(torch.isnan(y).any())

    def test_selective_scan_with_gate(self):
        """Test selective scan with gating (z parameter)."""
        from src.models.mamba_block import selective_scan_ref

        B, D, L, N = 2, 4, 8, 3
        u = torch.randn(B, D, L)
        delta = torch.randn(B, D, L)
        A = -torch.rand(D, N)
        B_param = torch.randn(B, N, L)
        C_param = torch.randn(B, N, L)
        z = torch.randn(B, D, L)

        y = selective_scan_ref(
            u, delta, A, B_param, C_param,
            z=z, delta_softplus=True,
        )

        self.assertEqual(y.shape, (B, D, L))
        self.assertFalse(torch.isnan(y).any())

    def test_selective_scan_step(self):
        """Test single-step selective scan."""
        from src.models.mamba_block import selective_scan_step

        B, D, N = 2, 4, 3
        x = torch.randn(B, D)
        delta = torch.randn(B, D)
        A = -torch.rand(D, N)
        B_param = torch.randn(B, N)
        C_param = torch.randn(B, N)
        D_param = torch.randn(D)

        y, state = selective_scan_step(
            x, delta, A, B_param, C_param,
            D=D_param, delta_softplus=True,
        )

        self.assertEqual(y.shape, (B, D))
        self.assertEqual(state.shape, (B, D, N))
        self.assertFalse(torch.isnan(y).any())


class TestMambaMixer(unittest.TestCase):
    """Test the MambaMixer module."""

    def setUp(self):
        self.d_model = 64
        self.d_state = 8
        self.d_conv = 4
        self.expand = 2
        self.batch_size = 2
        self.seq_len = 16

    def test_forward_shape(self):
        """Test output shape of MambaMixer."""
        from src.models.mamba_block import MambaMixer

        mixer = MambaMixer(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        out = mixer(x)

        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertFalse(torch.isnan(out).any())

    def test_forward_different_seq_lengths(self):
        """Test MambaMixer with different sequence lengths."""
        from src.models.mamba_block import MambaMixer

        mixer = MambaMixer(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )

        for seq_len in [1, 4, 16, 64]:
            x = torch.randn(self.batch_size, seq_len, self.d_model)
            out = mixer(x)
            self.assertEqual(out.shape, (self.batch_size, seq_len, self.d_model))


class TestMambaBlock(unittest.TestCase):
    """Test the MambaBlock (attention-compatible wrapper)."""

    def test_forward_signature(self):
        """Test that MambaBlock has attention-compatible forward signature."""
        from src.models.mamba_block import MambaBlock
        from types import SimpleNamespace

        config = SimpleNamespace(hidden_size=64)
        block = MambaBlock(config, layer_idx=0, d_state=8, d_conv=4, expand=2)

        x = torch.randn(2, 16, 64)
        attention_mask = torch.ones(2, 1, 1, 16)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)

        output, attn_weights, past_kv = block(
            hidden_states=x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=False,
            use_cache=False,
        )

        self.assertEqual(output.shape, (2, 16, 64))
        self.assertIsNone(attn_weights)

    def test_gradient_flow(self):
        """Test that gradients flow through MambaBlock."""
        from src.models.mamba_block import MambaBlock
        from types import SimpleNamespace

        config = SimpleNamespace(hidden_size=64)
        block = MambaBlock(config, layer_idx=0)

        x = torch.randn(2, 16, 64, requires_grad=True)
        output, _, _ = block(hidden_states=x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

        # Check that block parameters also have gradients
        for name, param in block.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")


class TestHybridConfig(unittest.TestCase):
    """Test the hybrid model configuration."""

    def test_layer_assignment(self):
        """Test that layers are correctly assigned as attention vs mamba."""
        from src.models.hybrid_model import QwenMambaHybridConfig

        config = QwenMambaHybridConfig(
            num_hidden_layers=12,
            attention_interval=4,
        )

        # With interval=4, layers 0, 4, 8 should be attention
        self.assertEqual(config.attention_layers, [0, 4, 8])
        self.assertEqual(config.mamba_layers, [1, 2, 3, 5, 6, 7, 9, 10, 11])
        self.assertTrue(config.is_attention_layer(0))
        self.assertTrue(config.is_mamba_layer(1))
        self.assertFalse(config.is_attention_layer(1))

    def test_explicit_attention_layers(self):
        """Test explicit attention layer specification."""
        from src.models.hybrid_model import QwenMambaHybridConfig

        config = QwenMambaHybridConfig(
            num_hidden_layers=12,
            attention_layers=[0, 3, 6, 9],
        )

        self.assertEqual(config.attention_layers, [0, 3, 6, 9])
        self.assertEqual(len(config.mamba_layers), 8)


class TestDistillationLoss(unittest.TestCase):
    """Test the distillation loss function."""

    def test_loss_computation(self):
        """Test that distillation loss computes correctly."""
        from src.training.distillation import DistillationLoss, DistillationConfig

        config = DistillationConfig(
            temperature=2.0,
            alpha_kd=0.5,
            alpha_ce=0.5,
        )
        criterion = DistillationLoss(config)

        B, L, V = 2, 16, 100
        student_logits = torch.randn(B, L, V)
        teacher_logits = torch.randn(B, L, V)
        labels = torch.randint(0, V, (B, L))

        loss, loss_dict = criterion(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
        )

        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0)
        self.assertIn("total_loss", loss_dict)
        self.assertIn("kd_loss", loss_dict)
        self.assertIn("ce_loss", loss_dict)

    def test_loss_gradient(self):
        """Test that loss is differentiable."""
        from src.training.distillation import DistillationLoss, DistillationConfig

        config = DistillationConfig(temperature=2.0, alpha_kd=0.5, alpha_ce=0.5)
        criterion = DistillationLoss(config)

        student_logits = torch.randn(2, 8, 50, requires_grad=True)
        teacher_logits = torch.randn(2, 8, 50)
        labels = torch.randint(0, 50, (2, 8))

        loss, _ = criterion(student_logits, teacher_logits, labels)
        loss.backward()

        self.assertIsNotNone(student_logits.grad)


class TestLayerAssignment(unittest.TestCase):
    """Test the layer assignment logic."""

    def test_get_attention_layer_indices(self):
        """Test attention layer index computation."""
        from src.models.architecture_surgery import get_attention_layer_indices

        # Interval of 4 with 36 layers
        attn, mamba = get_attention_layer_indices(36, 4)
        self.assertEqual(attn, [0, 4, 8, 12, 16, 20, 24, 28, 32])
        self.assertEqual(len(mamba), 27)
        self.assertEqual(len(attn) + len(mamba), 36)

        # Interval of 2 with 12 layers
        attn, mamba = get_attention_layer_indices(12, 2)
        self.assertEqual(attn, [0, 2, 4, 6, 8, 10])
        self.assertEqual(len(mamba), 6)

        # Explicit layers
        attn, mamba = get_attention_layer_indices(12, attention_layers=[0, 5, 11])
        self.assertEqual(attn, [0, 5, 11])
        self.assertEqual(len(mamba), 9)


class TestEndToEnd(unittest.TestCase):
    """End-to-end test with a small model."""

    def test_mamba_replaces_attention_in_simple_model(self):
        """Test replacing attention with Mamba in a simple transformer."""
        from src.models.mamba_block import MambaBlock
        from types import SimpleNamespace

        # Create a simple "transformer" layer
        config = SimpleNamespace(hidden_size=32)

        class SimpleLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Linear(32, 32)  # Placeholder
                self.mlp = nn.Linear(32, 32)
                self.norm = nn.LayerNorm(32)

            def forward(self, x):
                # Simulate: x + attn(norm(x))
                residual = x
                x = self.norm(x)
                if isinstance(self.self_attn, MambaBlock):
                    x, _, _ = self.self_attn(hidden_states=x)
                else:
                    x = self.self_attn(x)
                x = residual + x
                return x

        layer = SimpleLayer()
        x = torch.randn(2, 8, 32)

        # Before surgery
        out1 = layer(x)
        self.assertEqual(out1.shape, (2, 8, 32))

        # Replace attention with Mamba
        layer.self_attn = MambaBlock(config, layer_idx=0, d_state=4, d_conv=2, expand=2)

        # After surgery
        out2 = layer(x)
        self.assertEqual(out2.shape, (2, 8, 32))
        self.assertFalse(torch.isnan(out2).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
