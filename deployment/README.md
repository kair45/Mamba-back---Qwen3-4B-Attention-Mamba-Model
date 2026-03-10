# Edge Deployment Benchmark: Qwen3-4B-Mamba vs Qwen3-4B

This module benchmarks the memory and speed advantages of the **Mamba hybrid** model
over the original Transformer (Qwen3-4B) in an edge/mobile deployment scenario.

## Why Mamba wins on edge devices

| Aspect | Transformer (Qwen3-4B) | Mamba Hybrid |
|--------|----------------------|--------------|
| KV cache memory | **O(n)** — grows with context length | **O(1)** — fixed SSM state |
| Decode memory at 4096 ctx | ~2-4× baseline | ~constant |
| Inference mode | Causal attention (quadratic) | Linear recurrence (linear) |
| Model size | ~8GB (FP16) | ~8GB (FP16, same arch size) |

> The longer the conversation, the bigger Mamba's advantage.

---

## Quick Start

### Install dependencies
```bash
pip install -r deployment/requirements.txt
```

### Run full comparison (CPU, simulates mobile)
```bash
python deployment/compare_all.py \
    --transformer /root/autodl-tmp/models/Qwen3-4B \
    --mamba ./checkpoints/qwen3-4b-mamba-phase2/final \
    --device cpu \
    --context_lengths 128,256,512,1024,2048,4096 \
    --new_tokens 64 \
    --output ./deployment/results
```

### Run individual benchmarks
```bash
# Memory only
python deployment/benchmark_memory.py \
    --transformer /root/autodl-tmp/models/Qwen3-4B \
    --mamba ./checkpoints/qwen3-4b-mamba-phase2/final \
    --device cpu

# Speed only
python deployment/benchmark_speed.py \
    --transformer /root/autodl-tmp/models/Qwen3-4B \
    --mamba ./checkpoints/qwen3-4b-mamba-phase2/final \
    --device cpu

# Plot from saved JSON
python deployment/plot_results.py \
    --memory ./deployment/results/memory.json \
    --speed  ./deployment/results/speed.json \
    --output ./deployment/results/
```

---

## Output Files

```
deployment/results/
    memory.json             # Raw memory measurements
    speed.json              # Raw speed measurements
    memory_comparison.png   # Memory vs context length + reduction %
    speed_comparison.png    # Decode speed + speedup chart
    summary.png             # 2×2 combined figure (for README/slides)
```

---

## Metrics Explained

| Metric | Description |
|--------|-------------|
| **Peak Memory Delta (MB)** | Memory increase from baseline during inference; isolates KV cache growth |
| **Decode Speed (tok/s)** | Tokens generated per second during autoregressive generation |
| **Speedup (×)** | Mamba decode speed / Transformer decode speed at same context length |
| **Memory Reduction (%)** | How much less memory Mamba uses at each context length |

---

## Expected Results (reference)

Memory advantage grows significantly with context length because:
- Transformer KV cache: `2 × layers × kv_heads × head_dim × seq_len × 2 bytes`
  For Qwen3-4B: `2 × 32 × 8 × 128 × seq_len × 2 ≈ 131KB per token`
- Mamba SSM state: fixed `d_state × d_inner` regardless of sequence length

At 4096 tokens, Transformer KV cache alone ≈ **512 MB**; Mamba's recurrent state stays constant.

---

## Training Pipeline (for context)

```
Qwen3-4B (teacher)
       │
       │  Knowledge Distillation
       │  (Phase 1: Mamba-only warmup → Phase 2: Full distillation)
       ▼
Qwen3-4B-Mamba (student)
  - 75% Attention layers → replaced with Mamba SSM
  - 25% Attention layers → kept (hybrid)
  - Same vocab, same tokenizer
```
