# Mamba back - Qwen3-4B Attention-Mamba Model
## <center>In memory of Kobe bryant
![科比大战变形金刚 - Mamba项目插图](./images/mamba.jpg)


> 将 Qwen3-4B 的 75% Attention 层替换为线性复杂度层（Mamba / GatedDeltaNet），通过知识蒸馏恢复语言能力，实现长上下文下的推理加速与显存优化。

## 项目概览


| 模型变体 | 架构描述 | 关键特点 |
|---------|---------|---------|
| **Baseline** Qwen3-4B | 原始 36 层全 Attention | 基准参考，作为 Teacher |
| **Qwen3-4B-Mamba** | 9 层 Attention + 27 层 Mamba (S6 SSM) | 向量态递推，线性推理 |
| **Qwen3-4B-GatedDeltaNet** | 9 层 Attention + 27 层 GatedDeltaNet | 矩阵态递推，关联记忆更强 |

后续可扩展为 **MoE 版本**（Sparse Upcycling）。

### 架构图

```
Original Qwen3-4B (36 layers, all Attention):
  Layer 0:  [Attention] [MLP] [RMSNorm]
  Layer 1:  [Attention] [MLP] [RMSNorm]
  ...

After Surgery (attention_interval=4):
  Layer 0:  [Attention]  [MLP] [RMSNorm]  ← 保留 (每4层1层)
  Layer 1:  [Mamba/GDN]  [MLP] [RMSNorm]  ← 替换
  Layer 2:  [Mamba/GDN]  [MLP] [RMSNorm]  ← 替换
  Layer 3:  [Mamba/GDN]  [MLP] [RMSNorm]  ← 替换
  Layer 4:  [Attention]  [MLP] [RMSNorm]  ← 保留
  ...

MoE Expansion (optional):
  Layer 0:  [Attention]  [MoE-MLP: 8 experts, top-2] [RMSNorm]
  Layer 1:  [Mamba/GDN]  [MoE-MLP: 8 experts, top-2] [RMSNorm]
  ...
```

---

## 实验结果

基于 checkpoint-6000（Phase 2 蒸馏 6000 步），AutoDL A800 GPU 测量。

### 推理速度（`deployment/benchmark_speed.py`）

| 上下文长度 | Prefill Qwen3-4B | Prefill Mamba | Decode Qwen3-4B | Decode Mamba | TTFT Qwen3-4B | TTFT Mamba |
|-----------|-----------------|--------------|-----------------|-------------|--------------|-----------|
| 128       | 4,736 tok/s     | 5,394 tok/s  | 59.8 tok/s      | 57.9 tok/s  | 29.1 ms      | **23.7 ms** |
| 2048      | 24,854 tok/s    | 24,105 tok/s | 57.4 tok/s      | 54.9 tok/s  | 82.4 ms      | 85.0 ms   |
| 8192      | 23,982 tok/s    | **26,572 tok/s** | 46.8 tok/s  | 45.8 tok/s  | 341.6 ms     | **308.3 ms** |
| 16384     | 19,962 tok/s    | **24,554 tok/s** | 35.0 tok/s  | **36.4 tok/s** | 820.8 ms  | **667.3 ms** |
| 32768     | 15,448 tok/s    | **21,733 tok/s** | 18.8 tok/s  | **24.7 tok/s** | 2121.1 ms | **1507.8 ms** |

**长序列优势显著**：上下文 32768 时，Mamba Prefill 快 **+40.7%**，Decode 快 **+31.4%**，TTFT 短 **-28.9%**。

### KV Cache 显存（`deployment/benchmark_memory.py`）

Mamba 混合模型仅 9 层 Attention 有 KV Cache（Baseline 全部 36 层），显存增长率大幅降低：

| 上下文长度 | KV Cache Qwen3-4B | KV Cache Mamba | 节省 |
|-----------|-------------------|---------------|------|
| 512       | 72 MB             | 23 MB         | **68%** |
| 1024      | 171 MB            | 46 MB         | **73%** |
| 2048      | 288 MB            | 77 MB         | **73%** |
| 4096      | 576 MB            | 149 MB        | **74%** |

> KV Cache 随上下文线性增长；Mamba 的 SSM 状态固定为 O(1)（约 8 MB），不随长度增加。

### 语言建模质量（PPL）

在 FineWeb-Edu 验证集（200 条 × 2048 tokens，跳过训练区域前 50000 条）上测量：

| 模型 | PPL | 相对 Baseline |
|------|-----|--------------|
| Qwen3-4B (Baseline) | 14.064 | — |
| Mamba Hybrid (ckpt-1000) | 17.612 | +25.2% |

> checkpoint-1000 为 Phase 2 全参数蒸馏 1000 步的结果。log PPL 差距仅 0.22 nats（2.868 vs 2.644），在 75% Attention 层被替换为随机初始化 Mamba 层的情况下，模型已快速恢复大部分语言建模能力。随训练步数增加，PPL 差距将进一步缩小。

### 中文迁移能力

模型仅在英文 FineWeb-Edu 上蒸馏，测试中文是否保留（分语言 PPL，内嵌学术段落 × 1024 tokens × 30 chunks）：

| 模型 | English PPL | Chinese PPL | CN/EN ratio |
|------|------------|------------|-------------|
| Qwen3-4B (Baseline) | 6.515 | 5.985 | 0.919 |
| Mamba Hybrid (ckpt-1000) | 10.631 | 17.425 | 1.639 |

| 指标 | 数值 | 含义 |
|------|------|------|
| 英文退化 | +63.2% | 训练域内，主要因训练不足 |
| 中文退化 | +191.1% | 训练域外，退化更显著 |
| 中文额外损失 | +78.4% | 中文比英文多退化的部分 |

**中文续写示例（base model 续写，非指令跟随）：**

```
Prompt: 自然语言处理是人工智能领域中最重要的研究方向之一。近年来，随着深度学习技术的快速发展，

Qwen3-4B: 特别是Transformer模型的成功应用，使得NLP取得了显著进展。然而，在实际应用场景下，
传统的基于规则和统计的方法仍然存在诸多局限性...

Mamba Hybrid: 其在NLP、情感分析等领域的应用也逐渐增加，而语义理解（S2V）作为其中的
一个子问题，在许多情况下可以被转化成普通的NLP任务...
```

> Mamba Hybrid 保留了基础中文生成能力（能输出通顺中文句子），但准确性和连贯性有所下降，后半段出现中英混杂。纯英文蒸馏对中文能力有额外损害，加入多语言蒸馏数据可改善。

---

## 项目结构

```
mamba/
├── configs/
│   ├── default_config.yaml          # 主配置文件
│   └── ds_config_zero2.json         # DeepSpeed ZeRO-2 配置
├── src/
│   ├── models/
│   │   ├── mamba_block.py           # Mamba (S6 SSM) 实现
│   │   ├── gated_deltanet_block.py  # GatedDeltaNet 实现
│   │   ├── hybrid_model.py          # 统一混合模型（含 num_logits_to_keep 优化）
│   │   ├── architecture_surgery.py  # 架构手术（替换 Attention 层）
│   │   └── moe_expansion.py         # MoE 扩展模块
│   ├── training/
│   │   ├── distillation.py          # 本地 Teacher 蒸馏训练器
│   │   ├── api_distillation.py      # API 蒸馏（Qwen-Max）
│   │   └── data.py                  # 数据加载工具
│   └── utils/
│       └── helpers.py               # 通用工具函数
├── scripts/
│   ├── convert_model.py             # Step 1: 架构手术
│   ├── train_distill.py             # Step 2: 蒸馏训练
│   ├── evaluate.py                  # Step 3: 评估对比
│   ├── generate_api_data.py         # API 数据生成
│   ├── run_all_experiments.py       # 一键运行全部实验
│   ├── compare_summarization_ppl.py  # 总结对比 + PPL + 中文迁移测试
│   ├── run_lm_eval.py               # 下游任务评估
│   └── ablation_study.py            # 消融实验
├── deployment/                      # 推理基准测试模块
│   ├── README.md
│   ├── requirements.txt
│   ├── compare_all.py               # 一键对比（速度 + 显存）
│   ├── benchmark_memory.py          # 显存基准（KV Cache vs SSM 状态）
│   ├── benchmark_speed.py           # 速度基准（Prefill / Decode / TTFT）
│   ├── plot_results.py              # 结果可视化
│   └── download_models.py           # 模型下载工具
├── tests/
│   └── test_model.py
├── requirements.txt
└── README.md
```

---

## 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt

# 可选：安装 CUDA 加速内核（强烈推荐）
pip install mamba-ssm causal-conv1d   # Mamba selective scan CUDA 内核
pip install fla                        # GatedDeltaNet CUDA 内核
```

### 2. 架构手术（转换模型）

```bash
# 转换为 Mamba 混合模型
python scripts/convert_model.py \
    --base_model Qwen/Qwen3-4B \
    --linear_type mamba \
    --output_dir ./checkpoints/qwen3-4b-mamba-init

# 转换为 GatedDeltaNet 混合模型
python scripts/convert_model.py \
    --base_model Qwen/Qwen3-4B \
    --linear_type gated_deltanet \
    --output_dir ./checkpoints/qwen3-4b-gdn-init

# 仅查看层分配（不加载模型）
python scripts/convert_model.py --base_model Qwen/Qwen3-4B --dry_run
```

### 3. 知识蒸馏训练

#### 方式 A：本地 Teacher（需要 ≥40GB 显存）

```bash
python scripts/train_distill.py \
    --linear_type mamba \
    --base_model Qwen/Qwen3-4B \
    --output_dir ./checkpoints/qwen3-4b-mamba \
    --max_steps 10000 \
    --phase1_steps 2000 \
    --batch_size 2 --grad_accum 16 \
    --max_seq_length 2048
```

#### 方式 B：API 蒸馏（单卡 4090/3090，推荐）

API 蒸馏分为两步，避免 GPU 空等 API 返回：

**第一步：本地电脑生成数据（无需 GPU，只需网络）**

```bash
export DASHSCOPE_API_KEY="your-api-key-here"

# 正式训练（5000 条，约 1-3 小时）
python scripts/generate_api_data.py \
    --num_samples 5000 \
    --api_model qwen-max \
    --output_dir ./cache/api_distill
```

**第二步：上传到 GPU 服务器训练**

```bash
scp -r ./cache/api_distill user@server:/path/to/mamba/cache/api_distill

python scripts/train_distill.py \
    --linear_type mamba \
    --base_model Qwen/Qwen3-4B \
    --output_dir ./checkpoints/qwen3-4b-mamba-api \
    --use_api --cached_data_dir ./cache/api_distill \
    --max_steps 5000 \
    --batch_size 2 --grad_accum 8 \
    --max_seq_length 2048
```

> Mamba 和 GatedDeltaNet 两个变体共用同一份缓存数据，只需生成一次。

#### 方式 C：多卡训练（DeepSpeed ZeRO-2）

```bash
deepspeed --num_gpus 2 scripts/train_distill.py \
    --linear_type mamba \
    --base_model Qwen/Qwen3-4B \
    --output_dir ./checkpoints/qwen3-4b-mamba \
    --deepspeed configs/ds_config_zero2.json \
    --max_steps 10000 \
    --batch_size 2 --grad_accum 8 \
    --max_seq_length 2048
```

### 4. 推理基准测试

```bash
pip install -r deployment/requirements.txt

# 一键对比（速度 + 显存）
python deployment/compare_all.py \
    --baseline_paths /path/to/Qwen3-4B \
    --baseline_labels Qwen3-4B \
    --mamba ./checkpoints/qwen3-4b-mamba/final \
    --device cuda \
    --output ./deployment/results

# 仅速度
python deployment/benchmark_speed.py \
    --baseline_paths /path/to/Qwen3-4B \
    --baseline_labels Qwen3-4B \
    --mamba ./checkpoints/qwen3-4b-mamba/final \
    --context_lengths 128,2048,8192,16384,32768

# 仅显存
python deployment/benchmark_memory.py \
    --baseline_paths /path/to/Qwen3-4B \
    --baseline_labels Qwen3-4B \
    --mamba ./checkpoints/qwen3-4b-mamba/final \
    --context_lengths 128,256,512,1024,2048,4096

# 绘制图表
python deployment/plot_results.py \
    --memory ./deployment/results/memory.json \
    --speed  ./deployment/results/speed.json \
    --output ./deployment/results/
```

### 5. 评估对比

```bash
# PPL 测试（本地 FineEdu 数据）
python scripts/compare_summarization_ppl.py ppl \
    --mamba ./checkpoints/qwen3-4b-mamba/final \
    --qwen3 /path/to/Qwen3-4B \
    --data_path /path/to/fineweb-edu-cache \
    --skip_first 50000 --num_samples 200

# 中文迁移能力测试（续写 + 分语言 PPL，无需数据集）
python scripts/compare_summarization_ppl.py transfer \
    --mamba ./checkpoints/qwen3-4b-mamba/final \
    --qwen3 /path/to/Qwen3-4B

# 综合评估
python scripts/evaluate.py --compare_all \
    --baseline Qwen/Qwen3-4B \
    --mamba_path ./checkpoints/qwen3-4b-mamba/final \
    --gdn_path ./checkpoints/qwen3-4b-gdn/final \
    --output_dir ./eval_results
```

### 6. MoE 扩展（可选）

```bash
python scripts/run_all_experiments.py --mode moe \
    --mamba_model_path ./checkpoints/qwen3-4b-mamba/final \
    --moe_num_experts 8 --moe_top_k 2
```

---

## 实验设置

| 参数 | 设置 |
|------|------|
| 基础模型 | Qwen3-4B（36 层，hidden=3584） |
| Attention 保留比 | 25%（每 4 层保留 1 层，共 9 层） |
| 线性层 | Mamba（d_state=16, expand=1.5）/ GatedDeltaNet（head_dim=128） |
| 蒸馏温度 | T=2.0 |
| 蒸馏损失权重 | α_KD=0.5, α_CE=0.5 |
| 训练数据 | FineWeb-Edu（10B token 子集） |
| 序列长度 | 2048 |
| 有效 batch size | 32（batch=2 × grad_accum=16） |
| Phase 1（线性层 only） | 2000 steps，lr=1e-3 |
| Phase 2（全参数蒸馏） | 8000 steps，lr=5e-4（cosine decay） |
| 精度 | bf16 混合精度 |
| 总训练 token 量 | ~660M tokens |

---

## 蒸馏策略

### Phase 1：线性层 Warmup

- **目的**：新初始化的线性层是随机权重，先快速适应上下文
- **冻结**：Embedding、MLP、Attention、LM Head 全部冻结
- **训练**：仅 Mamba / GatedDeltaNet 层可训练
- **学习率**：1e-3

### Phase 2：全参数蒸馏

- **解冻**：所有参数可训练
- **损失函数**：
  ```
  L = 0.5 × KL(student || teacher, T=2.0) + 0.5 × CE(student, labels)
  ```
- **学习率**：5e-4（cosine decay）

### API 蒸馏（两阶段工作流）

```
┌──────────────────────────┐  scp/rsync  ┌──────────────────────────┐
│  本地电脑（无需 GPU）      │ ──────────→ │  GPU 服务器               │
│                          │             │                          │
│  generate_api_data.py    │             │  train_distill.py        │
│  → ./cache/api_distill/  │             │  --cached_data_dir ./cache│
│  耗时：1-3 小时（纯网络）  │             │  耗时：8-12 小时（纯 GPU） │
└──────────────────────────┘             └──────────────────────────┘
```

## 技术说明

### Mamba vs GatedDeltaNet

| 特性 | Mamba (S6 SSM) | GatedDeltaNet |
|------|---------------|---------------|
| **递推状态** | 向量 h ∈ R^{d_state} | 矩阵 S ∈ R^{d_v × d_k} |
| **状态大小/层** | ~0.3 MB | ~8 MB |
| **更新规则** | h = exp(ΔA)·h + ΔB·x | S = (1-β)·S + β·v⊗k |
| **关联记忆** | 弱（向量压缩） | 强（矩阵键值存储） |
| **CUDA 内核** | `mamba-ssm` 包 | `fla` 包 |

### 与 Qwen3.5 的对比

Qwen3.5 使用了相同的混合架构思路（25% Attention + 75% GatedDeltaNet），本项目在 4B 规模上进行同等设计的验证：


### num_logits_to_keep 优化

`hybrid_model.py` 实现了 `num_logits_to_keep` 参数（对齐 HuggingFace 新版接口）：generate() 推理时只对最后 1 个位置计算 lm_head，避免将完整 `(B, L, vocab_size)` 张量转为 fp32（L=4096 时节省约 3.4 GB 峰值显存）。训练时该参数为 0，行为不变。

### 关键注意事项

1. **`tie_word_embeddings=True`**：Qwen3 的 `lm_head` 和 `embed_tokens` 共享权重，架构手术后代码自动重新绑定。
2. **transformers 版本**：需要 `transformers>=4.51.0` 支持 Qwen3 模型。
3. **3090 不支持原生 bf16**：需在 config 中设置 `bf16: false, fp16: true`。
4. **mamba-ssm CUDA 内核**：正确安装后，selective scan 峰值显存从 ~2560 MB 降至 ~41 MB（L=4096）。


## MoE 扩展

基于 **Sparse Upcycling** 方法：

1. 将训练好的 MLP 权重复制为 N 个 Expert
2. 对每个 Expert 添加小噪声（scale=0.01）增加多样性
3. 保留一个不加噪声的"共享专家"始终激活
4. 添加可学习的 Top-K Router 网络
5. 冻结其他参数，仅训练 Router + Expert（~1000-2000 steps）

| 模型 | 总参数 | 激活参数/token | PPL 变化 |
|------|-------|--------------|---------|
| Mamba Dense | ~4B | ~4B | 基准 |
| Mamba MoE（8E，top-2） | ~16B | ~5B | PPL ↓ 5-10% |

---

## 引用

- **Mamba**: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
- **GatedDeltaNet**: Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule", 2024
- **Qwen3**: Qwen Team, "Qwen3 Technical Report", 2025
- **Sparse Upcycling**: Komatsuzaki et al., "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints", 2023

## License

Research use only. Base model weights subject to [Qwen License](https://huggingface.co/Qwen/Qwen3-4B/blob/main/LICENSE).
