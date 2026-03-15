#!/usr/bin/env python3
"""
长文档总结能力对比 + FineEdu PPL 测试

两个功能合一：
  1. summarization  -- 英/中长文档总结对比，记录显存与速度
  2. ppl            -- 在 FineEdu 验证集上测 Qwen3-4B vs Mamba-Hybrid PPL

用法：
    # 总结对比
    python scripts/compare_summarization_ppl.py summarization \
        --mamba  ./checkpoints/qwen3-4b-mamba-phase2-v3/checkpoint-1000 \
        --qwen3  /root/autodl-tmp/models/Qwen3-4B

    # PPL 对比
    python scripts/compare_summarization_ppl.py ppl \
        --mamba  ./checkpoints/qwen3-4b-mamba-phase2-v3/checkpoint-1000 \
        --qwen3  /root/autodl-tmp/models/Qwen3-4B \
        --num_samples 200 \
        --seq_len 2048
"""

import argparse
import gc
import math
import os
import sys
import time
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.hybrid_model import QwenHybridForCausalLM

# ─────────────────────────────────────────────────────────────────────────────
# 内嵌长文本样本（避免依赖外部文件）
# ─────────────────────────────────────────────────────────────────────────────

ENGLISH_PASSAGE = """\
Chapter 3: Neural Networks and Deep Learning

3.1 Introduction to Artificial Neural Networks

Artificial neural networks (ANNs) are computational systems inspired by the biological neural networks that constitute animal brains. These systems learn to perform tasks by considering examples, generally without being programmed with task-specific rules. An ANN is based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain. Each connection, like the synapses in a biological brain, can transmit a signal to other neurons. An artificial neuron receives a signal, then processes it, and can signal neurons connected to it. The signal at a connection between artificial neurons is a real number, and the output of each neuron is computed by some non-linear function of the sum of its inputs. The connections between neurons are called edges. Neurons and edges typically have a weight that adjusts as learning proceeds. The weight increases or decreases the strength of the signal at a connection. Neurons may have a threshold such that a signal is sent only if the aggregate signal crosses that threshold.

Typically, neurons are aggregated into layers. Different layers may perform different transformations on their inputs. Signals travel from the first layer (the input layer) to the last layer (the output layer), possibly after traversing the inner layers multiple times.

3.2 The Multilayer Perceptron

The multilayer perceptron (MLP) is a fully connected class of feedforward artificial neural network. A fully connected neural network consists of at least three layers of nodes: an input layer, one or more hidden layers, and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training. Its multiple layers and non-linear activation distinguish it from a linear perceptron. It can distinguish data that is not linearly separable.

Backpropagation is a widely used algorithm for training feedforward neural networks. Generalizations of backpropagation exist for other artificial neural networks (ANNs), and for functions generally – a class of algorithms referred to generically as "backpropagation". In fitting a neural network, backpropagation computes the gradient of the loss function with respect to the weights of the network for a single input–output example, and does so efficiently, unlike a naive direct computation of the gradient with respect to each weight individually.

3.3 Convolutional Neural Networks

Convolutional neural networks (CNNs) are a class of deep neural networks, most commonly applied to analyze visual imagery. They are also known as shift invariant or space invariant artificial neural networks (SIANN), based on the shared-weight architecture of the convolution kernels or filters that slide along input features and provide translation-equivariant responses known as feature maps. Counter-intuitively, most convolutional neural networks are not invariant to translation, due to the downsampling operation they apply to the input. They have applications in image and video recognition, recommender systems, image classification, image segmentation, medical image analysis, natural language processing, brain–computer interfaces, and financial time series.

CNNs are regularized versions of multilayer perceptrons. Multilayer perceptrons usually mean fully connected networks, that is, each neuron in one layer is connected to all neurons in the next layer. The "full connectivity" of these networks make them prone to overfitting data. Typical ways of regularization, or preventing overfitting, include: penalizing parameters during training (such as weight decay) or trimming connectivity (skipped connections, dropout etc.) CNNs take a different approach towards regularization: they take advantage of the hierarchical pattern in data and assemble patterns of increasing complexity using smaller and simpler patterns embossed in their filters. Therefore, on a scale of connectivity and complexity, CNNs are on the lower extreme.

3.4 Recurrent Neural Networks and Long Short-Term Memory

Recurrent neural networks (RNNs) are a class of artificial neural networks where connections between nodes can create a cycle, allowing output from some nodes to affect subsequent input to the same nodes. This allows it to exhibit temporal dynamic behavior. Derived from feedforward neural networks, RNNs can use their internal state (memory) to process variable length sequences of inputs. This makes them applicable to tasks such as unsegmented, connected handwriting recognition or speech recognition. Recurrent neural networks are theoretically Turing complete and can run arbitrary programs to process arbitrary sequences of inputs.

Long short-term memory (LSTM) networks are a modified version of recurrent neural networks, which makes it easier to remember past data in memory. The vanishing gradient problem of RNN is resolved here. LSTM is well-suited to classify, process, and predict time series given time lags of unknown duration. It trains the model by using back-propagation. In an LSTM network, three gates are present: input gate, output gate, and the forget gate. These gates determine which information is important enough to keep or throw away. Information passes through these gates and selective information is retained.

3.5 Attention Mechanisms and Transformers

The attention mechanism was proposed to address the limitations of recurrent neural networks when processing long sequences. Traditional sequence-to-sequence models encode the input sequence into a single fixed-length vector, which becomes a bottleneck for long sequences. The attention mechanism allows the model to focus on different parts of the input sequence when producing each element of the output sequence.

The Transformer architecture, introduced in "Attention Is All You Need" (Vaswani et al., 2017), eliminated the need for recurrent connections and relied entirely on attention mechanisms. The self-attention operation allows the model to attend to different positions of the same sequence to compute a representation of the sequence. Multi-head attention runs the attention mechanism multiple times in parallel, allowing the model to jointly attend to information from different representation subspaces.

The Transformer has become the dominant architecture for natural language processing tasks, leading to pre-trained language models such as BERT, GPT, T5, and their successors. These models are trained on large corpora of text and can be fine-tuned for specific tasks with relatively small amounts of labeled data.

3.6 Limitations of Transformer Architectures

Despite their success, Transformers have significant computational limitations. The self-attention mechanism has quadratic complexity with respect to sequence length, making it computationally expensive for very long sequences. The key-value cache required during autoregressive generation grows linearly with context length, consuming substantial GPU memory. These limitations have motivated research into more efficient alternatives, including linear attention mechanisms, state space models, and hybrid architectures that combine the strengths of both approaches.

Recent work has shown that hybrid architectures combining attention layers with linear recurrent layers can achieve competitive performance while significantly reducing memory consumption during inference. The Mamba architecture, based on selective state space models, demonstrates that linear complexity sequence modeling can match or exceed the performance of Transformers on many tasks while offering substantial efficiency advantages for long-context applications.
"""

CHINESE_PASSAGE = """\
第四章：大规模语言模型的训练与优化

4.1 预训练语言模型概述

大规模语言模型（Large Language Models，LLMs）是近年来自然语言处理领域最重要的技术突破之一。这类模型通过在海量文本数据上进行自监督预训练，学习语言的统计规律和语义表示，进而在各类下游任务上展现出卓越的泛化能力。预训练语言模型的核心思想是：通过预测下一个词或者填充被掩码的词，模型能够隐式地学习到丰富的语言知识，包括语法结构、语义关系、世界知识以及推理能力。

从技术架构的演进来看，预训练语言模型经历了从循环神经网络（RNN）到Transformer的重大转变。2017年，Vaswani等人提出的Transformer架构彻底改变了自然语言处理领域的面貌。Transformer摒弃了传统RNN的顺序计算方式，转而采用多头自注意力机制，使得模型能够并行处理序列中任意位置之间的依赖关系。这一架构不仅提升了训练效率，还使模型能够更好地捕获长距离语义依赖。

4.2 自注意力机制的数学原理

自注意力机制（Self-Attention）是Transformer架构的核心组件。给定输入序列X∈R^{n×d}，自注意力机制首先将其线性变换为查询矩阵Q、键矩阵K和值矩阵V，然后通过缩放点积计算注意力权重，最终得到加权求和的输出。

具体而言，注意力计算公式为：Attention(Q,K,V) = softmax(QK^T/√d_k)V

其中d_k为键向量的维度，除以√d_k是为了防止点积结果过大导致梯度消失。多头注意力机制（Multi-Head Attention）在多个不同的线性投影子空间中并行执行注意力操作，从而使模型能够同时关注来自不同表示子空间的信息。

自注意力机制的计算复杂度为O(n²d)，其中n为序列长度。这意味着当处理长序列时，计算量和内存占用会随序列长度的平方增长，这是Transformer架构在长文档处理中的主要瓶颈。

4.3 规模化定律与涌现能力

Kaplan等人（2020）的研究揭示了语言模型性能与训练计算量之间存在稳定的幂律关系，被称为"规模化定律"（Scaling Laws）。研究表明，在固定计算预算下，同时扩大模型参数量和训练数据量通常比单独扩大其中一个维度更为有效。

更令人惊奇的是，当模型规模超过某个阈值时，会出现所谓的"涌现能力"（Emergent Abilities）——即在较小模型上几乎不存在的能力突然出现。这些涌现能力包括：多步推理、少样本学习、算术计算、代码生成等。涌现现象表明，语言模型的能力并非随规模线性增长，而是存在质的飞跃节点。

然而，单纯依赖规模化策略面临日益严峻的挑战：训练大型语言模型需要消耗巨量的计算资源和能源，数据质量和多样性对模型性能的影响日趋重要，如何有效筛选高质量预训练数据成为关键问题。

4.4 高效训练技术

为了降低大规模语言模型的训练成本，研究者们提出了多种高效训练技术。

混合精度训练利用浮点数的不同精度格式（如FP16和BF16）来加速计算并减少内存占用，同时通过动态损失缩放技术保持数值稳定性。实践表明，混合精度训练通常能够将训练速度提升2-3倍，同时将显存占用减少约一半。

梯度检查点（Gradient Checkpointing）技术通过在前向传播中只保存部分中间激活值，在反向传播时重新计算所需的中间结果，从而以额外的计算量换取显存空间。这一技术使得在有限显存条件下训练更大批次或更长序列成为可能。

分布式训练是训练超大规模语言模型的必要手段。常见策略包括数据并行（Data Parallelism）、模型并行（Model Parallelism）和流水线并行（Pipeline Parallelism）。ZeRO（零冗余优化器）优化策略通过在多个设备间分片存储优化器状态、梯度和模型参数，显著降低了每台设备的显存需求。

4.5 知识蒸馏与模型压缩

知识蒸馏（Knowledge Distillation）是一种模型压缩技术，旨在将大型教师模型的知识迁移到小型学生模型中。其核心思想是：让学生模型不仅学习正确的标签，还学习教师模型输出的"软标签"（即各类别的预测概率分布）。软标签包含了教师模型对不同类别的相对置信度信息，这些信息比硬标签更加丰富，有助于学生模型学习到更泛化的表示。

在大语言模型的知识蒸馏中，通常采用KL散度作为蒸馏损失，衡量学生模型输出分布与教师模型输出分布之间的差异。引入温度参数T可以软化概率分布，使得低概率事件的相对关系也能被学生模型学习到。较高的温度值会使分布更加平滑，有助于知识迁移；较低的温度值则使分布更加尖锐，更接近硬标签训练。

4.6 混合架构的发展趋势

面对Transformer架构在长序列处理中的效率瓶颈，研究界正在积极探索混合架构方案。这类架构通常将少量注意力层与大量线性复杂度层相结合，在保持模型表达能力的同时，显著提升推理效率和降低显存占用。

典型的线性复杂度层包括：基于选择性状态空间模型的Mamba层、基于增量规则的GatedDeltaNet层等。这些线性层通过维护固定大小的隐状态来压缩历史信息，推理时的显存占用不随上下文长度增加，特别适合长文档理解和生成任务。

研究表明，在保留约25%的注意力层、将其余75%替换为线性层的混合架构中，模型能够在仅损失少量语言建模能力的情况下，实现显著的推理加速和显存节省。这一发现与Qwen3-Next等最新商业模型的架构选择高度吻合，表明混合架构是大语言模型部署优化的重要方向。
"""


# ─────────────────────────────────────────────────────────────────────────────
# 模型加载辅助
# ─────────────────────────────────────────────────────────────────────────────

def load_hybrid(path: str, device: str):
    model = QwenHybridForCausalLM.from_pretrained_hybrid(
        path, torch_dtype=torch.float16
    ).to(device)
    model.eval()
    return model


def load_qwen(path: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return model


def free_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# ① 总结对比
# ─────────────────────────────────────────────────────────────────────────────

CASES = [
    {
        "name": "English (FineEdu academic chapter)",
        "prompt_prefix": "Summarize the following text in detail:\n\n",
        "text": ENGLISH_PASSAGE,
    },
    {
        "name": "Chinese (科技教育论文章节)",
        "prompt_prefix": "请详细总结以下文本的核心观点：\n\n",
        "text": CHINESE_PASSAGE,
    },
]


def build_prompt(tokenizer, prefix: str, text: str) -> str:
    """拼接指令 + 正文，不使用 chat template（裸文本）。"""
    return prefix + text


@torch.no_grad()
def generate_with_stats(model, tokenizer, prompt: str, device: str,
                        max_new_tokens: int = 512,
                        temperature: float = 0.1,
                        repetition_penalty: float = 1.1):
    """生成文本，返回 (output_text, tokens_per_sec, peak_vram_mb)。"""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.05),
        temperature=temperature if temperature > 0.05 else None,
        repetition_penalty=repetition_penalty,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    new_tokens = output_ids.shape[1] - prompt_len
    tok_per_sec = new_tokens / elapsed if elapsed > 0 else 0

    if device == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        peak_vram = 0.0

    generated = tokenizer.decode(
        output_ids[0, prompt_len:], skip_special_tokens=True
    )
    return generated, tok_per_sec, peak_vram, new_tokens


def run_summarization(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.qwen3, trust_remote_code=True)

    sep = "─" * 80

    for case in CASES:
        prompt = build_prompt(tokenizer, case["prompt_prefix"], case["text"])
        input_len = len(tokenizer(prompt).input_ids)
        print(f"\n{'═' * 80}")
        print(f"  CASE: {case['name']}")
        print(f"  Prompt tokens: {input_len}")
        print(f"{'═' * 80}")

        # ── Qwen3-4B ──────────────────────────────────────────────────────────
        print("\n[1/2] Loading Qwen3-4B ...")
        model = load_qwen(args.qwen3, device)
        text_q, tps_q, vram_q, ntok_q = generate_with_stats(
            model, tokenizer, prompt, device,
            max_new_tokens=512, temperature=0.1, repetition_penalty=1.1
        )
        free_model(model)
        print(f"\n{'─'*40} Qwen3-4B Output {'─'*23}")
        print(text_q)
        print(f"\n  → {ntok_q} tokens | {tps_q:.1f} tok/s | Peak VRAM {vram_q:.0f} MB")

        # ── Mamba Hybrid ──────────────────────────────────────────────────────
        print(f"\n[2/2] Loading Mamba Hybrid ({args.mamba}) ...")
        model = load_hybrid(args.mamba, device)
        text_m, tps_m, vram_m, ntok_m = generate_with_stats(
            model, tokenizer, prompt, device,
            max_new_tokens=512, temperature=0.1, repetition_penalty=1.1
        )
        free_model(model)
        print(f"\n{'─'*40} Mamba Hybrid Output {'─'*19}")
        print(text_m)
        print(f"\n  → {ntok_m} tokens | {tps_m:.1f} tok/s | Peak VRAM {vram_m:.0f} MB")

        # ── Delta ─────────────────────────────────────────────────────────────
        print(f"\n{sep}")
        print(f"  Speed  : Mamba {tps_m:.1f} vs Qwen3 {tps_q:.1f} tok/s  "
              f"({(tps_m/tps_q-1)*100:+.1f}%)")
        print(f"  VRAM   : Mamba {vram_m:.0f} MB vs Qwen3 {vram_q:.0f} MB  "
              f"({(vram_m/vram_q-1)*100:+.1f}%)")
        print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# ② PPL 测试
# ─────────────────────────────────────────────────────────────────────────────

def _load_local_texts(path: str, max_texts: int = 10000) -> List[str]:
    """
    从本地路径读取文本，支持：
      - .txt          纯文本，整个文件作为一条文本
      - .jsonl        每行一个 JSON，取 "text" 字段
      - .parquet      pandas 读取，取 "text" 列（fineweb-edu 原始格式）
      - 目录           递归识别上述格式，或直接调用 datasets.load_from_disk()
    """
    import os

    # 目录：先尝试 load_from_disk，失败则递归扫描支持的文件
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "dataset_info.json")):
            try:
                from datasets import load_from_disk
                ds = load_from_disk(path)
                return [item["text"] for item in ds][:max_texts]
            except Exception:
                pass  # 不是合法 Dataset 目录，fallback

        texts = []
        for fname in sorted(os.listdir(path)):
            if len(texts) >= max_texts:
                break
            fpath = os.path.join(path, fname)
            if os.path.isfile(fpath):
                texts.extend(_load_local_texts(fpath, max_texts - len(texts)))
        return texts

    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return [f.read()]

    if ext in (".jsonl", ".json"):
        import json
        texts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text") or obj.get("content") or \
                       " ".join(v for v in obj.values() if isinstance(v, str))
                if text:
                    texts.append(text)
                if len(texts) >= max_texts:
                    break
        return texts

    if ext == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path, columns=["text"])
        return df["text"].dropna().tolist()[:max_texts]

    if ext == ".arrow":
        # HuggingFace datasets 缓存用的是 Arrow IPC stream 格式
        import pyarrow as pa
        try:
            with pa.memory_map(path, "r") as src:
                reader = pa.ipc.open_stream(src)
                table = reader.read_all()
        except pa.lib.ArrowInvalid:
            with pa.memory_map(path, "r") as src:
                table = pa.ipc.open_file(src).read_all()
        if "text" in table.column_names:
            return table.column("text").to_pylist()[:max_texts]
        return []

    return []  # 不支持的格式，跳过


class FineEduPPLDataset(Dataset):
    """
    PPL 评估数据集，两种来源：
      local=True  → 用脚本内嵌的英/中文段落，无需下载任何文件
      local=False → 流式加载 HuggingFace fineweb-edu（需要网络，约拉取 1-2 GB 第一分片）
    """

    def __init__(self, tokenizer, seq_len: int, num_samples: int,
                 local: bool = False, data_path: str = None,
                 skip_first: int = 0, split: str = "train"):
        self.examples: List[torch.Tensor] = []

        if local:
            # ── 内嵌文本（无需任何文件）────────────────────────────────────────
            combined = (ENGLISH_PASSAGE + "\n\n" + CHINESE_PASSAGE) * 200
            buf = tokenizer.encode(combined)
            for i in range(0, len(buf) - seq_len + 1, seq_len):
                if len(self.examples) >= num_samples:
                    break
                self.examples.append(torch.tensor(buf[i:i + seq_len], dtype=torch.long))
            print(f"  [local] Built {len(self.examples)} chunks × {seq_len} tokens "
                  f"from embedded passages (no download)")

        elif data_path is not None:
            # ── 本地文件/目录 ──────────────────────────────────────────────────
            # 多取一些文本以应对 skip_first 跳过前几条
            texts = _load_local_texts(data_path, (num_samples + skip_first) * 10)
            # 跳过前 skip_first 条（避开训练数据区域）
            texts = texts[skip_first:]
            buf: List[int] = []
            for text in texts:
                if len(self.examples) >= num_samples:
                    break
                buf.extend(tokenizer.encode(text))
                while len(buf) >= seq_len and len(self.examples) < num_samples:
                    self.examples.append(
                        torch.tensor(buf[:seq_len], dtype=torch.long)
                    )
                    buf = buf[seq_len:]
            skip_msg = f", skipped first {skip_first} docs" if skip_first else ""
            print(f"  [local path] Loaded {len(self.examples)} chunks × {seq_len} tokens "
                  f"from {data_path}{skip_msg}")

        else:
            # ── HuggingFace 流式模式 ──────────────────────────────────────────
            from datasets import load_dataset
            ds = load_dataset(
                "HuggingFaceFW/fineweb-edu",
                name="sample-10BT",
                split=split,
                streaming=True,
                trust_remote_code=True,
            )
            buf_hf: List[int] = []
            for item in ds:
                buf_hf.extend(tokenizer.encode(item["text"]))
                while len(buf_hf) >= seq_len and len(self.examples) < num_samples:
                    self.examples.append(
                        torch.tensor(buf_hf[:seq_len], dtype=torch.long)
                    )
                    buf_hf = buf_hf[seq_len:]
                if len(self.examples) >= num_samples:
                    break
            print(f"  [fineweb-edu] Loaded {len(self.examples)} chunks × {seq_len} tokens")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


@torch.no_grad()
def compute_ppl(model, dataloader, device: str) -> float:
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    for batch in dataloader:
        input_ids = batch.to(device)                  # (B, L)
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss                               # cross-entropy (mean over tokens)
        n = input_ids.numel()
        total_loss += loss.item() * n
        total_tokens += n
    return math.exp(total_loss / total_tokens)


def run_ppl(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Loading tokenizer from {args.qwen3} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.qwen3, trust_remote_code=True)

    if args.local:
        src = "embedded passages (no download)"
    elif args.data_path:
        src = args.data_path
    else:
        src = "HuggingFace fineweb-edu (streaming)"
    print(f"\nBuilding PPL dataset from: {src}")
    print(f"  {args.num_samples} chunks × {args.seq_len} tokens")
    dataset = FineEduPPLDataset(
        tokenizer, seq_len=args.seq_len,
        num_samples=args.num_samples,
        local=args.local,
        data_path=args.data_path,
        skip_first=args.skip_first,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = {}

    # ── Qwen3-4B ──────────────────────────────────────────────────────────────
    print(f"\n[1/2] Evaluating Qwen3-4B PPL ...")
    model = load_qwen(args.qwen3, device)
    t0 = time.perf_counter()
    ppl_q = compute_ppl(model, dataloader, device)
    elapsed_q = time.perf_counter() - t0
    free_model(model)
    results["Qwen3-4B"] = {"ppl": ppl_q, "elapsed_s": elapsed_q}
    print(f"  Qwen3-4B   PPL = {ppl_q:.3f}  ({elapsed_q:.1f}s)")

    # ── Mamba Hybrid ──────────────────────────────────────────────────────────
    print(f"\n[2/2] Evaluating Mamba Hybrid PPL ...")
    model = load_hybrid(args.mamba, device)
    t0 = time.perf_counter()
    ppl_m = compute_ppl(model, dataloader, device)
    elapsed_m = time.perf_counter() - t0
    free_model(model)
    results["Mamba"] = {"ppl": ppl_m, "elapsed_s": elapsed_m}
    print(f"  Mamba Hybrid PPL = {ppl_m:.3f}  ({elapsed_m:.1f}s)")

    # ── Summary ───────────────────────────────────────────────────────────────
    data_src = "local passages" if args.local else "FineEdu (HF)"
    print(f"\n{'═'*60}")
    print(f"  PPL COMPARISON  ({data_src}, {args.num_samples}×{args.seq_len} tokens)")
    print(f"{'═'*60}")
    print(f"  Qwen3-4B (Baseline) : {ppl_q:.3f}")
    print(f"  Mamba Hybrid        : {ppl_m:.3f}  ({(ppl_m/ppl_q-1)*100:+.2f}% vs baseline)")
    print(f"{'═'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# ③ 中文迁移能力测试（续写 + 分语言 PPL）
# ─────────────────────────────────────────────────────────────────────────────

CHINESE_CONTINUATIONS = [
    "自然语言处理是人工智能领域中最重要的研究方向之一。近年来，随着深度学习技术的快速发展，",
    "在计算机视觉领域，卷积神经网络已经取得了巨大的成功。然而，对于序列建模任务，",
    "知识蒸馏是一种有效的模型压缩技术，其核心思想是",
    "Transformer架构的自注意力机制虽然强大，但其计算复杂度为O(n²)，这意味着",
    "混合精度训练通过使用不同精度的浮点数格式来加速模型训练。具体来说，",
]

ENGLISH_CONTINUATIONS = [
    "Natural language processing has become one of the most important areas in artificial intelligence. With the rapid development of deep learning, ",
    "In the field of computer vision, convolutional neural networks have achieved great success. However, for sequence modeling tasks, ",
    "Knowledge distillation is an effective model compression technique. The core idea is to ",
    "The self-attention mechanism in Transformers is powerful but has O(n²) complexity, which means ",
    "Mixed precision training uses different floating point formats to accelerate training. Specifically, ",
]


@torch.no_grad()
def run_transfer(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.qwen3, trust_remote_code=True)

    # ── 测试 1：续写质量对比 ─────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print(f"  TEST 1: 中文续写质量对比（base model 用续写而非指令）")
    print(f"{'═'*80}")

    for model_label, load_fn, path in [
        ("Qwen3-4B", load_qwen, args.qwen3),
        ("Mamba Hybrid", load_hybrid, args.mamba),
    ]:
        print(f"\n  Loading {model_label} ...")
        model = load_fn(path, device)
        for i, prompt in enumerate(CHINESE_CONTINUATIONS[:3]):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            output = model.generate(
                input_ids,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.2,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"\n  [{model_label}] Prompt {i+1}: {prompt}")
            print(f"  续写: {generated[len(prompt):][:300]}")
        free_model(model)

    # ── 测试 2：分语言 PPL 对比 ──────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print(f"  TEST 2: 分语言 PPL（中文 vs 英文）")
    print(f"{'═'*80}")

    # 构建纯中文和纯英文的评估 chunks
    cn_tokens = tokenizer.encode(CHINESE_PASSAGE * 50)
    en_tokens = tokenizer.encode(ENGLISH_PASSAGE * 50)
    seq_len = 1024
    num_chunks = 30

    cn_chunks = [
        torch.tensor(cn_tokens[i:i+seq_len], dtype=torch.long)
        for i in range(0, len(cn_tokens) - seq_len + 1, seq_len)
    ][:num_chunks]
    en_chunks = [
        torch.tensor(en_tokens[i:i+seq_len], dtype=torch.long)
        for i in range(0, len(en_tokens) - seq_len + 1, seq_len)
    ][:num_chunks]

    results = {}
    for model_label, load_fn, path in [
        ("Qwen3-4B", load_qwen, args.qwen3),
        ("Mamba Hybrid", load_hybrid, args.mamba),
    ]:
        print(f"\n  Evaluating {model_label} ...")
        model = load_fn(path, device)
        model.eval()

        for lang, chunks in [("English", en_chunks), ("Chinese", cn_chunks)]:
            total_loss = 0.0
            total_tokens = 0
            for chunk in chunks:
                ids = chunk.unsqueeze(0).to(device)
                out = model(input_ids=ids, labels=ids.clone())
                total_loss += out.loss.item() * ids.numel()
                total_tokens += ids.numel()
            ppl = math.exp(total_loss / total_tokens)
            results[(model_label, lang)] = ppl
            print(f"    {lang:>8} PPL = {ppl:.3f}")

        free_model(model)

    # ── Summary ──────────────────────────────────────────────────────────────
    q_en = results[("Qwen3-4B", "English")]
    q_cn = results[("Qwen3-4B", "Chinese")]
    m_en = results[("Mamba Hybrid", "English")]
    m_cn = results[("Mamba Hybrid", "Chinese")]

    print(f"\n{'═'*70}")
    print(f"  TRANSFER ABILITY SUMMARY")
    print(f"{'═'*70}")
    print(f"  {'':>20} {'English':>12} {'Chinese':>12} {'CN/EN ratio':>14}")
    print(f"  {'─'*58}")
    print(f"  {'Qwen3-4B':>20} {q_en:>12.3f} {q_cn:>12.3f} {q_cn/q_en:>14.3f}")
    print(f"  {'Mamba Hybrid':>20} {m_en:>12.3f} {m_cn:>12.3f} {m_cn/m_en:>14.3f}")
    print(f"  {'─'*58}")
    print(f"  {'EN degradation':>20} {(m_en/q_en-1)*100:>+11.1f}%")
    print(f"  {'CN degradation':>20} {(m_cn/q_cn-1)*100:>+11.1f}%")
    cn_extra = (m_cn/q_cn) / (m_en/q_en)
    print(f"  {'CN extra loss':>20} {(cn_extra-1)*100:>+11.1f}%  "
          f"{'(中文额外退化)' if cn_extra > 1.05 else '(中文保持良好)'}")
    print(f"{'═'*70}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── summarization ─────────────────────────────────────────────────────────
    s = sub.add_parser("summarization", help="长文档总结对比")
    s.add_argument("--mamba",  required=True, help="Mamba Hybrid checkpoint 路径")
    s.add_argument("--qwen3",  required=True, help="Qwen3-4B 模型路径")

    # ── transfer ──────────────────────────────────────────────────────────────
    t = sub.add_parser("transfer", help="中文迁移能力测试（续写 + 分语言 PPL）")
    t.add_argument("--mamba",  required=True, help="Mamba Hybrid checkpoint 路径")
    t.add_argument("--qwen3",  required=True, help="Qwen3-4B 模型路径")

    # ── ppl ───────────────────────────────────────────────────────────────────
    p = sub.add_parser("ppl", help="FineEdu PPL 对比")
    p.add_argument("--mamba",       required=True, help="Mamba Hybrid checkpoint 路径")
    p.add_argument("--qwen3",       required=True, help="Qwen3-4B 模型路径")
    p.add_argument("--num_samples", type=int, default=200, help="评估样本数（每条 seq_len tokens）")
    p.add_argument("--seq_len",     type=int, default=2048, help="每条序列长度")
    p.add_argument("--local",       action="store_true",
                   help="用脚本内嵌文本计算 PPL，无需下载任何数据集")
    p.add_argument("--skip_first",  type=int, default=0,
                   help="跳过数据集前 N 条文档（用于跳过训练数据区域）")
    p.add_argument("--data_path",   type=str, default=None,
                   help="本地数据集路径，支持：\n"
                        "  目录（HF save_to_disk 或含 .jsonl/.parquet 文件）\n"
                        "  单个 .jsonl 文件（每行含 text 字段）\n"
                        "  单个 .parquet 文件（含 text 列）\n"
                        "  单个 .txt 纯文本文件")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "summarization":
        run_summarization(args)
    elif args.cmd == "ppl":
        run_ppl(args)
    elif args.cmd == "transfer":
        run_transfer(args)
