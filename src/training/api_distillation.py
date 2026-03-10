"""
API-based Knowledge Distillation Module.

Enables using large cloud-hosted models (e.g., Qwen-Max) as teachers
via API, eliminating the need to load a large teacher model locally.

Two distillation modes:
1. Sequence-level KD:
   - Generate high-quality text from the API teacher
   - Train student with standard cross-entropy on generated text
   - Most practical, works with any API

2. Top-K Logprob KD:
   - Query the API for token-level logprobs (DashScope supports top-20)
   - Use sparse teacher distribution for approximate KL divergence
   - Better signal than pure CE, but API-dependent

This approach enables training on a single GPU (4090/3090) since only
the student model needs to reside in GPU memory.
"""

import json
import logging
import math
import os
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class APIDistillConfig:
    """Configuration for API-based distillation."""

    # API settings
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""                # Set via env DASHSCOPE_API_KEY or arg
    teacher_model: str = "qwen-max"  # API model name
    api_max_retries: int = 3
    api_retry_delay: float = 1.0
    api_timeout: float = 60.0
    api_max_concurrent: int = 4

    # Generation settings (for sequence-level KD)
    gen_max_tokens: int = 2048
    gen_temperature: float = 0.7
    gen_top_p: float = 0.9
    gen_num_samples_per_prompt: int = 1

    # Logprob settings (for top-K KD)
    use_logprobs: bool = True
    logprob_top_k: int = 20       # API typically supports up to 20

    # Training settings
    alpha_ce: float = 0.7          # Weight for CE loss on API-generated text
    alpha_kd_sparse: float = 0.3   # Weight for sparse KD loss (from logprobs)
    temperature: float = 2.0       # KD temperature for sparse logprob distillation
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    num_train_epochs: int = 1
    max_steps: int = -1
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    bf16: bool = True

    # Data settings
    max_seq_length: int = 2048
    cache_dir: str = "./cache/api_distill"

    # Logging / saving
    logging_steps: int = 10
    save_steps: int = 500
    output_dir: str = "./checkpoints/api-distill"

    seed: int = 42


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------

class QwenAPIClient:
    """
    Client for interacting with Qwen-Max (DashScope) API.

    Uses the OpenAI-compatible API interface provided by DashScope.
    """

    def __init__(self, config: APIDistillConfig):
        self.config = config
        self.api_key = config.api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "No API key provided. Set DASHSCOPE_API_KEY env variable "
                "or pass api_key in config."
            )

        # Use openai client for compatibility
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=config.api_base_url,
                timeout=config.api_timeout,
            )
            self._has_client = True
        except ImportError:
            logger.warning(
                "openai package not installed. Install with: pip install openai"
            )
            self._has_client = False
            self.client = None

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate text from the API.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            logprobs: Whether to return logprobs
            top_logprobs: Number of top logprobs to return

        Returns:
            Dictionary with 'text', 'tokens', 'logprobs' (if requested)
        """
        if not self._has_client:
            raise RuntimeError("OpenAI client not available. Install: pip install openai")

        max_tokens = max_tokens or self.config.gen_max_tokens
        temperature = temperature or self.config.gen_temperature
        top_p = top_p or self.config.gen_top_p

        for attempt in range(self.config.api_max_retries):
            try:
                kwargs = {
                    "model": self.config.teacher_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }

                if logprobs:
                    kwargs["logprobs"] = True
                    kwargs["top_logprobs"] = top_logprobs or self.config.logprob_top_k

                response = self.client.chat.completions.create(**kwargs)

                result = {
                    "text": response.choices[0].message.content,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    },
                }

                # Extract logprobs if available
                if logprobs and response.choices[0].logprobs:
                    token_logprobs = []
                    for token_info in response.choices[0].logprobs.content:
                        entry = {
                            "token": token_info.token,
                            "logprob": token_info.logprob,
                            "top_logprobs": [
                                {"token": lp.token, "logprob": lp.logprob}
                                for lp in (token_info.top_logprobs or [])
                            ]
                        }
                        token_logprobs.append(entry)
                    result["logprobs"] = token_logprobs

                return result

            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.api_max_retries - 1:
                    time.sleep(self.config.api_retry_delay * (2 ** attempt))
                else:
                    raise

    def batch_generate(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate for a batch of prompts (sequential for simplicity)."""
        results = []
        for i, prompt in enumerate(prompts):
            try:
                result = self.generate(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to generate for prompt {i}: {e}")
                results.append({"text": "", "error": str(e)})

            # Rate limiting
            if i < len(prompts) - 1:
                time.sleep(0.5)

        return results


# ---------------------------------------------------------------------------
# API-Generated Dataset
# ---------------------------------------------------------------------------

class APIGeneratedDataset(Dataset):
    """
    Dataset created from API-generated text.

    Caches generated text to disk to avoid redundant API calls.
    """

    def __init__(
        self,
        tokenizer,
        texts: List[str],
        max_length: int = 2048,
        logprobs_data: Optional[List[List[Dict]]] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for i, text in enumerate(texts):
            if not text:
                continue

            tokens = tokenizer.encode(text, add_special_tokens=True)

            # Chunk into max_length sequences
            for start in range(0, len(tokens) - 1, max_length):
                chunk = tokens[start:start + max_length]
                if len(chunk) < 32:
                    continue
                self.examples.append({
                    "input_ids": chunk,
                    "logprobs": logprobs_data[i] if logprobs_data else None,
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)

        # Pad or truncate
        if len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            input_ids = F.pad(input_ids, (0, pad_len), value=self.tokenizer.pad_token_id or 0)
            attention_mask = torch.cat([
                torch.ones(len(item["input_ids"]), dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ])
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = torch.ones(self.max_length, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


# ---------------------------------------------------------------------------
# Data Generation Pipeline
# ---------------------------------------------------------------------------

def generate_training_data(
    api_client: QwenAPIClient,
    prompts: List[str],
    config: APIDistillConfig,
    tokenizer=None,
) -> Tuple[List[str], Optional[List[List[Dict]]]]:
    """
    Generate training data from the API teacher.

    Args:
        api_client: The API client
        prompts: List of seed prompts
        config: Distillation config
        tokenizer: Tokenizer for caching

    Returns:
        Tuple of (generated_texts, logprobs_data)
    """
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    texts = []
    logprobs_data = [] if config.use_logprobs else None

    for i, prompt in enumerate(prompts):
        # Check cache
        cache_key = hashlib.md5(
            f"{prompt}:{config.teacher_model}:{config.gen_temperature}".encode()
        ).hexdigest()
        cache_file = cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            texts.append(cached.get("text", ""))
            if logprobs_data is not None:
                logprobs_data.append(cached.get("logprobs", []))
            continue

        # Generate from API
        try:
            result = api_client.generate(
                prompt=prompt,
                max_tokens=config.gen_max_tokens,
                temperature=config.gen_temperature,
                top_p=config.gen_top_p,
                logprobs=config.use_logprobs,
                top_logprobs=config.logprob_top_k,
            )

            texts.append(result.get("text", ""))
            if logprobs_data is not None:
                logprobs_data.append(result.get("logprobs", []))

            # Cache to disk
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{len(prompts)} samples")

        except Exception as e:
            logger.error(f"Failed to generate for prompt {i}: {e}")
            texts.append("")
            if logprobs_data is not None:
                logprobs_data.append([])

    logger.info(f"Generated {len([t for t in texts if t])} valid samples out of {len(prompts)}")
    return texts, logprobs_data


def load_cached_data(
    cache_dir: str,
) -> Tuple[List[str], Optional[List[List[Dict]]]]:
    """
    从缓存目录加载已预生成的 API 数据（不调用 API）。

    用于两阶段工作流：
      1. 本地电脑运行 generate_api_data.py 生成数据
      2. GPU 服务器调用此函数直接加载缓存训练

    Args:
        cache_dir: 缓存目录路径（包含 *.json 文件）

    Returns:
        Tuple of (texts, logprobs_data)
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"缓存目录不存在: {cache_dir}\n"
            f"请先在本地运行: python scripts/generate_api_data.py --output_dir {cache_dir}"
        )

    json_files = sorted(cache_path.glob("*.json"))
    # 排除统计文件
    json_files = [f for f in json_files if not f.name.startswith("_")]

    if not json_files:
        raise FileNotFoundError(
            f"缓存目录为空: {cache_dir}\n"
            f"请先运行: python scripts/generate_api_data.py --output_dir {cache_dir}"
        )

    texts = []
    logprobs_data = []
    has_logprobs = False

    for f in json_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            text = data.get("text", "")
            if text:
                texts.append(text)
                lp = data.get("logprobs", [])
                logprobs_data.append(lp)
                if lp:
                    has_logprobs = True
        except Exception as e:
            logger.warning(f"跳过无法读取的缓存文件 {f.name}: {e}")

    logger.info(
        f"从缓存加载了 {len(texts)} 条数据 "
        f"(共 {len(json_files)} 个文件, logprobs={'有' if has_logprobs else '无'})"
    )

    return texts, logprobs_data if has_logprobs else None


# ---------------------------------------------------------------------------
# Sparse KD Loss (from top-K logprobs)
# ---------------------------------------------------------------------------

class SparseKDLoss(nn.Module):
    """
    Knowledge distillation loss using sparse teacher logprobs.

    Since the API only returns top-K logprobs (not full vocabulary),
    we compute an approximate KL divergence using only the teacher's
    top-K tokens plus a uniform background for the remaining vocabulary.

    Loss = sum over top-K tokens: p_teacher(t) * log(p_teacher(t) / p_student(t))
         + p_background * sum over rest: log(p_background / p_student(t))
    """

    def __init__(self, temperature: float = 2.0, vocab_size: int = 151936):
        super().__init__()
        self.temperature = temperature
        self.vocab_size = vocab_size

    def forward(
        self,
        student_logits: torch.Tensor,   # (B, L, V)
        teacher_top_k_ids: torch.Tensor, # (B, L, K) token ids
        teacher_top_k_logprobs: torch.Tensor, # (B, L, K) log probabilities
    ) -> torch.Tensor:
        """Compute sparse KD loss."""
        B, L, V = student_logits.shape
        K = teacher_top_k_ids.shape[-1]

        # Student log probabilities (temperature-scaled)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)

        # Gather student log probs for teacher's top-K tokens
        # teacher_top_k_ids: (B, L, K)
        student_topk_logprobs = torch.gather(
            student_log_probs, dim=-1, index=teacher_top_k_ids
        )  # (B, L, K)

        # Teacher probabilities from logprobs
        teacher_probs = torch.exp(teacher_top_k_logprobs / self.temperature)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # KL divergence on top-K tokens
        kl_topk = teacher_probs * (
            torch.log(teacher_probs.clamp(min=1e-8)) - student_topk_logprobs
        )

        loss = kl_topk.sum(dim=-1).mean() * (self.temperature ** 2)

        return loss


# ---------------------------------------------------------------------------
# API Distillation Trainer
# ---------------------------------------------------------------------------

class APIDistillationTrainer:
    """
    Trainer for API-based knowledge distillation.

    Only the student model needs to be on GPU. Teacher signals come from
    pre-generated API data (cached to disk).

    This is ideal for single-GPU training (4090/3090) since it avoids
    loading a large teacher model locally.
    """

    def __init__(
        self,
        student_model: nn.Module,
        config: APIDistillConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        tokenizer=None,
    ):
        self.student = student_model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.tokenizer = tokenizer

        self.device = next(self.student.parameters()).device
        self.use_amp = config.bf16
        self.amp_dtype = torch.bfloat16 if config.bf16 else torch.float32

        # Loss
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Total steps
        if config.max_steps > 0:
            self.total_steps = config.max_steps
        else:
            steps_per_epoch = math.ceil(
                len(train_dataloader) / config.gradient_accumulation_steps
            )
            self.total_steps = steps_per_epoch * config.num_train_epochs

        # Scheduler
        warmup_steps = int(self.total_steps * config.warmup_ratio)
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup = LinearLR(
            self.optimizer, start_factor=1e-7, end_factor=1.0,
            total_iters=max(warmup_steps, 1),
        )
        cosine = CosineAnnealingLR(
            self.optimizer, T_max=max(self.total_steps - warmup_steps, 1),
            eta_min=config.learning_rate * 0.1,
        )
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
        )

        self.global_step = 0

        # Gradient checkpointing
        if config.gradient_checkpointing:
            if hasattr(student_model, 'model') and hasattr(student_model.model, 'gradient_checkpointing_enable'):
                student_model.model.gradient_checkpointing_enable()

        # TensorBoard
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join(config.output_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        except ImportError:
            pass

    def train(self):
        """Main training loop for API distillation."""
        logger.info("=" * 60)
        logger.info("  API Distillation Training")
        logger.info("=" * 60)
        logger.info(f"  Teacher model (API): {self.config.teacher_model}")
        logger.info(f"  Total steps: {self.total_steps}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info("=" * 60)

        self.student.train()
        self.optimizer.zero_grad()

        accum_loss = 0.0
        start_time = time.time()

        for epoch in range(self.config.num_train_epochs):
            for step, batch in enumerate(self.train_dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch.get("labels", input_ids.clone()).to(self.device)

                with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    outputs = self.student(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                    # CE loss on API-generated text
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = self.ce_loss(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    loss = loss / self.config.gradient_accumulation_steps

                loss.backward()
                accum_loss += loss.item() * self.config.gradient_accumulation_steps

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.global_step += 1

                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = accum_loss / self.config.logging_steps
                        elapsed = time.time() - start_time
                        lr = self.scheduler.get_last_lr()[0]
                        ppl = math.exp(min(avg_loss, 20))

                        logger.info(
                            f"Step {self.global_step}/{self.total_steps} | "
                            f"Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | "
                            f"LR: {lr:.2e} | Time: {elapsed:.1f}s"
                        )

                        if self.writer:
                            self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                            self.writer.add_scalar("train/ppl", ppl, self.global_step)
                            self.writer.add_scalar("train/lr", lr, self.global_step)

                        accum_loss = 0.0
                        start_time = time.time()

                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint()

                    if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                        self._save_checkpoint(is_final=True)
                        return self.global_step

        self._save_checkpoint(is_final=True)
        logger.info("API distillation training completed!")
        return self.global_step

    def _save_checkpoint(self, is_final: bool = False):
        """Save checkpoint."""
        if is_final:
            save_dir = os.path.join(self.config.output_dir, "final")
        else:
            save_dir = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")

        os.makedirs(save_dir, exist_ok=True)

        if hasattr(self.student, 'save_pretrained'):
            self.student.save_pretrained(save_dir)
        else:
            torch.save(self.student.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

        if self.tokenizer:
            self.tokenizer.save_pretrained(save_dir)

        logger.info(f"Checkpoint saved to {save_dir}")


# ---------------------------------------------------------------------------
# Seed Prompts for Data Generation
# ---------------------------------------------------------------------------

DEFAULT_SEED_PROMPTS = [
    "请详细解释量子计算的基本原理，包括量子比特、叠加态和量子纠缠的概念。",
    "Explain the transformer architecture in detail, including self-attention, multi-head attention, and positional encoding.",
    "写一篇关于人工智能伦理的深度分析文章，讨论AI决策的公平性问题。",
    "Describe the mathematical foundations of neural networks, including backpropagation and gradient descent.",
    "详细介绍中国古代四大发明的历史背景和对世界文明的影响。",
    "Explain the concept of attention mechanisms in natural language processing, from Bahdanau attention to modern variants.",
    "请分析全球气候变化的主要原因、当前影响和未来预测。",
    "Describe the process of protein folding and explain why AlphaFold was a breakthrough in computational biology.",
    "详细解释经济学中的供需关系理论，并举例说明价格机制的作用。",
    "Explain the differences between supervised, unsupervised, and reinforcement learning with practical examples.",
    "请介绍CRISPR基因编辑技术的原理、应用前景和伦理挑战。",
    "Describe the mathematical foundations of state space models and their relationship to recurrent neural networks.",
    "写一篇关于区块链技术原理和去中心化金融（DeFi）的技术分析。",
    "Explain how large language models are trained, covering pretraining, fine-tuning, and RLHF.",
    "请详细分析太阳系各行星的特征，并讨论寻找外星生命的科学方法。",
    "Describe the key innovations in the Mamba architecture and how selective state spaces differ from attention.",
    "详细解释相对论的基本概念，包括狭义相对论和广义相对论的区别。",
    "Explain mixture of experts (MoE) models, their routing mechanisms, and load balancing strategies.",
    "请分析全球半导体产业链的现状、技术瓶颈和未来发展趋势。",
    "Describe the evolution of computer vision from CNNs to Vision Transformers to modern foundation models.",
]
