"""
Knowledge Distillation Trainer for Qwen-Mamba Hybrid Model.

This module implements the knowledge distillation training loop where:
- Teacher: Original pre-trained Qwen model (frozen)
- Student: Hybrid Attention-Mamba model (trainable)

Training Loss:
    L = α_kd * L_KD + α_ce * L_CE + α_hidden * L_hidden

Where:
    L_KD     = KL(softmax(student_logits/T), softmax(teacher_logits/T)) * T²
    L_CE     = CrossEntropy(student_logits, labels)
    L_hidden = MSE(student_hidden, teacher_hidden)  [optional]
"""

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation training."""

    # Distillation hyperparameters
    temperature: float = 2.0
    alpha_kd: float = 0.5            # Weight for KD loss (soft targets)
    alpha_ce: float = 0.5            # Weight for CE loss (hard labels)
    alpha_hidden: float = 0.0        # Weight for hidden state alignment (0 = disabled)
    hidden_align_layers: List[int] = field(default_factory=list)

    # Training hyperparameters
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    warmup_steps: int = 0            # If > 0, overrides warmup_ratio
    lr_scheduler_type: str = "cosine"
    num_train_epochs: int = 1
    max_steps: int = -1              # -1 = use num_train_epochs

    # Batch / accumulation
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 16

    # Precision
    bf16: bool = True
    fp16: bool = False

    # Gradient checkpointing
    gradient_checkpointing: bool = True

    # Logging / saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    output_dir: str = "./checkpoints/qwen-mamba-hybrid"

    # Misc
    seed: int = 42
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    # Two-phase training
    phase1_steps: int = 0            # Steps for phase 1 (Mamba-only training). 0 = skip
    phase1_learning_rate: float = 1e-3
    phase1_target_ppl: float = 0.0   # Early-stop Phase 1 when PPL drops below this value (0 = disabled)
    phase2_target_ppl: float = 0.0   # Phase 2 plateau early-stop: only active when PPL <= this value (0 = disabled)
    phase2_plateau_steps: int = 500  # Phase 2 plateau early-stop: stop if no CE improvement for this many steps

    # Memory optimization (for single 24GB GPU)
    use_8bit_adam: bool = False      # Use bitsandbytes 8-bit AdamW (~4x less optimizer VRAM)
    teacher_on_cpu: bool = False     # Keep teacher on CPU, only move logits to GPU (~8GB saved)
    top_k_logits: int = 0            # Top-k logits distillation: only KD on top-k teacher tokens (0=full vocab, 50 recommended for memory saving)

    @property
    def mixed_precision_dtype(self) -> torch.dtype:
        if self.bf16:
            return torch.bfloat16
        elif self.fp16:
            return torch.float16
        return torch.float32


class DistillationLoss(nn.Module):
    """
    Combined distillation loss function.

    Computes:
        L = α_kd * KL_div(student, teacher, T) * T²
          + α_ce * CrossEntropy(student, labels)
          + α_hidden * MSE(student_hidden, teacher_hidden)
    """

    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.temperature = config.temperature
        self.alpha_kd = config.alpha_kd
        self.alpha_ce = config.alpha_ce
        self.alpha_hidden = config.alpha_hidden
        self.hidden_align_layers = config.hidden_align_layers
        self.top_k_logits = config.top_k_logits
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        student_logits: torch.Tensor,      # (B, L, V)
        teacher_logits: torch.Tensor,       # (B, L, V)
        labels: torch.Tensor,               # (B, L)
        student_hidden: Optional[Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]] = None,
        teacher_hidden: Optional[Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the combined distillation loss.

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains
            individual loss components for logging.
        """
        loss_dict = {}
        shift_labels = labels[..., 1:].contiguous()

        # --- KD Loss: KL divergence on soft targets ---
        if self.alpha_kd > 0:
            # Shift for next-token prediction
            student_logits_shifted = student_logits[..., :-1, :].contiguous()
            teacher_logits_shifted = teacher_logits[..., :-1, :].contiguous()

            if self.top_k_logits > 0:
                # Top-k logits distillation: only compute KD on top-k teacher tokens.
                # Captures >99% of probability mass, reduces peak logit memory by ~3000x.
                k = min(self.top_k_logits, teacher_logits_shifted.size(-1))
                # Get top-k indices from teacher
                topk_vals, topk_idx = teacher_logits_shifted.topk(k, dim=-1)  # (B, L-1, k)
                # Gather student logits at those same indices
                student_topk = student_logits_shifted.gather(-1, topk_idx)    # (B, L-1, k)
                # Compute softmax/log_softmax only over the k tokens
                student_log_probs = F.log_softmax(student_topk / self.temperature, dim=-1)
                teacher_probs = F.softmax(topk_vals / self.temperature, dim=-1)
                # Free full logit tensors ASAP
                del teacher_logits_shifted, student_logits_shifted
            else:
                # Full-vocab KD (original behaviour)
                student_log_probs = F.log_softmax(
                    student_logits_shifted / self.temperature, dim=-1
                )
                teacher_probs = F.softmax(
                    teacher_logits_shifted / self.temperature, dim=-1
                )

            # KL divergence averaged by valid tokens (not by batch only)
            kd_per_token = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="none",
            ).sum(dim=-1)

            valid_mask = (shift_labels != -100).float()
            kd_loss = ((kd_per_token * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)) * (self.temperature ** 2)

            loss_dict["kd_loss"] = kd_loss.item()
        else:
            kd_loss = torch.tensor(0.0, device=student_logits.device)

        # --- CE Loss: Hard label cross-entropy ---
        if self.alpha_ce > 0:
            shift_logits = student_logits[..., :-1, :].contiguous()

            ce_loss = self.ce_loss(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            loss_dict["ce_loss"] = ce_loss.item()
        else:
            ce_loss = torch.tensor(0.0, device=student_logits.device)

        # --- Hidden State Alignment Loss (Hint Loss) ---
        if self.alpha_hidden > 0 and student_hidden is not None and teacher_hidden is not None:
            if isinstance(student_hidden, torch.Tensor) and isinstance(teacher_hidden, torch.Tensor):
                hidden_loss = F.mse_loss(student_hidden.float(), teacher_hidden.float())
            elif isinstance(student_hidden, (list, tuple)) and isinstance(teacher_hidden, (list, tuple)):
                n_layers = min(len(student_hidden), len(teacher_hidden))
                if n_layers == 0:
                    hidden_loss = torch.tensor(0.0, device=student_logits.device)
                else:
                    if self.hidden_align_layers:
                        layer_indices = sorted({i for i in self.hidden_align_layers if 0 <= i < n_layers})
                    else:
                        layer_indices = sorted({n_layers // 4, n_layers // 2, (3 * n_layers) // 4})

                    if not layer_indices:
                        layer_indices = [n_layers - 1]

                    hidden_terms = []
                    for idx in layer_indices:
                        hidden_terms.append(
                            F.mse_loss(student_hidden[idx].float(), teacher_hidden[idx].float())
                        )
                    hidden_loss = torch.stack(hidden_terms).mean()
            else:
                hidden_loss = torch.tensor(0.0, device=student_logits.device)

            loss_dict["hidden_loss"] = hidden_loss.item()
        else:
            hidden_loss = torch.tensor(0.0, device=student_logits.device)

        # --- Total Loss ---
        total_loss = (
            self.alpha_kd * kd_loss
            + self.alpha_ce * ce_loss
            + self.alpha_hidden * hidden_loss
        )
        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict


class DistillationTrainer:
    """
    Trainer for knowledge distillation from Qwen teacher to hybrid student.

    Supports:
    - Mixed precision training (bf16/fp16)
    - Gradient accumulation
    - Gradient checkpointing
    - Two-phase training (Mamba-only → full model)
    - Learning rate scheduling with warmup
    - Periodic evaluation and checkpointing
    - TensorBoard / WandB logging
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        config: DistillationConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        tokenizer=None,
    ):
        self.student = student_model
        self.teacher = teacher_model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.tokenizer = tokenizer

        # Ensure teacher is frozen
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Loss function
        self.criterion = DistillationLoss(config)

        # Determine device
        self.device = next(self.student.parameters()).device

        # Teacher device: CPU offload saves ~8GB GPU VRAM
        self.teacher_device = torch.device("cpu") if config.teacher_on_cpu else self.device
        if config.teacher_on_cpu:
            self.teacher = self.teacher.to("cpu")
            logger.info("Teacher model kept on CPU to save GPU VRAM (~8 GB).")

        # Mixed precision
        self.use_amp = config.bf16 or config.fp16
        self.amp_dtype = config.mixed_precision_dtype

        # Gradient scaler (only for fp16, not bf16)
        self.scaler = GradScaler(enabled=config.fp16)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.total_steps = self._compute_total_steps()
        self.scheduler = self._create_scheduler()

        # State
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")

        # TensorBoard writer
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join(config.output_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        except ImportError:
            logger.warning("TensorBoard not available. Logging to console only.")

        # Gradient checkpointing
        if config.gradient_checkpointing:
            if hasattr(self.student, 'model') and hasattr(self.student.model, 'gradient_checkpointing_enable'):
                self.student.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled for student model.")
            elif hasattr(self.student, 'gradient_checkpointing_enable'):
                self.student.gradient_checkpointing_enable()

    def _create_optimizer(self, params=None) -> AdamW:
        """Create AdamW optimizer with weight decay. Supports 8-bit Adam via bitsandbytes."""
        # Separate parameters that should and shouldn't have weight decay
        no_decay = ["bias", "layernorm", "layer_norm", "rmsnorm", "rms_norm"]
        decay_params = []
        no_decay_params = []

        param_iter = params if params is not None else [
            (n, p) for n, p in self.student.named_parameters() if p.requires_grad
        ]
        for name, param in param_iter:
            if any(nd in name.lower() for nd in no_decay) or hasattr(param, '_no_weight_decay'):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if self.config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                logger.info("Using bitsandbytes 8-bit AdamW (saves ~12 GB optimizer VRAM).")
                return bnb.optim.AdamW8bit(
                    optimizer_groups,
                    lr=self.config.learning_rate,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                )
            except ImportError:
                logger.warning("bitsandbytes not installed, falling back to FP32 AdamW. "
                               "Install with: pip install bitsandbytes")

        return AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    def _compute_total_steps(self) -> int:
        """Compute total training steps."""
        if self.config.max_steps > 0:
            return self.config.max_steps

        steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps
        )
        return steps_per_epoch * self.config.num_train_epochs

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        if self.config.warmup_steps > 0:
            warmup_steps = self.config.warmup_steps
        else:
            warmup_steps = int(self.total_steps * self.config.warmup_ratio)

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-7 / max(self.config.learning_rate, 1e-10),
            end_factor=1.0,
            total_iters=max(warmup_steps, 1),
        )

        if self.config.lr_scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(self.total_steps - warmup_steps, 1),
                eta_min=self.config.learning_rate * 0.1,
            )
        else:
            # Linear decay
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=max(self.total_steps - warmup_steps, 1),
            )

        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    def _training_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform a single training step.

        Args:
            batch: Dictionary with 'input_ids', 'attention_mask', 'labels'

        Returns:
            Tuple of (loss, loss_dict)
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch.get("labels", input_ids.clone()).to(self.device)

        with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            # Teacher forward (no gradient)
            # If teacher is on CPU, run inference there then move logits to GPU
            with torch.no_grad():
                if self.config.teacher_on_cpu:
                    teacher_outputs = self.teacher(
                        input_ids=input_ids.to("cpu"),
                        attention_mask=attention_mask.to("cpu"),
                        output_hidden_states=(self.config.alpha_hidden > 0),
                    )
                    teacher_logits = teacher_outputs.logits.to(self.device)
                    teacher_hidden = None
                    if self.config.alpha_hidden > 0 and hasattr(teacher_outputs, 'hidden_states'):
                        teacher_hidden = [h.to(self.device) for h in teacher_outputs.hidden_states[1:]]
                else:
                    teacher_outputs = self.teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=(self.config.alpha_hidden > 0),
                    )
                    teacher_logits = teacher_outputs.logits
                    teacher_hidden = None
                    if self.config.alpha_hidden > 0 and hasattr(teacher_outputs, 'hidden_states'):
                        teacher_hidden = list(teacher_outputs.hidden_states[1:])

            # Student forward
            student_outputs = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=(self.config.alpha_hidden > 0),
            )
            student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else student_outputs[0]
            student_hidden = None
            if self.config.alpha_hidden > 0 and hasattr(student_outputs, 'hidden_states'):
                student_hidden = list(student_outputs.hidden_states[1:])

            # Compute distillation loss
            loss, loss_dict = self.criterion(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                student_hidden=student_hidden,
                teacher_hidden=teacher_hidden,
            )

            # Scale by gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

        return loss, loss_dict

    def train(self):
        """
        Main training loop.

        Implements:
        - Phase 1 (optional): Train only Mamba layers
        - Phase 2: Train all parameters with distillation
        """
        logger.info("=" * 60)
        logger.info("  Starting Knowledge Distillation Training")
        logger.info("=" * 60)
        logger.info(f"  Total steps:          {self.total_steps}")
        logger.info(f"  Gradient accum steps: {self.config.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"  Learning rate:        {self.config.learning_rate}")
        logger.info(f"  Temperature:          {self.config.temperature}")
        logger.info(f"  Alpha (KD):           {self.config.alpha_kd}")
        logger.info(f"  Alpha (CE):           {self.config.alpha_ce}")
        logger.info(f"  Alpha (Hidden):       {self.config.alpha_hidden}")
        logger.info(f"  Hidden align layers:  {self.config.hidden_align_layers if self.config.hidden_align_layers else 'auto'}")
        logger.info(f"  Mixed precision:      {self.amp_dtype}")
        logger.info("=" * 60)

        # Phase 1: Mamba-only training (optional)
        if self.config.phase1_steps > 0:
            self._train_phase1()

        # Phase 2: Full distillation training
        self._train_phase2()

        # Close writer
        if self.writer is not None:
            self.writer.close()

        logger.info("Training completed!")
        return self.global_step

    def _train_phase1(self):
        """
        Phase 1: Train only Mamba layers with higher learning rate.

        This phase helps the randomly initialized Mamba layers catch up
        before fine-tuning the entire model.
        """
        from ..models.architecture_surgery import freeze_non_mamba_parameters, unfreeze_all_parameters

        logger.info("\n--- Phase 1: Mamba-only training ---")
        freeze_non_mamba_parameters(self.student, unfreeze_lm_head=False)

        # Create a temporary optimizer for phase 1
        phase1_optimizer = AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=self.config.phase1_learning_rate,
            betas=(0.9, 0.95),
        )

        self.student.train()
        step = 0
        accum_loss = 0.0
        accum_loss_dict = {}
        best_phase1_ce = float("inf")

        for batch in self.train_dataloader:
            loss, loss_dict = self._training_step(batch)
            self.scaler.scale(loss).backward()

            accum_loss += loss_dict["total_loss"]
            for k, v in loss_dict.items():
                accum_loss_dict[k] = accum_loss_dict.get(k, 0.0) + v
            step += 1

            if step % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(phase1_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.student.parameters() if p.requires_grad],
                    self.config.max_grad_norm,
                )
                self.scaler.step(phase1_optimizer)
                self.scaler.update()
                phase1_optimizer.zero_grad()

                opt_step = step // self.config.gradient_accumulation_steps

                if opt_step % self.config.logging_steps == 0:
                    avg_loss = accum_loss / (self.config.logging_steps * self.config.gradient_accumulation_steps)
                    n = self.config.logging_steps * self.config.gradient_accumulation_steps
                    avg_losses = {k: v / n for k, v in accum_loss_dict.items()}
                    ce = avg_losses.get("ce_loss", 0.0)
                    ppl = math.exp(min(ce, 20))
                    logger.info(
                        f"[Phase 1] Step {opt_step}/{self.config.phase1_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"KD: {avg_losses.get('kd_loss', 0):.4f} | "
                        f"CE: {ce:.4f} | "
                        f"PPL: {ppl:.2f}"
                    )
                    if self.writer is not None:
                        for k, v in avg_losses.items():
                            self.writer.add_scalar(f"phase1/{k}", v, opt_step)
                        self.writer.add_scalar("phase1/ppl", ppl, opt_step)
                    accum_loss = 0.0
                    accum_loss_dict = {}

                    # Save best phase1 checkpoint based on CE loss
                    # Only write to disk when PPL is already below target (avoids frequent disk writes early on)
                    ppl_eligible = (self.config.phase1_target_ppl <= 0 or ppl <= self.config.phase1_target_ppl)
                    if ce < best_phase1_ce:
                        best_phase1_ce = ce
                        if ppl_eligible:
                            self._save_phase1_checkpoint(opt_step, phase1_optimizer, is_best=True)
                            logger.info(f"[Phase 1] New best CE: {ce:.4f}, PPL: {ppl:.2f} — saved best checkpoint")

                    # Early stop if PPL target reached
                    if self.config.phase1_target_ppl > 0 and ppl <= self.config.phase1_target_ppl:
                        logger.info(
                            f"[Phase 1] PPL {ppl:.2f} ≤ target {self.config.phase1_target_ppl:.1f} "
                            f"at step {opt_step} — early stopping Phase 1"
                        )
                        self._save_phase1_checkpoint(opt_step, phase1_optimizer, is_final=True)
                        unfreeze_all_parameters(self.student)
                        logger.info("Phase 1 complete (early stop). All parameters unfrozen.\n")
                        return

                if self.config.save_steps > 0 and opt_step % self.config.save_steps == 0:
                    self._save_phase1_checkpoint(opt_step, phase1_optimizer)
                    # Keep only the latest phase1 checkpoint to save disk space
                    self._cleanup_phase1_checkpoints(opt_step)

                if opt_step >= self.config.phase1_steps:
                    break

        self._save_phase1_checkpoint(self.config.phase1_steps, phase1_optimizer, is_final=True)

        # Unfreeze all for phase 2
        unfreeze_all_parameters(self.student)
        logger.info("Phase 1 complete. All parameters unfrozen.\n")

    def _train_phase2(self):
        """Phase 2: Full distillation training with all parameters."""
        logger.info("\n--- Phase 2: Full distillation training ---")

        self.student.train()
        self.optimizer.zero_grad()

        accum_loss = 0.0
        accum_loss_dict = {}
        start_time = time.time()
        best_phase2_ce = float("inf")
        last_improvement_step = 0

        for epoch in range(self.config.num_train_epochs):
            self.epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_train_epochs}")

            for step_in_epoch, batch in enumerate(self.train_dataloader):
                # Forward + backward
                loss, loss_dict = self._training_step(batch)
                self.scaler.scale(loss).backward()

                # Accumulate losses for logging
                accum_loss += loss_dict["total_loss"]
                for k, v in loss_dict.items():
                    accum_loss_dict[k] = accum_loss_dict.get(k, 0.0) + v

                # Optimizer step at accumulation boundary
                if (step_in_epoch + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(),
                        self.config.max_grad_norm,
                    )

                    # Step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        elapsed = time.time() - start_time
                        n = self.config.logging_steps * self.config.gradient_accumulation_steps
                        avg_losses = {k: v / n for k, v in accum_loss_dict.items()}

                        lr = self.scheduler.get_last_lr()[0]
                        tokens_per_sec = (
                            self.config.per_device_train_batch_size
                            * self.config.gradient_accumulation_steps
                            * self.config.logging_steps
                            * batch["input_ids"].shape[1]
                            / elapsed
                        )

                        ce = avg_losses.get('ce_loss', 0)
                        ppl = math.exp(min(ce, 20))

                        logger.info(
                            f"Step {self.global_step}/{self.total_steps} | "
                            f"Loss: {avg_losses.get('total_loss', 0):.4f} | "
                            f"KD: {avg_losses.get('kd_loss', 0):.4f} | "
                            f"CE: {ce:.4f} | "
                            f"PPL: {ppl:.2f} | "
                            f"LR: {lr:.2e} | "
                            f"Tok/s: {tokens_per_sec:.0f}"
                        )

                        # TensorBoard logging
                        if self.writer is not None:
                            for k, v in avg_losses.items():
                                self.writer.add_scalar(f"train/{k}", v, self.global_step)
                            self.writer.add_scalar("train/lr", lr, self.global_step)
                            self.writer.add_scalar("train/tokens_per_sec", tokens_per_sec, self.global_step)

                        # Track best CE and plateau early-stop
                        if ce < best_phase2_ce:
                            best_phase2_ce = ce
                            last_improvement_step = self.global_step
                        if (
                            self.config.phase2_target_ppl > 0
                            and ppl <= self.config.phase2_target_ppl
                            and (self.global_step - last_improvement_step) >= self.config.phase2_plateau_steps
                        ):
                            logger.info(
                                f"[Phase 2] Plateau early-stop: PPL {ppl:.2f} ≤ target {self.config.phase2_target_ppl:.1f}, "
                                f"no improvement for {self.global_step - last_improvement_step} steps "
                                f"(best CE={best_phase2_ce:.4f} at step {last_improvement_step})"
                            )
                            self._save_checkpoint(is_final=True)
                            return

                        accum_loss = 0.0
                        accum_loss_dict = {}
                        start_time = time.time()

                    # Evaluation
                    if self.eval_dataloader is not None and self.global_step % self.config.eval_steps == 0:
                        eval_loss = self.evaluate()
                        self.student.train()

                    # Checkpointing
                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint()

                    # Check max steps
                    if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                        logger.info(f"Reached max_steps ({self.config.max_steps}). Stopping.")
                        self._save_checkpoint(is_final=True)
                        return

        # Save final checkpoint
        self._save_checkpoint(is_final=True)

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Run evaluation on the eval dataset.

        Returns:
            Average evaluation loss
        """
        if self.eval_dataloader is None:
            return float("inf")

        self.student.eval()
        total_loss = 0.0
        total_ce_loss = 0.0
        num_batches = 0

        for batch in self.eval_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch.get("labels", input_ids.clone()).to(self.device)

            with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                # Teacher
                teacher_outputs = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # Student
                student_outputs = self.student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else student_outputs[0]

                loss, loss_dict = self.criterion(
                    student_logits=student_logits,
                    teacher_logits=teacher_outputs.logits,
                    labels=labels,
                )

            total_loss += loss_dict["total_loss"]
            total_ce_loss += loss_dict.get("ce_loss", 0)
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_ce = total_ce_loss / max(num_batches, 1)
        ppl = math.exp(min(avg_ce, 20))  # Cap to avoid overflow

        logger.info(
            f"[Eval] Step {self.global_step} | "
            f"Loss: {avg_loss:.4f} | CE: {avg_ce:.4f} | PPL: {ppl:.2f}"
        )

        if self.writer is not None:
            self.writer.add_scalar("eval/loss", avg_loss, self.global_step)
            self.writer.add_scalar("eval/ce_loss", avg_ce, self.global_step)
            self.writer.add_scalar("eval/perplexity", ppl, self.global_step)

        if avg_loss < self.best_eval_loss:
            self.best_eval_loss = avg_loss
            self._save_checkpoint(is_best=True)

        return avg_loss

    def _save_checkpoint(self, is_final: bool = False, is_best: bool = False):
        """Save a training checkpoint."""
        if is_best:
            save_dir = os.path.join(self.config.output_dir, "best")
        elif is_final:
            save_dir = os.path.join(self.config.output_dir, "final")
        else:
            save_dir = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")

        os.makedirs(save_dir, exist_ok=True)

        # Save model
        if hasattr(self.student, 'save_pretrained'):
            self.student.save_pretrained(save_dir)
        else:
            torch.save(self.student.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_eval_loss": self.best_eval_loss,
        }
        torch.save(training_state, os.path.join(save_dir, "training_state.pt"))

        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_dir)

        logger.info(f"Checkpoint saved to {save_dir}")

        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def _save_phase1_checkpoint(self, phase1_step: int, phase1_optimizer: torch.optim.Optimizer, is_final: bool = False, is_best: bool = False):
        """Save phase1 checkpoint (Mamba-only warmup stage)."""
        if is_best:
            save_dir = os.path.join(self.config.output_dir, "phase1-best")
        elif is_final:
            save_dir = os.path.join(self.config.output_dir, "phase1-final")
        else:
            save_dir = os.path.join(self.config.output_dir, f"phase1-checkpoint-{phase1_step}")

        os.makedirs(save_dir, exist_ok=True)

        if hasattr(self.student, 'save_pretrained'):
            self.student.save_pretrained(save_dir)
        else:
            torch.save(self.student.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

        training_state = {
            "phase": "phase1",
            "phase1_step": phase1_step,
            "optimizer_state_dict": phase1_optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }
        torch.save(training_state, os.path.join(save_dir, "training_state.pt"))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_dir)

        logger.info(f"Phase 1 checkpoint saved to {save_dir}")

    def _cleanup_phase1_checkpoints(self, current_step: int):
        """Keep only the latest phase1 checkpoint (saves disk space)."""
        output_dir = self.config.output_dir
        if not os.path.exists(output_dir):
            return
        for name in os.listdir(output_dir):
            if name.startswith("phase1-checkpoint-"):
                try:
                    step = int(name.split("-")[-1])
                except (ValueError, IndexError):
                    continue
                if step != current_step:
                    import shutil
                    path = os.path.join(output_dir, name)
                    shutil.rmtree(path, ignore_errors=True)
                    logger.info(f"Removed old phase1 checkpoint: {path}")

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only save_total_limit."""
        if self.config.save_total_limit <= 0:
            return

        checkpoint_dirs = []
        output_dir = self.config.output_dir
        if not os.path.exists(output_dir):
            return

        for name in os.listdir(output_dir):
            if name.startswith("checkpoint-"):
                full_path = os.path.join(output_dir, name)
                if os.path.isdir(full_path):
                    try:
                        step = int(name.split("-")[1])
                        checkpoint_dirs.append((step, full_path))
                    except (ValueError, IndexError):
                        pass

        checkpoint_dirs.sort(key=lambda x: x[0])

        # Remove oldest checkpoints
        while len(checkpoint_dirs) > self.config.save_total_limit:
            _, path = checkpoint_dirs.pop(0)
            import shutil
            shutil.rmtree(path, ignore_errors=True)
            logger.info(f"Removed old checkpoint: {path}")

    def resume_from_checkpoint(self, checkpoint_dir: str):
        """Resume training from a checkpoint."""
        training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.exists(training_state_path):
            state = torch.load(training_state_path, map_location="cpu")
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
            self.scaler.load_state_dict(state["scaler_state_dict"])
            self.best_eval_loss = state.get("best_eval_loss", float("inf"))
            logger.info(f"Resumed from step {self.global_step}")
        else:
            logger.warning(f"No training state found in {checkpoint_dir}")
