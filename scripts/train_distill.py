#!/usr/bin/env python3
"""
Step 2: Knowledge Distillation Training.

Supports two distillation modes:
1. Local teacher: Load Qwen3-4B as teacher (needs ~16GB+ GPU per card)
2. API teacher:   Use Qwen-Max API as teacher (only ~10GB for student)

And two hybrid architectures:
- mamba:          Selective State Space Model
- gated_deltanet: Gated Delta Rule Network

Usage:
    # Mamba hybrid with local teacher
    python scripts/train_distill.py --linear_type mamba --dummy --max_steps 50

    # GatedDeltaNet hybrid with local teacher
    python scripts/train_distill.py --linear_type gated_deltanet --dummy --max_steps 50

    # API distillation (Qwen-Max, single GPU)
    python scripts/train_distill.py --linear_type mamba --use_api \
        --api_model qwen-max --dummy --max_steps 50

    # Multi-GPU with DeepSpeed
    deepspeed scripts/train_distill.py --linear_type mamba \
        --deepspeed configs/ds_config_zero2.json

    # Resume
    python scripts/train_distill.py --resume_from checkpoints/checkpoint-1000
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.models.architecture_surgery import (
    convert_qwen_to_hybrid,
    load_tokenizer,
    freeze_non_linear_parameters,
)
from src.training.distillation import DistillationTrainer, DistillationConfig
from src.training.data import build_dataset, build_dataloader, create_dummy_dataset
from src.utils.helpers import (
    load_config, setup_logging, print_model_summary, set_seed, get_gpu_memory_info,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Distillation Training for Qwen Hybrid")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--linear_type", type=str, default=None,
                        choices=["mamba", "gated_deltanet"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Phase 2 全参数训练学习率 (默认 5e-4，推荐 1e-4)")
    parser.add_argument("--phase1_learning_rate", type=float, default=None,
                        help="Phase 1 Mamba层预热学习率 (默认 1e-3)")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--alpha_kd", type=float, default=None)
    parser.add_argument("--alpha_ce", type=float, default=None)
    parser.add_argument("--alpha_hidden", type=float, default=None,
                        help="中间层 Hint Loss 权重，建议 0.02~0.1")
    parser.add_argument("--hidden_align_layers", type=str, default=None,
                        help="中间层对齐层索引，逗号分隔，如 8,16,24；不传则自动选中间层")
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--phase1_steps", type=int, default=None)
    parser.add_argument("--phase1_target_ppl", type=float, default=None,
                        help="Phase 1 早停 PPL 阈值：PPL 降到此值以下自动结束 Phase 1，进入 Phase 2（0=不启用）")
    parser.add_argument("--phase2_target_ppl", type=float, default=None,
                        help="Phase 2 plateau 早停：PPL 降到此阈值且连续 plateau_steps 步无改善则停止（0=不启用）")
    parser.add_argument("--phase2_plateau_steps", type=int, default=None,
                        help="Phase 2 plateau 判定窗口步数，默认 500")
    parser.add_argument("--save_total_limit", type=int, default=None,
                        help="Phase 2 最多保留几个 checkpoint，默认 2")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--dummy_samples", type=int, default=200)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--text_file", type=str, default=None)
    parser.add_argument("--processed_dataset", type=str, default=None,
                        help="直接指定已经 tokenize+group 好的 datasets 目录（load_from_disk），跳过所有预处理")
    parser.add_argument("--dataset_cache_dir", type=str, default="/root/autodl-tmp/dataset_cache",
                        help="Arrow cache directory for tokenized parquet data (shared across runs)")
    # API distillation
    parser.add_argument("--use_api", action="store_true",
                        help="Use API-based distillation (Qwen-Max)")
    parser.add_argument("--api_model", type=str, default="qwen-max")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--cached_data_dir", type=str, default=None,
                        help="预生成的 API 缓存数据目录 (跳过 API 调用, 直接训练)")
    # Memory optimization
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="使用 bitsandbytes 8-bit AdamW，节省约 12GB 优化器显存 (单卡4090推荐)")
    parser.add_argument("--teacher_on_cpu", action="store_true",
                        help="教师模型保留在 CPU 推理，节省约 8GB GPU 显存 (单卡4090推荐)")
    parser.add_argument("--no_grad_checkpoint", action="store_true",
                        help="关闭 gradient checkpointing，提升训练速度但占用更多显存 (5090 32GB 可选)")
    parser.add_argument("--top_k_logits", type=int, default=50,
                        help="Top-k logits 蒸馏：只对 teacher 概率最高的 k 个 token 算 KD loss，大幅降低显存峰值 (0=全词表, 默认50)")

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)

    config = load_config(args.config)
    model_config = config.get("model", {})
    mamba_config = config.get("mamba", {})
    gdn_config = config.get("gated_deltanet", {})
    distill_config = config.get("distillation", {})
    train_config = config.get("training", {})

    base_model = args.base_model or model_config.get("base_model_name", "Qwen/Qwen3-4B")
    linear_type = args.linear_type or model_config.get("linear_layer_type", "mamba")
    default_output = f"./checkpoints/qwen3-4b-{linear_type}"
    output_dir = args.output_dir or model_config.get("output_dir", default_output)
    seed = args.seed or train_config.get("seed", 42)
    set_seed(seed)
    max_seq_length = args.max_seq_length or train_config.get("max_seq_length", 2048)
    hidden_align_layers = distill_config.get("hidden_align_layers", [])
    if args.hidden_align_layers is not None:
        hidden_align_layers = [int(x.strip()) for x in args.hidden_align_layers.split(",") if x.strip()]

    # Distillation config
    distill_cfg = DistillationConfig(
        temperature=args.temperature or distill_config.get("temperature", 2.0),
        alpha_kd=args.alpha_kd or distill_config.get("alpha_kd", 0.5),
        alpha_ce=args.alpha_ce or distill_config.get("alpha_ce", 0.5),
        alpha_hidden=args.alpha_hidden if args.alpha_hidden is not None else distill_config.get("alpha_hidden", 0.0),
        hidden_align_layers=hidden_align_layers,
        learning_rate=args.learning_rate or train_config.get("learning_rate", 5e-4),
        phase1_learning_rate=args.phase1_learning_rate or train_config.get("phase1_learning_rate", 1e-3),
        weight_decay=train_config.get("weight_decay", 0.01),
        max_grad_norm=train_config.get("max_grad_norm", 1.0),
        warmup_ratio=train_config.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        num_train_epochs=train_config.get("num_train_epochs", 1),
        max_steps=args.max_steps if args.max_steps is not None else train_config.get("max_steps", -1),
        per_device_train_batch_size=args.batch_size or train_config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=args.grad_accum or train_config.get("gradient_accumulation_steps", 16),
        bf16=train_config.get("bf16", True),
        fp16=train_config.get("fp16", False),
        gradient_checkpointing=False if args.no_grad_checkpoint else train_config.get("gradient_checkpointing", True),
        logging_steps=train_config.get("logging_steps", 10),
        save_steps=train_config.get("save_steps", 500),
        eval_steps=train_config.get("eval_steps", 500),
        save_total_limit=args.save_total_limit if args.save_total_limit is not None else train_config.get("save_total_limit", 2),
        output_dir=output_dir,
        seed=seed,
        dataloader_num_workers=train_config.get("dataloader_num_workers", 4),
        phase1_steps=args.phase1_steps or 0,
        phase1_target_ppl=args.phase1_target_ppl if args.phase1_target_ppl is not None else 0.0,
        phase2_target_ppl=args.phase2_target_ppl if args.phase2_target_ppl is not None else 0.0,
        phase2_plateau_steps=args.phase2_plateau_steps if args.phase2_plateau_steps is not None else 500,
        use_8bit_adam=args.use_8bit_adam,
        teacher_on_cpu=args.teacher_on_cpu,
        top_k_logits=args.top_k_logits,
    )

    logger.info("=" * 60)
    logger.info(f"  Distillation Training ({linear_type})")
    logger.info("=" * 60)
    logger.info(f"  Base model:      {base_model}")
    logger.info(f"  Linear type:     {linear_type}")
    logger.info(f"  Use API:         {args.use_api}")
    logger.info(f"  Output dir:      {output_dir}")
    logger.info("=" * 60)

    # GPU info
    gpu_info = get_gpu_memory_info()
    if gpu_info.get("available", True) is not False:
        for k, v in gpu_info.items():
            if isinstance(v, dict):
                logger.info(f"  {k}: {v.get('name', '?')} - {v.get('total_gb', '?')} GB")

    # --- API Distillation Mode ---
    if args.use_api:
        _run_api_distillation(args, base_model, linear_type, mamba_config, gdn_config,
                              distill_cfg, max_seq_length, output_dir, logger)
        return

    # --- Local Teacher Distillation Mode ---
    attention_interval = model_config.get("attention_interval", 4)

    hybrid_model, teacher_model = convert_qwen_to_hybrid(
        model_name_or_path=base_model,
        attention_interval=attention_interval,
        linear_layer_type=linear_type,
        mamba_d_state=mamba_config.get("d_state", 16),
        mamba_d_conv=mamba_config.get("d_conv", 4),
        mamba_expand=mamba_config.get("expand", 2),
        gdn_key_head_dim=gdn_config.get("key_head_dim", 128),
        gdn_value_head_dim=gdn_config.get("value_head_dim", 128),
        gdn_conv_kernel=gdn_config.get("conv_kernel", 4),
        torch_dtype=torch.bfloat16 if distill_cfg.bf16 else torch.float16 if distill_cfg.fp16 else torch.float32,
        load_teacher=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hybrid_model = hybrid_model.to(device)
    # 教师模型：按配置决定是否放 GPU
    # --teacher_on_cpu: 教师留在 CPU, 省约 8GB GPU 显存 (DistillationTrainer 内部处理)
    if not args.teacher_on_cpu:
        teacher_model = teacher_model.to(device)
    else:
        logger.info("Teacher model will stay on CPU (--teacher_on_cpu). Saves ~8 GB GPU VRAM.")

    print_model_summary(hybrid_model, title=f"Student ({linear_type})")

    tokenizer = load_tokenizer(base_model)

    # Dataset
    if args.dummy:
        train_dataset = create_dummy_dataset(tokenizer, num_samples=args.dummy_samples, max_seq_length=max_seq_length)
        eval_dataset = create_dummy_dataset(tokenizer, num_samples=50, max_seq_length=max_seq_length)
    else:
        dataset_name = args.dataset_name or train_config.get("dataset_name", "HuggingFaceFW/fineweb-edu")
        dataset_config_name = train_config.get("dataset_config", "sample-10BT")
        if args.processed_dataset:
            train_dataset = build_dataset(tokenizer=tokenizer,
                                          max_seq_length=max_seq_length,
                                          processed_dataset=args.processed_dataset)
        elif args.text_file:
            train_dataset = build_dataset(tokenizer=tokenizer, text_file=args.text_file,
                                          max_seq_length=max_seq_length, cache_dir=args.dataset_cache_dir,
                                          processed_dataset=None)
        else:
            train_dataset = build_dataset(
                tokenizer=tokenizer, dataset_name=dataset_name, dataset_config=dataset_config_name,
                dataset_split=train_config.get("dataset_split", "train"),
                max_seq_length=max_seq_length,
                preprocessing_num_workers=train_config.get("preprocessing_num_workers", 8),
                streaming=True,
            )
        eval_dataset = None

    train_dataloader = build_dataloader(
        dataset=train_dataset, batch_size=distill_cfg.per_device_train_batch_size,
        shuffle=True, num_workers=distill_cfg.dataloader_num_workers,
        pin_memory=distill_cfg.dataloader_pin_memory,
    )
    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = build_dataloader(
            dataset=eval_dataset, batch_size=distill_cfg.per_device_eval_batch_size,
            shuffle=False, num_workers=distill_cfg.dataloader_num_workers,
            pin_memory=distill_cfg.dataloader_pin_memory,
        )

    trainer = DistillationTrainer(
        student_model=hybrid_model, teacher_model=teacher_model,
        config=distill_cfg, train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader, tokenizer=tokenizer,
    )

    if args.resume_from:
        trainer.resume_from_checkpoint(args.resume_from)

    total_steps = trainer.train()
    logger.info(f"\nTraining complete! Steps: {total_steps}, saved to: {output_dir}")


def _run_api_distillation(args, base_model, linear_type, mamba_config, gdn_config,
                          distill_cfg, max_seq_length, output_dir, logger):
    """
    Run API-based distillation.

    Supports three data source modes:
    1. --cached_data_dir: 直接从预生成的缓存目录加载 (推荐, 不调 API)
    2. --dummy:           使用随机数据测试流程
    3. 默认:              在线调 API 生成数据 (会阻塞等待)
    """
    from src.training.api_distillation import (
        APIDistillConfig, APIDistillationTrainer, QwenAPIClient,
        APIGeneratedDataset, generate_training_data, load_cached_data,
        DEFAULT_SEED_PROMPTS,
    )
    from src.training.data import build_dataloader

    # Create hybrid model (no teacher needed locally)
    hybrid_model, _ = convert_qwen_to_hybrid(
        model_name_or_path=base_model,
        linear_layer_type=linear_type,
        mamba_d_state=mamba_config.get("d_state", 16),
        mamba_d_conv=mamba_config.get("d_conv", 4),
        mamba_expand=mamba_config.get("expand", 2),
        gdn_key_head_dim=gdn_config.get("key_head_dim", 128),
        gdn_value_head_dim=gdn_config.get("value_head_dim", 128),
        gdn_conv_kernel=gdn_config.get("conv_kernel", 4),
        torch_dtype=torch.bfloat16 if distill_cfg.bf16 else torch.float32,
        load_teacher=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hybrid_model = hybrid_model.to(device)

    tokenizer = load_tokenizer(base_model)

    # API config
    api_config = APIDistillConfig(
        api_key=args.api_key or os.environ.get("DASHSCOPE_API_KEY", ""),
        teacher_model=args.api_model,
        learning_rate=distill_cfg.learning_rate,
        max_seq_length=max_seq_length,
        bf16=distill_cfg.bf16,
        gradient_checkpointing=distill_cfg.gradient_checkpointing,
        per_device_batch_size=distill_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=distill_cfg.gradient_accumulation_steps,
        max_steps=distill_cfg.max_steps,
        output_dir=output_dir,
    )

    # --- 数据加载 (三种模式) ---
    if args.dummy:
        # 模式 1: 随机数据测试
        from src.training.data import create_dummy_dataset
        logger.info("使用随机 dummy 数据测试流程...")
        train_dataset = create_dummy_dataset(tokenizer, num_samples=200, max_seq_length=max_seq_length)

    elif args.cached_data_dir:
        # 模式 2 (推荐): 从预生成缓存目录加载, 不调 API
        logger.info(f"从缓存目录加载预生成数据: {args.cached_data_dir}")
        logger.info("  (不会调用 API, GPU 立即开始训练)")
        texts, logprobs = load_cached_data(args.cached_data_dir)
        train_dataset = APIGeneratedDataset(tokenizer, texts, max_seq_length, logprobs)
        logger.info(f"  加载完成: {len(train_dataset)} 条训练样本")

    else:
        # 模式 3: 在线调 API 生成 (会阻塞)
        logger.warning(
            "即将在线调用 API 生成数据, GPU 在此期间空闲!\n"
            "  建议: 先在本地运行 generate_api_data.py 预生成数据, 再用 --cached_data_dir 加载。\n"
            "  示例: python scripts/generate_api_data.py --num_samples 5000 --output_dir ./cache/api_distill\n"
            "        python scripts/train_distill.py --use_api --cached_data_dir ./cache/api_distill"
        )
        api_client = QwenAPIClient(api_config)
        prompts = DEFAULT_SEED_PROMPTS * 5
        logger.info(f"Generating {len(prompts)} training samples from API...")
        texts, logprobs = generate_training_data(api_client, prompts, api_config, tokenizer)
        train_dataset = APIGeneratedDataset(tokenizer, texts, max_seq_length, logprobs)

    train_dataloader = build_dataloader(
        dataset=train_dataset, batch_size=api_config.per_device_batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )

    trainer = APIDistillationTrainer(
        student_model=hybrid_model, config=api_config,
        train_dataloader=train_dataloader, tokenizer=tokenizer,
    )

    total_steps = trainer.train()
    logger.info(f"\nAPI distillation complete! Steps: {total_steps}")


if __name__ == "__main__":
    main()
