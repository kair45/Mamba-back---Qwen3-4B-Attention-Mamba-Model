"""
Data Loading Utilities for Qwen-Mamba Hybrid Training.

Provides functions to load and preprocess text datasets for:
- Knowledge distillation (causal LM format)
- Continued pre-training

Supports HuggingFace datasets and custom text files.
"""

import logging
import os
import numpy as np
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    Simple text dataset that tokenizes and chunks text into fixed-length sequences.

    Used as a fallback when HuggingFace datasets are not available.
    """

    def __init__(self, file_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        logger.info(f"Loading text file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize the entire text
        tokens = tokenizer.encode(text)

        # Chunk into fixed-length sequences
        for i in range(0, len(tokens) - max_length, max_length):
            self.examples.append(tokens[i : i + max_length])

        logger.info(f"Created {len(self.examples)} examples of length {max_length}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        input_ids = torch.tensor(tokens, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }


def build_dataset(
    tokenizer,
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
    dataset_split: str = "train",
    text_file: Optional[str] = None,
    max_seq_length: int = 2048,
    preprocessing_num_workers: int = 8,
    max_samples: Optional[int] = None,
    streaming: bool = False,
    cache_dir: Optional[str] = None,
    processed_dataset: Optional[str] = None,
) -> Dataset:
    """
    Build a training dataset from HuggingFace hub or local text files.

    Args:
        tokenizer: The tokenizer to use
        dataset_name: HuggingFace dataset name (e.g., "HuggingFaceFW/fineweb-edu")
        dataset_config: Dataset config/subset name
        dataset_split: Which split to use
        text_file: Path to a local text file (alternative to dataset_name)
        max_seq_length: Maximum sequence length
        preprocessing_num_workers: Number of workers for preprocessing
        max_samples: Maximum number of samples (for debugging)
        streaming: Whether to use streaming mode for large datasets

    Returns:
        A PyTorch Dataset
    """
    # --- Shortcut: directly load pre-processed dataset from disk (highest priority) ---
    if processed_dataset is not None:
        import glob as _g
        if os.path.isdir(processed_dataset):
            from datasets import load_from_disk
            logger.info(f"Loading pre-processed dataset from: {processed_dataset}")
            ds = load_from_disk(processed_dataset)
        elif "*" in processed_dataset or processed_dataset.endswith(".arrow"):
            from datasets import Dataset, concatenate_datasets
            arrow_files = sorted(_g.glob(processed_dataset)) if "*" in processed_dataset else [processed_dataset]
            if not arrow_files:
                raise ValueError(f"No Arrow files found matching: {processed_dataset}")
            logger.info(f"Loading {len(arrow_files)} Arrow cache files...")
            ds = concatenate_datasets([Dataset.from_file(f) for f in arrow_files])
        else:
            raise ValueError(f"--processed_dataset path not found: {processed_dataset}")
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        logger.info(f"Dataset ready: {len(ds)} chunks of {max_seq_length} tokens")
        return ds

    if text_file is not None:
        import glob as _glob
        if text_file.endswith(".parquet") or os.path.isdir(text_file):
            from datasets import load_dataset as _load_ds

            if os.path.isdir(text_file):
                parquet_files = sorted(_glob.glob(os.path.join(text_file, "**", "*.parquet"), recursive=True))
                if not parquet_files:
                    raise ValueError(f"No parquet files found in directory: {text_file}")
            else:
                parquet_files = [text_file]

            # Arrow cache dir: shared between Mamba and GDN training
            _cache_dir = cache_dir or os.path.join(os.path.dirname(parquet_files[0]), ".arrow_cache")
            os.makedirs(_cache_dir, exist_ok=True)
            logger.info(f"Loading {len(parquet_files)} parquet shards, Arrow cache: {_cache_dir}")

            ds = _load_ds(
                "parquet",
                data_files={"train": parquet_files},
                split="train",
                cache_dir=_cache_dir,
            )

            text_col = next((c for c in ["text", "content", "document", "passage"]
                             if c in ds.column_names), ds.column_names[0])
            logger.info(f"Text column: '{text_col}', total rows: {len(ds)}")

            def _tokenize(examples):
                return tokenizer(
                    examples[text_col],
                    truncation=False,
                    padding=False,
                    return_attention_mask=False,
                )

            def _group(examples):
                cat = sum(examples["input_ids"], [])
                total = (len(cat) // max_seq_length) * max_seq_length
                chunks = [cat[i:i + max_seq_length] for i in range(0, total, max_seq_length)]
                return {
                    "input_ids": chunks,
                    "attention_mask": [[1] * max_seq_length] * len(chunks),
                    "labels": chunks,
                }

            logger.info("Tokenizing...")
            _tok_cache = os.path.join(_cache_dir, f"tokenized_seq{max_seq_length}.arrow")
            _grp_cache = os.path.join(_cache_dir, f"grouped_seq{max_seq_length}.arrow")

            # --- Tokenized cache ---
            if os.path.isdir(_tok_cache):
                from datasets import load_from_disk
                logger.info(f"Loading tokenized cache from {_tok_cache}")
                tokenized = load_from_disk(_tok_cache)
            else:
                tokenized = ds.map(
                    _tokenize, batched=True,
                    remove_columns=ds.column_names,
                    num_proc=preprocessing_num_workers,
                    desc="Tokenizing",
                )
                tokenized.save_to_disk(_tok_cache)
                logger.info(f"Tokenized cache saved to {_tok_cache}")

            # --- Grouped cache ---
            logger.info("Grouping into chunks...")
            if os.path.isdir(_grp_cache):
                from datasets import load_from_disk
                logger.info(f"Loading grouped cache from {_grp_cache}")
                processed = load_from_disk(_grp_cache)
            else:
                processed = tokenized.map(
                    _group, batched=True,
                    num_proc=preprocessing_num_workers,
                    desc="Grouping",
                )
                processed.save_to_disk(_grp_cache)
                logger.info(f"Grouped cache saved to {_grp_cache}")
            processed.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            logger.info(f"Dataset ready: {len(processed)} chunks of {max_seq_length} tokens")
            return processed
        return TextDataset(text_file, tokenizer, max_seq_length)

    if dataset_name is None:
        raise ValueError("Either dataset_name or text_file must be provided.")

    from datasets import load_dataset

    logger.info(f"Loading dataset: {dataset_name} (config={dataset_config}, split={dataset_split})")

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=dataset_split,
        streaming=streaming,
    )

    if max_samples is not None and not streaming:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Determine text column
    text_column = None
    if not streaming:
        columns = dataset.column_names
    else:
        # For streaming, peek at first example
        first_example = next(iter(dataset))
        columns = list(first_example.keys())

    for candidate in ["text", "content", "document", "passage"]:
        if candidate in columns:
            text_column = candidate
            break

    if text_column is None:
        text_column = columns[0]
        logger.warning(f"Could not find standard text column. Using '{text_column}'.")

    logger.info(f"Using text column: '{text_column}'")

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )

    # Group texts into chunks of max_seq_length
    def group_texts(examples):
        # Concatenate all texts
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])

        # Drop the last incomplete chunk
        total_length = (total_length // max_seq_length) * max_seq_length

        # Split into chunks
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated.items()
        }

        # Labels = input_ids (for causal LM)
        result["labels"] = result["input_ids"].copy()
        result["attention_mask"] = [
            [1] * max_seq_length for _ in range(len(result["input_ids"]))
        ]

        return result

    if streaming:
        # For streaming datasets, process on-the-fly
        tokenized = dataset.map(tokenize_function, batched=True, remove_columns=columns)
        processed = tokenized.map(group_texts, batched=True)
    else:
        # Tokenize
        logger.info("Tokenizing dataset...")
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=columns,
            desc="Tokenizing",
        )

        # Group into chunks
        logger.info("Grouping texts into chunks...")
        processed = tokenized.map(
            group_texts,
            batched=True,
            num_proc=preprocessing_num_workers,
            desc="Grouping texts",
        )

    # Set format
    if not streaming:
        processed.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    logger.info(f"Dataset ready. {'Streaming mode' if streaming else f'{len(processed)} examples'}")
    return processed


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Build a DataLoader from a dataset.

    Args:
        dataset: The dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory (for GPU training)
        drop_last: Whether to drop the last incomplete batch

    Returns:
        A PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=_collate_fn,
    )


def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for the DataLoader."""
    # If batch items are dicts (from TextDataset or HuggingFace)
    if isinstance(batch[0], dict):
        result = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            elif isinstance(values[0], list):
                result[key] = torch.tensor(values, dtype=torch.long)
            else:
                result[key] = values
        return result

    # If batch items are tuples
    return {"input_ids": torch.stack([item[0] for item in batch])}


def create_dummy_dataset(
    tokenizer,
    num_samples: int = 100,
    max_seq_length: int = 512,
) -> Dataset:
    """
    Create a dummy dataset for testing.

    Args:
        tokenizer: The tokenizer
        num_samples: Number of samples
        max_seq_length: Sequence length

    Returns:
        A list-based dataset
    """
    logger.info(f"Creating dummy dataset with {num_samples} samples...")

    class DummyDataset(Dataset):
        def __init__(self, num_samples, vocab_size, seq_length):
            self.num_samples = num_samples
            self.vocab_size = vocab_size
            self.seq_length = seq_length

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones(self.seq_length, dtype=torch.long),
                "labels": input_ids.clone(),
            }

    return DummyDataset(num_samples, tokenizer.vocab_size, max_seq_length)
