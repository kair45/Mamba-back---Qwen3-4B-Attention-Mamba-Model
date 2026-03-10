"""
Download Qwen3-4B and Qwen3.5-4B models from HuggingFace to local disk.

Usage (Windows):
    python deployment/download_models.py --output D:/models

Then upload to AutoDL:
    scp -r D:/models/Qwen3-4B   root@<your-autodl-ip>:/root/autodl-tmp/models/
    scp -r D:/models/Qwen3.5-4B root@<your-autodl-ip>:/root/autodl-tmp/models/

Or use rsync (faster for large files):
    rsync -avz --progress D:/models/Qwen3-4B/   root@<ip>:/root/autodl-tmp/models/Qwen3-4B/
    rsync -avz --progress D:/models/Qwen3.5-4B/ root@<ip>:/root/autodl-tmp/models/Qwen3.5-4B/
"""

import argparse
import os
import sys


def download(repo_id: str, local_dir: str, hf_endpoint: str):
    """Download a model from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not found. Installing...")
        os.system(f"{sys.executable} -m pip install huggingface_hub -q")
        from huggingface_hub import snapshot_download

    os.environ["HF_ENDPOINT"] = hf_endpoint
    print(f"\n{'='*60}")
    print(f"  Downloading: {repo_id}")
    print(f"  Destination: {local_dir}")
    print(f"  Mirror:      {hf_endpoint}")
    print(f"{'='*60}\n")

    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,   # 断点续传，网络中断可重跑
    )
    print(f"\n✓ Done: {repo_id}  →  {local_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Download Qwen models for benchmark")
    parser.add_argument(
        "--output", type=str, default="D:/models",
        help="Local directory to save models (default: D:/models)",
    )
    parser.add_argument(
        "--mirror", type=str, default="https://hf-mirror.com",
        help="HuggingFace mirror URL (default: hf-mirror.com for China)",
    )
    parser.add_argument(
        "--models", type=str, default="qwen3,qwen35",
        help="Comma-separated list of models to download: qwen3, qwen35 (default: both)",
    )
    args = parser.parse_args()

    model_map = {
        "qwen3":  ("Qwen/Qwen3-4B",   os.path.join(args.output, "Qwen3-4B")),
        "qwen35": ("Qwen/Qwen3.5-4B", os.path.join(args.output, "Qwen3.5-4B")),
    }

    targets = [m.strip() for m in args.models.split(",")]
    for key in targets:
        if key not in model_map:
            print(f"Unknown model key '{key}', choose from: {list(model_map.keys())}")
            continue
        repo_id, local_dir = model_map[key]
        download(repo_id, local_dir, args.mirror)

    print("=" * 60)
    print("  All downloads complete.")
    print(f"  Models saved to: {args.output}")
    print()
    print("  Next step — upload to AutoDL server:")
    print(f"    scp -r {args.output}\\Qwen3-4B   root@<autodl-ip>:/root/autodl-tmp/models/")
    print(f"    scp -r {args.output}\\Qwen3.5-4B root@<autodl-ip>:/root/autodl-tmp/models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
