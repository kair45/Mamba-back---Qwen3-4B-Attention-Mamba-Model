#!/usr/bin/env python3
"""
Downstream Task Evaluation using lm-eval-harness.

Evaluates the hybrid model and (optionally) the teacher model on standard
NLP benchmarks, then generates a comparison table for CV/paper use.

Benchmarks:
- HellaSwag     (commonsense reasoning)
- MMLU          (multitask knowledge, 5-shot)
- ARC-Challenge (science QA)
- WinoGrande    (coreference resolution)
- GSM8K         (math reasoning)
- TruthfulQA    (factual accuracy)

Requirements:
    pip install lm-eval>=0.4.0

Usage:
    # Evaluate hybrid model on all benchmarks
    python scripts/run_lm_eval.py \
        --hybrid_model ./checkpoints/qwen-mamba-hybrid/final \
        --output_dir ./eval_results

    # Compare with teacher
    python scripts/run_lm_eval.py \
        --hybrid_model ./checkpoints/qwen-mamba-hybrid/final \
        --teacher_model Qwen/Qwen3-4B \
        --output_dir ./eval_results

    # Quick eval (fewer benchmarks)
    python scripts/run_lm_eval.py \
        --hybrid_model ./checkpoints/qwen-mamba-hybrid/final \
        --teacher_model Qwen/Qwen3-4B \
        --tasks hellaswag,arc_challenge \
        --output_dir ./eval_results

    # Custom tasks and batch size
    python scripts/run_lm_eval.py \
        --hybrid_model ./checkpoints/qwen-mamba-hybrid/final \
        --tasks hellaswag,mmlu,arc_challenge,winogrande,gsm8k \
        --batch_size 8 \
        --output_dir ./eval_results
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.helpers import setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default benchmark configuration
# ---------------------------------------------------------------------------

DEFAULT_TASKS = [
    "hellaswag",
    "mmlu",
    "arc_challenge",
    "winogrande",
    "gsm8k",
    "truthfulqa_mc2",
]

TASK_DISPLAY_NAMES = {
    "hellaswag": "HellaSwag",
    "mmlu": "MMLU (5-shot)",
    "arc_challenge": "ARC-Challenge",
    "arc_easy": "ARC-Easy",
    "winogrande": "WinoGrande",
    "gsm8k": "GSM8K",
    "truthfulqa_mc2": "TruthfulQA MC2",
    "piqa": "PIQA",
    "boolq": "BoolQ",
    "openbookqa": "OpenBookQA",
}

TASK_NUM_FEWSHOT = {
    "hellaswag": 10,
    "mmlu": 5,
    "arc_challenge": 25,
    "arc_easy": 25,
    "winogrande": 5,
    "gsm8k": 5,
    "truthfulqa_mc2": 0,
    "piqa": 0,
    "boolq": 0,
    "openbookqa": 0,
}

# The metric to extract from each task's results
TASK_METRIC_KEY = {
    "hellaswag": "acc_norm,none",
    "mmlu": "acc,none",
    "arc_challenge": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "winogrande": "acc,none",
    "gsm8k": "exact_match,strict-match",
    "truthfulqa_mc2": "acc,none",
    "piqa": "acc_norm,none",
    "boolq": "acc,none",
    "openbookqa": "acc_norm,none",
}


# ---------------------------------------------------------------------------
# lm-eval-harness runner
# ---------------------------------------------------------------------------

def check_lm_eval_installed() -> bool:
    """Check if lm-eval-harness is installed."""
    try:
        import lm_eval
        return True
    except ImportError:
        return False


def run_lm_eval_python(
    model_path: str,
    tasks: List[str],
    batch_size: int = 4,
    dtype: str = "bfloat16",
    output_path: Optional[str] = None,
    trust_remote_code: bool = True,
    num_fewshot_override: Optional[int] = None,
) -> Dict:
    """
    Run lm-eval-harness using the Python API.

    Returns:
        Dictionary of results from lm-eval.
    """
    import lm_eval

    logger.info(f"Running lm-eval on: {model_path}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Batch size: {batch_size}")

    # Prepare task-specific num_fewshot
    task_str = ",".join(tasks)

    # Determine num_fewshot
    if num_fewshot_override is not None:
        num_fewshot = num_fewshot_override
    else:
        # Use default for each task (take the max if mixed)
        num_fewshot = None  # Let lm-eval use task defaults

    model_args = f"pretrained={model_path},dtype={dtype},trust_remote_code={str(trust_remote_code).lower()}"

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        log_samples=False,
    )

    # Save raw results
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Filter out non-serializable items
        serializable = {}
        if "results" in results:
            serializable["results"] = results["results"]
        if "config" in results:
            serializable["config"] = {
                k: str(v) for k, v in results.get("config", {}).items()
            }
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        logger.info(f"Raw results saved: {output_path}")

    return results


def run_lm_eval_cli(
    model_path: str,
    tasks: List[str],
    batch_size: int = 4,
    dtype: str = "bfloat16",
    output_dir: Optional[str] = None,
    trust_remote_code: bool = True,
) -> Optional[str]:
    """
    Run lm-eval-harness via CLI (fallback if Python API has issues).

    Returns:
        Path to the output directory with results.
    """
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype={dtype},trust_remote_code={str(trust_remote_code).lower()}",
        "--tasks", ",".join(tasks),
        "--batch_size", str(batch_size),
    ]

    if output_dir:
        cmd.extend(["--output_path", output_dir])

    cmd_str = " ".join(cmd)
    logger.info(f"Running: {cmd_str}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        print(result.stdout)
        if result.returncode != 0:
            logger.error(f"lm-eval failed:\n{result.stderr}")
            return None
        return output_dir
    except subprocess.TimeoutExpired:
        logger.error("lm-eval timed out (2 hour limit)")
        return None
    except Exception as e:
        logger.error(f"Failed to run lm-eval: {e}")
        return None


# ---------------------------------------------------------------------------
# Result extraction and formatting
# ---------------------------------------------------------------------------

def extract_scores(results: Dict, tasks: List[str]) -> Dict[str, float]:
    """
    Extract accuracy scores from lm-eval results.

    Returns:
        Dict mapping task name to accuracy (0-100 scale).
    """
    scores = {}

    raw_results = results.get("results", {})

    for task in tasks:
        task_data = raw_results.get(task, {})
        if not task_data:
            # Try with alias
            for key in raw_results:
                if task in key:
                    task_data = raw_results[key]
                    break

        if not task_data:
            logger.warning(f"No results found for task: {task}")
            scores[task] = -1
            continue

        # Try multiple metric keys
        metric_keys = [
            TASK_METRIC_KEY.get(task, "acc,none"),
            "acc_norm,none",
            "acc,none",
            "exact_match,strict-match",
            "exact_match,flexible-extract",
        ]

        found = False
        for mk in metric_keys:
            if mk in task_data:
                scores[task] = round(task_data[mk] * 100, 2)
                found = True
                break

        if not found:
            # Try any metric that looks like accuracy
            for k, v in task_data.items():
                if isinstance(v, (int, float)) and ("acc" in k or "match" in k):
                    scores[task] = round(v * 100, 2)
                    found = True
                    break

        if not found:
            logger.warning(f"Could not extract score for {task}. Keys: {list(task_data.keys())}")
            scores[task] = -1

    return scores


def generate_comparison_table(
    hybrid_scores: Dict[str, float],
    teacher_scores: Optional[Dict[str, float]],
    tasks: List[str],
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a markdown comparison table.

    Returns:
        Markdown-formatted table string.
    """
    lines = []
    lines.append("")
    lines.append("## Downstream Task Comparison")
    lines.append("")

    if teacher_scores:
        lines.append("| Benchmark | Teacher (Qwen3-4B) | Hybrid (Ours) | Retention |")
        lines.append("|-----------|:------------------:|:-------------:|:---------:|")
    else:
        lines.append("| Benchmark | Hybrid (Ours) |")
        lines.append("|-----------|:-------------:|")

    avg_retention = []

    for task in tasks:
        display_name = TASK_DISPLAY_NAMES.get(task, task)
        h_score = hybrid_scores.get(task, -1)

        if teacher_scores:
            t_score = teacher_scores.get(task, -1)
            if h_score >= 0 and t_score > 0:
                retention = h_score / t_score * 100
                avg_retention.append(retention)
                lines.append(
                    f"| {display_name} | {t_score:.1f} | {h_score:.1f} | {retention:.1f}% |"
                )
            elif h_score >= 0:
                lines.append(f"| {display_name} | N/A | {h_score:.1f} | N/A |")
            else:
                lines.append(f"| {display_name} | {t_score:.1f} | Error | N/A |")
        else:
            if h_score >= 0:
                lines.append(f"| {display_name} | {h_score:.1f} |")
            else:
                lines.append(f"| {display_name} | Error |")

    # Average row
    if teacher_scores and avg_retention:
        avg_ret = sum(avg_retention) / len(avg_retention)
        h_avg = sum(v for v in hybrid_scores.values() if v >= 0) / max(sum(1 for v in hybrid_scores.values() if v >= 0), 1)
        t_avg = sum(v for v in teacher_scores.values() if v >= 0) / max(sum(1 for v in teacher_scores.values() if v >= 0), 1)
        lines.append(f"| **Average** | **{t_avg:.1f}** | **{h_avg:.1f}** | **{avg_ret:.1f}%** |")

    lines.append("")

    table_str = "\n".join(lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(table_str)
        logger.info(f"Comparison table saved: {output_path}")

    return table_str


def generate_cv_bullet_points(
    hybrid_scores: Dict[str, float],
    teacher_scores: Optional[Dict[str, float]],
) -> str:
    """
    Generate ready-to-use CV bullet points from evaluation results.
    """
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("  READY-TO-USE CV BULLET POINTS")
    lines.append("=" * 60)
    lines.append("")

    if teacher_scores:
        # Calculate average retention
        retentions = []
        for task in hybrid_scores:
            h = hybrid_scores.get(task, -1)
            t = teacher_scores.get(task, -1)
            if h >= 0 and t > 0:
                retentions.append(h / t * 100)

        if retentions:
            avg_ret = sum(retentions) / len(retentions)

            lines.append(f"[Chinese]")
            lines.append(f"  - 在 {len(retentions)} 个标准 NLP 基准测试上平均保留教师模型 {avg_ret:.0f}% 的精度")
            lines.append("")

            lines.append(f"[English]")
            lines.append(f"  - Retained {avg_ret:.0f}% of teacher model accuracy across {len(retentions)} standard NLP benchmarks")
            lines.append("")

            # Best and worst retention
            task_ret = {t: hybrid_scores[t] / teacher_scores.get(t, 1) * 100
                        for t in hybrid_scores if hybrid_scores[t] >= 0 and teacher_scores.get(t, 0) > 0}
            if task_ret:
                best_task = max(task_ret, key=task_ret.get)
                worst_task = min(task_ret, key=task_ret.get)
                best_name = TASK_DISPLAY_NAMES.get(best_task, best_task)
                worst_name = TASK_DISPLAY_NAMES.get(worst_task, worst_task)

                lines.append(f"  Best retention:  {best_name} = {task_ret[best_task]:.1f}% "
                             f"({hybrid_scores[best_task]:.1f} vs {teacher_scores[best_task]:.1f})")
                lines.append(f"  Worst retention: {worst_name} = {task_ret[worst_task]:.1f}% "
                             f"({hybrid_scores[worst_task]:.1f} vs {teacher_scores[worst_task]:.1f})")
    else:
        lines.append("  (No teacher comparison available. Add --teacher_model for comparative metrics.)")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run lm-eval-harness benchmarks on Hybrid vs Teacher"
    )
    parser.add_argument("--hybrid_model", type=str, required=True, help="Path to hybrid model")
    parser.add_argument("--teacher_model", type=str, default=None, help="Path/name of teacher model")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task names (default: hellaswag,mmlu,arc_challenge,winogrande,gsm8k,truthfulqa_mc2)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--use_cli", action="store_true", help="Use CLI instead of Python API")
    parser.add_argument("--num_fewshot", type=int, default=None, help="Override num_fewshot for all tasks")

    return parser.parse_args()


def main():
    args = parse_args()

    setup_logging(log_level="INFO")

    # Check lm-eval is installed
    if not check_lm_eval_installed():
        logger.error(
            "lm-eval-harness is not installed!\n"
            "Install with: pip install lm-eval>=0.4.0\n"
            "Or: pip install lm_eval"
        )
        sys.exit(1)

    # Parse tasks
    tasks = args.tasks.split(",") if args.tasks else DEFAULT_TASKS
    tasks = [t.strip() for t in tasks]

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  LM-Eval Benchmark Runner")
    logger.info("=" * 60)
    logger.info(f"  Tasks:         {tasks}")
    logger.info(f"  Hybrid model:  {args.hybrid_model}")
    logger.info(f"  Teacher model: {args.teacher_model or 'None'}")
    logger.info(f"  Batch size:    {args.batch_size}")
    logger.info(f"  Output dir:    {args.output_dir}")
    logger.info("=" * 60)

    # --- Evaluate Hybrid Model ---
    logger.info("\n--- Evaluating Hybrid Model ---")

    if args.use_cli:
        hybrid_output_dir = os.path.join(args.output_dir, "hybrid_raw")
        run_lm_eval_cli(
            model_path=args.hybrid_model,
            tasks=tasks,
            batch_size=args.batch_size,
            dtype=args.dtype,
            output_dir=hybrid_output_dir,
        )
        # Load results from output
        hybrid_scores = {}
        logger.warning("CLI mode: manual result extraction needed from output directory.")
    else:
        hybrid_raw = run_lm_eval_python(
            model_path=args.hybrid_model,
            tasks=tasks,
            batch_size=args.batch_size,
            dtype=args.dtype,
            output_path=os.path.join(args.output_dir, "hybrid_lm_eval_raw.json"),
            num_fewshot_override=args.num_fewshot,
        )
        hybrid_scores = extract_scores(hybrid_raw, tasks)

    logger.info(f"\nHybrid Scores: {hybrid_scores}")

    # --- Evaluate Teacher Model ---
    teacher_scores = None
    if args.teacher_model:
        logger.info("\n--- Evaluating Teacher Model ---")

        if args.use_cli:
            teacher_output_dir = os.path.join(args.output_dir, "teacher_raw")
            run_lm_eval_cli(
                model_path=args.teacher_model,
                tasks=tasks,
                batch_size=args.batch_size,
                dtype=args.dtype,
                output_dir=teacher_output_dir,
            )
            teacher_scores = {}
        else:
            teacher_raw = run_lm_eval_python(
                model_path=args.teacher_model,
                tasks=tasks,
                batch_size=args.batch_size,
                dtype=args.dtype,
                output_path=os.path.join(args.output_dir, "teacher_lm_eval_raw.json"),
                num_fewshot_override=args.num_fewshot,
            )
            teacher_scores = extract_scores(teacher_raw, tasks)

        logger.info(f"\nTeacher Scores: {teacher_scores}")

    # --- Generate Comparison Table ---
    logger.info("\n--- Generating Comparison ---")

    table = generate_comparison_table(
        hybrid_scores=hybrid_scores,
        teacher_scores=teacher_scores,
        tasks=tasks,
        output_path=os.path.join(args.output_dir, "comparison_table.md"),
    )
    print(table)

    # Generate CV bullet points
    cv_text = generate_cv_bullet_points(hybrid_scores, teacher_scores)
    print(cv_text)

    cv_path = os.path.join(args.output_dir, "cv_bullet_points.txt")
    with open(cv_path, "w", encoding="utf-8") as f:
        f.write(cv_text)
    logger.info(f"CV bullet points saved: {cv_path}")

    # Save all scores to JSON
    all_scores = {"hybrid": hybrid_scores}
    if teacher_scores:
        all_scores["teacher"] = teacher_scores
    scores_path = os.path.join(args.output_dir, "all_scores.json")
    with open(scores_path, "w") as f:
        json.dump(all_scores, f, indent=2)
    logger.info(f"All scores saved: {scores_path}")

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
