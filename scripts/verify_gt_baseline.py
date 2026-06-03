"""Verify baseline scores by feeding GT as prediction on the train set.

Usage:
    python scripts/verify_gt_baseline.py             # hard only (no API)
    python scripts/verify_gt_baseline.py --soft      # hard + LLM judge (needs API)
"""
from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skillopt.envs.storyboard.hard_metrics import compute_hard_metrics


def main(run_soft: bool = False):
    data_path = os.path.join("data", "storyboard_split", "train", "items.json")
    with open(data_path, encoding="utf-8") as f:
        items = json.load(f)

    print(f"Train items: {len(items)}")
    print("=" * 70)

    hard_scores = []
    soft_scores = []
    all_results = []

    evaluate_fn = None
    if run_soft:
        from skillopt.envs.storyboard.evaluator import evaluate as _eval
        evaluate_fn = _eval

    for item in items:
        tid = item["id"]
        script = item["script_text"]
        gt_csv = item["ground_truth_csv"]

        # Hard metrics: GT vs GT (should be 1.0)
        hard_result = compute_hard_metrics(script, gt_csv, gt_csv)
        hard_scores.append(hard_result["hard_score"])

        row = {
            "id": tid,
            "hard_score": hard_result["hard_score"],
            "shot_count_dev": hard_result["shot_count_deviation"],
            "cut_align": hard_result["cut_alignment_rate"],
            "scale_edit": hard_result["shot_scale_edit_dist"],
        }

        soft_str = ""
        if evaluate_fn:
            try:
                eval_result = evaluate_fn(script, gt_csv, gt_csv, timeout=120)
            except Exception as e:
                eval_result = {"soft": 0.0, "score": 0.0, "reasoning": str(e), "dimensions": {}}
            soft_scores.append(eval_result["soft"])
            row["soft"] = eval_result["soft"]
            row["overall"] = eval_result.get("score", 0)
            row["dimensions"] = eval_result.get("dimensions", {})
            row["reasoning"] = eval_result.get("reasoning", "")
            soft_str = f" | soft={eval_result['soft']:.4f} overall={eval_result.get('score', 0):.1f}"

        all_results.append(row)

        print(
            f"  [{tid}] hard={hard_result['hard_score']:.4f} "
            f"(cnt_dev={hard_result['shot_count_deviation']:.3f} "
            f"cut={hard_result['cut_alignment_rate']:.3f} "
            f"scale={hard_result['shot_scale_edit_dist']:.3f})"
            f"{soft_str}"
        )

    print("=" * 70)
    avg_hard = sum(hard_scores) / len(hard_scores) if hard_scores else 0
    print(f"AVG hard={avg_hard:.4f}  MIN={min(hard_scores):.4f}  MAX={max(hard_scores):.4f}")
    if soft_scores:
        avg_soft = sum(soft_scores) / len(soft_scores)
        print(f"AVG soft={avg_soft:.4f}  MIN={min(soft_scores):.4f}  MAX={max(soft_scores):.4f}")

    out_path = os.path.join("data", "gt_baseline_scores.json")
    summary = {"avg_hard": avg_hard, "items": all_results}
    if soft_scores:
        summary["avg_soft"] = sum(soft_scores) / len(soft_scores)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft", action="store_true", help="Also run LLM judge (needs API)")
    args = parser.parse_args()

    if args.soft:
        from skillopt.config import load_config
        from skillopt.model import set_optimizer_backend, set_optimizer_deployment
        cfg_path = os.path.join("configs", "storyboard", "default.yaml")
        cfg = load_config(cfg_path)
        set_optimizer_backend(cfg.get("optimizer_backend", "claude_chat"))
        set_optimizer_deployment(cfg.get("optimizer_model", "global.anthropic.claude-opus-4-6-v1[1m]"))

    main(run_soft=args.soft)
