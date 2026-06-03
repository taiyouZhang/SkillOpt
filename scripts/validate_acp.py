#!/usr/bin/env python3
"""Validate ACP pipeline: run 1 item from each split (train/val/test).

Usage:
    python scripts/validate_acp.py

This script:
  1. Loads 1 item from train, val, test splits
  2. Composes the skill from the skill_base_dir
  3. For each item, prepares an ACP workspace and runs `claude -p`
  4. Evaluates the result (hard metrics + soft judge)
  5. Reports pass/fail for each split
"""
from __future__ import annotations

import json
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from skillopt.envs.storyboard.dataloader import StoryboardDataLoader
from skillopt.envs.storyboard.adapter import StoryboardAdapter
from skillopt.envs.storyboard.pipeline import compose_skill, decompose_skill
from skillopt.envs.storyboard.hard_metrics import compute_hard_metrics
from skillopt.model.acp_backend import prepare_acp_workspace, run_acp_pipeline


# ── Config ─────────────────────────────────────────────────────────────────

SPLIT_DIR = os.path.join(_PROJECT_ROOT, "data", "storyboard_split")
SKILL_BASE_DIR = r"C:\Users\44820\Documents\project\20260601_剧本分镜拆分\Agent-Skills\skills\script-to-shots"
OUT_ROOT = os.path.join(_PROJECT_ROOT, "runs", "validate_acp")
ACP_TIMEOUT = 600  # 10 minutes per item


def load_frozen_skills(skill_base_dir: str) -> dict[str, str]:
    frozen = {}
    for name in ("editor", "continuity", "qa"):
        path = os.path.join(skill_base_dir, name, "SKILL.md")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                frozen[name] = f.read()
        else:
            frozen[name] = ""
    return frozen


def compose_initial_skill(skill_base_dir: str) -> str:
    director_path = os.path.join(skill_base_dir, "director", "SKILL.md")
    dp_path = os.path.join(skill_base_dir, "dp", "SKILL.md")
    director = ""
    dp = ""
    if os.path.exists(director_path):
        with open(director_path, encoding="utf-8") as f:
            director = f.read()
    if os.path.exists(dp_path):
        with open(dp_path, encoding="utf-8") as f:
            dp = f.read()
    return compose_skill(director, dp)


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    # Load skills
    print("=" * 60)
    print("  ACP Pipeline Validation — 1 item per split")
    print("=" * 60)

    composite_skill = compose_initial_skill(SKILL_BASE_DIR)
    optimizable_skills = decompose_skill(composite_skill)
    frozen_skills = load_frozen_skills(SKILL_BASE_DIR)
    all_skills = {**frozen_skills, **optimizable_skills}

    print(f"  Skills loaded: {list(all_skills.keys())}")
    for name, content in all_skills.items():
        print(f"    {name}: {len(content)} chars")

    # Load 1 item from each split
    splits = {}
    for split_name in ("train", "val", "test"):
        items_path = os.path.join(SPLIT_DIR, split_name, "items.json")
        with open(items_path, encoding="utf-8") as f:
            items = json.load(f)
        splits[split_name] = items[0]  # Take first item
        print(f"\n  [{split_name}] id={items[0]['id']}, script_text={len(items[0]['script_text'])} chars")

    # Run ACP pipeline on each
    results = {}
    for split_name, item in splits.items():
        item_id = str(item["id"])
        script_text = item["script_text"]
        ground_truth_csv = item["ground_truth_csv"]

        print(f"\n{'─' * 60}")
        print(f"  Running ACP pipeline: split={split_name} id={item_id}")
        print(f"{'─' * 60}")

        work_dir = os.path.join(OUT_ROOT, split_name, item_id, "acp_workspace")
        prepare_acp_workspace(
            work_dir=work_dir,
            script_text=script_text,
            all_skills=all_skills,
        )

        t0 = time.time()
        final_csv, raw_meta = run_acp_pipeline(
            script_text=script_text,
            all_skills=all_skills,
            work_dir=work_dir,
            timeout=ACP_TIMEOUT,
        )
        elapsed = time.time() - t0

        # Evaluate
        has_output = bool(final_csv.strip())
        hard_result = None
        if has_output:
            hard_result = compute_hard_metrics(script_text, final_csv, ground_truth_csv)

        result = {
            "split": split_name,
            "id": item_id,
            "has_output": has_output,
            "output_chars": len(final_csv),
            "elapsed_s": round(elapsed, 1),
            "hard_metrics": hard_result,
            "raw_meta": raw_meta[:500],
        }
        results[split_name] = result

        # Save outputs
        out_dir = os.path.join(OUT_ROOT, split_name, item_id)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "final_csv.txt"), "w", encoding="utf-8") as f:
            f.write(final_csv)
        with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # Print summary
        if has_output and hard_result:
            print(f"  OUTPUT: {len(final_csv)} chars, {elapsed:.1f}s")
            print(f"  HARD METRICS:")
            print(f"    shot_count_deviation: {hard_result['shot_count_deviation']:.3f}")
            print(f"    cut_alignment_rate:   {hard_result['cut_alignment_rate']:.3f}")
            print(f"    shot_scale_edit_dist: {hard_result['shot_scale_edit_dist']:.3f}")
            print(f"    hard_score:           {hard_result['hard_score']:.3f}")
        elif has_output:
            print(f"  OUTPUT: {len(final_csv)} chars (hard metrics failed)")
        else:
            print(f"  FAILED: no output produced ({elapsed:.1f}s)")
            print(f"  Raw: {raw_meta[:200]}")

    # Final summary
    print(f"\n{'=' * 60}")
    print("  VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    all_pass = True
    for split_name, result in results.items():
        status = "PASS" if result["has_output"] else "FAIL"
        if not result["has_output"]:
            all_pass = False
        hard_str = ""
        if result["hard_metrics"]:
            hard_str = f" hard={result['hard_metrics']['hard_score']:.3f}"
        print(f"  [{status}] {split_name:5s} id={result['id']} "
              f"chars={result['output_chars']} "
              f"time={result['elapsed_s']}s{hard_str}")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"  Output: {OUT_ROOT}")

    # Save combined results
    with open(os.path.join(OUT_ROOT, "validation_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
