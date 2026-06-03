"""Storyboard rollout — generate shot list from script, then evaluate."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from skillopt.model import chat_target
from skillopt.envs.storyboard.evaluator import evaluate
from skillopt.envs.storyboard.hard_metrics import compute_hard_metrics
from skillopt.envs.storyboard.pipeline import run_pipeline, decompose_skill


def process_one(
    item: dict,
    out_root: str,
    skill_content: str,
    exec_timeout: int = 300,
    max_completion_tokens: int = 16384,
    frozen_skills: dict[str, str] | None = None,
    max_tokens_per_stage: dict[str, int] | None = None,
    pipeline_mode: str = "multi_agent",
    skill_base_dir: str = "",
) -> dict:
    """Process a single storyboard item via multi-agent pipeline."""
    item_id = str(item["id"])
    script_text = item["script_text"]
    ground_truth_csv = item["ground_truth_csv"]

    result = {
        "id": item_id,
        "task_type": "storyboard",
        "hard": 0,
        "soft": 0.0,
        "response": "",
        "fail_reason": "",
        "score": 0.0,
        "dimensions": {},
        "reasoning": "",
    }

    try:
        pred_dir = os.path.join(out_root, "predictions", item_id)
        os.makedirs(pred_dir, exist_ok=True)

        if pipeline_mode == "acp" and frozen_skills:
            from skillopt.model.acp_backend import prepare_acp_workspace, run_acp_pipeline

            optimizable_skills = decompose_skill(skill_content)
            all_skills = {**frozen_skills, **optimizable_skills}
            work_dir = os.path.join(pred_dir, "acp_workspace")
            prepare_acp_workspace(
                work_dir=work_dir,
                script_text=script_text,
                all_skills=all_skills,
                skill_base_dir=skill_base_dir,
            )
            response, _ = run_acp_pipeline(
                script_text=script_text,
                all_skills=all_skills,
                work_dir=work_dir,
                skill_base_dir=skill_base_dir,
                timeout=exec_timeout,
            )
            if not response.strip():
                result["fail_reason"] = "acp pipeline returned empty response"
                return result
        elif pipeline_mode == "multi_agent" and frozen_skills:
            pipeline_result = run_pipeline(
                script_text=script_text,
                composite_skill=skill_content,
                frozen_skills=frozen_skills,
                pred_dir=pred_dir,
                max_tokens_per_stage=max_tokens_per_stage,
                exec_timeout=exec_timeout,
            )
            if not pipeline_result.success:
                result["fail_reason"] = f"pipeline error: {pipeline_result.error}"
                return result
            response = pipeline_result.final_csv
        else:
            system = skill_content if skill_content.strip() else "你是专业的短剧分镜师，擅长将剧本拆分为结构化分镜头表。"
            user = f"## 剧本原文\n\n{script_text}\n\n## 任务\n\n请将上述剧本拆分为分镜头表。输出纯CSV格式。"
            response, _ = chat_target(
                system=system, user=user,
                max_completion_tokens=max_completion_tokens,
                retries=3, stage="rollout", timeout=exec_timeout,
            )
            with open(os.path.join(pred_dir, "system_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(system)
            with open(os.path.join(pred_dir, "user_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(user)

        result["response"] = response

        with open(os.path.join(pred_dir, "response.txt"), "w", encoding="utf-8") as f:
            f.write(response)
        with open(os.path.join(pred_dir, "system_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(skill_content)
        with open(os.path.join(pred_dir, "user_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(script_text)

        hard_result = compute_hard_metrics(script_text, response, ground_truth_csv)
        eval_result = evaluate(script_text, response, ground_truth_csv, timeout=exec_timeout)

        result["hard"] = hard_result["hard_score"]
        result["soft"] = eval_result["soft"]
        result["score"] = eval_result["score"]
        result["dimensions"] = eval_result["dimensions"]
        result["reasoning"] = eval_result["reasoning"]
        result["hard_metrics"] = {
            "shot_count_deviation": hard_result["shot_count_deviation"],
            "cut_alignment_rate": hard_result["cut_alignment_rate"],
            "shot_scale_edit_dist": hard_result["shot_scale_edit_dist"],
            "hard_score": hard_result["hard_score"],
        }

        if not hard_result.get("parse_ok", True):
            result["fail_reason"] = "CSV parse failure"
        elif eval_result["soft"] < 0.82:
            result["fail_reason"] = (
                f"Soft score below threshold (0.82): soft={eval_result['soft']:.2f} "
                f"hard={hard_result['hard_score']:.2f} "
                f"count_dev={hard_result['shot_count_deviation']:.2f} "
                f"cut_align={hard_result['cut_alignment_rate']:.2f} "
                f"scale_edit={hard_result['shot_scale_edit_dist']:.2f}"
            )

        combined_eval = {**eval_result, "hard_metrics": hard_result}
        with open(os.path.join(pred_dir, "eval_result.json"), "w", encoding="utf-8") as f:
            json.dump(combined_eval, f, ensure_ascii=False, indent=2)

    except Exception as e:  # noqa: BLE001
        result["fail_reason"] = f"error: {e}"

    return result


def run_batch(
    items: list[dict],
    out_root: str,
    skill_content: str,
    exec_timeout: int = 300,
    workers: int = 2,
    max_completion_tokens: int = 16384,
    task_timeout: int = 600,
    frozen_skills: dict[str, str] | None = None,
    max_tokens_per_stage: dict[str, int] | None = None,
    pipeline_mode: str = "multi_agent",
    skill_base_dir: str = "",
    **kwargs,
) -> list[dict]:
    """Run storyboard generation on all items. Resume-aware."""
    task_timeout = max(int(task_timeout), int(exec_timeout) + 60)
    results_path = os.path.join(out_root, "results.jsonl")
    os.makedirs(out_root, exist_ok=True)

    done_ids: set[str] = set()
    existing: list[dict] = []
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add(str(r["id"]))
                    existing.append(r)
                except Exception:
                    pass

    pending = [it for it in items if str(it["id"]) not in done_ids]
    if not pending:
        return existing

    total = len(existing) + len(pending)
    completed = len(existing)
    correct_count = sum(1 for r in existing if r.get("hard", 0))
    if existing:
        print(f"    [rollout] resuming: {completed}/{total} already done", flush=True)

    results = list(existing)
    started_at: dict[str, float] = {}

    def _run_one(item: dict) -> dict:
        started_at[str(item["id"])] = time.time()
        return process_one(
            item, out_root, skill_content, exec_timeout, max_completion_tokens,
            frozen_skills=frozen_skills,
            max_tokens_per_stage=max_tokens_per_stage,
            pipeline_mode=pipeline_mode,
            skill_base_dir=skill_base_dir,
        )

    def _timeout_result(item: dict) -> dict:
        return {
            "id": str(item["id"]),
            "task_type": "storyboard",
            "hard": 0,
            "soft": 0.0,
            "response": "",
            "fail_reason": f"task-timeout-{task_timeout}s",
            "score": 0.0,
            "dimensions": {},
            "reasoning": "",
        }

    with open(results_path, "a", encoding="utf-8") as outf:
        ex = ThreadPoolExecutor(max_workers=workers)
        try:
            futs = {ex.submit(_run_one, it): it for it in pending}
            pending_futs = set(futs)
            while pending_futs:
                done, _ = wait(pending_futs, timeout=5, return_when=FIRST_COMPLETED)
                now = time.time()
                timed_out = [
                    fut for fut in pending_futs - done
                    if str(futs[fut]["id"]) in started_at
                    and now - started_at[str(futs[fut]["id"])] >= task_timeout
                ]
                for fut in done:
                    pending_futs.remove(fut)
                    try:
                        res = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        res = _timeout_result(futs[fut])
                        res["fail_reason"] = f"error: {exc}"
                    results.append(res)
                    completed += 1
                    if res.get("hard", 0):
                        correct_count += 1
                    acc = correct_count / completed if completed else 0
                    print(
                        f"    [rollout] {completed}/{total} "
                        f"(score={res.get('score', 0):.1f}) id={res['id']} "
                        f"hard={res.get('hard', '?')}",
                        flush=True,
                    )
                    outf.write(json.dumps(res, ensure_ascii=False) + "\n")
                    outf.flush()
                for fut in timed_out:
                    pending_futs.remove(fut)
                    fut.cancel()
                    res = _timeout_result(futs[fut])
                    results.append(res)
                    completed += 1
                    print(
                        f"    [rollout] {completed}/{total} id={res['id']} TIMEOUT",
                        flush=True,
                    )
                    outf.write(json.dumps(res, ensure_ascii=False) + "\n")
                    outf.flush()
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    return results
