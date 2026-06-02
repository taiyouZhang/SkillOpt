"""Storyboard rollout — generate shot list from script, then evaluate."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from skillopt.model import chat_target
from skillopt.envs.storyboard.evaluator import evaluate


_USER_TEMPLATE = """\
## 剧本原文

{script_text}

## 任务

请将上述剧本拆分为分镜头表。输出纯CSV格式（不要markdown代码块），列头为：
分镜号,场景,画面内容,景别,拍摄角度,运镜,角色,台词

要求：
- 每行一个镜头，分镜号从0开始递增
- "画面内容"字段要具体、视觉化、可用于AI视频生成，描述画面动作和构图
- 景别使用：WS/FS/MFS/MS/MCU/CU/POV/OTS等
- 拍摄角度使用：平/微俯/微仰/仰拍/俯拍/倾斜/侧等
- 运镜使用：固定/推/拉/摇/移/手持/环绕/跟等
- 台词字段保留原文对白，无对白则留空
"""


def process_one(
    item: dict,
    out_root: str,
    skill_content: str,
    exec_timeout: int = 300,
    max_completion_tokens: int = 16384,
) -> dict:
    """Process a single storyboard item: generate CSV + evaluate."""
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

        system = skill_content if skill_content.strip() else "你是专业的短剧分镜师，擅长将剧本拆分为结构化分镜头表。"
        user = _USER_TEMPLATE.format(script_text=script_text)

        response, _ = chat_target(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=3,
            stage="rollout",
            timeout=exec_timeout,
        )

        result["response"] = response

        with open(os.path.join(pred_dir, "system_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(system)
        with open(os.path.join(pred_dir, "user_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(user)
        with open(os.path.join(pred_dir, "response.txt"), "w", encoding="utf-8") as f:
            f.write(response)

        eval_result = evaluate(script_text, response, ground_truth_csv, timeout=exec_timeout)
        result["hard"] = eval_result["hard"]
        result["soft"] = eval_result["soft"]
        result["score"] = eval_result["score"]
        result["dimensions"] = eval_result["dimensions"]
        result["reasoning"] = eval_result["reasoning"]

        if not eval_result.get("parse_ok", True):
            result["fail_reason"] = "CSV parse failure"
        elif eval_result["hard"] == 0:
            result["fail_reason"] = f"Score {eval_result['score']:.1f}/10 below threshold (7)"

        with open(os.path.join(pred_dir, "eval_result.json"), "w", encoding="utf-8") as f:
            json.dump(eval_result, f, ensure_ascii=False, indent=2)

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
        return process_one(item, out_root, skill_content, exec_timeout, max_completion_tokens)

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
