"""Storyboard evaluator — LLM-as-judge scoring."""
from __future__ import annotations

import csv
import io
import json
import re

from skillopt.model import chat_optimizer


_JUDGE_SYSTEM = """\
你是一位资深短剧导演兼分镜评审。你的核心评判视角是：导演技术的专业性——节拍把控、镜头语法、运镜手法。
忽略镜头数量差异（多几个少几个不重要），聚焦以下5个维度。

## 评分维度（每维度0-100分）

1. **节拍/叙事节奏 (beat_pacing)** 0-100
   - 剧本中的叙事节拍是否被精准识别并转化为镜头切换？
   - 情绪升降是否通过镜头节奏体现（紧张处密切、舒缓处留白）？
   - 关键戏剧转折点是否有专门的镜头处理（而非淹没在流水账里）？

2. **景别推进公式 (shot_progression)** 0-100
   - 是否运用了专业的景别推进逻辑（如：全景建立→中景叙事→近景情绪→特写强调）？
   - 景别跳跃是否合理（避免无意义的远-近跳切，也避免同景别堆叠）？
   - 关系镜头（OTS/MFS双人/关系镜头）是否在对话场景中正确使用？

3. **运镜手法 (camera_movement)** 0-100
   - 运镜选择是否服务于叙事意图（推镜=压迫/聚焦，拉镜=揭示/疏离，摇镜=环境交代，手持=紧张不安）？
   - 是否有"导演意识"的运镜设计（而非机械填写"固定"）？
   - 运镜变化是否与情绪弧线匹配？

4. **画面描述的AI生成友好度 (ai_gen_quality)** 0-100
   - 画面内容字段是否是具体、可执行的视觉指令（动作+构图+光影）？
   - 是否避免了抽象/文学化描述（"氛围紧张"不行，"人物攥紧拳头，指节发白"才行）？
   - 每个镜头是否是一个独立可生成的画面（而非多个动作塞进一格）？

5. **对剧本的忠实度 (script_fidelity)** 0-100
   - 剧本中的对白、动作、情绪是否被准确转化为画面？
   - 是否有自由发挥过度（加入剧本没有的内容）或遗漏剧本明确描述的画面？
   - 插入画面/闪回等特殊标记是否被正确处理？

## 评分原则

- 这是导演技术的比拼。一个好的分镜不在于镜头多少，而在于每个镜头选择背后的专业判断。
- 与标准分镜相比，如果AI版本用了不同但同样合理的镜头语法，不应扣分。
- 如果AI版本明显缺乏"导演思维"（如全程固定机位、无景别变化、描述像流水账），严厉扣分。
- 所有维度和overall都是百分制（0-100），不是10分制。

## 输出格式

严格输出JSON，不要输出其他内容：
```json
{
  "beat_pacing": <0-100>,
  "shot_progression": <0-100>,
  "camera_movement": <0-100>,
  "ai_gen_quality": <0-100>,
  "script_fidelity": <0-100>,
  "overall": <0-100>,
  "reasoning": "<简要评价，2-3句，指出最大优势和最明显不足>"
}
```
"""


def _extract_json(text: str) -> dict | None:
    patterns = [
        re.compile(r"```json\s*\n(.*?)\n\s*```", re.DOTALL),
        re.compile(r"```\s*\n(.*?)\n\s*```", re.DOTALL),
        re.compile(r"\{[^{}]*\}", re.DOTALL),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                return json.loads(match.group(1) if pattern.groups else match.group(0))
            except (json.JSONDecodeError, IndexError):
                continue
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def _parse_csv(csv_text: str) -> list[dict] | None:
    """Try to parse CSV text into list of row dicts. Returns None if unparseable."""
    cleaned = csv_text.strip()
    fence_match = re.search(r"```(?:csv)?\s*\n(.*?)\n\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    if not cleaned:
        return None
    try:
        reader = csv.DictReader(io.StringIO(cleaned))
        rows = list(reader)
        return rows if rows else None
    except Exception:
        return None


def evaluate(
    script_text: str,
    generated_csv: str,
    ground_truth_csv: str,
    timeout: int = 120,
) -> dict:
    """Score generated storyboard against ground truth using LLM-as-judge.

    Returns dict with: score (0-10), soft (0-1), hard (0|1), reasoning, dimensions.
    """
    gen_rows = _parse_csv(generated_csv)
    if gen_rows is None:
        return {
            "score": 0.0,
            "soft": 0.0,
            "hard": 0,
            "reasoning": "Generated output is not valid CSV",
            "dimensions": {},
            "parse_ok": False,
        }

    user_prompt = (
        f"## 剧本原文\n\n{script_text}\n\n"
        f"## AI生成的分镜（待评）\n\n{generated_csv}\n\n"
        f"## 导演标准分镜（参考答案）\n\n{ground_truth_csv}\n\n"
        f"请按评分维度打分，严格输出JSON。"
    )

    try:
        response, _ = chat_optimizer(
            system=_JUDGE_SYSTEM,
            user=user_prompt,
            max_completion_tokens=2048,
            retries=3,
            stage="judge",
            timeout=timeout,
        )
    except Exception as e:
        return {
            "score": 5.0,
            "soft": 0.5,
            "hard": 0,
            "reasoning": f"Judge call failed: {e}",
            "dimensions": {},
            "parse_ok": True,
        }

    parsed = _extract_json(response)
    if parsed is None:
        return {
            "score": 5.0,
            "soft": 0.5,
            "hard": 0,
            "reasoning": f"Judge returned unparseable response: {response[:200]}",
            "dimensions": {},
            "parse_ok": True,
        }

    overall = float(parsed.get("overall", 50))
    dimensions = {
        k: parsed.get(k)
        for k in ("beat_pacing", "shot_progression", "camera_movement", "ai_gen_quality", "script_fidelity")
        if parsed.get(k) is not None
    }

    return {
        "score": overall,
        "soft": max(0.0, min(1.0, overall / 100.0)),
        "hard": 1 if overall >= 70 else 0,
        "reasoning": str(parsed.get("reasoning", "")),
        "dimensions": dimensions,
        "parse_ok": True,
    }
