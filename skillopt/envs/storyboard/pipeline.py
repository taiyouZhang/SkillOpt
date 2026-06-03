"""Multi-agent serial pipeline for storyboard generation.

Executes 5 sequential agents (Director -> DP -> Editor -> Continuity -> QA),
each with its own SKILL.md as system prompt and the previous agent's output
as input context.

Only Director and DP skills are optimizable; Editor/Continuity/QA are frozen.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from skillopt.model import chat_target


SECTION_START = "<!-- SKILLFILE:{name} START -->"
SECTION_END = "<!-- SKILLFILE:{name} END -->"
SECTION_PATTERN = re.compile(
    r"<!-- SKILLFILE:(?P<name>[^\s]+) START -->\n(?P<content>.*?)<!-- SKILLFILE:(?P=name) END -->",
    re.DOTALL,
)

STAGES = ["director", "dp", "editor", "continuity", "qa"]
OPTIMIZABLE = {"director", "dp"}

DEFAULT_MAX_TOKENS = {
    "director": 4096,
    "dp": 8192,
    "editor": 8192,
    "continuity": 8192,
    "qa": 8192,
}


# ── Skill composition / decomposition ────────────────────────────────────────


def compose_skill(director_content: str, dp_content: str) -> str:
    parts = []
    for name, content in [("director", director_content), ("dp", dp_content)]:
        start = SECTION_START.format(name=name)
        end = SECTION_END.format(name=name)
        parts.append(f"{start}\n{content}\n{end}")
    return "\n\n".join(parts)


def decompose_skill(composite: str) -> dict[str, str]:
    result = {}
    for m in SECTION_PATTERN.finditer(composite):
        result[m.group("name")] = m.group("content").strip()
    return result


def load_composite_skill(director_path: str, dp_path: str) -> str:
    with open(director_path, encoding="utf-8") as f:
        director = f.read()
    with open(dp_path, encoding="utf-8") as f:
        dp = f.read()
    return compose_skill(director, dp)


# ── Per-stage user prompt templates ──────────────────────────────────────────

_DIRECTOR_USER = """\
## 剧本原文

{script_text}

## 任务

请阅读上述剧本，按照你的SKILL规范完成导演脚注（director_notes）：
1. 分析每场戏的Scene Intent
2. 划分Narrative Beats
3. 做取舍决策（密集拆镜/轻拆/省略）
4. 设定运镜基调和角度策略

输出完整的导演脚注markdown。
"""

_DP_USER = """\
## 导演脚注

{director_notes}

## 剧本原文

{script_text}

## 任务

请根据导演脚注和剧本原文，按照你的SKILL规范设计具体镜头。
输出纯CSV格式（不要markdown代码块），列头为：
镜头号,分镜组,场景,中文台词,英文台词,Prompt,时长,景别,视角,运镜,角色,道具,音效

要求：
- 每行一个镜头，镜头号从0开始递增
- Prompt字段要具体、视觉化、可用于AI视频生成
- 严格遵循景别推进公式、OTS精确格式、比喻禁令
"""

_EDITOR_USER = """\
## 导演脚注

{director_notes}

## DP初稿（dp_draft.csv）

{dp_draft}

## 任务

请根据你的SKILL规范优化分镜节奏：
- 删除冗余镜头
- 补充缺失反应/插入镜头
- 调整分镜组划分
- 校验运镜/角度多样性

输出优化后的纯CSV格式（保持相同列头）。
"""

_CONTINUITY_USER = """\
## 分镜表（editor_cut.csv）

{editor_cut}

## 任务

请根据你的SKILL规范做格式终检：
- 场景格式校验
- 编号连续性
- 字段完整性
- 台词/Prompt分离

输出修正后的纯CSV格式（保持相同列头）。
"""

_QA_USER = """\
## 分镜表（continuity_checked.csv）

{continuity_output}

## 任务

请根据你的SKILL规范做内容质量终检：
- 清除比喻污染
- 隔离非视觉内容
- 校验OTS精度
- 确认景别推进公式标注

输出最终修正后的纯CSV格式（保持相同列头）。
"""

USER_TEMPLATES = {
    "director": _DIRECTOR_USER,
    "dp": _DP_USER,
    "editor": _EDITOR_USER,
    "continuity": _CONTINUITY_USER,
    "qa": _QA_USER,
}


# ── Pipeline execution ───────────────────────────────────────────────────────


@dataclass
class StageResult:
    name: str
    system_prompt: str
    user_prompt: str
    output: str
    success: bool = True
    error: str = ""


@dataclass
class PipelineResult:
    stages: list[StageResult] = field(default_factory=list)
    final_csv: str = ""
    success: bool = True
    error: str = ""


def run_pipeline(
    script_text: str,
    composite_skill: str,
    frozen_skills: dict[str, str],
    pred_dir: str,
    max_tokens_per_stage: dict[str, int] | None = None,
    exec_timeout: int = 300,
) -> PipelineResult:
    """Execute the 5-agent serial pipeline."""
    if max_tokens_per_stage is None:
        max_tokens_per_stage = DEFAULT_MAX_TOKENS

    optimizable_skills = decompose_skill(composite_skill)
    all_skills = {**frozen_skills, **optimizable_skills}

    result = PipelineResult()
    intermediates: dict[str, str] = {"script_text": script_text}

    for stage_name in STAGES:
        system = all_skills.get(stage_name, "")
        if not system.strip():
            system = f"你是专业的短剧{stage_name}，请完成你的工作。"

        user = _build_user_prompt(stage_name, intermediates)
        max_tokens = max_tokens_per_stage.get(stage_name, 8192)

        try:
            response, _ = chat_target(
                system=system,
                user=user,
                max_completion_tokens=max_tokens,
                retries=2,
                stage=f"rollout_{stage_name}",
                timeout=exec_timeout,
            )
        except Exception as e:
            stage_result = StageResult(
                name=stage_name, system_prompt=system,
                user_prompt=user, output="", success=False, error=str(e),
            )
            result.stages.append(stage_result)
            result.success = False
            result.error = f"{stage_name} failed: {e}"
            break

        stage_result = StageResult(
            name=stage_name, system_prompt=system,
            user_prompt=user, output=response,
        )
        result.stages.append(stage_result)

        intermediates[stage_name] = response

        _save_stage_output(pred_dir, stage_name, stage_result)

    if result.success and result.stages:
        result.final_csv = result.stages[-1].output

    _save_conversation_json(pred_dir, result.stages)

    return result


def _build_user_prompt(stage_name: str, intermediates: dict[str, str]) -> str:
    template = USER_TEMPLATES[stage_name]
    if stage_name == "director":
        return template.format(script_text=intermediates["script_text"])
    elif stage_name == "dp":
        return template.format(
            director_notes=intermediates.get("director", ""),
            script_text=intermediates["script_text"],
        )
    elif stage_name == "editor":
        return template.format(
            director_notes=intermediates.get("director", ""),
            dp_draft=intermediates.get("dp", ""),
        )
    elif stage_name == "continuity":
        return template.format(editor_cut=intermediates.get("editor", ""))
    elif stage_name == "qa":
        return template.format(continuity_output=intermediates.get("continuity", ""))
    return ""


def _save_stage_output(pred_dir: str, stage_name: str, stage: StageResult) -> None:
    os.makedirs(pred_dir, exist_ok=True)
    with open(os.path.join(pred_dir, f"{stage_name}_system.txt"), "w", encoding="utf-8") as f:
        f.write(stage.system_prompt)
    with open(os.path.join(pred_dir, f"{stage_name}_input.txt"), "w", encoding="utf-8") as f:
        f.write(stage.user_prompt)
    with open(os.path.join(pred_dir, f"{stage_name}_output.txt"), "w", encoding="utf-8") as f:
        f.write(stage.output)


def _save_conversation_json(pred_dir: str, stages: list[StageResult]) -> None:
    """Save pipeline trace as conversation.json for the reflect stage."""
    os.makedirs(pred_dir, exist_ok=True)
    conversation = []
    for stage in stages:
        conversation.append({"role": "system", "content": f"[{stage.name.upper()} Agent]\n{stage.system_prompt[:2000]}"})
        conversation.append({"role": "user", "content": stage.user_prompt[:3000]})
        conversation.append({"role": "assistant", "content": stage.output[:4000]})
    with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)
