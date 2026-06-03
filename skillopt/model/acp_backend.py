"""ACP backend for running full pipeline via a Claude Code agent subprocess.

Prepares a workspace with script + skill files + CLAUDE.md instructions,
spawns Claude Code in non-interactive mode (`claude -p`), and lets it
autonomously execute the entire 5-stage storyboard pipeline.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import traceback

from skillopt.model.backend_config import get_acp_exec_config


# ── CLAUDE.md template for pipeline workspace ─────────────────────────────────

_CLAUDE_MD_TEMPLATE = """\
# Storyboard Pipeline Execution

You are executing a 5-stage storyboard generation pipeline. Follow these steps exactly in order.

## Workspace Structure

```
.agents/skills/
├── SKILL.md              # Top-level orchestration skill (read first for overview)
├── CONTRACT.md           # Input/output contract spec
├── director/SKILL.md     # Stage 1 skill
├── dp/SKILL.md           # Stage 2 skill
├── editor/SKILL.md       # Stage 3 skill
├── continuity/SKILL.md   # Stage 4 skill
├── qa/SKILL.md           # Stage 5 skill
├── references/           # Shared references (director styles, templates, gotchas, etc.)
└── scripts/              # Helper scripts (if any)
```

## Pipeline Stages

1. **Director** — Read `.agents/skills/director/SKILL.md` as your working guide. Read `script.txt` as input. Reference files under `.agents/skills/references/` as needed (director styles, gotchas, etc.). Produce director notes (scene intent, narrative beats, shot density decisions, camera strategy). Write output to `output/director_notes.md`.

2. **DP (Director of Photography)** — Read `.agents/skills/dp/SKILL.md` as your working guide. Using your director notes + the original script, reference `.agents/skills/references/` for prompt templates, schema, and examples. Design concrete shots as pure CSV. Write output to `output/dp_draft.csv`. CSV columns: 镜头号,分镜组,场景,中文台词,英文台词,Prompt,时长,景别,视角,运镜,角色,道具,音效

3. **Editor** — Read `.agents/skills/editor/SKILL.md` as your working guide. Optimize the DP draft for pacing, remove redundancy, add missing reaction shots. Write output to `output/editor_cut.csv`.

4. **Continuity** — Read `.agents/skills/continuity/SKILL.md` as your working guide. Validate format, numbering continuity, field completeness. Write output to `output/continuity_checked.csv`.

5. **QA** — Read `.agents/skills/qa/SKILL.md` as your working guide. Final quality check: remove metaphor pollution, isolate non-visual content, verify OTS precision. Write output to `output/storyboard.csv`.

## Rules

- Execute ALL 5 stages sequentially. Do not skip any stage.
- Each stage's output feeds into the next stage as input.
- Write intermediate outputs to the specified files in `output/`.
- The final deliverable is `output/storyboard.csv`.
- CSV output must be pure CSV (no markdown code blocks), with the header row specified above.
- Reference `.agents/skills/references/` for shared knowledge (director styles, prompt templates, schema, examples, gotchas).
- Do not ask for permission or clarification. Execute autonomously.
"""


# ── Workspace preparation ─────────────────────────────────────────────────────


def prepare_acp_workspace(
    *,
    work_dir: str,
    script_text: str,
    all_skills: dict[str, str],
    skill_base_dir: str = "",
) -> str:
    """Create a workspace directory with script, skills, and instructions.

    If skill_base_dir is provided, copies the entire skill directory tree
    (references/, scripts/, CONTRACT.md, top-level SKILL.md, etc.) into
    the workspace, then overlays the optimized skill content on top of
    the director/dp SKILL.md files.

    Returns the workspace root path.
    """
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)

    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.join(work_dir, "output"), exist_ok=True)

    with open(os.path.join(work_dir, "script.txt"), "w", encoding="utf-8") as f:
        f.write(script_text)

    with open(os.path.join(work_dir, "CLAUDE.md"), "w", encoding="utf-8") as f:
        f.write(_CLAUDE_MD_TEMPLATE)

    skill_dest = os.path.join(work_dir, ".agents", "skills")

    if skill_base_dir and os.path.isdir(skill_base_dir):
        shutil.copytree(skill_base_dir, skill_dest, dirs_exist_ok=True)
    else:
        os.makedirs(skill_dest, exist_ok=True)

    for stage_name, skill_content in all_skills.items():
        stage_dir = os.path.join(skill_dest, stage_name)
        os.makedirs(stage_dir, exist_ok=True)
        with open(os.path.join(stage_dir, "SKILL.md"), "w", encoding="utf-8") as f:
            f.write(skill_content)

    return work_dir


# ── Pipeline execution via Claude Code CLI ────────────────────────────────────


def _read_output_file(work_dir: str, filename: str) -> str:
    path = os.path.join(work_dir, "output", filename)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return ""


def collect_stage_outputs(work_dir: str) -> dict[str, str]:
    """Read all intermediate stage outputs from the workspace."""
    return {
        "director": _read_output_file(work_dir, "director_notes.md"),
        "dp": _read_output_file(work_dir, "dp_draft.csv"),
        "editor": _read_output_file(work_dir, "editor_cut.csv"),
        "continuity": _read_output_file(work_dir, "continuity_checked.csv"),
        "final": _read_output_file(work_dir, "storyboard.csv"),
    }


def run_acp_pipeline(
    *,
    script_text: str,
    all_skills: dict[str, str],
    work_dir: str,
    skill_base_dir: str = "",
    model: str = "",
    timeout: int = 0,
) -> tuple[str, str]:
    """Execute the full storyboard pipeline via Claude Code subprocess.

    Spawns `claude -p` with the prepared workspace. Claude Code reads
    CLAUDE.md for instructions and executes all 5 stages autonomously,
    writing intermediate and final outputs to `output/`.

    Returns (final_csv, raw_metadata_json).
    """
    config = get_acp_exec_config()
    claude_path = str(config["claude_path"])
    resolved = shutil.which(claude_path)
    if resolved:
        claude_path = resolved
    actual_model = model or str(config.get("model", ""))
    actual_timeout = timeout or int(config["timeout"])
    permission_mode = str(config["permission_mode"])

    cmd = [
        claude_path,
        "-p",
        "--output-format", "text",
        "--permission-mode", permission_mode,
        "--add-dir", work_dir,
    ]
    if actual_model:
        cmd.extend(["--model", actual_model])

    prompt_text = (
        "Execute the storyboard pipeline as described in CLAUDE.md. "
        "Read script.txt, then run all 5 stages (director -> DP -> editor -> continuity -> QA) "
        "writing outputs to the output/ directory. Start now."
    )
    cmd.extend(["--", prompt_text])

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=actual_timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        raw = json.dumps({
            "backend": "acp_exec",
            "is_error": True,
            "error": "timeout",
            "timeout": actual_timeout,
            "stdout_chars": len(stdout),
            "stderr": stderr[:2000],
        }, ensure_ascii=False)
        final_csv = _read_output_file(work_dir, "storyboard.csv")
        _persist_raw(work_dir, raw)
        return final_csv, raw
    except Exception as exc:
        raw = json.dumps({
            "backend": "acp_exec",
            "is_error": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }, ensure_ascii=False, indent=2)
        _persist_raw(work_dir, raw)
        return "", raw

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    final_csv = _read_output_file(work_dir, "storyboard.csv")
    if not final_csv.strip():
        final_csv = stdout.strip()

    raw = json.dumps({
        "backend": "acp_exec",
        "returncode": proc.returncode,
        "stdout_chars": len(stdout),
        "stderr": stderr[:2000] if stderr else "",
        "final_csv_chars": len(final_csv),
    }, ensure_ascii=False, indent=2)

    _persist_raw(work_dir, raw)
    return final_csv, raw


def _persist_raw(work_dir: str, raw: str) -> None:
    pred_dir = os.path.dirname(work_dir.rstrip(os.sep))
    raw_path = os.path.join(pred_dir, "acp_raw.txt")
    os.makedirs(pred_dir, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw)
