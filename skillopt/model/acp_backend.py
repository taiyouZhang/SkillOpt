"""ACP backend for running a skill via a Claude Code agent subprocess.

Prepares a workspace with input files + skill directory + minimal CLAUDE.md,
spawns Claude Code in non-interactive mode (`claude -p`), and lets it
autonomously execute the skill-defined workflow.

The CLAUDE.md only points Claude Code to the skill — all pipeline logic,
stage definitions, and output conventions are owned by the skill itself.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import traceback

from skillopt.model.backend_config import get_acp_exec_config




# ── Workspace preparation ─────────────────────────────────────────────────────


def prepare_acp_workspace(
    *,
    work_dir: str,
    script_text: str,
    all_skills: dict[str, str],
    skill_base_dir: str = "",
) -> str:
    """Create a workspace directory with input, skill files, and CLAUDE.md.

    If skill_base_dir is provided, copies the entire skill directory tree
    (including references/, scripts/, CONTRACT.md, top-level SKILL.md)
    into `.agents/skills/`, then overlays any optimized skill content.

    Returns the workspace root path.
    """
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)

    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.join(work_dir, "output"), exist_ok=True)

    with open(os.path.join(work_dir, "script.txt"), "w", encoding="utf-8") as f:
        f.write(script_text)

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
    """Read all output files from the workspace output/ directory."""
    output_dir = os.path.join(work_dir, "output")
    if not os.path.isdir(output_dir):
        return {}
    results = {}
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        if os.path.isfile(fpath):
            with open(fpath, encoding="utf-8", errors="replace") as f:
                results[fname] = f.read()
    return results


def run_acp_pipeline(
    *,
    script_text: str,
    all_skills: dict[str, str],
    work_dir: str,
    skill_base_dir: str = "",
    model: str = "",
    timeout: int = 0,
    output_filename: str = "storyboard.csv",
) -> tuple[str, str]:
    """Execute a skill via Claude Code subprocess.

    Spawns `claude -p` with the prepared workspace. Claude Code reads
    CLAUDE.md, which points it to the skill. The skill defines the
    workflow and writes outputs to `output/`.

    Parameters
    ----------
    output_filename : str
        The final output file to read from `output/` after execution.
        Defaults to "storyboard.csv" for the script-to-shots skill.

    Returns (final_output, raw_metadata_json).
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
        "Read .agents/skills/SKILL.md, then execute the workflow it defines. "
        "The input is in script.txt. Write all outputs to the output/ directory. "
        "Do not ask questions. Execute autonomously now."
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
        final_csv = _read_output_file(work_dir, output_filename)
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

    final_csv = _read_output_file(work_dir, output_filename)
    if not final_csv.strip():
        final_csv = stdout.strip()

    raw = json.dumps({
        "backend": "acp_exec",
        "returncode": proc.returncode,
        "stdout_chars": len(stdout),
        "stderr": stderr[:2000] if stderr else "",
        "output_chars": len(final_csv),
    }, ensure_ascii=False, indent=2)

    _persist_raw(work_dir, raw)
    return final_csv, raw


def _persist_raw(work_dir: str, raw: str) -> None:
    pred_dir = os.path.dirname(work_dir.rstrip(os.sep))
    raw_path = os.path.join(pred_dir, "acp_raw.txt")
    os.makedirs(pred_dir, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw)
