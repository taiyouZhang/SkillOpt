"""ACP (Agent Client Protocol) backend for running full pipeline via Claude Code agent.

Spawns a Claude Code instance as an ACP server over stdio (JSON-RPC 2.0),
dispatches the entire multi-agent pipeline as a single agentic session,
and extracts output from the workspace files.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import traceback
from typing import Any

from skillopt.model.backend_config import get_acp_exec_config


# ── CLAUDE.md template for pipeline workspace ─────────────────────────────────

_CLAUDE_MD_TEMPLATE = """\
# Storyboard Pipeline Execution

You are executing a 5-stage storyboard generation pipeline. Follow these steps exactly in order.

## Pipeline Stages

1. **Director** — Read `.agents/skills/director/SKILL.md` as your working guide. Read `script.txt` as input. Produce director notes (scene intent, narrative beats, shot density decisions, camera strategy). Write output to `output/director_notes.md`.

2. **DP (Director of Photography)** — Read `.agents/skills/dp/SKILL.md` as your working guide. Using your director notes + the original script, design concrete shots as pure CSV. Write output to `output/dp_draft.csv`. CSV columns: 镜头号,分镜组,场景,中文台词,英文台词,Prompt,时长,景别,视角,运镜,角色,道具,音效

3. **Editor** — Read `.agents/skills/editor/SKILL.md` as your working guide. Optimize the DP draft for pacing, remove redundancy, add missing reaction shots. Write output to `output/editor_cut.csv`.

4. **Continuity** — Read `.agents/skills/continuity/SKILL.md` as your working guide. Validate format, numbering continuity, field completeness. Write output to `output/continuity_checked.csv`.

5. **QA** — Read `.agents/skills/qa/SKILL.md` as your working guide. Final quality check: remove metaphor pollution, isolate non-visual content, verify OTS precision. Write output to `output/final.csv`.

## Rules

- Execute ALL 5 stages sequentially. Do not skip any stage.
- Each stage's output feeds into the next stage as input.
- Write intermediate outputs to the specified files in `output/`.
- The final deliverable is `output/final.csv`.
- CSV output must be pure CSV (no markdown code blocks), with the header row specified above.
- Do not ask for permission or clarification. Execute autonomously.
"""


# ── Workspace preparation ─────────────────────────────────────────────────────


def prepare_acp_workspace(
    *,
    work_dir: str,
    script_text: str,
    all_skills: dict[str, str],
) -> str:
    """Create a workspace directory with script, skills, and instructions.

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

    for stage_name, skill_content in all_skills.items():
        skill_dir = os.path.join(work_dir, ".agents", "skills", stage_name)
        os.makedirs(skill_dir, exist_ok=True)
        with open(os.path.join(skill_dir, "SKILL.md"), "w", encoding="utf-8") as f:
            f.write(skill_content)

    return work_dir


# ── ACP Session (JSON-RPC 2.0 over stdio) ────────────────────────────────────


class ACPSession:
    """Manages a Claude Code ACP agent subprocess over stdio JSON-RPC 2.0."""

    def __init__(
        self,
        *,
        claude_path: str = "claude",
        model: str = "",
        permission_mode: str = "bypassPermissions",
        max_turns: int = 50,
        timeout: int = 900,
    ):
        self._claude_path = claude_path
        self._model = model
        self._permission_mode = permission_mode
        self._max_turns = max_turns
        self._timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._request_id = 0
        self._session_id: str = ""
        self._lock = threading.Lock()
        self._trace: list[dict[str, Any]] = []

    def spawn(self, cwd: str) -> None:
        """Launch the Claude Code ACP server subprocess."""
        cmd = [
            self._claude_path,
            "--acp",
            "--permission-mode", self._permission_mode,
        ]
        if self._model:
            cmd.extend(["--model", self._model])
        if self._max_turns:
            cmd.extend(["--max-turns", str(self._max_turns)])

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            text=True,
            bufsize=1,
        )

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _send(self, method: str, params: dict[str, Any] | None = None) -> int:
        """Send a JSON-RPC 2.0 request."""
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("ACP session not spawned")
        req_id = self._next_id()
        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
        }
        if params:
            message["params"] = params
        line = json.dumps(message, ensure_ascii=False) + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()
        self._trace.append({"direction": "send", "message": message})
        return req_id

    def _recv(self, *, timeout: float | None = None) -> dict[str, Any]:
        """Read a single JSON-RPC message from stdout."""
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("ACP session not spawned")

        deadline = time.time() + (timeout or self._timeout)
        while time.time() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                if self._proc.poll() is not None:
                    raise RuntimeError(
                        f"ACP subprocess exited with code {self._proc.returncode}"
                    )
                continue
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                self._trace.append({"direction": "recv", "message": msg})
                return msg
            except json.JSONDecodeError:
                continue

        raise TimeoutError(f"ACP recv timed out after {timeout or self._timeout}s")

    def _recv_response(self, req_id: int) -> dict[str, Any]:
        """Read messages until we get the response matching req_id."""
        while True:
            msg = self._recv()
            if msg.get("id") == req_id:
                if "error" in msg:
                    raise RuntimeError(f"ACP error: {msg['error']}")
                return msg.get("result", {})

    def _recv_until_turn_complete(self, req_id: int) -> tuple[str, list[dict[str, Any]]]:
        """Read messages until turn completes. Returns (final_text, updates)."""
        updates: list[dict[str, Any]] = []
        while True:
            msg = self._recv()
            if msg.get("id") == req_id:
                if "error" in msg:
                    raise RuntimeError(f"ACP error: {msg['error']}")
                result = msg.get("result", {})
                return str(result.get("text", "")), updates
            if msg.get("method") == "session/update":
                updates.append(msg.get("params", {}))

    def initialize(self) -> dict[str, Any]:
        """Send initialize handshake."""
        req_id = self._send("initialize", {
            "clientInfo": {"name": "skillopt", "version": "1.0.0"},
            "capabilities": {},
        })
        return self._recv_response(req_id)

    def new_session(self, cwd: str) -> str:
        """Create a new ACP session."""
        req_id = self._send("session/new", {"cwd": cwd})
        result = self._recv_response(req_id)
        self._session_id = str(result.get("sessionId", ""))
        return self._session_id

    def prompt(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        """Send a prompt and wait for the turn to complete."""
        req_id = self._send("session/prompt", {
            "sessionId": self._session_id,
            "text": text,
        })
        return self._recv_until_turn_complete(req_id)

    def close(self) -> None:
        """Terminate the ACP subprocess."""
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            finally:
                self._proc = None

    @property
    def trace(self) -> list[dict[str, Any]]:
        return list(self._trace)


# ── High-level pipeline execution ─────────────────────────────────────────────


def _read_output_file(work_dir: str, filename: str) -> str:
    path = os.path.join(work_dir, "output", filename)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return ""


def _collect_stage_outputs(work_dir: str) -> dict[str, str]:
    """Read all intermediate stage outputs from the workspace."""
    return {
        "director": _read_output_file(work_dir, "director_notes.md"),
        "dp": _read_output_file(work_dir, "dp_draft.csv"),
        "editor": _read_output_file(work_dir, "editor_cut.csv"),
        "continuity": _read_output_file(work_dir, "continuity_checked.csv"),
        "final": _read_output_file(work_dir, "final.csv"),
    }


def _save_trace(work_dir: str, trace: list[dict[str, Any]], response: str) -> str:
    """Persist raw trace and return serialized version."""
    raw = json.dumps({
        "backend": "acp_exec",
        "trace_messages": len(trace),
        "response_chars": len(response),
        "trace": trace[-100:],  # Keep last 100 messages to avoid huge files
    }, ensure_ascii=False, indent=2)

    pred_dir = os.path.dirname(work_dir.rstrip(os.sep))
    raw_path = os.path.join(pred_dir, "acp_raw.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw)
    return raw


def run_acp_pipeline(
    *,
    script_text: str,
    all_skills: dict[str, str],
    work_dir: str,
    model: str = "",
    timeout: int = 0,
    max_turns: int = 0,
) -> tuple[str, str]:
    """Execute the full storyboard pipeline via ACP.

    Returns (final_csv, raw_trace_json).
    """
    config = get_acp_exec_config()
    claude_path = str(config["claude_path"])
    actual_model = model or str(config.get("model", ""))
    actual_timeout = timeout or int(config["timeout"])
    actual_max_turns = max_turns or int(config["max_turns"])
    permission_mode = str(config["permission_mode"])

    session = ACPSession(
        claude_path=claude_path,
        model=actual_model,
        permission_mode=permission_mode,
        max_turns=actual_max_turns,
        timeout=actual_timeout,
    )

    try:
        session.spawn(cwd=work_dir)
        session.initialize()
        session.new_session(cwd=work_dir)

        prompt_text = (
            "Execute the storyboard pipeline as described in CLAUDE.md. "
            "Read script.txt, then run all 5 stages (director -> DP -> editor -> continuity -> QA) "
            "writing outputs to the output/ directory. Start now."
        )
        response_text, updates = session.prompt(prompt_text)

    except Exception as exc:
        raw = json.dumps({
            "backend": "acp_exec",
            "is_error": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "trace": session.trace[-50:],
        }, ensure_ascii=False, indent=2)
        return "", raw
    finally:
        session.close()

    final_csv = _read_output_file(work_dir, "final.csv")
    if not final_csv.strip():
        final_csv = response_text

    raw = _save_trace(work_dir, session.trace, final_csv)

    return final_csv, raw


# ── Fallback: CLI-based execution (no ACP protocol, uses claude -p) ───────────


def run_acp_pipeline_cli_fallback(
    *,
    script_text: str,
    all_skills: dict[str, str],
    work_dir: str,
    model: str = "",
    timeout: int = 0,
) -> tuple[str, str]:
    """Fallback: run pipeline via `claude -p` with workspace + CLAUDE.md.

    This doesn't use the ACP JSON-RPC protocol but achieves the same result
    by letting Claude Code read the workspace's CLAUDE.md and execute autonomously.
    """
    config = get_acp_exec_config()
    claude_path = str(config["claude_path"])
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

    try:
        proc = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=actual_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        raw = json.dumps({
            "backend": "acp_exec_cli_fallback",
            "is_error": True,
            "error": "timeout",
            "stdout_chars": len(stdout),
            "stderr": stderr[:2000],
        }, ensure_ascii=False)
        final_csv = _read_output_file(work_dir, "final.csv")
        return final_csv, raw

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    final_csv = _read_output_file(work_dir, "final.csv")
    if not final_csv.strip():
        final_csv = stdout.strip()

    raw = json.dumps({
        "backend": "acp_exec_cli_fallback",
        "returncode": proc.returncode,
        "stdout_chars": len(stdout),
        "stderr": stderr[:2000] if stderr else "",
        "final_csv_chars": len(final_csv),
    }, ensure_ascii=False, indent=2)

    pred_dir = os.path.dirname(work_dir.rstrip(os.sep))
    with open(os.path.join(pred_dir, "acp_raw.txt"), "w", encoding="utf-8") as f:
        f.write(raw)

    return final_csv, raw
