"""
Claude via the official Claude Code CLI — runs on your Pro/Max plan login,
no API key needed.

Keeps ONE `claude -p` process alive in stream-json mode so follow-up
questions skip the ~1-3 s CLI startup. The process is restarted when the
model or system prompt changes, after CLICKY_CLAUDE_CLI_TURNS questions
(keeps context small and replies fast), or if a reply is cancelled.

Settings (.env, all optional):
  CLICKY_CLAUDE_CLI_MODEL   sonnet | opus | haiku | full model id  (default sonnet)
  CLICKY_CLAUDE_CLI_EFFORT  low | medium | high | max              (default: CLI default)
  CLICKY_CLAUDE_CLI_TURNS   questions per warm session            (default 6)
  CLICKY_CLAUDE_CLI_PATH    full path to claude.exe / claude.cmd   (default: PATH lookup)
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import AsyncIterator, List, Optional

from ai.base_provider import BaseLLMProvider, Message

_log = logging.getLogger("clicky.claude_cli")

# (model id passed to --model, label shown in the dropdown)
MODELS = [
    ("sonnet", "Claude Sonnet (plan)"),
    ("opus",   "Claude Opus (plan)"),
    ("haiku",  "Claude Haiku (plan)"),
]
DEFAULT_MODEL = os.getenv("CLICKY_CLAUDE_CLI_MODEL", "sonnet").strip() or "sonnet"
_MAX_TURNS = max(1, int(os.getenv("CLICKY_CLAUDE_CLI_TURNS", "6") or 6))
_REPLY_TIMEOUT = 180          # seconds of silence before we give up on a reply
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0   # CREATE_NO_WINDOW

# Env vars that would make the CLI bill an API key instead of the plan login
_API_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")


def find_claude() -> Optional[str]:
    """Locate the Claude Code CLI. Native claude.exe preferred over npm shim."""
    override = os.getenv("CLICKY_CLAUDE_CLI_PATH", "").strip().strip('"')
    if override and os.path.exists(override):
        return override
    home = Path.home()
    candidates = [
        home / ".local" / "bin" / "claude.exe",           # native installer
        home / ".claude" / "local" / "claude.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    found = shutil.which("claude")                         # PATHEXT → .exe/.cmd
    if found:
        return found
    appdata = os.environ.get("APPDATA")
    if appdata:
        npm = Path(appdata) / "npm" / "claude.cmd"
        if npm.exists():
            return str(npm)
    return None


def _clean_model(model: Optional[str]) -> str:
    """Only pass Claude model names to --model (other panels may hand us an
    Ollama/Gemini id when providers are switched from the tray)."""
    m = (model or "").strip()
    if m in {mid for mid, _ in MODELS} or m.startswith("claude-"):
        return m
    return DEFAULT_MODEL


def is_available() -> bool:
    return find_claude() is not None


class ClaudeCLIProvider(BaseLLMProvider):

    def __init__(self):
        self._exe = find_claude()
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._key: Optional[tuple] = None
        self._turns = 0
        self._lock = asyncio.Lock()
        self._stderr_tail: deque = deque(maxlen=20)
        self._stderr_task: Optional[asyncio.Task] = None
        self._workdir = Path(tempfile.gettempdir()) / "clicky-claude"
        self._workdir.mkdir(parents=True, exist_ok=True)

    # ── process management ───────────────────────────────────────────────

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def _kill(self):
        proc, self._proc = self._proc, None
        self._key = None
        self._turns = 0
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=3)
        except Exception:
            pass

    async def _drain_stderr(self, proc):
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
                txt = line.decode("utf-8", "replace").rstrip()
                if txt:
                    self._stderr_tail.append(txt)
                    _log.debug("claude stderr: %s", txt)
        except Exception:
            pass

    async def _ensure(self, model: str, system_prompt: str):
        key = (model, system_prompt)
        if self._alive() and self._key == key and self._turns < _MAX_TURNS:
            return
        await self._kill()
        if not self._exe:
            raise RuntimeError(
                "Claude Code CLI not found. Install it (npm i -g "
                "@anthropic-ai/claude-code), run `claude` once to sign in, "
                "or set CLICKY_CLAUDE_CLI_PATH in .env."
            )
        sys_file = self._workdir / "system_prompt.txt"
        sys_file.write_text(system_prompt or "", encoding="utf-8")

        args = [
            self._exe, "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--model", model,
            "--system-prompt-file", str(sys_file),
            "--tools", "",                      # pure chat: no file/shell tools
            "--strict-mcp-config",              # skip MCP servers → faster start
            "--setting-sources", "project",     # skip user hooks/plugins
            "--no-session-persistence",
        ]
        effort = os.getenv("CLICKY_CLAUDE_CLI_EFFORT", "").strip().lower()
        if effort in ("low", "medium", "high", "max"):
            args += ["--effort", effort]

        env = os.environ.copy()
        for k in _API_ENV:
            env.pop(k, None)                    # force plan (OAuth) login

        _log.info("starting claude CLI: model=%s exe=%s", model, self._exe)
        self._stderr_tail.clear()
        self._proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._workdir),             # empty dir → no project CLAUDE.md
            env=env,
            limit=32 * 1024 * 1024,             # stream-json lines can be large
            creationflags=_NO_WINDOW,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr(self._proc))
        self._key = key
        self._turns = 0

    def _err(self, fallback: str) -> str:
        tail = " | ".join(list(self._stderr_tail)[-3:])
        return f"{fallback}: {tail}" if tail else fallback

    # ── public API ───────────────────────────────────────────────────────

    async def prewarm(self, system_prompt: str, model: str | None = None):
        """Spawn the CLI ahead of the first question (called when the bar opens)."""
        async with self._lock:
            try:
                await self._ensure(_clean_model(model), system_prompt)
            except Exception:
                _log.debug("prewarm failed", exc_info=True)

    async def stream_response(
        self,
        user_text: str,
        screenshots_b64: List[str],
        history: List[Message],
        system_prompt: str,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        # `history` is unused: the warm CLI session keeps its own context.
        model = _clean_model(model)
        async with self._lock:
            await self._ensure(model, system_prompt)
            proc = self._proc

            content: list = [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/jpeg", "data": b}}
                for b in screenshots_b64
            ]
            content.append({"type": "text", "text": user_text})
            msg = {
                "type": "user",
                "message": {"role": "user", "content": content},
                "parent_tool_use_id": None,
                "session_id": "",
            }
            try:
                proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
                await proc.stdin.drain()
            except Exception as e:
                await self._kill()
                raise RuntimeError(self._err(f"Claude CLI stopped ({e})"))
            self._turns += 1

            finished = False
            yielded = False
            try:
                while True:
                    try:
                        line = await asyncio.wait_for(
                            proc.stdout.readline(), timeout=_REPLY_TIMEOUT)
                    except asyncio.TimeoutError:
                        raise RuntimeError("Claude CLI timed out")
                    if not line:
                        raise RuntimeError(self._err("Claude CLI exited"))
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    t = ev.get("type")
                    if t == "stream_event":
                        e = ev.get("event") or {}
                        if e.get("type") == "content_block_delta":
                            d = e.get("delta") or {}
                            if d.get("type") == "text_delta" and d.get("text"):
                                yielded = True
                                yield d["text"]
                    elif t == "assistant" and not yielded:
                        # Partial events missing (older CLI) → use whole message
                        for block in (ev.get("message") or {}).get("content") or []:
                            if block.get("type") == "text" and block.get("text"):
                                yielded = True
                                yield block["text"]
                    elif t == "result":
                        finished = True
                        if ev.get("is_error") or ev.get("subtype") != "success":
                            text = ev.get("result") or ev.get("subtype") or "error"
                            raise RuntimeError(f"Claude: {text}")
                        if not yielded and ev.get("result"):
                            yield ev["result"]
                        break
            finally:
                if not finished:
                    # Reply cancelled or broken → stream is out of sync; restart.
                    await self._kill()

    async def health_check(self) -> bool:
        return self._exe is not None

    async def close(self):
        async with self._lock:
            await self._kill()
