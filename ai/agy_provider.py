"""
Gemini via Google's Antigravity CLI (`agy`) — runs on your Google AI Pro
login. Replaces Gemini CLI, which Google cut off for personal accounts.

Headless agy only takes text, so the screenshot is saved as a JPEG in an
isolated work folder and agy is asked to open it with its file viewer.

Settings (.env, optional):
  CLICKY_AGY_MODELS   comma list for the model menu (default: "default" = agy's own pick)
  CLICKY_AGY_EFFORT   low | medium | high   (unset = agy default)
  CLICKY_AGY_PATH     full path to agy.exe
"""

import asyncio
import base64
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

_log = logging.getLogger("clicky.agy")
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_TIMEOUT = 180
_API_ENV = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "AGY_ADC_AUTH")   # keep plan login


def models() -> list:
    raw = os.getenv("CLICKY_AGY_MODELS", "default")
    return [m.strip() for m in raw.split(",") if m.strip()]


def find_agy() -> Optional[str]:
    override = os.getenv("CLICKY_AGY_PATH", "").strip().strip('"')
    if override and os.path.exists(override):
        return override
    local = os.environ.get("LOCALAPPDATA")
    if local:
        p = Path(local) / "agy" / "bin" / "agy.exe"      # official installer path
        if p.exists():
            return str(p)
    for c in (Path.home() / ".local" / "bin" / "agy",):
        if c.exists():
            return str(c)
    return shutil.which("agy")


def is_available() -> bool:
    return find_agy() is not None


def _text_of(ev: dict, *keys) -> str:
    """First non-empty text found under any of `keys` (string, or dict/list
    holding text) — agy's event field names vary between versions."""
    for k in keys:
        v = ev.get(k)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, dict):
            for kk in ("text", "content", "response", "message"):
                if isinstance(v.get(kk), str) and v[kk]:
                    return v[kk]
        if isinstance(v, list):
            parts = [x.get("text", "") for x in v if isinstance(x, dict)]
            if any(parts):
                return "".join(parts)
    return ""


class AgyProvider(BaseLLMProvider):

    def __init__(self):
        self._exe = find_agy()
        self._workdir = Path(tempfile.gettempdir()) / "clicky-agy"
        self._workdir.mkdir(parents=True, exist_ok=True)
        self._proc: Optional[asyncio.subprocess.Process] = None
        self.last_events: deque = deque(maxlen=40)   # raw stream, for debugging

    async def stream_response(
        self,
        user_text: str,
        screenshots_b64: List[str],
        history: List[Message],
        system_prompt: str,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        if not self._exe:
            raise RuntimeError("Antigravity CLI (agy) not found. Install: "
                               "irm https://antigravity.google/cli/install.ps1 | iex")

        for old in self._workdir.glob("shot_*.jpg"):
            try:
                old.unlink()
            except Exception:
                pass
        names = []
        for i, b in enumerate(screenshots_b64, start=1):
            p = self._workdir / f"shot_{i}.jpg"
            p.write_bytes(base64.b64decode(b))
            names.append(str(p))           # absolute path — agy tools want full paths

        prompt = f"INSTRUCTIONS:\n{system_prompt}\n\n"
        if history:
            prompt += "Earlier in this conversation:\n"
            for m in history[-4:]:
                prompt += f"{'User' if m.role == 'user' else 'You'}: {m.content}\n"
            prompt += "\n"
        if names:
            prompt += ("First use the view_file tool to look at this screenshot "
                       f"image: {', '.join(names)}. Do not run commands, browse, or "
                       "edit files. Then answer.\n\n")
        prompt += f"USER: {user_text}"

        args = [self._exe, "-p", prompt,
                "--output-format", "stream-json",
                "--disable-slash-commands",
                "--print-timeout", "3m"]
        if model and model != "default":
            args += ["--model", model]
        effort = os.getenv("CLICKY_AGY_EFFORT", "").strip().lower()
        if effort in ("low", "medium", "high"):
            args += ["--effort", effort]

        env = os.environ.copy()
        for k in _API_ENV:
            env.pop(k, None)
        env["NO_COLOR"] = "1"

        _log.info("agy: model=%s images=%d", model or "default", len(names))
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._workdir),
            env=env,
            limit=16 * 1024 * 1024,
            creationflags=_NO_WINDOW,
        )
        self._proc = proc
        self.last_events.clear()
        tail: deque = deque(maxlen=15)

        async def _drain():
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
                t = line.decode("utf-8", "replace").strip()
                if t:
                    tail.append(t)
                    _log.debug("agy stderr: %s", t)
        drain = asyncio.create_task(_drain())

        done = yielded = False
        try:
            while True:
                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=_TIMEOUT)
                except asyncio.TimeoutError:
                    raise RuntimeError("Antigravity CLI timed out")
                if not line:
                    break
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                self.last_events.append(line.decode("utf-8", "replace")[:600])
                kind = ev.get("event") or ev.get("type")
                # Real agy nests the payload: {"event":"result","result":{...}}
                if isinstance(ev.get(kind), dict):
                    ev = ev[kind]
                if kind == "step_update":
                    st = str(ev.get("step_type", "")).lower()
                    if st in ("user_input", "tool", "checkpoint"):
                        continue
                    delta = _text_of(ev, "text_delta", "delta", "text", "content")
                    if delta:
                        yielded = True
                        yield delta
                elif kind == "result":
                    done = True
                    status = str(ev.get("status", "")).upper()
                    if status not in ("SUCCESS", ""):
                        err = ev.get("error")
                        msg = err.get("message") if isinstance(err, dict) else err
                        raise RuntimeError(f"Antigravity: {msg or status}")
                    if not yielded:
                        final = _text_of(ev, "response", "result", "text", "structured_output")
                        if final:
                            yield final
                        else:
                            denied = ", ".join(d.get("display_name", d.get("action", "?"))
                                               for d in ev.get("denied_actions") or [])
                            raise RuntimeError("Antigravity gave no answer"
                                               + (f" (blocked tool: {denied})" if denied else ""))
                    break
            if not done:
                raise RuntimeError("Antigravity CLI exited: " +
                                   (" | ".join(list(tail)[-3:]) or "no output"))
        finally:
            if proc.returncode is None:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except Exception:
                pass
            drain.cancel()
            try:
                await drain
            except BaseException:
                pass
            self._proc = None

    async def health_check(self) -> bool:
        return self._exe is not None

    async def close(self):
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.kill()
            except Exception:
                pass
