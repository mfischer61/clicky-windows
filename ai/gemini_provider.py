"""
Google Gemini provider — uses the REST SSE streaming API directly (no extra deps).
Default model: gemini-2.5-flash (fast, cheap, vision-capable).
"""

import asyncio
import os
import json
from typing import AsyncIterator, List

import httpx

from ai.base_provider import BaseLLMProvider, Message
from config import cfg

DEFAULT_MODEL = "gemini-flash-lite-latest"
# Tried in order when the chosen model 404s or stays overloaded
FALLBACK_MODELS = ["gemini-flash-lite-latest"]
_RETRY_STATUS = {429, 500, 502, 503, 504}
STREAM_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
)


class GeminiProvider(BaseLLMProvider):

    def __init__(self):
        self._api_key = cfg.google_api_key

    async def stream_response(
        self,
        user_text: str,
        screenshots_b64: List[str],
        history: List[Message],
        system_prompt: str,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        model = model or DEFAULT_MODEL

        contents = []
        for msg in history:
            role = "user" if msg.role == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg.content}],
            })

        parts: list = []
        for img_b64 in screenshots_b64:
            parts.append({
                "inline_data": {"mime_type": "image/jpeg", "data": img_b64},
            })
        parts.append({"text": user_text})
        contents.append({"role": "user", "parts": parts})

        body = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {"maxOutputTokens": int(os.getenv("CLICKY_MAX_OUTPUT_TOKENS", "1024") or 1024), "temperature": 0.7},
        }

        headers = {"x-goog-api-key": self._api_key or ""}   # keep key out of URLs/logs

        # Try the chosen model, then fall back through FALLBACK_MODELS.
        # 404 (model retired / not enabled for this key) and overloads
        # (429/5xx after retries) both move on to the next model. Google's own
        # error message is kept so the bar shows WHY it failed.
        models = []
        for m in [model, *FALLBACK_MODELS]:
            if m and m not in models:
                models.append(m)
        errors: list[str] = []
        yielded = False
        # 40 s with no data → treat model as stuck and move to the next one
        async with httpx.AsyncClient(timeout=httpx.Timeout(40.0, connect=10.0)) as client:
            for m in models:
                url = f"{STREAM_URL.format(model=m)}?alt=sse"
                for attempt in range(3):
                    try:
                        async with client.stream("POST", url, json=body, headers=headers) as resp:
                            if resp.status_code != 200:
                                raw = (await resp.aread()).decode("utf-8", "replace")
                                try:
                                    msg = json.loads(raw)["error"]["message"]
                                except Exception:
                                    msg = raw[:200] or resp.reason_phrase
                                err = f"{m}: HTTP {resp.status_code} {msg}"
                                if resp.status_code in _RETRY_STATUS and attempt < 2:
                                    await asyncio.sleep(1.5 * (attempt + 1))
                                    continue
                                errors.append(err)
                                if resp.status_code in (400, 401, 403):
                                    raise RuntimeError(f"Gemini rejected the request — {err}")
                                break          # 404 / exhausted retries → next model
                            async for line in resp.aiter_lines():
                                if not line.startswith("data: "):
                                    continue
                                data = line[6:].strip()
                                if not data or data == "[DONE]":
                                    continue
                                try:
                                    obj = json.loads(data)
                                except json.JSONDecodeError:
                                    continue
                                for cand in obj.get("candidates", []):
                                    for part in cand.get("content", {}).get("parts", []):
                                        text = part.get("text", "")
                                        if text:
                                            yielded = True
                                            yield text
                            return
                    except httpx.ReadTimeout:
                        if yielded:
                            raise RuntimeError(f"Gemini ({m}) stopped mid-answer")
                        errors.append(f"{m}: no reply in 40 s")
                        break                  # slow model → next model, no retry
                    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                        if attempt == 2:
                            errors.append(f"{m}: {type(e).__name__}")
                            break
                        await asyncio.sleep(1.5 * (attempt + 1))
        raise RuntimeError("Gemini failed — " + " | ".join(errors[-3:]))

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(
                    "https://generativelanguage.googleapis.com/v1beta/models",
                    headers={"x-goog-api-key": self._api_key or ""},
                )
                return r.status_code == 200
        except Exception:
            return False
