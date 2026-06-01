from __future__ import annotations

import base64
import logging
import random
import time
from dataclasses import dataclass
from typing import Any

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover
    httpx = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class RetryConfig:
    max_retries: int
    initial_backoff_s: float
    max_backoff_s: float


def _should_retry(*, status_code: int | None, exc: Exception | None) -> bool:
    if exc is not None:
        # Network errors, timeouts, connection resets, etc.
        return True
    if status_code is None:
        return False
    if status_code in (408, 409, 429):
        return True
    if 500 <= status_code <= 599:
        return True
    return False


def _retry_sleep_seconds(attempt: int, *, initial: float, maximum: float) -> float:
    # Full jitter exponential backoff.
    base = min(maximum, initial * (2 ** (attempt - 1)))
    return random.uniform(0, base)


def request_with_retries(
    client,
    method: str,
    url: str,
    *,
    retry: RetryConfig,
    timeout_s: float,
    headers: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
) -> httpx.Response:
    if httpx is None:
        raise ModuleNotFoundError(
            "Missing optional dependency 'httpx'. Install it to enable provider HTTP calls."
        )
    last_exc: Exception | None = None
    for attempt in range(1, retry.max_retries + 1):
        try:
            resp = client.request(
                method,
                url,
                headers=headers,
                json=json_body,
                timeout=timeout_s,
            )
            if not _should_retry(status_code=resp.status_code, exc=None):
                return resp

            # Honor Retry-After when present.
            retry_after = resp.headers.get("retry-after")
            if retry_after is not None:
                try:
                    sleep_s = float(retry_after)
                except ValueError:
                    sleep_s = _retry_sleep_seconds(
                        attempt, initial=retry.initial_backoff_s, maximum=retry.max_backoff_s
                    )
            else:
                sleep_s = _retry_sleep_seconds(
                    attempt, initial=retry.initial_backoff_s, maximum=retry.max_backoff_s
                )

            if attempt == retry.max_retries:
                return resp
            time.sleep(sleep_s)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_exc = exc
            if attempt == retry.max_retries:
                raise
            time.sleep(
                _retry_sleep_seconds(
                    attempt, initial=retry.initial_backoff_s, maximum=retry.max_backoff_s
                )
            )
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("request_with_retries exhausted without response")


@dataclass(frozen=True)
class OpenAIClient:
    base_url: str
    api_key: str
    timeout_s: float
    retry: RetryConfig

    def _headers(self, trace_id: str) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
            "x-trace-id": trace_id,
        }

    def ocr_markdown_from_images(
        self,
        images: list[bytes],
        *,
        model: str,
        trace_id: str,
    ) -> tuple[str, Usage]:
        if httpx is None:
            raise ModuleNotFoundError(
                "Missing optional dependency 'httpx'. Install it to enable provider HTTP calls."
            )
        # Uses OpenAI-style vision message format; compatible providers may differ.
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "You are an OCR and document structure extractor. "
                    "Return clean, well-structured Markdown capturing the resume's reading order, "
                    "headings, and bullet lists. "
                    "Do not add information not present in the document."
                ),
            }
        ]
        for img in images:
            b64 = base64.b64encode(img).decode("ascii")
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            )

        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "temperature": 0,
        }

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        with httpx.Client() as client:
            resp = request_with_retries(
                client,
                "POST",
                url,
                retry=self.retry,
                timeout_s=self.timeout_s,
                headers=self._headers(trace_id),
                json_body=body,
            )

        if resp.status_code >= 400:
            logger.error(
                "OpenAI request failed",
                extra={
                    "event": "provider_error",
                    "trace_id": trace_id,
                    "provider": "openai",
                    "model": model,
                },
            )
            resp.raise_for_status()

        data = resp.json()
        usage = data.get("usage") or {}
        msg = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content")
        if not isinstance(msg, str) or not msg.strip():
            raise ValueError("Provider response missing message content")

        u = Usage(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )
        return msg.strip(), u

    def chat_text(self, *, prompt: str, model: str, trace_id: str) -> str:
        if httpx is None:
            raise ModuleNotFoundError(
                "Missing optional dependency 'httpx'. Install it to enable provider HTTP calls."
            )
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        with httpx.Client() as client:
            resp = request_with_retries(
                client,
                "POST",
                url,
                retry=self.retry,
                timeout_s=self.timeout_s,
                headers=self._headers(trace_id),
                json_body=body,
            )
        if resp.status_code >= 400:
            logger.error(
                "OpenAI request failed",
                extra={
                    "event": "provider_error",
                    "trace_id": trace_id,
                    "provider": "openai",
                    "model": model,
                },
            )
            resp.raise_for_status()
        data = resp.json()
        msg = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content")
        if not isinstance(msg, str) or not msg.strip():
            raise ValueError("Provider response missing message content")
        return msg.strip()

    def embed_texts(self, *, texts: list[str], model: str, trace_id: str) -> list[list[float]]:
        if httpx is None:
            raise ModuleNotFoundError(
                "Missing optional dependency 'httpx'. Install it to enable provider HTTP calls."
            )
        body: dict[str, Any] = {"model": model, "input": texts}
        url = f"{self.base_url.rstrip('/')}/embeddings"
        with httpx.Client() as client:
            resp = request_with_retries(
                client,
                "POST",
                url,
                retry=self.retry,
                timeout_s=self.timeout_s,
                headers=self._headers(trace_id),
                json_body=body,
            )
        if resp.status_code >= 400:
            logger.error(
                "OpenAI request failed",
                extra={
                    "event": "provider_error",
                    "trace_id": trace_id,
                    "provider": "openai",
                    "model": model,
                },
            )
            resp.raise_for_status()
        data = resp.json()
        items = data.get("data")
        if not isinstance(items, list):
            raise ValueError("Provider embeddings response missing data")
        # Preserve order.
        out: list[list[float]] = []
        for item in items:
            emb = (item or {}).get("embedding")
            if not isinstance(emb, list) or not emb:
                raise ValueError("Provider embeddings response missing embedding")
            out.append([float(x) for x in emb])
        if len(out) != len(texts):
            raise ValueError("Provider embeddings count mismatch")
        return out
