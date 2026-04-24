"""
v2.llm_provider — unified LLM call shim for the Harness.

The v1 code hard-wires ``anthropic.Anthropic()`` in two places
(``agents.base_agent`` and ``src.llm_sr_engine``). Under v2 we add a
provider shim so the backend can be switched by environment variable
without touching the call sites. Two providers are supported:

  anthropic  — Claude via the official SDK (v1 default).
  qwen       — Alibaba Qwen via its OpenAI-compatible DashScope endpoint.

Pick the provider via ``LLM_PROVIDER`` env var; default is ``anthropic``.

Credentials:
  LLM_PROVIDER=anthropic  ->  ANTHROPIC_API_KEY
  LLM_PROVIDER=qwen       ->  DASHSCOPE_API_KEY (or QWEN_API_KEY as alias)

Model selection (env override optional):
  LLM_MODEL defaults to the provider's strongest model:
    anthropic  ->  claude-sonnet-4-20250514
    qwen       ->  qwen-max
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

log = logging.getLogger("v2.llm_provider")


# -- provider detection ----------------------------------------------------


def _provider() -> str:
    return os.environ.get("LLM_PROVIDER", "anthropic").lower()


def _model() -> str:
    override = os.environ.get("LLM_MODEL")
    if override:
        return override
    if _provider() == "qwen":
        return "qwen-max"
    return "claude-sonnet-4-20250514"


# -- callable the existing code sites can invoke ---------------------------


# Per-call wall-clock cap. If the provider SDK doesn't honour the timeout
# kwarg (or the server hangs mid-stream), this is the outer guard.
_LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "90"))
_LLM_RETRIES = int(os.environ.get("LLM_RETRIES", "1"))


def call_llm(prompt: str, *, temperature: float = 0.7,
             max_tokens: int = 1024) -> dict[str, Any] | None:
    """
    Call the configured LLM and return the first JSON object parsed out of
    its response, or None if anything fails. Bounded by ``LLM_TIMEOUT_S``
    (default 90 s); retries up to ``LLM_RETRIES`` times on timeout.

    Mirrors the v1 ``_call_llm`` contract so the two existing call sites
    can drop in with minimal change.
    """
    provider = _provider()
    model = _model()

    attempt = 0
    while attempt <= _LLM_RETRIES:
        attempt += 1
        t0 = time.time()
        try:
            if provider == "qwen":
                text = _call_qwen(prompt, model, temperature, max_tokens)
            else:
                text = _call_anthropic(prompt, model, temperature, max_tokens)
        except TimeoutError as e:
            log.warning("LLM timeout (provider=%s model=%s attempt=%d): %s",
                        provider, model, attempt, e)
            if attempt > _LLM_RETRIES:
                return None
            continue
        except Exception as e:
            log.warning("LLM call failed (provider=%s model=%s): %s",
                        provider, model, e)
            return None

        dt = time.time() - t0
        log.info("LLM ok  provider=%s model=%s latency=%.2fs len(text)=%d",
                 provider, model, dt, len(text))

        # Same regex v1 used: first {...} block. Tolerant of code-fenced output.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            log.warning("LLM response contained no JSON object (head: %r)",
                        text[:120])
            return None
        try:
            return json.loads(m.group())
        except json.JSONDecodeError as e:
            log.warning("JSON parse failed: %s  (payload head: %r)",
                        e, m.group()[:200])
            return None
    return None


# -- backend-specific implementations --------------------------------------


def _call_anthropic(prompt: str, model: str, temperature: float,
                    max_tokens: int) -> str:
    import anthropic
    # Pass timeout to the SDK; raises anthropic.APITimeoutError if exceeded.
    try:
        client = anthropic.Anthropic(timeout=_LLM_TIMEOUT_S)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APITimeoutError as e:
        raise TimeoutError(f"anthropic timeout after {_LLM_TIMEOUT_S}s: {e}")
    return response.content[0].text


def _call_qwen(prompt: str, model: str, temperature: float,
               max_tokens: int) -> str:
    """
    Qwen DashScope exposes an OpenAI-compatible endpoint at
    ``https://dashscope.aliyuncs.com/compatible-mode/v1``. We use the
    openai SDK pointed at that URL with ``DASHSCOPE_API_KEY`` (or the
    ``QWEN_API_KEY`` alias) as the bearer token.
    """
    from openai import OpenAI, APITimeoutError
    api_key = (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("QWEN_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY / QWEN_API_KEY not set; cannot call Qwen."
        )
    base_url = os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    # The openai SDK exposes timeout at either client or request level.
    client = OpenAI(api_key=api_key, base_url=base_url,
                    timeout=_LLM_TIMEOUT_S)
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
    except APITimeoutError as e:
        raise TimeoutError(f"qwen timeout after {_LLM_TIMEOUT_S}s: {e}")
    return response.choices[0].message.content
