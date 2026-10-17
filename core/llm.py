"""Shared LLM client plus per-call latency/token accounting used by the runner and UI."""
import re
import threading
import time
from urllib.parse import urlparse

import requests

import config

_llm = None
_lock = threading.Lock()
_calls: list[dict] = []

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def get_llm():
    """Return the shared chat model (created lazily so tests can inject a fake first)."""
    global _llm
    if _llm is None:
        if config.LLM_MOCK:
            from core.mock_llm import MockLLM
            _llm = MockLLM()
        else:
            from langchain_openai import ChatOpenAI
            kwargs = {}
            if config.LLM_DISABLE_THINKING:
                kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            _llm = ChatOpenAI(
                base_url=config.LLM_BASE_URL,
                api_key=config.LLM_API_KEY,
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                timeout=config.LLM_TIMEOUT_S,
                max_retries=1,
                **kwargs,
            )
    return _llm


def set_llm(llm) -> None:
    """Override the chat model (fake LLM for tests). Pass None to reset."""
    global _llm
    _llm = llm


def model_label() -> str:
    llm = get_llm()
    return getattr(llm, "model_label", None) or config.LLM_MODEL


def endpoint_label() -> str:
    if config.LLM_MOCK or getattr(get_llm(), "model_label", None):
        return "offline mock"
    parsed = urlparse(config.LLM_BASE_URL)
    return parsed.netloc or config.LLM_BASE_URL


def chat(messages, node: str) -> str:
    """Invoke the LLM, record latency + token usage, return the raw text content."""
    t0 = time.perf_counter()
    response = get_llm().invoke(messages)
    latency = time.perf_counter() - t0
    usage = getattr(response, "usage_metadata", None) or {}
    with _lock:
        _calls.append({
            "node": node,
            "latency_s": round(latency, 3),
            "prompt_tokens": int(usage.get("input_tokens", 0) or 0),
            "completion_tokens": int(usage.get("output_tokens", 0) or 0),
        })
    content = response.content
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return content or ""


def drain_calls() -> list[dict]:
    """Return and clear the LLM calls recorded since the last drain."""
    with _lock:
        calls = list(_calls)
        _calls.clear()
    return calls


def strip_think(text: str) -> str:
    """Remove Qwen-style <think> reasoning, including an unterminated/unopened block."""
    text = _THINK_RE.sub("", text or "")
    if "</think>" in text:  # chat template opened <think> for us; only the close tag is in the output
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:  # truncated mid-thought
        text = text.split("<think>", 1)[0]
    return text.strip()


def ping_endpoint(timeout: float = 3.0) -> tuple[bool, str]:
    """Check that the OpenAI-compatible endpoint answers GET /models."""
    if config.LLM_MOCK:
        return True, "offline mock LLM"
    url = config.LLM_BASE_URL.rstrip("/") + "/models"
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {config.LLM_API_KEY}"}, timeout=timeout)
    except requests.exceptions.Timeout:
        return False, f"no answer within {timeout:.0f}s"
    except requests.exceptions.ConnectionError:
        return False, "connection refused or host unreachable"
    except requests.exceptions.RequestException as e:
        return False, type(e).__name__
    if r.status_code == 200:
        return True, "ok"
    return False, f"HTTP {r.status_code}"
