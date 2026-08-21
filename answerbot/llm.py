"""LLM providers behind one small protocol.

Only answer *generation* goes through here — embeddings are always local — so
switching providers later touches this file and one config key, nothing else.
"""

import json
import re
import tempfile
import time
import urllib.error
import urllib.request
from typing import Protocol

from . import config


# Cloudflare (in front of Groq) rejects Python's default urllib User-Agent
# with a bare 403. Any non-empty, non-urllib value is enough.
_HTTP_USER_AGENT = "answer-chat-history-bot/0.1.0"

# Groq on_demand TPM for gpt-oss-20b is 8000; each request reserves
# prompt + max_completion_tokens against it. Filling that window on every
# call 413s a too-large request and 429s the next question a few seconds later.
_GROQ_DEFAULT_MAX_REQUEST_TOKENS = 8000
_GROQ_DEFAULT_MAX_COMPLETION_TOKENS = 2048
_MIN_COMPLETION_TOKENS = 256
_CHAT_OVERHEAD_TOKENS = 24
_MAX_429_RETRIES = 1
_MAX_429_WAIT_S = 20.0
_RETRY_AFTER_RE = re.compile(r"try again in (\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def _estimate_tokens(text: str) -> int:
    """Conservative count so Groq's TPM check is not undershot."""
    return max(1, (len(text.encode("utf-8")) + 1) // 2)


def _completion_tokens(system: str, user: str, wanted: int, cap: int) -> int:
    """Shrink completion so prompt + completion stay under cap. cap 0 = no shrink."""
    if 0 < cap <= _GROQ_DEFAULT_MAX_REQUEST_TOKENS:
        wanted = min(wanted, _GROQ_DEFAULT_MAX_COMPLETION_TOKENS)
    if cap <= 0:
        return wanted
    prompt = _estimate_tokens(system) + _estimate_tokens(user) + _CHAT_OVERHEAD_TOKENS
    room = cap - prompt
    if room < _MIN_COMPLETION_TOKENS:
        raise RuntimeError(
            f"prompt is too large for the request budget "
            f"(~{prompt} tokens, limit {cap})"
        )
    return min(wanted, room)


def _retry_wait_s(detail: str) -> float | None:
    """Seconds Groq asked us to wait, or None if we should not retry."""
    m = _RETRY_AFTER_RE.search(detail)
    if not m:
        return None
    wait = float(m.group(1)) + 0.25
    if wait > _MAX_429_WAIT_S:
        return None
    return wait


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so Authorization is never forwarded to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _http_open(req, timeout):
    return urllib.request.build_opener(_NoRedirectHandler).open(req, timeout=timeout)


def _openai_api_error_text(raw: bytes, fallback: str) -> str:
    """Best-effort message from an OpenAI-style error body; never the raw payload."""
    try:
        data = json.loads(raw.decode(errors="replace"))
    except json.JSONDecodeError:
        return fallback
    if not isinstance(data, dict):
        return fallback
    err = data.get("error")
    if isinstance(err, dict):
        msg = err.get("message")
        return str(msg) if msg else fallback
    if isinstance(err, str) and err.strip():
        return err
    return fallback


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class ClaudeLLM:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic(
            api_key=api_key or config.ANTHROPIC_API_KEY,
            timeout=config.LLM_TIMEOUT,
        )
        self.model = model or config.ANSWER_MODEL

    def complete(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text").strip()


class GeminiLLM:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        from google import genai

        self.client = genai.Client(
            api_key=api_key or config.GEMINI_API_KEY,
            http_options={"timeout": int(config.LLM_TIMEOUT * 1000)},
        )
        self.model = model or config.ANSWER_MODEL

    def complete(self, system: str, user: str) -> str:
        resp = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config={
                "system_instruction": system,
                "max_output_tokens": 1024,
            },
        )
        text = (getattr(resp, "text", None) or "").strip()
        if not text:
            raise RuntimeError("Gemini returned no text")
        return text


def _openai_message_text(content) -> str:
    """Flatten OpenAI-style message.content (string or list of parts)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type", "text") == "text":
                parts.append(str(p.get("text") or ""))
        return "".join(parts).strip()
    return ""


class OpenAICompatLLM:
    """Groq and OpenRouter both speak the OpenAI Chat Completions API."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_headers: dict[str, str] | None = None,
        label: str = "LLM",
        max_request_tokens: int | None = None,
    ):
        self.model = model or config.ANSWER_MODEL
        self.api_key = api_key
        self.base_url = (base_url or "").rstrip("/")
        self.extra_headers = extra_headers or {}
        self.label = label
        self.max_request_tokens = (
            max_request_tokens
            if max_request_tokens is not None
            else config.ANSWER_MAX_REQUEST_TOKENS
        )

    def complete(self, system: str, user: str) -> str:
        if not self.api_key:
            raise RuntimeError(f"{self.label} API key is not set")
        try:
            max_tokens = _completion_tokens(
                system, user, config.ANSWER_MAX_TOKENS, self.max_request_tokens
            )
        except RuntimeError as e:
            raise RuntimeError(f"{self.label} {e}") from None
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                # Both names: reasoning models want max_completion_tokens;
                # older OpenAI-compat endpoints only honour max_tokens.
                "max_tokens": max_tokens,
                "max_completion_tokens": max_tokens,
                "stream": False,
            }
        ).encode()
        headers = {
            "Content-Type": "application/json",
            "User-Agent": _HTTP_USER_AGENT,
            **self.extra_headers,
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"
        req = urllib.request.Request(url, data=payload, headers=headers)
        retries = 0
        try:
            while True:
                try:
                    with _http_open(req, timeout=config.LLM_TIMEOUT) as resp:
                        body = json.loads(resp.read())
                    break
                except urllib.error.HTTPError as e:
                    detail = _openai_api_error_text(e.read(), e.reason)
                    wait = _retry_wait_s(detail) if e.code == 429 else None
                    if wait is not None and retries < _MAX_429_RETRIES:
                        retries += 1
                        time.sleep(wait)
                        continue
                    raise RuntimeError(f"{self.label} HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"{self.label} at {self.base_url} failed: {e}") from e
        except TimeoutError as e:
            raise RuntimeError(f"{self.label} timed out after {config.LLM_TIMEOUT}s") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"{self.label} returned invalid JSON") from e
        if not isinstance(body, dict):
            raise RuntimeError(f"{self.label} returned invalid JSON")
        err = body.get("error")
        if err:
            msg = err.get("message") if isinstance(err, dict) else err
            raise RuntimeError(f"{self.label} returned an error: {msg}")
        choices = body.get("choices") or []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        text = _openai_message_text(message.get("content"))
        if not text:
            if choice.get("finish_reason") == "length":
                raise RuntimeError(
                    f"{self.label} hit the token limit before producing an answer"
                )
            raise RuntimeError(f"{self.label} returned no text")
        return text


class GroqLLM(OpenAICompatLLM):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_request_tokens: int | None = None,
    ):
        if max_request_tokens is None:
            max_request_tokens = (
                config.ANSWER_MAX_REQUEST_TOKENS or _GROQ_DEFAULT_MAX_REQUEST_TOKENS
            )
        super().__init__(
            model=model or config.ANSWER_MODEL,
            api_key=api_key if api_key is not None else config.GROQ_API_KEY,
            base_url=base_url or "https://api.groq.com/openai/v1",
            label="Groq",
            max_request_tokens=max_request_tokens,
        )


class OpenRouterLLM(OpenAICompatLLM):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ):
        extra = {"X-Title": config.OPENROUTER_APP_TITLE}
        if config.OPENROUTER_HTTP_REFERER:
            extra["HTTP-Referer"] = config.OPENROUTER_HTTP_REFERER
        super().__init__(
            model=model or config.ANSWER_MODEL,
            api_key=api_key if api_key is not None else config.OPENROUTER_API_KEY,
            base_url=base_url or "https://openrouter.ai/api/v1",
            extra_headers=extra,
            label="OpenRouter",
        )


class OllamaLLM:
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or config.ANSWER_MODEL
        self.host = (host or config.OLLAMA_HOST).rstrip("/")

    def complete(self, system: str, user: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "system": system,
                "prompt": user,
                "stream": False,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as resp:
                body = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama at {self.host} failed: {e}") from e
        except TimeoutError as e:
            raise RuntimeError(f"Ollama at {self.host} timed out after {config.LLM_TIMEOUT}s") from e
        text = body.get("response")
        if not isinstance(text, str) or not text.strip():
            err = body.get("error") or "empty response"
            raise RuntimeError(f"Ollama at {self.host} returned no text: {err}")
        return text.strip()


def _cursor_sdk():
    """Lazy import so other providers do not need cursor-sdk installed."""
    try:
        from cursor_sdk import Agent, AgentOptions, CursorAgentError, LocalAgentOptions
    except ImportError as e:
        raise RuntimeError(
            "Cursor provider requires the cursor-sdk package. "
            "Install with: pip install 'answerbot[cursor]'"
        ) from e
    return Agent, AgentOptions, CursorAgentError, LocalAgentOptions


class CursorLLM:
    """Cursor SDK agent billed to the Cursor subscription, not a chat-completions API.

    Concatenates system + user into one prompt (the SDK has no system role) and
    offers no tools so the local agent can only return text.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        cwd: str | None = None,
    ):
        self.api_key = api_key if api_key is not None else config.CURSOR_API_KEY
        self.model = model or config.ANSWER_MODEL
        self.cwd = cwd

    def complete(self, system: str, user: str) -> str:
        if not self.api_key:
            raise RuntimeError("Cursor API key is not set")
        Agent, AgentOptions, CursorAgentError, LocalAgentOptions = _cursor_sdk()
        prompt = f"{system.strip()}\n\n{user.strip()}"

        def run(cwd: str):
            try:
                return Agent.prompt(
                    prompt,
                    AgentOptions(
                        api_key=self.api_key,
                        model=self.model,
                        tools=[],
                        local=LocalAgentOptions(cwd=cwd),
                    ),
                )
            except CursorAgentError as e:
                raise RuntimeError(f"Cursor: {e.message}") from e

        if self.cwd:
            result = run(self.cwd)
        else:
            with tempfile.TemporaryDirectory(prefix="answerbot-cursor-") as cwd:
                result = run(cwd)

        if result.status != "finished":
            run_id = getattr(result, "id", "") or ""
            suffix = f" ({run_id})" if run_id else ""
            raise RuntimeError(f"Cursor run {result.status}{suffix}")
        text = (result.result or "").strip()
        if not text:
            raise RuntimeError("Cursor returned no text")
        return text


def get_llm() -> LLM:
    provider = config.LLM_PROVIDER.lower()
    if provider == "claude":
        return ClaudeLLM()
    if provider == "gemini":
        return GeminiLLM()
    if provider == "ollama":
        return OllamaLLM()
    if provider == "groq":
        return GroqLLM()
    if provider == "openrouter":
        return OpenRouterLLM()
    if provider == "cursor":
        return CursorLLM()
    raise ValueError(f"unknown LLM_PROVIDER: {config.LLM_PROVIDER!r}")
