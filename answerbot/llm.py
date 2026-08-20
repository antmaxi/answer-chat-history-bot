"""LLM providers behind one small protocol.

Only answer *generation* goes through here — embeddings are always local — so
switching providers later touches this file and one config key, nothing else.
"""

from typing import Protocol

from . import config


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
    ):
        self.model = model or config.ANSWER_MODEL
        self.api_key = api_key
        self.base_url = (base_url or "").rstrip("/")
        self.extra_headers = extra_headers or {}
        self.label = label

    def complete(self, system: str, user: str) -> str:
        import json
        import urllib.error
        import urllib.request

        if not self.api_key:
            raise RuntimeError(f"{self.label} API key is not set")
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 1024,
                "stream": False,
            }
        ).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            **self.extra_headers,
        }
        url = f"{self.base_url}/chat/completions"
        req = urllib.request.Request(url, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            raise RuntimeError(f"{self.label} HTTP {e.code}: {detail or e.reason}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"{self.label} at {self.base_url} failed: {e}") from e
        except TimeoutError as e:
            raise RuntimeError(f"{self.label} timed out after {config.LLM_TIMEOUT}s") from e
        err = body.get("error")
        if err:
            msg = err.get("message") if isinstance(err, dict) else err
            raise RuntimeError(f"{self.label} returned an error: {msg}")
        choices = body.get("choices") or []
        content = choices[0].get("message", {}).get("content") if choices else None
        text = _openai_message_text(content)
        if not text:
            raise RuntimeError(f"{self.label} returned no text")
        return text


class GroqLLM(OpenAICompatLLM):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ):
        super().__init__(
            model=model or config.ANSWER_MODEL,
            api_key=api_key if api_key is not None else config.GROQ_API_KEY,
            base_url=base_url or "https://api.groq.com/openai/v1",
            label="Groq",
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
        import json
        import urllib.error
        import urllib.request

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
    raise ValueError(f"unknown LLM_PROVIDER: {config.LLM_PROVIDER!r}")
