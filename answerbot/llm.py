"""LLM providers behind one small protocol.

Only answer *generation* goes through here — embeddings are always local — so
switching from Claude to a local Ollama model later touches this file and one
config key, nothing else.
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
    if provider == "ollama":
        return OllamaLLM()
    raise ValueError(f"unknown LLM_PROVIDER: {config.LLM_PROVIDER!r}")
