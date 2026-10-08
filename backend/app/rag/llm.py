"""Chat model clients: Ollama for real runs, a fake model for tests and runs without a GPU."""

import json
import re
import time
from dataclasses import dataclass

import httpx

from app.config import Settings

RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """The model could not be reached, or kept failing after the retries."""


@dataclass
class LLMReply:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OllamaLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        backoff_s: float = 1.0,
        client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.client = client or httpx.Client(timeout=timeout_s)

    def chat(self, messages: list[dict[str, str]]) -> LLMReply:
        """Send one chat request. Retries 429, 5xx and connection errors with exponential backoff."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(f"{self.base_url}/api/chat", json=payload)
            except httpx.TransportError as exc:
                last_error = type(exc).__name__
            else:
                if response.status_code == 200:
                    data = response.json()
                    return LLMReply(
                        text=data["message"]["content"],
                        prompt_tokens=data.get("prompt_eval_count", 0),
                        completion_tokens=data.get("eval_count", 0),
                    )
                if response.status_code not in RETRY_STATUS:
                    raise LLMError(f"HTTP {response.status_code}: {response.text[:200]}")
                last_error = f"HTTP {response.status_code}"
            if attempt < self.max_retries:
                time.sleep(self.backoff_s * 2**attempt)
        raise LLMError(f"gave up after {self.max_retries + 1} attempts ({last_error})")


class FakeLLM:
    """Answers with the first clause in the evidence, or with a fixed reply if one is given."""

    def __init__(self, reply: str | None = None):
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> LLMReply:
        self.calls.append(messages)
        if self.reply is not None:
            return LLMReply(self.reply)
        match = re.search(r"^\[([^\]]+)\]", messages[-1]["content"], re.MULTILINE)
        if not match:
            return LLMReply(json.dumps({"status": "refused", "answer": "Not covered.", "citations": []}))
        return LLMReply(json.dumps({"status": "answered", "answer": "See the clause.", "citations": [match.group(1)]}))


def build_llm(settings: Settings) -> OllamaLLM | FakeLLM:
    if settings.llm_provider == "fake":
        return FakeLLM()
    return OllamaLLM(settings.ollama_url, settings.llm_model, settings.llm_timeout_s, settings.llm_max_retries)
