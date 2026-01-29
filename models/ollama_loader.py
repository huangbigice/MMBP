from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import httpx


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    model_name: str
    timeout_seconds: float = 120.0


class OllamaLoader:
    """
    Small async client for Ollama /api/chat streaming.

    Env:
    - OLLAMA_MODEL_URL: e.g. http://localhost:11434
    - OLLAMA_MODEL_NAME: e.g. llama3.2-vision:latest
    """

    def __init__(self, config: Optional[OllamaConfig] = None) -> None:
        if config is None:
            base_url = os.getenv("OLLAMA_MODEL_URL", "http://localhost:11434").strip().rstrip("/")
            model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3.2-vision:latest").strip()
            config = OllamaConfig(base_url=base_url, model_name=model_name)
        self._config = config

    @property
    def base_url(self) -> str:
        return self._config.base_url

    @property
    def model_name(self) -> str:
        return self._config.model_name

    async def stream_chat(
        self,
        *,
        user_message: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Yield assistant content chunks as they arrive.
        """
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if context:
            messages.append({"role": "system", "content": f"Context:\n{context}"})

        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model_name,
            "stream": True,
            "messages": messages,
        }

        url = f"{self.base_url}/api/chat"

        timeout = httpx.Timeout(self._config.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        # Ollama should send JSON lines; skip malformed fragments.
                        continue

                    # Ollama chat stream shape:
                    # { message: { role: 'assistant', content: '...' }, done: bool, ... }
                    message = obj.get("message") or {}
                    content = message.get("content")
                    if isinstance(content, str) and content:
                        yield content

                    if obj.get("done") is True:
                        break