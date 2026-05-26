from __future__ import annotations

from dataclasses import dataclass

from core.ollama.client import (
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_TIMEOUT,
)


@dataclass
class AgentConfig:
    ollama_host: str = DEFAULT_OLLAMA_HOST
    model: str = DEFAULT_OLLAMA_MODEL
    max_steps: int = 8
    temperature: float = DEFAULT_CHAT_TEMPERATURE
    default_collection: str = "kb_current"
    request_timeout: int = DEFAULT_OLLAMA_TIMEOUT