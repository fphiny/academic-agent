from __future__ import annotations

from langchain_ollama import ChatOllama

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


def build_chat_llm(
    *,
    model: str,
    think: bool = False,
    host: str = DEFAULT_OLLAMA_HOST,
    temperature: float = 0.0,
) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url=host,
        reasoning=think,
        temperature=temperature,
    )