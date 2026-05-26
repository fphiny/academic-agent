from __future__ import annotations

import json
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

import requests
from langchain_ollama import ChatOllama

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:31b"
DEFAULT_OLLAMA_TIMEOUT = 300
DEFAULT_CHAT_TEMPERATURE = 0.0
DEFAULT_NUM_CTX = 16384


class OllamaClient:
    def __init__(
        self,
        *,
        host: str = DEFAULT_OLLAMA_HOST,
        default_model: str = DEFAULT_OLLAMA_MODEL,
        timeout: int = DEFAULT_OLLAMA_TIMEOUT,
        default_num_ctx: int = DEFAULT_NUM_CTX,
    ):
        self.host = host.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self.default_num_ctx = default_num_ctx

    def resolve_model(self, model: Optional[str] = None) -> str:
        value = (model or "").strip()
        return value or self.default_model

    def build_chat_llm(
        self,
        *,
        model: Optional[str] = None,
        think: bool = False,
        temperature: float = DEFAULT_CHAT_TEMPERATURE,
        num_ctx: Optional[int] = None,
    ) -> ChatOllama:
        return ChatOllama(
            model=self.resolve_model(model),
            base_url=self.host,
            reasoning=think,
            temperature=temperature,
            num_ctx=num_ctx or self.default_num_ctx,
        )

    def _build_payload(
        self,
        *,
        model: Optional[str],
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.resolve_model(model),
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_ctx": self.default_num_ctx,
            },
        }

        if tools:
            payload["tools"] = tools

        if format:
            payload["format"] = format

        if options:
            merged_options = dict(payload["options"])
            merged_options.update(options)
            payload["options"] = merged_options

        return payload

    def _post_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("ollama response is not a dict")
        return data

    def _stream_chat(self, payload: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        response = requests.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=self.timeout,
            stream=True,
        )
        response.raise_for_status()

        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue

                line = raw_line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if isinstance(data, dict):
                    yield data
        finally:
            response.close()

    def chat(
        self,
        *,
        model: Optional[str],
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], Iterable[Dict[str, Any]]]:
        payload = self._build_payload(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            stream=stream,
            format=format,
            options=options,
        )

        if stream:
            return self._stream_chat(payload)

        return self._post_chat(payload)

    def chat_from_prompts(
        self,
        *,
        model: Optional[str],
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], Iterable[Dict[str, Any]]]:
        return self.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            tools=tools,
            stream=stream,
            format=format,
            options=options,
        )

    def chat_json(
        self,
        *,
        model: Optional[str],
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        data = self.chat_from_prompts(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            format="json",
            options=options,
        )

        if not isinstance(data, dict):
            raise TypeError("chat_json does not support stream responses")

        message = data.get("message") or {}
        content = self.normalize_text_content(message.get("content"))

        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {"raw": parsed}
        except Exception:
            return {"decision": "refine", "reason": content[:1000]}

    def chat_text(
        self,
        *,
        model: Optional[str],
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        data = self.chat_from_prompts(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            options=options,
        )

        if not isinstance(data, dict):
            raise TypeError("chat_text does not support stream responses")

        message = data.get("message") or {}
        content = self.normalize_text_content(message.get("content"))
        return content.strip() if content else ""

    def chat_stream(
        self,
        *,
        model: Optional[str],
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        result = self.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            stream=True,
            format=format,
            options=options,
        )

        if isinstance(result, dict):
            yield result
            return

        for chunk in result:
            if isinstance(chunk, dict):
                yield chunk

    def stream_chat(
        self,
        *,
        model: Optional[str],
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        for chunk in self.chat_stream(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            format=format,
            options=options,
        ):
            yield chunk

    def chat_text_stream(
        self,
        *,
        model: Optional[str],
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        options: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        stream = self.chat_from_prompts(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            stream=True,
            options=options,
        )

        if isinstance(stream, dict):
            message = stream.get("message") or {}
            content = self.normalize_text_content(message.get("content"))
            if content:
                yield content
            return

        for chunk in stream:
            text = self._extract_chunk_text(chunk)
            if text:
                yield text

    def stream_chat_text(
        self,
        *,
        model: Optional[str],
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        options: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        for token in self.chat_text_stream(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            options=options,
        ):
            yield token

    def chat_stream_text(
        self,
        *,
        model: Optional[str],
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        options: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        for token in self.chat_text_stream(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            options=options,
        ):
            yield token

    @classmethod
    def _extract_chunk_text(cls, chunk: Dict[str, Any]) -> str:
        if not isinstance(chunk, dict):
            return ""

        message = chunk.get("message") or {}
        content = cls.normalize_stream_text_content(message.get("content"))
        if content:
            return content

        response_text = cls.normalize_stream_text_content(chunk.get("response"))
        if response_text:
            return response_text

        return ""

    @staticmethod
    def normalize_tool_arguments(raw_args: Any) -> Dict[str, Any]:
        if raw_args is None:
            return {}

        if isinstance(raw_args, dict):
            return dict(raw_args)

        if isinstance(raw_args, str):
            raw_args = raw_args.strip()
            if not raw_args:
                return {}

            try:
                parsed = json.loads(raw_args)
                return parsed if isinstance(parsed, dict) else {"input": parsed}
            except json.JSONDecodeError:
                return {"input": raw_args}

        return {"input": raw_args}

    @classmethod
    def extract_tool_calls(cls, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        tool_calls = message.get("tool_calls") or []
        normalized: List[Dict[str, Any]] = []

        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            name = function.get("name")
            arguments = cls.normalize_tool_arguments(function.get("arguments"))

            if name:
                normalized.append(
                    {
                        "name": name,
                        "arguments": arguments,
                    }
                )

        return normalized

    @staticmethod
    def normalize_text_content(value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, str):
            return value

        if isinstance(value, list):
            parts: List[str] = []

            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                    continue

                if isinstance(item, dict):
                    for key in ("text", "content"):
                        v = item.get(key)
                        if isinstance(v, str):
                            parts.append(v)
                            break
                    continue

                parts.append(str(item))

            return "".join(parts)

        if isinstance(value, dict):
            for key in ("text", "content"):
                v = value.get(key)
                if isinstance(v, str):
                    return v

            return ""

        return str(value)

    @staticmethod
    def normalize_stream_text_content(value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, str):
            return value

        if isinstance(value, list):
            parts: List[str] = []

            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                    continue

                if isinstance(item, dict):
                    for key in ("text", "content"):
                        v = item.get(key)
                        if isinstance(v, str):
                            parts.append(v)
                            break
                    continue

                parts.append(str(item))

            return "".join(parts)

        if isinstance(value, dict):
            for key in ("text", "content"):
                v = value.get(key)
                if isinstance(v, str):
                    return v

            return ""

        return str(value)

    @classmethod
    def extract_reasoning_text(cls, data: Dict[str, Any]) -> str:
        message = data.get("message") or {}
        additional_kwargs = message.get("additional_kwargs") or {}

        candidates = [
            message.get("thinking"),
            message.get("reasoning"),
            message.get("reasoning_content"),
            message.get("thinking_content"),
            additional_kwargs.get("thinking"),
            additional_kwargs.get("reasoning"),
            additional_kwargs.get("reasoning_content"),
            data.get("thinking"),
            data.get("reasoning"),
            data.get("reasoning_content"),
            data.get("thinking_content"),
        ]

        for value in candidates:
            if isinstance(value, str) and value.strip():
                return value.strip()

        return ""

    @staticmethod
    def legacy_json_fallback(content: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(content)
        except json.JSONDecodeError:
            return None

        if isinstance(obj, dict) and ("tool" in obj or "final" in obj):
            return obj

        return None