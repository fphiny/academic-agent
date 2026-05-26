from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests

from modules.chroma.store import ChromaStore, get_store

try:
    from modules.chroma.alias_store import resolve_alias
except Exception:
    def resolve_alias(name: str) -> str:
        return name


@dataclass
class RAGConfig:
    ollama_host: str = "http://127.0.0.1:11434"
    model: str = "qwen3.5:122b"
    collection_name: str = "kb_current"

    # 1차 retrieval 후보 개수
    retrieval_k: int = 30

    # 최종 LLM 입력 개수
    rerank_top_k: int = 5

    max_context_chars: int = 6000
    temperature: float = 0.2
    think: bool = False
    distance_threshold: Optional[float] = None

    # reranker on/off
    rerank_enabled: bool = True

    # debug
    debug: bool = False

    system_prompt: str = (
        "너는 문서 기반 질의응답 어시스턴트다.\n"
        "반드시 제공된 검색 문맥을 우선 근거로 사용해 답변하라.\n"
        "이전 대화 기록은 사용자의 현재 질문 의도와 맥락을 이해하기 위한 참고 정보로만 사용하라.\n"
        "이전 대화 기록에 나온 정보만으로 사실을 단정하지 말고, 사실 판단은 반드시 검색 문맥에서 다시 확인하라.\n"
        "검색 문맥에 없는 내용은 추측하지 말고, 모르면 모른다고 답하라.\n"
        "답변은 간결하지만 필요한 정보는 빠뜨리지 말고 정리해서 답하라.\n"
    )


@dataclass
class RetrievedChunk:
    id: str
    document: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    distance: Optional[float] = None
    rerank_score: Optional[float] = None
    matched_query: Optional[str] = None


@dataclass
class RAGResult:
    answer: str
    collection_name: str
    resolved_collection_name: str
    query: str
    context: str
    sources: List[Dict[str, Any]]
    raw_retrievals: List[RetrievedChunk]
    debug_info: Dict[str, Any] = field(default_factory=dict)


class RAGService:
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        store: Optional[ChromaStore] = None,
    ):
        self.config = config or RAGConfig()
        self.store = store or get_store(
            persist_directory="./data/chroma",
            alias_resolver=resolve_alias,
        )

    def _debug(self, *args):
        if self.config.debug:
            print("[RAG]", *args)

    def _history_to_text(self, history_records) -> str:
        if not history_records:
            return ""

        lines: List[str] = []

        for msg in history_records:
            role = getattr(msg, "type", None) or msg.__class__.__name__.replace("Message", "").lower()
            content = getattr(msg, "content", "")

            if isinstance(content, list):
                content = " ".join(str(x) for x in content if x is not None)
            elif content is None:
                content = ""
            else:
                content = str(content)

            content = content.strip()
            if not content:
                continue

            if role == "human":
                role_name = "user"
            elif role == "ai":
                role_name = "assistant"
            else:
                role_name = str(role)

            lines.append(f"{role_name}: {content}")

        return "\n".join(lines).strip()

    def _normalize_query_text(self, text: str) -> str:
        text = (text or "").strip()
        return " ".join(text.split())

    def _rows_to_chunks(
        self,
        rows: List[Dict[str, Any]],
        matched_query: str,
    ) -> List[RetrievedChunk]:
        chunks: List[RetrievedChunk] = []

        for row in rows:
            distance = row.get("distance")
            if (
                self.config.distance_threshold is not None
                and distance is not None
                and distance > self.config.distance_threshold
            ):
                continue

            chunks.append(
                RetrievedChunk(
                    id=row.get("id"),
                    document=row.get("document") or "",
                    metadata=row.get("metadata") or {},
                    distance=distance,
                    rerank_score=row.get("rerank_score"),
                    matched_query=matched_query,
                )
            )

        return chunks

    def retrieve_once(
        self,
        query: str,
        collection_name: Optional[str] = None,
        k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        target_collection = collection_name or self.config.collection_name
        top_k = k or self.config.retrieval_k

        rows = self.store.similarity_search(
            collection_name=target_collection,
            query=query,
            k=top_k,
            where=where,
            rerank=False,
        )
        return self._rows_to_chunks(rows, matched_query=query)

    def retrieve(
        self,
        query: str,
        collection_name: Optional[str] = None,
        k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        normalized_query = self._normalize_query_text(query)
        self._debug("query:", normalized_query)

        chunks = self.retrieve_once(
            query=normalized_query,
            collection_name=collection_name,
            k=k,
            where=where,
        )

        self._debug("retrieve_once:", normalized_query, "->", len(chunks), "chunks")

        def sort_key(c: RetrievedChunk):
            return c.distance if c.distance is not None else 999999.0

        chunks.sort(key=sort_key)

        limit = k or self.config.retrieval_k
        return chunks[:limit]

    def search(
        self,
        query: str,
        collection_name: Optional[str] = None,
        retrieval_k: Optional[int] = None,
        final_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[RetrievedChunk], List[RetrievedChunk], Dict[str, Any]]:
        normalized_query = self._normalize_query_text(query)
        target_collection = collection_name or self.config.collection_name

        raw_rows = self.store.similarity_search(
            collection_name=target_collection,
            query=normalized_query,
            k=retrieval_k or self.config.retrieval_k,
            where=where,
            rerank=False,
        )
        retrieved_chunks = self._rows_to_chunks(raw_rows, matched_query=normalized_query)

        final_rows = self.store.similarity_search(
            collection_name=target_collection,
            query=normalized_query,
            k=final_k or self.config.rerank_top_k,
            where=where,
            rerank=self.config.rerank_enabled,
            retrieval_k=retrieval_k or self.config.retrieval_k,
        )
        final_chunks = self._rows_to_chunks(final_rows, matched_query=normalized_query)

        debug_info = {
            "retrieved_count": len(retrieved_chunks),
            "final_count": len(final_chunks),
            "rerank_enabled": self.config.rerank_enabled,
            "reranker_loaded": None,  # 이제 store/rerank 계층 책임
        }
        return retrieved_chunks, final_chunks, debug_info

    def build_context(
        self,
        chunks: List[RetrievedChunk],
        max_context_chars: Optional[int] = None,
    ) -> str:
        parts: List[str] = []

        for idx, chunk in enumerate(chunks, start=1):
            meta = chunk.metadata or {}

            header = (
                f"[출처 {idx}]\n"
                f"id: {chunk.id}\n"
                f"title: {meta.get('title', '')}\n"
                f"source: {meta.get('source', '')}\n"
                f"doc_id: {meta.get('doc_id', '')}\n"
                f"chunk_id: {meta.get('chunk_id', '')}\n"
                f"category: {meta.get('category', '')}\n"
                f"distance: {chunk.distance}\n"
                f"rerank_score: {chunk.rerank_score}\n"
                f"matched_query: {chunk.matched_query}\n"
            )

            body = (chunk.document or "").strip()
            piece = f"{header}\n내용:\n{body}\n"
            parts.append(piece)

        return "\n\n".join(parts).strip()
    
    def build_sources(self, chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks, start=1):
            meta = chunk.metadata or {}
            full_text = (chunk.document or "").strip()

            sources.append(
                {
                    "rank": idx,
                    "id": chunk.id,
                    "distance": chunk.distance,
                    "rerank_score": chunk.rerank_score,
                    "matched_query": chunk.matched_query,
                    "doc_id": meta.get("doc_id"),
                    "chunk_id": meta.get("chunk_id"),
                    "title": meta.get("title"),
                    "source": meta.get("source"),
                    "category": meta.get("category"),
                    "preview": full_text,
                    "content": full_text,
                }
            )

        return sources

    def build_messages(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        history_records=None,
    ) -> List[Dict[str, str]]:
        context_text = self.build_context(chunks)
        history_text = self._history_to_text(history_records)

        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": self.config.system_prompt,
            }
        ]

        if history_text:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "아래는 이전 대화 기록이다.\n"
                        "이 기록은 사용자의 현재 질문 의도와 지시 대상을 이해하기 위한 맥락으로만 참고하라.\n"
                        "이전 대화에 나온 내용만으로 사실을 단정하지 말고, 사실 근거는 반드시 아래 검색 문맥에서 다시 확인하라.\n"
                        "이전 대화와 검색 문맥이 충돌하면 검색 문맥을 우선하라.\n\n"
                        f"{history_text}"
                    ),
                }
            )

        messages.append(
            {
                "role": "system",
                "content": (
                    "아래는 검색된 문서 문맥이다.\n"
                    "답변의 사실 근거는 반드시 이 문맥을 우선 사용하라.\n\n"
                    f"{context_text}"
                ),
            }
        )

        messages.append(
            {
                "role": "user",
                "content": query,
            }
        )

        return messages

    def _chat_once(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        think: Optional[bool] = None,
        temperature: Optional[float] = None,
        timeout: int = 300,
    ) -> Dict[str, Any]:
        payload = {
            "model": model or self.config.model,
            "messages": messages,
            "stream": False,
            "think": self.config.think if think is None else think,
            "options": {
                "temperature": self.config.temperature if temperature is None else temperature,
            },
        }

        r = requests.post(
            f"{self.config.ollama_host}/api/chat",
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()

    def _chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        think: Optional[bool] = None,
        temperature: Optional[float] = None,
        timeout_connect: int = 10,
    ) -> Generator[Dict[str, Any], None, None]:
        payload = {
            "model": model or self.config.model,
            "messages": messages,
            "stream": True,
            "think": self.config.think if think is None else think,
            "options": {
                "temperature": self.config.temperature if temperature is None else temperature,
            },
        }

        try:
            with requests.post(
                f"{self.config.ollama_host}/api/chat",
                json=payload,
                stream=True,
                timeout=(timeout_connect, None),
            ) as r:
                r.raise_for_status()

                for raw_line in r.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue

                    try:
                        obj = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    thinking = ""
                    content = ""

                    if isinstance(obj, dict):
                        if obj.get("thinking"):
                            thinking = obj.get("thinking") or ""

                        msg = obj.get("message") or {}
                        if isinstance(msg, dict):
                            if not thinking and msg.get("thinking"):
                                thinking = msg.get("thinking") or ""
                            if msg.get("content"):
                                content = msg.get("content") or ""

                        if thinking:
                            yield {"type": "thinking", "delta": thinking}

                        if content:
                            yield {"type": "delta", "delta": content}

                        if obj.get("done") is True:
                            yield {"type": "done"}
                            break

        except Exception as e:
            yield {"type": "error", "error": str(e)}

    def answer(
        self,
        query: str,
        history_records=None,
        collection_name: Optional[str] = None,
        k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        think: Optional[bool] = None,
        temperature: Optional[float] = None,
    ) -> RAGResult:
        target_collection = collection_name or self.config.collection_name
        resolved_collection_name = resolve_alias(target_collection)

        retrieved_chunks, final_chunks, debug_info = self.search(
            query=query,
            collection_name=target_collection,
            retrieval_k=self.config.retrieval_k,
            final_k=k or self.config.rerank_top_k,
            where=where,
        )

        context = self.build_context(final_chunks)
        messages = self.build_messages(
            query=query,
            chunks=final_chunks,
            history_records=history_records,
        )

        resp = self._chat_once(
            messages=messages,
            model=model,
            think=think,
            temperature=temperature,
        )

        answer_text = ""
        msg = resp.get("message") or {}
        if isinstance(msg, dict):
            answer_text = msg.get("content") or ""

        return RAGResult(
            answer=answer_text,
            collection_name=target_collection,
            resolved_collection_name=resolved_collection_name,
            query=query,
            context=context,
            sources=self.build_sources(final_chunks),
            raw_retrievals=retrieved_chunks,
            debug_info=debug_info,
        )

    def answer_stream(
        self,
        query: str,
        history_records=None,
        collection_name: Optional[str] = None,
        k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        think: Optional[bool] = None,
        temperature: Optional[float] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        target_collection = collection_name or self.config.collection_name
        resolved_collection_name = resolve_alias(target_collection)

        retrieved_chunks, final_chunks, debug_info = self.search(
            query=query,
            collection_name=target_collection,
            retrieval_k=self.config.retrieval_k,
            final_k=k or self.config.rerank_top_k,
            where=where,
        )

        context = self.build_context(final_chunks)
        sources = self.build_sources(final_chunks)
        messages = self.build_messages(
            query=query,
            chunks=final_chunks,
            history_records=history_records,
        )

        yield {
            "type": "meta",
            "query": query,
            "collection_name": target_collection,
            "resolved_collection_name": resolved_collection_name,
            "model": model or self.config.model,
            "retrieval_k": self.config.retrieval_k,
            "rerank_top_k": k or self.config.rerank_top_k,
            "retrieved_count": len(retrieved_chunks),
            "final_count": len(final_chunks),
            "debug_info": debug_info,
        }

        yield {
            "type": "sources",
            "sources": sources,
        }

        yield {
            "type": "context",
            "context": context,
        }

        for event in self._chat_stream(
            messages=messages,
            model=model,
            think=think,
            temperature=temperature,
        ):
            yield event


_default_rag: Optional[RAGService] = None


def get_rag() -> RAGService:
    global _default_rag
    if _default_rag is None:
        _default_rag = RAGService()
    return _default_rag


def answer_query(
    query: str,
    history_records=None,
    collection_name: Optional[str] = None,
    k: Optional[int] = None,
    where: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    think: Optional[bool] = None,
    temperature: Optional[float] = None,
) -> RAGResult:
    return get_rag().answer(
        query=query,
        history_records=history_records,
        collection_name=collection_name,
        k=k,
        where=where,
        model=model,
        think=think,
        temperature=temperature,
    )


def answer_query_stream(
    query: str,
    history_records=None,
    collection_name: Optional[str] = None,
    k: Optional[int] = None,
    where: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    think: Optional[bool] = None,
    temperature: Optional[float] = None,
):
    yield from get_rag().answer_stream(
        query=query,
        history_records=history_records,
        collection_name=collection_name,
        k=k,
        where=where,
        model=model,
        think=think,
        temperature=temperature,
    )