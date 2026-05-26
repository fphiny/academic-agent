from __future__ import annotations

import os
from typing import Any, Dict, List

from modules.rag.service import RAGService

from .tools.base import ToolResult, debug_print
from .tools import (
    external_build_context,
    external_extract_main_content,
    external_fetch_raw,
    external_search,
    internal_multi_search,
    internal_search,
    list_collections,
    send_mail,
    get_menu,
)


class AgentTools:
    def __init__(self):
        self.rag = RAGService()

        self.google_api_key = os.getenv(
            "GOOGLE_API_KEY",
            "AIzaSyBxLH71mS_gDH6jZwr2jipISeHFLH8A1u0",
        )
        self.google_cx = os.getenv(
            "GOOGLE_CX",
            "a7ed23662474841ed",
        )

        self.naver_client_id = os.getenv(
            "NAVER_CLIENT_ID",
            "2ISGnq7d9_vhu_dHkc0M",
        )
        self.naver_client_secret = os.getenv(
            "NAVER_CLIENT_SECRET",
            "DbxEZBkQ_7",
        )

        self.debug = True

    def _debug_print(self, label: str, text: Any, max_len: int = 5000) -> None:
        debug_print(self.debug, label, text, max_len=max_len)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "list_collections",
                "description": "사용 가능한 내부 컬렉션 목록과 각 컬렉션의 메타데이터(domain, description 등)를 반환",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "internal_search",
                "description": "지정한 내부 컬렉션 목록 중 첫 번째 컬렉션에서 RAG 검색",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "collections": {
                            "oneOf": [
                                {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                {
                                    "type": "string",
                                },
                            ]
                        },
                        "k": {"type": "integer"},
                    },
                    "required": ["query", "collections"],
                },
            },
            {
                "name": "internal_multi_search",
                "description": "여러 내부 컬렉션에서 동시에 검색",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "collections": {
                            "oneOf": [
                                {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                {
                                    "type": "string",
                                },
                            ]
                        },
                        "k": {"type": "integer"},
                    },
                    "required": ["query", "collections"],
                },
            },
            {
                "name": "external_search",
                "description": (
                    "외부 웹 검색만 수행한다. "
                    "검색 결과 metadata(title, link, displayLink, source_engine)와 "
                    "fetch 후보 selected_items/selected_urls를 반환한다."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "num": {"type": "integer"},
                        "top_k_urls": {
                            "type": "integer",
                            "description": "fetch 후보로 고를 검색 결과 수",
                        },
                        "top_k_chunks": {
                            "type": "integer",
                            "description": "하위호환용 인자. 현재 external_search 내부에서는 사용하지 않음",
                        },
                        "engines": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "예: ['naver'], ['google'], ['google', 'naver']",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "external_fetch_raw",
                "description": (
                    "외부 URL 또는 검색 결과 items를 fetch해서 raw HTML/TEXT/PDF를 반환한다. "
                    "검색은 하지 않는다."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "원본 사용자 질의",
                        },
                        "url": {
                            "type": "string",
                            "description": "직접 fetch할 단일 URL",
                        },
                        "urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "직접 fetch할 URL 목록",
                        },
                        "items": {
                            "type": "array",
                            "description": "external_search 결과 item 목록",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "link": {"type": "string"},
                                    "snippet": {"type": "string"},
                                    "displayLink": {"type": "string"},
                                    "source_engine": {"type": "string"},
                                },
                            },
                        },
                        "max_fetch": {
                            "type": "integer",
                            "description": "최대 fetch 개수",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "HTTP timeout seconds",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "external_extract_main_content",
                "description": (
                    "external_fetch_raw가 반환한 raw documents를 바탕으로 "
                    "HTML 본문 영역을 추출하고 markdown/block 형태로 정제한다."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "documents": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "external_fetch_raw.data['documents']",
                        },
                    },
                    "required": ["documents"],
                },
            },
            {
                "name": "external_build_context",
                "description": (
                    "external_extract_main_content가 반환한 documents를 바탕으로 "
                    "관련 block 선택, chunking, embedding retrieval을 수행하고 "
                    "최종 context와 sources를 반환한다."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "documents": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "external_extract_main_content.data['documents']",
                        },
                        "top_k_chunks": {
                            "type": "integer",
                            "description": "최종 유사도 상위 chunk 수",
                        },
                    },
                    "required": ["query", "documents"],
                },
            },
            {
                "name": "get_menu",
                "description": "지정한 날짜의 메뉴를 조회한다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "조회 날짜. YYYYMMDD, YYYY-MM-DD, today, tomorrow, 오늘, 내일 등",
                        },
                    },
                    "required": ["date"],
                },
            },
            {
                "name": "send_mail",
                "description": "네이버 SMTP를 사용해 이메일을 전송한다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            ]
                        },
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                        "cc": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            ]
                        },
                        "bcc": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            ]
                        },
                        "html_body": {"type": "string"},
                    },
                    "required": ["to", "subject", "body"],
                },
            },
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        self._debug_print(
            "CALL_TOOL INPUT",
            {
                "name": name,
                "arguments": arguments,
            },
        )

        if name == "list_collections":
            return list_collections.run(
                rag=self.rag,
                debug_print=self._debug_print,
            )

        if name == "internal_search":
            return internal_search.run(
                rag=self.rag,
                query=arguments.get("query", ""),
                collections=arguments.get("collections", []),
                k=int(arguments.get("k", 5)),
            )

        if name == "internal_multi_search":
            return internal_multi_search.run(
                rag=self.rag,
                query=arguments.get("query", ""),
                collections=arguments.get("collections", []),
                k=int(arguments.get("k", 5)),
            )

        if name == "external_search":
            return external_search.run(
                query=arguments.get("query", ""),
                num=int(arguments.get("num", 5)),
                top_k_urls=int(arguments.get("top_k_urls", 5)),
                top_k_chunks=int(arguments.get("top_k_chunks", 5)),
                engines=arguments.get("engines", ["google"]),
                google_api_key=self.google_api_key,
                google_cx=self.google_cx,
                naver_client_id=self.naver_client_id,
                naver_client_secret=self.naver_client_secret,
                debug_print=self._debug_print,
            )

        if name == "external_fetch_raw":
            url = arguments.get("url", "")
            urls = arguments.get("urls")
            items = arguments.get("items")

            if isinstance(url, str) and url.startswith("[") and url.endswith("]") and not urls:
                try:
                    import json
                    parsed = json.loads(url)
                    if isinstance(parsed, list):
                        urls = parsed
                        url = ""
                except Exception:
                    pass

            return external_fetch_raw.run(
                query=arguments.get("query", ""),
                url=url,
                urls=urls,
                items=items,
                max_fetch=int(arguments.get("max_fetch", 3)),
                timeout=int(arguments.get("timeout", 20)),
                debug_print=self._debug_print,
            )

        if name == "external_extract_main_content":
            return external_extract_main_content.run(
                query=arguments.get("query", ""),
                documents=arguments.get("documents"),
                debug_print=self._debug_print,
            )

        if name == "external_build_context":
            return external_build_context.run(
                query=arguments.get("query", ""),
                documents=arguments.get("documents"),
                top_k_chunks=int(arguments.get("top_k_chunks", 5)),
                debug_print=self._debug_print,
            )

        if name == "get_menu":
            return get_menu.run(
                date=arguments.get("date", ""),
                debug_print=self._debug_print,
            )

        if name == "send_mail":
            return send_mail.run(
                to=arguments.get("to", ""),
                subject=arguments.get("subject", ""),
                body=arguments.get("body", ""),
                cc=arguments.get("cc"),
                bcc=arguments.get("bcc"),
                html_body=arguments.get("html_body"),
            )

        return ToolResult(
            name=name,
            ok=False,
            data={"error": f"unknown tool: {name}"},
        )