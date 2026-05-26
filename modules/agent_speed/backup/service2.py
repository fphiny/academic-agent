from __future__ import annotations

import json
import re
from collections import deque
from typing import Any, Deque, Dict, Generator, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urldefrag

from langchain_core.messages import HumanMessage, SystemMessage

from core.ollama.client import OllamaClient

from modules.chroma.alias_store import resolve_alias
from modules.chroma.store import get_store

from .config import AgentConfig
from .execution import execute_tool, extract_sources_from_result
from .tool import AgentTools


class AgentSpeedService:
    """
    개선 버전 + 보정 버전

    핵심 보정
    - 원본 사용자 질문(original_user_message)을 끝까지 보존
    - 검색 최적화용 질의(atomic / rewritten)는 retrieval 전용으로만 사용
    - external_build_context 결과를 후처리해
      같은 문서/같은 헤더/인접 청크를 최대한 이어 붙임
    - top_k_chunks 를 너무 작게 잡지 않음
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.tools = AgentTools()
        self.store = get_store(alias_resolver=resolve_alias)
        self.ollama = OllamaClient(
            host=self.config.ollama_host,
            default_model=self.config.model,
            timeout=self.config.request_timeout,
        )

    # ---------------------------------------------------------------------
    # basic utils
    # ---------------------------------------------------------------------

    def _get_available_collections(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        try:
            names = self.store.list_collections()
            for name in names:
                try:
                    c = self.store.get_collection(name)
                    items.append(
                        {
                            "name": c.name,
                            "metadata": getattr(c, "metadata", None) or {},
                        }
                    )
                except Exception:
                    items.append({"name": name, "metadata": {}})
        except Exception:
            return []
        return items

    def _normalize_stream_chunk_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        try:
            return self.ollama.normalize_text_content(content)
        except Exception:
            return str(content)

    def _safe_json_loads(self, text: str) -> Optional[Dict[str, Any]]:
        raw = (text or "").strip()
        if not raw:
            return None

        try:
            return json.loads(raw)
        except Exception:
            pass

        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            candidate = raw[start:end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return None
        return None

    def _normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def _truncate_text(self, text: Any, limit: int = 4000) -> str:
        value = str(text or "").strip()
        if len(value) <= limit:
            return value
        return value[:limit]

    def _canonicalize_url(self, url: str) -> str:
        raw = str(url or "").strip()
        if not raw:
            return ""
        if raw.startswith(("mailto:", "javascript:", "tel:")):
            return ""
        raw, _ = urldefrag(raw)
        parsed = urlparse(raw)
        if not parsed.scheme or not parsed.netloc:
            return raw
        normalized = parsed._replace(fragment="", params="")
        final = normalized.geturl()
        root = f"{parsed.scheme}://{parsed.netloc}/"
        if final.endswith("/") and final != root:
            final = final.rstrip("/")
        return final

    def _is_same_site(self, base_url: str, other_url: str) -> bool:
        try:
            a = urlparse(str(base_url or "").strip())
            b = urlparse(str(other_url or "").strip())
            return bool(a.netloc) and a.netloc == b.netloc
        except Exception:
            return False

    def _dedupe_texts(self, values: List[str]) -> List[str]:
        seen = set()
        result: List[str] = []
        for value in values:
            clean = self._normalize_whitespace(value)
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(clean)
        return result

    def _dedupe_sources(self, sources: Any) -> List[Dict[str, Any]]:
        if not isinstance(sources, list):
            return []

        result: List[Dict[str, Any]] = []
        seen = set()

        for src in sources:
            if not isinstance(src, dict):
                continue

            title = self._normalize_whitespace(src.get("title") or "")
            url = self._normalize_whitespace(src.get("url") or "")
            snippet = self._normalize_whitespace(src.get("snippet") or "")
            content = self._normalize_whitespace(src.get("content") or "")

            key = (title.lower(), url.lower(), (content or snippet)[:300].lower())
            if key in seen:
                continue
            seen.add(key)
            result.append(dict(src))

        return result

    def _extract_source_identity(self, src: Dict[str, Any]) -> Tuple[str, str]:
        title = self._normalize_whitespace(src.get("title") or "")
        url = self._normalize_whitespace(src.get("url") or "")
        return (title.lower(), url.lower())

    def _extract_header_key(self, src: Dict[str, Any]) -> str:
        """
        같은 헤더 반복 분할 문제를 완화하기 위한 느슨한 헤더 식별자.
        source metadata 에 header/section 계열 정보가 있으면 우선 사용하고,
        없으면 content/snippet 첫 줄을 약식 헤더처럼 사용.
        """
        for key in ["header", "section", "section_title", "heading", "parent_header"]:
            value = self._normalize_whitespace(src.get(key) or "")
            if value:
                return value.lower()

        content = str(src.get("content") or src.get("snippet") or "").strip()
        if not content:
            return ""

        first_line = content.splitlines()[0].strip()
        first_line = re.sub(r"\s+", " ", first_line)
        return first_line[:120].lower()

    def _extract_chunk_order(self, src: Dict[str, Any]) -> Optional[int]:
        for key in ["chunk_index", "order", "position", "seq", "chunk_id"]:
            value = src.get(key)
            try:
                return int(value)
            except Exception:
                continue
        return None

    def _merge_text_blocks(self, left: str, right: str) -> str:
        left = str(left or "").strip()
        right = str(right or "").strip()

        if not left:
            return right
        if not right:
            return left

        # 동일 prefix 중복 제거
        if right in left:
            return left
        if left in right:
            return right

        # 첫 줄 헤더가 완전히 같은 경우, 오른쪽 첫 줄 제거 후 이어붙임
        left_lines = left.splitlines()
        right_lines = right.splitlines()

        if left_lines and right_lines:
            l0 = self._normalize_whitespace(left_lines[0]).lower()
            r0 = self._normalize_whitespace(right_lines[0]).lower()
            if l0 and l0 == r0 and len(right_lines) > 1:
                right = "\n".join(right_lines[1:]).strip()

        if not right:
            return left

        return f"{left}\n{right}".strip()

    def _stitch_adjacent_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        같은 문서 + 같은 헤더 + 인접 chunk 로 보이는 source는 content 를 합친다.
        실제 split 로직을 못 건드리는 상황에서 orchestration 레벨에서 할 수 있는 최소 보정.
        """
        if not isinstance(sources, list) or not sources:
            return []

        enriched: List[Tuple[Tuple[str, str], str, Optional[int], Dict[str, Any]]] = []
        for src in sources:
            if not isinstance(src, dict):
                continue
            doc_key = self._extract_source_identity(src)
            header_key = self._extract_header_key(src)
            order = self._extract_chunk_order(src)
            enriched.append((doc_key, header_key, order, dict(src)))

        # 정렬: 문서 -> 헤더 -> 순서
        enriched.sort(key=lambda x: (x[0][0], x[0][1], x[1], 10**9 if x[2] is None else x[2]))

        stitched: List[Dict[str, Any]] = []

        for doc_key, header_key, order, src in enriched:
            text = str(src.get("content") or src.get("snippet") or "").strip()
            if not stitched:
                stitched.append(src)
                continue

            prev = stitched[-1]
            prev_doc_key = self._extract_source_identity(prev)
            prev_header_key = self._extract_header_key(prev)
            prev_order = self._extract_chunk_order(prev)

            same_doc = (doc_key == prev_doc_key)
            same_header = bool(header_key) and header_key == prev_header_key
            adjacent = (
                prev_order is not None and order is not None and order - prev_order <= 1
            )

            # 헤더/문서가 같고 chunk 순서도 붙어 있으면 병합
            if same_doc and (same_header or adjacent):
                prev_text = str(prev.get("content") or prev.get("snippet") or "").strip()
                merged_text = self._merge_text_blocks(prev_text, text)

                if prev.get("content"):
                    prev["content"] = merged_text
                else:
                    prev["snippet"] = merged_text

                # order 는 가장 앞 chunk 기준 유지
                continue

            stitched.append(src)

        return stitched

    def _extract_document_links(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(document, dict):
            return []

        raw_links = document.get("links")
        base_url = self._normalize_whitespace(document.get("url") or document.get("link") or "")

        result: List[Dict[str, Any]] = []
        seen = set()

        if isinstance(raw_links, list):
            for link in raw_links:
                if not isinstance(link, dict):
                    continue
                href = self._normalize_whitespace(link.get("url") or link.get("link") or "")
                if not href:
                    continue
                abs_url = self._canonicalize_url(urljoin(base_url, href)) if base_url else self._canonicalize_url(href)
                if not abs_url or abs_url in seen:
                    continue
                seen.add(abs_url)
                result.append(
                    {
                        "link": abs_url,
                        "title": self._normalize_whitespace(link.get("title") or link.get("anchor_text") or ""),
                        "snippet": self._normalize_whitespace(link.get("snippet") or link.get("context") or ""),
                    }
                )

        if result:
            return result

        blob = "\n".join(
            [
                str(document.get("content") or ""),
                str(document.get("raw_html") or ""),
                str(document.get("html") or ""),
                str(document.get("markdown") or ""),
            ]
        )

        if not blob:
            return []

        for href, anchor in re.findall(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', blob, flags=re.I | re.S):
            abs_url = self._canonicalize_url(urljoin(base_url, href)) if base_url else self._canonicalize_url(href)
            if not abs_url or abs_url in seen:
                continue
            seen.add(abs_url)
            anchor_text = self._normalize_whitespace(re.sub(r"<[^>]+>", " ", anchor))
            result.append({"link": abs_url, "title": anchor_text, "snippet": ""})

        for anchor, href in re.findall(r'\[([^\]]{0,200})\]\((https?://[^)\s]+|/[^)\s]+)\)', blob):
            abs_url = self._canonicalize_url(urljoin(base_url, href)) if base_url else self._canonicalize_url(href)
            if not abs_url or abs_url in seen:
                continue
            seen.add(abs_url)
            result.append({"link": abs_url, "title": self._normalize_whitespace(anchor), "snippet": ""})

        return result

    def _normalize_extracted_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(document, dict):
            return {}

        normalized = dict(document)
        url = self._canonicalize_url(document.get("url") or document.get("link") or "")
        if url:
            normalized["url"] = url
            normalized["link"] = url

        if "title" not in normalized:
            normalized["title"] = ""
        if "content" not in normalized:
            normalized["content"] = str(document.get("snippet") or "")

        normalized["links"] = self._extract_document_links(normalized)
        return normalized

    def _merge_documents_by_link(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen = set()

        for doc in documents:
            if not isinstance(doc, dict):
                continue
            normalized = self._normalize_extracted_document(doc)
            url = self._normalize_whitespace(normalized.get("url") or normalized.get("link") or "")
            if not url or url in seen:
                continue
            seen.add(url)
            merged.append(normalized)

        return merged

    def _fetch_and_extract_single_item(
        self,
        *,
        user_message: str,
        item: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None

        fetch_result = execute_tool(
            tools=self.tools,
            tool_name="external_fetch_raw",
            arguments={
                "query": user_message,
                "items": [item],
                "max_fetch": 1,
            },
            step=4,
            messages=[],
            user_message=user_message,
        )
        if not getattr(fetch_result, "ok", False) or not isinstance(fetch_result.data, dict):
            return None

        raw_documents = fetch_result.data.get("documents")
        if not isinstance(raw_documents, list) or not raw_documents:
            return None

        extract_result = execute_tool(
            tools=self.tools,
            tool_name="external_extract_main_content",
            arguments={
                "query": user_message,
                "documents": raw_documents,
            },
            step=4,
            messages=[],
            user_message=user_message,
        )
        if not getattr(extract_result, "ok", False) or not isinstance(extract_result.data, dict):
            return None

        extracted_documents = extract_result.data.get("documents")
        if not isinstance(extracted_documents, list) or not extracted_documents:
            return None

        return self._normalize_extracted_document(extracted_documents[0])

    def _judge_page_answerability_and_next_action(
        self,
        *,
        user_message: str,
        document: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(document, dict):
            return {
                "can_answer_now": False,
                "need_more_navigation": True,
                "confidence": "low",
                "reason": "document 가 dict 가 아님",
            }

        content = self._truncate_text(document.get("content") or document.get("snippet") or "", 5000)
        links = self._extract_document_links(document)
        link_lines: List[str] = []
        for idx, link in enumerate(links[:30], start=1):
            link_lines.append(
                f"{idx}. title={self._normalize_whitespace(link.get('title') or '')} url={self._normalize_whitespace(link.get('link') or '')} snippet={self._truncate_text(link.get('snippet') or '', 120)}"
            )

        system_prompt = (
            "너는 웹 페이지 기반 RAG 탐색 판단기다.\n"
            "현재 페이지 본문만으로 질문에 충분히 답할 수 있는지 판단하고,\n"
            "부족하면 추가 페이지 탐색이 필요한지 판단하라.\n"
            "반드시 JSON만 출력한다.\n"
            '\n형식: {"can_answer_now":true,"need_more_navigation":false,"confidence":"high","reason":"짧은 설명"}'
        )

        user_prompt = (
            f"[사용자 질문]\n{user_message}\n\n"
            f"[현재 페이지 URL]\n{self._normalize_whitespace(document.get('url') or document.get('link') or '')}\n\n"
            f"[현재 페이지 제목]\n{self._normalize_whitespace(document.get('title') or '')}\n\n"
            f"[현재 페이지 본문]\n{content or '(empty)'}\n\n"
            "[페이지 내 링크 후보]\n" + ("\n".join(link_lines) if link_lines else "(no links)")
        )

        try:
            llm = self.ollama.build_chat_llm(model=self.config.model, think=False)
            resp = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            raw = self._normalize_stream_chunk_content(getattr(resp, "content", ""))
            data = self._safe_json_loads(raw) or {}

            can_answer_now = bool(data.get("can_answer_now"))
            need_more_navigation = bool(data.get("need_more_navigation"))
            confidence = str(data.get("confidence") or "low").strip().lower()
            reason = str(data.get("reason") or "").strip()

            if confidence not in {"high", "medium", "low"}:
                confidence = "low"

            if can_answer_now and confidence == "high":
                need_more_navigation = False

            if not can_answer_now and not need_more_navigation:
                need_more_navigation = True

            return {
                "can_answer_now": can_answer_now,
                "need_more_navigation": need_more_navigation,
                "confidence": confidence,
                "reason": reason or "page judge 결과",
            }
        except Exception:
            return {
                "can_answer_now": False,
                "need_more_navigation": True,
                "confidence": "low",
                "reason": "page judge 호출 실패",
            }

    def _select_next_links_with_llm(
        self,
        *,
        user_message: str,
        current_document: Dict[str, Any],
        candidate_links: List[Dict[str, Any]],
        max_links: int = 5,
    ) -> List[Dict[str, Any]]:
        candidates = self._dedupe_external_candidates(candidate_links)
        if not candidates:
            return []

        safe_cap = max(1, min(int(max_links or 1), len(candidates)))
        numbered_lines: List[str] = []
        for idx, c in enumerate(candidates, start=1):
            numbered_lines.append(
                f"[{idx}]\n"
                f"title: {c['title']}\n"
                f"url: {c['link']}\n"
                f"snippet: {c['snippet']}\n"
            )

        system_prompt = (
            "너는 현재 페이지에서 다음에 방문할 링크를 고르는 웹 탐색 선택기다.\n"
            "사람이 만든 키워드 점수표는 사용하지 말고, 질문 해결 가능성이 높은 링크만 고른다.\n"
            "반드시 JSON만 출력한다.\n"
            '- 형식: {"selected_indices":[1,2],"reason":"짧은 설명"}'
        )

        current_url = self._normalize_whitespace(current_document.get("url") or current_document.get("link") or "")
        current_title = self._normalize_whitespace(current_document.get("title") or "")
        current_content = self._truncate_text(current_document.get("content") or current_document.get("snippet") or "", 2500)

        user_prompt = (
            f"[질문]\n{user_message}\n\n"
            f"[현재 페이지 URL]\n{current_url}\n\n"
            f"[현재 페이지 제목]\n{current_title}\n\n"
            f"[현재 페이지 본문 요약]\n{current_content}\n\n"
            f"[선택 상한]\n최대 {safe_cap}개\n\n"
            "[후보 링크 목록]\n" + "\n".join(numbered_lines)
        )

        try:
            llm = self.ollama.build_chat_llm(model=self.config.model, think=False)
            resp = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            raw = self._normalize_stream_chunk_content(getattr(resp, "content", ""))
            data = self._safe_json_loads(raw) or {}
            indices = data.get("selected_indices") or []
            if not isinstance(indices, list):
                indices = []

            selected: List[Dict[str, Any]] = []
            seen = set()

            for x in indices:
                try:
                    idx = int(x)
                except Exception:
                    continue

                if idx < 1 or idx > len(candidates):
                    continue

                item = dict(candidates[idx - 1])
                url = item["link"]
                if url in seen:
                    continue
                if current_url and not self._is_same_site(current_url, url):
                    continue

                seen.add(url)
                selected.append(item)
                if len(selected) >= safe_cap:
                    break

            if selected:
                return selected
        except Exception:
            pass

        fallback: List[Dict[str, Any]] = []
        for item in candidates:
            url = item.get("link") or ""
            if current_url and not self._is_same_site(current_url, url):
                continue
            fallback.append(item)
            if len(fallback) >= safe_cap:
                break
        return fallback

    def _ai_guided_bfs_expand(
        self,
        *,
        user_message: str,
        seed_documents: List[Dict[str, Any]],
        max_depth: int = 2,
        max_total_pages: int = 12,
        max_links_per_page: int = 4,
    ) -> List[Dict[str, Any]]:
        documents = self._merge_documents_by_link(seed_documents)
        if not documents:
            return []

        visited: Set[str] = set()
        queue: Deque[Tuple[int, Dict[str, Any]]] = deque()
        expanded: List[Dict[str, Any]] = []

        for doc in documents:
            url = self._normalize_whitespace(doc.get("url") or doc.get("link") or "")
            if url:
                visited.add(url)
            queue.append((0, doc))

        while queue and len(visited) < max_total_pages:
            depth, current_document = queue.popleft()
            if depth > max_depth:
                continue

            judge = self._judge_page_answerability_and_next_action(
                user_message=user_message,
                document=current_document,
            )

            if judge.get("can_answer_now") and not judge.get("need_more_navigation"):
                break

            if depth >= max_depth:
                continue

            candidate_links = self._extract_document_links(current_document)
            if not candidate_links:
                continue

            next_links = self._select_next_links_with_llm(
                user_message=user_message,
                current_document=current_document,
                candidate_links=candidate_links,
                max_links=max_links_per_page,
            )

            for link_item in next_links:
                next_url = self._canonicalize_url(link_item.get("link") or "")
                current_url = self._normalize_whitespace(current_document.get("url") or current_document.get("link") or "")

                if not next_url or next_url in visited:
                    continue
                if current_url and not self._is_same_site(current_url, next_url):
                    continue
                if len(visited) >= max_total_pages:
                    break

                visited.add(next_url)
                fetched_document = self._fetch_and_extract_single_item(
                    user_message=user_message,
                    item=link_item,
                )
                if not fetched_document:
                    continue

                expanded.append(fetched_document)
                queue.append((depth + 1, fetched_document))

        return self._merge_documents_by_link(expanded)

    # ---------------------------------------------------------------------
    # collection select
    # ---------------------------------------------------------------------

    def _select_collections_for_query(
        self,
        user_message: str,
        collections: List[Dict[str, Any]],
        fallback_collection: str,
        max_collections: int = 3,
    ) -> List[str]:
        raw_user = (user_message or "").strip()
        safe_fallback = (fallback_collection or "").strip()
        safe_cap = max(1, int(max_collections or 1))

        if not raw_user:
            return [safe_fallback] if safe_fallback else []

        if not collections:
            return [safe_fallback] if safe_fallback else []

        normalized_collections: List[Dict[str, str]] = []
        valid_names: List[str] = []

        for item in collections:
            if not isinstance(item, dict):
                continue

            name = str(item.get("name") or "").strip()
            metadata = item.get("metadata") or {}
            domain = str(metadata.get("domain") or "").strip()
            description = str(metadata.get("description") or "").strip()

            if not name:
                continue

            normalized_collections.append(
                {
                    "name": name,
                    "domain": domain,
                    "description": description,
                }
            )
            valid_names.append(name)

        if not normalized_collections:
            return [safe_fallback] if safe_fallback else []

        collection_lines: List[str] = []
        for idx, c in enumerate(normalized_collections, start=1):
            collection_lines.append(
                f"[{idx}]\n"
                f"name: {c['name']}\n"
                f"domain: {c['domain']}\n"
                f"description: {c['description']}\n"
            )

        system_prompt = (
            "너는 사용자 질문에 맞는 internal collection selector다.\n"
            "주어진 collection 목록 중에서 사용자 질문에 답하는 데 가장 적합한 collection만 고른다.\n"
            "\n"
            "[중요 원칙]\n"
            "1) 문자열 완전일치가 아니라 의미 기반으로 판단한다.\n"
            "2) 질문 속 표현을 collection description/domain의 일반화된 설명과 연결해서 해석한다.\n"
            "3) 위치/찾아가는 길/어디/몇 층/몇 호/호수 안내 맥락이면 위치 관련 컬렉션을 우선한다.\n"
            "4) 질문과 관련 없는 컬렉션은 고르지 않는다.\n"
            "5) 확신이 낮으면 적게 고른다.\n"
            "\n"
            "[출력 규칙]\n"
            "- 반드시 JSON만 출력한다.\n"
            "- 형식: {\"selected_names\":[\"collection1\",\"collection2\"],\"reason\":\"짧은 설명\"}\n"
            "- selected_names는 collection name 문자열 배열\n"
            "- 목록에 없는 이름 생성 금지\n"
            "- JSON 외 설명 금지"
        )

        user_prompt = (
            f"[사용자 질문]\n{raw_user}\n\n"
            f"[최대 선택 개수]\n{safe_cap}\n\n"
            "[collection 목록]\n"
            + "\n".join(collection_lines)
        )

        try:
            llm = self.ollama.build_chat_llm(model=self.config.model, think=False)
            resp = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )

            raw = self._normalize_stream_chunk_content(getattr(resp, "content", ""))
            data = self._safe_json_loads(raw) or {}
            selected_names = data.get("selected_names") or []

            if not isinstance(selected_names, list):
                selected_names = []

            valid_name_set = set(valid_names)
            result: List[str] = []
            seen = set()

            for name in selected_names:
                clean = str(name or "").strip()
                if not clean or clean not in valid_name_set or clean in seen:
                    continue
                seen.add(clean)
                result.append(clean)
                if len(result) >= safe_cap:
                    break

            if result:
                if safe_fallback and safe_fallback not in seen:
                    result.append(safe_fallback)
                return result
        except Exception:
            pass

        return [safe_fallback] if safe_fallback else []

    # ---------------------------------------------------------------------
    # atomic decomposition
    # ---------------------------------------------------------------------

    def _decompose_user_message_into_atomic_queries(
        self,
        *,
        user_message: str,
        max_atomic_queries: int = 4,
    ) -> List[str]:
        raw_user = self._normalize_whitespace(user_message)
        safe_cap = max(1, int(max_atomic_queries or 1))

        if not raw_user:
            return []

        system_prompt = (
            "너는 사용자 질문을 atomic query 들로 분해하는 질의 분해기다.\n"
            "질문에 두 개 이상의 대상(엔티티, 캠퍼스, 사람, 장소, 건물, 학과, 비교대상)이 있으면\n"
            "각 대상을 독립적으로 검색 가능한 완전한 검색 질의로 분해한다.\n"
            "\n"
            "[핵심 원칙]\n"
            "1) 공통 의도는 유지한다. 예: 위치, 홈페이지, 전화번호, 연구실, 프로필\n"
            "2) 각 atomic query 는 search/internal_search/build_context 에 바로 넣을 수 있게 완전한 문자열이어야 한다.\n"
            "3) 공통 접두어(기관명)는 각 query 에 반복 포함한다.\n"
            "4) 단일 대상 질문이면 queries 는 1개만 반환한다.\n"
            "5) 쓸데없이 잘게 쪼개지 마라.\n"
            "6) 비교/나열/병렬 구조(, / 및 / 와 / 그리고 / 랑)는 분해를 적극 고려한다.\n"
            "\n"
            "[예시]\n"
            "입력: 강원대학교 춘천캠퍼스, 원주캠퍼스 위치를 알려줘\n"
            "출력: {\"queries\":[\"강원대학교 춘천캠퍼스 위치\",\"강원대학교 원주캠퍼스 위치\"],\"reason\":\"캠퍼스 2개를 각각 검색\"}\n"
            "\n"
            "[출력 규칙]\n"
            "- 반드시 JSON만 출력\n"
            "- 형식: {\"queries\":[\"...\"],\"reason\":\"짧은 설명\"}\n"
            f"- queries 최대 {safe_cap}개\n"
            "- 중복 금지"
        )

        try:
            llm = self.ollama.build_chat_llm(model=self.config.model, think=False)
            resp = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=raw_user)]
            )
            raw = self._normalize_stream_chunk_content(getattr(resp, "content", ""))
            data = self._safe_json_loads(raw) or {}
            queries = data.get("queries") or []

            if isinstance(queries, list):
                cleaned = self._dedupe_texts([str(x) for x in queries])[:safe_cap]
                if cleaned:
                    return cleaned
        except Exception:
            pass

        split_markers = [",", "/", " 및 ", " 와 ", " 과 ", " 그리고 ", " 랑 "]
        has_parallel = any(marker in raw_user for marker in split_markers)
        if not has_parallel:
            return [raw_user]

        intent_word = ""
        for keyword in ["위치", "전화번호", "홈페이지", "연락처", "프로필", "연구실", "주소"]:
            if keyword in raw_user:
                intent_word = keyword
                break

        m = re.match(r"^(.*?)([^,\n]+(?:,\s*[^,\n]+)+)(.*)$", raw_user)
        if m:
            left = self._normalize_whitespace(m.group(1))
            middle = m.group(2)
            right = self._normalize_whitespace(m.group(3))

            parts = [
                self._normalize_whitespace(x)
                for x in middle.split(",")
                if self._normalize_whitespace(x)
            ]
            atomic: List[str] = []
            for part in parts[:safe_cap]:
                candidate = self._normalize_whitespace(f"{left} {part}")
                if intent_word and intent_word not in candidate:
                    candidate = self._normalize_whitespace(f"{candidate} {intent_word}")
                elif not intent_word and right:
                    candidate = self._normalize_whitespace(f"{candidate} {right}")
                atomic.append(candidate)

            atomic = self._dedupe_texts(atomic)
            if atomic:
                return atomic[:safe_cap]

        return [raw_user]

    # ---------------------------------------------------------------------
    # internal judge / merge
    # ---------------------------------------------------------------------

    def _has_meaningful_context(self, result_data: Any) -> bool:
        if not isinstance(result_data, dict):
            return False

        context = str(result_data.get("context") or "").strip()
        if len(context) >= 80:
            return True

        sources = result_data.get("sources")
        if isinstance(sources, list) and len(sources) >= 2:
            return True

        return False

    def _has_meaningful_internal_result(self, result_data: Any) -> bool:
        if not isinstance(result_data, dict):
            return False

        context = str(result_data.get("context") or "").strip()
        if len(context) >= 60:
            return True

        sources = result_data.get("sources")
        if isinstance(sources, list) and len(sources) >= 1:
            return True

        return False

    def _judge_internal_answerability(
        self,
        *,
        user_message: str,
        internal_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(internal_result, dict):
            return {
                "use_internal": False,
                "need_external": True,
                "confidence": "low",
                "reason": "internal_result 가 dict 가 아님",
            }

        context_text = str(internal_result.get("context") or "").strip()
        sources = internal_result.get("sources") or []

        source_lines: List[str] = []
        if isinstance(sources, list):
            for idx, src in enumerate(sources[:10], start=1):
                if not isinstance(src, dict):
                    continue
                title = str(src.get("title") or "").strip()
                snippet = str(src.get("snippet") or "").strip()
                content = str(src.get("content") or "").strip()
                text = content or snippet
                source_lines.append(f"{idx}. title={title} text={text[:400]}")

        if len(context_text) < 40 and not source_lines:
            return {
                "use_internal": False,
                "need_external": True,
                "confidence": "low",
                "reason": "internal context 가 거의 없음",
            }

        system_prompt = (
            "너는 RAG 중간판단기다.\n"
            "사용자 질문과 internal search 결과(context, sources)를 보고,\n"
            "이 internal 정보만으로 사용자 질문에 직접적이고 충분하게 답할 수 있는지 판단하라.\n"
            "\n"
            "[판단 기준]\n"
            "1) 질문의 핵심 슬롯이 실제로 있으면 use_internal=true 가능\n"
            "2) 질문과 관련은 있어도 핵심 답이 비어 있으면 need_external=true\n"
            "3) 멀티 엔티티 질문은 각 엔티티에 대한 정보가 충분히 있는지 같이 본다\n"
            "4) 최신성/검증/공식 확인이 필요하면 external 쪽으로 보수적으로 판단한다\n"
            "5) 추측 금지\n"
            "\n"
            "[출력 규칙]\n"
            "반드시 JSON만 출력한다.\n"
            '형식: {"use_internal":true,"need_external":false,"confidence":"high","reason":"짧은 설명"}'
        )

        user_prompt = (
            f"[사용자 질문]\n{user_message}\n\n"
            f"[internal context]\n{context_text or '(empty)'}\n\n"
            "[internal sources]\n"
            + ("\n".join(source_lines) if source_lines else "(no sources)")
        )

        try:
            llm = self.ollama.build_chat_llm(model=self.config.model, think=False)
            resp = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            raw = self._normalize_stream_chunk_content(getattr(resp, "content", ""))
            data = self._safe_json_loads(raw) or {}

            use_internal = bool(data.get("use_internal"))
            need_external = bool(data.get("need_external"))
            confidence = str(data.get("confidence") or "low").strip().lower()
            reason = str(data.get("reason") or "").strip()

            if confidence not in {"high", "medium", "low"}:
                confidence = "low"

            if use_internal and need_external:
                if confidence == "high":
                    need_external = False
                else:
                    use_internal = False
                    need_external = True

            if not use_internal and not need_external:
                need_external = True
                confidence = "low"

            return {
                "use_internal": use_internal,
                "need_external": need_external,
                "confidence": confidence,
                "reason": reason or "judge 결과",
            }
        except Exception:
            return {
                "use_internal": False,
                "need_external": True,
                "confidence": "low",
                "reason": "judge 호출 실패",
            }

    def _merge_result_payloads(
        self,
        *,
        query_to_result: List[Tuple[str, Dict[str, Any]]],
        prefer_longer_context: bool = False,
        stitch_sources: bool = False,
    ) -> Dict[str, Any]:
        merged_context_blocks: List[str] = []
        merged_sources: List[Dict[str, Any]] = []

        for atomic_query, result_data in query_to_result:
            if not isinstance(result_data, dict):
                continue

            context_text = self._normalize_whitespace(result_data.get("context") or "")
            if context_text:
                merged_context_blocks.append(
                    f"[atomic_query]\n{atomic_query}\n[context]\n{context_text}"
                )

            sources = result_data.get("sources") or []
            if isinstance(sources, list):
                for src in sources:
                    if not isinstance(src, dict):
                        continue
                    enriched = dict(src)
                    enriched["atomic_query"] = atomic_query
                    merged_sources.append(enriched)

        merged_sources = self._dedupe_sources(merged_sources)

        if stitch_sources:
            merged_sources = self._stitch_adjacent_sources(merged_sources)

        if prefer_longer_context:
            merged_context_blocks = sorted(merged_context_blocks, key=len, reverse=True)

        # sources 를 stitch 했으면 context 도 다시 합성해서 더 길게 준다.
        if stitch_sources and merged_sources:
            stitched_blocks: List[str] = []
            for src in merged_sources:
                atomic_query = str(src.get("atomic_query") or "").strip()
                content = str(src.get("content") or src.get("snippet") or "").strip()
                if not content:
                    continue
                if atomic_query:
                    stitched_blocks.append(
                        f"[atomic_query]\n{atomic_query}\n[context]\n{content}"
                    )
                else:
                    stitched_blocks.append(content)

            if stitched_blocks:
                merged_context_blocks = stitched_blocks

        return {
            "context": "\n\n".join(merged_context_blocks).strip(),
            "sources": merged_sources,
        }

    # ---------------------------------------------------------------------
    # external search helpers
    # ---------------------------------------------------------------------

    def _has_meaningful_search_items(self, result_data: Any) -> bool:
        if not isinstance(result_data, dict):
            return False

        selected_items = result_data.get("selected_items")
        if isinstance(selected_items, list) and len(selected_items) > 0:
            return True

        items = result_data.get("items")
        if isinstance(items, list) and len(items) > 0:
            return True

        return False

    def _collect_search_items_from_result(self, result_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(result_data, dict):
            return []

        selected_items = result_data.get("selected_items")
        if isinstance(selected_items, list) and selected_items:
            return [x for x in selected_items if isinstance(x, dict)]

        items = result_data.get("items")
        if isinstance(items, list) and items:
            return [x for x in items if isinstance(x, dict)]

        return []

    def _merge_search_items(self, items_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen_urls = set()

        for items in items_list:
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                url = self._normalize_whitespace(item.get("link") or "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                merged.append(item)

        return merged

    def _dedupe_external_candidates(self, items: Any) -> List[Dict[str, Any]]:
        if not isinstance(items, list):
            return []

        picked: List[Dict[str, Any]] = []
        seen_urls = set()

        for item in items:
            if not isinstance(item, dict):
                continue

            url = str(item.get("link") or "").strip()
            title = str(item.get("title") or "").strip()
            snippet = str(item.get("snippet") or item.get("description") or "").strip()

            if not url or url in seen_urls:
                continue

            seen_urls.add(url)
            picked.append(
                {
                    "title": title,
                    "link": url,
                    "snippet": snippet[:500],
                }
            )

        return picked

    def _build_search_queries_with_llm(
        self,
        *,
        user_message: str,
        max_queries: int = 3,
    ) -> List[str]:
        raw_user = (user_message or "").strip()
        safe_cap = max(1, int(max_queries or 1))

        if not raw_user:
            return []

        system_prompt = (
            "너는 검색어 생성기다.\n"
            "사용자의 대화형 질문을 웹 검색용으로 최적화된 짧은 질의 여러 개로 바꿔라.\n"
            "반드시 JSON만 출력한다.\n"
            "\n"
            "[목표]\n"
            "- 실제 검색엔진에서 잘 먹히는 질의 3개 이내를 만든다.\n"
            "- 각 질의는 짧고 핵심적이어야 한다.\n"
            "- 질의들은 서로 보완적이어야 한다.\n"
            "\n"
            "[규칙]\n"
            "1) 군더더기 표현 제거\n"
            "2) 조사/어미 제거\n"
            "3) 핵심 명사, 인물명, 기관명, 주제어 위주\n"
            "4) 너무 긴 문장형 질의 금지\n"
            "5) 1번째 질의는 대표 질의\n"
            "6) 2번째 질의는 기관명/정식명 보강 질의\n"
            "7) 3번째 질의는 의도 확장 질의(공식, 위치, 홈페이지, 전화번호 등)\n"
            "\n"
            "[출력 규칙]\n"
            '- 형식: {"queries":["검색어1","검색어2","검색어3"]}\n'
            f"- 최대 {safe_cap}개\n"
            "- 중복 금지"
        )

        user_prompt = f"[원문 질문]\n{raw_user}"

        try:
            llm = self.ollama.build_chat_llm(model=self.config.model, think=False)
            resp = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )

            raw = self._normalize_stream_chunk_content(getattr(resp, "content", ""))
            data = self._safe_json_loads(raw) or {}
            queries = data.get("queries") or []

            if not isinstance(queries, list):
                queries = []

            result: List[str] = []
            seen = set()

            for q in queries:
                clean = self._normalize_whitespace(q)
                if not clean:
                    continue
                key = clean.lower()
                if key in seen:
                    continue
                seen.add(key)
                result.append(clean)
                if len(result) >= safe_cap:
                    break

            if result:
                return result
        except Exception:
            pass

        fallback: List[str] = []
        for q in [raw_user, f"{raw_user} 공식", f"{raw_user} 정보"]:
            fallback.append(q)
        return self._dedupe_texts(fallback)[:safe_cap]

    def _select_urls_with_llm(
        self,
        *,
        user_message: str,
        items: List[Dict[str, Any]],
        max_items_cap: int = 10,
    ) -> List[Dict[str, Any]]:
        candidates = self._dedupe_external_candidates(items)
        if not candidates:
            return []

        safe_cap = max(1, min(int(max_items_cap or 1), len(candidates)))

        numbered_lines: List[str] = []
        for idx, c in enumerate(candidates, start=1):
            numbered_lines.append(
                f"[{idx}]\n"
                f"title: {c['title']}\n"
                f"url: {c['link']}\n"
                f"snippet: {c['snippet']}\n"
            )

        system_prompt = (
            "너는 검색 결과에서 fetch할 URL을 고르는 URL 선택기다.\n"
            "공식문서, 원문, 1차 출처를 최우선한다.\n"
            "제목/스니펫만으로 부정한다고 비관련이라고 단정하지 말고, 질문 키워드와 부분적으로라도 관련되면 우선 포함한다.\n"
            "특히 스니펫에 '제외', '별도 기준', '참고', '안내' 같은 표현이 있어도, 질문 주제와 같은 기관/학부/제도 문맥이면 탐색 후보로 남긴다.\n"
            "검색 결과는 불완전할 수 있으므로, 관련 가능성이 있으면 엄격한 precision보다 recall을 우선한다.\n"
            "중복 URL은 제거한다.\n"
            "최소 1개, 가능하면 10개까지 고른다.\n"
            "사용자 질문에 포함된 기관명/학부명/제도명 중 1개 이상이 일치하면 snippet에 부정 표현이 있어도 우선 선택한다.\n"
            "공식 사이트 결과는 애매하면 제외하지 말고 포함한다.\n"

            "\n"
            "[출력 규칙]\n"
            "- 반드시 JSON만 출력\n"
            "- 형식: {\"selected_indices\":[1,2],\"reason\":\"짧은 설명\"}"
        )

        user_prompt = (
            f"[질문]\n{user_message}\n\n"
            f"[선택 상한]\n최대 {safe_cap}개\n\n"
            "[후보 URL 목록]\n"
            + "\n".join(numbered_lines)
        )

        try:
            llm = self.ollama.build_chat_llm(model=self.config.model, think=False)
            resp = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )

            raw = self._normalize_stream_chunk_content(getattr(resp, "content", ""))
            data = self._safe_json_loads(raw) or {}
            indices = data.get("selected_indices") or []

            if not isinstance(indices, list):
                indices = []

            selected: List[Dict[str, Any]] = []
            seen = set()

            for x in indices:
                try:
                    idx = int(x)
                except Exception:
                    continue

                if idx < 1 or idx > len(candidates):
                    continue

                item = candidates[idx - 1]
                url = item["link"]
                if url in seen:
                    continue

                seen.add(url)
                selected.append(dict(item))
                if len(selected) >= safe_cap:
                    break

            if selected:
                return selected
        except Exception:
            pass

        return candidates[:safe_cap]

    # ---------------------------------------------------------------------
    # final answer
    # ---------------------------------------------------------------------

    def _generate_grounded_final_answer_stream(
        self,
        *,
        user_message: str,
        result_data: Dict[str, Any],
        mode: str,
    ) -> Generator[str, None, None]:
        context_text = str(result_data.get("context") or "").strip()
        sources = result_data.get("sources") or []

        source_lines: List[str] = []
        if isinstance(sources, list):
            for idx, src in enumerate(sources[:12], start=1):
                if not isinstance(src, dict):
                    continue
                atomic_query = str(src.get("atomic_query") or "").strip()
                title = str(src.get("title") or "").strip()
                url = str(src.get("url") or "").strip()

                label = f"{idx}. "
                if atomic_query:
                    label += f"[{atomic_query}] "
                if title or url:
                    label += f"{title} {url}".strip()
                source_lines.append(label)

        system_prompt = (
            "너는 근거 기반 답변기다.\n"
            "반드시 제공된 context와 sources만 사용해서 한국어로 답변하라.\n"
            "추측 금지.\n"
            "복합 질문이면 가능하면 엔티티별로 나눠 정리하라.\n"
            "한쪽 엔티티만 답하고 끝내지 마라.\n"
            "근거가 부족한 항목은 부족하다고 명시하라.\n"
            "특히 질문 문장을 임의로 바꾸거나 재해석하지 말고, 사용자 마지막 질문 그대로를 기준으로 답하라."
        )

        if mode == "refine":
            system_prompt += "\n근거가 제한적이면 제한사항을 분명히 밝히고 보수적으로 답하라."

        user_parts: List[str] = [
            f"[질문]\n{user_message}",
            f"[context]\n{context_text or '(no context)'}",
        ]

        if source_lines:
            user_parts.append("[sources]\n" + "\n".join(source_lines))

        user_parts.append(
            "[지시]\n"
            "위 근거만 사용해 답변하라.\n"
            "불필요하게 장황하지 않게 작성하라."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="\n\n".join(user_parts)),
        ]

        try:
            llm = self.ollama.build_chat_llm(model=self.config.model, think=False)
            streamed_any = False

            for chunk in llm.stream(messages):
                content = getattr(chunk, "content", None)
                token = self._normalize_stream_chunk_content(content)
                if not token:
                    continue
                streamed_any = True
                yield token

            if streamed_any:
                return
        except Exception:
            pass

        if context_text:
            yield context_text[:1500]

    def _emit_final_from_result(
        self,
        *,
        user_message: str,
        result_data: Dict[str, Any],
        step: int,
        mode: str,
    ) -> Generator[Dict[str, Any], None, bool]:
        if not self._has_meaningful_context(result_data):
            return False

        label = "[grounded-final/chat-stream] 수집된 근거를 chat stream으로 답변 생성"
        if mode == "refine":
            label = "[grounded-final/chat-stream] 근거가 제한적이므로 보수적으로 답변 생성"

        yield {"type": "thought", "step": step, "delta": label}

        emitted_any = False
        for chunk in self._generate_grounded_final_answer_stream(
            user_message=user_message,
            result_data=result_data,
            mode=mode,
        ):
            if not chunk:
                continue
            emitted_any = True
            yield {"type": "delta", "delta": str(chunk), "step": step}

        if emitted_any:
            sources = extract_sources_from_result(result_data)
            if sources:
                yield {
                    "type": "sources",
                    "tool_name": "grounded_final",
                    "sources": sources,
                    "step": step,
                }
            yield {"type": "done"}
            return True

        return False

    # ---------------------------------------------------------------------
    # run
    # ---------------------------------------------------------------------

    def run(
        self,
        user_message: str,
        collection_name: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        original_user_message = self._normalize_whitespace(user_message)
        fallback_collection = (collection_name or self.config.default_collection or "").strip()
        collections = self._get_available_collections()

        atomic_queries = self._decompose_user_message_into_atomic_queries(
            user_message=original_user_message,
            max_atomic_queries=4,
        )
        if not atomic_queries:
            atomic_queries = [original_user_message]

        yield {
            "type": "thought",
            "step": 1,
            "delta": "[agent-speed] atomic_queries="
            + json.dumps(atomic_queries, ensure_ascii=False),
        }

        # -----------------------------------------------------------------
        # 1) internal_search 를 atomic query 별로 각각 수행
        # -----------------------------------------------------------------
        internal_query_results: List[Tuple[str, Dict[str, Any]]] = []

        for idx, atomic_query in enumerate(atomic_queries, start=1):
            selected_collections = self._select_collections_for_query(
                user_message=atomic_query,
                collections=collections,
                fallback_collection=fallback_collection,
            )

            yield {
                "type": "thought",
                "step": 1,
                "delta": (
                    f"[agent-speed] internal_search {idx}/{len(atomic_queries)} "
                    f"atomic={json.dumps(atomic_query, ensure_ascii=False)} "
                    f"collections={json.dumps(selected_collections, ensure_ascii=False)}"
                ),
            }

            if not selected_collections:
                continue

            internal_result = yield from execute_tool(
                tools=self.tools,
                tool_name="internal_search",
                arguments={
                    "query": atomic_query,
                    "collections": selected_collections,
                    "k": 6,
                },
                step=1,
                messages=[],
                user_message=original_user_message,  # 원본 유지
            )

            if internal_result.ok and isinstance(internal_result.data, dict):
                if self._has_meaningful_internal_result(internal_result.data):
                    internal_query_results.append((atomic_query, internal_result.data))

        merged_internal = self._merge_result_payloads(
            query_to_result=internal_query_results,
            prefer_longer_context=True,
            stitch_sources=True,
        )

        if self._has_meaningful_internal_result(merged_internal):
            judge = self._judge_internal_answerability(
                user_message=original_user_message,
                internal_result=merged_internal,
            )

            yield {
                "type": "thought",
                "step": 1,
                "delta": (
                    "[agent-speed] merged internal judge 결과: "
                    f"use_internal={judge.get('use_internal')} "
                    f"need_external={judge.get('need_external')} "
                    f"confidence={judge.get('confidence')} "
                    f"reason={judge.get('reason')}"
                ),
            }

            if bool(judge.get("use_internal")) and not bool(judge.get("need_external")):
                done = yield from self._emit_final_from_result(
                    user_message=original_user_message,
                    result_data=merged_internal,
                    step=1,
                    mode="answer",
                )
                if done:
                    return
        else:
            yield {
                "type": "thought",
                "step": 1,
                "delta": "[agent-speed] merged internal 결과가 약함 -> external 로 전환",
            }

        # -----------------------------------------------------------------
        # 2) external_search 를 atomic query 별로 각각 수행
        # -----------------------------------------------------------------
        all_collected_search_items: List[List[Dict[str, Any]]] = []
        search_top_k = 10

        for atomic_idx, atomic_query in enumerate(atomic_queries, start=1):
            rewritten_queries = self._build_search_queries_with_llm(
                user_message=atomic_query,
                max_queries=3,
            )
            if not rewritten_queries:
                rewritten_queries = [atomic_query]

            yield {
                "type": "thought",
                "step": 2,
                "delta": (
                    f"[agent-speed] rewritten_queries {atomic_idx}/{len(atomic_queries)}="
                    f"{json.dumps(rewritten_queries, ensure_ascii=False)}"
                ),
            }

            chosen_result_data: Optional[Dict[str, Any]] = None
            adopted_query = ""

            for ridx, rewritten_query in enumerate(rewritten_queries, start=1):
                yield {
                    "type": "thought",
                    "step": 2,
                    "delta": (
                        f"[agent-speed] external_search {atomic_idx}.{ridx}/"
                        f"{len(rewritten_queries)}: {rewritten_query}"
                    ),
                }

                search_result = yield from execute_tool(
                    tools=self.tools,
                    tool_name="external_search",
                    arguments={
                        "query": rewritten_query,  # 검색용 질의만 여기서 사용
                        "num": 10,
                        "top_k_urls": search_top_k,
                    },
                    step=2,
                    messages=[],
                    user_message=original_user_message,  # 원본 유지
                )

                if not search_result.ok or not isinstance(search_result.data, dict):
                    continue

                if self._has_meaningful_search_items(search_result.data):
                    chosen_result_data = search_result.data
                    adopted_query = rewritten_query
                    break

            if not chosen_result_data:
                yield {
                    "type": "thought",
                    "step": 2,
                    "delta": f"[agent-speed] external_search 실패 atomic={atomic_query}",
                }
                continue

            collected_items = self._collect_search_items_from_result(chosen_result_data)
            if collected_items:
                all_collected_search_items.append(collected_items)

            yield {
                "type": "thought",
                "step": 2,
                "delta": (
                    f"[agent-speed] external_search 채택 atomic={json.dumps(atomic_query, ensure_ascii=False)} "
                    f"query={json.dumps(adopted_query, ensure_ascii=False)} "
                    f"items={len(collected_items)}"
                ),
            }

        merged_search_items = self._merge_search_items(all_collected_search_items)
        if not merged_search_items:
            yield {"type": "error", "error": "external_search failed for all atomic queries"}
            return

        llm_cap = min(search_top_k, len(merged_search_items))
        picked_items = self._select_urls_with_llm(
            user_message=original_user_message,
            items=merged_search_items,
            max_items_cap=llm_cap,
        )
        if not picked_items:
            yield {"type": "error", "error": "external_search returned no fetchable items"}
            return

        yield {
            "type": "thought",
            "step": 2,
            "delta": (
                "[agent-speed] merged URL selector 결과 "
                f"(selected={len(picked_items)}, merged_items={len(merged_search_items)})"
            ),
        }

        # -----------------------------------------------------------------
        # 3) fetch raw
        # -----------------------------------------------------------------
        yield {
            "type": "thought",
            "step": 3,
            "delta": f"[agent-speed] 선택된 URL {len(picked_items)}건 fetch 진행",
        }

        fetch_result = yield from execute_tool(
            tools=self.tools,
            tool_name="external_fetch_raw",
            arguments={
                "query": original_user_message,  # 원본 질문 유지
                "items": picked_items,
                "max_fetch": len(picked_items),
            },
            step=3,
            messages=[],
            user_message=original_user_message,
        )
        if not fetch_result.ok or not isinstance(fetch_result.data, dict):
            yield {"type": "error", "error": "external_fetch_raw failed"}
            return

        raw_documents = fetch_result.data.get("documents")
        if not isinstance(raw_documents, list) or not raw_documents:
            yield {"type": "error", "error": "external_fetch_raw returned no documents"}
            return

        # -----------------------------------------------------------------
        # 4) extract main content
        # -----------------------------------------------------------------
        yield {
            "type": "thought",
            "step": 4,
            "delta": "[agent-speed] external_extract_main_content 진행",
        }

        extract_result = yield from execute_tool(
            tools=self.tools,
            tool_name="external_extract_main_content",
            arguments={
                "query": original_user_message,  # 원본 질문 유지
                "documents": raw_documents,
            },
            step=4,
            messages=[],
            user_message=original_user_message,
        )
        if not extract_result.ok or not isinstance(extract_result.data, dict):
            yield {"type": "error", "error": "external_extract_main_content failed"}
            return

        extracted_documents = extract_result.data.get("documents")
        if not isinstance(extracted_documents, list) or not extracted_documents:
            yield {"type": "error", "error": "external_extract_main_content returned no documents"}
            return

        extracted_documents = self._merge_documents_by_link(extracted_documents)

        yield {
            "type": "thought",
            "step": 4,
            "delta": "[agent-speed] 현재 페이지 answerability judge + AI-guided BFS 확장 진행",
        }

        bfs_expanded_documents = self._ai_guided_bfs_expand(
            user_message=original_user_message,
            seed_documents=extracted_documents,
            max_depth=int(getattr(self.config, "agent_speed_bfs_max_depth", 2) or 2),
            max_total_pages=int(getattr(self.config, "agent_speed_bfs_max_total_pages", 12) or 12),
            max_links_per_page=int(getattr(self.config, "agent_speed_bfs_max_links_per_page", 4) or 4),
        )

        if bfs_expanded_documents:
            extracted_documents = self._merge_documents_by_link(extracted_documents + bfs_expanded_documents)

        yield {
            "type": "thought",
            "step": 4,
            "delta": (
                "[agent-speed] BFS 확장 완료 "
                f"(seed_docs={len(extract_result.data.get('documents') or [])}, total_docs={len(extracted_documents)})"
            ),
        }

        # -----------------------------------------------------------------
        # 5) build_context 도 atomic query 별로 각각 수행
        # -----------------------------------------------------------------
        per_atomic_build_results: List[Tuple[str, Dict[str, Any]]] = []

        for idx, atomic_query in enumerate(atomic_queries, start=1):
            yield {
                "type": "thought",
                "step": 5,
                "delta": (
                    f"[agent-speed] external_build_context {idx}/{len(atomic_queries)} "
                    f"atomic={json.dumps(atomic_query, ensure_ascii=False)}"
                ),
            }

            build_result = yield from execute_tool(
                tools=self.tools,
                tool_name="external_build_context",
                arguments={
                    "query": atomic_query,   # retrieval 용
                    "documents": extracted_documents,
                    "top_k_chunks": 8,       # 기존 4 -> 8
                },
                step=5,
                messages=[],
                user_message=original_user_message,  # 원본 질문 유지
            )

            if not build_result.ok or not isinstance(build_result.data, dict):
                continue

            if self._has_meaningful_context(build_result.data):
                per_atomic_build_results.append((atomic_query, build_result.data))

        # fallback: 전체 질문으로 한 번 더
        if not per_atomic_build_results:
            yield {
                "type": "thought",
                "step": 5,
                "delta": "[agent-speed] atomic build_context 가 약함 -> 전체 질문 build_context fallback",
            }

            build_result = yield from execute_tool(
                tools=self.tools,
                tool_name="external_build_context",
                arguments={
                    "query": original_user_message,
                    "documents": extracted_documents,
                    "top_k_chunks": 10,
                },
                step=5,
                messages=[],
                user_message=original_user_message,
            )

            if not build_result.ok or not isinstance(build_result.data, dict):
                yield {"type": "error", "error": "external_build_context failed"}
                return

            merged_build = build_result.data
            if isinstance(merged_build, dict):
                merged_build["sources"] = self._stitch_adjacent_sources(
                    merged_build.get("sources") or []
                )
                if merged_build.get("sources"):
                    rebuilt_context: List[str] = []
                    for src in merged_build["sources"]:
                        content = str(src.get("content") or src.get("snippet") or "").strip()
                        if content:
                            rebuilt_context.append(content)
                    if rebuilt_context:
                        merged_build["context"] = "\n\n".join(rebuilt_context).strip()
        else:
            merged_build = self._merge_result_payloads(
                query_to_result=per_atomic_build_results,
                prefer_longer_context=False,
                stitch_sources=True,
            )

        context_text = str(merged_build.get("context") or "").strip()
        yield {
            "type": "thought",
            "step": 5,
            "delta": (
                "[agent-speed] merged build_context 완료 "
                f"(context_len={len(context_text)}, sources={len(merged_build.get('sources') or [])})"
            ),
        }

        done = yield from self._emit_final_from_result(
            user_message=original_user_message,
            result_data=merged_build,
            step=5,
            mode="answer" if context_text else "refine",
        )
        if done:
            return

        yield {"type": "error", "error": "agent_speed could not produce final answer"}