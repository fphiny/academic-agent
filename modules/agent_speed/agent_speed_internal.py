from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from .agent_speed_utils import (
    dedupe_texts,
    normalize_stream_chunk_content,
    normalize_whitespace,
    safe_json_loads,
)


def normalize_linebreaks(text: Any) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n")


def normalize_for_display(text: Any) -> str:
    """
    본문 표시/LLM 요약 입력용 정규화.
    - 줄바꿈은 보존
    - 각 줄의 양끝 공백만 제거
    - 과도한 빈 줄은 최대 2줄로 축약
    """
    s = normalize_linebreaks(text)
    lines = [line.strip() for line in s.split("\n")]
    s = "\n".join(lines)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def normalize_for_debug_preview(text: Any) -> str:
    """
    디버그 미리보기/키 비교와 달리, 미리보기는 한 줄로 압축.
    """
    return normalize_whitespace(str(text or ""))


class AgentSpeedInternal:
    """
    internal retrieval 전용 helper

    역할
    - 질문 구조 판별(single vs multi)
    - 질문 atomic decomposition
    - internal search 전용 질의 생성
    - collection selection
    - query별 retrieval 결과 요약
    - query별 요약 병합 / 판정

    설계 원칙
    - atomic query는 필요할 때만 분해
    - 먼저 LLM으로 질문이 단일인지 복수인지 판정
    - 단일 질문이면 분해하지 않고 그대로 1개로 유지
    - 복수 질문일 때만 atomic decomposition 수행
    - atomic query마다 internal search query를 생성
    - judge 단계에서는 summary를 쓰지 않고 raw context/source를 사용
    - external fallback이 결정되면 최종 merge에서도 summary를 사용하지 않음
    """

    def __init__(self, *, ollama, sources):
        self.ollama = ollama
        self.sources = sources
        self.model = getattr(self.ollama, "default_model", None)

    # ---------------------------------------------------------------------
    # display/debug helpers
    # ---------------------------------------------------------------------

    def _extract_source_text(self, src: Dict[str, Any]) -> str:
        return str(
            src.get("content")
            or src.get("snippet")
            or src.get("preview")
            or src.get("document")
            or ""
        )

    def _extract_source_text_for_display(self, src: Dict[str, Any]) -> str:
        return normalize_for_display(self._extract_source_text(src))

    def _extract_source_text_for_debug(self, src: Dict[str, Any], limit: int = 220) -> str:
        return self._short_text(self._extract_source_text(src), limit)

    def _debug(self, title: str, payload: Any = None, max_len: int = 40000):
        print("\n" + "=" * 140)
        print(f"[AgentSpeedInternal DEBUG] {title}")
        print("=" * 140)
        if payload is not None:
            try:
                if isinstance(payload, (dict, list, tuple)):
                    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
                else:
                    rendered = str(payload)
            except Exception:
                rendered = str(payload)

            if len(rendered) > max_len:
                rendered = rendered[:max_len] + "\n... (truncated)"
            print(rendered)
        print("=" * 140 + "\n")

    def _short_text(self, text: Any, limit: int = 180) -> str:
        s = normalize_for_debug_preview(text)
        if len(s) <= limit:
            return s
        return s[:limit] + " ..."

    def _describe_source_for_debug(self, src: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": src.get("id"),
            "doc_id": src.get("doc_id"),
            "chunk_id": src.get("chunk_id"),
            "title": src.get("title"),
            "url": src.get("url") or src.get("link"),
            "rerank_score": src.get("rerank_score"),
            "distance": src.get("distance"),
            "dedupe_key": self._build_internal_source_key(src),
            "text_preview": self._extract_source_text_for_debug(src, 220),
        }

    # ---------------------------------------------------------------------
    # common
    # ---------------------------------------------------------------------

    def _build_llm(self):
        return self.ollama.build_chat_llm(model=self.model, think=False)

    # ---------------------------------------------------------------------
    # user-message structure classifier
    # ---------------------------------------------------------------------

    def classify_user_message_structure(
        self,
        *,
        user_message: str,
        max_atomic_queries: int = 3,
    ) -> Dict[str, Any]:
        """
        사용자 입력이 단일 질문인지, 복수 독립 질문인지 먼저 판정한다.

        반환 예시:
        {
            "is_multi": False,
            "num_queries": 1,
            "reason": "하나의 인물에 대한 속성 조회"
        }
        """
        raw_user = normalize_whitespace(user_message)
        safe_cap = max(1, int(max_atomic_queries or 1))

        if not raw_user:
            return {
                "is_multi": False,
                "num_queries": 0,
                "reason": "empty input",
            }

        system_prompt = (
            "너는 사용자 입력의 질문 구조를 판별하는 분류기다.\n"
            "목표는 이 입력이 단일 질문인지, 서로 독립된 복수 질문인지 판정하는 것이다.\n"
            "\n"
            "[판정 원칙]\n"
            "1) 하나의 대상에 대한 속성 나열(소속, 연락처, 연구분야, 이메일, 소개 등)은 단일 질문이다.\n"
            "2) 하나의 인물/기관/주제에 대한 설명 확장은 단일 질문이다.\n"
            "3) 서로 다른 대상에 대한 요청은 복수 질문이다.\n"
            "4) 서로 다른 행위/요청이 병렬로 묶여 있으면 복수 질문이다.\n"
            "5) 표현이 길거나 조건이 붙었다고 복수 질문으로 보지 말 것.\n"
            "6) 번역, 정규화, 말투 변경, 부연설명은 질문 수를 늘리는 이유가 아니다.\n"
            "7) 'A의 소속과 연락처'처럼 한 대상의 속성 병렬 나열은 반드시 단일 질문으로 본다.\n"
            "\n"
            "[예시]\n"
            "- '고영웅 교수님 정보를 알려줘' -> 단일\n"
            "- '고영웅 교수님 소속, 연락처, 연구분야 알려줘' -> 단일\n"
            "- '고영웅 교수님 정보와 한림대학교 주소 알려줘' -> 복수\n"
            "- 'A 연락처와 B 위치 알려줘' -> 복수\n"
            "- 'A의 소속과 연락처 알려줘' -> 단일\n"
            "\n"
            "[출력 규칙]\n"
            "- 반드시 JSON만 출력\n"
            '- 형식: {"is_multi":false,"num_queries":1,"reason":"짧은 설명"}\n'
            f"- num_queries는 1 이상 {safe_cap} 이하\n"
            "- 단일 질문이면 num_queries는 반드시 1\n"
            "- 확실하지 않으면 단일 질문 쪽으로 보수적으로 판단할 것\n"
        )

        try:
            llm = self._build_llm()
            resp = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=raw_user)]
            )
            raw = normalize_stream_chunk_content(
                getattr(resp, "content", ""),
                text_normalizer=self.ollama.normalize_text_content,
            )
            data = safe_json_loads(raw) or {}

            is_multi = bool(data.get("is_multi"))
            reason = normalize_whitespace(str(data.get("reason") or ""))

            try:
                num_queries = int(data.get("num_queries") or (2 if is_multi else 1))
            except Exception:
                num_queries = 2 if is_multi else 1

            num_queries = max(1, min(num_queries, safe_cap))

            if num_queries <= 1:
                is_multi = False
                num_queries = 1

            result = {
                "is_multi": is_multi,
                "num_queries": num_queries,
                "reason": reason or "structure classified",
            }
            self._debug("classify_user_message_structure", result)
            return result
        except Exception:
            return {
                "is_multi": False,
                "num_queries": 1,
                "reason": "classifier fallback",
            }

    # ---------------------------------------------------------------------
    # atomic decomposition
    # ---------------------------------------------------------------------

    def decompose_user_message_into_atomic_queries(
        self,
        *,
        user_message: str,
        max_atomic_queries: int = 4,
    ) -> List[str]:
        raw_user = normalize_whitespace(user_message)
        safe_cap = max(1, int(max_atomic_queries or 1))

        if not raw_user:
            return []

        structure = self.classify_user_message_structure(
            user_message=raw_user,
            max_atomic_queries=safe_cap,
        )

        # 단일 질문이면 무조건 그대로 1개 유지
        if not structure.get("is_multi"):
            result = [raw_user]
            self._debug("decompose_user_message_into_atomic_queries(single)", result)
            return result

        system_prompt = (
            "너는 사용자 질문을 검색용 query set으로 정리하는 질의 정리기다.\n"
            "현재 검색은 이미 특정 학부/학과 전용 컬렉션 내부에서 수행된다.\n"
            "따라서 입력에 포함된 학부명, 학과명, 전공명 등 컬렉션 범위와 중복되는 소속명은 반드시 제거하라.\n"
            "\n"
            "[최우선 규칙]\n"
            "1) 컬렉션 범위와 중복되는 학부명/학과명/소속명은 반드시 제거할 것\n"
            "2) 제거 후에도 사용자 질문의 핵심 의도는 유지할 것\n"
            "3) 필요할 때만 검색용 확장 query를 추가할 것\n"
            "4) 같은 의미를 불필요하게 반복하지 말 것\n"
            "5) 사용자 입력에 없는 새로운 고유명사는 추가하지 말 것\n"
            "6) 단일 질문이어도 검색 성능 향상을 위해 복수 query를 만들 수 있다\n"
            "7) 결과 query에는 학부명/학과명/소속명이 남아 있으면 안 된다\n"
            "\n"
            "[정규화 규칙]\n"
            "- '데이터사이언스학부 전공은?' -> ['전공', '상세전공', '세부전공', '전공 정보']\n"
            "- '데이터사이언스학부 교수진' -> ['교수진', '교수', '전임교원']\n"
            "- 학부명/학과명이 포함된 표현은 모두 제거 후 핵심 명사구만 남길 것\n"
            "\n"
            "[출력 규칙]\n"
            "- 반드시 JSON만 출력\n"
            '- 형식: {"queries":["..."],"reason":"짧은 설명"}\n'
            f"- queries 최대 {safe_cap}개\n"
            "- 중복 금지\n"
            "- 반드시 한글외에 모든 말은 번역하라\n"
        )

        try:
            llm = self._build_llm()
            resp = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=raw_user)]
            )
            raw = normalize_stream_chunk_content(
                getattr(resp, "content", ""),
                text_normalizer=self.ollama.normalize_text_content,
            )
            data = safe_json_loads(raw) or {}
            queries = data.get("queries") or []

            if isinstance(queries, list):
                cleaned = dedupe_texts(
                    [normalize_whitespace(str(x)) for x in queries if normalize_whitespace(str(x))]
                )[:safe_cap]
                if cleaned:
                    self._debug("decompose_user_message_into_atomic_queries(multi)", cleaned)
                    return cleaned
        except Exception:
            pass

        # 복수 질문 판정 이후 decomposition까지 실패한 경우에도
        # 보수적으로 원문 1개를 반환해서 과분해를 막는다.
        result = [raw_user]
        self._debug("decompose_user_message_into_atomic_queries(fallback)", result)
        return result

    # ---------------------------------------------------------------------
    # internal-search query builder
    # ---------------------------------------------------------------------

    def build_internal_search_queries(
        self,
        *,
        atomic_queries: List[str],
        original_user_message: str,
        max_queries_per_atomic: int = 3,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        safe_cap = max(1, int(max_queries_per_atomic or 1))

        for atomic_query in atomic_queries:
            clean_atomic = normalize_whitespace(atomic_query)
            if not clean_atomic:
                continue

            generated_queries = self._translate_for_internal_search(
                atomic_query=clean_atomic,
                original_user_message=original_user_message,
                max_queries=safe_cap,
            )

            internal_queries = dedupe_texts(
                [normalize_whitespace(x) for x in generated_queries if normalize_whitespace(x)]
            )

            if not internal_queries:
                internal_queries = [clean_atomic]

            results.append(
                {
                    "atomic_query": clean_atomic,
                    "internal_queries": internal_queries[:safe_cap],
                }
            )

        self._debug("build_internal_search_queries", results)
        return results

    def _translate_for_internal_search(
        self,
        *,
        atomic_query: str,
        original_user_message: str,
        max_queries: int = 2,
    ) -> List[str]:
        safe_cap = max(1, int(max_queries or 1))
        clean_atomic = normalize_whitespace(atomic_query)
        clean_original = normalize_whitespace(original_user_message)

        if not clean_atomic:
            return []

        system_prompt = (
            "너는 internal retrieval용 검색 질의 변환기다.\n"
            "atomic query를 내부검색에 사용할 검색 질의로 변환한다.\n"
            "현재 검색은 이미 특정 컬렉션 내부에서 수행된다고 가정한다.\n"
            "따라서 컬렉션 범위와 중복되는 상위 소속명(학부명, 학과명 등)은 제거하고 핵심 검색어만 남겨라.\n"
            "max_queries는 최대 허용 개수이며, 필요할 때만 여러 개를 만들 수 있다.\n"
            "\n"
            "[최우선 규칙]\n"
            "1) 원래 의미를 바꾸지 말 것\n"
            "2) 사용자 입력에 없는 새로운 엔티티를 추가하지 말 것\n"
            "3) 컬렉션 범위와 중복되는 상위 소속명은 반드시 제거할 것\n"
            "4) 내부 검색용 queries는 검색에 유리한 짧은 한글 명사구로 만들 것\n"
            "5) 입력에 영문이 있으면 한글로 번역하거나 음역할 것\n"
            "6) 이미 한글인 표현은 그대로 유지할 것\n"
            "7) 검색 recall을 높이기 위해 필요한 범위에서만 질의를 늘릴 것\n"
            "8) 같은 의미의 중복 질의는 금지\n"
            "9) 불필요한 수식어, 조사, 문장형 표현은 제거할 것\n"
            "10) 일반도가 지나치게 높은 단일어는 꼭 필요할 때만 사용할 것\n"
            "\n"
            "[질의 생성 원칙]\n"
            "- 질문형 표현은 검색용 명사구로 바꿀 것\n"
            "- 전공/세부전공/상세전공처럼 실제 탐색에 유용한 표현만 제한적으로 확장할 것\n"
            "- 정보/소개/프로필/약력 같은 일반 확장어는 입력 의미에 포함될 때만 사용할 것\n"
            "- 직함/소속 확장은 실제 검색 필요성이 있을 때만 사용할 것\n"
            "\n"
            "[예시]\n"
            "- '데이터사이언스학부 전공은?' -> ['전공', '세부전공', '상세전공', '전공 정보']\n"
            "- '데이터사이언스학부 교수진' -> ['교수진', '교수', '전임교원']\n"
            "- '졸업 요건이 뭐야?' -> ['졸업 요건']\n"
            "\n"
            "[실패 조건]\n"
            "- 컬렉션 범위와 중복되는 학부명/학과명이 남아 있으면 실패\n"
            "- queries 안에 불필요한 영문이 남아 있으면 실패\n"
            "- 사용자 입력에 없는 엔티티를 넣으면 실패\n"
            "- 거의 같은 표현만 반복하면 실패\n"
            "\n"
            "[출력 규칙]\n"
            "- 반드시 JSON만 출력\n"
            '- 형식: {"queries":["..."],"reason":"짧은 설명"}\n'
            f"- queries 최대 {safe_cap}개\n"
            "- 중복 금지\n"
            "- 한글외의 언어는 한글로 반드시 번역해라\n"
        )

        user_prompt = (
            f"[원래 사용자 질문]\n{clean_original}\n\n"
            f"[현재 atomic query]\n{clean_atomic}\n\n"
            "[지시]\n"
            "internal search용 query만 생성하라.\n"
            "Generate internal search queries only. You may normalize the search query into Korean for retrieval, but this must not change the final answer language."
        )

        cleaned: List[str] = []

        try:
            llm = self._build_llm()
            resp = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            raw = normalize_stream_chunk_content(
                getattr(resp, "content", ""),
                text_normalizer=self.ollama.normalize_text_content,
            )
            data = safe_json_loads(raw) or {}
            queries = data.get("queries") or []

            seen = set()
            if isinstance(queries, list):
                for q in queries:
                    candidate = normalize_whitespace(str(q))
                    if not candidate:
                        continue
                    key = candidate.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    cleaned.append(candidate)
                    if len(cleaned) >= safe_cap:
                        break
        except Exception:
            pass

        if cleaned:
            result = cleaned[:safe_cap]
            self._debug(
                "_translate_for_internal_search",
                {"atomic_query": clean_atomic, "queries": result},
            )
            return result

        return [clean_atomic]

    def _contains_non_korean(self, text: str) -> bool:
        for ch in text:
            if ch.isascii() and ch.isalpha():
                return True
        return False

    # ---------------------------------------------------------------------
    # collection select
    # ---------------------------------------------------------------------

    def select_collections_for_query(
        self,
        *,
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
            "3) 질문과 관련 없는 컬렉션은 고르지 않는다.\n"
            "4) 질문이 애매하거나 내부 컬렉션 적합성이 낮으면 fallback collection만 선택해도 된다.\n"
            "5) 무리해서 많이 고르지 말 것\n"
            "\n"
            "[출력 규칙]\n"
            "- 반드시 JSON만 출력한다.\n"
            '- 형식: {"selected_names":["collection1","collection2"],"reason":"짧은 설명"}\n'
            "- selected_names는 collection name 문자열 배열\n"
            "- 목록에 없는 이름 생성 금지\n"
            "- JSON 외 설명 금지"
        )

        user_prompt = (
            f"[사용자 질문]\n{raw_user}\n\n"
            f"[최대 선택 개수]\n{safe_cap}\n\n"
            "[collection 목록]\n" + "\n".join(collection_lines)
        )

        try:
            llm = self._build_llm()
            resp = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )

            raw = normalize_stream_chunk_content(
                getattr(resp, "content", ""),
                text_normalizer=self.ollama.normalize_text_content,
            )
            data = safe_json_loads(raw) or {}
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
    # summarization for per-query retrieval
    # ---------------------------------------------------------------------

    def summarize_result_payload(
        self,
        *,
        query: str,
        result_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        clean_query = normalize_whitespace(query)

        if not isinstance(result_data, dict):
            return {
                "query": clean_query,
                "summary": "",
                "sources": [],
            }

        # 본문 표시/요약 입력에는 줄바꿈 보존
        context_text = normalize_for_display(result_data.get("context") or "")
        sources = result_data.get("sources") or []
        if not isinstance(sources, list):
            sources = []

        if not clean_query:
            return {
                "query": "",
                "summary": context_text,
                "sources": sources,
            }

        if not context_text and not sources:
            return {
                "query": clean_query,
                "summary": "",
                "sources": [],
            }

        source_lines: List[str] = []
        for idx, src in enumerate(sources[:20], start=1):
            if not isinstance(src, dict):
                continue
            title = normalize_whitespace(str(src.get("title") or ""))
            content = self._extract_source_text_for_display(src)
            if title or content:
                source_lines.append(
                    f"[source {idx}]\n"
                    f"title: {title}\n"
                    f"text:\n{content}"
                )

        print("\n" + "=" * 120)
        print(f"[summarize_result_payload] FULL CONTEXT FOR QUERY: {clean_query}")
        print("=" * 120)
        print(context_text)
        print("=" * 120 + "\n")

        if source_lines:
            print("\n" + "=" * 120)
            print(f"[summarize_result_payload] FULL SOURCES FOR QUERY: {clean_query}")
            print("=" * 120)
            print("\n\n".join(source_lines))
            print("=" * 120 + "\n")

        system_prompt = (
            "너는 internal retrieval 결과 요약기다.\n"
            "주어진 query에 직접 관련된 정보만 남기고 검색 결과를 짧고 정확하게 요약하라.\n"
            "추측하지 말고 제공된 context와 sources에 명시된 내용만 사용하라.\n"
            "\n"
            "[핵심 규칙]\n"
            "1) query와 직접 관련 없는 문장 제거\n"
            "2) 중복 제거\n"
            "3) 핵심 엔티티, 핵심 사실, 수치, 날짜, 소속, 상태를 우선 유지\n"
            "4) 불확실하거나 충돌하는 내용은 '불명확' 또는 '상충 가능'처럼 표시\n"
            "5) 장황한 문장 대신 사실 위주로 정리\n"
            "6) 이 요약은 나중에 최종 종합답변의 재료가 되므로 정보 밀도를 높일 것\n"
            "\n"
            "[출력 형식]\n"
            "반드시 JSON만 출력\n"
            '- 형식: {"summary":"...","reason":"짧은 설명"}\n'
        )

        user_prompt = (
            f"[query]\n{clean_query}\n\n"
            f"[context]\n{context_text or '(empty)'}\n\n"
            f"[sources]\n{chr(10).join(source_lines) if source_lines else '(empty)'}"
        )

        try:
            llm = self._build_llm()
            resp = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            raw = normalize_stream_chunk_content(
                getattr(resp, "content", ""),
                text_normalizer=self.ollama.normalize_text_content,
            )
            data = safe_json_loads(raw) or {}
            summary = normalize_for_display(str(data.get("summary") or ""))

            if summary:
                return {
                    "query": clean_query,
                    "summary": summary,
                    "sources": sources,
                }
        except Exception:
            pass

        fallback_summary_blocks: List[str] = []
        if context_text:
            fallback_summary_blocks.append(context_text)

        if not fallback_summary_blocks and source_lines:
            fallback_summary_blocks.append("\n".join(source_lines))

        return {
            "query": clean_query,
            "summary": "\n\n".join(fallback_summary_blocks).strip(),
            "sources": sources,
        }

    def summarize_query_to_result_payloads(
        self,
        *,
        query_to_result: List[Tuple[str, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        summarized: List[Dict[str, Any]] = []

        for query_key, result_data in query_to_result:
            item = self.summarize_result_payload(
                query=query_key,
                result_data=result_data,
            )
            if item.get("summary"):
                summarized.append(item)

        return summarized

    # ---------------------------------------------------------------------
    # internal result judge / merge
    # ---------------------------------------------------------------------

    def has_meaningful_context(self, result_data: Any) -> bool:
        if not isinstance(result_data, dict):
            return False

        context = str(result_data.get("context") or "").strip()
        if len(context) >= 80:
            return True

        sources = result_data.get("sources")
        if isinstance(sources, list) and len(sources) >= 2:
            return True

        return False

    def has_meaningful_internal_result(self, result_data: Any) -> bool:
        if not isinstance(result_data, dict):
            return False

        context = str(result_data.get("context") or "").strip()
        if len(context) >= 60:
            return True

        sources = result_data.get("sources")
        if isinstance(sources, list) and len(sources) >= 1:
            return True

        return False

    def judge_internal_answerability(
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

        context_text = normalize_for_display(internal_result.get("context") or "")
        sources = internal_result.get("sources") or []

        source_lines: List[str] = []
        source_count = 0

        if isinstance(sources, list):
            for idx, src in enumerate(sources[:10], start=1):
                if not isinstance(src, dict):
                    continue
                title = str(src.get("title") or "").strip()
                snippet = normalize_for_display(src.get("snippet") or "")
                content = self._extract_source_text_for_display(src)
                text = content or snippet
                if text:
                    source_count += 1
                source_lines.append(f"{idx}. title={title}\ntext=\n{text}")

        if len(context_text) < 15 and source_count == 0:
            return {
                "use_internal": False,
                "need_external": True,
                "confidence": "low",
                "reason": "internal context 가 거의 없음",
            }

        system_prompt = (
            "너는 retrieval sufficiency judge 다.\n"
            "사용자 요청과 internal retrieval 결과(context, sources)를 보고,\n"
            "오직 '정보 요구사항'이 internal 정보만으로 충분한지 판단하라.\n"
            "\n"
            "[절대 금지]\n"
            "1) 메일 전송, 초안 작성, 일정 생성 등 행위 자체를 평가하지 말 것\n"
            "2) 실행 가능성, 기능 지원 여부, 시스템 동작 여부를 reason 에 쓰지 말 것\n"
            "3) 사용자가 어떤 행동 요청을 했더라도, 그 행동 자체는 평가하지 말 것\n"
            "4) 이 judge 는 오직 답변 본문에 들어갈 사실 정보가 internal 정보만으로 충분한지만 평가할 것\n"
            "\n"
            "[핵심 판단 질문]\n"
            "- internal 결과만으로 필요한 사실 정보를 충분히 정리할 수 있는가?\n"
            "- 질문의 핵심 대상이 internal 결과에서 명확히 식별되는가?\n"
            "- 결과들이 대체로 같은 대상과 같은 사실을 가리키는가?\n"
            "- external 확인이 꼭 필요한가?\n"
            "\n"
            "[최우선 원칙]\n"
            "1) 질문의 핵심 대상이 불명확하면 need_external=true\n"
            "2) 질문이 특정 역할/직위/신분(예: 총장, CEO, 대통령, 담당자)을 묻는 경우,\n"
            "   internal 결과에서 그 역할이 명시적으로 확인되어야 한다\n"
            "3) 관련 인물이나 기관 정보가 있다는 이유만으로 해당 역할을 추론해서는 안 된다\n"
            "4) 일부 값이 빠지거나 많아도, 핵심 엔티티와 핵심 사실이 명확하면 internal 사용 가능\n"
            "5) context 가 길거나 관련 문서가 많다고 자동 승인하지 말 것\n"
            "\n"
            "[중요 규칙: 역할/직위 질문]\n"
            "- 질문이 특정 역할 또는 직위를 요구하면,\n"
            "  internal 결과에 그 역할 또는 직위가 직접적으로 나타나야 한다\n"
            "- 단순히 같은 기관의 다른 인물 정보만 있는 경우는 충분하지 않다\n"
            "- 과거 정보만 있고 현재 여부가 불명확하면 확정된 답으로 볼 수 없다\n"
            "- 역할/직위를 유추, 추정, 보완해서 판단하지 말고 명시된 사실만 사용할 것\n"
            "\n"
            "[need_external=true 조건]\n"
            "- query 결과들이 서로 다른 대상을 가리키는 충돌이 명확함\n"
            "- 같은 대상이라고 보기 어려울 정도로 핵심 슬롯이 강하게 충돌함\n"
            "- internal source 들이 질문과 거의 무관함\n"
            "- 질문 핵심 엔티티 또는 핵심 역할/직위를 internal 결과에서 찾지 못함\n"
            "- 관련 정보는 있으나 질문에 대한 확정적 사실을 만들 수 없음\n"
            "- 특정 시점이 중요한 질문인데, 해당 시점 기준 정보가 internal 에서 확인되지 않음\n"
            "\n"
            "[internal 사용 가능 조건]\n"
            "- 질문 핵심 엔티티가 internal 결과에 직접 등장함\n"
            "- 질문이 요구하는 핵심 사실(예: 역할, 소속, 날짜, 상태)이 internal 결과에 명시적으로 확인됨\n"
            "- 약한 누락이나 경미한 불일치는 있으나, 필요한 사실 정보 정리에는 문제가 없음\n"
            "\n"
            "[판단 기준]\n"
            "1) internal 정보만으로 필요한 사실 정보를 직접적으로 정리할 수 있으면 use_internal=true\n"
            "2) 약한 불확실성은 confidence 를 낮추되 external 을 강제하지 말 것\n"
            "3) 명확한 충돌, 핵심 엔티티 부재, 핵심 역할 부재, 시점 불일치가 있을 때 need_external=true\n"
            "4) confidence 는 high/medium/low 중 하나\n"
            "5) 정보가 전혀 없거나, 있어도 질문에 대한 확정적 답을 만들 수 없으면 use_internal=false\n"
            "\n"
            "[출력 규칙]\n"
            "반드시 JSON만 출력한다.\n"
            '형식: {"use_internal":true,"need_external":false,"confidence":"high","reason":"짧은 설명"}\n'
            "reason 에는 오직 정보 sufficiency 이유만 써라.\n"
            "reason 에는 행위 가능성이나 시스템 동작과 관련된 표현을 쓰지 말 것.\n"
        )

        user_prompt = (
            f"[사용자 요청]\n{user_message}\n\n"
            "[평가 지시]\n"
            "위 사용자 요청에 메일 보내기/실행 요청이 포함되어 있어도 무시하고,\n"
            f"[internal context]\n{context_text or '(empty)'}\n\n"
            "[internal sources]\n" + ("\n".join(source_lines) if source_lines else "(no sources)")
        )

        try:
            llm = self._build_llm()
            resp = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            raw = normalize_stream_chunk_content(
                getattr(resp, "content", ""),
                text_normalizer=self.ollama.normalize_text_content,
            )
            data = safe_json_loads(raw) or {}

            use_internal = bool(data.get("use_internal"))
            need_external = bool(data.get("need_external"))
            confidence = str(data.get("confidence") or "low").strip().lower()
            reason = str(data.get("reason") or "").strip()

            if confidence not in {"high", "medium", "low"}:
                confidence = "low"

            lowered_reason = reason.lower()
            contaminated = any(
                token in lowered_reason
                for token in [
                    "send",
                    "action",
                    "capability",
                    "tool",
                    "execution",
                    "email",
                    "mail",
                ]
            )

            if contaminated:
                return {
                    "use_internal": False,
                    "need_external": True,
                    "confidence": "low",
                    "reason": "reason 형식이 오염됨",
                }

            if use_internal and need_external:
                if confidence in {"high", "medium"}:
                    need_external = False
                else:
                    use_internal = False
                    need_external = True
                    confidence = "low"

            if not use_internal and not need_external:
                need_external = True
                confidence = "low"
                reason = reason or "judge 결과가 불충분함"

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

    # ---------------------------------------------------------------------
    # raw merge helpers
    # ---------------------------------------------------------------------

    def _collect_valid_pairs(
        self,
        *,
        query_to_result: List[Tuple[str, Dict[str, Any]]],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        valid_pairs: List[Tuple[str, Dict[str, Any]]] = []

        for query_key, result_data in query_to_result:
            if not isinstance(result_data, dict):
                continue
            valid_pairs.append((normalize_whitespace(query_key), result_data))

        return valid_pairs

    def _build_internal_source_key(self, src: Dict[str, Any]) -> str:
        src_id = normalize_whitespace(src.get("id") or "")
        doc_id = normalize_whitespace(src.get("doc_id") or "")
        chunk_id = normalize_whitespace(src.get("chunk_id") or "")
        title = normalize_whitespace(src.get("title") or "")
        url = normalize_whitespace(src.get("url") or src.get("link") or "")

        # 텍스트 일부 겹침으로 dedupe하지 않음
        if src_id:
            return f"id::{src_id.lower()}"

        if doc_id and chunk_id:
            return f"doc_chunk::{doc_id.lower()}::{chunk_id.lower()}"

        if doc_id and title and url:
            return f"doc_title_url::{doc_id.lower()}::{title.lower()}::{url.lower()}"

        if title and url:
            return f"title_url::{title.lower()}::{url.lower()}"

        return ""

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _sort_sources_by_existing_score(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            [dict(x) for x in sources if isinstance(x, dict)],
            key=lambda x: (
                -self._safe_float(x.get("rerank_score"), -1e9),
                self._safe_float(x.get("distance"), 1e9),
            ),
        )

    def compress_query_to_result_preserve_coverage(
        self,
        *,
        query_to_result: List[Tuple[str, Dict[str, Any]]],
        max_total_sources: int = 8,
        preserve_min_per_query: int = 1,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        핵심 목표:
        - query마다 최소 1개 source는 최대한 보존
        - 같은 source(id/doc_id+chunk_id)는 하나만 유지
        - 남는 슬롯만 전역 점수(rerank_score/distance) 기준으로 채움
        - BM25처럼 query 구조를 깨지 않음
        """
        valid_pairs = self._collect_valid_pairs(query_to_result=query_to_result)
        if not valid_pairs:
            return []

        self._debug(
            "compress_query_to_result_preserve_coverage | BEFORE",
            [
                {
                    "query": q,
                    "source_count": len((r or {}).get("sources") or []),
                    "sources": [
                        self._describe_source_for_debug(x)
                        for x in ((r or {}).get("sources") or [])
                        if isinstance(x, dict)
                    ],
                }
                for q, r in valid_pairs
            ],
            max_len=50000,
        )

        key_to_hits: Dict[str, List[Dict[str, Any]]] = {}
        for query_key, result_data in valid_pairs:
            raw_sources = result_data.get("sources") or []
            if not isinstance(raw_sources, list):
                raw_sources = []
            for src in raw_sources:
                if not isinstance(src, dict):
                    continue
                key = self._build_internal_source_key(src) or "(empty-key)"
                key_to_hits.setdefault(key, []).append(
                    {
                        "query": query_key,
                        "title": src.get("title"),
                        "url": src.get("url") or src.get("link"),
                        "text_preview": self._short_text(self._extract_source_text(src), 180),
                    }
                )

        duplicated_before = {k: v for k, v in key_to_hits.items() if len(v) >= 2}
        self._debug(
            "compress_query_to_result_preserve_coverage | DUPLICATES BEFORE",
            duplicated_before if duplicated_before else {"message": "no duplicate keys before compress"},
            max_len=50000,
        )

        safe_total = max(1, int(max_total_sources or 1))
        safe_min_per_query = max(1, int(preserve_min_per_query or 1))

        prepared: List[Dict[str, Any]] = []

        for query_key, result_data in valid_pairs:
            if not isinstance(result_data, dict):
                continue

            raw_sources = result_data.get("sources") or []
            if not isinstance(raw_sources, list):
                raw_sources = []

            deduped_within_query: List[Dict[str, Any]] = []
            local_seen = set()
            local_dropped: List[Dict[str, Any]] = []

            for src in raw_sources:
                if not isinstance(src, dict):
                    continue

                key = self._build_internal_source_key(src)
                if key:
                    if key in local_seen:
                        local_dropped.append(self._describe_source_for_debug(src))
                        continue
                    local_seen.add(key)

                deduped_within_query.append(dict(src))

            ranked_sources = self._sort_sources_by_existing_score(deduped_within_query)

            self._debug(
                f"compress_query_to_result_preserve_coverage | PER QUERY | {query_key}",
                {
                    "raw_count": len(raw_sources),
                    "deduped_count": len(deduped_within_query),
                    "dropped_local_duplicates": local_dropped,
                    "ranked_sources": [self._describe_source_for_debug(x) for x in ranked_sources],
                },
                max_len=50000,
            )

            prepared.append(
                {
                    "query_key": query_key,
                    "result_data": result_data,
                    "sources": ranked_sources,
                }
            )

        if not prepared:
            return valid_pairs

        selected_by_query: Dict[str, List[Dict[str, Any]]] = {
            item["query_key"]: [] for item in prepared
        }
        globally_selected_keys = set()

        # 1) query별 최소 1개 보존
        for item in prepared:
            query_key = item["query_key"]
            sources = item["sources"]

            kept = 0
            for src in sources:
                key = self._build_internal_source_key(src)
                if key and key in globally_selected_keys:
                    continue

                selected_by_query[query_key].append(dict(src))
                if key:
                    globally_selected_keys.add(key)
                kept += 1

                if kept >= safe_min_per_query:
                    break

        # 2) 현재 선택 개수 계산
        total_selected = sum(len(v) for v in selected_by_query.values())
        effective_total_cap = max(safe_total, total_selected)

        # 3) 남은 후보 전역 점수 순으로 수집
        leftovers: List[Tuple[Tuple[float, float], str, Dict[str, Any]]] = []

        for item in prepared:
            query_key = item["query_key"]
            sources = item["sources"]

            already_keys = {
                self._build_internal_source_key(src)
                for src in selected_by_query.get(query_key, [])
                if self._build_internal_source_key(src)
            }

            for src in sources:
                key = self._build_internal_source_key(src)
                if key and key in already_keys:
                    continue

                rank_key = (
                    -self._safe_float(src.get("rerank_score"), -1e9),
                    self._safe_float(src.get("distance"), 1e9),
                )
                leftovers.append((rank_key, query_key, dict(src)))

        leftovers.sort(key=lambda x: x[0])

        # 4) 남은 슬롯만 전역 점수로 채움
        for _, query_key, src in leftovers:
            total_selected = sum(len(v) for v in selected_by_query.values())
            if total_selected >= effective_total_cap:
                break

            key = self._build_internal_source_key(src)
            if key and key in globally_selected_keys:
                continue

            selected_by_query[query_key].append(dict(src))
            if key:
                globally_selected_keys.add(key)

        # 5) query 구조 유지해서 rebuild
        rebuilt: List[Tuple[str, Dict[str, Any]]] = []

        for item in prepared:
            query_key = item["query_key"]
            original_result_data = item["result_data"]
            picked_sources = selected_by_query.get(query_key) or []

            if not picked_sources:
                continue

            new_result = dict(original_result_data)
            new_result["sources"] = picked_sources

            context_blocks: List[str] = []
            for src in picked_sources:
                block = self._extract_source_text_for_display(src)
                if block:
                    context_blocks.append(block)

            new_result["context"] = "\n\n".join(context_blocks).strip()
            rebuilt.append((query_key, new_result))

        self._debug(
            "compress_query_to_result_preserve_coverage | AFTER",
            [
                {
                    "query": q,
                    "source_count": len((r or {}).get("sources") or []),
                    "sources": [
                        self._describe_source_for_debug(x)
                        for x in ((r or {}).get("sources") or [])
                        if isinstance(x, dict)
                    ],
                }
                for q, r in rebuilt
            ],
            max_len=50000,
        )

        return rebuilt or valid_pairs

    def _merge_result_payloads_raw(
        self,
        *,
        query_to_result: List[Tuple[str, Dict[str, Any]]],
        prefer_longer_context: bool = False,
        stitch_sources: bool = False,
    ) -> Dict[str, Any]:
        merged_context_blocks: List[str] = []
        merged_sources: List[Dict[str, Any]] = []

        valid_pairs = self._collect_valid_pairs(query_to_result=query_to_result)

        self._debug(
            "_merge_result_payloads_raw | INPUT",
            [
                {
                    "query": q,
                    "source_count": len((r or {}).get("sources") or []),
                    "sources": [
                        self._describe_source_for_debug(x)
                        for x in ((r or {}).get("sources") or [])
                        if isinstance(x, dict)
                    ],
                }
                for q, r in valid_pairs
            ],
            max_len=50000,
        )

        seen_source_keys = set()
        dropped_global_duplicates: List[Dict[str, Any]] = []

        for query_key, result_data in valid_pairs:
            raw_sources = result_data.get("sources") or []
            if not isinstance(raw_sources, list):
                raw_sources = []

            unique_sources_for_query: List[Dict[str, Any]] = []

            for src in raw_sources:
                if not isinstance(src, dict):
                    continue

                source_key = self._build_internal_source_key(src)

                # key가 있는 internal source만 보수적으로 dedupe
                if source_key:
                    if source_key in seen_source_keys:
                        dropped = self._describe_source_for_debug(src)
                        dropped["query"] = query_key
                        dropped_global_duplicates.append(dropped)
                        continue
                    seen_source_keys.add(source_key)

                enriched = dict(src)
                enriched["query"] = query_key
                unique_sources_for_query.append(enriched)

            self._debug(
                f"_merge_result_payloads_raw | PER QUERY | {query_key}",
                {
                    "input_count": len(raw_sources),
                    "kept_count": len(unique_sources_for_query),
                    "kept_sources": [self._describe_source_for_debug(x) for x in unique_sources_for_query],
                },
                max_len=50000,
            )

            if not unique_sources_for_query:
                continue

            merged_sources.extend(unique_sources_for_query)

            rebuilt_context_blocks: List[str] = []
            for src in unique_sources_for_query:
                block = self._extract_source_text_for_display(src)
                if block:
                    rebuilt_context_blocks.append(block)

            rebuilt_context = "\n\n".join(rebuilt_context_blocks).strip()
            if rebuilt_context:
                merged_context_blocks.append(rebuilt_context)

        self._debug(
            "_merge_result_payloads_raw | DROPPED GLOBAL DUPLICATES",
            dropped_global_duplicates,
            max_len=50000,
        )

        merged_sources = self.sources.dedupe_sources(merged_sources)

        if stitch_sources:
            merged_sources = self.sources.stitch_adjacent_sources(merged_sources)

        if prefer_longer_context:
            merged_context_blocks = sorted(merged_context_blocks, key=len, reverse=True)

        if stitch_sources and merged_sources:
            stitched_blocks: List[str] = []

            for src in merged_sources:
                content = self._extract_source_text_for_display(src)
                if not content:
                    continue

                stitched_blocks.append(content)

            if stitched_blocks:
                merged_context_blocks = stitched_blocks

        final_context = "\n\n".join(merged_context_blocks).strip()

        print("\n" + "#" * 140)
        print("[_merge_result_payloads_raw] FINAL RAW MERGED CONTEXT")
        print("#" * 140)
        print(final_context)
        print("#" * 140 + "\n")

        self._debug(
            "_merge_result_payloads_raw | FINAL",
            {
                "merged_source_count": len(merged_sources),
                "merged_sources": [self._describe_source_for_debug(x) for x in merged_sources],
                "final_context_preview": self._short_text(final_context, 2000),
            },
            max_len=50000,
        )

        return {
            "context": final_context,
            "sources": merged_sources,
        }

    def merge_result_payloads_for_judge(
        self,
        *,
        query_to_result: List[Tuple[str, Dict[str, Any]]],
        prefer_longer_context: bool = False,
        stitch_sources: bool = False,
    ) -> Dict[str, Any]:
        """
        judge 전용 merge.
        절대 summary를 쓰지 않는다.
        절대 final answer rule을 붙이지 않는다.
        """
        return self._merge_result_payloads_raw(
            query_to_result=query_to_result,
            prefer_longer_context=prefer_longer_context,
            stitch_sources=stitch_sources,
        )

    def build_judge_ready_internal_result(
        self,
        *,
        query_to_result: List[Tuple[str, Dict[str, Any]]],
        prefer_longer_context: bool = False,
        stitch_sources: bool = False,
    ) -> Dict[str, Any]:
        """
        judge에 넣기 위한 안전한 raw internal_result 생성 entrypoint.
        external fallback 여부를 판정하기 전에는 이 결과만 사용해야 한다.
        """
        return self.merge_result_payloads_for_judge(
            query_to_result=query_to_result,
            prefer_longer_context=prefer_longer_context,
            stitch_sources=stitch_sources,
        )

    def merge_result_payloads(
        self,
        *,
        query_to_result: List[Tuple[str, Dict[str, Any]]],
        original_user_message: str = "",
        prefer_longer_context: bool = False,
        stitch_sources: bool = False,
        summarize_multi_query: bool = True,
        judge_result: Dict[str, Any] | None = None,
        force_no_summary: bool = False,
        include_final_answer_rule: bool = True,
    ) -> Dict[str, Any]:
        """
        여러 query 결과를 병합한다.

        핵심 규칙:
        - judge 전에 이 함수를 summary 모드로 쓰지 말 것
        - judge_result.need_external=True 이면 자동으로 no-summary
        - external fallback이면 raw context/source 기반으로만 병합
        - query가 1개 이하이면 summary를 타지 않고 바로 raw merge
        """
        valid_pairs = self._collect_valid_pairs(query_to_result=query_to_result)

        self._debug(
            "merge_result_payloads | INPUT",
            [
                {
                    "query": q,
                    "source_count": len((r or {}).get("sources") or []),
                    "sources": [
                        self._describe_source_for_debug(x)
                        for x in ((r or {}).get("sources") or [])
                        if isinstance(x, dict)
                    ],
                }
                for q, r in valid_pairs
            ],
            max_len=50000,
        )

        if len(valid_pairs) <= 1:
            summarize_multi_query = False

        if force_no_summary:
            summarize_multi_query = False

        if isinstance(judge_result, dict):
            if bool(judge_result.get("need_external")):
                summarize_multi_query = False

        if not summarize_multi_query:
            base_result = self._merge_result_payloads_raw(
                query_to_result=valid_pairs,
                prefer_longer_context=prefer_longer_context,
                stitch_sources=stitch_sources,
            )
            final_context = base_result.get("context") or ""

            if include_final_answer_rule:
                final_answer_instruction = (
                    "[FINAL ANSWER RULE]\n"
                    "최종 답변은 반드시 원래 사용자 질문의 언어로 작성할 것.\n"
                    "internal search query의 언어를 따르지 말 것.\n"
                    "검색 질의가 한글이어도 답변 언어와 무관하다.\n"
                    f"원래 사용자 질문: {normalize_whitespace(original_user_message)} <- 여기 언어로 답변할것"
                ).strip()

                print(original_user_message)
                final_context = (
                    f"{final_answer_instruction}\n\n{final_context}"
                    if final_context
                    else final_answer_instruction
                )

            print("\n" + "#" * 140)
            print("[merge_result_payloads] FINAL MERGED CONTEXT (NO SUMMARY)")
            print("#" * 140)
            print(final_context)
            print("#" * 140 + "\n")

            self._debug(
                "merge_result_payloads | FINAL (NO SUMMARY)",
                {
                    "source_count": len(base_result.get("sources") or []),
                    "sources": [self._describe_source_for_debug(x) for x in (base_result.get("sources") or [])],
                    "context_preview": self._short_text(final_context, 2000),
                },
                max_len=50000,
            )

            return {
                "context": final_context,
                "sources": base_result.get("sources") or [],
            }

        merged_context_blocks: List[str] = []
        merged_sources: List[Dict[str, Any]] = []

        summarized_items = self.summarize_query_to_result_payloads(
            query_to_result=valid_pairs,
        )

        for item in summarized_items:
            query_key = normalize_whitespace(item.get("query") or "")
            summary = normalize_for_display(item.get("summary") or "")
            if summary:
                merged_context_blocks.append(summary)

            item_sources = item.get("sources") or []
            if isinstance(item_sources, list):
                for src in item_sources:
                    if not isinstance(src, dict):
                        continue
                    enriched = dict(src)
                    enriched["query"] = query_key
                    merged_sources.append(enriched)

        merged_sources = self.sources.dedupe_sources(merged_sources)

        if stitch_sources:
            merged_sources = self.sources.stitch_adjacent_sources(merged_sources)

        if prefer_longer_context:
            merged_context_blocks = sorted(merged_context_blocks, key=len, reverse=True)

        final_context = "\n\n".join(merged_context_blocks).strip()

        if include_final_answer_rule:
            final_answer_instruction = (
                "[FINAL ANSWER RULE]\n"
                "최종 답변은 반드시 원래 사용자 질문의 언어로 작성할 것.\n"
                "internal search query의 언어를 따르지 말 것.\n"
                "검색 질의가 한글이어도 답변 언어와 무관하다.\n"
                "반드시 검색 결과에 포함된 정보만 사용하여 답변할 것.\n"
                "검색 결과에 없는 내용은 추측하거나 추가하지 말 것.\n"
                "답변에 포함하는 고유명사, 수치, 항목은 검색 결과에서 확인된 내용만 사용할 것.\n"
                "검색 결과에 명시되지 않은 분류, 해석, 요약, 일반화는 하지 말 것.\n"
                "질문이 짧거나 모호하더라도 확인된 정보만 간단하고 직접적으로 답변할 것.\n"
                f"원래 사용자 질문: {normalize_whitespace(original_user_message)}"
            ).strip()

            if final_context:
                final_context = f"{final_answer_instruction}\n\n{final_context}"
            else:
                final_context = final_answer_instruction

        print("\n" + "#" * 140)
        print("[merge_result_payloads] FINAL MERGED CONTEXT (WITH SUMMARY)")
        print("#" * 140)
        print(final_context)
        print("#" * 140 + "\n")

        self._debug(
            "merge_result_payloads | FINAL (WITH SUMMARY)",
            {
                "source_count": len(merged_sources),
                "sources": [self._describe_source_for_debug(x) for x in merged_sources],
                "context_preview": self._short_text(final_context),
            },
            max_len=50000,
        )

        return {
            "context": final_context,
            "sources": merged_sources,
        }

    # ---------------------------------------------------------------------
    # helper for service.run
    # ---------------------------------------------------------------------

    def build_internal_execution_plan(
        self,
        *,
        original_user_message: str,
        max_atomic_queries: int = 4,
        max_queries_per_atomic: int = 5,
    ) -> List[Dict[str, Any]]:
        atomic_queries = self.decompose_user_message_into_atomic_queries(
            user_message=original_user_message,
            max_atomic_queries=max_atomic_queries,
        )
        if not atomic_queries:
            atomic_queries = [normalize_whitespace(original_user_message)]

        return self.build_internal_search_queries(
            atomic_queries=atomic_queries,
            original_user_message=original_user_message,
            max_queries_per_atomic=max_queries_per_atomic,
        )

    def debug_dump_internal_plan(
        self,
        *,
        original_user_message: str,
        max_atomic_queries: int = 5,
        max_queries_per_atomic: int = 5,
    ) -> str:
        structure = self.classify_user_message_structure(
            user_message=original_user_message,
            max_atomic_queries=max_atomic_queries,
        )
        plan = self.build_internal_execution_plan(
            original_user_message=original_user_message,
            max_atomic_queries=max_atomic_queries,
            max_queries_per_atomic=max_queries_per_atomic,
        )
        return json.dumps(
            {
                "original_user_message": original_user_message,
                "structure": structure,
                "plan": plan,
            },
            ensure_ascii=False,
            indent=2,
        )