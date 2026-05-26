from __future__ import annotations

import json
from typing import Any, Dict, Generator, List, Optional, Tuple

from .agent_speed_utils import normalize_whitespace
from .execution import execute_tool


class RetrievalFlow:
    """
    retrieval orchestration 전용 모듈

    책임
    - research_tasks -> atomic_queries 변환
    - internal retrieval 실행/판정
    - external retrieval 실행
    - fetch/extract/BFS/build_context
    - internal/external 결과 merge
    """

    def __init__(self, *, tools, internal, external, sources, config):
        self.tools = tools
        self.internal = internal
        self.external = external
        self.sources = sources
        self.config = config

    # ---------------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------------
    def run(
        self,
        *,
        original_user_message: str,
        collection_name: Optional[str],
        plan: Dict[str, Any],
        collections: List[Dict[str, Any]],
    ) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        fallback_collection = (collection_name or self.config.default_collection or "").strip()

        atomic_queries = self._prepare_atomic_queries(
            original_user_message=original_user_message,
            plan=plan,
        )

        (
            raw_internal_result,
            realized_atomic_queries,
            internal_query_results,
        ) = yield from self._run_internal_phase(
            original_user_message=original_user_message,
            atomic_queries=atomic_queries,
            plan=plan,
            collections=collections,
            fallback_collection=fallback_collection,
        )

        need_external = yield from self._decide_external_needed(
            original_user_message=original_user_message,
            plan=plan,
            internal_result=raw_internal_result,
        )

        internal_context_len = len(str(raw_internal_result.get("context") or "").strip())
        internal_sources_len = len(raw_internal_result.get("sources") or [])

        if not need_external:
            if not internal_query_results:
                return raw_internal_result

            structure = self.internal.classify_user_message_structure(
                user_message=original_user_message,
                max_atomic_queries=4,
            )

            final_internal_result = self.internal.merge_result_payloads(
                query_to_result=internal_query_results,
                original_user_message=original_user_message,
                prefer_longer_context=True,
                stitch_sources=True,
                summarize_multi_query=False,
            )
            return final_internal_result

        (
            extracted_documents,
            adopted_queries_by_atomic,
        ) = yield from self._run_external_document_phase(
            original_user_message=original_user_message,
            realized_atomic_queries=realized_atomic_queries,
            internal_result=raw_internal_result,
        )

        if not extracted_documents:
            if internal_context_len > 0 or internal_sources_len > 0:
                return raw_internal_result
            return {}

        external_result = yield from self._run_build_context_phase(
            original_user_message=original_user_message,
            realized_atomic_queries=realized_atomic_queries,
            adopted_queries_by_atomic=adopted_queries_by_atomic,
            extracted_documents=extracted_documents,
            internal_result=raw_internal_result,
        )

        if not external_result:
            if internal_context_len > 0 or internal_sources_len > 0:
                return raw_internal_result
            return {}

        return external_result

    # ---------------------------------------------------------------------
    # atomic query preparation
    # ---------------------------------------------------------------------

    def _prepare_atomic_queries(
        self,
        *,
        original_user_message: str,
        plan: Dict[str, Any],
    ) -> List[str]:
        normalized_original = normalize_whitespace(original_user_message)
        if not normalized_original:
            return []

        structure = self.internal.classify_user_message_structure(
            user_message=normalized_original,
            max_atomic_queries=4,
        )

        if not structure.get("is_multi"):
            return [normalized_original]

        research_tasks = plan.get("research_tasks") or []
        atomic_queries: List[str] = []

        for item in research_tasks:
            if not isinstance(item, dict):
                continue

            topic = normalize_whitespace(str(item.get("topic") or ""))
            goal = normalize_whitespace(str(item.get("goal") or ""))

            if topic:
                atomic_queries.append(topic)
            elif goal:
                atomic_queries.append(goal)

        atomic_queries = self._dedupe_strings(atomic_queries)

        if atomic_queries:
            return atomic_queries[:4]

        decomposed = self.internal.decompose_user_message_into_atomic_queries(
            user_message=normalized_original,
            max_atomic_queries=4,
        )
        if decomposed:
            return decomposed

        return [normalized_original]

    # ---------------------------------------------------------------------
    # collection resolution
    # ---------------------------------------------------------------------

    def _resolve_planned_collections(
        self,
        *,
        planned_candidates: List[str],
        collections: List[Dict[str, Any]],
        fallback_selected: List[str],
    ) -> List[str]:
        available_names = {
            str(item.get("name") or "").strip()
            for item in collections
            if isinstance(item, dict)
        }

        picked = [name for name in planned_candidates if name in available_names]
        if picked:
            return picked

        return fallback_selected

    # ---------------------------------------------------------------------
    # internal phase
    # ---------------------------------------------------------------------
    def _run_internal_phase(
        self,
        *,
        original_user_message: str,
        atomic_queries: List[str],
        plan: Dict[str, Any],
        collections: List[Dict[str, Any]],
        fallback_collection: str,
    ) -> Generator[
        Dict[str, Any],
        None,
        Tuple[Dict[str, Any], List[str], List[Tuple[str, Dict[str, Any]]]],
    ]:
        structure = self.internal.classify_user_message_structure(
            user_message=original_user_message,
            max_atomic_queries=4,
        )

        yield {
            "type": "thought",
            "step": 1,
            "delta": (
                "[검색 흐름] 질문 구조 분석 완료 | "
                f"질문 수={'복수' if structure.get('is_multi') else '단일'} | "
                f"예상 atomic 수={int(structure.get('num_queries') or 1)} | "
                f"설명={str(structure.get('reason') or '').strip()}"
            ),
        }

        max_atomic_queries = 1
        if structure.get("is_multi"):
            try:
                max_atomic_queries = max(1, min(int(structure.get("num_queries") or 2), 4))
            except Exception:
                max_atomic_queries = 2

        internal_plan = self.internal.build_internal_execution_plan(
            original_user_message=original_user_message,
            max_atomic_queries=max_atomic_queries,
            max_queries_per_atomic=2,
        )

        if not internal_plan:
            internal_plan = [{"atomic_query": q, "internal_queries": [q]} for q in atomic_queries]

        yield {
            "type": "thought",
            "step": 1,
            "delta": "[검색 흐름] 내부 검색 계획=" + json.dumps(internal_plan, ensure_ascii=False),
        }

        if not bool(plan.get("use_internal_first")):
            yield {
                "type": "thought",
                "step": 1,
                "delta": "[검색 흐름] 설정상 내부 검색 우선 사용이 꺼져 있어 바로 다음 단계로 이동",
            }
            return {}, atomic_queries[:], []

        internal_query_results: List[Tuple[str, Dict[str, Any]]] = []
        realized_atomic_queries: List[str] = []

        for idx, item in enumerate(internal_plan, start=1):
            atomic_query = normalize_whitespace(str(item.get("atomic_query") or ""))
            internal_queries = item.get("internal_queries") or []

            if not atomic_query:
                continue

            realized_atomic_queries.append(atomic_query)

            fallback_selected = self.internal.select_collections_for_query(
                user_message=atomic_query,
                collections=collections,
                fallback_collection=fallback_collection,
            )

            selected_collections = self._resolve_planned_collections(
                planned_candidates=self._normalize_string_list(plan.get("collection_candidates")),
                collections=collections,
                fallback_selected=fallback_selected,
            )

            yield {
                "type": "thought",
                "step": 1,
                "delta": (
                    f"[내부 검색 {idx}/{len(internal_plan)}] "
                    f"질문={json.dumps(atomic_query, ensure_ascii=False)} | "
                    f"검색어 후보={json.dumps(internal_queries, ensure_ascii=False)} | "
                    f"대상 컬렉션={json.dumps(selected_collections, ensure_ascii=False)}"
                ),
            }

            if not selected_collections:
                yield {
                    "type": "thought",
                    "step": 1,
                    "delta": f"[내부 검색 {idx}/{len(internal_plan)}] 사용할 컬렉션이 없어 건너뜀",
                }
                continue

            for iq_idx, internal_query in enumerate(internal_queries, start=1):
                internal_query = normalize_whitespace(str(internal_query or ""))
                if not internal_query:
                    continue

                yield {
                    "type": "thought",
                    "step": 1,
                    "delta": (
                        f"[내부 검색 실행 {idx}.{iq_idx}/{len(internal_queries)}] "
                        f"검색어={json.dumps(internal_query, ensure_ascii=False)}"
                    ),
                }

                internal_result = yield from execute_tool(
                    tools=self.tools,
                    tool_name="internal_search",
                    arguments={
                        "query": internal_query,
                        "collections": selected_collections,
                        "k": 3,
                    },
                    step=1,
                    messages=[],
                    user_message=original_user_message,
                )

                if not internal_result.ok or not isinstance(internal_result.data, dict):
                    yield {
                        "type": "thought",
                        "step": 1,
                        "delta": (
                            f"[내부 검색 결과 {idx}.{iq_idx}/{len(internal_queries)}] "
                            "검색 실패 또는 비정상 응답"
                        ),
                    }
                    continue

                enriched_result = dict(internal_result.data)
                enriched_sources: List[Dict[str, Any]] = []

                raw_sources = enriched_result.get("sources") or []
                if isinstance(raw_sources, list):
                    for src in raw_sources:
                        if not isinstance(src, dict):
                            continue
                        s = dict(src)
                        s["atomic_query"] = atomic_query
                        s["internal_query"] = internal_query
                        enriched_sources.append(s)

                enriched_result["sources"] = enriched_sources

                context_text = str(enriched_result.get("context") or "").strip()
                yield {
                    "type": "thought",
                    "step": 1,
                    "delta": (
                        f"[내부 검색 결과 {idx}.{iq_idx}/{len(internal_queries)}] "
                        f"context 길이={len(context_text)} | "
                        f"source 수={len(enriched_sources)}"
                    ),
                }

                query_key = f"atomic={atomic_query} | internal={internal_query}"
                internal_query_results.append((query_key, enriched_result))

        if not realized_atomic_queries:
            realized_atomic_queries = atomic_queries[:]

        if not realized_atomic_queries:
            realized_atomic_queries = atomic_queries[:]

        # 핵심 변경:
        # BM25 전역 top-k 제거
        # 각 query마다 최소 1개는 보존하고, 남은 슬롯만 전역 점수로 채움
        # 핵심 변경:
        # query coverage 보존 대신, 전체 internal source를 전역 rerank 해서 top-3만 남김
        if internal_query_results:
            before_queries = len(internal_query_results)
            before_sources = 0
            for _, result_data in internal_query_results:
                if isinstance(result_data, dict):
                    srcs = result_data.get("sources") or []
                    if isinstance(srcs, list):
                        before_sources += len(srcs)

            yield {
                "type": "thought",
                "step": 1,
                "delta": (
                    "[내부 검색 후처리] 전역 rerank top-3 수행 | "
                    f"before_queries={before_queries} | before_sources={before_sources}"
                ),
            }

            internal_query_results = self.sources.rerank_sources_with_bm25(
                user_query=original_user_message,
                query_to_result=internal_query_results,
                top_k=3,
            )

            after_queries = len(internal_query_results)
            after_sources = 0
            for _, result_data in internal_query_results:
                if isinstance(result_data, dict):
                    srcs = result_data.get("sources") or []
                    if isinstance(srcs, list):
                        after_sources += len(srcs)

            yield {
                "type": "thought",
                "step": 1,
                "delta": (
                    "[내부 검색 후처리] 전역 rerank top-3 완료 | "
                    f"after_queries={after_queries} | after_sources={after_sources}"
                ),
            }

        raw_internal = self.internal.build_judge_ready_internal_result(
            query_to_result=internal_query_results,
            prefer_longer_context=True,
            stitch_sources=False,
        )

        merged_internal_context_len = len(str(raw_internal.get("context") or "").strip())
        merged_internal_sources_len = len(raw_internal.get("sources") or [])

        yield {
            "type": "thought",
            "step": 1,
            "delta": (
                "[내부 검색 병합 완료] "
                f"context 길이={merged_internal_context_len} | "
                f"source 수={merged_internal_sources_len}"
            ),
        }

        return raw_internal, realized_atomic_queries, internal_query_results

    def _decide_external_needed(
        self,
        *,
        original_user_message: str,
        plan: Dict[str, Any],
        internal_result: Dict[str, Any],
    ) -> Generator[Dict[str, Any], None, bool]:
        plan_wants_external = bool(plan.get("use_external_search"))

        internal_context = str(internal_result.get("context") or "").strip()
        internal_sources = internal_result.get("sources") or []
        internal_has_result = bool(internal_context) or bool(internal_sources)

        if not internal_has_result:
            yield {
                "type": "thought",
                "step": 1,
                "delta": (
                    "[판단] 내부 검색 결과가 없어 외부 검색으로 바로 전환 | "
                    "use_internal=False | need_external=True | confidence=low"
                ),
            }
            return True

        judge = self.internal.judge_internal_answerability(
            user_message=original_user_message,
            internal_result=internal_result,
        )

        use_internal = bool(judge.get("use_internal"))
        need_external_by_judge = bool(judge.get("need_external"))
        confidence = str(judge.get("confidence") or "").strip().lower()
        reason = str(judge.get("reason") or "").strip()

        yield {
            "type": "thought",
            "step": 1,
            "delta": (
                "[판단] 내부 정보만으로 답변 가능한지 평가 | "
                f"use_internal={use_internal} | "
                f"need_external={need_external_by_judge} | "
                f"confidence={confidence or 'low'} | "
                f"이유={reason}"
            ),
        }

        if need_external_by_judge:
            yield {
                "type": "thought",
                "step": 1,
                "delta": "[판단] 외부 검색이 필요하다고 판단되어 외부 검색 진행",
            }
            return True

        if use_internal and not need_external_by_judge and confidence in {"high", "medium"}:
            yield {
                "type": "thought",
                "step": 1,
                "delta": "[판단] 내부 정보만으로 충분하여 외부 검색 없이 진행",
            }
            return False

        if internal_context or internal_sources:
            yield {
                "type": "thought",
                "step": 1,
                "delta": "[판단] 내부 정보는 있으나 충분하지 않아 외부 검색으로 보강",
            }
            return True

        return plan_wants_external

    # ---------------------------------------------------------------------
    # external document phase
    # ---------------------------------------------------------------------
    def _run_external_document_phase(
        self,
        *,
        original_user_message: str,
        realized_atomic_queries: List[str],
        internal_result: Dict[str, Any],
    ) -> Generator[Dict[str, Any], None, Tuple[List[Dict[str, Any]], Dict[str, str]]]:
        all_collected_search_items: List[List[Dict[str, Any]]] = []
        adopted_queries_by_atomic: Dict[str, str] = {}
        search_top_k = 10

        for atomic_idx, atomic_query in enumerate(realized_atomic_queries, start=1):
            rewritten_queries = self.external.build_search_queries_with_llm(
                user_message=atomic_query,
                max_queries=3,
            )
            if not rewritten_queries:
                rewritten_queries = [atomic_query]

            yield {
                "type": "thought",
                "step": 2,
                "delta": (
                    f"[외부 검색 준비 {atomic_idx}/{len(realized_atomic_queries)}] "
                    f"질문={json.dumps(atomic_query, ensure_ascii=False)} | "
                    f"검색어 후보={json.dumps(rewritten_queries, ensure_ascii=False)}"
                ),
            }

            successful_queries: List[str] = []
            collected_for_atomic: List[List[Dict[str, Any]]] = []

            for ridx, rewritten_query in enumerate(rewritten_queries, start=1):
                yield {
                    "type": "thought",
                    "step": 2,
                    "delta": (
                        f"[외부 검색 실행 {atomic_idx}.{ridx}/{len(rewritten_queries)}] "
                        f"검색어={json.dumps(rewritten_query, ensure_ascii=False)}"
                    ),
                }

                search_result = yield from execute_tool(
                    tools=self.tools,
                    tool_name="external_search",
                    arguments={
                        "query": rewritten_query,
                        "num": 10,
                        "top_k_urls": search_top_k,
                    },
                    step=2,
                    messages=[],
                    user_message=original_user_message,
                )

                if not search_result.ok or not isinstance(search_result.data, dict):
                    continue

                if not self.external.has_meaningful_search_items(search_result.data):
                    continue

                successful_queries.append(rewritten_query)

                collected_items = self.sources.collect_search_items_from_result(search_result.data)
                if collected_items:
                    collected_for_atomic.append(collected_items)

                yield {
                    "type": "thought",
                    "step": 2,
                    "delta": (
                        f"[외부 검색 채택 {atomic_idx}.{ridx}/{len(rewritten_queries)}] "
                        f"질문={json.dumps(atomic_query, ensure_ascii=False)} | "
                        f"채택 검색어={json.dumps(rewritten_query, ensure_ascii=False)} | "
                        f"수집 URL 수={len(collected_items)}"
                    ),
                }

            if not collected_for_atomic:
                yield {
                    "type": "thought",
                    "step": 2,
                    "delta": f"[외부 검색 실패 {atomic_idx}/{len(realized_atomic_queries)}] 질문={atomic_query}",
                }
                continue

            adopted_queries_by_atomic[atomic_query] = " | ".join(successful_queries)
            all_collected_search_items.extend(collected_for_atomic)

        merged_search_items = self.sources.merge_search_items(all_collected_search_items)
        if not merged_search_items:
            yield {
                "type": "thought",
                "step": 2,
                "delta": "[외부 검색] 수집된 검색 결과 URL이 없어 종료",
            }
            return [], adopted_queries_by_atomic

        llm_cap = min(search_top_k, len(merged_search_items))
        picked_items = self.external.select_urls_with_llm(
            user_message=original_user_message,
            items=merged_search_items,
            max_items_cap=llm_cap,
        )

        if not picked_items:
            yield {
                "type": "thought",
                "step": 2,
                "delta": "[외부 검색] 최종 선택된 URL이 없어 종료",
            }
            return [], adopted_queries_by_atomic

        selected_urls = self._format_selected_urls_for_log(picked_items)
        selected_url_text = " | ".join(selected_urls) if selected_urls else "-"

        yield {
            "type": "thought",
            "step": 2,
            "delta": (
                "[외부 검색] 최종 선택 URL 완료 | "
                f"선택 수={len(picked_items)} | "
                f"전체 후보 수={len(merged_search_items)} | "
                f"선택 URL={selected_url_text}"
            ),
        }

        raw_documents = yield from self._fetch_raw_documents(
            original_user_message=original_user_message,
            picked_items=picked_items,
        )
        if not raw_documents:
            return [], adopted_queries_by_atomic

        extracted_documents = yield from self._extract_and_expand_documents(
            original_user_message=original_user_message,
            raw_documents=raw_documents,
        )
        return extracted_documents, adopted_queries_by_atomic

    def _fetch_raw_documents(
        self,
        *,
        original_user_message: str,
        picked_items: List[Dict[str, Any]],
    ) -> Generator[Dict[str, Any], None, List[Dict[str, Any]]]:
        selected_urls = self._format_selected_urls_for_log(picked_items)
        selected_url_text = " | ".join(selected_urls) if selected_urls else "-"

        yield {
            "type": "thought",
            "step": 3,
            "delta": (
                "[문서 수집] 선택된 URL fetch 시작 | "
                f"개수={len(picked_items)} | "
                f"URL={selected_url_text}"
            ),
        }

        fetch_result = yield from execute_tool(
            tools=self.tools,
            tool_name="external_fetch_raw",
            arguments={
                "query": original_user_message,
                "items": picked_items,
                "max_fetch": len(picked_items),
            },
            step=3,
            messages=[],
            user_message=original_user_message,
        )

        if not fetch_result.ok or not isinstance(fetch_result.data, dict):
            yield {
                "type": "thought",
                "step": 3,
                "delta": "[문서 수집] fetch 실패 또는 비정상 응답",
            }
            return []

        raw_documents = fetch_result.data.get("documents")
        if not isinstance(raw_documents, list) or not raw_documents:
            yield {
                "type": "thought",
                "step": 3,
                "delta": "[문서 수집] fetch 성공했지만 문서가 비어 있음",
            }
            return []

        yield {
            "type": "thought",
            "step": 3,
            "delta": f"[문서 수집] fetch 완료 | raw 문서 수={len(raw_documents)}",
        }

        return raw_documents

    def _extract_and_expand_documents(
        self,
        *,
        original_user_message: str,
        raw_documents: List[Dict[str, Any]],
    ) -> Generator[Dict[str, Any], None, List[Dict[str, Any]]]:
        yield {
            "type": "thought",
            "step": 4,
            "delta": "[문서 정리] 본문 추출 시작",
        }

        extract_result = yield from execute_tool(
            tools=self.tools,
            tool_name="external_extract_main_content",
            arguments={
                "query": original_user_message,
                "documents": raw_documents,
            },
            step=4,
            messages=[],
            user_message=original_user_message,
        )

        if not extract_result.ok or not isinstance(extract_result.data, dict):
            yield {
                "type": "thought",
                "step": 4,
                "delta": "[문서 정리] 본문 추출 실패 또는 비정상 응답",
            }
            return []

        extracted_documents = extract_result.data.get("documents")
        if not isinstance(extracted_documents, list) or not extracted_documents:
            yield {
                "type": "thought",
                "step": 4,
                "delta": "[문서 정리] 본문 추출 결과가 비어 있음",
            }
            return []

        extracted_documents = self.sources.merge_documents_by_link(extracted_documents)

        yield {
            "type": "thought",
            "step": 4,
            "delta": (
                "[문서 정리] 본문 추출 완료 후 BFS 확장 시작 | "
                f"초기 문서 수={len(extracted_documents)}"
            ),
        }

        bfs_expanded_documents = self.external.ai_guided_bfs_expand(
            user_message=original_user_message,
            seed_documents=extracted_documents,
            max_depth=int(getattr(self.config, "agent_speed_bfs_max_depth", 2) or 2),
            max_total_pages=int(getattr(self.config, "agent_speed_bfs_max_total_pages", 3) or 3),
            max_links_per_page=int(getattr(self.config, "agent_speed_bfs_max_links_per_page", 4) or 4),
        )

        if bfs_expanded_documents:
            extracted_documents = self.sources.merge_documents_by_link(
                extracted_documents + bfs_expanded_documents
            )

        yield {
            "type": "thought",
            "step": 4,
            "delta": (
                "[문서 정리] BFS 확장 완료 | "
                f"추가 문서 수={len(bfs_expanded_documents or [])} | "
                f"최종 문서 수={len(extracted_documents)}"
            ),
        }

        return extracted_documents

    # ---------------------------------------------------------------------
    # build context phase
    # ---------------------------------------------------------------------
    def _run_build_context_phase(
        self,
        *,
        original_user_message: str,
        realized_atomic_queries: List[str],
        adopted_queries_by_atomic: Dict[str, str],
        extracted_documents: List[Dict[str, Any]],
        internal_result: Dict[str, Any],
    ) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        per_atomic_build_results: List[Tuple[str, Dict[str, Any]]] = []

        for idx, atomic_query in enumerate(realized_atomic_queries, start=1):
            ai_search_query = normalize_whitespace(
                adopted_queries_by_atomic.get(atomic_query) or atomic_query
            )

            yield {
                "type": "thought",
                "step": 5,
                "delta": (
                    f"[답변 컨텍스트 구성 {idx}/{len(realized_atomic_queries)}] "
                    f"질문={json.dumps(atomic_query, ensure_ascii=False)} | "
                    f"AI검색질의={json.dumps(ai_search_query, ensure_ascii=False)}"
                ),
            }

            build_result = yield from execute_tool(
                tools=self.tools,
                tool_name="external_build_context",
                arguments={
                    "query": ai_search_query,
                    "original_query": original_user_message,
                    "atomic_query": atomic_query,
                    "documents": extracted_documents,
                    "top_k_chunks": 5,
                },
                step=5,
                messages=[],
                user_message=original_user_message,
            )

            if not build_result.ok or not isinstance(build_result.data, dict):
                continue

            if self.internal.has_meaningful_context(build_result.data):
                per_atomic_build_results.append((atomic_query, build_result.data))

        if not per_atomic_build_results:
            yield {
                "type": "thought",
                "step": 5,
                "delta": "[답변 컨텍스트 구성] atomic 기준 결과가 약해 전체 질문 기준으로 다시 구성",
            }

            fallback_query = ""
            for atomic_query in realized_atomic_queries:
                fallback_query = normalize_whitespace(
                    adopted_queries_by_atomic.get(atomic_query) or ""
                )
                if fallback_query:
                    break

            if not fallback_query:
                fallback_query = normalize_whitespace(original_user_message)

            build_result = yield from execute_tool(
                tools=self.tools,
                tool_name="external_build_context",
                arguments={
                    "query": fallback_query,
                    "original_query": original_user_message,
                    "atomic_query": fallback_query,
                    "documents": extracted_documents,
                    "top_k_chunks": 10,
                },
                step=5,
                messages=[],
                user_message=original_user_message,
            )

            if not build_result.ok or not isinstance(build_result.data, dict):
                return {}

            merged_build = build_result.data
            if isinstance(merged_build, dict):
                merged_build["sources"] = self.sources.stitch_adjacent_sources(
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
            structure = self.internal.classify_user_message_structure(
                user_message=original_user_message,
                max_atomic_queries=4,
            )

            merged_build = self.internal.merge_result_payloads(
                query_to_result=per_atomic_build_results,
                original_user_message=original_user_message,
                prefer_longer_context=False,
                stitch_sources=True,
                summarize_multi_query=False
            )

        context_text = str(merged_build.get("context") or "").strip()
        yield {
            "type": "thought",
            "step": 5,
            "delta": (
                "[답변 컨텍스트 구성 완료] "
                f"context 길이={len(context_text)} | "
                f"source 수={len(merged_build.get('sources') or [])}"
            ),
        }

        return merged_build

    # ---------------------------------------------------------------------
    # merge
    # ---------------------------------------------------------------------

    def _merge_results(
        self,
        *,
        internal_result: Dict[str, Any],
        external_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        internal_context_len = len(str(internal_result.get("context") or "").strip())
        internal_sources_len = len(internal_result.get("sources") or [])

        if internal_context_len > 0 or internal_sources_len > 0:
            return self.internal.merge_result_payloads(
                query_to_result=[
                    ("internal", internal_result),
                    ("external", external_result),
                ],
                prefer_longer_context=False,
                stitch_sources=True,
                summarize_multi_query=False,
            )

        return external_result

    # ---------------------------------------------------------------------
    # small helpers
    # ---------------------------------------------------------------------

    def _normalize_string_list(self, value: Any) -> List[str]:
        if value is None:
            return []

        if isinstance(value, list):
            out: List[str] = []
            for item in value:
                s = str(item).strip()
                if s:
                    out.append(s)
            return out

        s = str(value).strip()
        return [s] if s else []

    def _dedupe_strings(self, items: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for item in items:
            clean = normalize_whitespace(str(item))
            if not clean or clean in seen:
                continue
            seen.add(clean)
            out.append(clean)
        return out

    def _format_selected_urls_for_log(self, items: List[Dict[str, Any]]) -> List[str]:
        formatted: List[str] = []

        for item in items:
            if not isinstance(item, dict):
                continue

            url = normalize_whitespace(
                str(
                    item.get("url")
                    or item.get("link")
                    or item.get("href")
                    or ""
                )
            )

            if not url:
                continue

            formatted.append(url)

        return formatted