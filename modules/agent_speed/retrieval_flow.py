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

    추가 원칙
    - 질문별(atomic query)로 결과를 service 에 즉시 넘길 수 있어야 함
    - 질문별 컬렉션 선택이 전역 planner 후보에 오염되지 않도록 함
    - mixed request 에서 task_type=research 인 항목만 retrieval 대상으로 삼음
    - planner topic 과 internal atomic_query 문자열 불일치로 plan 이 비지 않도록
      research_tasks 기반으로 internal_plan 을 직접 구성함
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

    def get_atomic_queries(
        self,
        *,
        original_user_message: str,
        plan: Dict[str, Any],
    ) -> List[str]:
        return self._prepare_atomic_queries(
            original_user_message=original_user_message,
            plan=plan,
        )

    def run(
        self,
        *,
        original_user_message: str,
        collection_name: Optional[str],
        plan: Dict[str, Any],
        collections: List[Dict[str, Any]],
    ) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        """
        기존 호환용:
        - 모든 atomic query 결과를 모은 뒤
        - 최종 merged result 하나를 반환
        """
        per_atomic_results: List[Tuple[str, Dict[str, Any]]] = []

        for item in self.run_per_atomic_query(
            original_user_message=original_user_message,
            collection_name=collection_name,
            plan=plan,
            collections=collections,
        ):
            if isinstance(item, dict) and str(item.get("type") or "") == "atomic_result":
                atomic_query = normalize_whitespace(str(item.get("atomic_query") or ""))
                result_data = item.get("result_data") or {}
                if atomic_query and isinstance(result_data, dict):
                    per_atomic_results.append((atomic_query, result_data))
                continue

            yield item

        if not per_atomic_results:
            return {}

        structure = self.internal.classify_user_message_structure(
            user_message=original_user_message,
            max_atomic_queries=4,
        )

        merged = self.internal.merge_result_payloads(
            query_to_result=per_atomic_results,
            original_user_message=original_user_message,
            prefer_longer_context=False,
            stitch_sources=True,
            summarize_multi_query=bool(structure.get("is_multi")),
        )
        return merged

    def run_per_atomic_query(
        self,
        *,
        original_user_message: str,
        collection_name: Optional[str],
        plan: Dict[str, Any],
        collections: List[Dict[str, Any]],
    ) -> Generator[Dict[str, Any], None, None]:
        """
        질문별 결과를 즉시 service 로 넘기기 위한 메서드.

        yield 형식:
        - 기존 thought/tool/sources 이벤트는 그대로 전달
        - 질문 하나가 끝날 때마다:
          {
              "type": "atomic_result",
              "atomic_query": "...",
              "result_data": {...}
          }
        """
        fallback_collection = (collection_name or self.config.default_collection or "").strip()

        atomic_queries = self._prepare_atomic_queries(
            original_user_message=original_user_message,
            plan=plan,
        )

        if not atomic_queries:
            return

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

        # 핵심 수정:
        # research_tasks 기반으로 internal_plan 을 직접 구성한다.
        # planner topic 과 internal planner atomic_query 의 문자열 mismatch 를 피한다.
        research_tasks = [
            item
            for item in (plan.get("research_tasks") or [])
            if isinstance(item, dict)
            and normalize_whitespace(str(item.get("task_type") or "research")).lower() == "research"
        ]

        if research_tasks:
            internal_plan: List[Dict[str, Any]] = []

            for item in research_tasks[:max_atomic_queries]:
                topic = normalize_whitespace(str(item.get("topic") or ""))
                if not topic:
                    continue

                generated = self.internal.build_internal_search_queries(
                    atomic_queries=[topic],
                    original_user_message=original_user_message,
                    max_queries_per_atomic=2,
                )

                if generated:
                    internal_plan.extend(generated)
                else:
                    internal_plan.append(
                        {
                            "atomic_query": topic,
                            "internal_queries": [topic],
                        }
                    )
        else:
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

        for idx, item in enumerate(internal_plan, start=1):
            atomic_query = normalize_whitespace(str(item.get("atomic_query") or ""))
            internal_queries = item.get("internal_queries") or []

            if not atomic_query:
                continue

            yield {
                "type": "thought",
                "step": 1,
                "delta": (
                    f"[질문 처리 시작 {idx}/{len(internal_plan)}] "
                    f"질문={json.dumps(atomic_query, ensure_ascii=False)}"
                ),
            }

            raw_internal_result, internal_query_results = yield from self._run_internal_phase_for_atomic(
                original_user_message=original_user_message,
                atomic_query=atomic_query,
                internal_queries=internal_queries,
                plan=plan,
                collections=collections,
                fallback_collection=fallback_collection,
                atomic_index=idx,
                atomic_total=len(internal_plan),
            )

            need_external = yield from self._decide_external_needed(
                original_user_message=original_user_message,
                plan=plan,
                internal_result=raw_internal_result,
            )

            internal_context_len = len(str(raw_internal_result.get("context") or "").strip())
            internal_sources_len = len(raw_internal_result.get("sources") or [])

            if not need_external:
                final_internal_result = raw_internal_result
                if internal_query_results:
                    final_internal_result = self.internal.merge_result_payloads(
                        query_to_result=internal_query_results,
                        original_user_message=original_user_message,
                        prefer_longer_context=True,
                        stitch_sources=True,
                        summarize_multi_query=False,
                    )

                yield {
                    "type": "thought",
                    "step": 1,
                    "delta": (
                        f"[질문 처리 완료 {idx}/{len(internal_plan)}] "
                        f"질문={json.dumps(atomic_query, ensure_ascii=False)} | "
                        "내부 검색 결과 사용"
                    ),
                }

                yield {
                    "type": "atomic_result",
                    "atomic_query": atomic_query,
                    "result_data": final_internal_result,
                }
                continue

            extracted_documents, adopted_queries_by_atomic = yield from self._run_external_document_phase(
                original_user_message=original_user_message,
                realized_atomic_queries=[atomic_query],
                internal_result=raw_internal_result,
            )

            if not extracted_documents:
                fallback_result = raw_internal_result if (internal_context_len > 0 or internal_sources_len > 0) else {}
                yield {
                    "type": "thought",
                    "step": 5,
                    "delta": (
                        f"[질문 처리 완료 {idx}/{len(internal_plan)}] "
                        f"질문={json.dumps(atomic_query, ensure_ascii=False)} | "
                        "외부 검색 성과 부족, 내부 결과 fallback"
                    ),
                }
                yield {
                    "type": "atomic_result",
                    "atomic_query": atomic_query,
                    "result_data": fallback_result,
                }
                continue

            external_result = yield from self._run_build_context_phase(
                original_user_message=original_user_message,
                realized_atomic_queries=[atomic_query],
                adopted_queries_by_atomic=adopted_queries_by_atomic,
                extracted_documents=extracted_documents,
                internal_result=raw_internal_result,
            )

            if not external_result:
                fallback_result = raw_internal_result if (internal_context_len > 0 or internal_sources_len > 0) else {}
                yield {
                    "type": "thought",
                    "step": 5,
                    "delta": (
                        f"[질문 처리 완료 {idx}/{len(internal_plan)}] "
                        f"질문={json.dumps(atomic_query, ensure_ascii=False)} | "
                        "컨텍스트 구성 실패, 내부 결과 fallback"
                    ),
                }
                yield {
                    "type": "atomic_result",
                    "atomic_query": atomic_query,
                    "result_data": fallback_result,
                }
                continue

            merged_result = self._merge_results(
                internal_result=raw_internal_result,
                external_result=external_result,
            )

            yield {
                "type": "thought",
                "step": 5,
                "delta": (
                    f"[질문 처리 완료 {idx}/{len(internal_plan)}] "
                    f"질문={json.dumps(atomic_query, ensure_ascii=False)} | "
                    "외부 보강 결과 사용"
                ),
            }

            yield {
                "type": "atomic_result",
                "atomic_query": atomic_query,
                "result_data": merged_result,
            }

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

        # planner 가 질문별 task 를 이미 분해해 둔 경우 우선 사용
        research_tasks = plan.get("research_tasks") or []
        task_topics: List[str] = []
        for item in research_tasks:
            if not isinstance(item, dict):
                continue

            topic = normalize_whitespace(str(item.get("topic") or ""))
            task_type = normalize_whitespace(str(item.get("task_type") or "research")).lower()

            # retrieval 대상은 research task 만
            if topic and task_type == "research":
                task_topics.append(topic)

        task_topics = self._dedupe_strings(task_topics)
        if task_topics:
            return task_topics[:4]

        structure = self.internal.classify_user_message_structure(
            user_message=normalized_original,
            max_atomic_queries=4,
        )

        if not structure.get("is_multi"):
            return [normalized_original]

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

    def _run_internal_phase_for_atomic(
        self,
        *,
        original_user_message: str,
        atomic_query: str,
        internal_queries: List[str],
        plan: Dict[str, Any],
        collections: List[Dict[str, Any]],
        fallback_collection: str,
        atomic_index: int,
        atomic_total: int,
    ) -> Generator[
        Dict[str, Any],
        None,
        Tuple[Dict[str, Any], List[Tuple[str, Dict[str, Any]]]],
    ]:
        if not bool(plan.get("use_internal_first")):
            yield {
                "type": "thought",
                "step": 1,
                "delta": (
                    f"[내부 검색 {atomic_index}/{atomic_total}] "
                    "설정상 내부 검색 우선 사용이 꺼져 있어 건너뜀"
                ),
            }
            return {}, []

        fallback_selected = self.internal.select_collections_for_query(
            user_message=atomic_query,
            collections=collections,
            fallback_collection=fallback_collection,
        )

        # 질문별 컬렉션을 독립적으로 선택한다.
        selected_collections = fallback_selected

        yield {
            "type": "thought",
            "step": 1,
            "delta": (
                f"[내부 검색 {atomic_index}/{atomic_total}] "
                f"질문={json.dumps(atomic_query, ensure_ascii=False)} | "
                f"검색어 후보={json.dumps(internal_queries, ensure_ascii=False)} | "
                f"대상 컬렉션={json.dumps(selected_collections, ensure_ascii=False)}"
            ),
        }

        if not selected_collections:
            yield {
                "type": "thought",
                "step": 1,
                "delta": (
                    f"[내부 검색 {atomic_index}/{atomic_total}] "
                    "사용할 컬렉션이 없어 건너뜀"
                ),
            }
            return {}, []

        internal_query_results: List[Tuple[str, Dict[str, Any]]] = []

        for iq_idx, internal_query in enumerate(internal_queries, start=1):
            internal_query = normalize_whitespace(str(internal_query or ""))
            if not internal_query:
                continue

            yield {
                "type": "thought",
                "step": 1,
                "delta": (
                    f"[내부 검색 실행 {atomic_index}.{iq_idx}/{len(internal_queries)}] "
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
                        f"[내부 검색 결과 {atomic_index}.{iq_idx}/{len(internal_queries)}] "
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
                    f"[내부 검색 결과 {atomic_index}.{iq_idx}/{len(internal_queries)}] "
                    f"context 길이={len(context_text)} | "
                    f"source 수={len(enriched_sources)}"
                ),
            }

            query_key = f"atomic={atomic_query} | internal={internal_query}"
            internal_query_results.append((query_key, enriched_result))

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
                    "[내부 검색 후처리] 전역 rerank 수행 | "
                    f"before_queries={before_queries} | before_sources={before_sources}"
                ),
            }

            internal_query_results = self.sources.rerank_sources_with_bm25(
                user_query=atomic_query,
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
                    "[내부 검색 후처리] 전역 rerank 완료 | "
                    f"after_queries={after_queries} | after_sources={after_sources}"
                ),
            }

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
                    "[내부 검색 후처리] coverage-preserving dedupe 수행 | "
                    f"before_queries={before_queries} | before_sources={before_sources}"
                ),
            }

            internal_query_results = self.internal.compress_query_to_result_preserve_coverage(
                query_to_result=internal_query_results,
                max_total_sources=3,
                preserve_min_per_query=1,
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
                    "[내부 검색 후처리] coverage-preserving dedupe 완료 | "
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

        return raw_internal, internal_query_results

    # ---------------------------------------------------------------------
    # external need judge
    # ---------------------------------------------------------------------

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

            all_collected_items_for_atomic: List[List[Dict[str, Any]]] = []
            adopted_queries: List[str] = []

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
                    yield {
                        "type": "thought",
                        "step": 2,
                        "delta": (
                            f"[외부 검색 결과 {atomic_idx}.{ridx}/{len(rewritten_queries)}] "
                            "검색 실패 또는 비정상 응답"
                        ),
                    }
                    continue

                if not self.external.has_meaningful_search_items(search_result.data):
                    yield {
                        "type": "thought",
                        "step": 2,
                        "delta": (
                            f"[외부 검색 결과 {atomic_idx}.{ridx}/{len(rewritten_queries)}] "
                            "의미 있는 검색 결과 없음"
                        ),
                    }
                    continue

                adopted_queries.append(rewritten_query)

                collected_items = self.sources.collect_search_items_from_result(search_result.data)
                if collected_items:
                    all_collected_items_for_atomic.append(collected_items)

                yield {
                    "type": "thought",
                    "step": 2,
                    "delta": (
                        f"[외부 검색 결과 {atomic_idx}.{ridx}/{len(rewritten_queries)}] "
                        f"수집 URL 수={len(collected_items)}"
                    ),
                }

            if not all_collected_items_for_atomic:
                yield {
                    "type": "thought",
                    "step": 2,
                    "delta": (
                        f"[외부 검색 실패 {atomic_idx}/{len(realized_atomic_queries)}] "
                        f"질문={json.dumps(atomic_query, ensure_ascii=False)}"
                    ),
                }
                continue

            adopted_queries_by_atomic[atomic_query] = " | ".join(adopted_queries)

            merged_atomic_items = self.sources.merge_search_items(all_collected_items_for_atomic)
            if merged_atomic_items:
                all_collected_search_items.append(merged_atomic_items)

            yield {
                "type": "thought",
                "step": 2,
                "delta": (
                    f"[외부 검색 완료 {atomic_idx}/{len(realized_atomic_queries)}] "
                    f"질문={json.dumps(atomic_query, ensure_ascii=False)} | "
                    f"실행 검색어 수={len(rewritten_queries)} | "
                    f"성공 검색어 수={len(adopted_queries)} | "
                    f"수집 URL 수={len(merged_atomic_items)}"
                ),
            }

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

            if self.external.has_meaningful_context(build_result.data):
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

            merged_build = dict(build_result.data)
            merged_sources = merged_build.get("sources") or []
            if isinstance(merged_sources, list):
                merged_build["sources"] = self.sources.stitch_adjacent_sources(merged_sources)

                rebuilt_context: List[str] = []
                for src in merged_build["sources"]:
                    if not isinstance(src, dict):
                        continue
                    content = str(
                        src.get("content")
                        or src.get("snippet")
                        or src.get("preview")
                        or ""
                    ).strip()
                    if content:
                        rebuilt_context.append(content)

                if rebuilt_context:
                    merged_build["context"] = "\n\n".join(rebuilt_context).strip()

        else:
            merged_build = self.external.merge_result_payloads(
                query_to_result=per_atomic_build_results,
                original_user_message=original_user_message,
                prefer_longer_context=False,
                stitch_sources=True,
                summarize_multi_query=False,
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
                original_user_message="",
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