from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from .agent_speed_utils import normalize_whitespace
from .config import AgentConfig


class AgentPlanner:
    """
    planner 전용 모듈

    책임
    - 사용자 요청과 collection catalog를 보고 실행 계획 수립
    - direct send_mail 인자 추출
    - mixed request 에서 질문별 task 분해
    - 질문별 task_type 부여 (예: get_menu_direct / research)
    - LLM JSON 응답 파싱/정규화
    """

    VALID_INTENTS = {
        "answer_only",
        "send_mail_direct",
        "research_then_send_mail",
        "research_then_answer",
        "get_menu_direct",
    }

    VALID_TASK_TYPES = {
        "research",
        "get_menu_direct",
    }

    def __init__(self, ollama, config: AgentConfig):
        self.ollama = ollama
        self.config = config

    # ---------------------------------------------------------------------
    # low-level helpers
    # ---------------------------------------------------------------------

    def _safe_json_loads_dict(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}

        content = text.strip()

        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        return {}

    def _call_llm_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            llm = self.ollama.build_chat_llm(model=self.config.model, think=False)
            response = llm.invoke(messages)
            content = getattr(response, "content", "") or ""
            if not isinstance(content, str):
                content = str(content)
            return self._safe_json_loads_dict(content)
        except Exception:
            return {}

    def _clean_scalar_or_list(self, value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, str):
            return value.strip()

        if isinstance(value, list):
            out: List[str] = []
            for item in value:
                s = str(item).strip()
                if s:
                    out.append(s)
            return out

        return str(value).strip()

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
            clean = normalize_whitespace(str(item or ""))
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(clean)

        return out

    # ---------------------------------------------------------------------
    # task normalization
    # ---------------------------------------------------------------------

    def _normalize_task_type(self, value: Any) -> str:
        raw = normalize_whitespace(str(value or "")).lower()
        if raw in self.VALID_TASK_TYPES:
            return raw
        return "research"

    def _normalize_research_tasks(self, value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []

        normalized: List[Dict[str, Any]] = []

        for item in value:
            if not isinstance(item, dict):
                continue

            topic = normalize_whitespace(str(item.get("topic") or ""))
            goal = normalize_whitespace(str(item.get("goal") or ""))
            task_type = self._normalize_task_type(item.get("task_type"))
            menu_date = normalize_whitespace(str(item.get("menu_date") or ""))

            if not topic and not goal:
                continue

            if not topic:
                topic = goal
            if not goal:
                goal = "사용자 질문에 답하기"

            normalized.append(
                {
                    "topic": topic,
                    "goal": goal,
                    "task_type": task_type,
                    "menu_date": menu_date,
                }
            )

        deduped: List[Dict[str, Any]] = []
        seen = set()

        for item in normalized:
            key = (
                normalize_whitespace(item.get("topic") or "").lower(),
                normalize_whitespace(item.get("task_type") or "").lower(),
                normalize_whitespace(item.get("menu_date") or "").lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        return deduped

    def _infer_menu_date_from_message(self, user_message: str) -> str:
        text = normalize_whitespace(user_message).lower()

        if any(word in text for word in ["내일", "tomorrow"]):
            return "tomorrow"
        if any(word in text for word in ["모레"]):
            return "day_after_tomorrow"
        if any(word in text for word in ["어제", "yesterday"]):
            return "yesterday"
        if any(word in text for word in ["오늘", "today", "학식", "메뉴"]):
            return "today"

        return "today"

    def _post_normalize_plan(
        self,
        *,
        user_message: str,
        raw_plan: Dict[str, Any],
        collection_catalog: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        normalized_intent = normalize_whitespace(str(raw_plan.get("intent") or ""))
        if normalized_intent not in self.VALID_INTENTS:
            normalized_intent = "research_then_answer"

        available_collections = {
            normalize_whitespace(str(item.get("name") or ""))
            for item in collection_catalog
            if isinstance(item, dict)
        }

        collection_candidates = [
            name
            for name in self._normalize_string_list(raw_plan.get("collection_candidates"))
            if name in available_collections
        ]
        collection_candidates = self._dedupe_strings(collection_candidates)

        research_tasks = self._normalize_research_tasks(raw_plan.get("research_tasks"))
        menu_date = normalize_whitespace(str(raw_plan.get("menu_date") or ""))

        use_internal_first = bool(raw_plan.get("use_internal_first"))
        use_external_search = bool(raw_plan.get("use_external_search"))
        needs_grounded_synthesis = bool(raw_plan.get("needs_grounded_synthesis"))

        mail_to = self._clean_scalar_or_list(raw_plan.get("mail_to"))
        mail_cc = self._clean_scalar_or_list(raw_plan.get("mail_cc"))
        mail_bcc = self._clean_scalar_or_list(raw_plan.get("mail_bcc"))
        mail_subject_hint = normalize_whitespace(str(raw_plan.get("mail_subject_hint") or ""))
        mail_body_hint = normalize_whitespace(str(raw_plan.get("mail_body_hint") or ""))

        if normalized_intent == "get_menu_direct":
            if not menu_date:
                menu_date = self._infer_menu_date_from_message(user_message)
            if not research_tasks:
                research_tasks = [
                    {
                        "topic": normalize_whitespace(user_message),
                        "goal": "학식 메뉴 조회",
                        "task_type": "get_menu_direct",
                        "menu_date": menu_date,
                    }
                ]

        if normalized_intent in {"answer_only", "research_then_answer"} and not research_tasks:
            research_tasks = [
                {
                    "topic": normalize_whitespace(user_message),
                    "goal": "사용자 질문에 답하기",
                    "task_type": "research",
                    "menu_date": "",
                }
            ]

        if any(task.get("task_type") == "get_menu_direct" for task in research_tasks):
            if normalized_intent == "answer_only":
                normalized_intent = "research_then_answer"

        if research_tasks and normalized_intent in {"research_then_answer", "research_then_send_mail"}:
            needs_grounded_synthesis = True

        return {
            "intent": normalized_intent,
            "collection_candidates": collection_candidates,
            "research_tasks": research_tasks,
            "use_internal_first": use_internal_first,
            "use_external_search": use_external_search,
            "needs_grounded_synthesis": needs_grounded_synthesis,
            "menu_date": menu_date,
            "mail_to": mail_to,
            "mail_cc": mail_cc,
            "mail_bcc": mail_bcc,
            "mail_subject_hint": mail_subject_hint,
            "mail_body_hint": mail_body_hint,
        }

    # ---------------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------------

    def plan(self, *, user_message: str, collection_catalog: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        collection-aware planner

        - 내부 collection 존재/도메인/설명/태그를 보고 internal 우선 여부 결정
        - direct send 와 research_then_send 구분
        - 메뉴 조회 direct tool 구분
        - mixed request 에서는 research_tasks 에 task_type 을 넣어 분해
        - 최종적으로 normalized plan 반환
        """
        system_prompt = (
            "너는 agent orchestration planner다.\n"
            "사용자 요청과 내부 collection catalog 를 함께 보고 어떤 액션 순서로 처리할지 결정하라.\n"
            "매우 중요:\n"
            "- 내부 collection catalog 에 관련성이 있는 collection 이 있으면 internal search 를 우선 고려하라.\n"
            "- 메일을 지금 바로 보내라는 요청이면 send_mail_direct 로 분류하라.\n"
            "- 조사/검색 후 메일 작성이 필요하면 research_then_send_mail 로 분류하라.\n"
            "- 메뉴만 직접 조회하면 되는 요청이면 get_menu_direct 로 분류하라.\n"
            "- 일반 질의응답/조사형은 research_then_answer 또는 answer_only 로 분류하라.\n"
            "\n"
            "[혼합 질문 처리 규칙]\n"
            "- 하나의 사용자 메시지 안에 서로 다른 하위 질문이 섞여 있을 수 있다.\n"
            "- 이 경우 research_tasks 에 하위 질문들을 각각 분해해서 넣어라.\n"
            "- 각 research task 는 반드시 task_type 을 가져야 한다.\n"
            "- task_type 은 research 또는 get_menu_direct 중 하나만 허용한다.\n"
            "- 학식/메뉴/식단 조회성 항목이면 해당 하위 질문만 task_type=get_menu_direct 로 넣어라.\n"
            "- 사람 정보/장학금/공지/연락처/설명/위치 등은 task_type=research 로 넣어라.\n"
            "- top-level intent 는 전체 흐름 제어용이므로, mixed request 는 보통 research_then_answer 로 두어라.\n"
            "\n"
            "[메뉴 관련 규칙]\n"
            "- 사용자가 오늘/내일/어제 등의 메뉴를 묻는 경우 menu_date 를 채워라.\n"
            "- 메뉴 질문이 하위 task 이면 그 task 의 menu_date 도 채워라.\n"
            "\n"
            "[출력 규칙]\n"
            "- 반드시 JSON만 출력하라.\n"
            "- 형식:\n"
            "{\n"
            '  "intent": "answer_only|send_mail_direct|research_then_send_mail|research_then_answer|get_menu_direct",\n'
            '  "collection_candidates": ["..."],\n'
            '  "research_tasks": [\n'
            '    {\n'
            '      "topic": "...",\n'
            '      "goal": "...",\n'
            '      "task_type": "research|get_menu_direct",\n'
            '      "menu_date": "today|tomorrow|yesterday|... or empty"\n'
            "    }\n"
            "  ],\n"
            '  "use_internal_first": true,\n'
            '  "use_external_search": false,\n'
            '  "needs_grounded_synthesis": true,\n'
            '  "menu_date": "",\n'
            '  "mail_to": "",\n'
            '  "mail_cc": "",\n'
            '  "mail_bcc": "",\n'
            '  "mail_subject_hint": "",\n'
            '  "mail_body_hint": ""\n'
            "}\n"
            "- collection_candidates 는 catalog 에 실제 존재하는 이름 위주로 고르라.\n"
            "- 확실하지 않으면 intent 는 research_then_answer 로 두어라.\n"
        )

        user_prompt = (
            f"[사용자 요청]\n{normalize_whitespace(user_message)}\n\n"
            f"[내부 collection catalog]\n{json.dumps(collection_catalog, ensure_ascii=False, indent=2)}"
        )

        raw_plan = self._call_llm_json(system_prompt, user_prompt)
        normalized = self._post_normalize_plan(
            user_message=user_message,
            raw_plan=raw_plan,
            collection_catalog=collection_catalog,
        )

        return normalized

    def extract_direct_mail_args(self, *, user_message: str) -> Dict[str, Any]:
        system_prompt = (
            "너는 메일 전송 요청에서 to/cc/bcc/subject/body 를 뽑아내는 정보 추출기다.\n"
            "반드시 JSON만 출력하라.\n"
            "{\n"
            '  "to": string | string[],\n'
            '  "cc": string | string[] | null,\n'
            '  "bcc": string | string[] | null,\n'
            '  "subject": string,\n'
            '  "body": string,\n'
            '  "html_body": string | null\n'
            "}\n"
            "모르면 빈 문자열 또는 null 로 두어라.\n"
        )

        result = self._call_llm_json(system_prompt, normalize_whitespace(user_message))
        return {
            "to": self._clean_scalar_or_list(result.get("to")),
            "cc": self._clean_scalar_or_list(result.get("cc")),
            "bcc": self._clean_scalar_or_list(result.get("bcc")),
            "subject": self._clean_scalar_or_list(result.get("subject")),
            "body": self._clean_scalar_or_list(result.get("body")),
            "html_body": self._clean_scalar_or_list(result.get("html_body")),
        }