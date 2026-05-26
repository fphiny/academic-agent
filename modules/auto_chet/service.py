from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from core.ollama.client import OllamaClient


ROUTE_CHAT = "chat"
ROUTE_AGENT = "agent_speed"
VALID_ROUTES = {ROUTE_CHAT, ROUTE_AGENT}


@dataclass
class AutoRouteDecision:
    route: str
    reason: str
    scores: dict[str, int] = field(default_factory=dict)
    matched_keywords: dict[str, list[str]] = field(default_factory=dict)
    contextualized_query: Optional[str] = None


class AutoChetService:
    """
    LLM prompt 기반 router.
    conversation history를 함께 보고
    지금 바로 답 가능한지(chat),
    아니면 새 작업/새 조사/새 분석이 필요한지(agent_speed) 판단한다.
    """

    ROUTER_SYSTEM_PROMPT = """
너는 사용자 요청을 정확히 분류하는 라우터다.
반드시 아래 2개 route 중 하나만 선택한다.

[route 정의]
1. chat
- 일반 대화
- 주관적 조언
- 간단한 설명
- 창작/잡담
- conversation_history 안의 정보만으로 충분히 답할 수 있는 요청
- 이미 이전 대화에서 확실한 답이 있고, 다룬 내용을 다시 묻거나 표현만 바꿔서 묻는 요청
- 기존 답변을 재서술, 요약, 보완하는 요청
- 새 검색, 새 조사, 새 계산, 새 계획 수립 없이 바로 응답 가능한 요청

2. agent_speed
- 여러 단계를 거쳐 분석/비교/계획/계산/정리/의사결정이 필요한 요청
- conversation_history 안의 정보만으로는 답하기 부족한 요청
- 새 검색, 새 조사, 새 근거 확인, 새 계산, 새 정리가 필요한 요청
- 이전 답변을 그대로 반복하는 것으로는 부족하고 추가 작업이 필요한 요청
- 문서/가이드/정책/규정/절차/공지/매뉴얼/근거 기반으로 다시 확인하거나 정리해야 하는 요청
- 최신성, 정확성, 검증, 비교, 재분석이 중요해서 추가 작업이 필요한 요청
- 메일 전송 기능 필요시 에이전트를 호출한다.

[판단 원칙]
- 먼저 conversation_history를 보고 이미 답할 수 있는지 판단해라.
- 과거 대화에 이미 충분한 답변 근거가 있으면 chat을 선택해라.
- 같은 주제의 반복 질문이라고 해서 무조건 agent_speed로 보내지 마라.
- 사용자가 같은 대상을 다시 물어보더라도, history만으로 충분히 응답 가능하면 chat이다.
- 이전 답변만으로 부족해서 새로 조사/확인/비교/정리/계산해야 하면 agent_speed다.
- 키워드 매칭처럼 기계적으로 판단하지 말고 사용자 의도를 해석해라.
- 사용자의 말투가 거칠어도 표현은 무시하고 의도만 보고 분류해라.
- 애매하면 chat 쪽으로 보수적으로 선택해라.

[contextualized_query 작성 규칙]
- route가 chat 또는 agent_speed이면 contextualized_query는 latest_query와 동일하게 둬라.

반드시 JSON만 출력해라.
추가 설명, 코드블록, 마크다운, 서문 없이 아래 형식만 반환해라.

{
  "route": "chat" | "agent_speed",
  "reason": "한 문장 설명",
  "contextualized_query": "문자열"
}
""".strip()

    def __init__(self, router_model: Optional[str] = None):
        self.ollama = OllamaClient()
        self.router_model = router_model

    def history_to_text(
        self,
        history_records: Optional[Iterable[Any]],
        max_messages: int = 12,
        max_chars: int = 4000,
    ) -> str:
        if not history_records:
            return ""

        items = list(history_records)[-max_messages:]
        lines: list[str] = []

        for msg in items:
            role = getattr(msg, "type", None) or msg.__class__.__name__.replace("Message", "").lower()
            content = getattr(msg, "content", "")

            if isinstance(content, list):
                content = " ".join(str(x) for x in content if x is not None)
            elif content is None:
                content = ""
            else:
                content = str(content)

            content = re.sub(r"\s+", " ", content).strip()
            if not content:
                continue

            if role == "human":
                role_name = "user"
            elif role == "ai":
                role_name = "assistant"
            else:
                role_name = str(role)

            lines.append(f"{role_name}: {content}")

        text = "\n".join(lines).strip()
        if len(text) > max_chars:
            text = text[-max_chars:]

        return text

    def decide(
        self,
        query: str,
        history_records: Optional[Iterable[Any]] = None,
        forced_route: Optional[str] = None,
        router_model: Optional[str] = None,
        rag_available: bool = True,
        rag_status: str = "",
        requested_collection: str = "",
    ) -> AutoRouteDecision:
        clean_query = (query or "").strip()
        route_override = (forced_route or "").strip().lower()

        if not clean_query:
            return AutoRouteDecision(
                route=ROUTE_CHAT,
                reason="빈 질문이어서 기본 chat으로 처리",
                scores={ROUTE_CHAT: 1, ROUTE_AGENT: 0},
                matched_keywords={ROUTE_AGENT: []},
                contextualized_query="",
            )

        if route_override in VALID_ROUTES:
            return AutoRouteDecision(
                route=route_override,
                reason=f"사용자 지정 route={route_override}",
                scores={
                    ROUTE_CHAT: 1 if route_override == ROUTE_CHAT else 0,
                    ROUTE_AGENT: 1 if route_override == ROUTE_AGENT else 0,
                },
                matched_keywords={ROUTE_AGENT: []},
                contextualized_query=clean_query,
            )

        history_text = self.history_to_text(history_records)

        try:
            llm = self.ollama.build_chat_llm(
                model=router_model or self.router_model or self.ollama.resolve_model(None),
                think=False,
            )

            prompt = (
                f"[conversation_history]\n{history_text or '(empty)'}\n\n"
                f"[latest_query]\n{clean_query}"
            )

            response = llm.invoke(
                [
                    SystemMessage(content=self.ROUTER_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )

            raw_text = self._message_to_text(response)
            data = self._parse_json_object(raw_text)

            route = str(data.get("route") or "").strip().lower()
            reason = str(data.get("reason") or "").strip()
            contextualized_query = str(data.get("contextualized_query") or "").strip()

            if route == "agent":
                route = ROUTE_AGENT

            if route not in VALID_ROUTES:
                route = ROUTE_CHAT

            if not reason:
                reason = "LLM router 기본 판단"

            return AutoRouteDecision(
                route=route,
                reason=reason,
                scores={
                    ROUTE_CHAT: 1 if route == ROUTE_CHAT else 0,
                    ROUTE_AGENT: 1 if route == ROUTE_AGENT else 0,
                },
                matched_keywords={ROUTE_AGENT: []},
                contextualized_query=contextualized_query or clean_query,
            )

        except Exception as e:
            return AutoRouteDecision(
                route=ROUTE_CHAT,
                reason=f"router fallback: {str(e)}",
                scores={ROUTE_CHAT: 1, ROUTE_AGENT: 0},
                matched_keywords={ROUTE_AGENT: []},
                contextualized_query=clean_query,
            )

    def _message_to_text(self, message: Any) -> str:
        content = getattr(message, "content", "")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                elif item is not None:
                    parts.append(str(item))
            return "\n".join(parts).strip()

        return str(content).strip()

    def _parse_json_object(self, text: str) -> dict[str, Any]:
        cleaned = (text or "").strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            candidate = match.group(0)
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data

        raise ValueError("router output is not valid JSON")

    def decide_route_only(
        self,
        query: str,
        history_records: Optional[Iterable[Any]] = None,
        forced_route: Optional[str] = None,
        router_model: Optional[str] = None,
        rag_available: bool = True,
        rag_status: str = "",
        requested_collection: str = "",
    ) -> str:
        decision = self.decide(
            query=query,
            history_records=history_records,
            forced_route=forced_route,
            router_model=router_model,
            rag_available=rag_available,
            rag_status=rag_status,
            requested_collection=requested_collection,
        )
        return decision.route


auto_chet_service = AutoChetService()