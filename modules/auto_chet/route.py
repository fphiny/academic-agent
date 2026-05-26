from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage

from core.ollama.client import OllamaClient
from core.parse import (
    clamp,
    get_request_data,
    parse_bool,
    parse_float,
    parse_int,
    parse_json_str,
    stringify_observation,
)
from core.session import get_logged_in_student_id
from core.streaming.sse import sse, sse_comment
from modules.agent_speed.service import AgentConfig, AgentSpeedService
from modules.auto_chet.service import ROUTE_AGENT, ROUTE_CHAT, auto_chet_service
from modules.log.service import log_service

DEFAULT_HISTORY_LIMIT = 20
DEFAULT_AGENT_MAX_STEPS = 6
DEFAULT_AGENT_TEMPERATURE = 0.2
DEFAULT_AGENT_COLLECTION = "kb_current"

router = APIRouter(prefix="/api/auto_chet", tags=["auto_chet"])
ollama = OllamaClient()


def _streaming_headers() -> Dict[str, str]:
    return {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


def _build_agent_service(
    model: str,
    max_steps: int,
    temperature: float,
    default_collection: str,
) -> AgentSpeedService:
    return AgentSpeedService(
        AgentConfig(
            ollama_host=ollama.host,
            model=model,
            max_steps=max_steps,
            temperature=temperature,
            default_collection=default_collection,
        )
    )


def _pick_agent_collection(collection_name: str) -> str:
    original = (collection_name or "").strip()
    if original:
        return original
    return DEFAULT_AGENT_COLLECTION


def _make_kakao_simple_text(text: str) -> Dict[str, Any]:
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": (text or "")
                    }
                }
            ]
        }
    }


def _extract_kakao_user_message(body: Dict[str, Any]) -> str:
    user_request = body.get("userRequest") or {}
    value = body.get("value") or {}

    return (
        user_request.get("utterance")
        or body.get("utterance")
        or value.get("resolved")
        or value.get("origin")
        or body.get("message")
        or ""
    ).strip()


def _extract_kakao_callback_url(body: Dict[str, Any]) -> str:
    user_request = body.get("userRequest") or {}
    action = body.get("action") or {}
    client_extra = action.get("clientExtra") or {}

    return (
        user_request.get("callbackUrl")
        or body.get("callbackUrl")
        or client_extra.get("callbackUrl")
        or ""
    ).strip()


def _send_kakao_callback(callback_url: str, text: str) -> None:
    import requests

    payload = _make_kakao_simple_text(text)
    requests.post(
        callback_url,
        json=payload,
        timeout=15,
        headers={"Content-Type": "application/json; charset=utf-8"},
    ).raise_for_status()


def _run_auto_chet_final_only(
    *,
    user_message: str,
    model: str,
    mode: str,
    think: bool,
    sid: str,
    history_limit: int,
    collection_name: str,
    where: Any,
    max_steps: int,
    temperature: float,
    forced_route: str,
    student_id: Optional[int],
    route: str = "/api/auto_chet/kakao",
) -> str:
    try:
        history = (
            log_service.build_langchain_history(
                student_id=student_id,
                sid=sid,
                limit=history_limit,
                include_roles=["user", "assistant", "system"],
            )
            if student_id
            else []
        )
    except Exception:
        history = []

    effective_agent_collection = _pick_agent_collection(collection_name)

    decision = auto_chet_service.decide(
        query=user_message,
        history_records=history,
        forced_route=forced_route,
        router_model=model,
    )

    runtime_state: dict[str, Any] = {
        "selected_route": decision.route,
        "route_reason": decision.reason,
        "effective_agent_collection": effective_agent_collection,
    }

    saved_event_ids: list[int] = []
    final_buf: list[str] = []

    def save_event(
        event_type: str,
        content: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not student_id:
            return
        try:
            ev = log_service.append_event(
                student_id=student_id,
                sid=sid,
                mode="auto_chet_kakao",
                event_type=event_type,
                content=content,
                metadata=metadata or {},
            )
            if ev and getattr(ev, "id", None):
                saved_event_ids.append(ev.id)
        except Exception:
            pass

    if student_id:
        try:
            log_service.append_user_message(
                student_id=student_id,
                sid=sid,
                mode="auto_chet_kakao",
                content=user_message,
                metadata={
                    "model": model,
                    "mode": mode,
                    "think": think,
                    "route": route,
                    "history_limit": history_limit,
                    "selected_route": decision.route,
                    "route_reason": decision.reason,
                    "route_scores": decision.scores,
                    "matched_keywords": decision.matched_keywords,
                    "collection_name": collection_name,
                    "effective_agent_collection": effective_agent_collection,
                    "where": where,
                    "max_steps": max_steps,
                    "temperature": temperature,
                    "channel": "kakao",
                },
            )
        except Exception:
            pass

    save_event(
        "route_decision",
        content=runtime_state["route_reason"],
        metadata={
            "selected_route": runtime_state["selected_route"],
            "scores": decision.scores,
            "matched_keywords": decision.matched_keywords,
            "effective_agent_collection": runtime_state["effective_agent_collection"],
        },
    )

    if runtime_state["selected_route"] == ROUTE_CHAT:
        llm = ollama.build_chat_llm(model=model, think=think)
        messages = history + [HumanMessage(content=user_message)]

        for chunk in llm.stream(messages):
            content = getattr(chunk, "content", None)
            if content:
                token = content if isinstance(content, str) else str(content)
                if token:
                    final_buf.append(token)
                    save_event("delta", content=token)

            additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}
            reasoning_content = additional_kwargs.get("reasoning_content")
            if reasoning_content:
                reasoning_text = (
                    reasoning_content
                    if isinstance(reasoning_content, str)
                    else str(reasoning_content)
                )
                save_event("thinking", content=reasoning_text)

    elif runtime_state["selected_route"] == ROUTE_AGENT:
        agent_service = _build_agent_service(
            model=model,
            max_steps=max_steps,
            temperature=temperature,
            default_collection=effective_agent_collection,
        )

        for event in agent_service.run(
            user_message=user_message,
            collection_name=effective_agent_collection,
        ):
            etype = str(event.get("type", "message"))

            if etype in {"agent_step", "thinking", "thought"}:
                thought_text = str(
                    event.get("delta")
                    or event.get("step")
                    or event.get("thought")
                    or ""
                ).strip()
                if thought_text:
                    save_event("thought", content=thought_text, metadata={"event": event})
                continue

            if etype == "tool_call":
                save_event("tool_call", metadata={"event": event})
                continue

            if etype == "tool_result":
                obs = stringify_observation(event.get("result"))
                save_event("tool_result", content=obs)
                continue

            if etype == "sources":
                save_event("sources", metadata={"event": event})
                continue

            if etype == "delta":
                chunk_text = str(event.get("delta", "") or "")
                if chunk_text:
                    final_buf.append(chunk_text)
                    save_event("delta", content=chunk_text, metadata={"event": event})
                continue

            if etype == "done":
                break

            if etype == "error":
                error_message = str(event.get("error", "unknown error") or "unknown error")
                save_event("error", content=error_message)
                raise RuntimeError(error_message)

            save_event("unknown", content=str(event), metadata={"event": event})

    else:
        raise RuntimeError(f"unsupported route: {runtime_state['selected_route']}")

    final_text = "".join(final_buf).strip()
    if not final_text:
        raise RuntimeError("auto_chet ended without final text")

    if student_id:
        try:
            assistant_msg = log_service.append_assistant_message(
                student_id=student_id,
                sid=sid,
                mode="auto_chet_kakao",
                content=final_text,
                metadata={
                    "model": model,
                    "mode": mode,
                    "think": think,
                    "route": route,
                    "history_limit": history_limit,
                    "selected_route": runtime_state["selected_route"],
                    "route_reason": runtime_state["route_reason"],
                    "route_scores": decision.scores,
                    "matched_keywords": decision.matched_keywords,
                    "collection_name": collection_name,
                    "effective_agent_collection": runtime_state["effective_agent_collection"],
                    "where": where,
                    "max_steps": max_steps,
                    "temperature": temperature,
                    "channel": "kakao",
                },
            )
            if assistant_msg is not None and saved_event_ids:
                try:
                    log_service.bind_events_to_message(
                        student_id=student_id,
                        sid=sid,
                        message_id=assistant_msg.id,
                        event_ids=saved_event_ids,
                    )
                except Exception:
                    pass
        except Exception:
            pass

    return final_text


def _process_kakao_callback(
    *,
    callback_url: str,
    user_message: str,
    model: str,
    mode: str,
    think: bool,
    sid: str,
    history_limit: int,
    collection_name: str,
    where: Any,
    max_steps: int,
    temperature: float,
    forced_route: str,
    student_id: Optional[int],
) -> None:
    try:
        answer = _run_auto_chet_final_only(
            user_message=user_message,
            model=model,
            mode=mode,
            think=think,
            sid=sid,
            history_limit=history_limit,
            collection_name=collection_name,
            where=where,
            max_steps=max_steps,
            temperature=temperature,
            forced_route=forced_route,
            student_id=student_id,
            route="/api/auto_chet/kakao",
        )
    except Exception:
        answer = "지금 답변이 어려워요. 잠시 후 다시 시도해 주세요."

    try:
        _send_kakao_callback(callback_url, answer)
    except Exception:
        pass


@router.post("/stream")
async def api_auto_chet_stream(request: Request):
    student_id = get_logged_in_student_id(request)
    if not student_id:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")

    payload = await get_request_data(request)

    user_message = str(payload.get("message") or "").strip()
    if not user_message:
        return StreamingResponse(
            iter([sse("error", {"error": "message is required"})]),
            status_code=400,
            media_type="text/event-stream",
            headers=_streaming_headers(),
        )

    model = ollama.resolve_model(payload.get("model"))
    mode = str(payload.get("mode") or "normal").strip().lower()
    think = parse_bool(payload.get("think"), default=(mode == "think"))
    sid = str(payload.get("sid") or str(uuid.uuid4())).strip()

    history_limit = parse_int(payload.get("history_limit"), DEFAULT_HISTORY_LIMIT)
    if history_limit < 0:
        history_limit = 0

    collection_name = str(payload.get("collection") or DEFAULT_AGENT_COLLECTION).strip() or DEFAULT_AGENT_COLLECTION
    _ = parse_int(payload.get("k"), 5)  # 하위 호환용으로만 consume
    where = parse_json_str(payload.get("where"), default=None)

    max_steps = parse_int(payload.get("max_steps"), DEFAULT_AGENT_MAX_STEPS)
    if max_steps < 1:
        max_steps = 1

    temperature = clamp(
        parse_float(payload.get("temperature"), DEFAULT_AGENT_TEMPERATURE),
        0.0,
        2.0,
    )

    forced_route = str(payload.get("route") or "").strip().lower()

    try:
        history = log_service.build_langchain_history(
            student_id=student_id,
            sid=sid,
            limit=history_limit,
            include_roles=["user", "assistant", "system"],
        )
    except Exception:
        history = []

    effective_agent_collection = _pick_agent_collection(collection_name)

    decision = auto_chet_service.decide(
        query=user_message,
        history_records=history,
        forced_route=forced_route,
        router_model=model,
    )

    runtime_state: dict[str, Any] = {
        "selected_route": decision.route,
        "route_reason": decision.reason,
        "effective_agent_collection": effective_agent_collection,
    }

    try:
        log_service.append_user_message(
            student_id=student_id,
            sid=sid,
            mode="auto_chet",
            content=user_message,
            metadata={
                "model": model,
                "mode": mode,
                "think": think,
                "route": "/api/auto_chet/stream",
                "history_limit": history_limit,
                "selected_route": decision.route,
                "route_reason": decision.reason,
                "route_scores": decision.scores,
                "matched_keywords": decision.matched_keywords,
                "collection_name": collection_name,
                "effective_agent_collection": effective_agent_collection,
                "where": where,
                "max_steps": max_steps,
                "temperature": temperature,
            },
        )
    except Exception as e:
        return StreamingResponse(
            iter([sse("error", {"error": f"log save failed: {str(e)}"})]),
            status_code=500,
            media_type="text/event-stream",
            headers=_streaming_headers(),
        )

    def finalize_assistant_message(final_text: str) -> None:
        try:
            log_service.append_assistant_message(
                student_id=student_id,
                sid=sid,
                mode="auto_chet",
                content=final_text,
                metadata={
                    "model": model,
                    "mode": mode,
                    "think": think,
                    "route": "/api/auto_chet/stream",
                    "history_limit": history_limit,
                    "selected_route": runtime_state["selected_route"],
                    "route_reason": runtime_state["route_reason"],
                    "route_scores": decision.scores,
                    "matched_keywords": decision.matched_keywords,
                    "collection_name": collection_name,
                    "effective_agent_collection": runtime_state["effective_agent_collection"],
                    "where": where,
                    "max_steps": max_steps,
                    "temperature": temperature,
                },
            )
        except Exception:
            pass

    def save_event(event_type: str, content: str = "", metadata: Optional[Dict[str, Any]] = None) -> None:
        try:
            log_service.append_event(
                student_id=student_id,
                sid=sid,
                mode="auto_chet",
                event_type=event_type,
                content=content,
                metadata=metadata or {},
            )
        except Exception:
            pass

    def generate():
        yield sse_comment("connected")
        yield sse("sid", {"sid": sid})
        yield sse(
            "route",
            {
                "selected_route": runtime_state["selected_route"],
                "reason": runtime_state["route_reason"],
                "scores": decision.scores,
                "matched_keywords": decision.matched_keywords,
                "effective_agent_collection": runtime_state["effective_agent_collection"],
            },
        )
        yield sse(
            "meta",
            {
                "model": model,
                "mode": mode,
                "think": think,
                "history_limit": history_limit,
                "collection_name": collection_name,
                "effective_agent_collection": effective_agent_collection,
                "max_steps": max_steps,
                "temperature": temperature,
                "selected_route": runtime_state["selected_route"],
            },
        )

        save_event(
            "route_decision",
            content=runtime_state["route_reason"],
            metadata={
                "selected_route": runtime_state["selected_route"],
                "scores": decision.scores,
                "matched_keywords": decision.matched_keywords,
                "effective_agent_collection": runtime_state["effective_agent_collection"],
            },
        )

        last_ping = time.time()
        final_buf: list[str] = []

        def maybe_ping() -> str | None:
            nonlocal last_ping
            now = time.time()
            if now - last_ping >= 15:
                last_ping = now
                return sse_comment("ping")
            return None

        def stream_agent(agent_collection: str, route_reason: Optional[str] = None):
            runtime_state["selected_route"] = ROUTE_AGENT
            if route_reason:
                runtime_state["route_reason"] = route_reason
            runtime_state["effective_agent_collection"] = agent_collection

            save_event(
                "route_decision",
                content=runtime_state["route_reason"],
                metadata={
                    "selected_route": ROUTE_AGENT,
                    "effective_agent_collection": runtime_state["effective_agent_collection"],
                },
            )

            yield sse(
                "route",
                {
                    "selected_route": ROUTE_AGENT,
                    "reason": runtime_state["route_reason"],
                    "effective_agent_collection": runtime_state["effective_agent_collection"],
                },
            )

            agent_service = _build_agent_service(
                model=model,
                max_steps=max_steps,
                temperature=temperature,
                default_collection=agent_collection,
            )

            for event in agent_service.run(
                user_message=user_message,
                collection_name=agent_collection,
            ):
                ping = maybe_ping()
                if ping:
                    yield ping

                etype = str(event.get("type", "message"))

                if etype in {"agent_step", "thinking", "thought"}:
                    thought_text = str(
                        event.get("delta")
                        or event.get("step")
                        or event.get("thought")
                        or ""
                    ).strip()
                    if thought_text:
                        save_event("thought", content=thought_text, metadata={"event": event})
                        yield sse("thought", {"text": thought_text, "raw": event})
                    continue

                if etype == "tool_call":
                    save_event("tool_call", metadata={"event": event})
                    yield sse(
                        "action",
                        {
                            "tool": event.get("tool_name"),
                            "arguments": event.get("arguments", {}) or {},
                        },
                    )
                    continue

                if etype == "tool_result":
                    obs = stringify_observation(event.get("result"))
                    save_event("tool_result", content=obs)
                    yield sse("observation", {"text": obs})
                    continue

                if etype == "sources":
                    save_event("sources", metadata={"event": event})
                    yield sse(
                        "sources",
                        {
                            "tool_name": event.get("tool_name"),
                            "sources": event.get("sources", []) or [],
                        },
                    )
                    continue

                if etype == "delta":
                    chunk_text = str(event.get("delta", "") or "")
                    if chunk_text:
                        final_buf.append(chunk_text)
                        yield sse("chunk", {"text": chunk_text})
                    continue

                if etype == "done":
                    final_text = "".join(final_buf).strip()
                    finalize_assistant_message(final_text)
                    yield sse("final", {"text": final_text})
                    yield sse("done", {"done": True})
                    return

                if etype == "error":
                    error_message = str(event.get("error", "unknown error") or "unknown error")
                    save_event("error", content=error_message)
                    yield sse("error", {"error": error_message})
                    return

                save_event("unknown", content=str(event), metadata={"event": event})

            final_text = "".join(final_buf).strip()
            if final_text:
                finalize_assistant_message(final_text)
                yield sse("final", {"text": final_text})
                yield sse("done", {"done": True})
                return

            yield sse("error", {"error": "agent stream ended without done"})

        try:
            if runtime_state["selected_route"] == ROUTE_CHAT:
                llm = ollama.build_chat_llm(model=model, think=think)
                messages = history + [HumanMessage(content=user_message)]

                for chunk in llm.stream(messages):
                    ping = maybe_ping()
                    if ping:
                        yield ping

                    content = getattr(chunk, "content", None)
                    if content:
                        token = content if isinstance(content, str) else str(content)
                        if token:
                            final_buf.append(token)
                            yield sse("delta", {"delta": token})

                    additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}
                    reasoning_content = additional_kwargs.get("reasoning_content")
                    if reasoning_content:
                        reasoning_text = (
                            reasoning_content
                            if isinstance(reasoning_content, str)
                            else str(reasoning_content)
                        )
                        save_event("thinking", content=reasoning_text)
                        yield sse("thinking", {"delta": reasoning_text})

                final_text = "".join(final_buf).strip()
                finalize_assistant_message(final_text)
                yield sse("done", {"done": True})
                return

            if runtime_state["selected_route"] == ROUTE_AGENT:
                yield from stream_agent(
                    effective_agent_collection,
                    route_reason=runtime_state["route_reason"],
                )
                return

            yield sse("error", {"error": f"unsupported route: {runtime_state['selected_route']}"})

        except Exception as e:
            error_message = str(e)
            save_event("error", content=error_message)
            yield sse("error", {"error": error_message})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=_streaming_headers(),
    )


@router.post("/kakao")
async def api_auto_chet_kakao(request: Request):
    payload = await get_request_data(request)
    user_message = _extract_kakao_user_message(payload)
    callback_url = _extract_kakao_callback_url(payload)

    if not user_message:
        return JSONResponse(_make_kakao_simple_text("사용자 메시지를 읽지 못했습니다."))

    user_message = (
        f"{user_message}\n\n"
        "반드시 일반 텍스트로만 답변하세요. "
        "마크다운, 제목, 목록기호, 코드블록, 굵게, 링크 포맷을 사용하지 마세요."
    )

    model = ollama.resolve_model(payload.get("model"))
    mode = str(payload.get("mode") or "normal").strip().lower()
    think = parse_bool(payload.get("think"), default=(mode == "think"))
    sid = str(payload.get("sid") or str(uuid.uuid4())).strip()

    history_limit = parse_int(payload.get("history_limit"), DEFAULT_HISTORY_LIMIT)
    if history_limit < 0:
        history_limit = 0

    collection_name = str(payload.get("collection") or DEFAULT_AGENT_COLLECTION).strip() or DEFAULT_AGENT_COLLECTION
    _ = parse_int(payload.get("k"), 5)  # 하위 호환용으로만 consume
    where = parse_json_str(payload.get("where"), default=None)

    max_steps = parse_int(payload.get("max_steps"), DEFAULT_AGENT_MAX_STEPS)
    if max_steps < 1:
        max_steps = 1

    temperature = clamp(
        parse_float(payload.get("temperature"), DEFAULT_AGENT_TEMPERATURE),
        0.0,
        2.0,
    )

    forced_route = str(payload.get("route") or "").strip().lower()

    student_id = None
    try:
        student_id = get_logged_in_student_id(request)
    except Exception:
        student_id = None

    if callback_url:
        worker = threading.Thread(
            target=_process_kakao_callback,
            kwargs={
                "callback_url": callback_url,
                "user_message": user_message,
                "model": model,
                "mode": mode,
                "think": think,
                "sid": sid,
                "history_limit": history_limit,
                "collection_name": collection_name,
                "where": where,
                "max_steps": max_steps,
                "temperature": temperature,
                "forced_route": forced_route,
                "student_id": student_id,
            },
            daemon=True,
        )
        worker.start()
        return JSONResponse(
            {
                "version": "2.0",
                "useCallback": True,
            }
        )

    try:
        final_text = _run_auto_chet_final_only(
            user_message=user_message,
            model=model,
            mode=mode,
            think=think,
            sid=sid,
            history_limit=history_limit,
            collection_name=collection_name,
            where=where,
            max_steps=max_steps,
            temperature=temperature,
            forced_route=forced_route,
            student_id=student_id,
            route="/api/auto_chet/kakao",
        )
        return JSONResponse(_make_kakao_simple_text(final_text))
    except Exception:
        return JSONResponse(
            _make_kakao_simple_text("지금 답변이 어려워요. 잠시 후 다시 시도해 주세요.")
        )