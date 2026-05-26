from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from core.ollama.client import OllamaClient
from core.parse import clamp, get_request_data, parse_float, parse_int, stringify_observation
from core.session import get_logged_in_student_id
from core.streaming.sse import fail, ok, sse, sse_comment
from .config import AgentConfig
from .tool import AgentTools
from .service import AgentSpeedService
from modules.log.service import log_service

DEFAULT_COLLECTION = ""
DEFAULT_AGENT_MAX_STEPS = 5
DEFAULT_AGENT_TEMPERATURE = 0.0
DEFAULT_HISTORY_LIMIT = 20

router = APIRouter(prefix="/api/agent_speed", tags=["agent_speed"])
agent_tools = AgentTools()
ollama = OllamaClient()


def build_agent_speed_service(
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


def _streaming_headers() -> Dict[str, str]:
    return {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


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


def _run_agent_speed_final_only(
    *,
    user_message: str,
    model: str,
    max_steps: int,
    temperature: float,
    collection_name: str,
    sid: str,
    student_id: Optional[int] = None,
    history_limit: int = DEFAULT_HISTORY_LIMIT,
    route: str = "/api/agent_speed/kakao",
) -> str:
    agent_service = build_agent_speed_service(
        model=model,
        max_steps=max_steps,
        temperature=temperature,
        default_collection=collection_name,
    )

    final_buf: list[str] = []
    saved_event_ids: list[int] = []

    def save_event(
        *,
        event_type: str,
        content: str = "",
        step_index: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not student_id:
            return
        try:
            ev = log_service.append_event(
                student_id=student_id,
                sid=sid,
                mode="agent_speed_kakao",
                event_type=event_type,
                content=content,
                step_index=step_index,
                model=model,
                metadata=metadata or {},
            )
            if ev and getattr(ev, "id", None):
                saved_event_ids.append(ev.id)
        except Exception:
            pass

    def parse_step_index(step_value: Any) -> Optional[int]:
        try:
            if step_value is None:
                return None
            return int(step_value)
        except (TypeError, ValueError):
            return None

    if student_id:
        try:
            log_service.append_user_message(
                student_id=student_id,
                sid=sid,
                mode="agent_speed_kakao",
                content=user_message,
                model=model,
                metadata={
                    "model": model,
                    "max_steps": max_steps,
                    "temperature": temperature,
                    "collection_name": collection_name,
                    "route": route,
                    "history_limit": history_limit,
                    "channel": "kakao",
                },
            )
        except Exception:
            pass

    for event in agent_service.run(user_message=user_message, collection_name=collection_name):
        etype = str(event.get("type", "message"))

        if etype in {"agent_step", "thinking", "thought"}:
            thought_text = str(event.get("delta", "") or event.get("step", "") or "")
            step_value = event.get("step")
            step_index = parse_step_index(step_value)
            save_event(
                event_type=etype,
                content=thought_text,
                step_index=step_index,
                metadata={"step": step_value},
            )
            continue

        if etype == "tool_call":
            save_event(
                event_type="tool_call",
                content="",
                metadata={
                    "tool": event.get("tool_name"),
                    "arguments": event.get("arguments", {}) or {},
                },
            )
            continue

        if etype == "tool_result":
            tool_payload = event.get("result", event.get("data"))
            obs = stringify_observation(tool_payload)
            save_event(event_type="tool_result", content=obs, metadata={})
            continue

        if etype == "sources":
            save_event(
                event_type="sources",
                content="",
                metadata={
                    "event": {
                        "tool_name": str(event.get("tool_name") or ""),
                        "sources": event.get("sources", []) or [],
                    }
                },
            )
            continue

        if etype == "delta":
            chunk_text = str(event.get("delta", "") or "")
            step_value = event.get("step")
            step_index = parse_step_index(step_value)
            if chunk_text:
                final_buf.append(chunk_text)
                save_event(
                    event_type="delta",
                    content=chunk_text,
                    step_index=step_index,
                    metadata={"step": step_value},
                )
            continue

        if etype == "done":
            break

        if etype == "error":
            error_message = str(event.get("error", "unknown error") or "unknown error")
            save_event(event_type="error", content=error_message, metadata={})
            raise RuntimeError(error_message)

    final_text = "".join(final_buf).strip()
    if not final_text:
        raise RuntimeError("agent_speed ended without final text")

    if student_id:
        try:
            assistant_msg = log_service.append_assistant_message(
                student_id=student_id,
                sid=sid,
                mode="agent_speed_kakao",
                content=final_text,
                model=model,
                metadata={
                    "model": model,
                    "max_steps": max_steps,
                    "temperature": temperature,
                    "collection_name": collection_name,
                    "route": route,
                    "history_limit": history_limit,
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


def _send_kakao_callback(callback_url: str, text: str) -> None:
    import requests

    payload = _make_kakao_simple_text(text)
    requests.post(
        callback_url,
        json=payload,
        timeout=15,
        headers={"Content-Type": "application/json; charset=utf-8"},
    ).raise_for_status()


def _process_kakao_callback(
    *,
    callback_url: str,
    user_message: str,
    model: str,
    max_steps: int,
    temperature: float,
    collection_name: str,
    sid: str,
    student_id: Optional[int],
    history_limit: int,
) -> None:
    try:
        answer = _run_agent_speed_final_only(
            user_message=user_message,
            model=model,
            max_steps=max_steps,
            temperature=temperature,
            collection_name=collection_name,
            sid=sid,
            student_id=student_id,
            history_limit=history_limit,
            route="/api/agent_speed/kakao",
        )
    except Exception:
        answer = "지금 답변이 어려워요. 잠시 후 다시 시도해 주세요."

    try:
        _send_kakao_callback(callback_url, answer)
    except Exception:
        pass


@router.post("/stream")
async def api_agent_speed_stream(request: Request):
    student_id = get_logged_in_student_id(request)
    if not student_id:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")

    payload = await get_request_data(request)
    query = str(payload.get("message") or "").strip()
    if not query:
        return StreamingResponse(
            iter([sse("error", {"error": "message is required"})]),
            status_code=400,
            media_type="text/event-stream",
            headers=_streaming_headers(),
        )

    model = ollama.resolve_model(payload.get("model"))
    max_steps = parse_int(payload.get("max_steps"), DEFAULT_AGENT_MAX_STEPS)
    if max_steps < 1:
        max_steps = 1

    temperature = clamp(
        parse_float(payload.get("temperature"), DEFAULT_AGENT_TEMPERATURE),
        0.0,
        2.0,
    )

    sid = str(payload.get("sid") or str(uuid.uuid4())).strip()
    history_limit = parse_int(payload.get("history_limit"), DEFAULT_HISTORY_LIMIT)
    if history_limit < 0:
        history_limit = 0

    collection_name = DEFAULT_COLLECTION
    agent_service = build_agent_speed_service(
        model=model,
        max_steps=max_steps,
        temperature=temperature,
        default_collection=collection_name,
    )

    try:
        log_service.append_user_message(
            student_id=student_id,
            sid=sid,
            mode="agent_speed",
            content=query,
            model=model,
            metadata={
                "model": model,
                "max_steps": max_steps,
                "temperature": temperature,
                "collection_name": collection_name,
                "route": "/api/agent_speed/stream",
                "history_limit": history_limit,
            },
        )
    except Exception as e:
        return StreamingResponse(
            iter([sse("error", {"error": f"log save failed: {str(e)}"})]),
            status_code=500,
            media_type="text/event-stream",
            headers=_streaming_headers(),
        )

    def generate():
        yield sse_comment("connected")
        yield sse("sid", {"sid": sid})
        yield sse("route", {"route": "/api/agent_speed/stream"})
        yield sse("mode", {"mode": "agent_speed"})
        yield sse("input", {"text": query})
        yield sse(
            "meta",
            {
                "model": model,
                "max_steps": max_steps,
                "temperature": temperature,
                "collection_name": collection_name,
                "history_limit": history_limit,
            },
        )

        last_ping = time.time()
        final_buf: list[str] = []
        saved_event_ids: list[int] = []
        final_persisted = False

        def save_event(
            *,
            event_type: str,
            content: str = "",
            step_index: Optional[int] = None,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> None:
            try:
                ev = log_service.append_event(
                    student_id=student_id,
                    sid=sid,
                    mode="agent_speed",
                    event_type=event_type,
                    content=content,
                    step_index=step_index,
                    model=model,
                    metadata=metadata or {},
                )
                if ev and getattr(ev, "id", None):
                    saved_event_ids.append(ev.id)
            except Exception:
                pass

        def parse_step_index(step_value: Any) -> Optional[int]:
            try:
                if step_value is None:
                    return None
                return int(step_value)
            except (TypeError, ValueError):
                return None

        def persist_final_answer(final_text: str) -> None:
            nonlocal final_persisted

            if not final_text or final_persisted:
                return

            assistant_msg = None
            try:
                assistant_msg = log_service.append_assistant_message(
                    student_id=student_id,
                    sid=sid,
                    mode="agent_speed",
                    content=final_text,
                    model=model,
                    metadata={
                        "model": model,
                        "max_steps": max_steps,
                        "temperature": temperature,
                        "collection_name": collection_name,
                        "route": "/api/agent_speed/stream",
                        "history_limit": history_limit,
                    },
                )
            except Exception:
                assistant_msg = None

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

            final_persisted = True

        try:
            for event in agent_service.run(user_message=query, collection_name=collection_name):
                now = time.time()
                if now - last_ping >= 15:
                    yield sse_comment("ping")
                    last_ping = now

                etype = str(event.get("type", "message"))

                if etype in {"agent_step", "thinking", "thought"}:
                    thought_text = str(event.get("delta", "") or event.get("step", "") or "")
                    step_value = event.get("step")
                    step_index = parse_step_index(step_value)
                    save_event(
                        event_type=etype,
                        content=thought_text,
                        step_index=step_index,
                        metadata={"step": step_value},
                    )
                    yield sse("thought", {"text": thought_text, "step": step_value})
                    continue

                if etype == "tool_call":
                    tool_name = event.get("tool_name")
                    arguments = event.get("arguments", {}) or {}
                    save_event(
                        event_type="tool_call",
                        content="",
                        metadata={"tool": tool_name, "arguments": arguments},
                    )
                    yield sse("action", {"tool": tool_name, "arguments": arguments})
                    continue

                if etype == "tool_result":
                    tool_payload = event.get("result", event.get("data"))
                    obs = stringify_observation(tool_payload)
                    save_event(event_type="tool_result", content=obs, metadata={})
                    yield sse("observation", {"text": obs})
                    continue

                if etype == "sources":
                    tool_name = str(event.get("tool_name") or "")
                    sources = event.get("sources", []) or []
                    save_event(
                        event_type="sources",
                        content="",
                        metadata={"event": {"tool_name": tool_name, "sources": sources}},
                    )
                    yield sse("sources", {"tool_name": tool_name, "sources": sources})
                    continue

                if etype == "delta":
                    chunk_text = str(event.get("delta", "") or "")
                    step_value = event.get("step")
                    step_index = parse_step_index(step_value)

                    if chunk_text:
                        final_buf.append(chunk_text)
                        save_event(
                            event_type="delta",
                            content=chunk_text,
                            step_index=step_index,
                            metadata={"step": step_value},
                        )
                        yield sse("chunk", {"text": chunk_text, "step": step_value})
                    continue

                if etype == "done":
                    final_text = "".join(final_buf).strip()
                    if not final_text:
                        yield sse("error", {"error": "agent_speed stream ended without final text"})
                        return

                    persist_final_answer(final_text)
                    yield sse("final", {"text": final_text})
                    yield sse("done", {"done": True})
                    return

                if etype == "error":
                    error_message = str(event.get("error", "unknown error") or "unknown error")
                    save_event(event_type="error", content=error_message, metadata={})
                    yield sse("error", {"error": error_message})
                    return

            final_text = "".join(final_buf).strip()
            if final_text:
                persist_final_answer(final_text)
                yield sse("final", {"text": final_text})
                yield sse("done", {"done": True})
                return

            yield sse("error", {"error": "agent_speed stream ended without done"})
        except Exception as e:
            yield sse("error", {"error": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=_streaming_headers(),
    )


@router.post("/kakao")
async def api_agent_speed_kakao(request: Request):
    payload = await get_request_data(request)
    user_message = _extract_kakao_user_message(payload)
    callback_url = _extract_kakao_callback_url(payload)

    if not user_message:
        return JSONResponse(_make_kakao_simple_text("사용자 메시지를 읽지 못했습니다."))

    user_message = (
        f"(Markdown prohibited) {user_message}\n\n"
        ""
    )

    model = ollama.resolve_model(payload.get("model"))
    max_steps = parse_int(payload.get("max_steps"), DEFAULT_AGENT_MAX_STEPS)
    if max_steps < 1:
        max_steps = 1

    temperature = clamp(
        parse_float(payload.get("temperature"), DEFAULT_AGENT_TEMPERATURE),
        0.0,
        2.0,
    )

    sid = str(payload.get("sid") or str(uuid.uuid4())).strip()
    history_limit = parse_int(payload.get("history_limit"), DEFAULT_HISTORY_LIMIT)
    if history_limit < 0:
        history_limit = 0

    collection_name = DEFAULT_COLLECTION

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
                "max_steps": max_steps,
                "temperature": temperature,
                "collection_name": collection_name,
                "sid": sid,
                "student_id": student_id,
                "history_limit": history_limit,
            },
            daemon=True,
        )
        worker.start()
        return JSONResponse({
            "version": "2.0",
            "useCallback": True,
        })

    try:
        final_text = _run_agent_speed_final_only(
            user_message=user_message,
            model=model,
            max_steps=max_steps,
            temperature=temperature,
            collection_name=collection_name,
            sid=sid,
            student_id=student_id,
            history_limit=history_limit,
            route="/api/agent_speed/kakao",
        )
        return JSONResponse(_make_kakao_simple_text(final_text))
    except Exception:
        return JSONResponse(
            _make_kakao_simple_text("지금 답변이 어려워요. 잠시 후 다시 시도해 주세요.")
        )


@router.get("/tools")
async def api_agent_speed_tools(request: Request):
    student_id = get_logged_in_student_id(request)
    if not student_id:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    return ok({"tools": agent_tools.get_tool_schemas()})


@router.post("/tools/call")
async def api_agent_speed_tools_call(request: Request):
    student_id = get_logged_in_student_id(request)
    if not student_id:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    try:
        payload = await get_request_data(request)
        name = str(payload.get("name") or "").strip()
        arguments = payload.get("arguments") or {}
        if not name:
            return fail("name is required", 400)
        result = agent_tools.call_tool(name, arguments)
        return ok({"name": result.name, "ok_result": result.ok, "data": result.data})
    except Exception as e:
        return fail(str(e), 500)