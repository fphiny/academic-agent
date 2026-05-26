from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage

from core.ollama.client import OllamaClient
from core.parse import get_request_data, parse_bool, parse_int
from core.session import get_logged_in_student_id
from core.streaming.sse import sse, sse_comment
from modules.log.service import log_service
from .recommendations import stream_recommend_courses_for_career_question

# login/route.py 에서 쓰는 실제 성적 로딩 흐름을 그대로 재사용
from settings.storage import (
    GRADE_STORE,
    load_student_grades,
    session_records_to_dataframe,
)

DEFAULT_HISTORY_LIMIT = 20

router = APIRouter(prefix="/api/chat", tags=["chat"])
ollama = OllamaClient()


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


def _send_kakao_callback(callback_url: str, text: str) -> None:
    import requests

    payload = _make_kakao_simple_text(text)
    requests.post(
        callback_url,
        json=payload,
        timeout=15,
        headers={"Content-Type": "application/json; charset=utf-8"},
    ).raise_for_status()


def _run_chat_final_only(
    *,
    user_message: str,
    model: str,
    mode: str,
    think: bool,
    sid: str,
    history_limit: int,
    student_id: Optional[str],
    route: str = "/api/chat/kakao",
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

    final_buf: list[str] = []
    saved_event_ids: list[int] = []

    def save_event(
        *,
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
                mode="chat_kakao",
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
                mode="chat_kakao",
                content=user_message,
                metadata={
                    "model": model,
                    "mode": mode,
                    "think": think,
                    "route": route,
                    "history_limit": history_limit,
                    "channel": "kakao",
                },
            )
        except Exception:
            pass

    llm = ollama.build_chat_llm(
        model=model,
        think=think,
    )
    messages = history + [HumanMessage(content=user_message)]

    for chunk in llm.stream(messages):
        content = getattr(chunk, "content", None)
        if content:
            if isinstance(content, str):
                if content:
                    final_buf.append(content)
                    save_event(event_type="delta", content=content)
            else:
                token = str(content)
                if token:
                    final_buf.append(token)
                    save_event(event_type="delta", content=token)

        additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}
        reasoning_content = additional_kwargs.get("reasoning_content")

        if reasoning_content:
            reasoning_text = (
                reasoning_content
                if isinstance(reasoning_content, str)
                else str(reasoning_content)
            )
            save_event(event_type="thinking", content=reasoning_text)

    final_text = "".join(final_buf).strip()
    if not final_text:
        raise RuntimeError("chat ended without final text")

    if student_id:
        try:
            assistant_msg = log_service.append_assistant_message(
                student_id=student_id,
                sid=sid,
                mode="chat_kakao",
                content=final_text,
                metadata={
                    "model": model,
                    "mode": mode,
                    "think": think,
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


def _process_kakao_callback(
    *,
    callback_url: str,
    user_message: str,
    model: str,
    mode: str,
    think: bool,
    sid: str,
    history_limit: int,
    student_id: Optional[str],
) -> None:
    try:
        answer = _run_chat_final_only(
            user_message=user_message,
            model=model,
            mode=mode,
            think=think,
            sid=sid,
            history_limit=history_limit,
            student_id=student_id,
            route="/api/chat/kakao",
        )
    except Exception:
        answer = "지금 답변이 어려워요. 잠시 후 다시 시도해 주세요."

    try:
        _send_kakao_callback(callback_url, answer)
    except Exception:
        pass


def _load_student_grade_df(request: Request, student_id: str):
    store_key = request.session.get("grade_store_key")

    grade_df = None

    if store_key and store_key in GRADE_STORE:
        grade_records = GRADE_STORE[store_key].get("grades", [])
        grade_df = session_records_to_dataframe(grade_records)

    if grade_df is None or grade_df.empty:
        grade_df = load_student_grades(student_id)

    if grade_df is None or grade_df.empty:
        raise RuntimeError("성적 데이터가 없습니다.")

    return grade_df


@router.post("/stream")
async def api_chat_stream(request: Request):
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

    try:
        history = log_service.build_langchain_history(
            student_id=student_id,
            sid=sid,
            limit=history_limit,
            include_roles=["user", "assistant", "system"],
        )
    except Exception:
        history = []

    try:
        log_service.append_user_message(
            student_id=student_id,
            sid=sid,
            mode=mode,
            content=user_message,
            metadata={
                "model": model,
                "mode": mode,
                "think": think,
                "route": "/api/chat/stream",
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
        yield sse(
            "meta",
            {
                "model": model,
                "mode": mode,
                "think": think,
                "history_limit": history_limit,
            },
        )

        last_ping = time.time()
        final_buf: list[str] = []

        try:
            llm = ollama.build_chat_llm(
                model=model,
                think=think,
            )
            messages = history + [HumanMessage(content=user_message)]

            for chunk in llm.stream(messages):
                now = time.time()
                if now - last_ping >= 15:
                    yield sse_comment("ping")
                    last_ping = now

                content = getattr(chunk, "content", None)
                if content:
                    if isinstance(content, str):
                        if content:
                            final_buf.append(content)
                            yield sse("delta", {"delta": content})
                    else:
                        token = str(content)
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

                    try:
                        log_service.append_event(
                            student_id=student_id,
                            sid=sid,
                            mode="chat",
                            event_type="thinking",
                            content=reasoning_text,
                            metadata={},
                        )
                    except Exception:
                        pass

                    yield sse("thinking", {"delta": reasoning_text})

            final_text = "".join(final_buf).strip()

            try:
                log_service.append_assistant_message(
                    student_id=student_id,
                    sid=sid,
                    mode="chat",
                    content=final_text,
                    metadata={
                        "model": model,
                        "mode": mode,
                        "think": think,
                        "route": "/api/chat/stream",
                        "history_limit": history_limit,
                    },
                )
            except Exception:
                pass

            yield sse("done", {"done": True})

        except Exception as e:
            error_message = str(e)

            try:
                log_service.append_event(
                    student_id=student_id,
                    sid=sid,
                    mode="chat",
                    event_type="error",
                    content=error_message,
                    metadata={},
                )
            except Exception:
                pass

            yield sse("error", {"error": error_message})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=_streaming_headers(),
    )


@router.post("/recommend-courses/stream")
async def api_recommend_courses_stream(request: Request):
    student_id = get_logged_in_student_id(request)
    if not student_id:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")

    payload = await get_request_data(request)
    question = str(payload.get("question") or "").strip()
    sid = str(payload.get("sid") or str(uuid.uuid4())).strip()

    if not question:
        return StreamingResponse(
            iter([sse("error", {"error": "question is required"})]),
            status_code=400,
            media_type="text/event-stream",
            headers=_streaming_headers(),
        )

    try:
        grade_df = _load_student_grade_df(request, student_id)
    except Exception as e:
        return StreamingResponse(
            iter([sse("error", {"error": f"grade load failed: {str(e)}"})]),
            status_code=500,
            media_type="text/event-stream",
            headers=_streaming_headers(),
        )

    try:
        log_service.append_user_message(
            student_id=student_id,
            sid=sid,
            mode="career_recommendation",
            content=question,
            metadata={
                "route": "/api/chat/recommend-courses/stream",
                "channel": "dashboard",
                "streaming": True,
            },
        )
    except Exception:
        pass

    def generate():
        yield sse_comment("connected")
        yield sse(
            "meta",
            {
                "sid": sid,
                "type": "course_recommendation",
            },
        )

        last_ping = time.time()
        final_answer_parts: list[str] = []
        final_courses: list[dict] = []

        try:
            for item in stream_recommend_courses_for_career_question(
                student_id=str(student_id),
                grade_df=grade_df,
                question=question,
            ):
                now = time.time()
                if now - last_ping >= 15:
                    yield sse_comment("ping")
                    last_ping = now

                event_name = str(item.get("event") or "").strip()
                data = item.get("data") or {}

                if event_name == "delta":
                    delta = str(data.get("delta") or "")
                    if delta:
                        final_answer_parts.append(delta)

                elif event_name == "courses":
                    courses = data.get("recommended_courses")
                    if isinstance(courses, list):
                        final_courses = courses

                yield sse(event_name, data)

            final_answer = "".join(final_answer_parts).strip()

            try:
                log_service.append_assistant_message(
                    student_id=student_id,
                    sid=sid,
                    mode="career_recommendation",
                    content=final_answer,
                    metadata={
                        "route": "/api/chat/recommend-courses/stream",
                        "recommended_courses": final_courses,
                        "channel": "dashboard",
                        "streaming": True,
                    },
                )
            except Exception:
                pass

        except Exception as e:
            error_message = str(e)

            try:
                log_service.append_event(
                    student_id=student_id,
                    sid=sid,
                    mode="career_recommendation",
                    event_type="error",
                    content=error_message,
                    metadata={},
                )
            except Exception:
                pass

            yield sse("error", {"error": error_message})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=_streaming_headers(),
    )


@router.post("/kakao")
async def api_chat_kakao(request: Request):
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
        final_text = _run_chat_final_only(
            user_message=user_message,
            model=model,
            mode=mode,
            think=think,
            sid=sid,
            history_limit=history_limit,
            student_id=student_id,
            route="/api/chat/kakao",
        )
        return JSONResponse(_make_kakao_simple_text(final_text))
    except Exception:
        return JSONResponse(
            _make_kakao_simple_text("지금 답변이 어려워요. 잠시 후 다시 시도해 주세요.")
        )