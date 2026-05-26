from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from core.ollama.client import OllamaClient
from core.parse import get_request_data, parse_bool, parse_int, parse_json_str
from core.session import get_logged_in_student_id
from core.streaming.sse import sse, sse_comment
from modules.chroma.alias_store import resolve_alias
from modules.log.service import log_service
from modules.rag.service import answer_query_stream
from settings.config import TEMPLATES_DIR

DEFAULT_COLLECTION = ""
DEFAULT_TOP_K = 5
DEFAULT_HISTORY_LIMIT = 20

router = APIRouter(tags=["rag"])
templates = Jinja2Templates(directory=TEMPLATES_DIR)
ollama = OllamaClient()


def get_can_access_db(request: Request) -> bool:
    return bool(request.session.get("can_access_db"))


def require_login(request: Request) -> str:
    student_id = get_logged_in_student_id(request)

    if not student_id:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")

    return student_id


def require_db_access(request: Request) -> str:
    student_id = require_login(request)

    if not get_can_access_db(request):
        raise HTTPException(status_code=403, detail="DB 접속 권한이 없습니다.")

    return student_id


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


def _run_rag_final_only(
    *,
    query: str,
    collection_name: str,
    model: str,
    mode: str,
    think: bool,
    k: int,
    where: Any,
    sid: str,
    history_limit: int,
    student_id: Optional[int],
    route: str = "/api/rag/kakao",
) -> str:
    resolved_collection_name = resolve_alias(collection_name)

    try:
        history_records = (
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
        history_records = []

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
                mode="rag_kakao",
                sid=sid,
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
                mode="rag_kakao",
                content=query,
                metadata={
                    "model": model,
                    "mode": mode,
                    "think": think,
                    "k": k,
                    "collection_name": collection_name,
                    "resolved_collection_name": resolved_collection_name,
                    "where": where,
                    "route": route,
                    "history_limit": history_limit,
                    "source_mode": "rag",
                    "channel": "kakao",
                },
            )
        except Exception:
            pass

    for event in answer_query_stream(
        query=query,
        history_records=history_records,
        collection_name=collection_name,
        k=k,
        where=where,
        model=model,
        think=think,
    ):
        etype = str(event.get("type", "message"))

        if etype == "thinking":
            thinking_text = str(event.get("delta", "") or "")
            if thinking_text:
                save_event(
                    event_type="thinking",
                    content=thinking_text,
                    metadata={"source_mode": "rag"},
                )
            continue

        if etype == "delta":
            delta_text = str(event.get("delta", "") or "")
            if delta_text:
                final_buf.append(delta_text)
                save_event(
                    event_type="delta",
                    content=delta_text,
                    metadata={"source_mode": "rag"},
                )
            continue

        if etype == "sources":
            save_event(
                event_type="sources",
                content="",
                metadata={
                    "source_mode": "rag",
                    "event": event,
                },
            )
            continue

        if etype == "context":
            save_event(
                event_type="context",
                content="",
                metadata={
                    "source_mode": "rag",
                    "event": event,
                },
            )
            continue

        if etype == "meta":
            save_event(
                event_type="meta",
                content="",
                metadata={
                    "source_mode": "rag",
                    "event": event,
                },
            )
            continue

        if etype == "done":
            break

        if etype == "error":
            error_message = str(event.get("error", "unknown error") or "unknown error")
            save_event(
                event_type="error",
                content=error_message,
                metadata={"source_mode": "rag"},
            )
            raise RuntimeError(error_message)

        save_event(
            event_type=etype,
            content="",
            metadata={
                "source_mode": "rag",
                "event": event,
            },
        )

    final_text = "".join(final_buf).strip()
    if not final_text:
        raise RuntimeError("rag ended without final text")

    if student_id:
        try:
            assistant_msg = log_service.append_assistant_message(
                student_id=student_id,
                mode="rag_kakao",
                sid=sid,
                content=final_text,
                metadata={
                    "model": model,
                    "mode": mode,
                    "think": think,
                    "k": k,
                    "collection_name": collection_name,
                    "resolved_collection_name": resolved_collection_name,
                    "where": where,
                    "route": route,
                    "history_limit": history_limit,
                    "source_mode": "rag",
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
    query: str,
    collection_name: str,
    model: str,
    mode: str,
    think: bool,
    k: int,
    where: Any,
    sid: str,
    history_limit: int,
    student_id: Optional[int],
) -> None:
    try:
        answer = _run_rag_final_only(
            query=query,
            collection_name=collection_name,
            model=model,
            mode=mode,
            think=think,
            k=k,
            where=where,
            sid=sid,
            history_limit=history_limit,
            student_id=student_id,
            route="/api/rag/kakao",
        )
    except Exception:
        answer = "지금 답변이 어려워요. 잠시 후 다시 시도해 주세요."

    try:
        _send_kakao_callback(callback_url, answer)
    except Exception:
        pass


@router.get("/rag", response_class=HTMLResponse)
async def rag_index(request: Request):
    path = os.path.join(TEMPLATES_DIR, "index.html")
    student_id = get_logged_in_student_id(request)

    if not student_id:
        return RedirectResponse(url="/", status_code=302)

    if os.path.exists(path):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "student_id": student_id,
                "can_access_db": get_can_access_db(request),
            },
        )
    return HTMLResponse("<h2>index.html 없음</h2>")


@router.get("/rag/documents", response_class=HTMLResponse)
async def rag_documents_page(request: Request):
    path = os.path.join(TEMPLATES_DIR, "documents.html")
    student_id = require_db_access(request)

    if os.path.exists(path):
        return templates.TemplateResponse(
            "documents.html",
            {
                "request": request,
                "student_id": student_id,
                "can_access_db": True,
            },
        )
    return HTMLResponse("<h2>documents.html 없음</h2>")


@router.get("/rag/collections", response_class=HTMLResponse)
async def rag_collections_page(request: Request):
    path = os.path.join(TEMPLATES_DIR, "collections.html")
    student_id = require_db_access(request)

    if os.path.exists(path):
        return templates.TemplateResponse(
            "collections.html",
            {
                "request": request,
                "student_id": student_id,
                "can_access_db": True,
            },
        )
    return HTMLResponse("<h2>collections.html 없음</h2>")


@router.get("/rag/web_docs", response_class=HTMLResponse)
async def rag_web_docs_page(request: Request):
    path = os.path.join(TEMPLATES_DIR, "web_docs.html")
    student_id = require_db_access(request)

    if os.path.exists(path):
        return templates.TemplateResponse(
            "web_docs.html",
            {
                "request": request,
                "student_id": student_id,
                "can_access_db": True,
            },
        )
    return HTMLResponse("<h2>web_docs.html 없음</h2>")


@router.post("/api/rag/stream")
async def api_rag_stream(request: Request):
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

    collection_name = str(payload.get("collection") or DEFAULT_COLLECTION).strip()
    model = ollama.resolve_model(payload.get("model"))
    mode = str(payload.get("mode") or "normal").strip().lower()
    think = parse_bool(payload.get("think"), default=(mode == "think"))
    k = parse_int(payload.get("k"), DEFAULT_TOP_K)
    where = parse_json_str(payload.get("where"), default=None)
    sid = str(payload.get("sid") or str(uuid.uuid4())).strip()

    history_limit = parse_int(payload.get("history_limit"), DEFAULT_HISTORY_LIMIT)
    if history_limit < 0:
        history_limit = 0

    resolved_collection_name = resolve_alias(collection_name)

    try:
        history_records = log_service.build_langchain_history(
            student_id=student_id,
            sid=sid,
            limit=history_limit,
            include_roles=["user", "assistant", "system"],
        )
    except Exception:
        history_records = []

    try:
        log_service.append_user_message(
            student_id=student_id,
            sid=sid,
            mode=mode,
            content=query,
            metadata={
                "model": model,
                "mode": mode,
                "think": think,
                "k": k,
                "collection_name": collection_name,
                "resolved_collection_name": resolved_collection_name,
                "where": where,
                "route": "/api/rag/stream",
                "history_limit": history_limit,
                "source_mode": "rag",
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
                "query": query,
                "collection_name": collection_name,
                "resolved_collection_name": resolved_collection_name,
                "model": model,
                "think": think,
                "k": k,
                "where": where,
                "history_limit": history_limit,
            },
        )

        last_ping = time.time()
        final_buf: list[str] = []

        try:
            for event in answer_query_stream(
                query=query,
                history_records=history_records,
                collection_name=collection_name,
                k=k,
                where=where,
                model=model,
                think=think,
            ):
                now = time.time()
                if now - last_ping >= 15:
                    yield sse_comment("ping")
                    last_ping = now

                etype = event.get("type", "message")

                if etype == "thinking":
                    thinking_text = str(event.get("delta", ""))

                    try:
                        log_service.append_event(
                            student_id=student_id,
                            mode=mode,
                            sid=sid,
                            event_type="thinking",
                            content=thinking_text,
                            metadata={
                                "source_mode": "rag",
                            },
                        )
                    except Exception:
                        pass

                    yield sse("thinking", {"delta": thinking_text})

                elif etype == "delta":
                    delta_text = str(event.get("delta", ""))
                    if delta_text:
                        final_buf.append(delta_text)
                    yield sse("delta", {"delta": delta_text})

                elif etype == "sources":
                    try:
                        log_service.append_event(
                            student_id=student_id,
                            mode=mode,
                            sid=sid,
                            event_type="sources",
                            content="",
                            metadata={
                                "source_mode": "rag",
                                "event": event,
                            },
                        )
                    except Exception:
                        pass

                    yield sse("sources", event)

                elif etype == "context":
                    try:
                        log_service.append_event(
                            student_id=student_id,
                            mode=mode,
                            sid=sid,
                            event_type="context",
                            content="",
                            metadata={
                                "source_mode": "rag",
                                "event": event,
                            },
                        )
                    except Exception:
                        pass

                    yield sse("context", event)

                elif etype == "meta":
                    try:
                        log_service.append_event(
                            student_id=student_id,
                            mode=mode,
                            sid=sid,
                            event_type="meta",
                            content="",
                            metadata={
                                "source_mode": "rag",
                                "event": event,
                            },
                        )
                    except Exception:
                        pass

                    yield sse("rag_meta", event)

                elif etype == "done":
                    final_text = "".join(final_buf).strip()

                    try:
                        log_service.append_assistant_message(
                            student_id=student_id,
                            mode=mode,
                            sid=sid,
                            content=final_text,
                            metadata={
                                "model": model,
                                "mode": mode,
                                "think": think,
                                "k": k,
                                "collection_name": collection_name,
                                "resolved_collection_name": resolved_collection_name,
                                "where": where,
                                "route": "/api/rag/stream",
                                "history_limit": history_limit,
                                "source_mode": "rag",
                            },
                        )
                    except Exception:
                        pass

                    yield sse("done", {"done": True})

                elif etype == "error":
                    error_message = str(event.get("error", "unknown error"))

                    try:
                        log_service.append_event(
                            student_id=student_id,
                            mode=mode,
                            sid=sid,
                            event_type="error",
                            content=error_message,
                            metadata={
                                "source_mode": "rag",
                            },
                        )
                    except Exception:
                        pass

                    yield sse("error", {"error": error_message})

                else:
                    yield sse(etype, event)

        except Exception as e:
            error_message = str(e)

            try:
                log_service.append_event(
                    student_id=student_id,
                    mode=mode,
                    sid=sid,
                    event_type="error",
                    content=error_message,
                    metadata={
                        "source_mode": "rag",
                    },
                )
            except Exception:
                pass

            yield sse("error", {"error": error_message})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=_streaming_headers(),
    )


@router.post("/api/rag/kakao")
async def api_rag_kakao(request: Request):
    payload = await get_request_data(request)
    query = _extract_kakao_user_message(payload)
    callback_url = _extract_kakao_callback_url(payload)

    if not query:
        return JSONResponse(_make_kakao_simple_text("사용자 메시지를 읽지 못했습니다."))

    query = (
        f"{query}\n\n"
        "반드시 일반 텍스트로만 답변하세요. "
        "마크다운, 제목, 목록기호, 코드블록, 굵게, 링크 포맷을 사용하지 마세요."
    )

    collection_name = str(payload.get("collection") or DEFAULT_COLLECTION).strip()
    model = ollama.resolve_model(payload.get("model"))
    mode = str(payload.get("mode") or "normal").strip().lower()
    think = parse_bool(payload.get("think"), default=(mode == "think"))
    k = parse_int(payload.get("k"), DEFAULT_TOP_K)
    where = parse_json_str(payload.get("where"), default=None)
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
                "query": query,
                "collection_name": collection_name,
                "model": model,
                "mode": mode,
                "think": think,
                "k": k,
                "where": where,
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
        final_text = _run_rag_final_only(
            query=query,
            collection_name=collection_name,
            model=model,
            mode=mode,
            think=think,
            k=k,
            where=where,
            sid=sid,
            history_limit=history_limit,
            student_id=student_id,
            route="/api/rag/kakao",
        )
        return JSONResponse(_make_kakao_simple_text(final_text))
    except Exception:
        return JSONResponse(
            _make_kakao_simple_text("지금 답변이 어려워요. 잠시 후 다시 시도해 주세요.")
        )

@router.get("/rag/guideline", response_class=HTMLResponse)
async def rag_guideline_page(request: Request):
    path = os.path.join(TEMPLATES_DIR, "guideline.html")
    student_id = require_db_access(request)

    if os.path.exists(path):
        return templates.TemplateResponse(
            "guideline.html",
            {
                "request": request,
                "student_id": student_id,
                "can_access_db": True,
            },
        )
    return HTMLResponse("<h2>guideline.html 없음</h2>")