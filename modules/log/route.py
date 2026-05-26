from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from modules.log.service import log_service
from settings.storage import GRADE_STORE

router = APIRouter(prefix="/api/log", tags=["log"])


def get_session(request: Request) -> Dict[str, Any]:
    return request.session


def get_logged_in_store_key(session: Dict[str, Any]) -> Optional[str]:
    store_key = session.get("grade_store_key")
    if not store_key:
        return None
    if store_key not in GRADE_STORE:
        return None
    return store_key


def get_logged_in_student_id(session: Dict[str, Any]) -> Optional[str]:
    store_key = get_logged_in_store_key(session)
    if not store_key:
        return None
    return GRADE_STORE[store_key].get("student_id")


async def require_logged_in_student_id(
    session: Dict[str, Any] = Depends(get_session),
) -> str:
    student_id = get_logged_in_student_id(session)
    if not student_id:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    return student_id


def ok(data: Dict[str, Any], status_code: int = 200) -> JSONResponse:
    payload = {"ok": True}
    payload.update(data)
    return JSONResponse(content=payload, status_code=status_code)


def fail(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        content={"ok": False, "error": message},
        status_code=status_code,
    )


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.isoformat()

    if is_dataclass(value):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]

    return value

@router.delete("/conversations")
async def api_delete_conversations(request: Request):
    student_id = get_logged_in_student_id(request.session)  # 맞음
    if not student_id:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")

    deleted_count = log_service.delete_conversations(
        student_id=student_id,
    )

    return ok({
        "deleted": True,
        "deleted_count": deleted_count,
    })
    
@router.get("/conversations")
async def api_log_list_conversations(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    student_id: str = Depends(require_logged_in_student_id),
):
    try:
        items = log_service.list_conversations(
            student_id=student_id,
            limit=limit,
            offset=offset,
        )

        return ok(
            {
                "student_id": student_id,
                "items": [to_jsonable(item) for item in items],
                "count": len(items),
                "limit": limit,
                "offset": offset,
            }
        )
    except Exception as e:
        return fail(str(e), 500)


@router.get("/conversations/{sid}")
async def api_log_get_conversation_detail(
    sid: str,
    student_id: str = Depends(require_logged_in_student_id),
):
    sid = sid.strip()

    if not sid:
        return fail("sid is required", 400)

    try:
        detail = log_service.get_conversation_detail(
            student_id=student_id,
            sid=sid,
        )
        if detail is None:
            return fail("conversation not found", 404)

        return ok(
            {
                "student_id": student_id,
                "sid": sid,
                "conversation": to_jsonable(detail["conversation"]),
                "messages": [to_jsonable(item) for item in detail["messages"]],
                "events": [to_jsonable(item) for item in detail["events"]],
            }
        )
    except Exception as e:
        return fail(str(e), 500)


@router.get("/conversations/{sid}/messages")
async def api_log_list_messages(
    sid: str,
    limit: Optional[int] = Query(default=None, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    ascending: bool = Query(default=True),
    student_id: str = Depends(require_logged_in_student_id),
):
    sid = sid.strip()

    if not sid:
        return fail("sid is required", 400)

    try:
        conversation = log_service.get_conversation(
            student_id=student_id,
            sid=sid,
        )
        if conversation is None:
            return fail("conversation not found", 404)

        items = log_service.list_messages(
            student_id=student_id,
            sid=sid,
            limit=limit,
            offset=offset,
            ascending=ascending,
        )

        return ok(
            {
                "student_id": student_id,
                "sid": sid,
                "conversation": to_jsonable(conversation),
                "items": [to_jsonable(item) for item in items],
                "count": len(items),
                "limit": limit,
                "offset": offset,
                "ascending": ascending,
            }
        )
    except Exception as e:
        return fail(str(e), 500)



@router.get("/conversations/{sid}/events")
async def api_log_list_events(
    sid: str,
    limit: Optional[int] = Query(default=None, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    ascending: bool = Query(default=True),
    event_type: Optional[str] = Query(default=None),
    student_id: str = Depends(require_logged_in_student_id),
):
    sid = sid.strip()
    normalized_event_type = str(event_type).strip().lower() if event_type else None

    if not sid:
        return fail("sid is required", 400)

    try:
        conversation = log_service.get_conversation(
            student_id=student_id,
            sid=sid,
        )
        if conversation is None:
            return fail("conversation not found", 404)

        items = log_service.list_events(
            student_id=student_id,
            sid=sid,
            limit=limit,
            offset=offset,
            ascending=ascending,
            event_type=normalized_event_type,
        )

        return ok(
            {
                "student_id": student_id,
                "sid": sid,
                "conversation": to_jsonable(conversation),
                "items": [to_jsonable(item) for item in items],
                "count": len(items),
                "limit": limit,
                "offset": offset,
                "ascending": ascending,
                "event_type": normalized_event_type,
            }
        )
    except Exception as e:
        return fail(str(e), 500)


@router.delete("/conversations/{sid}")
async def api_log_delete_conversation(
    sid: str,
    student_id: str = Depends(require_logged_in_student_id),
):
    sid = sid.strip()

    if not sid:
        return fail("sid is required", 400)

    try:
        deleted = log_service.delete_conversation(
            student_id=student_id,
            sid=sid,
        )

        if not deleted:
            return fail("conversation not found", 404)

        return ok(
            {
                "student_id": student_id,
                "sid": sid,
                "deleted": True,
            }
        )
    except Exception as e:
        return fail(str(e), 500)
