from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import Depends, Request

from settings.storage import GRADE_STORE


def get_session(request: Request) -> Dict[Any, Any]:
    return request.session


def get_logged_in_store_key(session: Dict[Any, Any]) -> Optional[str]:
    store_key = session.get("grade_store_key")
    if not store_key:
        return None
    if store_key not in GRADE_STORE:
        return None
    return store_key


def get_logged_in_student_id(session: Dict[Any, Any]) -> Optional[str]:
    store_key = get_logged_in_store_key(session)
    if not store_key:
        return None
    return GRADE_STORE[store_key].get("student_id")


async def check_login_status(
    session: Dict[Any, Any] = Depends(get_session),
) -> Optional[str]:
    return get_logged_in_student_id(session)