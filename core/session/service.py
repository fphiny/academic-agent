from __future__ import annotations

from typing import Optional

from fastapi import Request

from settings.storage import GRADE_STORE


def get_logged_in_store_key(request: Request) -> Optional[str]:
    store_key = request.session.get("grade_store_key")
    if not store_key:
        return None
    if store_key not in GRADE_STORE:
        return None
    return store_key


def get_logged_in_student_id(request: Request) -> Optional[str]:
    store_key = get_logged_in_store_key(request)
    if not store_key:
        return None
    return GRADE_STORE[store_key].get("student_id")