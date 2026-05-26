from __future__ import annotations

import json
from typing import Any, Dict


def sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\n" + f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_comment(text: str) -> str:
    return f": {text}\n\n"

def ok(data: Dict[str, Any], status_code: int = 200):
    payload = {"ok": True}
    payload.update(data)
    return JSONResponse(content=payload, status_code=status_code)


def fail(message: str, status_code: int = 400):
    return JSONResponse(
        content={"ok": False, "error": message},
        status_code=status_code,
    )
