from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests

from ..base import ToolResult

MENU_API_URL = "http://oslab.hallym.ac.kr:8001/menu"
KST = ZoneInfo("Asia/Seoul")


def _safe_debug(debug_print: Optional[Callable[..., None]], label: str, payload: Any) -> None:
    if debug_print is None:
        return
    try:
        debug_print(label, payload)
    except Exception:
        pass


def _today_kst() -> datetime:
    return datetime.now(KST)


def _normalize_date_string(date: Any) -> str:
    raw = str(date or "").strip()
    if not raw:
        return _today_kst().strftime("%Y%m%d")

    lowered = raw.lower().strip()
    today = _today_kst().date()

    if lowered in {"today", "오늘"}:
        return today.strftime("%Y%m%d")
    if lowered in {"tomorrow", "내일"}:
        return (today + timedelta(days=1)).strftime("%Y%m%d")
    if lowered in {"모레"}:
        return (today + timedelta(days=2)).strftime("%Y%m%d")
    if lowered in {"글피"}:
        return (today + timedelta(days=3)).strftime("%Y%m%d")

    digits_only = re.sub(r"[^0-9]", "", raw)
    if len(digits_only) == 8:
        try:
            dt = datetime.strptime(digits_only, "%Y%m%d")
            return dt.strftime("%Y%m%d")
        except Exception:
            pass

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.strftime("%Y%m%d")
        except Exception:
            pass

    month_day_match = re.fullmatch(r"\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일?\s*", raw)
    if month_day_match:
        month = int(month_day_match.group(1))
        day = int(month_day_match.group(2))
        year = today.year
        dt = datetime(year, month, day)
        return dt.strftime("%Y%m%d")

    month_day_slash_match = re.fullmatch(r"\s*(\d{1,2})\s*[/.-]\s*(\d{1,2})\s*", raw)
    if month_day_slash_match:
        month = int(month_day_slash_match.group(1))
        day = int(month_day_slash_match.group(2))
        year = today.year
        dt = datetime(year, month, day)
        return dt.strftime("%Y%m%d")

    raise ValueError(f"지원하지 않는 날짜 형식입니다: {raw}")


def _flatten_menu_items(value: Any) -> List[str]:
    result: List[str] = []

    def visit(node: Any) -> None:
        if node is None:
            return

        if isinstance(node, str):
            text = node.strip()
            if text:
                result.append(text)
            return

        if isinstance(node, (int, float, bool)):
            result.append(str(node))
            return

        if isinstance(node, list):
            for item in node:
                visit(item)
            return

        if isinstance(node, dict):
            preferred_keys = [
                "menu",
                "menus",
                "items",
                "list",
                "data",
                "content",
                "text",
                "name",
                "value",
                "lunch",
                "dinner",
                "breakfast",
            ]
            found_preferred = False
            for key in preferred_keys:
                if key in node:
                    found_preferred = True
                    visit(node.get(key))

            if found_preferred:
                return

            for item in node.values():
                visit(item)
            return

    visit(value)

    deduped: List[str] = []
    seen = set()
    for item in result:
        clean = re.sub(r"\s+", " ", item).strip()
        if not clean:
            continue
        if clean in seen:
            continue
        seen.add(clean)
        deduped.append(clean)
    return deduped


def _extract_menu_payload(payload: Any, normalized_date: str) -> Dict[str, Any]:
    if isinstance(payload, dict):
        resolved_date = str(
            payload.get("date")
            or payload.get("menu_date")
            or payload.get("target_date")
            or normalized_date
        ).strip() or normalized_date

        items = _flatten_menu_items(
            payload.get("menu")
            if "menu" in payload
            else payload.get("menus")
            if "menus" in payload
            else payload.get("items")
            if "items" in payload
            else payload.get("data")
            if "data" in payload
            else payload
        )

        text = str(payload.get("text") or "").strip()
        if not text and items:
            text = "\n".join(items)

        return {
            "date": resolved_date,
            "items": items,
            "text": text,
            "raw": payload,
        }

    if isinstance(payload, list):
        items = _flatten_menu_items(payload)
        text = "\n".join(items) if items else ""
        return {
            "date": normalized_date,
            "items": items,
            "text": text,
            "raw": payload,
        }

    text = str(payload or "").strip()
    items = [line.strip() for line in text.splitlines() if line.strip()]
    return {
        "date": normalized_date,
        "items": items,
        "text": text,
        "raw": payload,
    }


def _response_to_payload(response: requests.Response) -> Any:
    content_type = (response.headers.get("Content-Type") or "").lower()

    if "application/json" in content_type:
        return response.json()

    text = response.text.strip()

    try:
        return response.json()
    except Exception:
        pass

    if text.startswith("{") or text.startswith("["):
        try:
            return json.loads(text)
        except Exception:
            pass

    return text


def run(
    *,
    date: Any,
    debug_print: Optional[Callable[..., None]] = None,
) -> ToolResult:
    try:
        normalized_date = _normalize_date_string(date)
    except Exception as e:
        return ToolResult(
            name="get_menu",
            ok=False,
            data={
                "error": str(e),
                "input_date": date,
            },
        )

    params = {"date": normalized_date}

    _safe_debug(
        debug_print,
        "GET_MENU REQUEST",
        {
            "url": MENU_API_URL,
            "params": params,
            "input_date": date,
            "normalized_date": normalized_date,
        },
    )

    try:
        response = requests.get(MENU_API_URL, params=params, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return ToolResult(
            name="get_menu",
            ok=False,
            data={
                "error": f"menu api request failed: {str(e)}",
                "url": MENU_API_URL,
                "params": params,
                "date": normalized_date,
            },
        )

    try:
        payload = _response_to_payload(response)
        parsed = _extract_menu_payload(payload, normalized_date)

        if not parsed["text"] and parsed["items"]:
            parsed["text"] = "\n".join(parsed["items"])

        if not parsed["text"]:
            parsed["text"] = f"{parsed['date']} 메뉴 정보가 없습니다."

        result_data = {
            "date": parsed["date"],
            "text": parsed["text"],
            "items": parsed["items"],
            "url": str(response.url),
            "status_code": response.status_code,
            "raw": parsed["raw"],
        }

        _safe_debug(debug_print, "GET_MENU RESPONSE", result_data)

        return ToolResult(
            name="get_menu",
            ok=True,
            data=result_data,
        )
    except Exception as e:
        return ToolResult(
            name="get_menu",
            ok=False,
            data={
                "error": f"menu response parse failed: {str(e)}",
                "date": normalized_date,
                "url": str(response.url),
                "status_code": response.status_code,
                "response_text_preview": response.text[:1000],
            },
        )