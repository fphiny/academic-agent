from __future__ import annotations

import json
import os
from typing import Any

import requests


HTML_TABLE_API_BASE_URL = os.getenv(
    "HTML_TABLE_API_BASE_URL",
    "http://210.115.229.254:8010",
)

HTML_TABLE_API_TIMEOUT = float(
    os.getenv("HTML_TABLE_API_TIMEOUT", "30")
)


def _post_preprocess_table(html: str) -> dict[str, Any]:
    html = str(html or "").strip()
    if not html:
        return {
            "ok": True,
            "row_count": 0,
            "rows": [],
            "data": "",
        }

    url = f"{HTML_TABLE_API_BASE_URL.rstrip('/')}/preprocess-table"

    try:
        response = requests.post(
            url,
            json={"html": html},
            timeout=HTML_TABLE_API_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"HTML table API 요청 실패: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("HTML table API 응답이 JSON 형식이 아닙니다.") from exc

    if not payload.get("ok"):
        detail = payload.get("detail") or payload.get("error") or payload
        raise RuntimeError(f"HTML table API 처리 실패: {detail}")

    return payload


def process_html_table_be(html: str) -> list:
    """
    기존 함수 이름 유지.

    기존 로컬 파서처럼 list[dict] 반환을 우선한다.
    API가 rows를 내려주지 않는 구버전이면 data 텍스트를 단순 행 형태로 감싸서 반환한다.
    """
    payload = _post_preprocess_table(html)

    rows = payload.get("rows")
    if isinstance(rows, list):
        return rows

    # API가 아직 rows 없이 data만 반환하는 경우의 fallback
    data = payload.get("data", "")
    if not data:
        return []

    return [
        {
            "data": str(data)
        }
    ]


def json_to_multiline_text(json_str: str) -> str:
    """
    기존 함수 이름 유지.

    기존에는 JSON 문자열을 multiline text로 바꿨지만,
    API 구조에서는 process_html_table_be() 결과 또는 JSON 문자열을 받아
    최대한 기존 출력 형식으로 변환한다.
    """
    if not json_str:
        return ""

    # 이미 API 응답 JSON이면 data 우선 사용
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, dict):
            if isinstance(parsed.get("data"), str):
                return parsed["data"]

            if isinstance(parsed.get("rows"), list):
                return _rows_to_multiline_text(parsed["rows"])

        if isinstance(parsed, list):
            return _rows_to_multiline_text(parsed)

    except Exception:
        pass

    # JSON이 아니면 기존 문자열 그대로 반환
    return str(json_str).strip()


def _rows_to_multiline_text(rows: list[Any]) -> str:
    lines: list[str] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        for key, value in row.items():
            key = str(key).strip()
            value = str(value).strip()

            if not key:
                continue

            lines.append(f"{key}: {value}")

    return "\n".join(lines)
# ---------------------------------------------------------------------
# compatibility stub: service.py debug check용
# ---------------------------------------------------------------------
def get_header_structure(thead, table):
    if not table:
        return []

    first_row = table.find("tr")
    if not first_row:
        return []

    return [
        cell.get_text(strip=True)
        for cell in first_row.find_all(["th", "td"])
    ]
