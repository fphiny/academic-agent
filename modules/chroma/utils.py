from __future__ import annotations

import hashlib
import json
import re
from typing import Any
from urllib.parse import urljoin, urlparse

import pandas as pd
from fastapi.responses import JSONResponse


def ok(data: dict[str, Any], status_code: int = 200) -> JSONResponse:
    payload = {"ok": True}
    payload.update(data)
    return JSONResponse(content=payload, status_code=status_code)


def fail(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        content={"ok": False, "error": message},
        status_code=status_code,
    )


def parse_int(value: Any, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def parse_json_str(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def sanitize_text(text: str) -> str:
    text = re.sub(r"\r", "\n", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def normalize_url(url: str, base: str | None = None) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if base:
        u = urljoin(base, u)
    parsed = urlparse(u)
    if not parsed.scheme:
        u = "https://" + u
    return u


def same_domain(candidate: str, root_url: str, domain: str | None = None) -> bool:
    try:
        c_host = urlparse(candidate).netloc.lower()
        if domain:
            return c_host.endswith(domain.lower())
        r_host = urlparse(root_url).netloc.lower()
        return c_host.endswith(r_host) or r_host.endswith(c_host)
    except Exception:
        return False


def unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def extract_urls_from_text(text: str) -> list[str]:
    pattern = r"https?://[^\s<>()\"']+"
    return unique_keep_order(re.findall(pattern, text or ""))


def infer_url_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        col_name = str(col).strip().lower()
        if any(key in col_name for key in ["url", "link", "주소", "링크", "웹"]):
            cols.append(str(col))
    return cols


def dataframe_to_urls(df: pd.DataFrame) -> list[str]:
    urls: list[str] = []
    preferred_cols = infer_url_columns(df)
    target_cols = preferred_cols or [str(c) for c in df.columns]

    for col in target_cols:
        if col not in df.columns:
            continue
        series = df[col]
        for value in series.dropna().astype(str).tolist():
            urls.extend(extract_urls_from_text(value))
            raw = value.strip()
            if raw.startswith("http://") or raw.startswith("https://"):
                urls.append(raw)

    return unique_keep_order([normalize_url(v) for v in urls if v.strip()])


def build_doc_id(url: str, collection_name: str) -> str:
    key = f"{collection_name}::{url}".encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()[:16]
    return f"web_{digest}"


def sanitize_metadata_for_chroma(data: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}

    for key, value in (data or {}).items():
        if value is None:
            continue

        if isinstance(value, list):
            filtered = [v for v in value if v is not None and str(v).strip() != ""]
            if not filtered:
                continue
            cleaned[key] = (
                str(filtered[0])
                if len(filtered) == 1
                else ", ".join(str(v) for v in filtered)
            )
            continue

        if isinstance(value, dict):
            if value:
                cleaned[key] = json.dumps(value, ensure_ascii=False)
            continue

        if isinstance(value, str):
            if value.strip():
                cleaned[key] = value
            continue

        if isinstance(value, (int, float, bool)):
            cleaned[key] = value
            continue

        cleaned[key] = str(value)

    return cleaned