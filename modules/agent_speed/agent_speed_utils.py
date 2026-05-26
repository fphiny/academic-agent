from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urldefrag, urlparse


def normalize_stream_chunk_content(
    content: Any,
    text_normalizer: Optional[Callable[[Any], str]] = None,
) -> str:
    """
    스트리밍 chunk content 를 안전하게 문자열로 정규화한다.

    Parameters
    ----------
    content:
        원본 content 값
    text_normalizer:
        예: self.ollama.normalize_text_content 같은 callable.
        주입되지 않으면 fallback 으로 str(content) 사용.

    Returns
    -------
    str
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if text_normalizer is not None:
        try:
            return text_normalizer(content)
        except Exception:
            pass

    return str(content)


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """
    JSON 문자열을 안전하게 파싱한다.
    응답 앞뒤에 잡텍스트가 섞인 경우 첫 '{' ~ 마지막 '}' 구간도 재시도한다.
    """
    raw = (text or "").strip()
    if not raw:
        return None

    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidate = raw[start : end + 1]
        try:
            data = json.loads(candidate)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    return None


def normalize_whitespace(text: str) -> str:
    """
    연속 공백/개행을 하나의 공백으로 정규화한다.
    """
    return re.sub(r"\s+", " ", str(text or "")).strip()


def truncate_text(text: Any, limit: int = 4000) -> str:
    """
    텍스트를 지정 길이까지만 잘라 반환한다.
    """
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit]


def canonicalize_url(url: str) -> str:
    """
    URL canonicalization:
    - mailto:, javascript:, tel: 제외
    - fragment 제거
    - params 제거
    - 루트가 아닌 trailing slash 제거
    """
    raw = str(url or "").strip()
    if not raw:
        return ""

    if raw.startswith(("mailto:", "javascript:", "tel:")):
        return ""

    raw, _ = urldefrag(raw)
    parsed = urlparse(raw)

    if not parsed.scheme or not parsed.netloc:
        return raw

    normalized = parsed._replace(fragment="", params="")
    final = normalized.geturl()
    root = f"{parsed.scheme}://{parsed.netloc}/"

    if final.endswith("/") and final != root:
        final = final.rstrip("/")

    return final


def is_same_site(base_url: str, other_url: str) -> bool:
    """
    두 URL 의 netloc 이 같은지 비교한다.
    """
    try:
        a = urlparse(str(base_url or "").strip())
        b = urlparse(str(other_url or "").strip())
        return bool(a.netloc) and a.netloc == b.netloc
    except Exception:
        return False


def dedupe_texts(values: List[str]) -> List[str]:
    """
    공백 정규화 + case-insensitive 기준으로 중복 제거.
    """
    seen = set()
    result: List[str] = []

    for value in values:
        clean = normalize_whitespace(value)
        if not clean:
            continue

        key = clean.lower()
        if key in seen:
            continue

        seen.add(key)
        result.append(clean)

    return result