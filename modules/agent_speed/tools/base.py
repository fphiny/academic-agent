from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List
from urllib.parse import urlparse


@dataclass
class ToolResult:
    name: str
    ok: bool
    data: Dict[str, Any]


def debug_print(enabled: bool, label: str, text: Any, max_len: int = 5000) -> None:
    if not enabled:
        return

    try:
        rendered = text if isinstance(text, str) else repr(text)
    except Exception:
        rendered = "<unprintable>"

    print("\n" + "=" * 120)
    print(f"[DEBUG] {label}")
    print("-" * 120)
    print(rendered[:max_len])
    if len(rendered) > max_len:
        print(f"\n... (truncated, total={len(rendered)} chars)")
    print("=" * 120 + "\n")


def normalize_string_list_input(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if isinstance(value, tuple):
        return [str(x).strip() for x in value if str(x).strip()]

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []

        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

        return [s]

    return [str(value).strip()]


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\xa0", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_block_text(text: str) -> str:
    return normalize_whitespace(text)


def strip_html_tags(text: str) -> str:
    if not text:
        return ""

    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def safe_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().strip()
    except Exception:
        return ""