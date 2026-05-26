from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from ..base import (
    ToolResult,
    normalize_whitespace,
    strip_html_tags,
)


def google_search_items(
    query: str,
    google_api_key: str,
    google_cx: str,
    num: int = 5,
    debug_print=None,
) -> List[Dict[str, Any]]:
    if not google_api_key or not google_cx:
        raise ValueError("GOOGLE_API_KEY or GOOGLE_CX is missing")

    resp = requests.get(
        "https://oslab.hallym.ac.kr/jj_search.php?type=json&key=oslab_agent_bot&provider=all",
        params={
            "key": "oslab_agent_bot", # google_api_key,
            "cx": google_cx,
            "query": query,
            "num": 10,
            "hl": "ko",
            "gl": "kr",
            "lr": "lang_ko",
            # "safe": "off" 또는 "active"
            # "start": 1
        },
        timeout=30,
    )
    resp.raise_for_status()

    data = resp.json()

    items = []
    for item in data.get("items", []):
        items.append(
            {
                "title": normalize_whitespace(item.get("title", "")),
                "link": item.get("link", ""),
                "displayLink": item.get("displayLink", ""),
                "snippet": normalize_whitespace(item.get("snippet", "")),
                "source_engine": "google_cse",
            }
        )

    if debug_print:
        debug_print("GOOGLE_SEARCH_RAW QUERY", {"query": query, "num": num})
        debug_print("GOOGLE_SEARCH_RAW ITEMS", items)

    return items


def naver_web_search_items(
    query: str,
    naver_client_id: str,
    naver_client_secret: str,
    num: int = 5,
    debug_print=None,
) -> List[Dict[str, Any]]:
    if not naver_client_id or not naver_client_secret:
        raise ValueError("NAVER_CLIENT_ID or NAVER_CLIENT_SECRET is missing")

    display = max(1, min(num, 100))

    resp = requests.get(
        "https://openapi.naver.com/v1/search/webkr.json",
        params={
            "query": query,
            "display": display,
            "start": 1,
            "sort": "sim",
        },
        headers={
            "X-Naver-Client-Id": naver_client_id,
            "X-Naver-Client-Secret": naver_client_secret,
        },
        timeout=30,
    )
    resp.raise_for_status()

    data = resp.json()

    items = []
    for item in data.get("items", []):
        title = strip_html_tags(item.get("title", ""))
        link = item.get("link", "")
        snippet = strip_html_tags(item.get("description", ""))

        items.append(
            {
                "title": title,
                "link": link,
                "displayLink": "search.naver.com",
                "snippet": normalize_whitespace(snippet),
                "source_engine": "naver_webkr",
            }
        )

    if debug_print:
        debug_print("NAVER_WEB_SEARCH QUERY", {"query": query, "num": num})
        debug_print("NAVER_WEB_SEARCH ITEMS", items)

    return items


def search_items(
    query: str,
    num: int = 5,
    engines: Optional[List[str]] = None,
    google_api_key: str = "",
    google_cx: str = "",
    naver_client_id: str = "",
    naver_client_secret: str = "",
    debug_print=None,
) -> List[Dict[str, Any]]:
    engines = engines or ["google", "naver"]

    merged: List[Dict[str, Any]] = []

    for engine in engines:
        try:
            if engine == "google":
                merged.extend(
                    google_search_items(
                        query=query,
                        google_api_key=google_api_key,
                        google_cx=google_cx,
                        num=num,
                        debug_print=debug_print,
                    )
                )
            elif engine == "naver":
                merged.extend(
                    naver_web_search_items(
                        query=query,
                        naver_client_id=naver_client_id,
                        naver_client_secret=naver_client_secret,
                        num=num,
                        debug_print=debug_print,
                    )
                )
            else:
                if debug_print:
                    debug_print("UNKNOWN SEARCH ENGINE", engine)
        except Exception as e:
            if debug_print:
                debug_print(f"SEARCH ENGINE ERROR | {engine}", str(e))

    dedup: List[Dict[str, Any]] = []
    seen = set()

    for item in merged:
        link = (item.get("link") or "").strip()
        if not link:
            continue
        if link in seen:
            continue
        seen.add(link)
        dedup.append(item)

    if debug_print:
        debug_print(
            "MERGED SEARCH ITEMS",
            {
                "query": query,
                "engines": engines,
                "count_before_dedup": len(merged),
                "count_after_dedup": len(dedup),
            },
        )

    return dedup


def build_search_candidates_text(items: List[Dict[str, Any]]) -> str:
    lines = []

    for i, item in enumerate(items, start=1):
        lines.append(
            f"[{i}]\n"
            f"title: {item.get('title', '')}\n"
            f"link: {item.get('link', '')}\n"
            f"engine: {item.get('source_engine', '')}\n"
            f"snippet: {item.get('snippet', '')}\n"
        )

    return "\n".join(lines)


def select_search_items_for_fetch(
    items: List[Dict[str, Any]],
    max_select: int = 5,
    min_select: int = 1,
    debug_print=None,
) -> List[Dict[str, Any]]:
    if not items:
        return []

    selected: List[Dict[str, Any]] = []
    for item in items:
        if len(selected) >= max_select:
            break
        selected.append(dict(item))

    if len(selected) < min_select and items:
        selected = [dict(items[0])]

    if debug_print:
        debug_print(
            "SELECT_SEARCH_ITEMS FINAL",
            [
                {
                    "title": s.get("title"),
                    "link": s.get("link"),
                    "engine": s.get("source_engine"),
                    "snippet": s.get("snippet"),
                }
                for s in selected
            ],
            max_len=12000,
        )

    return selected


def run(
    query: str,
    google_api_key: str,
    google_cx: str,
    naver_client_id: str,
    naver_client_secret: str,
    num: int = 5,
    top_k_urls: int = 5,
    top_k_chunks: int = 5,  # 하위호환용. 지금은 안 씀.
    engines: Optional[List[str]] = None,
    debug_print=None,
) -> ToolResult:
    try:
        _ = top_k_chunks
        engines = engines or ["google"]

        if debug_print:
            debug_print(
                "EXTERNAL_SEARCH START",
                {
                    "query": query,
                    "num": num,
                    "top_k_urls": top_k_urls,
                    "engines": engines,
                },
            )

        items = search_items(
            query=query,
            num=num,
            engines=engines,
            google_api_key=google_api_key,
            google_cx=google_cx,
            naver_client_id=naver_client_id,
            naver_client_secret=naver_client_secret,
            debug_print=debug_print,
        )

        if not items:
            return ToolResult(
                name="external_search",
                ok=False,
                data={"error": "no search items found"},
            )

        if debug_print:
            debug_print(
                "SEARCH CANDIDATES TEXT",
                build_search_candidates_text(items),
                max_len=12000,
            )

        selected_items = select_search_items_for_fetch(
            items=items,
            max_select=top_k_urls,
            min_select=1,
            debug_print=debug_print,
        )

        selected_urls = []
        for item in selected_items:
            link = (item.get("link") or "").strip()
            if link:
                selected_urls.append(link)

        if debug_print:
            debug_print("EXTERNAL_SEARCH SELECTED URLS", selected_urls)

        return ToolResult(
            name="external_search",
            ok=True,
            data={
                "query": query,
                "engines": engines,
                "items": items,
                "selected_items": selected_items,
                "selected_urls": selected_urls,
            },
        )

    except Exception as e:
        if debug_print:
            debug_print("EXTERNAL_SEARCH ERROR", str(e))
        return ToolResult(
            name="external_search",
            ok=False,
            data={"error": str(e)},
        )