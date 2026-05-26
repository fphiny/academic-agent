from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from bs4 import BeautifulSoup
from fastapi import Request
from fastapi.responses import JSONResponse

from modules.chroma.alias_store import resolve_alias
from modules.chroma.ingest import ingest_html_semantic
from modules.chroma.schemas import EmbedUrlsRequest
from modules.chroma.store import get_store
from modules.chroma.utils import (
    build_doc_id,
    fail,
    normalize_url,
    ok,
    sanitize_metadata_for_chroma,
    sanitize_text,
    unique_keep_order,
)

logger = logging.getLogger("modules.chroma.scrape_service")
store = get_store(alias_resolver=resolve_alias)

SCRAPE_TIMEOUT = float(os.getenv("SCRAPE_TIMEOUT", "20"))
SCRAPE_UA = os.getenv(
    "SCRAPE_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
)


async def fetch_url_html(url: str) -> str:
    headers = {
        "User-Agent": SCRAPE_UA,
        "Accept-Language": "ko,en-US;q=0.9,en;q=0.8",
    }
    timeout = httpx.Timeout(SCRAPE_TIMEOUT)

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        verify=False,
    ) as client:
        res = await client.get(url, headers=headers)
        res.raise_for_status()
        res.encoding = res.encoding or "utf-8"
        return res.text


def build_selector_candidates(soup: BeautifulSoup) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    classes: list[str] = []

    strong_tags = soup.select(
        "main, article, section, .content, .container, .article, .post, .entry"
    )
    for tag in strong_tags[:50]:
        if tag.get("id"):
            ids.append(str(tag.get("id")).strip())
        for cls in tag.get("class", []) or []:
            cls = str(cls).strip()
            if cls and len(cls) <= 80:
                classes.append(cls)

    all_ids = [t.get("id") for t in soup.find_all(attrs={"id": True})]
    ids.extend([str(v).strip() for v in all_ids if v])

    all_classes: list[str] = []
    for t in soup.find_all(class_=True):
        for cls in t.get("class", []) or []:
            if cls:
                all_classes.append(str(cls).strip())
    classes.extend(all_classes)

    ids = [v for v in unique_keep_order(ids) if 1 < len(v) <= 80]
    classes = [v for v in unique_keep_order(classes) if 1 < len(v) <= 80]

    bad_tokens = {
        "active",
        "show",
        "hide",
        "open",
        "close",
        "disabled",
        "selected",
        "current",
        "btn",
        "button",
        "row",
        "col",
        "wrap",
        "inner",
        "outer",
        "box",
        "item",
    }
    classes = [c for c in classes if c.lower() not in bad_tokens]

    return ids[:50], classes[:80]


def extract_html_by_selectors(
    html: str,
    element_ids: list[str],
    element_classes: list[str],
) -> tuple[str, str, dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    selected_nodes = []

    for element_id in element_ids:
        element_id = element_id.strip()
        if not element_id:
            continue
        node = soup.find(id=element_id)
        if node is not None:
            selected_nodes.append(node)

    for class_name in element_classes:
        class_name = class_name.strip()
        if not class_name:
            continue
        nodes = soup.find_all(
            class_=lambda c: c
            and class_name in (c if isinstance(c, list) else str(c).split())
        )
        if nodes:
            selected_nodes.extend(nodes)

    if not selected_nodes:
        fallback = (
            soup.find("main")
            or soup.find("article")
            or soup.find("section")
            or soup.body
            or soup
        )
        selected_nodes = [fallback]

    dedup_nodes = []
    seen_ids = set()
    for node in selected_nodes:
        key = id(node)
        if key not in seen_ids:
            seen_ids.add(key)
            dedup_nodes.append(node)

    html_parts: list[str] = []
    text_parts: list[str] = []

    for node in dedup_nodes:
        html_parts.append(str(node))
        text = sanitize_text(node.get_text("\n", strip=True))
        if text:
            text_parts.append(text)

    selected_html = "\n".join(html_parts).strip()
    selected_text = "\n\n".join(text_parts).strip()

    meta = {
        "selected_count": len(dedup_nodes),
        "used_element_id": element_ids,
        "used_element_class": element_classes,
    }
    return selected_html, selected_text, meta


async def emit_progress(
    request: Request,
    channel: str | None,
    message: str,
    extra: dict[str, Any] | None = None,
) -> None:
    sio_server = getattr(request.app.state, "sio", None)
    if sio_server is None:
        return

    payload = {"message": message}
    if extra:
        payload.update(extra)

    try:
        if channel:
            await sio_server.emit(channel, payload)
        await sio_server.emit("scrape_progress", payload)
    except Exception as e:
        logger.warning("socket emit failed: %s", e)


async def process_embed_urls(
    payload: EmbedUrlsRequest,
    request: Request,
    *,
    mode: str = "embed",
) -> JSONResponse:
    if not payload.index.strip():
        return fail("index 값이 비어 있습니다.")
    if not payload.urls:
        return fail("urls 값이 비어 있습니다.")

    channel = None
    if isinstance(payload.meta, dict):
        channel = str(payload.meta.get("channel", "")).strip() or None

    collection_name = resolve_alias(payload.index.strip())

    try:
        store.create_collection(collection_name, get_or_create=True)
    except Exception as e:
        return fail(f"collection 생성 실패: {e}", 500)

    results: list[dict[str, Any]] = []
    stored = 0
    failed = 0

    await emit_progress(
        request,
        channel,
        f"{mode} 시작: collection={collection_name}, url_count={len(payload.urls)}",
        {"index": collection_name, "mode": mode},
    )

    for i, raw_url in enumerate(payload.urls, start=1):
        url = normalize_url(raw_url)

        await emit_progress(
            request,
            channel,
            f"[{i}/{len(payload.urls)}] 요청 중: {url}",
            {"step": i, "total": len(payload.urls), "url": url, "mode": mode},
        )

        try:
            html = await fetch_url_html(url)

            selected_html, selected_text, selector_meta = extract_html_by_selectors(
                html,
                payload.element_id,
                payload.element_class,
            )

            soup = BeautifulSoup(html, "html.parser")
            title = sanitize_text(soup.title.text) if soup.title and soup.title.text else ""

            doc_id = build_doc_id(url=url, collection_name=collection_name)

            raw_doc_meta = {
                "url": url,
                "title": title,
                "index": collection_name,
                "source": "web_scrape",
                "mode": mode,
                "doc_type": "web_page",
                "element_id": payload.element_id[0] if payload.element_id else None,
                "element_class": payload.element_class[0] if payload.element_class else None,
                "used_element_id": payload.element_id[0] if payload.element_id else None,
                "used_element_class": payload.element_class[0] if payload.element_class else None,
                "selected_count": selector_meta.get("selected_count"),
                **(payload.meta or {}),
            }

            doc_meta = sanitize_metadata_for_chroma(raw_doc_meta)

            ingest_result = ingest_html_semantic(
                raw_html=selected_html if selected_html else html,
                url=url,
                collection_name=collection_name,
                doc_id=doc_id,
                title=title,
                metadata=doc_meta,
            )

            stored += 1
            results.append(
                {
                    "doc_id": doc_id,
                    "url": url,
                    "title": ingest_result.get("title"),
                    "text_length": len(selected_text),
                    "selected_html_length": len(selected_html),
                    "chunks": ingest_result.get("chunks"),
                    "metadata": doc_meta,
                    "semantic_meta": ingest_result.get("meta"),
                    "items": ingest_result.get("items", []),
                }
            )

            await emit_progress(
                request,
                channel,
                f"[{i}/{len(payload.urls)}] 완료: {url} (chunks={ingest_result.get('chunks', 0)})",
                {
                    "step": i,
                    "total": len(payload.urls),
                    "url": url,
                    "doc_id": doc_id,
                    "chunks": ingest_result.get("chunks", 0),
                    "mode": mode,
                },
            )

        except Exception as e:
            failed += 1
            logger.exception("%s failed: %s", mode, url)
            results.append(
                {
                    "doc_id": build_doc_id(url=url, collection_name=collection_name),
                    "url": url,
                    "error": str(e),
                }
            )

            await emit_progress(
                request,
                channel,
                f"[{i}/{len(payload.urls)}] 실패: {url} - {e}",
                {
                    "step": i,
                    "total": len(payload.urls),
                    "url": url,
                    "error": str(e),
                    "mode": mode,
                },
            )

    await emit_progress(
        request,
        channel,
        f"{mode} 종료: success={len(results) - failed}, failed={failed}, stored={stored}",
        {
            "success": len(results) - failed,
            "failed": failed,
            "stored": stored,
            "index": collection_name,
            "mode": mode,
        },
    )

    return ok(
        {
            "index": collection_name,
            "total": len(payload.urls),
            "stored": stored,
            "failed": failed,
            "items": results,
            "results": results,
            "mode": mode,
        }
    )