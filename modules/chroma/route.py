#/home/user/hallym_guide/modules/chroma/route.py
from __future__ import annotations

import io
import uuid
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup
from fastapi import APIRouter, File, Form, Request, UploadFile

from modules.chroma.alias_store import (
    delete_alias,
    list_aliases,
    resolve_alias,
    set_alias,
)
from modules.chroma.ingest import ingest_file, ingest_text
from modules.chroma.schemas import FetchHtmlRequest, EmbedUrlsRequest, ScrapeRequest
from modules.chroma.scrape_service import (
    build_selector_candidates,
    fetch_url_html,
    process_embed_urls,
)
from modules.chroma.store import get_store
from modules.chroma.utils import (
    dataframe_to_urls,
    fail,
    normalize_url,
    ok,
    parse_int,
    parse_json_str,
    same_domain,
    unique_keep_order,
)

DEFAULT_COLLECTION = ""

router = APIRouter(tags=["chroma"])
store = get_store(alias_resolver=resolve_alias)


# -------------------------------------------------------
# db: collections
# -------------------------------------------------------


@router.post("/api/db/collections")
async def api_create_collection(request: Request):
    try:
        payload = await request.json()
        name = str(payload.get("name") or "").strip()
        metadata = payload.get("metadata") or {}

        if not name:
            return fail("name is required", 400)

        collection = store.create_collection(
            name=name,
            metadata=metadata,
            get_or_create=True,
        )

        return ok(
            {
                "collection": collection.name,
                "metadata": getattr(collection, "metadata", None),
            }
        )
    except Exception as exc:
        return fail(str(exc), 500)


@router.get("/api/db/collections")
async def api_list_collections():
    try:
        collections = [
            {
                "name": collection.name,
                "metadata": getattr(collection, "metadata", None),
            }
            for collection in (
                store.get_collection(name) for name in store.list_collections()
            )
        ]
        return ok({"collections": collections})
    except Exception as exc:
        return fail(str(exc), 500)


@router.delete("/api/db/collections/{name}")
async def api_delete_collection(name: str):
    try:
        store.delete_collection(name)
        return ok({"deleted": name})
    except Exception as exc:
        return fail(str(exc), 500)


@router.get("/api/db/collections/{name}/count")
async def api_count_collection(name: str):
    try:
        count = store.count_documents(name)
        return ok({"collection": name, "count": count})
    except Exception as exc:
        return fail(str(exc), 500)


# -------------------------------------------------------
# db: documents
# -------------------------------------------------------


@router.post("/api/db/documents/ingest")
async def api_ingest_document(request: Request):
    try:
        payload = await request.json()

        text = payload.get("text")
        collection_name = str(payload.get("collection") or DEFAULT_COLLECTION).strip()
        doc_id = payload.get("doc_id")
        metadata = payload.get("metadata") or {}

        if not text:
            return fail("text is required", 400)

        result = ingest_text(
            text=text,
            collection_name=collection_name,
            doc_id=doc_id,
            metadata=metadata,
        )

        return ok(
            {
                "collection": collection_name,
                "result": result,
            }
        )
    except Exception as exc:
        return fail(str(exc), 500)


@router.post("/api/db/documents/upload")
async def api_upload_document(
    collection: str = Form(DEFAULT_COLLECTION),
    doc_id: str = Form(""),
    metadata: str = Form("{}"),
    files: list[UploadFile] = File(...),
):
    try:
        collection_name = str(collection or DEFAULT_COLLECTION).strip()
        doc_id_prefix = str(doc_id or "").strip()
        metadata_obj = parse_json_str(metadata, default={}) or {}

        if not files:
            return fail("file is required", 400)

        valid_files = [file for file in files if file and (file.filename or "").strip()]
        if not valid_files:
            return fail("valid file is required", 400)

        results = []
        errors = []

        for index, file in enumerate(valid_files, start=1):
            file_doc_id = f"{doc_id_prefix}-{index}" if doc_id_prefix else None

            try:
                result = ingest_file(
                    file_storage=file,
                    collection_name=collection_name,
                    doc_id=file_doc_id,
                    metadata=metadata_obj,
                )
                results.append(
                    {
                        "index": index,
                        "file_name": file.filename,
                        "doc_id": file_doc_id,
                        "ok": True,
                        "result": result,
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "index": index,
                        "file_name": file.filename,
                        "doc_id": file_doc_id,
                        "ok": False,
                        "error": str(exc),
                    }
                )

        return ok(
            {
                "collection": collection_name,
                "total_files": len(valid_files),
                "success_count": len(results),
                "error_count": len(errors),
                "results": results,
                "errors": errors,
            }
        )
    except Exception as exc:
        return fail(str(exc), 500)


@router.post("/api/db/documents/upsert")
async def api_upsert_documents(request: Request):
    try:
        payload = await request.json()

        collection_name = str(payload.get("collection") or "").strip()
        ids = payload.get("ids") or []
        documents = payload.get("documents") or []
        metadatas = payload.get("metadatas")

        if not collection_name:
            return fail("collection is required", 400)
        if not ids:
            return fail("ids is required", 400)
        if not documents:
            return fail("documents is required", 400)

        store.upsert_documents(
            collection_name=collection_name,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        return ok(
            {
                "collection": collection_name,
                "count": len(ids),
            }
        )
    except Exception as exc:
        return fail(str(exc), 500)


@router.get("/api/db/documents")
async def api_get_documents(request: Request):
    try:
        collection_name = str(
            request.query_params.get("collection") or DEFAULT_COLLECTION
        ).strip()
        ids = request.query_params.getlist("id")
        limit = parse_int(request.query_params.get("limit"), 10)
        offset = parse_int(request.query_params.get("offset"), 0)
        where = parse_json_str(request.query_params.get("where"), default=None)

        result = store.get_documents(
            collection_name=collection_name,
            ids=ids or None,
            where=where,
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"],
        )

        return ok({"result": result})
    except Exception as exc:
        return fail(str(exc), 500)


@router.delete("/api/db/documents")
async def api_delete_documents(request: Request):
    try:
        payload = await request.json()

        collection_name = str(payload.get("collection") or DEFAULT_COLLECTION).strip()
        ids = payload.get("ids")
        where = payload.get("where")

        if ids is None and where is None:
            return fail("either ids or where is required", 400)

        store.delete_documents(
            collection_name=collection_name,
            ids=ids,
            where=where,
        )

        return ok({"collection": collection_name})
    except Exception as exc:
        return fail(str(exc), 500)


# -------------------------------------------------------
# db: search
# -------------------------------------------------------


@router.get("/api/db/search")
async def api_search(request: Request):
    try:
        query = str(request.query_params.get("query") or "").strip()
        collection_name = str(
            request.query_params.get("collection") or DEFAULT_COLLECTION
        ).strip()
        k = parse_int(request.query_params.get("k"), 5)
        where = parse_json_str(request.query_params.get("where"), default=None)

        if not query:
            return fail("query is required", 400)

        results = store.similarity_search(
            collection_name=collection_name,
            query=query,
            k=k,
            where=where,
        )

        return ok(
            {
                "query": query,
                "collection": collection_name,
                "k": k,
                "results": results,
            }
        )
    except Exception as exc:
        return fail(str(exc), 500)


# -------------------------------------------------------
# aliases
# -------------------------------------------------------


@router.get("/api/aliases")
async def api_aliases():
    try:
        return ok({"aliases": list_aliases()})
    except Exception as exc:
        return fail(str(exc), 500)


@router.post("/api/aliases/set")
async def api_alias_set(request: Request):
    try:
        payload = await request.json()
        alias = str(payload.get("alias") or "").strip()
        collection_name = str(payload.get("collection_name") or "").strip()

        if not alias:
            return fail("alias is required", 400)
        if not collection_name:
            return fail("collection_name is required", 400)

        set_alias(alias, collection_name)

        return ok(
            {
                "alias": alias,
                "collection_name": collection_name,
                "aliases": list_aliases(),
            }
        )
    except Exception as exc:
        return fail(str(exc), 500)


@router.post("/api/aliases/delete")
async def api_alias_delete(request: Request):
    try:
        payload = await request.json()
        alias = str(payload.get("alias") or "").strip()

        if not alias:
            return fail("alias is required", 400)

        deleted = delete_alias(alias)

        return ok(
            {
                "deleted": deleted,
                "alias": alias,
                "aliases": list_aliases(),
            }
        )
    except Exception as exc:
        return fail(str(exc), 500)


# -------------------------------------------------------
# web scraping / web embedding
# -------------------------------------------------------


@router.get("/get_channel")
async def api_get_channel():
    return ok({"channel": f"scrape_{uuid.uuid4().hex}"})


@router.post("/extract_urls")
async def api_extract_urls(
    url: str = Form(...),
    domain: str | None = Form(None),
):
    target_url = normalize_url(url)

    try:
        html = await fetch_url_html(target_url)
    except Exception as exc:
        return fail(f"URL fetch failed: {exc}", 502)

    soup = BeautifulSoup(html, "html.parser")
    collected: list[str] = []

    for anchor in soup.find_all("a", href=True):
        href = normalize_url(anchor.get("href", ""), base=target_url)
        if not href:
            continue

        parsed = urlparse(href)
        if parsed.scheme not in {"http", "https"}:
            continue

        if same_domain(href, target_url, domain):
            collected.append(href)

    urls = unique_keep_order(collected)
    return ok({"urls": urls, "count": len(urls), "source_url": target_url})


@router.post("/get_gpt_recommendations")
async def api_get_gpt_recommendations(url: str = Form(...)):
    target_url = normalize_url(url)

    try:
        html = await fetch_url_html(target_url)
    except Exception as exc:
        return fail(f"URL fetch failed: {exc}", 502)

    soup = BeautifulSoup(html, "html.parser")
    ids, classes = build_selector_candidates(soup)

    return ok(
        {
            "url": target_url,
            "element_id": ids[:20],
            "element_class": classes[:30],
            "ids": ids[:20],
            "classes": classes[:30],
        }
    )
    
@router.patch("/api/db/collections/{name}")
async def api_update_collection(name: str, request: Request):
    try:
        payload = await request.json()
        metadata = payload.get("metadata")

        if metadata is None:
            return fail("metadata is required", 400)
        if not isinstance(metadata, dict):
            return fail("metadata must be an object", 400)

        collection = store.get_collection(name)
        collection.modify(metadata=metadata)
        updated_collection = store.get_collection(name)

        return ok(
            {
                "collection": updated_collection.name,
                "metadata": getattr(updated_collection, "metadata", metadata),
            }
        )
    except Exception as exc:
        return fail(str(exc), 500)


@router.post("/fetch_html")
async def api_fetch_html(payload: FetchHtmlRequest):
    target_url = normalize_url(payload.url)

    try:
        html = await fetch_url_html(target_url)
    except Exception as exc:
        return fail(f"URL fetch failed: {exc}", 502)

    return ok({"url": target_url, "html": html})


@router.post("/excel_url")
async def api_excel_url(file: UploadFile = File(...)):
    filename = file.filename or "upload"
    content = await file.read()

    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
    except Exception as exc:
        return fail(f"파일 파싱 실패: {exc}")

    urls = dataframe_to_urls(df)
    return ok({"filename": filename, "urls": urls, "count": len(urls)})


@router.post("/embed_urls")
async def api_embed_urls(payload: EmbedUrlsRequest, request: Request):
    return await process_embed_urls(payload, request, mode="embed")


@router.post("/scrape")
async def api_scrape(payload: ScrapeRequest, request: Request):
    return await process_embed_urls(payload, request, mode="scrape")