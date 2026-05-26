from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from bs4 import BeautifulSoup, NavigableString
from pypdf import PdfReader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from modules.chroma.chunking import split_text
from modules.chroma.embeddings import embed_documents, embed_query

from .base import (
    ToolResult,
    clean_block_text,
    normalize_whitespace,
    strip_html_tags,
)


def html_to_markdown(soup: BeautifulSoup, title: str = "") -> str:
    """
    HTML 문서를 markdown 텍스트로 변환한다.

    - script/style 등 검색에 불필요한 태그는 제거
    - img alt 텍스트는 텍스트로 보존
      예: <img alt="문영식"> -> [image: 문영식]
    - markdownify 사용 가능하면 markdown으로 변환
    - 실패하면 plain text로 fallback
    - title이 있으면 맨 위에 # 제목 형태로 추가
    """
    # 이미지 alt 텍스트를 먼저 보존
    for img in soup.find_all("img"):
        alt = normalize_whitespace(img.get("alt", ""))
        if alt:
            img.replace_with(NavigableString(f" [image: {alt}] "))
        else:
            img.decompose()

    # 텍스트 검색에 불필요한 태그 제거
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "canvas"]):
        tag.decompose()

    body = soup.body or soup

    try:
        from markdownify import markdownify as md

        raw_html = str(body)
        markdown_text = md(
            raw_html,
            heading_style="ATX",
            bullets="-",
            strip=["script", "style", "noscript", "svg", "iframe", "canvas"],
        )
        markdown_text = normalize_whitespace(markdown_text)

        if title and not markdown_text.startswith("# "):
            markdown_text = f"# {clean_block_text(title)}\n\n{markdown_text}"

        return markdown_text

    except Exception:
        # markdown 변환 실패 시 텍스트만 추출
        text = body.get_text("\n", strip=True)
        text = normalize_whitespace(text)

        if title:
            return f"# {clean_block_text(title)}\n\n{text}"
        return text


def split_markdown_with_langchain(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    markdown 텍스트를 2단계로 청킹한다.

    1) MarkdownHeaderTextSplitter 로 헤더(#, ##, ### ...) 기준 분리
    2) 각 섹션이 길면 RecursiveCharacterTextSplitter 로 다시 분리

    기본값:
    - chunk_size=500
    - chunk_overlap=50
    """
    text = normalize_whitespace(text)
    if not text:
        return []

    # 1차: 헤더 기준 분리
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
            ("#####", "h5"),
            ("######", "h6"),
        ],
        strip_headers=False,  # 헤더를 유지해서 문맥 보존
    )

    docs = header_splitter.split_text(text)

    # 2차: 길이 기준 재분할
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: List[str] = []

    for doc in docs:
        content = normalize_whitespace(doc.page_content)
        if not content:
            continue

        sub_chunks = char_splitter.split_text(content)
        for chunk in sub_chunks:
            cleaned = normalize_whitespace(chunk)
            if cleaned:
                chunks.append(cleaned)

    return chunks


def fetch_url(url: str, debug_print=None) -> Dict[str, Any]:
    """
    URL을 가져와서 content_type에 따라 텍스트를 추출한다.

    지원:
    - HTML -> markdown 변환
    - text/plain -> 텍스트 그대로 사용
    - PDF -> pypdf로 텍스트 추출

    반환 예시:
    {
        "url": "...",
        "text": "...",
        "content_type": "html" | "text" | "pdf",
        "title": "..."
    }

    에러 시:
    {
        "url": "...",
        "error": "..."
    }
    """
    try:
        resp = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            },
        )
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "").lower()

        if debug_print:
            debug_print(
                f"FETCH_URL RESPONSE META | {url}",
                {
                    "status_code": resp.status_code,
                    "content_type": content_type,
                    "final_url": resp.url,
                },
            )

        # HTML 처리
        if "text/html" in content_type:
            soup = BeautifulSoup(resp.text, "html.parser")

            title = ""
            if soup.title and soup.title.string:
                title = clean_block_text(soup.title.string)

            markdown_text = html_to_markdown(soup=soup, title=title)

            if debug_print:
                debug_print(f"FETCH_URL HTML RAW TITLE | {url}", title)
                debug_print(
                    f"FETCH_URL HTML -> MARKDOWN | {url}",
                    markdown_text,
                    max_len=12000,
                )

            return {
                "url": url,
                "text": markdown_text,
                "content_type": "html",
                "title": title,
            }

        # text/plain 처리
        if "text/plain" in content_type:
            text = normalize_whitespace(resp.text)

            if debug_print:
                debug_print(
                    f"FETCH_URL TEXT | {url}",
                    text,
                    max_len=12000,
                )

            return {
                "url": url,
                "text": text,
                "content_type": "text",
                "title": "",
            }

        # PDF 처리
        if "application/pdf" in content_type:
            reader = PdfReader(BytesIO(resp.content))

            text = ""
            for page_idx, page in enumerate(reader.pages, start=1):
                extracted = page.extract_text() or ""
                text += extracted + "\n"

                if debug_print:
                    debug_print(
                        f"FETCH_URL PDF PAGE {page_idx} | {url}",
                        extracted,
                        max_len=6000,
                    )

            text = normalize_whitespace(text)

            if debug_print:
                debug_print(
                    f"FETCH_URL PDF FULLTEXT | {url}",
                    text,
                    max_len=12000,
                )

            return {
                "url": url,
                "text": text,
                "content_type": "pdf",
                "title": "",
            }

        # 지원하지 않는 타입
        if debug_print:
            debug_print(
                f"FETCH_URL UNSUPPORTED CONTENT TYPE | {url}",
                content_type,
            )

        return {
            "url": url,
            "text": "",
            "content_type": content_type,
            "title": "",
        }

    except Exception as e:
        if debug_print:
            debug_print(f"FETCH_URL ERROR | {url}", str(e))
        return {
            "url": url,
            "error": str(e),
        }


def google_search_items(
    query: str,
    google_api_key: str,
    google_cx: str,
    num: int = 5,
    debug_print=None,
) -> List[Dict[str, Any]]:
    """
    Google Custom Search API로 검색 결과를 가져온다.
    """
    if not google_api_key or not google_cx:
        raise ValueError("GOOGLE_API_KEY or GOOGLE_CX is missing")

    resp = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={
            "key": google_api_key,
            "cx": google_cx,
            "q": query,
            "num": max(1, min(num, 10)),
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
                "snippet": normalize_whitespace(item.get("snippet", "")),
                "displayLink": item.get("displayLink", ""),
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
    """
    네이버 웹검색 API로 검색 결과를 가져온다.
    """
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
                "snippet": snippet,
                "displayLink": "search.naver.com",
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
    """
    검색 엔진(google/naver) 결과를 합쳐서 dedup 후 반환한다.
    """
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

    # URL 기준 dedup
    dedup = []
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
    """
    디버깅용: 검색 후보들을 사람이 읽기 쉬운 텍스트로 만든다.
    """
    lines = []

    for i, item in enumerate(items, start=1):
        lines.append(
            f"[{i}]\n"
            f"title: {item.get('title', '')}\n"
            f"link: {item.get('link', '')}\n"
            f"snippet: {item.get('snippet', '')}\n"
            f"engine: {item.get('source_engine', '')}\n"
        )

    return "\n".join(lines)


def score_search_item_for_fetch(user_query: str, item: Dict[str, Any]) -> float:
    """
    fetch할 검색 결과의 점수 함수.

    현재는 미구현이라 모두 0.0 리턴.
    즉, 현재는 검색 결과 순서대로 선택되는 것과 거의 같다.

    나중에 아래 기준으로 확장 가능:
    - query/title lexical match
    - snippet match
    - domain prior
    - pdf/html 선호도
    """
    _ = user_query
    _ = item
    return 0.0


def select_search_items_for_fetch(
    user_query: str,
    items: List[Dict[str, Any]],
    max_select: int = 5,
    min_select: int = 1,
    debug_print=None,
) -> List[Dict[str, Any]]:
    """
    검색 결과 중 실제로 fetch할 URL 후보를 고른다.

    현재는 score_search_item_for_fetch()가 0.0만 반환하므로
    사실상 dedup된 검색 결과 상위부터 max_select개를 선택한다.
    """
    if not items:
        return []

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for item in items:
        score = score_search_item_for_fetch(user_query, item)
        enriched = dict(item)
        enriched["_fetch_score"] = float(score)
        scored.append((score, enriched))

    scored.sort(key=lambda x: x[0], reverse=True)

    if debug_print:
        debug_print(
            "SELECT_SEARCH_ITEMS SCORED",
            [
                {
                    "score": round(score, 4),
                    "title": one.get("title"),
                    "link": one.get("link"),
                    "engine": one.get("source_engine"),
                }
                for score, one in scored
            ],
            max_len=12000,
        )

    selected: List[Dict[str, Any]] = []

    for score, item in scored:
        if len(selected) >= max_select:
            break
        selected.append(item)

    if len(selected) < min_select and scored:
        selected = [scored[0][1]]

    if debug_print:
        debug_print(
            "SELECT_SEARCH_ITEMS FINAL",
            [
                {
                    "score": s.get("_fetch_score"),
                    "title": s.get("title"),
                    "link": s.get("link"),
                    "engine": s.get("source_engine"),
                }
                for s in selected
            ],
            max_len=12000,
        )

    return selected


def memory_similarity_search(
    query: str,
    documents: List[Dict[str, Any]],
    k: int = 5,
    debug_print=None,
) -> List[Dict[str, Any]]:
    """
    청크 문서들을 임베딩한 뒤 query와 cosine similarity를 계산해서
    상위 k개 chunk를 반환한다.

    반환값 예시:
    [
        {
            "rank": 1,
            "url": "...",
            "title": "...",
            "content_type": "html",
            "preview": "...",
            "score": 0.82
        },
        ...
    ]
    """
    texts = [d["text"] for d in documents]

    if debug_print:
        debug_print(
            "MEMORY_SIMILARITY_SEARCH INPUT META",
            {
                "query": query,
                "num_documents": len(documents),
                "top_k": k,
            },
        )

        for idx, doc in enumerate(documents, start=1):
            debug_print(
                f"DOCUMENT BEFORE EMBEDDING #{idx}",
                {
                    "url": doc.get("url"),
                    "content_type": doc.get("content_type"),
                    "title": doc.get("title"),
                    "text_preview": doc.get("text", "")[:2500],
                },
                max_len=7000,
            )

    # 문서와 질의 임베딩
    doc_embeddings = embed_documents(texts)
    query_embedding = embed_query(query)

    scores = []

    # cosine similarity 계산
    for i, emb in enumerate(doc_embeddings):
        denom = np.linalg.norm(query_embedding) * np.linalg.norm(emb)
        sim = 0.0 if denom == 0 else float(np.dot(query_embedding, emb) / denom)
        scores.append((sim, i))

    scores.sort(reverse=True)

    results = []
    for sim, idx in scores[:k]:
        results.append(
            {
                "rank": len(results) + 1,
                "url": documents[idx]["url"],
                "title": documents[idx].get("title", ""),
                "content_type": documents[idx].get("content_type", ""),
                "preview": documents[idx]["text"][:500],  # 미리보기는 500자까지만
                "score": float(sim),
            }
        )

    if debug_print:
        debug_print(
            "MEMORY_SIMILARITY_SEARCH TOP RESULTS",
            results,
            max_len=12000,
        )

    return results


def run(
    query: str,
    google_api_key: str,
    google_cx: str,
    naver_client_id: str,
    naver_client_secret: str,
    num: int = 5,
    top_k_urls: int = 5,
    top_k_chunks: int = 5,
    engines: Optional[List[str]] = None,
    debug_print=None,
) -> ToolResult:
    """
    외부 검색 전체 실행 함수.

    전체 흐름:
    1) 검색 엔진에서 결과 가져오기
    2) fetch할 URL 선택
    3) 각 URL 문서 가져오기
    4) HTML이면 markdown 헤더 기반 + 길이 기반 청킹
       PDF/TXT면 기존 split_text 사용
    5) 모든 청크 임베딩 후 query와 유사도 검색
    6) 상위 chunk들을 context로 합치고,
       마지막에 '사용자질문: {query}'를 붙여서 반환

    기본값:
    - 검색 결과 num=5
    - fetch할 URL 수 top_k_urls=5
    - 최종 사용할 chunk 수 top_k_chunks=5
    """
    try:
        engines = engines or ["google", "naver"]

        if debug_print:
            debug_print(
                "EXTERNAL_SEARCH START",
                {
                    "query": query,
                    "num": num,
                    "top_k_urls": top_k_urls,
                    "top_k_chunks": top_k_chunks,
                    "engines": engines,
                },
            )

        # 1) 검색 결과 수집
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

        # 2) fetch할 URL 고르기
        selected_items = select_search_items_for_fetch(
            user_query=query,
            items=items,
            max_select=top_k_urls,
            min_select=1,
            debug_print=debug_print,
        )

        urls = []
        for item in selected_items:
            link = (item.get("link") or "").strip()
            if not link:
                continue
            urls.append(link)

        if debug_print:
            debug_print("EXTERNAL_SEARCH SELECTED URLS", urls)

        documents = []

        # 3) 각 URL fetch + 청킹
        for url_idx, url in enumerate(urls, start=1):
            if debug_print:
                debug_print(f"PROCESS URL {url_idx}/{len(urls)}", url)

            page = fetch_url(url, debug_print=debug_print)

            if page.get("error"):
                if debug_print:
                    debug_print(f"SKIP URL DUE TO ERROR | {url}", page["error"])
                continue

            text = page.get("text", "")
            content_type = page.get("content_type", "")
            title = page.get("title", "").strip()

            if not text:
                if debug_print:
                    debug_print(
                        f"SKIP URL DUE TO EMPTY TEXT | {url}",
                        {
                            "content_type": content_type,
                            "title": title,
                        },
                    )
                continue

            # HTML은 markdown 헤더 기반 분리 + 길이 기반 재분할
            if content_type == "html":
                chunks = split_markdown_with_langchain(text)

                if debug_print:
                    debug_print(
                        f"HTML MARKDOWN HEADER CHUNK COUNT | {url}",
                        {
                            "title": title,
                            "num_chunks": len(chunks),
                        },
                    )

                for idx, chunk in enumerate(chunks, start=1):
                    if debug_print:
                        debug_print(
                            f"HTML MARKDOWN HEADER CHUNK {idx}/{len(chunks)} | {url}",
                            chunk,
                            max_len=12000,
                        )

            # PDF / plain text 등은 기존 split_text 사용
            else:
                chunks = split_text(text)

                if debug_print:
                    debug_print(
                        f"NON-HTML CHUNK COUNT | {url}",
                        {
                            "content_type": content_type,
                            "num_chunks": len(chunks),
                        },
                    )

                for idx, chunk in enumerate(chunks, start=1):
                    if debug_print:
                        debug_print(
                            f"NON-HTML CHUNK {idx}/{len(chunks)} | {url}",
                            chunk,
                            max_len=12000,
                        )

            # 청크 정리 후 documents에 적재
            for idx, chunk in enumerate(chunks, start=1):
                cleaned = normalize_whitespace(chunk)
                if not cleaned:
                    if debug_print:
                        debug_print(
                            f"DROP EMPTY CLEANED CHUNK {idx} | {url}",
                            chunk,
                        )
                    continue

                # HTML이고 title이 있으면 chunk 앞에 제목 보강
                # 단, 이미 # 헤더로 시작하면 그대로 둔다.
                if content_type == "html" and title:
                    if not cleaned.startswith("# "):
                        chunk_text = f"# {title}\n\n{cleaned}"
                    else:
                        chunk_text = cleaned
                else:
                    chunk_text = cleaned

                if debug_print:
                    debug_print(
                        f"FINAL CHUNK BEFORE EMBEDDING {idx} | {url}",
                        chunk_text,
                        max_len=12000,
                    )

                documents.append(
                    {
                        "url": url,
                        "text": chunk_text,
                        "content_type": content_type,
                        "title": title,
                    }
                )

        if not documents:
            if debug_print:
                debug_print("EXTERNAL_SEARCH NO DOCUMENTS", "no documents fetched")
            return ToolResult(
                name="external_search",
                ok=False,
                data={"error": "no documents fetched"},
            )

        if debug_print:
            debug_print("EXTERNAL_SEARCH DOCUMENT COUNT", len(documents))

        # 4) 임베딩 유사도 검색
        results = memory_similarity_search(
            query=query,
            documents=documents,
            k=top_k_chunks,
            debug_print=debug_print,
        )

        for r in results:
            if debug_print:
                debug_print(
                    f"RETRIEVED RESULT rank={r['rank']} score={r['score']}",
                    r["preview"],
                    max_len=6000,
                )

        # 5) LLM에 넣을 context 생성
        #    여기서 마지막에 반드시 사용자 질문을 붙인다.
        retrieved_context = "\n\n".join([r["preview"] for r in results])

        context = (
            "아래는 검색으로 찾은 참고 문서 조각들입니다.\n\n"
            f"{retrieved_context}\n\n"
            f"사용자질문: {query}"
        )

        if debug_print:
            debug_print(
                "FINAL CONTEXT FOR LLM",
                context,
                max_len=12000,
            )

        return ToolResult(
            name="external_search",
            ok=True,
            data={
                "query": query,                    # 원본 사용자 질문
                "engines": engines,                # 사용한 검색 엔진
                "items": items,                    # 검색 결과 전체
                "selected_items": selected_items,  # fetch 대상으로 고른 결과
                "selected_urls": urls,             # 실제 fetch한 URL 목록
                "sources": results,                # 최종 top-k chunk 검색 결과
                "context": context,                # LLM에 넘길 최종 컨텍스트
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