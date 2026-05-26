from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

from .agent_speed_utils import (
    canonicalize_url,
    normalize_whitespace,
)


class AgentSpeedSources:
    """
    source / document 후처리 전용 헬퍼.

    책임:
    - source dedupe
    - chunk stitching
    - document link extraction
    - extracted document normalization
    - document merge

    - external candidate dedupe / search item merge
    """

    def _build_source_identity_key(self, src: Dict[str, Any]) -> Tuple[Any, ...]:
        """
        내부 source dedupe 기준:
        1) id
        2) doc_id + chunk_id
        3) 둘 다 없으면 텍스트로 dedupe하지 않음 (각각 별개로 유지)

        외부 source dedupe 기준:
        - title + url + snippet/content 일부
        """
        src_id = normalize_whitespace(src.get("id") or "")
        title = normalize_whitespace(src.get("title") or "")
        url = normalize_whitespace(src.get("url") or src.get("link") or "")
        snippet = normalize_whitespace(src.get("snippet") or "")
        content = normalize_whitespace(src.get("content") or "")
        doc_id = normalize_whitespace(src.get("doc_id") or "")
        chunk_id = normalize_whitespace(src.get("chunk_id") or "")

        is_internal = bool(doc_id or chunk_id or src_id)

        if is_internal and src_id:
            return ("internal_id", src_id.lower())

        if is_internal and doc_id and chunk_id:
            return ("internal_doc_chunk", doc_id.lower(), chunk_id.lower())

        if is_internal:
            # 내부 source인데 id/chunk_id가 없으면 텍스트 일부 겹침으로 dedupe하지 않음
            # title/url/doc_id 조합 정도만 쓰고, 동일성 확신이 없으면 각각 유지
            return (
                "internal_unique",
                doc_id.lower(),
                title.lower(),
                url.lower(),
                id(src),
            )

        return (
            "external",
            title.lower(),
            url.lower(),
            (content or snippet)[:300].lower(),
        )

    def dedupe_sources(self, sources: Any) -> List[Dict[str, Any]]:
        if not isinstance(sources, list):
            return []

        result: List[Dict[str, Any]] = []
        seen = set()

        for src in sources:
            if not isinstance(src, dict):
                continue

            key = self._build_source_identity_key(src)

            if key in seen:
                continue

            seen.add(key)
            result.append(dict(src))

        return result

    def extract_source_identity(self, src: Dict[str, Any]) -> Tuple[str, str]:
        title = normalize_whitespace(src.get("title") or "")
        url = normalize_whitespace(src.get("url") or "")
        doc_id = normalize_whitespace(src.get("doc_id") or "")
        chunk_id = normalize_whitespace(src.get("chunk_id") or "")
        src_id = normalize_whitespace(src.get("id") or "")

        # 내부 source면 가능한 한 더 안정적인 identity 사용
        if src_id:
            return (src_id.lower(), chunk_id.lower())

        if doc_id or chunk_id:
            return (doc_id.lower() or title.lower(), chunk_id.lower())

        return (title.lower(), url.lower())

    def extract_header_key(self, src: Dict[str, Any]) -> str:
        """
        같은 헤더 반복 분할 문제를 완화하기 위한 느슨한 헤더 식별자.
        metadata 에 header/section 계열 정보가 있으면 우선 사용하고,
        없으면 content/snippet 첫 줄을 약식 헤더처럼 사용.
        """
        for key in ["header", "section", "section_title", "heading", "parent_header"]:
            value = normalize_whitespace(src.get(key) or "")
            if value:
                return value.lower()

        content = str(src.get("content") or src.get("snippet") or "").strip()
        if not content:
            return ""

        first_line = content.splitlines()[0].strip()
        first_line = re.sub(r"\s+", " ", first_line)
        return first_line[:120].lower()

    def extract_chunk_order(self, src: Dict[str, Any]) -> Optional[int]:
        for key in ["chunk_index", "order", "position", "seq", "chunk_id"]:
            value = src.get(key)
            try:
                return int(value)
            except Exception:
                continue
        return None

    def dedupe_internal_sources_by_doc_id(
        self,
        sources: Any,
        *,
        keep_per_doc: int = 5,
    ) -> List[Dict[str, Any]]:
        if not isinstance(sources, list):
            return []

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        no_doc_id: List[Dict[str, Any]] = []

        for src in sources:
            if not isinstance(src, dict):
                continue

            doc_id = normalize_whitespace(src.get("doc_id") or "")
            if not doc_id:
                no_doc_id.append(dict(src))
                continue

            grouped.setdefault(doc_id, []).append(dict(src))

        result: List[Dict[str, Any]] = []

        for _, items in grouped.items():
            items.sort(
                key=lambda x: (
                    -float(x.get("rerank_score") or -1e9),
                    float(x.get("distance") or 1e9),
                )
            )
            result.extend(items[:keep_per_doc])

        result.extend(no_doc_id)
        return result

    def merge_text_blocks(self, left: str, right: str) -> str:
        left = str(left or "").strip()
        right = str(right or "").strip()

        if not left:
            return right
        if not right:
            return left

        if right in left:
            return left
        if left in right:
            return right

        left_lines = left.splitlines()
        right_lines = right.splitlines()

        if left_lines and right_lines:
            l0 = normalize_whitespace(left_lines[0]).lower()
            r0 = normalize_whitespace(right_lines[0]).lower()
            if l0 and l0 == r0 and len(right_lines) > 1:
                right = "\n".join(right_lines[1:]).strip()

        if not right:
            return left

        return f"{left}\n{right}".strip()

    def stitch_adjacent_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        같은 문서 + 같은 헤더 + 인접 chunk 로 보이는 source 는 content 를 합친다.
        """
        if not isinstance(sources, list) or not sources:
            return []

        enriched: List[Tuple[Tuple[str, str], str, Optional[int], Dict[str, Any]]] = []
        for src in sources:
            if not isinstance(src, dict):
                continue

            doc_key = self.extract_source_identity(src)
            header_key = self.extract_header_key(src)
            order = self.extract_chunk_order(src)
            enriched.append((doc_key, header_key, order, dict(src)))

        enriched.sort(
            key=lambda x: (
                x[0][0],
                x[0][1],
                x[1],
                10**9 if x[2] is None else x[2],
            )
        )

        stitched: List[Dict[str, Any]] = []

        for doc_key, header_key, order, src in enriched:
            text = str(src.get("content") or src.get("snippet") or "").strip()

            if not stitched:
                stitched.append(src)
                continue

            prev = stitched[-1]
            prev_doc_key = self.extract_source_identity(prev)
            prev_header_key = self.extract_header_key(prev)
            prev_order = self.extract_chunk_order(prev)

            same_doc = doc_key == prev_doc_key
            same_header = bool(header_key) and header_key == prev_header_key
            adjacent = (
                prev_order is not None
                and order is not None
                and order - prev_order <= 1
            )

            if same_doc and (same_header or adjacent):
                prev_text = str(prev.get("content") or prev.get("snippet") or "").strip()
                merged_text = self.merge_text_blocks(prev_text, text)

                if prev.get("content"):
                    prev["content"] = merged_text
                else:
                    prev["snippet"] = merged_text

                continue

            stitched.append(src)

        return stitched

    def extract_document_links(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(document, dict):
            return []

        raw_links = document.get("links")
        base_url = normalize_whitespace(document.get("url") or document.get("link") or "")

        result: List[Dict[str, Any]] = []
        seen = set()

        if isinstance(raw_links, list):
            for link in raw_links:
                if not isinstance(link, dict):
                    continue

                href = normalize_whitespace(link.get("url") or link.get("link") or "")
                if not href:
                    continue

                abs_url = (
                    canonicalize_url(urljoin(base_url, href))
                    if base_url
                    else canonicalize_url(href)
                )
                if not abs_url or abs_url in seen:
                    continue

                seen.add(abs_url)
                result.append(
                    {
                        "link": abs_url,
                        "title": normalize_whitespace(
                            link.get("title") or link.get("anchor_text") or ""
                        ),
                        "snippet": normalize_whitespace(
                            link.get("snippet") or link.get("context") or ""
                        ),
                    }
                )

        if result:
            return result

        blob = "\n".join(
            [
                str(document.get("content") or ""),
                str(document.get("raw_html") or ""),
                str(document.get("html") or ""),
                str(document.get("markdown") or ""),
            ]
        )

        if not blob:
            return []

        for href, anchor in re.findall(
            r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
            blob,
            flags=re.I | re.S,
        ):
            abs_url = (
                canonicalize_url(urljoin(base_url, href))
                if base_url
                else canonicalize_url(href)
            )
            if not abs_url or abs_url in seen:
                continue

            seen.add(abs_url)
            anchor_text = normalize_whitespace(re.sub(r"<[^>]+>", " ", anchor))
            result.append({"link": abs_url, "title": anchor_text, "snippet": ""})

        for anchor, href in re.findall(
            r'\[([^\]]{0,200})\]\((https?://[^)\s]+|/[^)\s]+)\)',
            blob,
        ):
            abs_url = (
                canonicalize_url(urljoin(base_url, href))
                if base_url
                else canonicalize_url(href)
            )
            if not abs_url or abs_url in seen:
                continue

            seen.add(abs_url)
            result.append(
                {
                    "link": abs_url,
                    "title": normalize_whitespace(anchor),
                    "snippet": "",
                }
            )

        return result

    def normalize_extracted_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(document, dict):
            return {}

        normalized = dict(document)
        url = canonicalize_url(document.get("url") or document.get("link") or "")

        if url:
            normalized["url"] = url
            normalized["link"] = url

        if "title" not in normalized:
            normalized["title"] = ""

        if "content" not in normalized:
            normalized["content"] = str(document.get("snippet") or "")

        normalized["links"] = self.extract_document_links(normalized)
        return normalized

    def merge_documents_by_link(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen = set()

        for doc in documents:
            if not isinstance(doc, dict):
                continue

            normalized = self.normalize_extracted_document(doc)
            url = normalize_whitespace(normalized.get("url") or normalized.get("link") or "")

            if not url or url in seen:
                continue

            seen.add(url)
            merged.append(normalized)

        return merged

    def dedupe_external_candidates(self, items: Any) -> List[Dict[str, Any]]:
        if not isinstance(items, list):
            return []

        picked: List[Dict[str, Any]] = []
        seen_urls = set()

        for item in items:
            if not isinstance(item, dict):
                continue

            url = str(item.get("link") or "").strip()
            title = str(item.get("title") or "").strip()
            snippet = str(item.get("snippet") or item.get("description") or "").strip()

            if not url or url in seen_urls:
                continue

            seen_urls.add(url)
            picked.append(
                {
                    "title": title,
                    "link": url,
                    "snippet": snippet[:500],
                }
            )

        return picked

    def rerank_sources_with_bm25(
        self,
        *,
        user_query: str,
        query_to_result: List[Tuple[str, Dict[str, Any]]],
        top_k: int = 5,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        import math
        import re
        from collections import Counter

        safe_query = normalize_whitespace(user_query)
        safe_top_k = max(1, int(top_k or 1))

        if not safe_query:
            return query_to_result

        if not isinstance(query_to_result, list) or not query_to_result:
            return []

        def _tokenize(text: str) -> List[str]:
            normalized = normalize_whitespace(text).lower()
            if not normalized:
                return []
            return re.findall(r"[0-9a-zA-Z가-힣]+", normalized)

        def _build_source_text(src: Dict[str, Any]) -> str:
            if not isinstance(src, dict):
                return ""
            parts = [
                str(src.get("content") or ""),
                str(src.get("snippet") or ""),
                str(src.get("preview") or ""),
                str(src.get("document") or ""),
            ]
            return normalize_whitespace(" ".join(x for x in parts if str(x).strip()))

        class _SimpleBM25Okapi:
            def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
                self.corpus_tokens = corpus_tokens
                self.k1 = k1
                self.b = b
                self.doc_freqs: List[Counter] = []
                self.doc_len: List[int] = []
                self.idf: Dict[str, float] = {}
                self.corpus_size = len(corpus_tokens)
                self.avgdl = 0.0

                if not corpus_tokens:
                    return

                nd: Counter = Counter()
                total_len = 0

                for tokens in corpus_tokens:
                    freqs = Counter(tokens)
                    self.doc_freqs.append(freqs)
                    dl = len(tokens)
                    self.doc_len.append(dl)
                    total_len += dl
                    for token in freqs.keys():
                        nd[token] += 1

                self.avgdl = total_len / max(1, self.corpus_size)

                for token, freq in nd.items():
                    self.idf[token] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

            def get_scores(self, query_tokens: List[str]) -> List[float]:
                if not self.corpus_tokens or not query_tokens:
                    return [0.0] * self.corpus_size

                scores = [0.0] * self.corpus_size
                query_terms = Counter(query_tokens)

                for idx, freqs in enumerate(self.doc_freqs):
                    dl = self.doc_len[idx]
                    norm = self.k1 * (1 - self.b + self.b * dl / max(1e-12, self.avgdl))
                    score = 0.0

                    for term, qtf in query_terms.items():
                        tf = freqs.get(term, 0)
                        if tf <= 0:
                            continue
                        idf = self.idf.get(term, 0.0)
                        numer = tf * (self.k1 + 1)
                        denom = tf + norm
                        score += idf * (numer / max(1e-12, denom)) * qtf

                    scores[idx] = score

                return scores

        flattened: List[Dict[str, Any]] = []
        seen_source_keys = set()

        for query_idx, item in enumerate(query_to_result):
            if not isinstance(item, tuple) or len(item) != 2:
                continue

            query_key, result_data = item
            if not isinstance(result_data, dict):
                continue

            sources = result_data.get("sources") or []
            if not isinstance(sources, list):
                continue

            for source_idx, src in enumerate(sources):
                if not isinstance(src, dict):
                    continue

                source_key = self._build_source_identity_key(src)
                if source_key in seen_source_keys:
                    continue
                seen_source_keys.add(source_key)

                text = _build_source_text(src)
                if not text:
                    continue

                flattened.append(
                    {
                        "query_idx": query_idx,
                        "query_key": normalize_whitespace(query_key),
                        "result_data": result_data,
                        "source_idx": source_idx,
                        "source": dict(src),
                        "text": text,
                    }
                )

        if not flattened:
            return query_to_result

        corpus_tokens = [_tokenize(item["text"]) for item in flattened]
        query_tokens = _tokenize(safe_query)

        if not query_tokens:
            return query_to_result

        bm25 = _SimpleBM25Okapi(corpus_tokens)
        scores = bm25.get_scores(query_tokens)

        ranked_rows = sorted(
            [
                (float(score), idx, flattened[idx])
                for idx, score in enumerate(scores)
            ],
            key=lambda x: x[0],
            reverse=True,
        )

        selected_rows = ranked_rows[:safe_top_k]
        if not selected_rows:
            return query_to_result

        rebuilt: List[Tuple[str, Dict[str, Any]]] = []

        for picked_rank, (bm25_score, _, row) in enumerate(selected_rows, start=1):
            src = dict(row["source"])
            src["bm25_score"] = float(bm25_score)
            src["bm25_rank"] = picked_rank

            new_result = {
                "context": normalize_whitespace(
                    str(
                        src.get("content")
                        or src.get("snippet")
                        or src.get("preview")
                        or src.get("document")
                        or ""
                    )
                ),
                "sources": [src],
            }
            rebuilt.append((row["query_key"], new_result))

        return rebuilt or query_to_result

    def collect_search_items_from_result(self, result_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(result_data, dict):
            return []

        selected_items = result_data.get("selected_items")
        if isinstance(selected_items, list) and selected_items:
            return [x for x in selected_items if isinstance(x, dict)]

        items = result_data.get("items")
        if isinstance(items, list) and items:
            return [x for x in items if isinstance(x, dict)]

        return []

    def merge_search_items(self, items_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen_urls = set()

        for items in items_list:
            if not isinstance(items, list):
                continue

            for item in items:
                if not isinstance(item, dict):
                    continue

                url = normalize_whitespace(item.get("link") or "")
                if not url or url in seen_urls:
                    continue

                seen_urls.add(url)
                merged.append(item)

        return merged