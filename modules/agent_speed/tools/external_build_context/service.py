from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.chroma.embeddings import embed_documents, embed_query
from ..base import ToolResult, normalize_whitespace


# -----------------------------------------------------------------------------
# optional reranker
# -----------------------------------------------------------------------------

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


_RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_RERANKER_INSTANCE = None


def _get_reranker():
    global _RERANKER_INSTANCE

    if _RERANKER_INSTANCE is not None:
        return _RERANKER_INSTANCE

    if CrossEncoder is None:
        return None

    try:
        _RERANKER_INSTANCE = CrossEncoder(_RERANKER_MODEL_NAME)
        return _RERANKER_INSTANCE
    except Exception:
        return None


# -----------------------------------------------------------------------------
# text utils
# -----------------------------------------------------------------------------

def _normalize_text(text: Any) -> str:
    return normalize_whitespace(str(text or ""))


def _normalize_compact(text: Any) -> str:
    return re.sub(r"\s+", "", _normalize_text(text).lower())


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _pick_first_text(mapping: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = mapping.get(key)
        text = _normalize_text(value)
        if text:
            return text
    return ""


def _normalize_score_weights(original_weight: float, atomic_weight: float) -> Tuple[float, float]:
    total = max(1e-12, float(original_weight) + float(atomic_weight))
    return float(original_weight) / total, float(atomic_weight) / total


# -----------------------------------------------------------------------------
# debug helpers
# -----------------------------------------------------------------------------

def _debug_print_dense_candidates(
    *,
    query: str,
    documents: List[Dict[str, Any]],
    dense_rows: List[Tuple[float, int]],
) -> None:
    print("\n" + "=" * 120)
    print(f"[DEBUG][DENSE CANDIDATES] query={query}")
    print("=" * 120)

    for rank, (dense_score, doc_idx) in enumerate(dense_rows, start=1):
        doc = documents[doc_idx]
        full_text = _normalize_text(doc.get("text", ""))
        print(
            f"[DENSE {rank:02d}] "
            f"doc_index={doc_idx} | "
            f"dense_score={dense_score:.6f}"
        )
        print(f"  URL   : {doc.get('url', '')}")
        print(f"  TITLE : {doc.get('title', '')}")
        print(f"  HEAD  : {doc.get('heading', '')}")
        print(f"  TEXT_LEN: {len(full_text)}")
        print(f"  PREVIEW: {full_text[:1000]}")
        print("-" * 120)


def _debug_print_rerank_candidates(
    *,
    query: str,
    documents: List[Dict[str, Any]],
    rerank_rows: List[Tuple[float, float, int]],
) -> None:
    print("\n" + "=" * 120)
    print(f"[DEBUG][RERANK INPUT] query={query}")
    print("=" * 120)

    for rank, (dense_score, bm25_score, doc_idx) in enumerate(rerank_rows, start=1):
        doc = documents[doc_idx]
        full_text = _normalize_text(doc.get("text", ""))
        print(
            f"[RERANK-IN {rank:02d}] "
            f"doc_index={doc_idx} | "
            f"dense_score={dense_score:.6f} | "
            f"bm25={bm25_score:.4f}"
        )
        print(f"  URL   : {doc.get('url', '')}")
        print(f"  TITLE : {doc.get('title', '')}")
        print(f"  HEAD  : {doc.get('heading', '')}")
        print(f"  TEXT_LEN: {len(full_text)}")
        print(f"  PREVIEW: {full_text[:1000]}")
        print("-" * 120)


def _debug_print_final_ranking(
    *,
    query: str,
    original_query: str,
    atomic_query: str,
    documents: List[Dict[str, Any]],
    scores: List[Tuple[float, float, float, float, float, float, int]],
    top_k: int,
) -> None:
    print("\n" + "=" * 120)
    print(f"[DEBUG][FINAL RANKING] retrieval_query={query}")
    print(f"[DEBUG][FINAL RANKING] original_query={original_query}")
    print(f"[DEBUG][FINAL RANKING] atomic_query={atomic_query}")
    print("=" * 120)

    for rank, (
        final_score,
        rerank_blended_score,
        rerank_original_score,
        rerank_atomic_score,
        dense_score,
        bm25_score,
        doc_idx,
    ) in enumerate(scores, start=1):
        doc = documents[doc_idx]
        full_text = _normalize_text(doc.get("text", ""))
        print(
            f"[FINAL {rank:02d}] "
            f"doc_index={doc_idx} | "
            f"final_score={final_score:.6f} | "
            f"rerank_blended={rerank_blended_score:.6f} | "
            f"rerank_original={rerank_original_score:.6f} | "
            f"rerank_atomic={rerank_atomic_score:.6f} | "
            f"dense_score={dense_score:.6f} | "
            f"bm25={bm25_score:.4f}"
        )
        print(f"  URL   : {doc.get('url', '')}")
        print(f"  TITLE : {doc.get('title', '')}")
        print(f"  HEAD  : {doc.get('heading', '')}")
        print(f"  TEXT_LEN: {len(full_text)}")
        print(f"  PREVIEW: {full_text[:1000]}")
        if rank == top_k:
            print("-" * 120)
            print(f"[DEBUG] ---- above is top_k={top_k} cutoff ----")
        print("-" * 120)


# -----------------------------------------------------------------------------
# flatten
# -----------------------------------------------------------------------------

def _normalize_block_record(
    *,
    parent_meta: Dict[str, Any],
    doc: Dict[str, Any],
    block: Dict[str, Any],
    fallback_block_id: str,
) -> Optional[Dict[str, Any]]:
    text = _pick_first_text(block, "text", "content", "raw_text", "preview", "body")

    normalized = {
        "url": block.get("url", "") or parent_meta["url"],
        "final_url": block.get("final_url", "") or parent_meta["final_url"],
        "title": _normalize_text(block.get("title", "")) or parent_meta["title"],
        "heading": _normalize_text(block.get("heading", "")) or _normalize_text(doc.get("heading", "")),
        "page": _safe_int(block.get("page")),
        "content_type": block.get("content_type", "") or doc.get("content_type", ""),
        "block_id": block.get("block_id", "") or fallback_block_id,
        "search_title": _normalize_text(block.get("search_title", "")) or parent_meta["search_title"],
        "search_snippet": _normalize_text(block.get("search_snippet", "")) or parent_meta["search_snippet"],
        "display_link": _normalize_text(block.get("display_link", "")) or parent_meta["display_link"],
        "source_engine": _normalize_text(block.get("source_engine", "")) or parent_meta["source_engine"],
        "block_score": block.get("block_score"),
        "text": text,
    }

    if not text:
        return None

    return normalized


def _flatten_documents_to_blocks(
    documents: List[Dict[str, Any]],
    debug_print=None,
) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []

    for doc_idx, doc in enumerate(documents, start=1):
        if not isinstance(doc, dict):
            continue

        parent_meta = {
            "url": doc.get("url", ""),
            "final_url": doc.get("final_url", ""),
            "title": _normalize_text(doc.get("title", "")),
            "search_title": _normalize_text(doc.get("search_title", "")),
            "search_snippet": _normalize_text(doc.get("search_snippet", "")),
            "display_link": _normalize_text(doc.get("display_link", "")),
            "source_engine": _normalize_text(doc.get("source_engine", "")),
        }

        blocks = doc.get("blocks") or []
        if isinstance(blocks, list) and blocks:
            for idx, block in enumerate(blocks, start=1):
                if not isinstance(block, dict):
                    continue
                normalized = _normalize_block_record(
                    parent_meta=parent_meta,
                    doc=doc,
                    block=block,
                    fallback_block_id=f"doc-{doc_idx}-block-{idx}",
                )
                if normalized:
                    flattened.append(normalized)
            continue

        chunks = doc.get("chunks") or []
        if isinstance(chunks, list) and chunks:
            for idx, chunk in enumerate(chunks, start=1):
                if isinstance(chunk, dict):
                    normalized = _normalize_block_record(
                        parent_meta=parent_meta,
                        doc=doc,
                        block=chunk,
                        fallback_block_id=f"doc-{doc_idx}-chunk-{idx}",
                    )
                    if normalized:
                        flattened.append(normalized)
                    continue

                text = _normalize_text(chunk)
                if not text:
                    continue

                flattened.append(
                    {
                        "url": parent_meta["url"],
                        "final_url": parent_meta["final_url"],
                        "title": parent_meta["title"],
                        "heading": _normalize_text(doc.get("heading", "")) or parent_meta["title"],
                        "page": _safe_int(doc.get("page")),
                        "content_type": doc.get("content_type", ""),
                        "block_id": f"doc-{doc_idx}-chunk-{idx}",
                        "search_title": parent_meta["search_title"],
                        "search_snippet": parent_meta["search_snippet"],
                        "display_link": parent_meta["display_link"],
                        "source_engine": parent_meta["source_engine"],
                        "block_score": None,
                        "text": text,
                    }
                )
            continue

        text = _pick_first_text(doc, "text", "content", "raw_text", "preview", "body")
        if not text:
            continue

        flattened.append(
            {
                "url": parent_meta["url"],
                "final_url": parent_meta["final_url"],
                "title": parent_meta["title"],
                "heading": _normalize_text(doc.get("heading", "")) or parent_meta["title"],
                "page": _safe_int(doc.get("page")),
                "content_type": doc.get("content_type", ""),
                "block_id": doc.get("block_id", "") or f"doc-{doc_idx}-full",
                "search_title": parent_meta["search_title"],
                "search_snippet": parent_meta["search_snippet"],
                "display_link": parent_meta["display_link"],
                "source_engine": parent_meta["source_engine"],
                "block_score": doc.get("block_score"),
                "text": text,
            }
        )

    return flattened


def build_embedding_documents(
    documents: List[Dict[str, Any]],
    debug_print=None,
) -> List[Dict[str, Any]]:
    return _flatten_documents_to_blocks(documents, debug_print=debug_print)


# -----------------------------------------------------------------------------
# query helpers
# -----------------------------------------------------------------------------

def _extract_focus_terms(query: str, max_terms: int = 18) -> List[str]:
    normalized = _normalize_text(query)
    if not normalized:
        return []

    parts = re.findall(r"[0-9A-Za-z가-힣]+", normalized)
    out: List[str] = []
    seen = set()

    for part in parts:
        token = _normalize_text(part)
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(token)
        if len(out) >= max_terms:
            break

    return out


def _build_semantic_query(query: str, focus_terms: List[str]) -> str:
    return _normalize_text(query)


def _extract_entity_hint(query: str) -> str:
    return ""


def _is_multi_entity_query(query: str) -> bool:
    return False


# -----------------------------------------------------------------------------
# bm25
# -----------------------------------------------------------------------------

def _tokenize_ko_en(text: str) -> List[str]:
    normalized = _normalize_text(text).lower()
    if not normalized:
        return []
    return re.findall(r"[0-9a-zA-Z가-힣]+", normalized)


def _unique_query_tokens(query: str) -> List[str]:
    tokens = _tokenize_ko_en(query)
    seen = set()
    out = []

    for token in tokens:
        t = _normalize_text(token).lower()
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(token)

    return out


class SimpleBM25Okapi:
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


def _normalize_scores(values: List[float]) -> List[float]:
    if not values:
        return []

    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


# -----------------------------------------------------------------------------
# retrieval config
# -----------------------------------------------------------------------------

MAX_CONTEXT_SOURCES = 10

RERANK_ENABLED = True

DENSE_RETRIEVAL_MULTIPLIER = 8
DENSE_RETRIEVAL_MIN_CANDIDATES = 30

RERANK_CANDIDATE_MULTIPLIER = 6
RERANK_MIN_CANDIDATES = 20

BM25_BONUS_WEIGHT = 0.80

RERANK_ORIGINAL_QUERY_WEIGHT = 0.65
RERANK_ATOMIC_QUERY_WEIGHT = 0.35


# -----------------------------------------------------------------------------
# embedding utils
# -----------------------------------------------------------------------------

def cosine_similarity(query_embedding: np.ndarray, doc_embedding: np.ndarray) -> float:
    denom = np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
    if denom == 0:
        return 0.0
    return float(np.dot(query_embedding, doc_embedding) / denom)


# -----------------------------------------------------------------------------
# rerank utils
# -----------------------------------------------------------------------------

def _compute_rerank_scores(
    query: str,
    docs: List[Dict[str, Any]],
) -> Optional[List[float]]:
    reranker = _get_reranker()
    if reranker is None:
        return None

    try:
        pairs = [(query, _normalize_text(doc.get("text", ""))) for doc in docs]
        scores = reranker.predict(pairs)
        return [float(x) for x in scores]
    except Exception:
        return None


# -----------------------------------------------------------------------------
# dense retrieval
# -----------------------------------------------------------------------------

def _select_dense_candidates(
    documents: List[Dict[str, Any]],
    query: str,
    k: int,
    debug_print=None,
) -> Tuple[List[Tuple[float, int]], List[float]]:
    if not documents:
        return [], []

    dense_k = max(DENSE_RETRIEVAL_MIN_CANDIDATES, k * DENSE_RETRIEVAL_MULTIPLIER)
    dense_k = min(dense_k, len(documents))

    semantic_query = _build_semantic_query(query, _extract_focus_terms(query))
    query_embedding = embed_query(semantic_query)

    texts = [doc.get("text", "") for doc in documents]
    doc_embeddings = embed_documents(texts)

    dense_rows: List[Tuple[float, int]] = []
    dense_scores_all: List[float] = []

    for idx, doc_embedding in enumerate(doc_embeddings):
        dense_score = cosine_similarity(query_embedding, doc_embedding)
        dense_scores_all.append(dense_score)
        dense_rows.append((dense_score, idx))

    dense_rows.sort(key=lambda x: x[0], reverse=True)
    dense_rows = dense_rows[:dense_k]

    if debug_print:
        debug_print(
            "DENSE CANDIDATE META",
            {
                "query": query,
                "dense_candidate_size": len(dense_rows),
                "dense_top_k": dense_k,
                "top_candidates": [
                    {
                        "doc_index": idx,
                        "dense_score": dense_score,
                    }
                    for dense_score, idx in dense_rows[:10]
                ],
            },
        )

    _debug_print_dense_candidates(
        query=query,
        documents=documents,
        dense_rows=dense_rows,
    )

    return dense_rows, dense_scores_all


# -----------------------------------------------------------------------------
# retrieval
# -----------------------------------------------------------------------------

def similarity_search(
    query: str,
    documents: List[Dict[str, Any]],
    k: int = 5,
    debug_print=None,
    rerank_enabled: bool = RERANK_ENABLED,
    rerank_top_n: Optional[int] = None,
    original_query: Optional[str] = None,
    atomic_query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not documents:
        return []

    normalized_query = _normalize_text(query)
    if not normalized_query:
        return []

    normalized_atomic_query = _normalize_text(atomic_query or normalized_query)
    normalized_original_query = _normalize_text(original_query or normalized_atomic_query)

    retrieval_query = normalized_atomic_query
    final_query = normalized_original_query or retrieval_query

    dense_rows, dense_scores_all = _select_dense_candidates(
        documents=documents,
        query=retrieval_query,
        k=max(1, int(k)),
        debug_print=debug_print,
    )
    if not dense_rows:
        return []

    query_tokens = _unique_query_tokens(retrieval_query)
    corpus_tokens = [_tokenize_ko_en(doc.get("text", "")) for doc in documents]
    bm25 = SimpleBM25Okapi(corpus_tokens)
    bm25_raw_all = bm25.get_scores(query_tokens)
    bm25_norm_all = _normalize_scores(bm25_raw_all)

    rerank_n = rerank_top_n or max(RERANK_MIN_CANDIDATES, k * RERANK_CANDIDATE_MULTIPLIER)
    rerank_n = min(rerank_n, len(dense_rows))

    rerank_pool_dense = dense_rows[:rerank_n]

    rerank_rows: List[Tuple[float, float, int]] = []
    for dense_score, idx in rerank_pool_dense:
        rerank_rows.append((dense_score, bm25_raw_all[idx], idx))

    original_weight, atomic_weight = _normalize_score_weights(
        RERANK_ORIGINAL_QUERY_WEIGHT,
        RERANK_ATOMIC_QUERY_WEIGHT,
    )

    if debug_print:
        debug_print(
            "RERANK META",
            {
                "query": normalized_query,
                "retrieval_query": retrieval_query,
                "original_query": final_query,
                "atomic_query": normalized_atomic_query,
                "rerank_enabled": rerank_enabled,
                "rerank_top_n": rerank_n,
                "dense_candidate_count": len(dense_rows),
                "reranker_model": _RERANKER_MODEL_NAME,
                "bm25_bonus_weight": BM25_BONUS_WEIGHT,
                "rerank_original_query_weight": original_weight,
                "rerank_atomic_query_weight": atomic_weight,
            },
        )

    _debug_print_rerank_candidates(
        query=retrieval_query,
        documents=documents,
        rerank_rows=rerank_rows,
    )

    reranker_applied = False
    final_scores: List[Tuple[float, float, float, float, float, float, int]] = []

    if rerank_enabled:
        rerank_docs = [documents[idx] for _, _, idx in rerank_rows]

        rerank_scores_atomic_raw = _compute_rerank_scores(normalized_atomic_query, rerank_docs)

        rerank_scores_original_raw: Optional[List[float]]
        if final_query == normalized_atomic_query:
            rerank_scores_original_raw = rerank_scores_atomic_raw
        else:
            rerank_scores_original_raw = _compute_rerank_scores(final_query, rerank_docs)

        if rerank_scores_original_raw is not None or rerank_scores_atomic_raw is not None:
            reranker_applied = True

            if rerank_scores_original_raw is None:
                rerank_scores_original_raw = rerank_scores_atomic_raw
            if rerank_scores_atomic_raw is None:
                rerank_scores_atomic_raw = rerank_scores_original_raw

            if rerank_scores_original_raw is not None and rerank_scores_atomic_raw is not None:
                rerank_original_norm = _normalize_scores(rerank_scores_original_raw)
                rerank_atomic_norm = _normalize_scores(rerank_scores_atomic_raw)

                for i, (dense_score, bm25_raw_score, idx) in enumerate(rerank_rows):
                    bm25_bonus = BM25_BONUS_WEIGHT * bm25_norm_all[idx]
                    rerank_blended_score = (
                        original_weight * rerank_original_norm[i]
                        + atomic_weight * rerank_atomic_norm[i]
                    )
                    final_score = rerank_blended_score + bm25_bonus

                    final_scores.append(
                        (
                            final_score,
                            rerank_blended_score,
                            float(rerank_scores_original_raw[i]),
                            float(rerank_scores_atomic_raw[i]),
                            dense_score,
                            bm25_raw_score,
                            idx,
                        )
                    )

                final_scores.sort(key=lambda x: x[0], reverse=True)

    if not final_scores:
        dense_norm = _normalize_scores([score for score, _ in dense_rows])

        for i, (dense_score, idx) in enumerate(dense_rows):
            bm25_bonus = BM25_BONUS_WEIGHT * bm25_norm_all[idx]
            final_score = dense_norm[i] + bm25_bonus

            final_scores.append(
                (
                    final_score,
                    0.0,
                    0.0,
                    0.0,
                    dense_score,
                    bm25_raw_all[idx],
                    idx,
                )
            )

        final_scores.sort(key=lambda x: x[0], reverse=True)

    _debug_print_final_ranking(
        query=retrieval_query,
        original_query=final_query,
        atomic_query=normalized_atomic_query,
        documents=documents,
        scores=final_scores,
        top_k=max(1, int(k)),
    )

    results: List[Dict[str, Any]] = []
    for rank, (
        score,
        rerank_blended_score,
        rerank_original_score,
        rerank_atomic_score,
        dense_score,
        bm25_score,
        idx,
    ) in enumerate(final_scores[: max(1, int(k))], start=1):
        doc = documents[idx]
        results.append(
            {
                "rank": rank,
                "url": doc.get("url", ""),
                "final_url": doc.get("final_url", ""),
                "title": doc.get("title", ""),
                "heading": doc.get("heading", ""),
                "page": doc.get("page"),
                "content_type": doc.get("content_type", ""),
                "block_id": doc.get("block_id", ""),
                "search_title": doc.get("search_title", ""),
                "search_snippet": doc.get("search_snippet", ""),
                "display_link": doc.get("display_link", ""),
                "source_engine": doc.get("source_engine", ""),
                "block_score": doc.get("block_score"),
                "preview": doc.get("text", ""),
                "content": doc.get("text", ""),
                "score": score,
                "rerank_score": rerank_blended_score,
                "rerank_blended_score": rerank_blended_score,
                "rerank_original_score": rerank_original_score,
                "rerank_atomic_score": rerank_atomic_score,
                "reranker_applied": reranker_applied,
                "embedding_score": dense_score,
                "bm25_score": bm25_score,
                "lexical_hit_count": 0.0,
                "lexical_hit_ratio": 0.0,
                "matched_tokens": [],
                "lexical_bonus": BM25_BONUS_WEIGHT * bm25_norm_all[idx],
                "bm25_strong": False,
                "bonus_detail": {
                    "bm25_bonus_weight": BM25_BONUS_WEIGHT,
                    "bm25_norm": bm25_norm_all[idx],
                },
                "score_weights": {
                    "embedding_first_stage": 1.0,
                    "reranker": _RERANKER_MODEL_NAME if reranker_applied else None,
                    "bm25_bonus_weight": BM25_BONUS_WEIGHT,
                    "rerank_original_query_weight": original_weight,
                    "rerank_atomic_query_weight": atomic_weight,
                },
            }
        )

    return results


# -----------------------------------------------------------------------------
# source grouping / context aggregation
# -----------------------------------------------------------------------------

def _dedupe_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()

    for src in sources:
        key = (
            _normalize_text(src.get("url", "")),
            _normalize_text(src.get("block_id", "")),
            _normalize_text(src.get("heading", "")),
            _normalize_compact(src.get("preview", ""))[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(src)

    return out


def _boost_sources_for_original_query(
    sources: List[Dict[str, Any]],
    original_query: str,
    atomic_query: str,
) -> List[Dict[str, Any]]:
    return sources


def _group_sources_by_entity(
    sources: List[Dict[str, Any]],
    original_query: str,
    atomic_query: str,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    return [("", sources)]


def _render_source_header(result: Dict[str, Any]) -> str:
    lines = [
        f"title={result.get('title', '')}",
        f"heading={result.get('heading', '')}",
        f"block_id={result.get('block_id', '')}",
        f"content_type={result.get('content_type', '')}",
        f"score={result.get('score', 0.0):.4f}",
        f"rerank_score={result.get('rerank_score', 0.0):.4f}",
        f"rerank_original_score={result.get('rerank_original_score', 0.0):.4f}",
        f"rerank_atomic_score={result.get('rerank_atomic_score', 0.0):.4f}",
        f"embedding_score={result.get('embedding_score', 0.0):.4f}",
        f"bm25_score={result.get('bm25_score', 0.0):.4f}",
        f"lexical_bonus={result.get('lexical_bonus', 0.0):.4f}",
        f"reranker_applied={str(bool(result.get('reranker_applied', False))).lower()}",
    ]

    page = result.get("page")
    if page is not None:
        lines.append(f"page={page}")

    return "\n".join(lines)


def _render_grouped_context(
    *,
    original_query: str,
    atomic_query: str,
    grouped_sources: List[Tuple[str, List[Dict[str, Any]]]],
) -> str:
    lines: List[str] = []
    lines.append("아래는 외부 문서에서 추출한 참고 문서 조각들입니다.")
    lines.append("")
    lines.append("[original_query]")
    lines.append(_normalize_text(original_query))
    lines.append("")
    lines.append("[atomic_query]")
    lines.append(_normalize_text(atomic_query))
    lines.append("")

    source_rank = 1
    for _, entity_sources in grouped_sources:
        if not entity_sources:
            continue

        for src in entity_sources:
            lines.append(f"[source {source_rank}]")
            lines.append(_render_source_header(src))
            lines.append(str(src.get("preview", "") or "").strip())
            lines.append("")
            source_rank += 1

    lines.append(f"사용자원문질문: {_normalize_text(original_query)}")
    lines.append(f"현재검색질문: {_normalize_text(atomic_query)}")
    lines.append("답변 시에는 현재검색질문 하나만 대표로 답하지 말고 사용자원문질문 기준으로 해석해야 한다.")
    return "\n".join(lines).strip()


# -----------------------------------------------------------------------------
# run
# -----------------------------------------------------------------------------

def run(
    query: str,
    documents: Optional[List[Dict[str, Any]]] = None,
    top_k_chunks: int = 5,
    original_query: Optional[str] = None,
    atomic_query: Optional[str] = None,
    debug_print=None,
) -> ToolResult:
    try:
        documents = documents or []

        if not isinstance(documents, list) or not documents:
            return ToolResult(
                name="external_build_context",
                ok=False,
                data={"error": "documents is required"},
            )

        normalized_original_query = _normalize_text(original_query or query)
        normalized_query = _normalize_text(query or atomic_query or normalized_original_query)
        normalized_atomic_query = _normalize_text(atomic_query or normalized_query)

        focus_terms = _extract_focus_terms(normalized_query)
        semantic_query = _build_semantic_query(normalized_query, focus_terms)

        embedded_documents = build_embedding_documents(
            documents,
            debug_print=debug_print,
        )

        if not embedded_documents:
            return ToolResult(
                name="external_build_context",
                ok=False,
                data={"error": "no documents to embed"},
            )

        atomic_results = similarity_search(
            query=semantic_query,
            documents=embedded_documents,
            k=max(1, int(top_k_chunks)),
            debug_print=debug_print,
            rerank_enabled=RERANK_ENABLED,
            original_query=normalized_original_query,
            atomic_query=normalized_atomic_query,
        )

        merged_sources = _dedupe_sources(atomic_results or [])
        merged_sources = _boost_sources_for_original_query(
            sources=merged_sources,
            original_query=normalized_original_query,
            atomic_query=normalized_atomic_query,
        )
        merged_sources = merged_sources[:MAX_CONTEXT_SOURCES]

        if not merged_sources:
            return ToolResult(
                name="external_build_context",
                ok=False,
                data={"error": "no similarity results found"},
            )

        grouped_sources = _group_sources_by_entity(
            sources=merged_sources,
            original_query=normalized_original_query,
            atomic_query=normalized_atomic_query,
        )

        context = _render_grouped_context(
            original_query=normalized_original_query,
            atomic_query=normalized_atomic_query,
            grouped_sources=grouped_sources,
        )

        if debug_print:
            debug_print(
                "EXTERNAL_BUILD_CONTEXT FINAL META",
                {
                    "query": normalized_query,
                    "original_query": normalized_original_query,
                    "atomic_query": normalized_atomic_query,
                    "semantic_query": semantic_query,
                    "original_semantic_query": None,
                    "multi_entity_query": _is_multi_entity_query(normalized_original_query),
                    "num_documents_embedded": len(embedded_documents),
                    "atomic_sources": len(atomic_results),
                    "original_sources": 0,
                    "merged_sources": len(merged_sources),
                    "entity_groups": [label for label, _ in grouped_sources],
                },
            )

        exported_sources: List[Dict[str, Any]] = []
        for entity_label, entity_items in grouped_sources:
            for src in entity_items:
                enriched = dict(src)
                enriched["original_query"] = normalized_original_query
                enriched["atomic_query"] = normalized_atomic_query
                enriched["entity_label"] = entity_label
                enriched["content"] = str(src.get("preview", "") or "").strip()
                exported_sources.append(enriched)

        return ToolResult(
            name="external_build_context",
            ok=True,
            data={
                "query": normalized_query,
                "original_query": normalized_original_query,
                "atomic_query": normalized_atomic_query,
                "semantic_query": semantic_query,
                "original_semantic_query": None,
                "focus_terms": focus_terms,
                "original_focus_terms": [],
                "multi_entity_query": _is_multi_entity_query(normalized_original_query),
                "entity_hint": _extract_entity_hint(normalized_atomic_query),
                "entity_groups": [label for label, _ in grouped_sources],
                "num_documents_embedded": len(embedded_documents),
                "sources": exported_sources,
                "context": context,
                "top_k_chunks": top_k_chunks,
            },
        )

    except Exception as e:
        if debug_print:
            debug_print("EXTERNAL_BUILD_CONTEXT ERROR", str(e))
        return ToolResult(
            name="external_build_context",
            ok=False,
            data={"error": str(e)},
        )