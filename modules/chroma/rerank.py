from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from FlagEmbedding import FlagReranker


@dataclass
class RerankerConfig:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    use_fp16: bool = True
    device: Optional[str] = "cuda:4"


class BGEReranker:
    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig()
        self.model = self._load_model()

    def _load_model(self):
        kwargs = {
            "model_name_or_path": self.config.model_name,
            "use_fp16": self.config.use_fp16,
        }

        if self.config.device is not None:
            kwargs["devices"] = self.config.device

        try:
            return FlagReranker(**kwargs)
        except Exception:
            if self.config.use_fp16:
                fallback_kwargs = dict(kwargs)
                fallback_kwargs["use_fp16"] = False
                return FlagReranker(**fallback_kwargs)
            raise

    def _to_float_score(self, score: Any) -> Optional[float]:
        if score is None:
            return None

        if isinstance(score, (int, float)):
            return float(score)

        if hasattr(score, "tolist"):
            score = score.tolist()

        while isinstance(score, (list, tuple)) and len(score) > 0:
            score = score[0]

        try:
            return float(score)
        except Exception:
            return None

    def compute_scores(
        self,
        query: str,
        texts: Sequence[str],
    ) -> List[Optional[float]]:
        if not texts:
            return []

        pairs = [[query, str(text or "")] for text in texts]
        scores = self.model.compute_score(pairs)

        if hasattr(scores, "tolist"):
            scores = scores.tolist()

        if not isinstance(scores, list):
            scores = [scores]

        return [self._to_float_score(score) for score in scores]

    def rerank_texts(
        self,
        query: str,
        texts: Sequence[str],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        scores = self.compute_scores(query, texts)

        rows: List[Dict[str, Any]] = []
        for idx, (text, score) in enumerate(zip(texts, scores)):
            rows.append(
                {
                    "index": idx,
                    "text": text,
                    "score": score,
                }
            )

        rows.sort(
            key=lambda x: x["score"] if x["score"] is not None else float("-inf"),
            reverse=True,
        )

        if top_k is not None:
            rows = rows[:top_k]

        return rows

    def rerank_rows(
        self,
        query: str,
        rows: Sequence[Dict[str, Any]],
        text_key: str = "document",
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not rows:
            return []

        texts = [str((row or {}).get(text_key) or "") for row in rows]
        scores = self.compute_scores(query, texts)

        reranked: List[Dict[str, Any]] = []
        for row, score in zip(rows, scores):
            item = dict(row)
            item["rerank_score"] = score
            reranked.append(item)

        reranked.sort(
            key=lambda x: x["rerank_score"] if x["rerank_score"] is not None else float("-inf"),
            reverse=True,
        )

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked


_default_instance: Optional[BGEReranker] = None


def init_reranker(config: Optional[RerankerConfig] = None) -> BGEReranker:
    global _default_instance
    if _default_instance is None:
        _default_instance = BGEReranker(config)
    return _default_instance


def get_reranker() -> BGEReranker:
    global _default_instance
    if _default_instance is None:
        _default_instance = BGEReranker()
    return _default_instance


def reset_reranker() -> None:
    global _default_instance
    _default_instance = None


def rerank_rows(
    query: str,
    rows: Sequence[Dict[str, Any]],
    text_key: str = "document",
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    return get_reranker().rerank_rows(
        query=query,
        rows=rows,
        text_key=text_key,
        top_k=top_k,
    )