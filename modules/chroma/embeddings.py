from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from FlagEmbedding import BGEM3FlagModel


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-m3"
    use_fp16: bool = True
    batch_size: int = 2
    normalize_embeddings: bool = True
    device: Optional[str] = "cuda:5"
    query_max_length: int = 256
    passage_max_length: int = 512


class BGEEmbeddings:
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()

        self.model = BGEM3FlagModel(
            self.config.model_name,
            use_fp16=self.config.use_fp16,
            normalize_embeddings=self.config.normalize_embeddings,
            devices=self.config.device,   # 핵심: device 아님, devices
            batch_size=self.config.batch_size,
            query_max_length=self.config.query_max_length,
            passage_max_length=self.config.passage_max_length,
        )

    @property
    def target_devices(self):
        return getattr(self.model, "target_devices", None)

    def _sanitize_text(self, text: str) -> str:
        if text is None:
            return " "
        text = str(text).strip()
        return text if text else " "

    def _sanitize_texts(self, texts: Sequence[str]) -> List[str]:
        if not isinstance(texts, (list, tuple)):
            raise TypeError("texts must be a list or tuple of strings")
        return [self._sanitize_text(t) for t in texts]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        cleaned = self._sanitize_texts(texts)
        if not cleaned:
            return []

        result = self.model.encode_corpus(
            cleaned,
            batch_size=self.config.batch_size,
            max_length=self.config.passage_max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        dense_vecs = result["dense_vecs"]
        return dense_vecs.tolist() if hasattr(dense_vecs, "tolist") else dense_vecs

    def embed_query(self, text: str) -> List[float]:
        cleaned = self._sanitize_texts([text])
        if not cleaned:
            return []

        result = self.model.encode_queries(
            cleaned,
            batch_size=1,
            max_length=self.config.query_max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        dense_vecs = result["dense_vecs"]
        dense_vecs = dense_vecs.tolist() if hasattr(dense_vecs, "tolist") else dense_vecs
        return dense_vecs[0] if dense_vecs else []


_default_instance: Optional[BGEEmbeddings] = None


def init_embeddings(config: Optional[EmbeddingConfig] = None) -> BGEEmbeddings:
    global _default_instance
    if _default_instance is None:
        _default_instance = BGEEmbeddings(config)
    return _default_instance


def get_embeddings() -> BGEEmbeddings:
    global _default_instance
    if _default_instance is None:
        _default_instance = BGEEmbeddings()
    return _default_instance


def reset_embeddings() -> None:
    global _default_instance
    _default_instance = None


def embed_documents(texts: Sequence[str]) -> List[List[float]]:
    return get_embeddings().embed_documents(texts)


def embed_query(text: str) -> List[float]:
    return get_embeddings().embed_query(text)