from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ChunkConfig:
    chunk_size: int = 800
    chunk_overlap: int = 120
    min_chunk_size: int = 80
    separator_pattern: str = r"\n\n|\n|\. "


@dataclass
class Chunk:
    id: str
    document: str
    metadata: Dict


class TextChunker:
    """
    RAG용 텍스트 chunk 분할기
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()

    # ---------------------------------------------------------
    # text split
    # ---------------------------------------------------------

    def _split_units(self, text: str) -> List[str]:
        """
        문단/문장 단위로 먼저 분할
        """
        pattern = self.config.separator_pattern
        parts = re.split(pattern, text)

        units: List[str] = []
        for p in parts:
            p = p.strip()
            if p:
                units.append(p)

        return units

    def split_text(self, text: str) -> List[str]:
        """
        텍스트를 chunk 단위로 분할
        """
        if not text:
            return []

        units = self._split_units(text)

        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        chunks: List[str] = []
        current = ""

        for unit in units:
            candidate = (current + " " + unit).strip()

            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)

            # overlap 처리
            if overlap > 0 and chunks:
                prev = chunks[-1]
                current = prev[-overlap:] + " " + unit
            else:
                current = unit

        if current:
            chunks.append(current)

        # 너무 작은 chunk 제거
        filtered: List[str] = []
        for c in chunks:
            if len(c) >= self.config.min_chunk_size:
                filtered.append(c)

        return filtered

    # ---------------------------------------------------------
    # document -> chunks
    # ---------------------------------------------------------

    def chunk_document(
        self,
        text: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> List[Chunk]:
        """
        문서를 chunk 리스트로 변환
        """

        if doc_id is None:
            doc_id = str(uuid.uuid4())

        metadata = metadata or {}

        pieces = self.split_text(text)

        chunks: List[Chunk] = []

        for i, piece in enumerate(pieces, start=1):

            chunk_id = f"{doc_id}:chunk{i:04d}"

            meta = dict(metadata)
            meta.update(
                {
                    "doc_id": doc_id,
                    "chunk_id": i,
                }
            )

            chunks.append(
                Chunk(
                    id=chunk_id,
                    document=piece,
                    metadata=meta,
                )
            )

        return chunks

    # ---------------------------------------------------------
    # helper
    # ---------------------------------------------------------

    def chunks_to_lists(
        self,
        chunks: List[Chunk],
    ) -> Tuple[List[str], List[str], List[Dict]]:
        """
        Chroma insert용 포맷 변환
        """

        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict] = []

        for c in chunks:
            ids.append(c.id)
            docs.append(c.document)
            metas.append(c.metadata)

        return ids, docs, metas


# ---------------------------------------------------------
# singleton helper
# ---------------------------------------------------------

_default_chunker: Optional[TextChunker] = None


def get_chunker() -> TextChunker:
    global _default_chunker

    if _default_chunker is None:
        _default_chunker = TextChunker()

    return _default_chunker


# ---------------------------------------------------------
# 편의 함수
# ---------------------------------------------------------

def split_text(text: str) -> List[str]:
    return get_chunker().split_text(text)


def chunk_document(
    text: str,
    doc_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> List[Chunk]:
    return get_chunker().chunk_document(
        text=text,
        doc_id=doc_id,
        metadata=metadata,
    )