# chroma.py / store.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import chromadb
from chromadb.api.models.Collection import Collection

from .embeddings import embed_documents, embed_query
from .rerank import rerank_rows


AliasResolver = Callable[[str], str]


@dataclass
class ChromaConfig:
    persist_directory: str = "./data/chroma"


class ChromaStore:
    """
    Chroma 벡터 저장소 래퍼

    기능:
    - client 초기화
    - 컬렉션 CRUD
    - 문서 CRUD
    - 검색
    - alias resolve 지원

    문서 저장 권장 형식:
        id = "doc123:chunk0001"
        metadata = {
            "doc_id": "doc123",
            "chunk_id": 1,
            "title": "...",
            "source": "...",
            "category": "...",
            "created_at": "...",
            "updated_at": "...",
        }
    """

    def __init__(
        self,
        config: Optional[ChromaConfig] = None,
        alias_resolver: Optional[AliasResolver] = None,
    ):
        self.config = config or ChromaConfig()
        self.alias_resolver = alias_resolver

        self.client = chromadb.PersistentClient(
            path=self.config.persist_directory
        )

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _resolve_collection_name(self, name: str) -> str:
        if not name:
            raise ValueError("collection name is required")

        if self.alias_resolver:
            resolved = self.alias_resolver(name)
            return resolved or name
        return name

    def _get_collection_obj(self, name: str) -> Collection:
        resolved_name = self._resolve_collection_name(name)
        return self.client.get_collection(name=resolved_name)

    def _validate_lengths(
        self,
        ids: Sequence[str],
        documents: Optional[Sequence[str]] = None,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        n = len(ids)

        if documents is not None and len(documents) != n:
            raise ValueError("len(documents) must match len(ids)")

        if metadatas is not None and len(metadatas) != n:
            raise ValueError("len(metadatas) must match len(ids)")

    def _normalize_row(
        self,
        row: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "id": row.get("id"),
            "document": row.get("document"),
            "metadata": row.get("metadata"),
            "distance": row.get("distance"),
            "rerank_score": row.get("rerank_score"),
        }

    # ------------------------------------------------------------------
    # collection CRUD
    # ------------------------------------------------------------------

    def create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        get_or_create: bool = True,
    ) -> Collection:
        """
        컬렉션 생성
        """
        resolved_name = self._resolve_collection_name(name)
        return self.client.get_or_create_collection(
            name=resolved_name,
            metadata=metadata,
        ) if get_or_create else self.client.create_collection(
            name=resolved_name,
            metadata=metadata,
        )

    def update_collection_metadata(
        self,
        name: str,
        metadata: Dict[str, Any],
    ) -> Collection:
        """
        컬렉션 메타데이터 수정

        주의:
        - Chroma의 collection.modify(metadata=...) 동작을 그대로 사용
        - 전달한 metadata로 컬렉션 metadata 전체가 갱신됨
        """
        if metadata is None:
            raise ValueError("metadata is required")
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dict")

        col = self._get_collection_obj(name)
        col.modify(metadata=metadata)
        return col

    def get_collection(self, name: str) -> Collection:
        """
        컬렉션 조회
        """
        return self._get_collection_obj(name)

    def list_collections(self) -> List[str]:
        """
        컬렉션 목록
        """
        collections = self.client.list_collections()
        names: List[str] = []

        for c in collections:
            # Chroma 버전별 반환 형태 차이 방어
            if hasattr(c, "name"):
                names.append(c.name)
            else:
                names.append(str(c))

        return names

    def delete_collection(self, name: str) -> None:
        """
        컬렉션 삭제
        """
        resolved_name = self._resolve_collection_name(name)
        self.client.delete_collection(name=resolved_name)

    def rename_collection(self, name: str, new_name: str) -> Collection:
        """
        컬렉션 이름 변경
        """
        col = self._get_collection_obj(name)
        col.modify(name=new_name)
        return col

    def count_documents(self, collection_name: str) -> int:
        """
        컬렉션 내 문서 수
        """
        col = self._get_collection_obj(collection_name)
        return col.count()

    # ------------------------------------------------------------------
    # document CRUD
    # ------------------------------------------------------------------

    def upsert_documents(
        self,
        collection_name: str,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """
        문서 추가/수정 (재색인 포함)
        """
        self._validate_lengths(ids=ids, documents=documents, metadatas=metadatas)

        if not ids:
            return

        col = self._get_collection_obj(collection_name)
        embeddings = embed_documents(documents)

        col.upsert(
            ids=list(ids),
            documents=list(documents),
            metadatas=list(metadatas) if metadatas is not None else None,
            embeddings=embeddings,
        )

    def add_documents(
        self,
        collection_name: str,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """
        문서 추가
        이미 존재하는 ID면 에러 날 수 있으므로 보통 upsert_documents 권장
        """
        self._validate_lengths(ids=ids, documents=documents, metadatas=metadatas)

        if not ids:
            return

        col = self._get_collection_obj(collection_name)
        embeddings = embed_documents(documents)

        col.add(
            ids=list(ids),
            documents=list(documents),
            metadatas=list(metadatas) if metadatas is not None else None,
            embeddings=embeddings,
        )

    def get_documents(
        self,
        collection_name: str,
        ids: Optional[Sequence[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        문서 조회
        include 예:
            ["documents", "metadatas", "embeddings"]
        """
        col = self._get_collection_obj(collection_name)

        kwargs: Dict[str, Any] = {}
        if ids is not None:
            kwargs["ids"] = list(ids)
        if where is not None:
            kwargs["where"] = where
        if limit is not None:
            kwargs["limit"] = limit
        if offset is not None:
            kwargs["offset"] = offset
        if include is not None:
            kwargs["include"] = include

        return col.get(**kwargs)

    def update_documents(
        self,
        collection_name: str,
        ids: Sequence[str],
        documents: Optional[Sequence[str]] = None,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """
        문서 수정

        주의:
        - documents가 바뀌면 재임베딩 필요
        - metadatas만 바꿀 수도 있음
        """
        if not ids:
            return

        self._validate_lengths(
            ids=ids,
            documents=documents if documents is not None else None,
            metadatas=metadatas if metadatas is not None else None,
        )

        col = self._get_collection_obj(collection_name)

        kwargs: Dict[str, Any] = {
            "ids": list(ids),
        }

        if documents is not None:
            kwargs["documents"] = list(documents)
            kwargs["embeddings"] = embed_documents(documents)

        if metadatas is not None:
            kwargs["metadatas"] = list(metadatas)

        col.update(**kwargs)

    def delete_documents(
        self,
        collection_name: str,
        ids: Optional[Sequence[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        문서 삭제
        - ids 기준 삭제
        - where 기준 삭제 가능
        """
        if ids is None and where is None:
            raise ValueError("either ids or where must be provided")

        col = self._get_collection_obj(collection_name)

        kwargs: Dict[str, Any] = {}
        if ids is not None:
            kwargs["ids"] = list(ids)
        if where is not None:
            kwargs["where"] = where

        col.delete(**kwargs)

    def delete_document_by_doc_id(
        self,
        collection_name: str,
        doc_id: str,
    ) -> None:
        """
        같은 doc_id를 가진 chunk 전부 삭제
        """
        self.delete_documents(
            collection_name=collection_name,
            where={"doc_id": doc_id},
        )

    # ------------------------------------------------------------------
    # query / search
    # ------------------------------------------------------------------

    def query_documents(
        self,
        collection_name: str,
        query_texts: Optional[Sequence[str]] = None,
        query_embeddings: Optional[Sequence[Sequence[float]]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        벡터 검색

        일반적으로는 query_texts만 넣으면 내부에서 embed_query 수행
        """
        if not query_texts and not query_embeddings:
            raise ValueError("either query_texts or query_embeddings must be provided")

        col = self._get_collection_obj(collection_name)

        if include is None:
            include = ["documents", "metadatas", "distances"]

        kwargs: Dict[str, Any] = {
            "n_results": n_results,
            "include": include,
        }

        if where is not None:
            kwargs["where"] = where

        if query_embeddings is not None:
            kwargs["query_embeddings"] = [list(vec) for vec in query_embeddings]
        else:
            embedded_queries = [embed_query(q) for q in query_texts or []]
            kwargs["query_embeddings"] = embedded_queries

        return col.query(**kwargs)

    def similarity_search(
        self,
        collection_name: str,
        query: str,
        k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        *,
        rerank: bool = False,
        retrieval_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        query 1개에 대한 검색 결과를 보기 쉽게 정리해서 반환

        호환성:
        - 기존 호출: similarity_search(..., k=5) 그대로 동작
        - 확장 호출: rerank=True 주면 1차 후보를 더 가져와서 리랭킹 후 상위 k개 반환

        파라미터:
        - k: 최종 반환 개수
        - rerank: True면 리랭킹 수행
        - retrieval_k: 리랭킹 전 1차 후보 개수. None이면 max(k, 20)
        """
        if not query or not str(query).strip():
            return []

        final_k = max(int(k), 1)

        if rerank:
            candidate_k = retrieval_k if retrieval_k is not None else max(final_k, 20)
            candidate_k = max(int(candidate_k), final_k)
        else:
            candidate_k = final_k

        result = self.query_documents(
            collection_name=collection_name,
            query_texts=[query],
            n_results=candidate_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        rows: List[Dict[str, Any]] = []

        for i in range(len(ids)):
            rows.append(
                {
                    "id": ids[i],
                    "document": docs[i] if i < len(docs) else None,
                    "metadata": metas[i] if i < len(metas) else None,
                    "distance": dists[i] if i < len(dists) else None,
                    "rerank_score": None,
                }
            )

        if not rows:
            return []

        if rerank:
            rows = rerank_rows(
                query=query,
                rows=rows,
                text_key="document",
                top_k=final_k,
            )
        else:
            rows = rows[:final_k]

        return [self._normalize_row(row) for row in rows]


# ----------------------------------------------------------------------
# 편하게 바로 쓰는 전역 인스턴스
# ----------------------------------------------------------------------

_default_store: Optional[ChromaStore] = None


def get_store(
    persist_directory: str = "./data/chroma",
    alias_resolver: Optional[AliasResolver] = None,
) -> ChromaStore:
    global _default_store

    if _default_store is None:
        _default_store = ChromaStore(
            config=ChromaConfig(persist_directory=persist_directory),
            alias_resolver=alias_resolver,
        )

    return _default_store