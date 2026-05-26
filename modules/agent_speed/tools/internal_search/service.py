from __future__ import annotations

from ..base import ToolResult, normalize_string_list_input


def run(rag, query: str, collections, k: int = 5, retrieval_k: int = 30) -> ToolResult:
    try:
        collections = normalize_string_list_input(collections)
        collection = collections[0] if collections else ""

        if not collection:
            return ToolResult(
                name="internal_search",
                ok=False,
                data={"error": "collections is required"},
            )

        retrieved_chunks, chunks, debug_info = rag.search(
            query=query,
            collection_name=collection,
            retrieval_k=retrieval_k,
            final_k=k,
        )

        sources = rag.build_sources(chunks)
        context = rag.build_context(chunks)

        if not sources or not (context or "").strip():
            return ToolResult(
                name="internal_search",
                ok=False,
                data={
                    "query": query,
                    "collections": collections,
                    "sources": [],
                    "context": "",
                    "error": "no internal results",
                    "debug_info": debug_info,
                },
            )

        return ToolResult(
            name="internal_search",
            ok=True,
            data={
                "query": query,
                "collections": collections,
                "sources": sources,
                "context": context,
                "retrieved_count": len(retrieved_chunks),
                "final_count": len(chunks),
                "debug_info": debug_info,
            },
        )

    except Exception as e:
        return ToolResult(
            name="internal_search",
            ok=False,
            data={"error": str(e)},
        )