from __future__ import annotations

from ..base import ToolResult, normalize_string_list_input
from ..internal_search import run as internal_search_run


def run(rag, query: str, collections, k: int = 5) -> ToolResult:
    try:
        collections = normalize_string_list_input(collections)

        merged_sources = []
        context_parts = []
        searched_collections = []

        for collection in collections:
            one = internal_search_run(
                rag=rag,
                query=query,
                collections=[collection],
                k=k,
            )

            searched_collections.append(
                {
                    "collection": collection,
                    "ok": one.ok,
                    "error": one.data.get("error") if isinstance(one.data, dict) else None,
                }
            )

            if not one.ok:
                continue

            one_sources = one.data.get("sources", []) or []
            one_context = one.data.get("context", "") or ""

            for src in one_sources:
                if isinstance(src, dict):
                    enriched = dict(src)
                    enriched["collection"] = collection
                    merged_sources.append(enriched)
                else:
                    merged_sources.append(
                        {
                            "source": str(src),
                            "collection": collection,
                        }
                    )

            if one_context:
                context_parts.append(f"[{collection}]\n{one_context}")

        if not merged_sources and not context_parts:
            return ToolResult(
                name="internal_multi_search",
                ok=False,
                data={
                    "query": query,
                    "collections": collections,
                    "searched_collections": searched_collections,
                    "error": "no results from selected collections",
                },
            )

        context = "\n\n".join(context_parts)

        return ToolResult(
            name="internal_multi_search",
            ok=True,
            data={
                "query": query,
                "collections": collections,
                "searched_collections": searched_collections,
                "sources": merged_sources,
                "context": context,
            },
        )

    except Exception as e:
        return ToolResult(
            name="internal_multi_search",
            ok=False,
            data={"error": str(e)},
        )
