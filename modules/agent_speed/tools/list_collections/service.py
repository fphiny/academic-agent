from __future__ import annotations

from ..base import ToolResult


def normalize_collection_record(raw):
    metadata = raw.get("metadata", {}) if isinstance(raw, dict) else {}

    return {
        "name": raw.get("name", "") if isinstance(raw, dict) else "",
        "engine": (
            raw.get("engine")
            or raw.get("distance")
            or metadata.get("engine")
            or ""
        ),
        "domain": metadata.get(
            "domain",
            raw.get("domain", "") if isinstance(raw, dict) else "",
        ),
        "description": metadata.get(
            "description",
            raw.get("description", "") if isinstance(raw, dict) else "",
        ),
        "metadata": metadata,
    }


def run(rag, debug_print) -> ToolResult:
    try:
        if not hasattr(rag, "list_collections"):
            return ToolResult(
                name="list_collections",
                ok=False,
                data={"error": "RAGService.list_collections is not available"},
            )

        raw = rag.list_collections()
        if not isinstance(raw, list):
            return ToolResult(
                name="list_collections",
                ok=False,
                data={"error": "invalid list_collections response"},
            )

        collections = [normalize_collection_record(x) for x in raw]

        debug_print("LIST_COLLECTIONS RAW", raw, max_len=12000)
        debug_print("LIST_COLLECTIONS NORMALIZED", collections, max_len=12000)

        return ToolResult(
            name="list_collections",
            ok=True,
            data={"collections": collections},
        )

    except Exception as e:
        return ToolResult(
            name="list_collections",
            ok=False,
            data={"error": str(e)},
        )