from __future__ import annotations

import json
from typing import Any, Callable, Dict, Generator, List, Tuple


def extract_sources_from_result(result_data: Any) -> List[Dict[str, Any]]:
    if not isinstance(result_data, dict):
        return []

    raw_sources = result_data.get("sources")
    if not isinstance(raw_sources, list):
        return []

    normalized_sources: List[Dict[str, Any]] = []
    for i, src in enumerate(raw_sources, start=1):
        if isinstance(src, dict):
            normalized_sources.append(
                {
                    "index": i,
                    "title": src.get("title") or src.get("source") or src.get("url") or "",
                    "url": src.get("url") or src.get("link") or "",
                    "score": src.get("score"),
                    "raw": src,
                }
            )
        else:
            normalized_sources.append(
                {
                    "index": i,
                    "title": str(src),
                    "url": "",
                    "score": None,
                    "raw": src,
                }
            )

    return normalized_sources


def execute_tool(
    *,
    tools: Any,
    tool_name: str,
    arguments: Dict[str, Any],
    step: int,
    messages: List[Dict[str, Any]],
    user_message: str,
) -> Generator[Dict[str, Any], None, Any]:
    yield {
        "type": "tool_call",
        "tool_name": tool_name,
        "arguments": arguments,
        "step": step,
    }

    result = tools.call_tool(tool_name, arguments)

    yield {
        "type": "tool_result",
        "tool_name": tool_name,
        "ok": result.ok,
        "result": result.data,
        "step": step,
    }

    sources = extract_sources_from_result(result.data)
    if sources:
        yield {
            "type": "sources",
            "tool_name": tool_name,
            "sources": sources,
            "step": step,
        }

    tool_payload = {
        "ok": result.ok,
        "data": result.data,
    }

    messages.append(
        {
            "role": "tool",
            "name": tool_name,
            "content": (
                f"{json.dumps(tool_payload, ensure_ascii=False)}\n\n"
                f"사용자질문: {user_message}"
            ),
        }
    )

    return result


def emit_build_context_debug(
    *,
    result_data: Dict[str, Any],
    step: int,
) -> Generator[Dict[str, Any], None, None]:
    context_text = str(result_data.get("context") or "").strip()
    sources = result_data.get("sources") if isinstance(result_data.get("sources"), list) else []
    selected_blocks = (
        result_data.get("selected_blocks")
        if isinstance(result_data.get("selected_blocks"), list)
        else []
    )

    context_sources = []
    for src in sources[:8]:
        if not isinstance(src, dict):
            continue
        context_sources.append(
            {
                "rank": src.get("rank"),
                "title": src.get("title", ""),
                "heading": src.get("heading", ""),
                "url": src.get("url", ""),
                "score": src.get("score"),
                "preview": str(src.get("preview") or "")[:500],
            }
        )

    selected_block_metas = []
    for block in selected_blocks[:20]:
        if not isinstance(block, dict):
            continue
        selected_block_metas.append(
            {
                "url": block.get("url", ""),
                "title": block.get("title", ""),
                "heading": block.get("heading", ""),
                "page": block.get("page"),
                "block_id": block.get("block_id", ""),
            }
        )

    yield {
        "type": "context",
        "step": step,
        "label": "final_build_context",
        "num_sources": len(sources),
        "num_selected_blocks": result_data.get("num_selected_blocks", 0),
        "num_documents_embedded": result_data.get("num_documents_embedded", 0),
        "context": context_text,
        "context_preview": context_text[:4000],
        "context_sources": context_sources,
        "selected_blocks": selected_block_metas,
    }

    preview_lines: List[str] = []
    preview_lines.append("[build-context-preview]")
    preview_lines.append(
        f"selected_blocks={result_data.get('num_selected_blocks', 0)} / "
        f"embedded_docs={result_data.get('num_documents_embedded', 0)} / "
        f"sources={len(sources)}"
    )
    if context_text:
        preview_lines.append(context_text)

    yield {
        "type": "thought",
        "step": step,
        "delta": "\n".join(preview_lines),
    }


def handle_build_context_result(
    *,
    result_data: Dict[str, Any],
    user_message: str,
    step: int,
    messages: List[Dict[str, Any]],
    external_build_context_retry_used: bool,
    judge_external_build_context_result: Callable[..., Dict[str, Any]],
    generate_grounded_final_answer_stream: Callable[..., Generator[str, None, None]],
) -> Generator[Dict[str, Any], None, Tuple[bool, bool]]:
    yield from emit_build_context_debug(
        result_data=result_data,
        step=step,
    )

    try:
        build_judged = judge_external_build_context_result(
            user_query=user_message,
            result_data=result_data,
        )
    except Exception as e:
        build_judged = {
            "decision": "refine",
            "reason": f"build_context judge failed: {str(e)}",
            "rewrite_query": "",
        }

    yield {
        "type": "thought",
        "step": step,
        "delta": (
            f"[build-context-judge] "
            f"decision={build_judged['decision']} / reason={build_judged['reason']}"
        ),
    }

    if build_judged["decision"] in {"answer", "refine"}:
        emitted_any = False

        try:
            for chunk in generate_grounded_final_answer_stream(
                user_message=user_message,
                result_data=result_data,
                mode=build_judged["decision"],
            ):
                if not chunk:
                    continue
                emitted_any = True
                yield {
                    "type": "delta",
                    "delta": str(chunk),
                    "step": step,
                }
        except Exception as e:
            yield {
                "type": "error",
                "error": f"final grounded answer failed: {str(e)}",
            }
            return True, external_build_context_retry_used

        if emitted_any:
            yield {"type": "done"}
            return True, external_build_context_retry_used

        fallback_context = str(result_data.get("context") or "").strip()
        if fallback_context:
            yield {
                "type": "delta",
                "delta": fallback_context[:1500],
                "step": step,
            }
            yield {"type": "done"}
            return True, external_build_context_retry_used

        yield {"type": "error", "error": "empty grounded final answer"}
        return True, external_build_context_retry_used

    if (
        build_judged["decision"] == "search_again"
        and not external_build_context_retry_used
    ):
        rewrite_query = str(build_judged.get("rewrite_query") or "").strip()
        if rewrite_query:
            external_build_context_retry_used = True
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "[external_build_context_followup_instruction]\n"
                        f"다음에는 external_search를 query={rewrite_query!r} 로 재시도하라.\n"
                        "사용자 의도를 바꾸지 마라.\n\n"
                        f"사용자질문: {user_message}"
                    ),
                }
            )

    return False, external_build_context_retry_used