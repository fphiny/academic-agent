from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from modules.log.repository import LogRepository
from modules.log.schemas import Conversation, LogEvent, LogMessage


class LogService:
    def __init__(self, repository: Optional[LogRepository] = None) -> None:
        self.repository = repository or LogRepository()

    # ------------------------------------------------------------------
    # validation / normalize
    # ------------------------------------------------------------------
    def _normalize_student_id(self, student_id: str) -> str:
        value = str(student_id or "").strip()
        if not value:
            raise ValueError("student_id is required")
        return value

    def _normalize_sid(self, sid: str) -> str:
        value = str(sid or "").strip()
        if not value:
            raise ValueError("sid is required")
        return value

    def _normalize_mode(self, mode: Optional[str]) -> str:
        value = str(mode or "").strip().lower()
        if not value:
            raise ValueError("mode is required")
        return value

    def _normalize_role(self, role: str) -> str:
        value = str(role or "").strip().lower()
        if value not in {"user", "assistant", "system", "tool"}:
            raise ValueError("role must be one of: user, assistant, system, tool")
        return value

    def _normalize_event_type(self, event_type: str) -> str:
        value = str(event_type or "").strip().lower()
        if not value:
            raise ValueError("event_type is required")
        return value

    def _normalize_model(self, model: Optional[str]) -> str:
        return str(model or "").strip()

    def _normalize_thinking(self, thinking: Optional[str]) -> str:
        return str(thinking or "").strip()

    def _normalize_message_id(self, message_id: Optional[int]) -> Optional[int]:
        if message_id is None:
            return None
        try:
            value = int(message_id)
        except Exception:
            raise ValueError("message_id must be an integer")
        if value <= 0:
            raise ValueError("message_id must be greater than 0")
        return value

    def _safe_metadata(self, metadata: Optional[dict]) -> dict:
        return metadata if isinstance(metadata, dict) else {}

    def _default_title_from_content(self, content: str, max_length: int = 40) -> str:
        text = str(content or "").strip()
        if not text:
            return ""
        one_line = " ".join(text.split())
        if len(one_line) <= max_length:
            return one_line
        return one_line[:max_length].rstrip() + "..."

    def _safe_created_at(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        return datetime.min

    def _safe_id(self, value: Any) -> int:
        try:
            return int(value)
        except Exception:
            return 0

    def _message_sort_key(self, item: LogMessage) -> Tuple[datetime, int]:
        return (
            self._safe_created_at(getattr(item, "created_at", None)),
            self._safe_id(getattr(item, "id", 0)),
        )

    def _event_sort_key(self, item: LogEvent) -> Tuple[datetime, int]:
        return (
            self._safe_created_at(getattr(item, "created_at", None)),
            self._safe_id(getattr(item, "id", 0)),
        )

    def _message_to_langchain(self, item: LogMessage) -> Optional[BaseMessage]:
        metadata = item.metadata or {}

        additional_kwargs: Dict[str, Any] = {}
        if item.mode:
            additional_kwargs["mode"] = item.mode
        if item.model:
            additional_kwargs["model"] = item.model
        if item.thinking:
            additional_kwargs["thinking"] = item.thinking

        if item.role == "user":
            return HumanMessage(content=item.content, additional_kwargs=additional_kwargs)

        if item.role == "assistant":
            return AIMessage(content=item.content, additional_kwargs=additional_kwargs)

        if item.role == "system":
            return SystemMessage(content=item.content, additional_kwargs=additional_kwargs)

        if item.role == "tool":
            tool_call_id = str(metadata.get("tool_call_id") or "")
            name = str(metadata.get("name") or metadata.get("tool_name") or "")
            artifact = metadata.get("artifact")

            if tool_call_id:
                return ToolMessage(
                    content=item.content,
                    tool_call_id=tool_call_id,
                    name=name or None,
                    artifact=artifact,
                    additional_kwargs=additional_kwargs,
                )

            return AIMessage(content=item.content, additional_kwargs=additional_kwargs)

        return None

    # ------------------------------------------------------------------
    # conversation
    # ------------------------------------------------------------------
    def get_conversation(
        self,
        student_id: str,
        sid: str,
    ) -> Optional[Conversation]:
        student_id = self._normalize_student_id(student_id)
        sid = self._normalize_sid(sid)

        return self.repository.get_conversation(
            student_id=student_id,
            sid=sid,
        )

    def get_or_create_conversation(
        self,
        student_id: str,
        sid: str,
        mode: str,
        title: str = "",
        metadata: Optional[dict] = None,
    ) -> Conversation:
        student_id = self._normalize_student_id(student_id)
        sid = self._normalize_sid(sid)
        normalized_mode = self._normalize_mode(mode)

        return self.repository.get_or_create_conversation(
            student_id=student_id,
            sid=sid,
            mode=normalized_mode,
            title=title.strip(),
            metadata=self._safe_metadata(metadata),
        )

    def ensure_conversation(
        self,
        student_id: str,
        sid: str,
        mode: str,
        title_from_message: str = "",
        metadata: Optional[dict] = None,
    ) -> Conversation:
        conversation = self.get_conversation(
            student_id=student_id,
            sid=sid,
        )
        if conversation is not None:
            return conversation

        title = self._default_title_from_content(title_from_message)
        return self.get_or_create_conversation(
            student_id=student_id,
            sid=sid,
            mode=mode,
            title=title,
            metadata=metadata,
        )

    def list_conversations(
        self,
        student_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Conversation]:
        student_id = self._normalize_student_id(student_id)

        return self.repository.list_conversations(
            student_id=student_id,
            mode=None,
            limit=limit,
            offset=offset,
        )

    def delete_conversation(
        self,
        student_id: str,
        sid: str,
    ) -> bool:
        student_id = self._normalize_student_id(student_id)
        sid = self._normalize_sid(sid)

        return self.repository.delete_conversation(
            student_id=student_id,
            sid=sid,
        )

    def delete_conversations(
        self,
        student_id: str,
    ) -> int:
        student_id = self._normalize_student_id(student_id)

        return self.repository.delete_conversations(
            student_id=student_id,
        )
    
    def update_conversation_title(
        self,
        student_id: str,
        sid: str,
        title: str,
    ) -> Optional[Conversation]:
        conversation = self.get_conversation(student_id=student_id, sid=sid)
        if conversation is None:
            return None

        return self.repository.update_conversation(
            conversation_id=conversation.id,
            title=title.strip(),
            touch=True,
        )

    def set_conversation_metadata(
        self,
        student_id: str,
        sid: str,
        metadata: dict,
    ) -> Optional[Conversation]:
        conversation = self.get_conversation(student_id=student_id, sid=sid)
        if conversation is None:
            return None

        return self.repository.update_conversation(
            conversation_id=conversation.id,
            metadata=self._safe_metadata(metadata),
            touch=True,
        )

    # ------------------------------------------------------------------
    # messages
    # ------------------------------------------------------------------
    def append_message(
        self,
        student_id: str,
        sid: str,
        mode: str,
        role: str,
        content: str,
        model: str = "",
        thinking: str = "",
        metadata: Optional[dict] = None,
    ) -> LogMessage:
        student_id = self._normalize_student_id(student_id)
        sid = self._normalize_sid(sid)
        normalized_mode = self._normalize_mode(mode)
        role = self._normalize_role(role)
        normalized_model = self._normalize_model(model)
        normalized_thinking = self._normalize_thinking(thinking)

        conversation = self.ensure_conversation(
            student_id=student_id,
            sid=sid,
            mode=normalized_mode,
            title_from_message=content if role == "user" else "",
        )

        if not conversation.title and role == "user":
            self.repository.update_conversation(
                conversation_id=conversation.id,
                title=self._default_title_from_content(content),
                touch=False,
            )

        return self.repository.append_message(
            conversation_id=conversation.id,
            role=role,
            content=str(content or ""),
            mode=normalized_mode,
            model=normalized_model,
            thinking=normalized_thinking,
            metadata=self._safe_metadata(metadata),
        )

    def append_user_message(
        self,
        student_id: str,
        sid: str,
        mode: str,
        content: str,
        model: str = "",
        thinking: str = "",
        metadata: Optional[dict] = None,
    ) -> LogMessage:
        return self.append_message(
            student_id=student_id,
            sid=sid,
            mode=mode,
            role="user",
            content=content,
            model=model,
            thinking=thinking,
            metadata=metadata,
        )

    def append_assistant_message(
        self,
        student_id: str,
        sid: str,
        mode: str,
        content: str,
        model: str = "",
        thinking: str = "",
        metadata: Optional[dict] = None,
    ) -> LogMessage:
        return self.append_message(
            student_id=student_id,
            sid=sid,
            mode=mode,
            role="assistant",
            content=content,
            model=model,
            thinking=thinking,
            metadata=metadata,
        )

    def append_system_message(
        self,
        student_id: str,
        sid: str,
        mode: str,
        content: str,
        model: str = "",
        thinking: str = "",
        metadata: Optional[dict] = None,
    ) -> LogMessage:
        return self.append_message(
            student_id=student_id,
            sid=sid,
            mode=mode,
            role="system",
            content=content,
            model=model,
            thinking=thinking,
            metadata=metadata,
        )

    def append_tool_message(
        self,
        student_id: str,
        sid: str,
        mode: str,
        content: str,
        model: str = "",
        thinking: str = "",
        metadata: Optional[dict] = None,
    ) -> LogMessage:
        return self.append_message(
            student_id=student_id,
            sid=sid,
            mode=mode,
            role="tool",
            content=content,
            model=model,
            thinking=thinking,
            metadata=metadata,
        )

    def list_messages(
        self,
        student_id: str,
        sid: str,
        limit: Optional[int] = None,
        offset: int = 0,
        ascending: bool = True,
    ) -> List[LogMessage]:
        conversation = self.get_conversation(student_id=student_id, sid=sid)
        if conversation is None:
            return []

        return self.repository.list_messages(
            conversation_id=conversation.id,
            limit=limit,
            offset=offset,
            ascending=ascending,
            mode=None,
        )

    def get_recent_messages(
        self,
        student_id: str,
        sid: str,
        limit: int = 20,
        roles: Optional[List[str]] = None,
    ) -> List[LogMessage]:
        conversation = self.get_conversation(student_id=student_id, sid=sid)
        if conversation is None:
            return []

        normalized_roles = None
        if roles is not None:
            normalized_roles = [self._normalize_role(role) for role in roles]

        return self.repository.get_recent_messages(
            conversation_id=conversation.id,
            limit=limit,
            roles=normalized_roles,
            mode=None,
        )

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------
    def append_event(
        self,
        student_id: str,
        sid: str,
        mode: str,
        event_type: str,
        content: str = "",
        step_index: Optional[int] = None,
        message_id: Optional[int] = None,
        model: str = "",
        thinking: str = "",
        metadata: Optional[dict] = None,
    ) -> LogEvent:
        student_id = self._normalize_student_id(student_id)
        sid = self._normalize_sid(sid)
        normalized_mode = self._normalize_mode(mode)
        normalized_event_type = self._normalize_event_type(event_type)
        normalized_message_id = self._normalize_message_id(message_id)
        normalized_model = self._normalize_model(model)
        normalized_thinking = self._normalize_thinking(thinking)

        conversation = self.ensure_conversation(
            student_id=student_id,
            sid=sid,
            mode=normalized_mode,
        )

        return self.repository.append_event(
            conversation_id=conversation.id,
            message_id=normalized_message_id,
            event_type=normalized_event_type,
            content=str(content or ""),
            step_index=step_index,
            mode=normalized_mode,
            model=normalized_model,
            thinking=normalized_thinking,
            metadata=self._safe_metadata(metadata),
        )

    def list_events(
        self,
        student_id: str,
        sid: str,
        limit: Optional[int] = None,
        offset: int = 0,
        ascending: bool = True,
        event_type: Optional[str] = None,
        message_id: Optional[int] = None,
    ) -> List[LogEvent]:
        conversation = self.get_conversation(student_id=student_id, sid=sid)
        if conversation is None:
            return []

        normalized_event_type = (
            self._normalize_event_type(event_type) if event_type is not None else None
        )
        normalized_message_id = self._normalize_message_id(message_id)

        return self.repository.list_events(
            conversation_id=conversation.id,
            limit=limit,
            offset=offset,
            ascending=ascending,
            event_type=normalized_event_type,
            mode=None,
            message_id=normalized_message_id,
        )

    def list_events_for_message(
        self,
        student_id: str,
        sid: str,
        message_id: int,
        limit: Optional[int] = None,
        offset: int = 0,
        ascending: bool = True,
        event_type: Optional[str] = None,
    ) -> List[LogEvent]:
        return self.list_events(
            student_id=student_id,
            sid=sid,
            limit=limit,
            offset=offset,
            ascending=ascending,
            event_type=event_type,
            message_id=message_id,
        )

    # ------------------------------------------------------------------
    # detail
    # ------------------------------------------------------------------
    def get_conversation_detail(
        self,
        student_id: str,
        sid: str,
    ) -> Optional[dict]:
        student_id = self._normalize_student_id(student_id)
        sid = self._normalize_sid(sid)

        detail = self.repository.get_conversation_detail(
            student_id=student_id,
            sid=sid,
            mode=None,
        )
        if detail is None:
            return None

        messages = detail.get("messages", [])
        events = detail.get("events", [])

        if isinstance(messages, list):
            messages.sort(key=self._message_sort_key)
        if isinstance(events, list):
            events.sort(key=self._event_sort_key)

        return {
            "conversation": detail.get("conversation"),
            "messages": messages,
            "events": events,
        }

    # ------------------------------------------------------------------
    # chat history transform
    # ------------------------------------------------------------------
    def build_langchain_history(
        self,
        student_id: str,
        sid: str,
        limit: int = 20,
        include_roles: Optional[List[str]] = None,
    ) -> List[BaseMessage]:
        messages = self.get_recent_messages(
            student_id=student_id,
            sid=sid,
            limit=limit,
            roles=include_roles,
        )

        history: List[BaseMessage] = []

        for item in messages:
            lc_msg = self._message_to_langchain(item)
            if lc_msg is not None:
                history.append(lc_msg)

        return history

    def build_openai_style_history(
        self,
        student_id: str,
        sid: str,
        limit: int = 20,
        include_roles: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        messages = self.get_recent_messages(
            student_id=student_id,
            sid=sid,
            limit=limit,
            roles=include_roles,
        )

        items: List[Dict[str, Any]] = []
        for item in messages:
            record: Dict[str, Any] = {
                "role": item.role,
                "content": item.content,
            }
            if item.mode:
                record["mode"] = item.mode
            if item.model:
                record["model"] = item.model
            if item.thinking:
                record["thinking"] = item.thinking
            if item.metadata:
                record["metadata"] = item.metadata
            items.append(record)

        return items

    def bind_events_to_message(
        self,
        student_id: str,
        sid: str,
        message_id: int,
        event_ids: List[int],
    ) -> int:
        student_id = self._normalize_student_id(student_id)
        sid = self._normalize_sid(sid)
        normalized_message_id = self._normalize_message_id(message_id)

        if not event_ids:
            return 0

        conversation = self.get_conversation(student_id=student_id, sid=sid)
        if conversation is None:
            return 0

        normalized_event_ids: List[int] = []
        for event_id in event_ids:
            try:
                value = int(event_id)
            except Exception:
                continue
            if value > 0:
                normalized_event_ids.append(value)

        if not normalized_event_ids:
            return 0

        return self.repository.bind_events_to_message(
            conversation_id=conversation.id,
            message_id=normalized_message_id,
            event_ids=normalized_event_ids,
        )

    

log_service = LogService()