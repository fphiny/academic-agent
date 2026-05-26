from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class Conversation:
    id: int
    student_id: str
    sid: str
    mode: str
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class LogMessage:
    id: int
    conversation_id: int
    role: str
    content: str
    mode: str = ""
    model: str = ""
    thinking: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None


@dataclass
class LogEvent:
    id: int
    conversation_id: int
    message_id: Optional[int] = None
    event_type: str = ""
    content: str = ""
    step_index: Optional[int] = None
    mode: str = ""
    model: str = ""
    thinking: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None