from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Any, List, Optional

from modules.log.schemas import Conversation, LogEvent, LogMessage


DEFAULT_DB_PATH = os.getenv("LOG_DB_PATH", "data/logs.sqlite3")

KST = timezone(timedelta(hours=9))


class LogRepository:
    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        os.makedirs(db_dir, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # time helpers
    # ------------------------------------------------------------------
    def _now_kst_str(self) -> str:
        """
        Store timestamps explicitly as KST with timezone offset.

        Example:
            2026-05-21T17:42:03+09:00
        """
        return datetime.now(KST).isoformat(timespec="seconds")

    # ------------------------------------------------------------------
    # db helpers
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _table_exists(self, conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            LIMIT 1
            """,
            (table_name,),
        ).fetchone()
        return row is not None

    def _column_exists(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
    ) -> bool:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return any(row["name"] == column_name for row in rows)

    def _index_exists(self, conn: sqlite3.Connection, index_name: str) -> bool:
        row = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'index' AND name = ?
            LIMIT 1
            """,
            (index_name,),
        ).fetchone()
        return row is not None

    def _ensure_messages_columns(self, conn: sqlite3.Connection) -> None:
        if not self._column_exists(conn, "messages", "mode"):
            conn.execute("ALTER TABLE messages ADD COLUMN mode TEXT NOT NULL DEFAULT ''")
        if not self._column_exists(conn, "messages", "model"):
            conn.execute("ALTER TABLE messages ADD COLUMN model TEXT NOT NULL DEFAULT ''")
        if not self._column_exists(conn, "messages", "thinking"):
            conn.execute("ALTER TABLE messages ADD COLUMN thinking TEXT NOT NULL DEFAULT ''")

    def _ensure_events_columns(self, conn: sqlite3.Connection) -> None:
        if not self._column_exists(conn, "events", "message_id"):
            conn.execute("ALTER TABLE events ADD COLUMN message_id INTEGER NULL")
        if not self._column_exists(conn, "events", "mode"):
            conn.execute("ALTER TABLE events ADD COLUMN mode TEXT NOT NULL DEFAULT ''")
        if not self._column_exists(conn, "events", "model"):
            conn.execute("ALTER TABLE events ADD COLUMN model TEXT NOT NULL DEFAULT ''")
        if not self._column_exists(conn, "events", "thinking"):
            conn.execute("ALTER TABLE events ADD COLUMN thinking TEXT NOT NULL DEFAULT ''")

    def _rebuild_conversations_table(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                sid TEXT NOT NULL,
                mode TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S+09:00', 'now', '+9 hours')),
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S+09:00', 'now', '+9 hours')),
                UNIQUE(student_id, sid)
            );
            """
        )

        old_rows = conn.execute(
            """
            SELECT *
            FROM conversations
            ORDER BY student_id ASC, sid ASC, created_at ASC, id ASC
            """
        ).fetchall()

        grouped: dict[tuple[str, str], dict[str, Any]] = {}

        for row in old_rows:
            key = (row["student_id"], row["sid"])
            if key not in grouped:
                grouped[key] = {
                    "student_id": row["student_id"],
                    "sid": row["sid"],
                    "mode": row["mode"] or "",
                    "title": row["title"] or "",
                    "metadata_json": row["metadata_json"] or "{}",
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "old_ids": [row["id"]],
                }
                continue

            item = grouped[key]
            item["old_ids"].append(row["id"])

            old_updated = str(item["updated_at"] or "")
            new_updated = str(row["updated_at"] or "")
            if new_updated >= old_updated:
                item["updated_at"] = row["updated_at"]
                item["mode"] = row["mode"] or item["mode"]
                if not item["title"] and row["title"]:
                    item["title"] = row["title"]
                if item["metadata_json"] in ("", "{}", None) and row["metadata_json"]:
                    item["metadata_json"] = row["metadata_json"]

        old_to_new_id: dict[int, int] = {}

        for item in grouped.values():
            cur = conn.execute(
                """
                INSERT INTO conversations_new (
                    student_id, sid, mode, title, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["student_id"],
                    item["sid"],
                    item["mode"] or "",
                    item["title"] or "",
                    item["metadata_json"] or "{}",
                    item["created_at"],
                    item["updated_at"],
                ),
            )
            new_id = cur.lastrowid
            for old_id in item["old_ids"]:
                old_to_new_id[old_id] = new_id

        for old_id, new_id in old_to_new_id.items():
            conn.execute(
                "UPDATE messages SET conversation_id = ? WHERE conversation_id = ?",
                (new_id, old_id),
            )
            conn.execute(
                "UPDATE events SET conversation_id = ? WHERE conversation_id = ?",
                (new_id, old_id),
            )

        conn.execute("DROP TABLE conversations")
        conn.execute("ALTER TABLE conversations_new RENAME TO conversations")

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    sid TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S+09:00', 'now', '+9 hours')),
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S+09:00', 'now', '+9 hours')),
                    UNIQUE(student_id, sid)
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    mode TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    thinking TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S+09:00', 'now', '+9 hours')),
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    message_id INTEGER NULL,
                    event_type TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    step_index INTEGER NULL,
                    mode TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    thinking TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S+09:00', 'now', '+9 hours')),
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                    FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE
                );
                """
            )

            self._ensure_messages_columns(conn)
            self._ensure_events_columns(conn)

            recreate_conversations = False

            if self._table_exists(conn, "conversations"):
                indexes = conn.execute("PRAGMA index_list(conversations)").fetchall()
                unique_indexes = [row for row in indexes if int(row["unique"]) == 1]

                has_target_unique = False
                for index_row in unique_indexes:
                    idx_name = index_row["name"]
                    cols = conn.execute(f"PRAGMA index_info({idx_name})").fetchall()
                    col_names = [col["name"] for col in cols]
                    if col_names == ["student_id", "sid"]:
                        has_target_unique = True
                        break

                if not has_target_unique:
                    recreate_conversations = True

            if recreate_conversations:
                self._rebuild_conversations_table(conn)

            if not self._index_exists(conn, "idx_conversations_student_updated"):
                conn.execute(
                    """
                    CREATE INDEX idx_conversations_student_updated
                    ON conversations(student_id, updated_at DESC)
                    """
                )

            if not self._index_exists(conn, "idx_conversations_student_sid"):
                conn.execute(
                    """
                    CREATE INDEX idx_conversations_student_sid
                    ON conversations(student_id, sid)
                    """
                )

            if not self._index_exists(conn, "idx_messages_conversation_created"):
                conn.execute(
                    """
                    CREATE INDEX idx_messages_conversation_created
                    ON messages(conversation_id, created_at ASC, id ASC)
                    """
                )

            if not self._index_exists(conn, "idx_events_conversation_created"):
                conn.execute(
                    """
                    CREATE INDEX idx_events_conversation_created
                    ON events(conversation_id, created_at ASC, id ASC)
                    """
                )

            if not self._index_exists(conn, "idx_events_conversation_type"):
                conn.execute(
                    """
                    CREATE INDEX idx_events_conversation_type
                    ON events(conversation_id, event_type, created_at ASC, id ASC)
                    """
                )

            if not self._index_exists(conn, "idx_events_message_created"):
                conn.execute(
                    """
                    CREATE INDEX idx_events_message_created
                    ON events(message_id, created_at ASC, id ASC)
                    """
                )

    def _json_dump(self, value: Optional[dict]) -> str:
        if not isinstance(value, dict):
            value = {}
        return json.dumps(value, ensure_ascii=False)

    def _json_load(self, value: Any) -> dict:
        if not value:
            return {}
        if isinstance(value, dict):
            return value
        try:
            return json.loads(value)
        except Exception:
            return {}

    def _parse_dt(self, value: Any) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, datetime):
            return value

        text = str(value).strip()

        try:
            return datetime.fromisoformat(text)
        except Exception:
            pass

        try:
            return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

    def _row_to_conversation(self, row: sqlite3.Row) -> Conversation:
        return Conversation(
            id=row["id"],
            student_id=row["student_id"],
            sid=row["sid"],
            mode=row["mode"] or "",
            title=row["title"] or "",
            metadata=self._json_load(row["metadata_json"]),
            created_at=self._parse_dt(row["created_at"]),
            updated_at=self._parse_dt(row["updated_at"]),
        )

    def _row_to_message(self, row: sqlite3.Row) -> LogMessage:
        return LogMessage(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=row["content"] or "",
            mode=(row["mode"] if "mode" in row.keys() else "") or "",
            model=(row["model"] if "model" in row.keys() else "") or "",
            thinking=(row["thinking"] if "thinking" in row.keys() else "") or "",
            metadata=self._json_load(row["metadata_json"]),
            created_at=self._parse_dt(row["created_at"]),
        )

    def _row_to_event(self, row: sqlite3.Row) -> LogEvent:
        return LogEvent(
            id=row["id"],
            conversation_id=row["conversation_id"],
            message_id=(row["message_id"] if "message_id" in row.keys() else None),
            event_type=row["event_type"],
            content=row["content"] or "",
            step_index=row["step_index"],
            mode=(row["mode"] if "mode" in row.keys() else "") or "",
            model=(row["model"] if "model" in row.keys() else "") or "",
            thinking=(row["thinking"] if "thinking" in row.keys() else "") or "",
            metadata=self._json_load(row["metadata_json"]),
            created_at=self._parse_dt(row["created_at"]),
        )

    # ------------------------------------------------------------------
    # conversation
    # ------------------------------------------------------------------
    def get_conversation(
        self,
        student_id: str,
        sid: str,
    ) -> Optional[Conversation]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM conversations
                WHERE student_id = ? AND sid = ?
                LIMIT 1
                """,
                (student_id, sid),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_conversation(row)

    def get_or_create_conversation(
        self,
        student_id: str,
        sid: str,
        mode: str,
        title: str = "",
        metadata: Optional[dict] = None,
    ) -> Conversation:
        existing = self.get_conversation(student_id=student_id, sid=sid)
        if existing is not None:
            return existing

        now = self._now_kst_str()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO conversations (
                    student_id, sid, mode, title, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    student_id,
                    sid,
                    mode or "",
                    title or "",
                    self._json_dump(metadata),
                    now,
                    now,
                ),
            )

            row = conn.execute(
                """
                SELECT *
                FROM conversations
                WHERE student_id = ? AND sid = ?
                LIMIT 1
                """,
                (student_id, sid),
            ).fetchone()

        if row is None:
            raise RuntimeError("failed to create conversation")

        return self._row_to_conversation(row)

    def update_conversation(
        self,
        conversation_id: int,
        title: Optional[str] = None,
        metadata: Optional[dict] = None,
        touch: bool = True,
    ) -> Optional[Conversation]:
        fields: List[str] = []
        params: List[Any] = []

        if title is not None:
            fields.append("title = ?")
            params.append(title)

        if metadata is not None:
            fields.append("metadata_json = ?")
            params.append(self._json_dump(metadata))

        if touch:
            fields.append("updated_at = ?")
            params.append(self._now_kst_str())

        if not fields:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM conversations WHERE id = ? LIMIT 1",
                    (conversation_id,),
                ).fetchone()
            return self._row_to_conversation(row) if row else None

        params.append(conversation_id)

        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE conversations
                SET {", ".join(fields)}
                WHERE id = ?
                """,
                params,
            )
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ? LIMIT 1",
                (conversation_id,),
            ).fetchone()

        return self._row_to_conversation(row) if row else None

    def touch_conversation(
        self,
        conversation_id: int,
        mode: Optional[str] = None,
    ) -> None:
        now = self._now_kst_str()

        with self._connect() as conn:
            if mode is not None:
                conn.execute(
                    """
                    UPDATE conversations
                    SET updated_at = ?,
                        mode = ?
                    WHERE id = ?
                    """,
                    (now, mode, conversation_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE conversations
                    SET updated_at = ?
                    WHERE id = ?
                    """,
                    (now, conversation_id),
                )

    def list_conversations(
        self,
        student_id: str,
        mode: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Conversation]:
        query = """
            SELECT *
            FROM conversations
            WHERE student_id = ?
        """
        params: List[Any] = [student_id]

        if mode:
            query += " AND mode = ?"
            params.append(mode)

        query += " ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_conversation(row) for row in rows]

    def delete_conversation(
        self,
        student_id: str,
        sid: str,
    ) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                """
                DELETE FROM conversations
                WHERE student_id = ? AND sid = ?
                """,
                (student_id, sid),
            )
            return cur.rowcount > 0

    def delete_conversations(
        self,
        student_id: str,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                DELETE FROM conversations
                WHERE student_id = ?
                """,
                (student_id,),
            )
            return cur.rowcount
        
    # ------------------------------------------------------------------
    # messages
    # ------------------------------------------------------------------
    def append_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        mode: str = "",
        model: str = "",
        thinking: str = "",
        metadata: Optional[dict] = None,
    ) -> LogMessage:
        now = self._now_kst_str()

        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO messages (
                    conversation_id, role, content, mode, model, thinking, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    role,
                    content,
                    mode or "",
                    model or "",
                    thinking or "",
                    self._json_dump(metadata),
                    now,
                ),
            )
            message_id = cur.lastrowid

            conn.execute(
                """
                UPDATE conversations
                SET updated_at = ?,
                    mode = ?
                WHERE id = ?
                """,
                (now, mode or "", conversation_id),
            )

            row = conn.execute(
                """
                SELECT *
                FROM messages
                WHERE id = ?
                LIMIT 1
                """,
                (message_id,),
            ).fetchone()

        if row is None:
            raise RuntimeError("failed to append message")

        return self._row_to_message(row)

    def list_messages(
        self,
        conversation_id: int,
        limit: Optional[int] = None,
        offset: int = 0,
        ascending: bool = True,
        mode: Optional[str] = None,
    ) -> List[LogMessage]:
        order = "ASC" if ascending else "DESC"
        query = """
            SELECT *
            FROM messages
            WHERE conversation_id = ?
        """
        params: List[Any] = [conversation_id]

        if mode:
            query += " AND mode = ?"
            params.append(mode)

        query += f" ORDER BY created_at {order}, id {order}"

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_message(row) for row in rows]

    def get_recent_messages(
        self,
        conversation_id: int,
        limit: int = 20,
        roles: Optional[List[str]] = None,
        mode: Optional[str] = None,
    ) -> List[LogMessage]:
        query = """
            SELECT *
            FROM messages
            WHERE conversation_id = ?
        """
        params: List[Any] = [conversation_id]

        if roles:
            placeholders = ",".join(["?"] * len(roles))
            query += f" AND role IN ({placeholders})"
            params.extend(roles)

        if mode:
            query += " AND mode = ?"
            params.append(mode)

        query += """
            ORDER BY created_at DESC, id DESC
            LIMIT ?
        """
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        items = [self._row_to_message(row) for row in rows]
        items.reverse()
        return items

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------
    def append_event(
        self,
        conversation_id: int,
        event_type: str,
        content: str = "",
        step_index: Optional[int] = None,
        message_id: Optional[int] = None,
        mode: str = "",
        model: str = "",
        thinking: str = "",
        metadata: Optional[dict] = None,
    ) -> LogEvent:
        now = self._now_kst_str()

        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO events (
                    conversation_id, message_id, event_type, content, step_index,
                    mode, model, thinking, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    message_id,
                    event_type,
                    content,
                    step_index,
                    mode or "",
                    model or "",
                    thinking or "",
                    self._json_dump(metadata),
                    now,
                ),
            )
            event_id = cur.lastrowid

            conn.execute(
                """
                UPDATE conversations
                SET updated_at = ?,
                    mode = ?
                WHERE id = ?
                """,
                (now, mode or "", conversation_id),
            )

            row = conn.execute(
                """
                SELECT *
                FROM events
                WHERE id = ?
                LIMIT 1
                """,
                (event_id,),
            ).fetchone()

        if row is None:
            raise RuntimeError("failed to append event")

        return self._row_to_event(row)

    def list_events(
        self,
        conversation_id: int,
        limit: Optional[int] = None,
        offset: int = 0,
        ascending: bool = True,
        event_type: Optional[str] = None,
        mode: Optional[str] = None,
        message_id: Optional[int] = None,
    ) -> List[LogEvent]:
        order = "ASC" if ascending else "DESC"
        query = """
            SELECT *
            FROM events
            WHERE conversation_id = ?
        """
        params: List[Any] = [conversation_id]

        if event_type is not None:
            query += " AND event_type = ?"
            params.append(event_type)

        if mode:
            query += " AND mode = ?"
            params.append(mode)

        if message_id is not None:
            query += " AND message_id = ?"
            params.append(message_id)

        query += f" ORDER BY created_at {order}, id {order}"

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_event(row) for row in rows]

    # ------------------------------------------------------------------
    # detail
    # ------------------------------------------------------------------
    def get_conversation_detail(
        self,
        student_id: str,
        sid: str,
        mode: Optional[str] = None,
    ) -> Optional[dict]:
        conversation = self.get_conversation(student_id=student_id, sid=sid)
        if conversation is None:
            return None

        messages = self.list_messages(
            conversation_id=conversation.id,
            ascending=True,
            mode=mode,
        )
        events = self.list_events(
            conversation_id=conversation.id,
            ascending=True,
            mode=mode,
        )

        return {
            "conversation": conversation,
            "messages": messages,
            "events": events,
        }

    def bind_events_to_message(
        self,
        conversation_id: int,
        message_id: int,
        event_ids: List[int],
    ) -> int:
        if not event_ids:
            return 0

        placeholders = ",".join(["?"] * len(event_ids))
        params: List[Any] = [message_id, conversation_id, *event_ids]

        with self._connect() as conn:
            cur = conn.execute(
                f"""
                UPDATE events
                SET message_id = ?
                WHERE conversation_id = ?
                  AND id IN ({placeholders})
                """,
                params,
            )
            return cur.rowcount