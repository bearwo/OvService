from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from config import DB_PATH, DB_COMPRESS_MAX_RATIO, get_model_context_length

SUMMARIZE_EVERY = 10


class ConversationMemory:
    def __init__(self, db_path: Path | str = DB_PATH):
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._ensure_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute('PRAGMA journal_mode=WAL')
        return self._conn

    def _ensure_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                metadata TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                message_count INTEGER NOT NULL,
                level INTEGER DEFAULT 0,
                timestamp REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id);
        """)
        conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_summaries_level ON summaries(session_id, level);
        """)
        conn.commit()

    def save_message(self, session_id: str, role: str, content: str) -> None:
        conn = self._get_conn()
        now = time.time()
        conn.execute(
            "INSERT INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET updated_at = ?",
            (session_id, now, now, now),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now),
        )
        conn.commit()

    def get_messages(self, session_id: str, limit: int = 100) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT role, content, timestamp FROM messages "
            "WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]} for r in reversed(rows)]

    def save_summary(self, session_id: str, summary: str, message_count: int, level: int = 0) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO summaries (session_id, summary, message_count, level, timestamp) VALUES (?, ?, ?, ?, ?)",
            (session_id, summary, message_count, level, time.time()),
        )
        conn.commit()

    def get_latest_summary(self, session_id: str, level: int | None = None) -> str | None:
        conn = self._get_conn()
        if level is not None:
            row = conn.execute(
                "SELECT summary FROM summaries WHERE session_id = ? AND level = ? ORDER BY timestamp DESC LIMIT 1",
                (session_id, level),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT summary FROM summaries WHERE session_id = ? ORDER BY level DESC, timestamp DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return row["summary"] if row else None

    def get_all_summaries(self, session_id: str, level: int | None = None) -> list[dict]:
        conn = self._get_conn()
        if level is not None:
            rows = conn.execute(
                "SELECT id, summary, message_count, level, timestamp FROM summaries "
                "WHERE session_id = ? AND level = ? ORDER BY timestamp",
                (session_id, level),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, summary, message_count, level, timestamp FROM summaries "
                "WHERE session_id = ? ORDER BY level, timestamp",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_summaries(self, session_id: str, ids: list[int]) -> None:
        if not ids:
            return
        conn = self._get_conn()
        placeholders = ",".join("?" * len(ids))
        conn.execute(
            f"DELETE FROM summaries WHERE session_id = ? AND id IN ({placeholders})",
            [session_id] + ids,
        )
        conn.commit()

    def get_context_memory(self, session_id: str, max_chars: int = 8000) -> str:
        summaries = self.get_all_summaries(session_id)
        if not summaries:
            return ""
        best = summaries[-1]
        text = best["summary"]
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text

    def get_message_count(self, session_id: str) -> int:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row["cnt"] if row else 0

    def list_sessions(self, limit: int = 50) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT s.session_id, s.created_at, s.updated_at, "
            "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.session_id) as msg_count "
            "FROM sessions s ORDER BY s.updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def export_session(self, session_id: str) -> str:
        messages = self.get_messages(session_id, limit=10000)
        summary = self.get_latest_summary(session_id)
        data = {
            "session_id": session_id,
            "summary": summary,
            "messages": messages,
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    def search(self, query: str, session_id: str | None = None, limit: int = 20) -> list[dict]:
        conn = self._get_conn()
        escaped = query.replace('%', '\\%').replace('_', '\\_')
        if session_id:
            rows = conn.execute(
                "SELECT session_id, role, content, timestamp FROM messages "
                "WHERE session_id = ? AND content LIKE ? ESCAPE '\\' ORDER BY timestamp DESC LIMIT ?",
                (session_id, f"%{escaped}%", limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT session_id, role, content, timestamp FROM messages "
                "WHERE content LIKE ? ESCAPE '\\' ORDER BY timestamp DESC LIMIT ?",
                (f"%{escaped}%", limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
