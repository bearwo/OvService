from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from config import KNOWLEDGE_DIR, DB_PATH
from features.file_parser import parse_file, chunk_text


class KnowledgeBase:
    def __init__(self, db_path: Path | str = DB_PATH, kb_dir: Path | str = KNOWLEDGE_DIR):
        self._db_path = str(db_path)
        self._kb_dir = Path(kb_dir)
        self._kb_dir.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._db_lock = threading.Lock()
        self._ensure_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute('PRAGMA journal_mode=WAL')
        return self._conn

    def _ensure_db(self) -> None:
        with self._db_lock:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS knowledge_docs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    imported_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES knowledge_docs(id)
                );
            """)
            conn.commit()

    def add_file(self, file_path: str | Path) -> dict:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        text = parse_file(path)
        chunks = chunk_text(text)

        with self._db_lock:
            conn = self._get_conn()
            now = time.time()
            cursor = conn.execute(
                "INSERT INTO knowledge_docs (filename, filepath, chunk_count, imported_at) VALUES (?, ?, ?, ?)",
                (path.name, str(path), len(chunks), now),
            )
            doc_id = cursor.lastrowid

            for i, chunk in enumerate(chunks):
                conn.execute(
                    "INSERT INTO knowledge_chunks (doc_id, chunk_index, content) VALUES (?, ?, ?)",
                    (doc_id, i, chunk),
                )
            conn.commit()

            dest = self._kb_dir / path.name
            if not dest.exists():
                dest.write_text(text, encoding="utf-8")

        return {
            "doc_id": doc_id,
            "filename": path.name,
            "chunk_count": len(chunks),
            "imported_at": now,
        }

    def search(self, query: str, limit: int = 5) -> list[dict]:
        with self._db_lock:
            conn = self._get_conn()
            words = [w.strip().replace('%', '\\%').replace('_', '\\_') for w in query.split() if w.strip()]
            if not words:
                return []
            where = " AND ".join(["kc.content LIKE ? ESCAPE '\\'"] * len(words))
            params = [f"%{w}%" for w in words] + [limit]
            rows = conn.execute(
                "SELECT kc.content, kd.filename, kd.filepath, kc.chunk_index "
                "FROM knowledge_chunks kc "
                "JOIN knowledge_docs kd ON kc.doc_id = kd.id "
                f"WHERE {where} "
                "LIMIT ?",
                params,
            ).fetchall()
            return [dict(r) for r in rows]

    def list_documents(self) -> list[dict]:
        with self._db_lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT id, filename, filepath, chunk_count, imported_at "
                "FROM knowledge_docs ORDER BY imported_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_context(self, query: str, max_chunks: int = 3) -> str:
        results = self.search(query, limit=max_chunks)
        if not results:
            return ""
        parts = []
        for r in results:
            parts.append(f"[{r['filename']}] {r['content'][:500]}")
        return "\n\n".join(parts)

    def delete_document(self, doc_id: int) -> bool:
        with self._db_lock:
            conn = self._get_conn()
            row = conn.execute("SELECT filename FROM knowledge_docs WHERE id = ?", (doc_id,)).fetchone()
            if not row:
                return False
            kb_file = self._kb_dir / row["filename"]
            if kb_file.exists():
                kb_file.unlink()
            conn.execute("DELETE FROM knowledge_chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM knowledge_docs WHERE id = ?", (doc_id,))
            conn.commit()
            return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self) -> None:
        with self._db_lock:
            if self._conn:
                self._conn.close()
                self._conn = None
