from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from config import MAX_HISTORY_TURNS

CLEANUP_INTERVAL_SECONDS = 300


@dataclass
class Session:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    max_turns: int = MAX_HISTORY_TURNS
    _history: list[dict] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})
        max_msgs = self.max_turns * 2
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]
        self.last_active = time.time()

    def to_messages(self) -> list[dict]:
        self.last_active = time.time()
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()
        self.last_active = time.time()

    @property
    def age_seconds(self) -> float:
        return time.time() - self.last_active


class SessionManager:
    def __init__(self, timeout_seconds: int = 1800):
        self._sessions: dict[str, Session] = {}
        self._timeout = timeout_seconds
        self._cleanup_timer: threading.Timer | None = None
        self._start_cleanup_timer()

    def _start_cleanup_timer(self) -> None:
        self._cleanup_timer = threading.Timer(CLEANUP_INTERVAL_SECONDS, self._cleanup_and_reschedule)
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()

    def _cleanup_and_reschedule(self) -> None:
        self.cleanup_expired()
        self._start_cleanup_timer()

    def stop_cleanup(self) -> None:
        if self._cleanup_timer is not None:
            self._cleanup_timer.cancel()
            self._cleanup_timer = None

    def get_or_create(self, session_id: str | None = None) -> Session:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        sid = session_id or str(uuid.uuid4())
        session = Session(session_id=sid)
        self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def cleanup_expired(self) -> int:
        expired = [
            sid for sid, s in self._sessions.items()
            if s.age_seconds > self._timeout
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    def list_sessions(self) -> list[dict]:
        return [
            {
                "session_id": s.session_id,
                "created_at": s.created_at,
                "last_active": s.last_active,
                "message_count": len(s._history),
            }
            for s in self._sessions.values()
        ]
