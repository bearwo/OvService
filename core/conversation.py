from __future__ import annotations

from dataclasses import dataclass

from config import MAX_HISTORY_TURNS, COMPRESS_CONTEXT_RATIO, get_model_context_length


@dataclass
class Turn:
    role: str
    content: str


class Conversation:
    def __init__(
        self,
        max_turns: int = MAX_HISTORY_TURNS,
    ):
        self.max_turns = max_turns
        self._max_tokens = get_model_context_length()
        self._compress_at = int(self._max_tokens * COMPRESS_CONTEXT_RATIO)
        self._history: list[Turn] = []
        self._memory_context: str = ""
        self._turn_count: int = 0
        self._needs_compression: bool = False

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def estimated_tokens(self) -> int:
        total = len(self._memory_context) // 2 if self._memory_context else 0
        total += sum(len(t.content) // 2 for t in self._history)
        return total

    @property
    def needs_compression(self) -> bool:
        return self._needs_compression

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def compress_at(self) -> int:
        return self._compress_at

    def consume_compression(self) -> None:
        self._needs_compression = False

    def set_memory_context(self, text: str) -> None:
        self._memory_context = text

    def add_user(self, content: str) -> None:
        self._history.append(Turn(role="user", content=content))
        self._turn_count += 1
        self._check_compression()

    def add_assistant(self, content: str) -> None:
        self._history.append(Turn(role="assistant", content=content))
        self._check_compression()

    def clear(self) -> None:
        self._history.clear()
        self._turn_count = 0
        self._memory_context = ""
        self._needs_compression = False

    def to_messages(self) -> list[dict]:
        messages = []
        if self._memory_context:
            messages.append({"role": "system", "content": self._memory_context})
        for turn in self._history:
            messages.append({"role": turn.role, "content": turn.content})
        return messages

    def summary(self) -> str:
        if not self._history:
            return "(empty conversation)"
        lines = []
        for turn in self._history[-6:]:
            prefix = "You" if turn.role == "user" else "AI"
            text = turn.content[:80] + ("..." if len(turn.content) > 80 else "")
            lines.append(f"  {prefix}: {text}")
        return "\n".join(lines)

    def _check_compression(self) -> None:
        if len(self._history) > self.max_turns * 2:
            self._needs_compression = True
        if self.estimated_tokens > self.compress_at:
            self._needs_compression = True

    def get_old_messages_for_summary(self, keep_recent: int = 6) -> list[dict]:
        if len(self._history) <= keep_recent * 2:
            return []
        old = self._history[: -keep_recent * 2]
        return [{"role": t.role, "content": t.content} for t in old]

    def trim_after_summary(self, keep_recent: int = 6) -> None:
        if len(self._history) > keep_recent * 2:
            self._history = self._history[-keep_recent * 2:]
