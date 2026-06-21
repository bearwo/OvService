from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from config import MAX_HISTORY_TURNS, COMPRESS_CONTEXT_RATIO, get_model_context_length

IMAGE_KEYWORDS = re.compile(
    r"图[片片照]|照[片]|图[像]|图[文]|图[片]?[内容描述]|那[张份个].*[图照]|之前.*[图照]|刚[才才].*[图照]|上[一]?[张份].*[图照]|前.*[图照]|你[看见过].*[图照]|描述.*[图照]|[图照][片像].*怎么样|[图照][片像].*是什么|[图照][片像].*有[什么哪]",
    re.IGNORECASE,
)

FILE_KEYWORDS = re.compile(
    r"文[件档]|文[本档]?[内容]|那[份个].*文[件档]|之前.*文[件档]|刚[才才].*文[件档]|上[一]?[份个].*文[件档]|文[件档].*怎么样|文[件档].*是什么",
    re.IGNORECASE,
)


@dataclass
class Turn:
    role: str
    content: str
    image_path: Optional[str] = None


class Conversation:
    def __init__(
        self,
        max_turns: int = MAX_HISTORY_TURNS,
        tokenizer=None,
    ):
        self.max_turns = max_turns
        self._max_tokens = get_model_context_length()
        self._compress_at = int(self._max_tokens * COMPRESS_CONTEXT_RATIO)
        self._history: list[Turn] = []
        self._memory_context: str = ""
        self._turn_count: int = 0
        self._needs_compression: bool = False
        self._tokenizer = tokenizer
        self._token_cache: dict[int, int] = {}

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def estimated_tokens(self) -> int:
        total = self._count_tokens(self._memory_context) if self._memory_context else 0
        total += sum(self._count_tokens(t.content) for t in self._history)
        return total

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        text_id = hash(text)
        if text_id in self._token_cache:
            return self._token_cache[text_id]
        if self._tokenizer is not None:
            try:
                encoded = self._tokenizer.encode(text)
                count = encoded.input_ids.shape[1] if len(encoded.input_ids.shape) > 1 else len(encoded.input_ids.data)
            except Exception:
                count = len(text) // 2
        else:
            count = len(text) // 2
        self._token_cache[text_id] = count
        return count

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

    def add_user_with_image(self, content: str, image_path: str) -> None:
        self._history.append(Turn(role="user", content=content, image_path=image_path))
        self._turn_count += 1
        self._check_compression()

    def add_file_ref(self, content: str, file_path: str) -> None:
        self._history.append(Turn(role="user", content=content, image_path=file_path))
        self._turn_count += 1
        self._check_compression()

    def get_latest_image_path(self) -> str | None:
        for turn in reversed(self._history):
            if turn.image_path:
                return turn.image_path
        return None

    def get_recent_image_path(self, max_turns_ago: int = 10) -> str | None:
        recent_turns = self._history[-max_turns_ago * 2:]
        for turn in reversed(recent_turns):
            if turn.image_path and turn.image_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
                return turn.image_path
        return None

    def get_recent_file_path(self, max_turns_ago: int = 10) -> str | None:
        recent_turns = self._history[-max_turns_ago * 2:]
        for turn in reversed(recent_turns):
            if turn.image_path and not turn.image_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
                return turn.image_path
        return None

    def is_asking_about_image(self, text: str) -> bool:
        return bool(IMAGE_KEYWORDS.search(text))

    def is_asking_about_file(self, text: str) -> bool:
        return bool(FILE_KEYWORDS.search(text))

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

    def to_messages_with_limit(self, max_chars: int = 0) -> list[dict]:
        if max_chars <= 0:
            return self.to_messages()
        messages = []
        if self._memory_context:
            messages.append({"role": "system", "content": self._memory_context})
        total = 0
        for turn in reversed(self._history):
            msg = {"role": turn.role, "content": turn.content}
            msg_len = len(turn.content)
            if total + msg_len > max_chars:
                break
            messages.append(msg)
            total += msg_len
        messages.reverse()
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
