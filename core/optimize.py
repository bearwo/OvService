from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from core.base import BaseModelAdapter, GenerateConfig, GenerateResult


@dataclass
class PerfStats:
    total_requests: int = 0
    total_tokens: int = 0
    total_ms: float = 0.0
    last_latency_ms: float = 0.0
    start_time: float = field(default_factory=time.time)

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_ms / self.total_requests

    @property
    def requests_per_sec(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return 0.0
        return self.total_requests / elapsed

    def record(self, elapsed_ms: float, tokens: int = 0) -> None:
        self.total_requests += 1
        self.total_ms += elapsed_ms
        self.last_latency_ms = elapsed_ms
        self.total_tokens += tokens

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_ms": round(self.total_ms, 1),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "last_latency_ms": round(self.last_latency_ms, 1),
            "requests_per_sec": round(self.requests_per_sec, 2),
            "uptime_seconds": round(time.time() - self.start_time, 1),
        }


class LRUModelManager:
    def __init__(self, max_loaded: int = 2):
        self._max_loaded = max_loaded
        self._order: OrderedDict[str, BaseModelAdapter] = OrderedDict()
        self._lock = threading.Lock()

    def touch(self, name: str) -> None:
        with self._lock:
            if name in self._order:
                self._order.move_to_end(name)

    def register(self, name: str, adapter: BaseModelAdapter) -> None:
        with self._lock:
            self._order[name] = adapter

    def evict_if_needed(self) -> list[str]:
        evicted = []
        with self._lock:
            while len(self._order) > self._max_loaded:
                oldest_name, oldest_adapter = next(iter(self._order.items()))
                if oldest_adapter._loaded:
                    oldest_adapter.unload()
                    evicted.append(oldest_name)
                self._order.move_to_end(oldest_name)
                self._order.popitem(last=True)
        return evicted

    @property
    def loaded_count(self) -> int:
        with self._lock:
            return sum(1 for a in self._order.values() if a._loaded)


class InferenceParams:
    def __init__(self):
        self._lock = threading.Lock()
        self._params: dict[str, Any] = {
            "max_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
        }

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._params.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._params[key] = value

    def update(self, **kwargs) -> None:
        with self._lock:
            self._params.update(kwargs)

    def to_config(self) -> GenerateConfig:
        with self._lock:
            return GenerateConfig(
                max_length=self._params.get("max_length", 2048),
                temperature=self._params.get("temperature", 0.7),
                top_p=self._params.get("top_p", 0.9),
                top_k=self._params.get("top_k", 50),
                repetition_penalty=self._params.get("repetition_penalty", 1.1),
            )

    def to_dict(self) -> dict:
        with self._lock:
            return dict(self._params)
