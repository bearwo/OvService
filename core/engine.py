from __future__ import annotations

import threading

from core.base import BaseModelAdapter, GenerateConfig, GenerateResult, ModelStatus


class ModelEngine:
    _instance: ModelEngine | None = None
    _lock = threading.Lock()

    def __new__(cls) -> ModelEngine:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._adapters = {}
                    cls._instance._active = None
        return cls._instance

    def register(self, adapter: BaseModelAdapter) -> None:
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> BaseModelAdapter | None:
        return self._adapters.get(name)

    def set_active(self, name: str) -> bool:
        if name not in self._adapters:
            return False
        self._active = name
        return True

    def active(self) -> BaseModelAdapter | None:
        if self._active and self._active in self._adapters:
            return self._adapters[self._active]
        return None

    def list_models(self) -> list[ModelStatus]:
        return [a.status() for a in self._adapters.values()]

    def load(self, name: str) -> bool:
        adapter = self._adapters.get(name)
        if adapter is None:
            return False
        adapter.load()
        return True

    def unload(self, name: str) -> bool:
        adapter = self._adapters.get(name)
        if adapter is None:
            return False
        adapter.unload()
        return True

    def generate(self, messages: list[dict], config: GenerateConfig | None = None, **kwargs) -> GenerateResult:
        adapter = self.active()
        if adapter is None:
            raise RuntimeError("No active model. Use /model load <name> first.")
        return adapter.generate(messages, config, **kwargs)

    def generate_stream(self, messages: list[dict], config: GenerateConfig | None = None, **kwargs) -> str:
        adapter = self.active()
        if adapter is None:
            raise RuntimeError("No active model. Use /model load <name> first.")
        return adapter.generate_stream(messages, config, **kwargs)
