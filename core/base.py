from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GenerateConfig:
    max_length: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    stop_strings: list[str] | None = None


@dataclass
class GenerateResult:
    text: str
    tokens: int = 0
    elapsed_ms: float = 0.0

    @property
    def tok_per_sec(self) -> float:
        if self.elapsed_ms <= 0:
            return 0.0
        return self.tokens / (self.elapsed_ms / 1000)


@dataclass
class ModelStatus:
    name: str
    loaded: bool = False
    device: str = ""
    model_path: str = ""
    load_time_ms: float = 0.0


class BaseModelAdapter(ABC):
    name: str = "base"

    def __init__(self, model_path: Path, device: str = "GPU"):
        self.model_path = model_path
        self.device = device
        self._loaded = False
        self._load_time_ms = 0.0

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def unload(self) -> None:
        ...

    @abstractmethod
    def generate(self, messages: list[dict], config: GenerateConfig | None = None) -> GenerateResult:
        ...

    @abstractmethod
    def generate_stream(self, messages: list[dict], config: GenerateConfig | None = None) -> str:
        ...

    def status(self) -> ModelStatus:
        return ModelStatus(
            name=self.name,
            loaded=self._loaded,
            device=self.device,
            model_path=str(self.model_path),
            load_time_ms=self._load_time_ms,
        )
