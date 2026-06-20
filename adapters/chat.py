from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import openvino as ov
import openvino_genai as ov_genai

from core.base import BaseModelAdapter, GenerateConfig, GenerateResult


def _is_garbled(text: str) -> bool:
    if len(text) < 20:
        return False
    alpha_count = sum(1 for c in text if c.isalpha())
    alpha_ratio = alpha_count / len(text)
    if alpha_ratio < 0.4:
        return True
    punct_count = sum(1 for c in text if c in "*#<>{}[]|\\~`^")
    if punct_count > len(text) * 0.1:
        return True
    return False


class _TextCollector(ov_genai.StreamerBase):
    def __init__(self, tokenizer: ov_genai.Tokenizer):
        super().__init__()
        self._tokenizer = tokenizer
        self._tokens: list[str] = []
        self._ready = threading.Event()
        self._done = False
        self._lock = threading.Lock()

    def write(self, token) -> ov_genai.StreamingStatus:
        if isinstance(token, (list, tuple)):
            for t in token:
                text = self._tokenizer.decode([int(t)])
                with self._lock:
                    self._tokens.append(text)
        else:
            text = self._tokenizer.decode([int(token)])
            with self._lock:
                self._tokens.append(text)
        self._ready.set()
        return ov_genai.StreamingStatus.RUNNING

    def end(self):
        with self._lock:
            self._done = True
        self._ready.set()

    def stream(self, on_token):
        idx = 0
        while True:
            with self._lock:
                if idx < len(self._tokens):
                    text = self._tokens[idx]
                    idx += 1
                elif self._done:
                    return
                else:
                    text = None
            if text is not None:
                on_token(text)
            else:
                self._ready.wait(timeout=0.1)
                self._ready.clear()


class ChatAdapter(BaseModelAdapter):
    name = "chat"

    def __init__(self, model_path, device="GPU"):
        super().__init__(model_path, device)
        self._pipe = None
        self.thinking = True

    def load(self) -> None:
        if self._loaded:
            return
        t0 = time.perf_counter()
        self._pipe = ov_genai.VLMPipeline(str(self.model_path), self.device)
        self._load_time_ms = (time.perf_counter() - t0) * 1000
        self._loaded = True

    def unload(self) -> None:
        self._pipe = None
        self._loaded = False

    def _build_config(self, config: GenerateConfig | None) -> ov_genai.GenerationConfig:
        gc = ov_genai.GenerationConfig()
        if config is None:
            config = GenerateConfig()
        gc.max_new_tokens = config.max_length
        gc.temperature = config.temperature
        gc.top_p = config.top_p
        gc.top_k = config.top_k
        gc.repetition_penalty = config.repetition_penalty
        gc.do_sample = config.do_sample
        if config.stop_strings:
            gc.stop_strings = config.stop_strings
        return gc

    def generate(
        self,
        messages: list[dict],
        config: GenerateConfig | None = None,
        images: list[ov.Tensor] | None = None,
        knowledge_context: str = "",
    ) -> GenerateResult:
        for attempt in range(2):
            if not self._loaded:
                self.load()
            gc = self._build_config(config)
            use_messages = messages if attempt == 0 else self._truncate_messages(messages)
            prompt = self._messages_to_prompt(use_messages, knowledge_context)
            t0 = time.perf_counter()
            if images:
                result = self._pipe.generate(prompt, images=images, generation_config=gc)
            else:
                result = self._pipe.generate(prompt, generation_config=gc)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            text = str(result.texts[0]) if hasattr(result, "texts") else str(result)
            if not _is_garbled(text) or attempt == 1:
                tok = self._pipe.get_tokenizer()
                try:
                    encoded = tok.encode(text)
                    tokens = encoded.input_ids.shape[1] if len(encoded.input_ids.shape) > 1 else len(encoded.input_ids.data)
                except Exception:
                    tokens = 0
                return GenerateResult(text=text, tokens=tokens, elapsed_ms=elapsed_ms)

    def generate_stream(
        self,
        messages: list[dict],
        config: GenerateConfig | None = None,
        images: list[ov.Tensor] | None = None,
        knowledge_context: str = "",
    ) -> str:
        for attempt in range(2):
            if not self._loaded:
                self.load()
            gc = self._build_config(config)
            use_messages = messages if attempt == 0 else self._truncate_messages(messages)
            prompt = self._messages_to_prompt(use_messages, knowledge_context)
            tokenizer = self._pipe.get_tokenizer()
            collector = _TextCollector(tokenizer)

            def run_generate():
                if images:
                    self._pipe.generate(prompt, images=images, generation_config=gc, streamer=collector)
                else:
                    self._pipe.generate(prompt, generation_config=gc, streamer=collector)

            result = []
            cleared = False

            def on_token(text):
                nonlocal cleared
                result.append(text)
                if not cleared:
                    sys.stdout.write(text)
                    sys.stdout.flush()

            gen_thread = threading.Thread(target=run_generate, daemon=True)
            gen_thread.start()
            collector.stream(on_token)
            gen_thread.join()

            full_text = "".join(result)
            if not _is_garbled(full_text) or attempt == 1:
                if cleared:
                    print(full_text)
                sys.stdout.write("\n")
                sys.stdout.flush()
                return full_text
            else:
                result.clear()
                cleared = True
                print("[Output garbled, retrying with shorter context...]")

    def _messages_to_prompt(self, messages: list[dict], knowledge_context: str = "") -> str:
        tok = self._pipe.get_tokenizer()
        chat_msgs = []
        if knowledge_context:
            chat_msgs.append({"role": "system", "content": "Use the following context to answer the question:\n\n" + knowledge_context})
        else:
            chat_msgs.append({"role": "system", "content": "You are a helpful assistant."})
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                content = " ".join(text_parts)
            chat_msgs.append({"role": role, "content": content})
        return tok.apply_chat_template(
            chat_msgs,
            add_generation_prompt=True,
            extra_context={"enable_thinking": self.thinking},
        )

    def _truncate_messages(self, messages: list[dict]) -> list[dict]:
        if len(messages) <= 4:
            return messages
        system = [m for m in messages if m.get("role") == "system"]
        recent = messages[-4:]
        if system:
            return system + recent
        return recent
