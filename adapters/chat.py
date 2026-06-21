from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import openvino as ov
import openvino_genai as ov_genai

from config import get_model_context_length
from core.base import BaseModelAdapter, GenerateConfig, GenerateResult

logger = logging.getLogger(__name__)

# 上下文安全余量：为 template 开销、图片 token、thinking token 留出空间
_CONTEXT_SAFETY_MARGIN = 2048


def _is_garbled(text: str) -> bool:
    """检测模型输出是否为真正的编码乱码（而非正常中文/代码）。

    仅针对编码错误的强信号：
    1. U+FFFD 替换符（UTF-8 解码失败标志）
    2. 非换行控制字符（\x00-\x1f 除 \\n\\t\\r）
    3. 极低唯一字符数（重复填充型乱码）
    """
    if len(text) < 20:
        return False
    # 1. 替换符 U+FFFD —— 真乱码的强信号
    if text.count("\ufffd") / len(text) > 0.03:
        return True
    # 2. 非换行控制字符
    ctrl = sum(
        1 for c in text
        if ord(c) < 32 and c not in "\n\t\r"
    )
    if ctrl / len(text) > 0.03:
        return True
    # 3. 极低唯一字符数（重复填充型，如 "锘匡丰锘匡丰..."）
    if len(set(text)) < max(3, len(text) * 0.02) and len(text) > 50:
        return True
    return False


class _TextCollector(ov_genai.StreamerBase):
    def __init__(self, tokenizer: ov_genai.Tokenizer):
        super().__init__()
        self._tokenizer = tokenizer
        self._tokens: list[str] = []
        self._ready = threading.Event()
        self._done = False
        self._interrupted = False
        self._lock = threading.Lock()

    def write(self, token) -> ov_genai.StreamingStatus:
        if self._interrupted:
            return ov_genai.StreamingStatus.STOP
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

    def interrupt(self):
        with self._lock:
            self._interrupted = True
        self._ready.set()

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
        self._gen_lock = threading.Lock()
        self._current_collector: _TextCollector | None = None

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

    def interrupt(self):
        collector = self._current_collector
        if collector:
            collector.interrupt()

    def _try_recover(self):
        try:
            self._pipe.finish_chat()
        except Exception:
            pass
        try:
            self.unload()
            self.load()
            self._pipe.finish_chat()
        except Exception:
            pass

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

    def _count_message_tokens(self, text: str, tokenizer) -> int:
        """用 tokenizer 真实编码计数，失败时回退到 len//2 估算。"""
        if not text:
            return 0
        try:
            encoded = tokenizer.encode(text)
            if len(encoded.input_ids.shape) > 1:
                return encoded.input_ids.shape[1]
            return len(encoded.input_ids.data)
        except Exception:
            return len(text) // 2

    def _trim_messages_by_budget(
        self,
        messages: list[dict],
        max_new_tokens: int,
        tokenizer,
    ) -> list[dict]:
        """按 token 预算从尾部往前保留 messages，确保 prompt 不超出模型上下文窗口。

        策略：
        - 始终保留 system 消息
        - 始终保留最后一条 user 消息（当前提问）
        - 从倒数第二条开始往前累加 token，直到预算用完
        - 预算 = 模型上下文长度 - max_new_tokens - 安全余量
        """
        context_length = get_model_context_length()
        budget = context_length - max_new_tokens - _CONTEXT_SAFETY_MARGIN
        if budget < 512:
            budget = 512

        # 分离 system 和非 system
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if not non_system:
            return messages

        # 始终保留最后一条消息（通常是当前 user 提问）
        last_msg = non_system[-1]
        remaining = non_system[:-1]

        # 计算 system + 最后一条消息的 token 开销
        system_tokens = sum(
            self._count_message_tokens(m.get("content", ""), tokenizer) for m in system_msgs
        )
        last_tokens = self._count_message_tokens(last_msg.get("content", ""), tokenizer)

        # template 开销估算（chat template 会添加特殊 token）
        template_overhead = 50 + (2 * len(remaining))  # <im_start>/<im_end> 等
        remaining_budget = budget - system_tokens - last_tokens - template_overhead

        if remaining_budget <= 0:
            # 极端情况：system + 最后一条已经超预算，只返回它们
            logger.warning(
                "Context budget exhausted even with only system + last message "
                "(system=%d, last=%d, budget=%d)",
                system_tokens, last_tokens, budget,
            )
            return system_msgs + [last_msg]

        # 从倒数第二条往前累加，直到预算用完
        kept = []
        for msg in reversed(remaining):
            msg_tokens = self._count_message_tokens(msg.get("content", ""), tokenizer)
            if remaining_budget - msg_tokens < 0:
                break
            remaining_budget -= msg_tokens
            kept.append(msg)

        kept.reverse()
        trimmed = system_msgs + kept + [last_msg]

        dropped = len(remaining) - len(kept)
        if dropped > 0:
            logger.info(
                "Trimmed %d messages to fit token budget (budget=%d, kept=%d/%d)",
                dropped, budget, len(trimmed), len(messages),
            )

        return trimmed

    def generate(
        self,
        messages: list[dict],
        config: GenerateConfig | None = None,
        images: list[ov.Tensor] | None = None,
        knowledge_context: str = "",
    ) -> GenerateResult:
        with self._gen_lock:
            for attempt in range(2):
                if not self._loaded:
                    self.load()
                gc = self._build_config(config)
                # 第一次尝试：按 token 预算智能截断；重试：进一步激进截断
                if attempt == 0:
                    use_messages = self._trim_messages_by_budget(
                        messages, gc.max_new_tokens, self._pipe.get_tokenizer(),
                    )
                else:
                    use_messages = self._truncate_messages(messages)
                if images:
                    img_items = [{"type": "image"} for _ in images]
                    last_user_idx = next(
                        (i for i in range(len(use_messages) - 1, -1, -1) if use_messages[i].get("role") == "user"),
                        -1,
                    )
                    if last_user_idx >= 0:
                        msg = use_messages[last_user_idx]
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            use_messages = list(use_messages)
                            use_messages[last_user_idx] = {"role": "user", "content": [{"type": "text", "text": content}] + img_items}
                prompt = self._messages_to_prompt(use_messages, knowledge_context)
                t0 = time.perf_counter()
                try:
                    if images:
                        result = self._pipe.generate(prompt, images=images, generation_config=gc)
                    else:
                        result = self._pipe.generate(prompt, generation_config=gc)
                except Exception as e:
                    if attempt == 0:
                        continue
                    self._try_recover()
                    raise RuntimeError(f"Generation failed after retry: {e}") from e
                finally:
                    try:
                        self._pipe.finish_chat()
                    except Exception:
                        pass
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
        on_token=None,
    ):
        with self._gen_lock:
            for attempt in range(2):
                if not self._loaded:
                    self.load()
                gc = self._build_config(config)
                # 第一次尝试：按 token 预算智能截断；重试：进一步激进截断
                if attempt == 0:
                    use_messages = self._trim_messages_by_budget(
                        messages, gc.max_new_tokens, self._pipe.get_tokenizer(),
                    )
                else:
                    use_messages = self._truncate_messages(messages)
                prompt = self._messages_to_prompt(use_messages, knowledge_context)
                tokenizer = self._pipe.get_tokenizer()
                collector = _TextCollector(tokenizer)
                self._current_collector = collector

                def run_generate():
                    try:
                        if images:
                            self._pipe.generate(prompt, images=images, generation_config=gc, streamer=collector)
                        else:
                            self._pipe.generate(prompt, generation_config=gc, streamer=collector)
                    finally:
                        self._current_collector = None

                result = []

                def handle_token(text):
                    result.append(text)
                    if on_token:
                        on_token(text)
                    else:
                        sys.stdout.write(text)
                        sys.stdout.flush()

                try:
                    gen_thread = threading.Thread(target=run_generate, daemon=True)
                    gen_thread.start()
                    collector.stream(handle_token)
                    gen_thread.join()
                finally:
                    try:
                        self._pipe.finish_chat()
                    except Exception:
                        self._try_recover()

                full_text = "".join(result)
                if not _is_garbled(full_text) or attempt == 1:
                    if not on_token:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                    return full_text
                else:
                    if not on_token:
                        print("[Output garbled, retrying with shorter context...]")

    def _messages_to_prompt(self, messages: list[dict], knowledge_context: str = "") -> str:
        tok = self._pipe.get_tokenizer()
        chat_msgs = []
        has_system = messages and messages[0].get("role") == "system"
        if not has_system:
            if knowledge_context:
                chat_msgs.append({"role": "system", "content": "Use the following context to answer the question:\n\n" + knowledge_context})
            else:
                chat_msgs.append({"role": "system", "content": "You are a helpful assistant."})
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                chat_msgs.append({"role": role, "content": content})
            else:
                chat_msgs.append({"role": role, "content": content})
        prompt = tok.apply_chat_template(
            chat_msgs,
            add_generation_prompt=True,
            extra_context={"enable_thinking": self.thinking},
        )
        if not self.thinking and "💭" in prompt and "🍈" not in prompt:
            prompt = prompt.replace("💭", "💭\n\n🍈")
        return prompt

    def _truncate_messages(self, messages: list[dict]) -> list[dict]:
        """乱码重试时的激进截断：保留 system + 最近 8 条消息。"""
        if len(messages) <= 8:
            return messages
        system = [m for m in messages if m.get("role") == "system"]
        recent = messages[-8:]
        if system:
            return system + recent
        return recent
