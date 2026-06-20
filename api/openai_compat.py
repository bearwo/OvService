from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.base import GenerateConfig
from core.engine import ModelEngine

router = APIRouter()


class OpenAIMessage(BaseModel):
    role: str
    content: str | None = None


class OpenAIChatRequest(BaseModel):
    model: str = "qwen3.6-35b-a3b"
    messages: list[OpenAIMessage]
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_tokens: int | None = Field(None, ge=1, le=32768)
    stream: bool = False
    stop: str | list[str] | None = None


class OpenAIChoice(BaseModel):
    index: int = 0
    message: dict
    finish_reason: str = "stop"


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage


class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "ovservice"
    permission: list = []
    root: str = ""
    parent: str | None = None


class OpenAIModelList(BaseModel):
    object: str = "list"
    data: list[OpenAIModel]


def _get_engine() -> ModelEngine:
    return ModelEngine()


@router.get("/v1/models", response_model=OpenAIModelList)
def list_models():
    engine = _get_engine()
    models = []
    for s in engine.list_models():
        display_name = "Qwen3.6-35B-A3B (OpenVINO GPU)" if s.name == "chat" else s.name
        models.append(OpenAIModel(
            id=s.name,
            created=0,
            owned_by="ovservice",
            root=display_name,
        ))
    return OpenAIModelList(data=models)


@router.post("/v1/chat/completions")
async def chat_completions(req: OpenAIChatRequest):
    engine = _get_engine()
    if not engine.active():
        raise HTTPException(status_code=400, detail="No model loaded")

    messages = [{"role": m.role, "content": m.content or ""} for m in req.messages]
    if not messages or not any(m.get("role") == "user" for m in messages):
        messages = [{"role": "user", "content": "Hello"}]

    config = GenerateConfig(
        max_length=req.max_tokens or 4096,
        temperature=req.temperature,
        top_p=req.top_p,
    )

    if req.stop is not None:
        stop_list = req.stop if isinstance(req.stop, list) else [req.stop]
        config.stop_strings = stop_list

    if req.stream:
        return StreamingResponse(
            _stream_chat(engine, messages, config, req.model),
            media_type="text/event-stream",
        )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, lambda: engine.generate(messages, config)
    )

    text = result.text if hasattr(result, "text") else str(result)

    prompt_tokens = 0
    completion_tokens = result.tokens if result.tokens else 0
    total_tokens = prompt_tokens + completion_tokens

    return OpenAIChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=req.model,
        choices=[OpenAIChoice(
            message={"role": "assistant", "content": text},
            finish_reason="stop",
        )],
        usage=OpenAIUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


async def _stream_chat(engine, messages, config, model):
    if not messages or not any(m.get("role") == "user" for m in messages):
        messages = [{"role": "user", "content": "Hello"}]

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    done_event = asyncio.Event()

    def on_token(text):
        loop.call_soon_threadsafe(queue.put_nowait, text)

    def run_generate():
        try:
            engine.generate_stream(messages, config, on_token=on_token)
        except Exception as e:
            error_chunk = {
                'id': chunk_id,
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': model,
                'choices': [{'index': 0, 'delta': {'content': f'Error: {e}'}, 'finish_reason': 'stop'}],
            }
            loop.call_soon_threadsafe(queue.put_nowait, f'data: {json.dumps(error_chunk)}\n\n')
        finally:
            loop.call_soon_threadsafe(done_event.set)

    gen_thread = threading.Thread(target=run_generate, daemon=True)
    gen_thread.start()

    while True:
        try:
            token = await asyncio.wait_for(queue.get(), timeout=0.5)
            chunk_data = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"
        except asyncio.TimeoutError:
            if done_event.is_set():
                while not queue.empty():
                    token = queue.get_nowait()
                    chunk_data = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                break

    final_data = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_data)}\n\n"
    yield "data: [DONE]\n\n"
