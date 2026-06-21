from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse

VALID_ROLES = {"user", "assistant", "system"}


def _validate_messages(messages: list[dict]) -> None:
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role not in VALID_ROLES:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i}: invalid role '{role}', must be one of {sorted(VALID_ROLES)}",
            )
        content = msg.get("content")
        if not isinstance(content, str):
            raise HTTPException(
                status_code=400,
                detail=f"Message {i}: 'content' must be a string, got {type(content).__name__}",
            )
from api.schemas import (
    ChatRequest,
    ChatResponse,
    ModelsResponse,
    ModelInfo,
    HealthResponse,
    KnowledgeAddRequest,
    KnowledgeSearchRequest,
    KnowledgeResponse,
    KnowledgeDocInfo,
)
from api.session import SessionManager
from config import UPLOADS_DIR
from core.base import GenerateConfig
from core.engine import ModelEngine
from features.memory import ConversationMemory
from features.knowledge import KnowledgeBase
from features.image import load_image_from_bytes

router = APIRouter()
session_mgr = SessionManager()
memory = ConversationMemory()
knowledge = KnowledgeBase()


def _get_engine() -> ModelEngine:
    return ModelEngine()


@router.get("/health", response_model=HealthResponse)
def health():
    engine = _get_engine()
    active = engine.active()
    model_loaded = False
    database_ok = False
    try:
        memory._get_conn().execute("SELECT 1")
        database_ok = True
    except Exception:
        pass
    if active:
        model_loaded = active._loaded
    return HealthResponse(
        active_model=active.name if active else None,
        database_ok=database_ok,
        model_loaded=model_loaded,
    )


@router.post("/chat/stream/interrupt")
def interrupt_generation():
    engine = _get_engine()
    adapter = engine.active()
    if adapter and hasattr(adapter, "interrupt"):
        adapter.interrupt()
        return {"status": "interrupted"}
    return {"status": "no active generation"}


@router.get("/models", response_model=ModelsResponse)
def list_models():
    engine = _get_engine()
    statuses = engine.list_models()
    return ModelsResponse(
        models=[
            ModelInfo(
                name=s.name,
                loaded=s.loaded,
                device=s.device,
                model_path=s.model_path,
                load_time_ms=s.load_time_ms,
            )
            for s in statuses
        ]
    )


@router.post("/models/{name}/load")
def load_model(name: str):
    engine = _get_engine()
    if engine.load(name):
        engine.set_active(name)
        return {"status": "loaded", "name": name}
    raise HTTPException(status_code=404, detail=f"Model '{name}' not found")


@router.post("/models/{name}/unload")
def unload_model(name: str):
    engine = _get_engine()
    if engine.unload(name):
        return {"status": "unloaded", "name": name}
    raise HTTPException(status_code=404, detail=f"Model '{name}' not found")


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, session_id: str = Query(default="default")):
    _validate_messages(req.messages)
    engine = _get_engine()
    if not engine.active():
        raise HTTPException(status_code=400, detail="No model loaded")

    config = GenerateConfig(
        max_length=req.max_length,
        temperature=req.temperature,
        top_p=req.top_p,
    )

    session = session_mgr.get_or_create(session_id)

    if req.messages:
        last_user_msg = next((m for m in reversed(req.messages) if m["role"] == "user"), req.messages[-1])
        session.add_message(last_user_msg["role"], last_user_msg["content"])
        memory.save_message(session_id, last_user_msg["role"], last_user_msg["content"])

    knowledge_context = ""
    if req.use_knowledge and req.messages:
        last_msg = req.messages[-1].get("content", "")
        query = req.knowledge_query or last_msg
        knowledge_context = knowledge.get_context(query)

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, lambda: engine.generate(req.messages, config, knowledge_context=knowledge_context)
    )

    session.add_message("assistant", result.text)
    memory.save_message(session_id, "assistant", result.text)

    return ChatResponse(text=result.text, elapsed_ms=result.elapsed_ms)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, session_id: str = Query(default="default")):
    _validate_messages(req.messages)
    engine = _get_engine()
    if not engine.active():
        raise HTTPException(status_code=400, detail="No model loaded")

    config = GenerateConfig(
        max_length=req.max_length,
        temperature=req.temperature,
        top_p=req.top_p,
    )

    session = session_mgr.get_or_create(session_id)

    if req.messages:
        last_user_msg = next((m for m in reversed(req.messages) if m["role"] == "user"), req.messages[-1])
        session.add_message(last_user_msg["role"], last_user_msg["content"])
        memory.save_message(session_id, last_user_msg["role"], last_user_msg["content"])

    knowledge_context = ""
    if req.use_knowledge and req.messages:
        last_msg = req.messages[-1].get("content", "")
        query = req.knowledge_query or last_msg
        knowledge_context = knowledge.get_context(query)

    async def event_generator():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        done_event = asyncio.Event()
        full_response = []

        def on_token(text):
            full_response.append(text)
            loop.call_soon_threadsafe(queue.put_nowait, text)

        def run_generate():
            try:
                engine.generate_stream(
                    req.messages, config,
                    knowledge_context=knowledge_context,
                    on_token=on_token,
                )
            finally:
                loop.call_soon_threadsafe(done_event.set)

        gen_thread = threading.Thread(target=run_generate, daemon=True)
        gen_thread.start()

        while True:
            try:
                token = await asyncio.wait_for(queue.get(), timeout=0.5)
                yield f"data: {json.dumps({'token': token})}\n\n"
            except asyncio.TimeoutError:
                if done_event.is_set():
                    while not queue.empty():
                        token = queue.get_nowait()
                        yield f"data: {json.dumps({'token': token})}\n\n"
                    break

        response_text = "".join(full_response)
        session.add_message("assistant", response_text)
        memory.save_message(session_id, "assistant", response_text)

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/chat/image/upload", response_model=ChatResponse)
async def chat_image_upload(
    session_id: str = Query(default="default"),
    message: str = Form(default="Describe this image"),
    max_length: int = Form(default=2048),
    temperature: float = Form(default=0.7),
    top_p: float = Form(default=0.9),
    file: UploadFile = File(...),
):
    engine = _get_engine()
    if not engine.active():
        raise HTTPException(status_code=400, detail="No model loaded")

    image_data = await file.read()
    try:
        tensor = load_image_from_bytes(image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    config = GenerateConfig(max_length=max_length, temperature=temperature, top_p=top_p)
    session = session_mgr.get_or_create(session_id)
    session.add_message("user", f"[Image] {message}")
    memory.save_message(session_id, "user", f"[Image] {message}")

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, lambda: engine.generate(
            session.to_messages(), config, images=[tensor]
        )
    )

    session.add_message("assistant", result.text)
    memory.save_message(session_id, "assistant", result.text)

    return ChatResponse(text=result.text, elapsed_ms=result.elapsed_ms)


@router.get("/sessions")
def list_sessions():
    sessions = memory.list_sessions(limit=50)
    return {"sessions": sessions}


@router.get("/sessions/{session_id}/messages")
def get_session_messages(session_id: str, limit: int = 100):
    messages = memory.get_messages(session_id, limit=limit)
    return {"session_id": session_id, "messages": messages}


@router.get("/sessions/{session_id}/export")
def export_session(session_id: str):
    exported = memory.export_session(session_id)
    return json.loads(exported)


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    session_mgr.delete(session_id)
    return {"status": "deleted", "session_id": session_id}


ALLOWED_KNOWLEDGE_DIRS = [
    os.path.realpath(r"D:\AISpace\Workspace"),
    os.path.realpath(r"D:\AISpace\Models"),
    os.path.realpath(r"D:\AISpace\Tools"),
]

ALLOWED_UPLOAD_EXTENSIONS = {".txt", ".md", ".pdf", ".json", ".csv"}


@router.post("/knowledge/add", response_model=KnowledgeDocInfo)
def knowledge_add(req: KnowledgeAddRequest):
    resolved = os.path.realpath(req.file_path)
    if not any(Path(resolved).is_relative_to(Path(d)) for d in ALLOWED_KNOWLEDGE_DIRS):
        raise HTTPException(status_code=403, detail="Access denied: path outside allowed directories")
    try:
        result = knowledge.add_file(resolved)
        return KnowledgeDocInfo(id=result['doc_id'], filename=result['filename'], chunk_count=result['chunk_count'], filepath=resolved, imported_at=result['imported_at'])
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")


MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

@router.post("/knowledge/upload", response_model=KnowledgeDocInfo)
async def knowledge_upload(file: UploadFile = File(...)):
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File extension '{ext}' not allowed")
    if file.size and file.size > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large: {file.size} bytes (max: {MAX_UPLOAD_SIZE})")
    dest = UPLOADS_DIR / (uuid.uuid4().hex[:8] + '_' + safe_name)
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large: {len(content)} bytes (max: {MAX_UPLOAD_SIZE})")
    dest.write_bytes(content)
    result = knowledge.add_file(dest)
    if dest.exists():
        dest.unlink()
    return KnowledgeDocInfo(id=result['doc_id'], filename=result['filename'], chunk_count=result['chunk_count'], filepath=str(dest), imported_at=result['imported_at'])


@router.post("/knowledge/search", response_model=KnowledgeResponse)
def knowledge_search(req: KnowledgeSearchRequest):
    results = knowledge.search(req.query, limit=req.limit)
    context = knowledge.get_context(req.query)
    return KnowledgeResponse(results=results, context=context)


@router.get("/knowledge")
def knowledge_list():
    docs = knowledge.list_documents()
    return {"documents": docs}


@router.delete("/knowledge/{doc_id}")
def knowledge_delete(doc_id: int):
    knowledge.delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}
