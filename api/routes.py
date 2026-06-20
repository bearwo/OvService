from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form

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
from fastapi.responses import StreamingResponse

from api.schemas import (
    ChatRequest,
    ChatResponse,
    ModelsResponse,
    ModelInfo,
    HealthResponse,
    TaskResponse,
    KnowledgeAddRequest,
    KnowledgeSearchRequest,
    KnowledgeResponse,
    KnowledgeDocInfo,
)
from api.session import SessionManager
from api.task_queue import TaskQueue, TaskStatus
from config import MAX_CONCURRENT, UPLOADS_DIR
from core.base import GenerateConfig
from core.engine import ModelEngine
from features.memory import ConversationMemory
from features.knowledge import KnowledgeBase
from features.image import load_image_from_bytes

router = APIRouter()
session_mgr = SessionManager()
task_queue = TaskQueue(max_concurrent=MAX_CONCURRENT)
memory = ConversationMemory()
knowledge = KnowledgeBase()


def _get_engine() -> ModelEngine:
    return ModelEngine()


@router.get("/health", response_model=HealthResponse)
def health():
    engine = _get_engine()
    active = engine.active()
    database_ok = False
    model_loaded = False
    try:
        import sqlite3
        from config import DB_PATH
        conn = sqlite3.connect(str(DB_PATH), timeout=2)
        conn.execute("SELECT 1")
        conn.close()
        database_ok = True
    except Exception:
        database_ok = False
    if active:
        model_loaded = active._loaded
    return HealthResponse(
        active_model=active.name if active else None,
        database_ok=database_ok,
        model_loaded=model_loaded,
    )


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
    for msg in req.messages:
        session.add_message(msg["role"], msg["content"])
        memory.save_message(session_id, msg["role"], msg["content"])

    knowledge_context = ""
    if req.use_knowledge and req.messages:
        last_msg = req.messages[-1].get("content", "")
        query = req.knowledge_query or last_msg
        knowledge_context = knowledge.get_context(query)

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, lambda: engine.generate(session.to_messages(), config, knowledge_context=knowledge_context)
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
    for msg in req.messages:
        session.add_message(msg["role"], msg["content"])
        memory.save_message(session_id, msg["role"], msg["content"])

    knowledge_context = ""
    if req.use_knowledge and req.messages:
        last_msg = req.messages[-1].get("content", "")
        query = req.knowledge_query or last_msg
        knowledge_context = knowledge.get_context(query)

    async def event_generator():
        loop = asyncio.get_running_loop()
        full_response = []

        def run_stream():
            for token in engine.generate_stream(
                session.to_messages(), config, knowledge_context=knowledge_context
            ):
                full_response.append(token)
                yield token

        gen = run_stream()

        def next_token():
            try:
                return next(gen)
            except StopIteration:
                return None

        while True:
            token = await loop.run_in_executor(None, next_token)
            if token is None:
                break
            yield f"data: {json.dumps({'token': token})}\n\n"

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

    config = GenerateConfig(max_length=max_length, temperature=temperature)
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
    if not any(resolved.startswith(d) for d in ALLOWED_KNOWLEDGE_DIRS):
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
    dest = UPLOADS_DIR / safe_name
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large: {len(content)} bytes (max: {MAX_UPLOAD_SIZE})")
    dest.write_bytes(content)
    result = knowledge.add_file(dest)
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


@router.post("/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    if task_queue.cancel(task_id):
        return {"status": "cancelled", "task_id": task_id}
    raise HTTPException(status_code=404, detail="Task not found or not cancellable")
