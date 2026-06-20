from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    messages: list[dict] = Field(..., description="Message history")
    max_length: int = Field(2048, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    use_knowledge: bool = Field(False, description="Search knowledge base for context")
    knowledge_query: str = Field("", description="Custom query for knowledge search")


class ChatResponse(BaseModel):
    text: str
    elapsed_ms: float = 0.0


class TaskResponse(BaseModel):
    task_id: str
    status: str


class ModelInfo(BaseModel):
    name: str
    loaded: bool
    device: str
    model_path: str
    load_time_ms: float = 0.0


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str = "ok"
    active_model: str | None = None
    database_ok: bool = True
    model_loaded: bool = False


class KnowledgeAddRequest(BaseModel):
    file_path: str = Field(..., description="Path to file to add")


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(5, ge=1, le=20)


class KnowledgeResponse(BaseModel):
    results: list[dict]
    context: str = ""


class KnowledgeDocInfo(BaseModel):
    id: int
    filename: str
    filepath: str
    chunk_count: int
    imported_at: float
