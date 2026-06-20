from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import setup_openvino, CHAT_MODEL, DEFAULT_DEVICE, API_HOST, API_PORT
from core.engine import ModelEngine
from adapters.chat import ChatAdapter
from api.routes import router
from api.openai_compat import router as openai_router

import uvicorn


def init_model():
    engine = ModelEngine()
    adapter = ChatAdapter(CHAT_MODEL, DEFAULT_DEVICE)
    engine.register(adapter)
    engine.set_active("chat")
    print(f"Loading {adapter.name} from {adapter.model_path}...")
    adapter.load()
    print(f"Model loaded in {adapter.status().load_time_ms:.0f}ms")


def create_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="OvService", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # development only — restrict in production
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(openai_router)
    return app


def main():
    setup_openvino()
    init_model()

    app = create_app()
    print(f"Starting OvService API on {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")


if __name__ == "__main__":
    main()
