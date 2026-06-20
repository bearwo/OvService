from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
config.setup_openvino()

from core.engine import ModelEngine
from adapters.chat import ChatAdapter
from api.routes import router
from api.openai_compat import router as openai_router

import os
import uvicorn


def init_model():
    engine = ModelEngine()
    adapter = ChatAdapter(config.CHAT_MODEL, config.DEFAULT_DEVICE)
    engine.register(adapter)
    engine.set_active("chat")
    print(f"Loading {adapter.name} from {adapter.model_path}...")
    adapter.load()
    print(f"Model loaded in {adapter.status().load_time_ms:.0f}ms")


def create_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="OvService", version="1.0.0")
    origins = ["*"] if os.environ.get("OVSERVICE_CORS_ALLOW_ALL", "1") == "1" else ["http://localhost:3000", "http://127.0.0.1:3000"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(openai_router)
    return app


def main():
    init_model()

    app = create_app()
    print(f"Starting OvService API on {config.API_HOST}:{config.API_PORT}")
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, log_level="info")


if __name__ == "__main__":
    main()
