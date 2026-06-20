import os
import sys
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONUTF8"] = "1"
os.environ["ONEDNN_VERBOSE"] = "0"
os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = "0"

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

OPENVINO_LIB_PATHS = [
    Path(r"D:\AISpace\Tools\openvino_genai\runtime\bin\intel64\Release"),
    Path(r"D:\AISpace\Tools\openvino_genai\runtime\3rdparty\tbb\bin"),
]

OPENVINO_PYTHON_DIR = Path(r"D:\AISpace\Tools\openvino_genai\python")

MODELS_DIR = Path(r"D:\AISpace\Models")
CHAT_MODEL = MODELS_DIR / "Qwen3.6-35B-A3B-int4-ov"

DATA_DIR = Path(__file__).parent / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "conversations.db"

API_HOST = "0.0.0.0"
API_PORT = 8000

DEFAULT_DEVICE = "GPU"
MAX_CONCURRENT = int(os.environ.get("OVSERVICE_MAX_CONCURRENT", "2"))

MAX_HISTORY_TURNS = 50
MAX_CONTEXT_TOKENS = 0
COMPRESS_CONTEXT_RATIO = 0.33
DB_COMPRESS_MAX_COUNT = 200
DB_COMPRESS_MAX_RATIO = 0.50


def get_model_context_length() -> int:
    global MAX_CONTEXT_TOKENS
    if MAX_CONTEXT_TOKENS > 0:
        return MAX_CONTEXT_TOKENS
    config_path = CHAT_MODEL / "config.json"
    if config_path.exists():
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        tc = cfg.get("text_config", cfg)
        mpe = tc.get("max_position_embeddings", 0)
        if mpe > 0:
            MAX_CONTEXT_TOKENS = mpe
            return mpe
    MAX_CONTEXT_TOKENS = 32768
    return MAX_CONTEXT_TOKENS


def setup_openvino():
    lib_paths_str = ";".join(str(p) for p in OPENVINO_LIB_PATHS if p.exists())
    os.environ["OPENVINO_LIB_PATHS"] = lib_paths_str

    for p in OPENVINO_LIB_PATHS:
        if p.exists():
            os.add_dll_directory(str(p))

    py_dir = str(OPENVINO_PYTHON_DIR)
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)


setup_openvino()
