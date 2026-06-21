from __future__ import annotations

import http.client
import json
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

WEBUI_DIR = Path(__file__).parent
STATIC_DIR = WEBUI_DIR / "static"
TEMPLATE_DIR = WEBUI_DIR / "templates"

sys.path.insert(0, str(WEBUI_DIR.parent))


class WebUIHandler(SimpleHTTPRequestHandler):
    api_host: str = "localhost"
    api_port: int = 8000

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_file(TEMPLATE_DIR / "index.html", "text/html")
        elif self.path.startswith("/static/"):
            file_path = (STATIC_DIR / self.path[8:]).resolve()
            if not file_path.is_relative_to(STATIC_DIR.resolve()):
                self.send_error(403)
                return
            content_type = self._get_content_type(file_path)
            self._serve_file(file_path, content_type)
        elif self.path.startswith("/v1/") or self.path.startswith("/health"):
            self._proxy_request("GET")
        else:
            self.send_error(404)

    def do_POST(self):
        if not self.path.startswith("/static/"):
            self._proxy_request("POST")
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _serve_file(self, file_path: Path, content_type: str):
        if not file_path.exists():
            self.send_error(404)
            return
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _proxy_request(self, method: str):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else None
        is_stream = False
        if body:
            try:
                is_stream = json.loads(body).get("stream", False)
            except Exception:
                pass

        try:
            conn = http.client.HTTPConnection(self.api_host, self.api_port, timeout=600)
            headers = {"Content-Type": "application/json"}
            conn.request(method, self.path, body=body, headers=headers)
            resp = conn.getresponse()

            self.send_response(resp.status)
            self.send_header("Content-Type", "text/event-stream" if is_stream else "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("X-Accel-Buffering", "no")
            if is_stream:
                self.send_header("Connection", "keep-alive")
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            if is_stream:
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    self.wfile.write(line)
                    self.wfile.flush()
            else:
                self.wfile.write(resp.read())

            conn.close()

        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _get_content_type(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        types = {
            ".css": "text/css",
            ".js": "application/javascript",
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".svg": "image/svg+xml",
        }
        return types.get(ext, "application/octet-stream")

    def log_message(self, format, *args):
        pass


def start_api_server():
    import config
    config.setup_openvino()

    from core.engine import ModelEngine
    from core.base import GenerateConfig
    from adapters.chat import ChatAdapter

    engine = ModelEngine()
    adapter = ChatAdapter(config.CHAT_MODEL, config.DEFAULT_DEVICE)
    engine.register(adapter)
    engine.set_active("chat")
    print(f"Loading model from {adapter.model_path}...")
    adapter.load()
    print(f"Model loaded in {adapter._load_time_ms:.0f}ms")
    print("Warming up model...")
    adapter.generate([{"role": "user", "content": "hi"}], GenerateConfig(max_length=1))
    print("Warmup complete.")

    from app import create_app
    import uvicorn

    app = create_app()
    uv_config = uvicorn.Config(app, host=config.API_HOST, port=config.API_PORT, log_level="warning")
    server = uvicorn.Server(uv_config)
    server.run()


def wait_for_api(api_host: str, api_port: int, timeout: int = 300):
    start = time.time()
    while time.time() - start < timeout:
        try:
            conn = http.client.HTTPConnection(api_host, api_port, timeout=2)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            return True
        except Exception:
            time.sleep(2)
    return False


def parse_port(arg, default):
    if not arg:
        return default
    arg = arg.strip()
    if "://" in arg:
        from urllib.parse import urlparse
        return int(urlparse(arg).port or default)
    return int(arg)


def main():
    api_port = parse_port(sys.argv[1] if len(sys.argv) > 1 else None, 8000)
    webui_port = parse_port(sys.argv[2] if len(sys.argv) > 2 else None, 3000)

    print("=" * 50)
    print("OvService - Starting all services")
    print("=" * 50)

    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()

    print(f"Waiting for API server on port {api_port}...")
    if wait_for_api("localhost", api_port):
        print(f"API server ready on http://localhost:{api_port}")
    else:
        print("API server failed to start")
        sys.exit(1)

    WebUIHandler.api_host = "localhost"
    WebUIHandler.api_port = api_port
    server = HTTPServer(("0.0.0.0", webui_port), WebUIHandler)
    print(f"Web UI: http://localhost:{webui_port}")
    print("=" * 50)
    print("Press Ctrl+C to stop all services")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
        server.server_close()


if __name__ == "__main__":
    main()
