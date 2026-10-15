"""A minimal Ollama-compatible HTTP server, so e2e tests exercise the real `ollama` client over real HTTP.

Implements just what chat_agent uses: GET /api/tags (ollama.list) and POST /api/chat, both streaming
(NDJSON) and non-streaming. Responses are scripted with the same `Turn` objects as the in-process fake.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tests.fakes import DEFAULT_TEST_MODEL, RECAP_TEXT, Turn, is_recap_request


class FakeOllamaServer:
    def __init__(self, models: list[str] | None = None):
        self.models = models if models is not None else [DEFAULT_TEST_MODEL]
        self.turns: deque[Turn] = deque()
        self.recap: str = RECAP_TEXT
        self.chat_requests: list[dict] = []
        self.recap_requests: list[dict] = []
        # One record per streamed reply: did the client read it to the end, or hang up early?
        self.streams: list[dict] = []
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler_class())
        self._httpd.daemon_threads = True
        self._httpd.handle_error = lambda *_a: None  # client resets are expected (Esc, app killed)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def host(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def script(self, *turns: Turn) -> "FakeOllamaServer":
        with self._lock:
            self.turns.extend(turns)
        return self

    def start(self) -> "FakeOllamaServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def wait_for_requests(self, n: int, timeout: float = 10.0) -> list[dict]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.chat_requests) >= n:
                    return list(self.chat_requests)
            time.sleep(0.02)
        raise AssertionError(f"expected {n} chat requests, got {len(self.chat_requests)}")

    # ---------------------------------------------------------------------------------------------
    def _handler_class(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):  # keep test output clean
                pass

            def _json(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._json(200, {"models": [
                        {"name": m, "model": m, "modified_at": "2026-01-01T00:00:00Z", "size": 1,
                         "digest": "0" * 64, "details": {}} for m in server.models]})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                if self.path != "/api/chat":
                    self._json(404, {"error": "not found"})
                    return
                length = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(length) or b"{}")
                messages = req.get("messages", [])
                model = req.get("model", DEFAULT_TEST_MODEL)

                if not req.get("stream", True) and is_recap_request(messages):
                    with server._lock:
                        server.recap_requests.append(req)
                    self._json(200, _chunk(model, server.recap, done=True))
                    return

                with server._lock:
                    server.chat_requests.append(req)
                    turn = server.turns.popleft() if server.turns else None
                if turn is None:
                    self._json(500, {"error": "fake server: script exhausted"})
                    return
                if turn.error is not None and turn.error_after_chunks is None:
                    self._json(500, {"error": str(turn.error)})
                    return

                if not req.get("stream", True):
                    self._json(200, _chunk(model, turn.content, done=True,
                                           tool_calls=[c.to_json() for c in turn.tool_calls]))
                    return

                record = {"started": time.monotonic(), "finished": None, "client_hung_up": False,
                          "chunks_sent": 0}
                with server._lock:
                    server.streams.append(record)
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    for i, piece in enumerate(turn.content_chunks()):
                        if turn.chunk_delay:
                            time.sleep(turn.chunk_delay)
                        if turn.error_after_chunks is not None and i >= turn.error_after_chunks:
                            break
                        self._send_chunk(_chunk(model, piece))
                        record["chunks_sent"] += 1
                    if turn.error_after_chunks is not None:
                        self._send_chunk({"error": str(turn.error)})
                    else:
                        if turn.chunk_delay and not turn.content:
                            time.sleep(turn.chunk_delay)
                        if turn.split_tool_calls:
                            for c in turn.tool_calls:
                                self._send_chunk(_chunk(model, "", tool_calls=[c.to_json()]))
                            final_calls = None
                        else:
                            final_calls = [c.to_json() for c in turn.tool_calls] or None
                        self._send_chunk(_chunk(model, "", done=True, tool_calls=final_calls))
                    self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    record["client_hung_up"] = True  # e.g. the user pressed Esc and the client closed
                record["finished"] = time.monotonic()

            def _send_chunk(self, payload: dict) -> None:
                data = (json.dumps(payload) + "\n").encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

        return Handler


def _chunk(model: str, content: str, done: bool = False, tool_calls: list | None = None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    payload = {"model": model, "created_at": datetime.now(timezone.utc).isoformat(),
               "message": message, "done": done}
    if done:
        payload["done_reason"] = "stop"
    return payload
