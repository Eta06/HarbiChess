"""Standalone low-overhead HTTP/SSE server for HarbiChess telemetry."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from harbichess.dashboard.state import SnapshotStore, demo_snapshot

STATIC_ROOT = Path(__file__).with_name("static")
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}


class DashboardHTTPServer(ThreadingHTTPServer):
    """Threaded server that ignores routine browser disconnects."""

    daemon_threads = True

    def handle_error(self, request: object, client_address: object) -> None:
        if isinstance(sys.exception(), BrokenPipeError | ConnectionResetError):
            return
        super().handle_error(request, client_address)


def handler_for(store: SnapshotStore, static_root: Path = STATIC_ROOT):
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/api/snapshot":
                self._send_json(store.read().to_json())
            elif path == "/api/events":
                self._send_events()
            elif path == "/healthz":
                self._send_json(json.dumps({"status": "ok"}))
            elif path in ("/", "/index.html"):
                self._send_file(static_root / "index.html")
            elif path in ("/styles.css", "/app.js"):
                self._send_file(static_root / path.removeprefix("/"))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _send_json(self, payload: str) -> None:
            encoded = payload.encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_file(self, path: Path) -> None:
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            encoded = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", CONTENT_TYPES[path.suffix])
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_events(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                while True:
                    payload = store.read().to_json()
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(1)
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, format: str, *args: object) -> None:
            return

    return DashboardHandler


def create_server(
    host: str,
    port: int,
    store: SnapshotStore,
    static_root: Path = STATIC_ROOT,
) -> ThreadingHTTPServer:
    return DashboardHTTPServer((host, port), handler_for(store, static_root))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state", type=Path, default=Path("artifacts/dashboard/state.json"))
    parser.add_argument("--demo", action="store_true")
    arguments = parser.parse_args(argv)

    store = SnapshotStore(arguments.state)
    if arguments.demo:
        store.write_atomic(demo_snapshot())
    server = create_server(arguments.host, arguments.port, store)
    print(f"HarbiChess dashboard: http://{arguments.host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
