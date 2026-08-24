import json
import threading
import urllib.request
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from harbichess.dashboard.server import (
    STATIC_ROOT,
    DashboardHTTPServer,
    create_server,
    handler_for,
)
from harbichess.dashboard.state import SnapshotStore, demo_snapshot


def test_dashboard_serves_ui_health_and_snapshot(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "state.json")
    store.write_atomic(demo_snapshot())
    server = create_server("127.0.0.1", 0, store, STATIC_ROOT)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(f"{base}/", timeout=2) as response:
            assert response.status == 200
            assert b"Training Command" in response.read()
        with urllib.request.urlopen(f"{base}/healthz", timeout=2) as response:
            assert json.load(response) == {"status": "ok"}
        with urllib.request.urlopen(f"{base}/api/snapshot", timeout=2) as response:
            assert json.load(response)["active_games"] == 64
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_suppresses_routine_client_disconnects(tmp_path: Path) -> None:
    server = DashboardHTTPServer(
        ("127.0.0.1", 0),
        handler_for(SnapshotStore(tmp_path / "unused")),
    )
    try:
        with redirect_stderr(StringIO()) as stderr:
            try:
                raise ConnectionResetError
            except ConnectionResetError:
                server.handle_error(object(), ("127.0.0.1", 1))
        assert stderr.getvalue() == ""
    finally:
        server.server_close()
