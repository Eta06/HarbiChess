import json
import threading
import urllib.error
import urllib.request
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import pytest

import harbichess.dashboard.server as dashboard_server
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
            assert b"HarbiChess Dashboard" in response.read()
        with urllib.request.urlopen(f"{base}/healthz", timeout=2) as response:
            assert json.load(response) == {"status": "ok"}
        with urllib.request.urlopen(f"{base}/api/snapshot", timeout=2) as response:
            assert json.load(response)["active_games"] == 64
        with urllib.request.urlopen(f"{base}/assets/app.css", timeout=2) as response:
            assert response.headers["Content-Type"] == "text/css; charset=utf-8"
        with urllib.request.urlopen(f"{base}/assets/app.js", timeout=2) as response:
            assert response.headers["Content-Type"] == "text/javascript; charset=utf-8"
        with urllib.request.urlopen(f"{base}/api/events", timeout=2) as response:
            assert json.loads(response.readline().removeprefix(b"data: "))["demo"]
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(f"{base}/missing", timeout=2)
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_main_writes_demo_and_closes_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeServer:
        server_port = 8765
        closed = False

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def server_close(self) -> None:
            self.closed = True

    fake_server = FakeServer()
    monkeypatch.setattr(dashboard_server, "create_server", lambda *args: fake_server)
    state_path = tmp_path / "state.json"

    assert dashboard_server.main(["--demo", "--state", str(state_path)]) == 0
    assert SnapshotStore(state_path).read().demo
    assert fake_server.closed


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
