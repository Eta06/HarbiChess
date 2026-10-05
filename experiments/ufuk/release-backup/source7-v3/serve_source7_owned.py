"""Owned fixed18089 SHA allowlist server and authorized HTTP2 QuickTunnel."""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

END = 1791180000
SERVER_HELPER = Path(
    "/content/harbichess-fullgame-method2-inputs/"
    "formal45-release-reviewed-v3-20261005/formal45_release_backup.py"
)
SERVER_SHA = "5936b67303c178ab2b8ba97fc50742e3bf6c63274d7c2f792bdd1a4c4469a4f8"


def serve(package):
    assert hashlib.sha256(SERVER_HELPER.read_bytes()).hexdigest() == SERVER_SHA
    spec = importlib.util.spec_from_file_location("frozen_allowlist_server", SERVER_HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    result = json.loads((package / "pack-result.json").read_text())
    path = Path(result["archive"])
    assert path.parent == package and not path.is_symlink()
    assert helper.sha256_file(path) == result["sha256"]
    assert path.name == f"sha256-{result['sha256']}.tar.gz"
    path.chmod(0o600)  # Newly generated archive only; original qualification files untouched.
    st = path.stat()
    mapping = {"/" + path.name: (path, st.st_size, result["sha256"], st.st_dev, st.st_ino)}
    server = helper.AllowlistHTTPServer(("127.0.0.1", 18089), helper.ReadOnlyHandler, mapping, END)
    try:
        while time.time() < END:
            server.handle_request()
    finally:
        server.server_close()


def terminate(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--root", type=Path)
    p.add_argument("--serve-only", action="store_true")
    a = p.parse_args()
    if a.serve_only:
        serve(a.package)
    else:
        owner = json.loads((a.package.parent / "owner-result.json").read_text())
        assert owner["status"] == "pass" and owner["finished_epoch"] <= owner["deadline_epoch"]
        assert time.time() < owner["deadline_epoch"]
        a.root.mkdir(exist_ok=False)
        owned = []
        try:

            def launch(name, argv):
                out = (a.root / (name + ".stdout")).open("x")
                err = (a.root / (name + ".stderr")).open("x")
                child = subprocess.Popen(
                    argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True
                )
                owned.append((child, out, err))
                return child

            server = launch(
                "server", ["python3", __file__, "--package", str(a.package), "--serve-only"]
            )
            tunnel = launch(
                "tunnel",
                [
                    "cloudflared",
                    "tunnel",
                    "--url",
                    "http://127.0.0.1:18089",
                    "--no-autoupdate",
                    "--protocol",
                    "http2",
                ],
            )
            deadline = min(time.time() + 90, owner["deadline_epoch"])
            url = None
            while time.time() < deadline:
                assert server.poll() is None and tunnel.poll() is None
                matches = re.findall(
                    r"https://[a-z0-9-]+\.trycloudflare\.com",
                    (a.root / "tunnel.stderr").read_text(errors="replace"),
                )
                if matches:
                    url = matches[0]
                    break
                time.sleep(0.2)
            if url is None:
                raise RuntimeError("original-pack600-public-relay-bootstrap-failed")
            result = {
                "schema": "source7-one-archive-readonly18089-owned-relay-v1",
                "status": "relay-ready-public-verification-pending",
                "base_url": url,
                "server_pid": server.pid,
                "tunnel_pid": tunnel.pid,
                "port": 18089,
                "expires_epoch": END,
                "observed_epoch": time.time(),
                "original_pack_deadline_epoch": owner["deadline_epoch"],
            }
            (a.root / "relay-ready.json").write_text(json.dumps(result, indent=2) + "\n")
            while time.time() < END:
                if server.poll() is not None or tunnel.poll() is not None:
                    raise RuntimeError("owned-public-relay-child-stopped")
                time.sleep(1)
        except Exception as exc:
            (a.root / "relay-failure.json").write_text(
                json.dumps(
                    {
                        "status": "failed-preserved",
                        "error_type": type(exc).__name__,
                        "finished_epoch": time.time(),
                    }
                )
                + "\n"
            )
            raise
        finally:
            for child, out, err in owned:
                terminate(child)
                out.close()
                err.close()
