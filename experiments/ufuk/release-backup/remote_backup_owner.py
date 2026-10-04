"""Owned packaging and temporary public SHA allowlist; existing training is untouched."""

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

END = 1791180000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def stop(child):
    if child and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--helper", type=Path, required=True)
    p.add_argument("--helper-sha256", required=True)
    p.add_argument("--packaging-seconds", type=int, default=1200)
    a = p.parse_args()
    assert sha(a.helper) == a.helper_sha256
    assert a.packaging_seconds == 1200
    a.root.mkdir(exist_ok=False, parents=True)
    first = time.time()
    deadline = first + a.packaging_seconds
    assert deadline < END
    owned = []
    phases = []

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Owned backup interruption {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def launch(name, command):
        out = (a.root / (name + ".stdout")).open("xb")
        err = (a.root / (name + ".stderr")).open("xb")
        child = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True
        )
        owned.append((child, out, err))
        publish(
            a.root / (name + "-owner.json"),
            {
                "pid": child.pid,
                "command": command,
                "started_epoch": time.time(),
                "packaging_deadline_epoch": deadline,
            },
        )
        return child

    def finish(name, child):
        while child.poll() is None:
            if time.time() >= deadline:
                raise TimeoutError("Original1200 backup packaging ceiling")
            time.sleep(0.2)
        phase = {"name": name, "returncode": child.returncode, "finished_epoch": time.time()}
        phases.append(phase)
        assert child.returncode == 0

    try:
        # All remote source files have explicit fixed roots. Never expose private/auth/log files.
        tunnel = launch(
            "public-tunnel",
            [
                "cloudflared",
                "tunnel",
                "--url",
                "http://127.0.0.1:18087",
                "--no-autoupdate",
                "--protocol",
                "http2",
            ],
        )
        url = None
        url_deadline = min(deadline, first + 60)
        while time.time() < url_deadline:
            assert tunnel.poll() is None
            text = (a.root / "public-tunnel.stderr").read_text(errors="replace")
            matches = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", text)
            if matches:
                url = matches[0]
                break
            time.sleep(0.2)
        assert url is not None
        packages = []
        package_root = Path("/content/harbichess-runs/release-packages")
        for seed in (20261205, 20261206):
            before = set(package_root.glob(f"seed-{seed}-through-*"))
            name = "pack-" + str(seed)
            child = launch(
                name,
                ["/usr/bin/python3", str(a.helper), "pack", "--seed", str(seed), "--base-url", url],
            )
            finish(name, child)
            created = set(package_root.glob(f"seed-{seed}-through-*")) - before
            assert len(created) == 1
            directory = created.pop()
            manifest = json.loads((directory / "package.json").read_text())
            assert manifest["seed"] == seed and manifest["max_closed_epoch"] == 40
            assert manifest["source_commit"] == "2312652dc52a894e9726f48321117cf114270355"
            assert len(manifest["archives"]) == 42
            packages.append(directory)
        manifest_path = package_root / "fullgame40-transport-manifest-20261004.json"
        argv = [
            "/usr/bin/python3",
            str(a.helper),
            "manifest",
            "--base-url",
            url,
            "--output",
            str(manifest_path),
        ]
        for package in packages:
            argv += ["--package-dir", str(package)]
        finish("manifest", launch("manifest", argv))
        argv = ["/usr/bin/python3", str(a.helper), "serve", "--deadline-epoch", str(END)]
        for package in packages:
            argv += ["--package-dir", str(package)]
        server = launch("readonly-server", argv)
        manifest = json.loads(manifest_path.read_text())
        assert len(manifest["assets"]) == 84 and manifest["seeds"] == [20261205, 20261206]
        result = {
            "status": "packaged84-full-native-and-input-source-archives-not-yet-Release-backup",
            "started_epoch": first,
            "finished_epoch": time.time(),
            "original_packaging_deadline_epoch": deadline,
            "source_commit": "2312652dc52a894e9726f48321117cf114270355",
            "helper_sha256": a.helper_sha256,
            "controller_sha256": sha(__file__),
            "server_expiry_epoch": END,
            "server_pid": server.pid,
            "tunnel_pid": tunnel.pid,
            "base_url": url,
            "package_dirs": list(map(str, packages)),
            "transport_manifest": str(manifest_path),
            "transport_manifest_sha256": sha(manifest_path),
            "asset_count": 84,
            "phases": phases,
            "scope": "No model promotion; claim Release backup only after verified readback.",
        }
        publish(a.root / "packaging-result.json", result)
        while time.time() < END:
            assert server.poll() is None and tunnel.poll() is None
            time.sleep(min(5, END - time.time()))
    except BaseException as exc:
        publish(
            a.root / "failure.json",
            {
                "status": "backup-transport-failed-preserved",
                "error_type": type(exc).__name__,
                "finished_epoch": time.time(),
                "original_packaging_deadline_epoch": deadline,
                "phases": phases,
            },
        )
        raise
    finally:
        for child, out, err in owned:
            stop(child)
            out.close()
            err.close()


if __name__ == "__main__":
    main()
