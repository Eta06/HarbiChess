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


SOURCE5 = "4515a7c0dda3b4f9615c2fc78a47c872ab14699d"
REGISTRATION5 = "a9ae975e6ecb486f964ee0514d98fb4bc9d50d3b949ecabd0cf1e441bd75d271"
QUALIFICATION5 = "5fa34fdc69a49b78fecae17ee9ade86da994c548bbda81473d86c454c6cc733a"


def _read_pinned_receipt(path, expected_sha):
    assert isinstance(expected_sha, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha)
    path = Path(path)
    assert not path.is_symlink()
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected_sha
    return json.loads(raw)


def validate_failed5_barrier(barrier, terminal, release):
    """No readiness/latency/eligibility alias: explicit terminal failure plus ownership release."""
    assert barrier["kind"] == "failed5-terminal-owned-release-v1"
    assert barrier["source_commit"] == terminal["source_commit"] == SOURCE5
    assert barrier["original_registration_sha256"] == REGISTRATION5
    assert terminal["original_qualification_config_sha256"] == QUALIFICATION5
    assert terminal["fixed_epochs"] == 8
    assert terminal["no_strength_qualification_or_model_promotion"] is True
    assert terminal["method5_replay_retry_or_budget_reset"] is False
    assert terminal["immutable_source_and_native_inputs_untouched"] is True
    assert terminal["original_training_started_epoch"] == 1791156456.0081534
    assert (
        terminal["schema"]
        == barrier["terminal_schema"]
        == ("own5-incomplete-original-freshCLI600-terminal-v1")
    )
    assert (
        terminal["status"]
        == barrier["terminal_status"]
        == ("INCOMPLETE-original-freshCLI-replay-deadline-exhausted")
    )
    assert (
        release["schema"]
        == barrier["release_schema"]
        == ("own5-incomplete-owned-compute-terminated-witness-v1")
    )
    assert release["all_registered_owned_groups_and_identities_terminated"] is True
    assert release["method5_ready_latency_final_receipts_fabricated"] is False
    assert release["prior_failure_and_current_native_prefixes_preserved"] is True
    assert release["terminal_receipt_sha256"] == barrier["terminal_sha256"]
    assert terminal["observed_epoch"] <= release["observed_epoch"] <= END
    assert release["before_inventory_sha256"] == barrier["before_inventory_sha256"]
    assert release["after_inventory_sha256"] == barrier["after_inventory_sha256"]
    return {
        "status": "pass-explicit-failed5-terminal-owned-release-no-promotion",
        "source_commit": SOURCE5,
        "fixed_epochs": 8,
        "terminal_sha256": barrier["terminal_sha256"],
        "release_sha256": barrier["release_sha256"],
        "finished_epoch": release["observed_epoch"],
        "genuine_latency_claim": False,
    }


def await_backup_barrier(barrier, deadline):
    assert barrier["kind"] in ("failed5-terminal-owned-release-v1", "genuine5-latency-v1")
    while True:
        try:
            if barrier["kind"] == "failed5-terminal-owned-release-v1":
                terminal = _read_pinned_receipt(
                    barrier["terminal_path"], barrier["terminal_sha256"]
                )
                release = _read_pinned_receipt(barrier["release_path"], barrier["release_sha256"])
                for phase in ("before", "after"):
                    inventory = _read_pinned_receipt(
                        release[phase + "_inventory"], release[phase + "_inventory_sha256"]
                    )
                    validate_owned_inventory(barrier, inventory)
                return validate_failed5_barrier(barrier, terminal, release)
            latency = _read_pinned_receipt(barrier["latency_result"], barrier["latency_sha256"])
            process = _read_pinned_receipt(barrier["process_result"], barrier["process_sha256"])
            assert process["returncode"] == 0
            assert process["finished_epoch"] <= process["deadline_epoch"] <= END
            assert latency["source_commit"] == SOURCE5
            assert latency["fixed_epochs"] == 8 and latency["qualification_ledger_slot"] == 5
            assert latency["script_sha256"] == barrier["latency_script_sha256"]
            assert latency["quiescent_declared"] is True
            for seed in (20261525, 20261526):
                final = Path(
                    f"/content/harbichess-runs/search-acting-method5-seed-{seed}/run/"
                    "checkpoints/epoch-00000008/model.safetensors"
                )
                assert latency["weights_sha256"][str(seed)] == sha(final)
            return {
                "status": "pass-genuine5-latency-completion-no-gate-selection",
                "source_commit": SOURCE5,
                "latency_result_sha256": barrier["latency_sha256"],
                "latency_process_sha256": barrier["process_sha256"],
                "finished_epoch": process["finished_epoch"],
            }
        except (FileNotFoundError, json.JSONDecodeError):
            if time.time() >= deadline:
                raise TimeoutError("Original1200 backup ceiling includes barrier wait") from None
            time.sleep(0.5)


def validate_owned_inventory(barrier, inventory):
    assert inventory["schema"] == "own5-terminal-owned-process-inventory-v1"
    assert inventory["source_commit"] == SOURCE5
    assert inventory["qualification_ledger_slot"] == 5
    assert inventory["terminal_receipt_sha256"] == barrier["terminal_sha256"]
    assert inventory["remaining_owned_pid_startticks"] == []
    assert inventory["remaining_live_known_method5_compute_commands"] == []
    assert inventory["observed_epoch"] <= END


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--helper", type=Path, required=True)
    p.add_argument("--helper-sha256", required=True)
    p.add_argument("--packaging-seconds", type=int, default=1200)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--spec-sha256", required=True)
    a = p.parse_args()
    assert sha(a.spec) == a.spec_sha256
    spec = json.loads(a.spec.read_text())
    assert spec["schema"] == "formal45-release-backup-spec-v1"
    assert spec["status"] == "frozen-reviewed-artifact-allowlist"
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
        barrier = spec["backup_barrier"]
        receipt = await_backup_barrier(barrier, deadline)
        publish(a.root / "verified-backup-scheduling-barrier.json", receipt)
        # All remote source files have explicit fixed roots. Never expose private/auth/log files.
        tunnel = launch(
            "public-tunnel",
            [
                "cloudflared",
                "tunnel",
                "--url",
                "http://127.0.0.1:18088",
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
        package_root = Path("/content/harbichess-runs/release-packages-formal45")
        for seed in (20261425, 20261426, 20261525, 20261526):
            name = "pack-" + str(seed)
            child = launch(
                name,
                [
                    "/usr/bin/python3",
                    str(a.helper),
                    "--spec",
                    str(a.spec),
                    "--spec-sha256",
                    a.spec_sha256,
                    "pack",
                    "--seed",
                    str(seed),
                    "--base-url",
                    url,
                ],
            )
            finish(name, child)
            report = json.loads((a.root / (name + ".stdout")).read_text())
            assert report["status"] == "package-ready" and report["seed"] == seed
            directory = Path(report["package_dir"])
            assert (
                not directory.is_symlink() and directory.parent.resolve() == package_root.resolve()
            )
            assert sha(directory / "package.json") == report["package_manifest_sha256"]
            manifest = json.loads((directory / "package.json").read_text())
            assert manifest["seed"] == seed and manifest["max_closed_epoch"] >= 1
            assert manifest["source_commit"] == spec["jobs"][str(seed)]["source_commit"]
            assert len(manifest["archives"]) == manifest["max_closed_epoch"] + 2
            packages.append(directory)
        manifest_path = package_root / ("formal45-transport-manifest-" + str(int(first)) + ".json")
        argv = [
            "/usr/bin/python3",
            str(a.helper),
            "--spec",
            str(a.spec),
            "--spec-sha256",
            a.spec_sha256,
            "manifest",
            "--base-url",
            url,
            "--output",
            str(manifest_path),
        ]
        for package in packages:
            argv += ["--package-dir", str(package)]
        finish("manifest", launch("manifest", argv))
        argv = [
            "/usr/bin/python3",
            str(a.helper),
            "--spec",
            str(a.spec),
            "--spec-sha256",
            a.spec_sha256,
            "serve",
            "--deadline-epoch",
            str(END),
        ]
        for package in packages:
            argv += ["--package-dir", str(package)]
        server = launch("readonly-server", argv)
        manifest = json.loads(manifest_path.read_text())
        assert 1 <= len(manifest["assets"]) <= 200
        assert {20261425, 20261426, 20261525, 20261526} <= set(manifest["seeds"])
        result = {
            "status": "packaged-formal45-closed-prefixes-not-yet-Release-backup",
            "started_epoch": first,
            "finished_epoch": time.time(),
            "original_packaging_deadline_epoch": deadline,
            "source_commits": sorted({j["source_commit"] for j in spec["jobs"].values()}),
            "spec_sha256": a.spec_sha256,
            "helper_sha256": a.helper_sha256,
            "controller_sha256": sha(__file__),
            "server_expiry_epoch": END,
            "server_pid": server.pid,
            "tunnel_pid": tunnel.pid,
            "base_url": url,
            "package_dirs": list(map(str, packages)),
            "transport_manifest": str(manifest_path),
            "transport_manifest_sha256": sha(manifest_path),
            "asset_count": len(manifest["assets"]),
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
