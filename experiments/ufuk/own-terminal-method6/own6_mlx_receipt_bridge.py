"""Owner-invoked bridge: fixed OWN models -> actual local MLXCPU -> immutable receipt.

Transport is an explicitly supplied owner adapter, not embedded SSH/auth code.
Importing this module or running its pure tests executes no transport or inference.
"""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from own6_audit_support import check_source, publish, sha
from own6_strength_config import SEEDS, validate_config

END = 1791180000
PROBE_SHA = "1089fd0cca308c24bb040a840352d0ad18aa85a7bdcd17608133456bf08bed9f"


def localize_manifest(manifest, config, mirror, expected_runs):
    assert manifest["qualification_ledger_slot"] == 6
    assert manifest["source_commit"] == config["source_commit"]
    assert manifest["fixed_epochs"] == config["fixed_epochs"]
    assert [row["seed"] for row in manifest["seeds"]] == list(SEEDS)
    localized = {**manifest, "seeds": []}
    for row in manifest["seeds"]:
        expected = (
            Path(expected_runs[str(row["seed"])])
            / "checkpoints"
            / (f"epoch-{config['fixed_epochs']:08d}")
            / "model.safetensors"
        )
        remote = Path(row["candidate"])
        assert remote == expected and remote.is_relative_to("/content/harbichess-runs")
        local = Path(mirror) / remote.relative_to("/content")
        assert local.is_file() and not local.is_symlink()
        assert sha(local) == row["candidate_sha256"]
        localized["seeds"].append({**row, "candidate": str(local.resolve())})
    return localized


def parity_command(python, helper, bindings, bindings_sha, manifest, probes, output):
    return [
        str(python),
        str(helper),
        "--qualification-config",
        str(bindings),
        "--qualification-config-sha256",
        bindings_sha,
        "--manifest",
        str(manifest),
        "--probes",
        str(probes),
        "--backend",
        "mlx",
        "--output",
        str(output),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--config-sha256", required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    assert sha(a.config) == a.config_sha256
    c = json.loads(a.config.read_text())
    bindings = Path(c["qualification_config"])
    assert sha(bindings) == c["qualification_config_sha256"]
    q = validate_config(json.loads(bindings.read_text()))
    repo, helper, probes = (Path(c[k]) for k in ("repo", "helper", "probes"))
    assert sha(helper) == q["helper_sha256"]["own6_parity.py"]
    assert sha(probes) == PROBE_SHA
    assert c["remote_receipt"].endswith("/method6-final-mlx-cpu-parity.json")
    assert "method6" in c["remote_manifest"]
    check_source(repo, q["source_commit"])
    adapter = c["owner_transport_argv"]
    assert (
        adapter
        and sha(c["owner_transport_script"]) == c["owner_transport_script_sha256"]
    )
    assert c["owner_transport_script"] in adapter
    if not a.execute:
        print(json.dumps({"status": "bridge-plan-validated-no-transport-or-MLX"}))
        return
    root = Path(c["root"])
    root.mkdir(parents=True, exist_ok=False)
    process = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Owned MLX bridge interruption {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        packet = None
        while time.time() < END:
            response = subprocess.run(
                [*adapter, "read", c["remote_manifest"]],
                capture_output=True,
                timeout=min(60, END - time.time()),
                check=False,
            )
            if response.returncode == 0 and response.stdout:
                try:
                    packet = json.loads(response.stdout)
                    break
                except json.JSONDecodeError:
                    pass
            time.sleep(min(5, max(0, END - time.time())))
        if packet is None:
            raise TimeoutError(
                "Harddeadline before fixed BOTH eligible OWN model manifest"
            )
        publish(root / "remote-model-manifest.json", packet)
        # Semantic manifest failures are never masked as publication-inflight.
        assert packet["source_commit"] == q["source_commit"]
        assert packet["fixed_epochs"] == q["fixed_epochs"]
        assert [row["seed"] for row in packet["seeds"]] == list(SEEDS)
        for row in packet["seeds"]:
            remote = Path(row["candidate"])
            expected = (
                Path(c["expected_runs"][str(row["seed"])])
                / "checkpoints"
                / (f"epoch-{q['fixed_epochs']:08d}")
                / "model.safetensors"
            )
            assert remote == expected and remote.is_relative_to(
                "/content/harbichess-runs"
            )
            local = Path(c["mirror"]) / remote.relative_to("/content")
            while not local.is_file() and time.time() < END:
                time.sleep(min(5, max(0, END - time.time())))
        localized = localize_manifest(packet, q, c["mirror"], c["expected_runs"])
        manifest = root / "local-model-manifest.json"
        publish(manifest, localized)
        output = root / "method6-final-mlx-cpu-parity.json"
        argv = parity_command(
            c["python"],
            helper,
            bindings,
            c["qualification_config_sha256"],
            manifest,
            probes,
            output,
        )
        started = time.time()
        deadline = started + 180
        assert deadline < END
        publish(
            root / "command.json",
            {
                "argv": argv,
                "started_epoch": started,
                "absolute_deadline_epoch": deadline,
                "source_commit": q["source_commit"],
                "fixed_epochs": q["fixed_epochs"],
            },
        )
        env = {
            **os.environ,
            "PYTHONPATH": str(repo / "src"),
            "PYTHONOPTIMIZE": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        with (
            (root / "stdout.log").open("x") as out,
            (root / "stderr.log").open("x") as err,
        ):
            process = subprocess.Popen(
                argv,
                cwd=repo,
                env=env,
                stdout=out,
                stderr=err,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            while process.poll() is None:
                if time.time() >= deadline:
                    raise TimeoutError("Original180 MLX qualification deadline")
                time.sleep(0.5)
        assert process.returncode == 0 and time.time() <= deadline
        receipt = json.loads(output.read_text())
        assert (
            receipt["status"] == "pass-sameweights-full-and-legalmasked18fullhistories"
        )
        assert (
            receipt["backend"] == "mlx"
            and receipt["source_commit"] == q["source_commit"]
        )
        assert (
            receipt["qualification_ledger_slot"] == 6
            and receipt["fixed_epochs"] == q["fixed_epochs"]
        )
        assert {row["seed"]: row["weights_sha256"] for row in receipt["models"]} == {
            row["seed"]: row["candidate_sha256"] for row in packet["seeds"]
        }
        check_source(repo, q["source_commit"])
        digest = sha(output)
        response = subprocess.run(
            [*adapter, "publish", c["remote_receipt"], str(output), digest],
            capture_output=True,
            timeout=min(120, END - time.time()),
            check=False,
        )
        assert response.returncode == 0
        transferred = json.loads(response.stdout)
        assert (
            transferred["status"]
            == "immutable-published-receipt-and-sha-readback-verified"
        )
        assert (
            transferred["sha256"] == digest
            and transferred["path"] == c["remote_receipt"]
        )
        publish(root / "transport-receipt.json", transferred)
    except BaseException as exc:
        publish(
            root / "failure.json",
            {
                "status": "bridge-failed-preserved-no-retry",
                "error_type": type(exc).__name__,
            },
        )
        raise
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
