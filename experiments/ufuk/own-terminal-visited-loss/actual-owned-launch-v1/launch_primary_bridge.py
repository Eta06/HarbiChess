"""Fetch immutable admitted8 binding and launch qualified PRIMARY MLXCPU bridge once."""

import argparse
import base64
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/HarbiChess")
WORK = Path("/workspace/work/harbichess/source8-actual-qualification")
HELPERS = ROOT / "experiments/ufuk/own-terminal-visited-loss/frozen-helper-v1"
QREMOTE = (
    "/content/harbichess-fullgame-method2-inputs/formal8-production/frozen/strength-bindings.json"
)
BASE = "/content/harbichess-fullgame-method2-inputs/formal8-production"
END = 1791180000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "validated-static-primary8-bridge-plan-no-SSH-no-processes",
                    "source_commit": "3be5b87db27a0fbde83464e7ea7157f0d9a76ae4",
                    "remote_binding": QREMOTE,
                    "remote_manifest": BASE + "/method8-portable-exchange-manifest.json",
                    "remote_receipt": BASE + "/method8-final-mlx-cpu-parity.json",
                }
            )
        )
        return
    sys.path.insert(0, "/workspace/work/harbichess/a100")
    from ssh_colab_access import call

    WORK.mkdir(parents=True, exist_ok=True)
    assert time.time() < END
    code = (
        "import base64,hashlib,json;from pathlib import Path;"
        f"p=Path({QREMOTE!r});b=p.read_bytes();"
        'print(json.dumps({"sha256":hashlib.sha256(b).hexdigest(),'
        '"body":base64.b64encode(b).decode()}))'
    )
    r = call("python3 -c " + shlex.quote(code), capture_output=True, timeout=60)
    assert r.returncode == 0
    packet = json.loads(r.stdout)
    body = base64.b64decode(packet["body"], validate=True)
    assert hashlib.sha256(body).hexdigest() == packet["sha256"]
    q = json.loads(body)
    assert q["source_commit"] == "3be5b87db27a0fbde83464e7ea7157f0d9a76ae4"
    assert q["qualification_ledger_slot"] == 8 and q["fixed_epochs"] == 8
    assert q["status"] == "frozen-before-formal-execution"
    bind = WORK / "actual-formal8-strength-bindings.json"
    with bind.open("xb") as f:
        f.write(body)
    script = HELPERS / "owner8_mlx_transport.py"
    mirror = WORK / "eligible-bridge-mirror"
    config = {
        "repo": "/workspace/work/harbichess/own8-source-3be5",
        "python": str(ROOT / ".venv/bin/python"),
        "helper": str(HELPERS / "own8_parity.py"),
        "probes": str(ROOT / "docs/research/UFUK-width-latency-probes-20261004.json"),
        "qualification_config": str(bind),
        "qualification_config_sha256": packet["sha256"],
        "root": str(WORK / "actual-owned-primary-mlx-bridge"),
        "mirror": str(mirror),
        "expected_runs": {
            str(s): f"/content/harbichess-runs/visited-loss-method8-seed-{s}/run"
            for s in (20261825, 20261826)
        },
        "remote_manifest": BASE + "/method8-portable-exchange-manifest.json",
        "remote_receipt": BASE + "/method8-final-mlx-cpu-parity.json",
        "owner_transport_argv": [
            str(ROOT / ".venv/bin/python"),
            str(script),
            "--mirror",
            str(mirror),
        ],
        "owner_transport_script": str(script),
        "owner_transport_script_sha256": sha(script),
    }
    cp = WORK / "actual-primary-mlx-bridge-config.json"
    with cp.open("x") as f:
        json.dump(config, f, sort_keys=True, indent=2)
        f.write("\n")
    argv = [
        str(ROOT / ".venv/bin/python"),
        str(HELPERS / "own8_mlx_receipt_bridge.py"),
        "--config",
        str(cp),
        "--config-sha256",
        sha(cp),
    ]
    subprocess.run(argv, cwd=ROOT, check=True, capture_output=True, timeout=15)
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONOPTIMIZE": "0",
    }
    with (
        (WORK / "actual-primary-mlx-bridge.stdout.log").open("x") as out,
        (WORK / "actual-primary-mlx-bridge.stderr.log").open("x") as err,
    ):
        p = subprocess.Popen(
            [*argv, "--execute"],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
    stat = Path(f"/proc/{p.pid}/stat").read_text().rsplit(")", 1)[1].split()
    record = {
        "schema": "actual-owned-primary-MLXCPU-bridge-launch-v1",
        "pid": p.pid,
        "start_ticks": int(stat[19]),
        "pgid": p.pid,
        "argv": [*argv, "--execute"],
        "observed_epoch": time.time(),
        "hard_deadline_epoch": END,
        "qualification_config_sha256": packet["sha256"],
        "bridge_config_sha256": sha(cp),
        "adapter_sha256": sha(script),
        "source_commit": q["source_commit"],
    }
    with (WORK / "actual-primary-mlx-bridge-launch.json").open("x") as f:
        json.dump(record, f, sort_keys=True, indent=2)
        f.write("\n")
    print(
        json.dumps(
            {"status": "actual-primary-bridge-launched", "pid": p.pid, "source": q["source_commit"]}
        )
    )


if __name__ == "__main__":
    main()
