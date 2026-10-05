"""Fixed CPU E8 posttraining only: fail-closed receipts and hard06 owned phases."""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import time
from pathlib import Path

SOURCE = "3be5b87db27a0fbde83464e7ea7157f0d9a76ae4"
SEEDS = (20261925, 20261926)
END = 1791180000
PROBE = "1089fd0cca308c24bb040a840352d0ad18aa85a7bdcd17608133456bf08bed9f"
BUSY = re.compile(
    r"portable_arena|torch_\w+_run|_full_audit\.py|_fresh_cli_replay\.py|_parity\.py|qualify|profile_e1|pytest|benchmark"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def wait_json(path, deadline):
    while True:
        try:
            # Complete historical receipt remains valid after its own phase deadline.
            value = read(path)
            return value
        except (FileNotFoundError, json.JSONDecodeError):
            if time.time() >= deadline:
                raise TimeoutError("waiting immutable publication: " + str(path)) from None
            time.sleep(0.25)


def busy_processes():
    found = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        if int(path.parent.name) == os.getpid():
            continue
        try:
            cmd = path.read_bytes().replace(b"\x00", b" ").decode(errors="replace")
        except (FileNotFoundError, ProcessLookupError):
            continue
        if BUSY.search(cmd):
            found.append(dict(pid=int(path.parent.name), command=cmd))
    return found


def quiet(deadline):
    while True:
        busy = busy_processes()
        if not busy:
            return {"status": "quiescent-localCPU", "observed_epoch": time.time(), "busy": []}
        if time.time() >= deadline:
            raise TimeoutError(json.dumps({"quiescence_failure": busy}))
        time.sleep(0.25)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--config-sha256", required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    assert sha(a.config) == a.config_sha256
    cfg = read(a.config)
    assert cfg["schema"] == "cpu-posttraining-config-v1" and cfg["source_commit"] == SOURCE
    assert cfg["seeds"] == list(SEEDS) and cfg["hard_end_epoch"] == END
    helpers = Path(cfg["helpers"])
    Q = read(cfg["qualification_config"]["path"])
    assert sha(cfg["qualification_config"]["path"]) == cfg["qualification_config"]["sha256"]
    assert Q["native_schema"] == "torch-search-acting-native-cpu-v3" and Q["seeds"] == list(SEEDS)
    for name, digest in Q["helper_sha256"].items():
        assert sha(helpers / name) == digest
    assert sha(cfg["probes"]) == PROBE
    assert (
        sha(cfg["stockfish"]) == "0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19"
    )
    assert sha(cfg["owned_runtime"]["path"]) == cfg["owned_runtime"]["sha256"]
    runtime_spec = importlib.util.spec_from_file_location(
        "owned_runtime", cfg["owned_runtime"]["path"]
    )
    runtime = importlib.util.module_from_spec(runtime_spec)
    runtime_spec.loader.exec_module(runtime)
    assert sha(cfg["cohort_clock"]["path"]) == cfg["cohort_clock"]["sha256"]
    clock = read(cfg["cohort_clock"]["path"])
    assert clock["hard_end_epoch"] == END
    assert [row["seed"] for row in cfg["candidates"]] == list(SEEDS)
    for row in cfg["candidates"]:
        assert sha(row["baseline_arena"]["path"]) == row["baseline_arena"]["sha256"]
        assert sha(row["book"]) == Q["books_sha256"][str(row["seed"])]
        baseline = read(row["baseline_result"]["path"])
        assert sha(row["baseline_result"]["path"]) == row["baseline_result"]["sha256"]
        assert (
            baseline["status"]
            == "completed-frozen-baseline-outcomes-withheld-from-training-decisions"
        )
        assert baseline["finished_epoch"] <= baseline["original_deadline_epoch"]
    # A missing future blind result is permitted at stage time, never at final-arena admission.
    blind = cfg["blind_overlap"]
    assert blind["sha256"] != "ROOT_UNKNOWN"
    if not a.execute:
        print(json.dumps({"status": "static-validation-no-clock-no-process"}))
        return
    root = Path(cfg["root"])
    root.mkdir(parents=True, exist_ok=False)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONPATH=str(Path(cfg["checkout"]) / "src") + ":" + str(helpers),
    )
    owners = []
    result = {
        "schema": "cpu-posttraining-owned-result-v1",
        "status": "incomplete",
        "source_commit": SOURCE,
        "config_sha256": sha(a.config),
        "hard_end_epoch": END,
        "started_epoch": time.time(),
    }

    def run(name, helper, arguments, ceiling, qualified=True):
        deadline = min(time.time() + ceiling, END - 1)
        assert time.time() < deadline
        cmd = [cfg["python"], str(helpers / helper)]
        if qualified:
            cmd += [
                "--qualification-config",
                cfg["qualification_config"]["path"],
                "--qualification-config-sha256",
                cfg["qualification_config"]["sha256"],
            ]
        cmd += arguments
        with (
            (root / (name + ".stdout.log")).open("x") as out,
            (root / (name + ".stderr.log")).open("x") as err,
        ):
            child = subprocess.Popen(
                cmd,
                cwd=cfg["checkout"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
        identity = runtime.ticks(child.pid)
        assert identity is not None
        tracked = {child.pid: identity[0]}
        owners.append((name, child, identity[0], deadline, tracked))
        runtime.publish(
            root / (name + ".owner.json"),
            {
                "command": cmd,
                "pid": child.pid,
                "startticks": identity[0],
                "pgid": child.pid,
                "deadline_epoch": deadline,
                "started_epoch": time.time(),
            },
        )
        while child.poll() is None:
            tracked.update(runtime.track_group(child, identity[0]))
            if time.time() >= deadline:
                raise TimeoutError(name + " original owned phase ceiling")
            time.sleep(0.25)
        assert child.returncode == 0, name + " returned " + str(child.returncode)
        assert time.time() <= deadline
        runtime.publish(
            root / (name + ".process-result.json"),
            {
                "status": "completed",
                "returncode": 0,
                "finished_epoch": time.time(),
                "deadline_epoch": deadline,
            },
        )

    try:
        completed = wait_json(cfg["cohort_result"], clock["audit_deadline"])
        assert completed["status"] == "completed-both-fixedCPU8-fullaudits-and-independent-replays"
        assert completed["original_first_epoch"] == clock["original_first_epoch"]
        assert all(row["returncode"] == 0 for row in completed["owners"])
        assert completed["finished_epoch"] <= clock["audit_deadline"]
        overlap = wait_json(blind["path"], END - 1)
        assert sha(blind["path"]) == blind["sha256"]
        for key, value in blind["required_values"].items():
            assert overlap[key] == value
        runtime.publish(
            root / "blind-overlap-binding.json",
            {"receipt_sha256": sha(blind["path"]), "required_values": blind["required_values"]},
        )
        eligibility = {
            "source_commit": SOURCE,
            "qualification_ledger_slot": 8,
            "fixed_epochs": 8,
            "audit_helper_sha256": sha(helpers / "cpu_contingency_full_audit.py"),
            "replay_helper_sha256": sha(helpers / "cpu_contingency_fresh_cli_replay.py"),
            "seeds": [],
        }
        for row in cfg["candidates"]:
            full = Path(row["audit_root"]) / "full-audit-result.json"
            replay = Path(row["replay_run"]) / "audit-result.json"
            eligibility["seeds"].append(
                {
                    "seed": row["seed"],
                    "run": row["run"],
                    "full_search_acting_audit": {"path": str(full), "sha256": sha(full)},
                    "fresh_search_acting_replay": {"path": str(replay), "sha256": sha(replay)},
                }
            )
        em = root / "eligibility-manifest.json"
        runtime.publish(em, eligibility)
        eligible = root / "eligibility.json"
        run(
            "eligibility",
            "cpu_contingency_eligibility.py",
            ["--manifest", str(em), "--output", str(eligible)],
            120,
            qualified=False,
        )
        parity = {
            "source_commit": SOURCE,
            "fixed_epochs": 8,
            "qualification_ledger_slot": 8,
            "seeds": [],
        }
        for row in cfg["candidates"]:
            model = Path(row["run"]) / "checkpoints/epoch-00000008/model.safetensors"
            manifest = Path(row["audit_root"]) / "audit-manifest.json"
            parity["seeds"].append(
                {
                    "seed": row["seed"],
                    "candidate": str(model),
                    "candidate_sha256": sha(model),
                    "native_audit_manifest": str(manifest),
                    "native_audit_manifest_sha256": sha(manifest),
                }
            )
        pm = root / "parity-manifest.json"
        runtime.publish(pm, parity)
        for backend in ("cpu", "mlx"):
            run(
                "parity-" + backend,
                "cpu_contingency_parity.py",
                [
                    "--manifest",
                    str(pm),
                    "--probes",
                    cfg["probes"],
                    "--backend",
                    backend,
                    "--output",
                    str(root / ("parity-" + backend + ".json")),
                ],
                120,
            )
        runtime.publish(root / "quiescence.json", quiet(min(time.time() + 60, END - 1)))
        latency = root / "latency.json"
        args = [
            "--initial",
            cfg["initial"],
            "--probes",
            cfg["probes"],
            "--output",
            str(latency),
            "--quiescent",
        ]
        for row in parity["seeds"]:
            args += ["--final-" + str(row["seed"]), row["candidate"]]
        run("latency", "cpu_contingency_latency.py", args, 180)
        # Blind evidence is reverified immediately before heldout final outcomes are generated.
        assert sha(blind["path"]) == blind["sha256"]
        for row in cfg["candidates"]:
            seed = row["seed"]
            final_root = root / f"final-{seed}"
            run(
                f"final-{seed}",
                "cpu_contingency_final_arms.py",
                [
                    "--python",
                    cfg["python"],
                    "--repo",
                    cfg["checkout"],
                    "--candidate",
                    str(Path(row["run"]) / "checkpoints/epoch-00000008/model.safetensors"),
                    "--initial",
                    cfg["initial"],
                    "--book",
                    row["book"],
                    "--stockfish",
                    cfg["stockfish"],
                    "--run-dir",
                    str(final_root),
                    "--baseline-arena",
                    row["baseline_arena"]["path"],
                    "--baseline-arena-sha256",
                    row["baseline_arena"]["sha256"],
                    "--eligible-receipt",
                    str(eligible),
                    "--eligible-receipt-sha256",
                    sha(eligible),
                    "--source-commit",
                    SOURCE,
                    "--seed",
                    str(seed),
                    "--deadline-epoch",
                    str(END - 1),
                    "--arm-wall-seconds",
                    "3600",
                ],
                7300,
            )
        strength = {
            "source_commit": SOURCE,
            "fixed_epochs": 8,
            "qualification_ledger_slot": 8,
            "seeds": [],
        }
        for row, candidate in zip(cfg["candidates"], parity["seeds"], strict=True):
            seed = row["seed"]
            strength["seeds"].append(
                {
                    "seed": seed,
                    "book": row["book"],
                    "initial_sha256": sha(cfg["initial"]),
                    "final_sha256": candidate["candidate_sha256"],
                    "direct": str(root / f"final-{seed}/direct/arena.json"),
                    "final_sf": str(root / f"final-{seed}/final_sf/arena.json"),
                    "initial_sf": row["baseline_arena"]["path"],
                }
            )
        sm = root / "strength-manifest.json"
        runtime.publish(sm, strength)
        run(
            "analysis",
            "cpu_contingency_strength_analysis.py",
            ["--manifest", str(sm), "--output", str(root / "strength-analysis.json")],
            120,
        )
        gates = {"source_commit": SOURCE, "fixed_epochs": 8, "qualification_ledger_slot": 8}
        for key, path in [
            ("fixed_candidate_eligibility", eligible),
            ("strength_analysis", root / "strength-analysis.json"),
            ("strength_manifest", sm),
            ("latency", latency),
            ("cpu_export_parity", root / "parity-cpu.json"),
            ("mlx_cpu_parity", root / "parity-mlx.json"),
        ]:
            gates[key] = {"path": str(path), "sha256": sha(path)}
        gm = root / "all-gates-manifest.json"
        runtime.publish(gm, gates)
        run(
            "all-gates",
            "cpu_contingency_all_gates.py",
            ["--manifest", str(gm), "--output", str(root / "all-gates.json")],
            120,
        )
        result["status"] = "all-requiredCPU-native-parity-latency-strength-gates-actually-pass"
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        failures = []
        for name, child, identity, _deadline, tracked in owners:
            try:
                runtime.stop_owned(child, identity, tracked)
            except BaseException as exc:
                failures.append({"name": name, "error": repr(exc)})
        if failures:
            result.update(status="incomplete-owned-cleanup-failure", cleanup_errors=failures)
        result.update(
            finished_epoch=time.time(),
            owners=[
                {
                    "name": name,
                    "pid": child.pid,
                    "startticks": identity,
                    "returncode": child.returncode,
                    "deadline_epoch": deadline,
                }
                for name, child, identity, deadline, _ in owners
            ],
        )
        runtime.publish(root / "result.json", result)


if __name__ == "__main__":
    main()
