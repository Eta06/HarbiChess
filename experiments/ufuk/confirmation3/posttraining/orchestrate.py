"""Owner-invoked remote confirmation3 stages. No SSH, retries, or checkpoint selection."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

SOURCE = "2312652dc52a894e9726f48321117cf114270355"
SEEDS = (20261205, 20261206)
END = 1791170400
REG_SHA = "36a055c69e03fddcc170e1bb4514c24a72c978f0983d3e04d2169afe0e8cfad6"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def publish(p, value):
    with Path(p).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def wait_ready(paths, deadline):
    while not all(Path(p).is_file() for p in paths):
        if time.time() >= deadline:
            raise TimeoutError("Original stage deadline while waiting")
        time.sleep(1)


def wait_json(path, deadline):
    while True:
        try:
            return read(path)
        except (FileNotFoundError, json.JSONDecodeError):
            if time.time() >= deadline:
                raise TimeoutError("Original receipt deadline exhausted") from None
            time.sleep(1)


def stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def quiescent():
    # Never terminate preexisting processes. Fail if training/auditing/arena still active.
    text = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    forbidden = (
        "harbichess.training.torch_fullgame_run",
        "a100-mc-native-terminal-audit-v3-incremental.py",
        "harbichess.evaluation.portable_arena",
        "a100-mc-fresh-cli-replay.py",
        "harbichess.training.torch_online_run",
        "harbichess.training.torch_search_run",
        "a100-mc-whole-training-controller",
        "a100-mc-incremental-audit-controller",
        "a100-mc-baseline-strength-controller",
        "a100-mc-final-two-arm-controller",
        "devqualification",
        "dev-qualification",
        "pytest",
    )
    assert not any(any(token in line for token in forbidden) for line in text.splitlines()), (
        "Not quiescent"
    )
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    )
    assert not gpu.strip(), "GPU compute context still active"


def wait_quiescent(deadline):
    while True:
        try:
            quiescent()
            return
        except AssertionError:
            if time.time() >= deadline:
                raise TimeoutError("Processes or GPU context did not quiesce") from None
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    assert sha(args.config) == args.config_sha256
    cfg = read(args.config)
    assert cfg["source_commit"] == SOURCE and cfg["qualification_ledger_slot"] == 3
    assert [row["seed"] for row in cfg["seeds"]] == list(SEEDS)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "plan-only",
                    "stages": [
                        "both original40+audit gates",
                        "two serialized exact600replays",
                        "eligibility",
                        "CUDAparity",
                        "quiescent latency",
                        "two parallel final controllers",
                        "wait actual transported MLX receipt",
                        "strength analysis",
                        "all-gates",
                    ],
                    "no_jobs_launched": True,
                }
            )
        )
        return
    repo, helpers, root = map(Path, (cfg["repo"], cfg["helpers"], cfg["root"]))
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() == SOURCE
    )
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).strip()
    assert sha(cfg["registration"]) == REG_SHA
    reg = read(cfg["registration"])
    assert reg["qualification_ledger_slot"] == 3 and reg["source_commit"] == SOURCE
    for name, digest in reg["helper_sha256"].items():
        assert sha(helpers / name) == digest
    expected_runtime = cfg["expected_runtime"]
    assert sha(cfg["python"]) == expected_runtime["python_executable_sha256"]
    assert os.uname().nodename == expected_runtime["hostname"]
    assert (
        Path("/proc/sys/kernel/random/boot_id").read_text().strip() == expected_runtime["boot_id"]
    )
    assert (
        subprocess.check_output(
            [cfg["python"], "-c", "import torch; print(torch.__version__)"],
            text=True,
            timeout=30,
        ).strip()
        == expected_runtime["torch_version"]
    )
    root.mkdir(parents=True, exist_ok=False)
    publish(root / "runtime-witness.json", expected_runtime)
    owned = []
    env = {
        **os.environ,
        "PYTHONPATH": str(repo / "src"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Owner interruption {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def launch(name, script, arguments, seconds, gpu=True, started=None):
        started = time.time() if started is None else started
        assert started + seconds < END, "Insufficient lease for full original stage ceiling"
        deadline = started + seconds
        command = [cfg["python"], str(helpers / script), *map(str, arguments)]
        publish(
            root / (name + "-command.json"),
            {
                "argv": command,
                "started_epoch": started,
                "deadline_epoch": deadline,
                "final_runtime_witness_sha256": sha(root / "final-arms-runtime-witness.json")
                if name.startswith("final-")
                else None,
            },
        )
        out = (root / (name + ".stdout")).open("x")
        err = (root / (name + ".stderr")).open("x")
        process = subprocess.Popen(
            command,
            cwd=repo,
            env={**env, **({} if gpu else {"CUDA_VISIBLE_DEVICES": ""})},
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        owned.append((process, out, err))
        return name, process, deadline

    def finish(task):
        name, process, deadline = task
        while process.poll() is None:
            if time.time() >= deadline:
                stop(process)
                raise TimeoutError(name)
            time.sleep(0.5)
        publish(
            root / (name + "-process-result.json"),
            {
                "returncode": process.returncode,
                "finished_epoch": time.time(),
                "deadline_epoch": deadline,
            },
        )
        assert process.returncode == 0 and time.time() <= deadline, name

    try:
        eligible_rows = []
        audited_inventory = []
        for row in cfg["seeds"]:
            run = Path(row["run"])
            audit_manifest = read(row["audit_manifest"])
            assert sha(row["audit_manifest"]) == row["audit_manifest_sha256"]
            assert audit_manifest["source_commit"] == SOURCE
            assert (
                audit_manifest["inputs"]["protocol"]["sha256"]
                == reg["fixed_lineage"]["original_training_protocol_sha256"]
            )
            training_deadline = audit_manifest["original_training_deadline_epoch"]
            assert training_deadline == audit_manifest["original_training_started_epoch"] + 18000
            result = wait_json(run.parent / "result.json", training_deadline)
            assert result["status"] == (
                "completed-fixedCURRENT40-awaiting-independent-integrity-"
                "latency-portability-strength"
            )
            assert result["finished_epoch"] <= training_deadline
            audit = wait_json(row["full_audit"], min(training_deadline + 3000, END))
            assert audit["status"] == "pass-all41native-all40epochs-independent-own-data"
            assert audit["finished_epoch"] <= min(training_deadline + 3000, END)
            assert audit["source_commit"] == SOURCE and audit["seed"] == row["seed"]
            assert audit["manifest_sha256"] == row["audit_manifest_sha256"]
            controller = wait_json(
                Path(row["full_audit"]).parent / "result.json",
                min(training_deadline + 3000, END),
            )
            assert controller["status"] == "completed-independent-all41native-all40own-data-audit"
            assert controller["full_audit_sha256"] == sha(row["full_audit"])
            assert controller["finished_epoch"] <= min(training_deadline + 3000, END)
            assert controller["returncode"] == 0
            receipt_shas = audit["immutable_percheckpoint_audit_receipt_sha256"]
            assert len(receipt_shas) == 41
            for path, digest in receipt_shas.items():
                assert sha(path) == digest
            audited_inventory.append(
                {
                    row["full_audit"]: sha(row["full_audit"]),
                    **receipt_shas,
                    str(Path(row["full_audit"]).parent / "result.json"): sha(
                        Path(row["full_audit"]).parent / "result.json"
                    ),
                }
            )
            assert (
                audit["audited_native_checkpoints"] == 41
                and audit["independently_replayed_fresh_transitions"] == 1310720
            )
            assert (
                audit["audit_sha256"]
                == reg["helper_sha256"]["a100-mc-native-terminal-audit-v3-incremental.py"]
            )
            model = run / "checkpoints/epoch-00000040/model.safetensors"
            assert sha(model) == result["fixed_current40_sha256"]
            eligible_rows.append(
                {
                    "seed": row["seed"],
                    "run": str(run),
                    "candidate": str(model),
                    "full_native_audit": {
                        "path": row["full_audit"],
                        "sha256": sha(row["full_audit"]),
                    },
                }
            )
        wait_quiescent(min(time.time() + 60, END))
        for row, eligible in zip(cfg["seeds"], eligible_rows, strict=True):
            replay = Path(row["run"]).parent / "confirmation3-replay-audit"
            assert not replay.exists()
            replay_started = time.time()
            finish(
                launch(
                    "replay-" + str(row["seed"]),
                    "a100-mc-fresh-cli-replay.py",
                    [
                        "--run",
                        row["run"],
                        "--manifest",
                        row["audit_manifest"],
                        "--audit-run",
                        replay,
                        "--deadline-epoch",
                        replay_started + 600,
                    ],
                    600,
                    started=replay_started,
                )
            )
            eligible["fresh_replay_audit"] = {
                "path": str(replay / "audit-result.json"),
                "sha256": sha(replay / "audit-result.json"),
            }
        eligibility = root / "eligibility.json"
        publish(
            root / "eligibility-manifest.json",
            {"source_commit": SOURCE, "qualification_ledger_slot": 3, "seeds": eligible_rows},
        )
        finish(
            launch(
                "eligibility",
                "a100-mc-fixed-candidate-eligibility-v2-neural.py",
                ["--manifest", root / "eligibility-manifest.json", "--output", eligibility],
                120,
            )
        )
        models = [
            {
                "seed": r["seed"],
                "candidate": r["candidate"],
                "candidate_sha256": sha(r["candidate"]),
            }
            for r in eligible_rows
        ]
        publish(root / "portable-model-manifest.json", {"seeds": models})
        publish(cfg["exchange_manifest"], {"seeds": models})
        finish(
            launch(
                "cuda-parity",
                "a100-mc-final-portable-parity.py",
                [
                    "--manifest",
                    root / "portable-model-manifest.json",
                    "--probes",
                    cfg["probes"],
                    "--backend",
                    "cuda",
                    "--output",
                    root / "cuda-parity.json",
                ],
                120,
            )
        )
        wait_quiescent(min(time.time() + 60, END))
        finish(
            launch(
                "latency",
                "a100-mc-strength-latency.py",
                [
                    "--initial",
                    cfg["initial"],
                    "--final-20261205",
                    models[0]["candidate"],
                    "--final-20261206",
                    models[1]["candidate"],
                    "--probes",
                    cfg["probes"],
                    "--quiescent",
                    "--output",
                    root / "latency.json",
                ],
                180,
                False,
            )
        )
        # Fresh factual runtime snapshot bound to actual final-controller launch clocks.
        assert sha(cfg["python"]) == expected_runtime["python_executable_sha256"]
        assert os.uname().nodename == expected_runtime["hostname"]
        assert (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            == expected_runtime["boot_id"]
        )
        observed_torch = subprocess.check_output(
            [cfg["python"], "-c", "import torch; print(torch.__version__)"],
            text=True,
            timeout=30,
        ).strip()
        assert observed_torch == expected_runtime["torch_version"]
        final_runtime = {
            **expected_runtime,
            "observed_epoch": time.time(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "cpu_model": next(
                line
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "producer_source_commit": SOURCE,
            "helper_source_commit": reg["helper_source_commit"],
            "registration_sha256": REG_SHA,
            "torch_cpu_threads": 1,
            "cuda_visible_for_arenas": "disabled",
            "stockfish_sha256": sha(cfg["stockfish"]),
        }
        publish(root / "final-arms-runtime-witness.json", final_runtime)
        tasks = []
        for row, model in zip(cfg["seeds"], models, strict=True):
            assert sha(row["baseline_arena"]) == row["baseline_arena_sha256"]
            assert sha(row["book"]) == reg["strength"]["frozen_books"][str(row["seed"])]["sha256"]
            final = root / ("final-" + str(row["seed"]))
            tasks.append(
                launch(
                    "final-" + str(row["seed"]),
                    "a100-mc-final-two-arm-controller.py",
                    [
                        "--repo",
                        repo,
                        "--python",
                        cfg["python"],
                        "--candidate",
                        model["candidate"],
                        "--initial",
                        cfg["initial"],
                        "--book",
                        row["book"],
                        "--stockfish",
                        cfg["stockfish"],
                        "--run-dir",
                        final,
                        "--baseline-arena",
                        row["baseline_arena"],
                        "--baseline-arena-sha256",
                        row["baseline_arena_sha256"],
                        "--eligible-receipt",
                        eligibility,
                        "--eligible-receipt-sha256",
                        sha(eligibility),
                        "--source-commit",
                        SOURCE,
                        "--seed",
                        row["seed"],
                        "--deadline-epoch",
                        END,
                        "--arm-wall-seconds",
                        3600,
                    ],
                    7300,
                    False,
                )
            )
        for task in tasks:
            finish(task)
        # Owner transports exact models and local MLX receipt; no SSH/credential code.
        wait_ready([cfg["mlx_receipt"], cfg["mlx_receipt_sha_file"]], END)
        mlx = wait_json(cfg["mlx_receipt"], END)
        while True:
            mlx_sha = Path(cfg["mlx_receipt_sha_file"]).read_text().strip()
            if len(mlx_sha) == 64 and all(c in "0123456789abcdef" for c in mlx_sha):
                break
            if time.time() >= END:
                raise TimeoutError("MLX receipt checksum publication incomplete")
            time.sleep(1)
        assert sha(cfg["mlx_receipt"]) == mlx_sha
        assert (
            mlx["backend"] == "mlx"
            and mlx["status"] == "pass-sameweights-full-and-legalmasked18fullhistories"
        )
        assert {r["seed"]: r["weights_sha256"] for r in mlx["models"]} == {
            r["seed"]: r["candidate_sha256"] for r in models
        }
        strength_rows = []
        for row, model in zip(cfg["seeds"], models, strict=True):
            final = root / ("final-" + str(row["seed"]))
            strength_rows.append(
                {
                    "seed": row["seed"],
                    "book": row["book"],
                    "direct": str(final / "direct/arena.json"),
                    "final_sf": str(final / "final_sf/arena.json"),
                    "initial_sf": row["baseline_arena"],
                    "initial_sha256": sha(cfg["initial"]),
                    "final_sha256": model["candidate_sha256"],
                }
            )
        publish(
            root / "strength-manifest.json",
            {"source_commit": SOURCE, "qualification_ledger_slot": 3, "seeds": strength_rows},
        )
        finish(
            launch(
                "analysis",
                "a100-mc-strength-analysis.py",
                [
                    "--manifest",
                    root / "strength-manifest.json",
                    "--output",
                    root / "strength-analysis.json",
                ],
                300,
                False,
            )
        )
        for immutable in audited_inventory:
            for path, digest in immutable.items():
                assert sha(path) == digest, "Immutable audit evidence changed"
        paths = {
            "fixed_candidate_eligibility": eligibility,
            "strength_analysis": root / "strength-analysis.json",
            "strength_manifest": root / "strength-manifest.json",
            "latency": root / "latency.json",
            "cuda_parity": root / "cuda-parity.json",
            "mlx_cpu_parity": Path(cfg["mlx_receipt"]),
        }
        publish(
            root / "join-manifest.json",
            {name: {"path": str(p), "sha256": sha(p)} for name, p in paths.items()},
        )
        finish(
            launch(
                "all-gates",
                "a100-mc-all-gates-qualification.py",
                [
                    "--manifest",
                    root / "join-manifest.json",
                    "--output",
                    root / "qualification.json",
                ],
                120,
                False,
            )
        )
        publish(
            root / "orchestration-result.json",
            {
                "status": "all-helper-gates-pass",
                "qualification_sha256": sha(root / "qualification.json"),
                "finished_epoch": time.time(),
            },
        )
    except BaseException as exc:
        publish(
            root / "orchestration-failure.json",
            {
                "status": "failed-or-incomplete-preserved-no-retry",
                "error": repr(exc),
                "finished_epoch": time.time(),
            },
        )
        raise
    finally:
        for process, out, err in owned:
            stop(process)
            out.close()
            err.close()


if __name__ == "__main__":
    main()
