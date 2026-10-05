"""Root-invoked fixed SEARCH-ACTING-v2 posttraining coordinator; no SSH or outcome selection."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from own6_audit_support import check_source, guard, publish, sha
from own6_strength_config import BOOKS, SEEDS, validate_config
from own6_training_controller import stop

END = 1791180000


def wait_json(path, deadline):
    while True:
        try:
            return json.loads(Path(path).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            if time.time() >= deadline:
                raise TimeoutError(str(path)) from None
            time.sleep(0.5)


def quiescent(deadline):
    forbidden = (
        "torch_ownsearch_run",
        "torch_search_acting_run",
        "search_acting_audit",
        "ownv2_training_controller",
        "ownv2_audit_controller",
        "ownv2_baseline",
        "ownv2_final_arms",
        "profile_search_acting",
        "qualify_search_acting",
        "torch_fullgame_run",
        "torch_online_run",
        "portable_arena",
        "ownv1_training_controller",
        "ownv1_audit_controller",
        "ownv1_baseline",
        "ownv1_final_arms",
        "ownv1_full_audit",
        "own5_training_controller",
        "own5_audit_controller",
        "own5_baseline",
        "own5_final_arms",
        "own5_full_audit",
        "own6_training_controller",
        "own6_audit_controller",
        "own6_baseline",
        "own6_final_arms",
        "baseline-strength-controller",
        "final-two-arm-controller",
        "incremental-audit-controller",
        "whole-training-controller",
        "audit_profile",
        "own6_full_audit",
        "native-terminal-audit",
        "pytest",
        "qualify_cuda",
        "fresh_cli_replay",
        "fresh-cli-replay",
    )
    while True:
        if time.time() >= deadline:
            raise TimeoutError("Fixed60s SEARCH-ACTING-v2 quiescence ceiling")
        busy = []
        for path in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                text = path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if int(path.parent.name) != os.getpid() and any(
                word in text for word in forbidden
            ):
                busy.append(path.parent.name)
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        ).strip()
        if not busy and not gpu:
            return
        time.sleep(0.5)


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
    assert c["source_commit"] == q["source_commit"]
    assert [row["seed"] for row in c["seeds"]] == list(SEEDS)
    helpers, repo, root = (Path(c[k]) for k in ("helpers", "repo", "root"))
    for name, digest in q["helper_sha256"].items():
        assert sha(helpers / name) == digest
    check_source(repo, q["source_commit"])
    assert "cohort" not in c, "Method6 must not mutate/join the original45 cohort"
    assert "previous45_completion_barrier" in c
    assert (
        sha(helpers / "own6_previous_methods_barrier.py")
        == q["helper_sha256"]["own6_previous_methods_barrier.py"]
    )
    from own6_previous_methods_barrier import LATEST_LATENCY_START, wait_previous

    if not a.execute:
        print(
            json.dumps(
                {
                    "status": "validated-plan-only-no-jobs",
                    "fixed_epochs": q["fixed_epochs"],
                }
            )
        )
        return
    root.mkdir(parents=True, exist_ok=False)
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
        raise KeyboardInterrupt(f"Owned coordinator interruption {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def launch(name, helper, arguments, seconds, qualified=False):
        started = time.time()
        deadline = started + seconds
        assert deadline < END
        argv = [c["python"], str(helpers / helper)]
        if qualified:
            argv += [
                "--qualification-config",
                str(bindings),
                "--qualification-config-sha256",
                c["qualification_config_sha256"],
            ]
        argv += list(map(str, arguments))
        publish(
            root / (name + "-command.json"),
            {"argv": argv, "started_epoch": started, "deadline_epoch": deadline},
        )
        out, err = (
            (root / (name + suffix)).open("x") for suffix in (".stdout", ".stderr")
        )
        process = subprocess.Popen(
            argv,
            cwd=repo,
            env=env,
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
            guard(deadline, root)
            time.sleep(0.5)
        publish(
            root / (name + "-process-result.json"),
            {
                "returncode": process.returncode,
                "finished_epoch": time.time(),
                "deadline_epoch": deadline,
            },
        )
        assert process.returncode == 0 and time.time() <= deadline

    def manifest(name, value):
        path = root / name
        publish(
            path,
            {
                "source_commit": q["source_commit"],
                "qualification_ledger_slot": 6,
                "fixed_epochs": q["fixed_epochs"],
                **value,
            },
        )
        return path

    try:
        rows, models = [], []
        cohort_artifacts = {}
        for row in c["seeds"]:
            first = row["original_training_started_epoch"]
            train = wait_json(
                row["training_result"], first + c["whole_training_seconds"]
            )
            assert (
                train["status"]
                == "completed-fixedSEARCH_ACTINGv2-currentE-awaiting-independent-qualification"
            )
            assert train["source_commit"] == q["source_commit"]
            assert train["fixed_epochs"] == q["fixed_epochs"]
            assert train["finished_epoch"] <= first + c["whole_training_seconds"]
            audit = wait_json(
                row["audit_controller_result"], first + c["whole_audit_seconds"]
            )
            assert (
                audit["status"]
                == "completed-independent-search-acting-v2-allnative-data-search-ledgers"
            )
            assert (
                audit["returncode"] == 0
                and audit["source_commit"] == q["source_commit"]
            )
            assert audit["finished_epoch"] <= first + c["whole_audit_seconds"]
            assert sha(row["full_audit"]) == audit["full_audit_sha256"]
            full = wait_json(row["full_audit"], END)
            assert full["audited_native_checkpoints"] == q["fixed_epochs"] + 1
            assert (
                full["independently_replayed_fresh_transitions"]
                == 32768 * q["fixed_epochs"]
            )
            assert (
                full["prescribed_neural_roots_verified"]
                == full["raw_actor_packet_roots_verified"]
                == 108
            )
            receipts = full["immutable_percheckpoint_audit_receipt_sha256"]
            assert len(receipts) == q["fixed_epochs"] + 1
            for path, digest in receipts.items():
                assert sha(path) == digest
                cohort_artifacts[path] = digest
            cohort_artifacts[row["full_audit"]] = sha(row["full_audit"])
            for field in ("training_result", "audit_controller_result"):
                cohort_artifacts[row[field]] = sha(row[field])
            assert sha(row["book"]) == BOOKS[row["seed"]]
        # Both fixed candidates/all data complete BEFORE any final strength observation.
        for row in c["seeds"]:
            run = Path(row["run"])
            native = run / "checkpoints" / f"epoch-{q['fixed_epochs']:08d}"
            candidate = native / "model.safetensors"
            replay = run.parent / "own6-independent-replay"
            clock = time.time()
            finish(
                launch(
                    "replay-" + str(row["seed"]),
                    "own6_fresh_cli_replay.py",
                    [
                        "--run",
                        run,
                        "--manifest",
                        row["audit_manifest"],
                        "--manifest-sha256",
                        sha(row["audit_manifest"]),
                        "--audit-run",
                        replay,
                        "--deadline-epoch",
                        clock + 600,
                    ],
                    600,
                )
            )
            rows.append(
                {
                    "seed": row["seed"],
                    "run": str(run),
                    "full_search_acting_audit": {
                        "path": row["full_audit"],
                        "sha256": sha(row["full_audit"]),
                    },
                    "fresh_search_acting_replay": {
                        "path": str(replay / "audit-result.json"),
                        "sha256": sha(replay / "audit-result.json"),
                    },
                }
            )
            models.append(
                {
                    "seed": row["seed"],
                    "candidate": str(candidate),
                    "candidate_sha256": sha(candidate),
                }
            )
        em = manifest(
            "eligibility-manifest.json",
            {
                "seeds": rows,
                "audit_helper_sha256": sha(helpers / "own6_full_audit.py"),
                "replay_helper_sha256": sha(helpers / "own6_fresh_cli_replay.py"),
            },
        )
        eligible = root / "eligibility.json"
        finish(
            launch(
                "eligibility",
                "own6_eligibility.py",
                ["--manifest", em, "--output", eligible],
                120,
            )
        )
        portable = manifest("portable-model-manifest.json", {"seeds": models})
        publish(Path(c["exchange_manifest"]), json.loads(portable.read_text()))
        finish(
            launch(
                "cuda-parity",
                "own6_parity.py",
                [
                    "--manifest",
                    portable,
                    "--probes",
                    c["probes"],
                    "--backend",
                    "cuda",
                    "--output",
                    root / "cuda-parity.json",
                ],
                120,
                True,
            )
        )
        # Baseline completion metadata only; actual score packets stay unread here.
        for row in c["seeds"]:
            baseline = wait_json(
                Path(row["baseline_root"]) / "result.json",
                row["baseline_deadline_epoch"],
            )
            assert baseline["status"] == (
                "completed-frozen-baseline-outcomes-withheld-from-training-decisions"
            )
            assert baseline["returncode"] == 0
            assert baseline["finished_epoch"] <= row["baseline_deadline_epoch"]
        previous = wait_previous(c, wait_json, sha)
        publish(
            root / "original45-latency-and-all-four-final-owner-completion.json",
            previous,
        )
        quiescent(min(time.time() + 60, LATEST_LATENCY_START))
        assert time.time() < LATEST_LATENCY_START
        finish(
            launch(
                "latency",
                "own6_latency.py",
                [
                    "--initial",
                    c["initial"],
                    "--final-20261625",
                    models[0]["candidate"],
                    "--final-20261626",
                    models[1]["candidate"],
                    "--probes",
                    c["probes"],
                    "--quiescent",
                    "--output",
                    root / "latency.json",
                ],
                180,
                True,
            )
        )
        runtime = {
            "hostname": os.uname().nodename,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "python_executable_sha256": sha(c["python"]),
            "torch_version": subprocess.check_output(
                [c["python"], "-c", "import torch; print(torch.__version__)"],
                text=True,
                timeout=30,
            ).strip(),
        }
        assert runtime == c["expected_A100_runtime"]
        publish(
            root / "final-arms-runtime-owner-receipt.json",
            {
                **runtime,
                "source_commit": q["source_commit"],
                "qualification_config_sha256": c["qualification_config_sha256"],
                "same_runtime_all_six_arms": True,
                "observed_epoch": time.time(),
                "coordinator_sha256": sha(Path(__file__)),
            },
        )
        finals = []
        for row, model in zip(c["seeds"], models, strict=True):
            baseline = wait_json(Path(row["baseline_root"]) / "result.json", END)
            assert (
                baseline["status"]
                == "completed-frozen-baseline-outcomes-withheld-from-training-decisions"
            )
            arena = Path(row["baseline_root"]) / "arena.json"
            assert sha(arena) == baseline["arena_sha256"]
            finals.append(
                launch(
                    "final-" + str(row["seed"]),
                    "own6_final_arms.py",
                    [
                        "--repo",
                        repo,
                        "--python",
                        c["python"],
                        "--candidate",
                        model["candidate"],
                        "--initial",
                        c["initial"],
                        "--book",
                        row["book"],
                        "--stockfish",
                        c["stockfish"],
                        "--run-dir",
                        row["final_root"],
                        "--baseline-arena",
                        arena,
                        "--baseline-arena-sha256",
                        sha(arena),
                        "--eligible-receipt",
                        eligible,
                        "--eligible-receipt-sha256",
                        sha(eligible),
                        "--source-commit",
                        q["source_commit"],
                        "--seed",
                        row["seed"],
                        "--deadline-epoch",
                        END,
                    ],
                    7300,
                    True,
                )
            )
        for task in finals:
            finish(task)
        strength_rows = []
        for row, model in zip(c["seeds"], models, strict=True):
            strength_rows.append(
                {
                    "seed": row["seed"],
                    "book": row["book"],
                    "initial_sha256": sha(c["initial"]),
                    "final_sha256": model["candidate_sha256"],
                    "direct": str(Path(row["final_root"]) / "direct/arena.json"),
                    "final_sf": str(Path(row["final_root"]) / "final_sf/arena.json"),
                    "initial_sf": str(Path(row["baseline_root"]) / "arena.json"),
                }
            )
        sm = manifest("strength-manifest.json", {"seeds": strength_rows})
        analysis = root / "strength-analysis.json"
        finish(
            launch(
                "analysis",
                "own6_strength_analysis.py",
                ["--manifest", sm, "--output", analysis],
                120,
                True,
            )
        )
        mlx = wait_json(c["mlx_receipt"], END)
        while True:
            try:
                checksum = Path(c["mlx_receipt_sha256_file"]).read_text().strip()
            except FileNotFoundError:
                checksum = ""
            if len(checksum) == 64 and all(x in "0123456789abcdef" for x in checksum):
                break
            if time.time() >= END:
                raise TimeoutError("MLX checksum publication incomplete")
            time.sleep(0.5)
        assert sha(c["mlx_receipt"]) == checksum and mlx["backend"] == "mlx"
        entries = {
            "fixed_candidate_eligibility": eligible,
            "strength_analysis": analysis,
            "strength_manifest": sm,
            "latency": root / "latency.json",
            "cuda_parity": root / "cuda-parity.json",
            "mlx_cpu_parity": Path(c["mlx_receipt"]),
        }
        join = manifest(
            "all-gates-manifest.json",
            {k: {"path": str(v), "sha256": sha(v)} for k, v in entries.items()},
        )
        finish(
            launch(
                "all-gates",
                "own6_all_gates.py",
                ["--manifest", join, "--output", root / "all-gates.json"],
                120,
                True,
            )
        )
    except BaseException as error:
        publish(
            root / "failure.json",
            {
                "status": "failed-or-incomplete-preserved-no-retry",
                "error": repr(error),
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
