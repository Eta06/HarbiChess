"""ROOT executable synthetic native bridge qualification, NOT collection or strength."""

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "human-prior-own-v1-synthetic-qualification-registration"
END = 1791448916.685839
RUNTIME = (
    "model.py",
    "native.py",
    "train.py",
    "prove.py",
    "convert.py",
    "contracts.py",
    "collector.py",
    "run_collection.py",
    "metadata_factory.py",
    "audit_collection_six.py",
    "specs.py",
    "parent_bridge.py",
    "admit_parent.py",
    "parent_seal.py",
    "ledger.py",
    "zero_parent.py",
    "initialize.py",
    "prepare_zero.py",
    "prepare_collection.py",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(262144), b""):
            h.update(b)
    return h.hexdigest()


def ref(p):
    p = Path(p).resolve()
    return dict(path=str(p), sha256=sha(p))


def read(path):
    if Path(path).stat().st_size > 8 * 2**20:
        raise ValueError("JSON8MiB ceiling")
    return json.loads(Path(path).read_bytes())


def write(path, obj):
    with Path(path).open("xb") as f:
        f.write(
            json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        )


def clock(r, now):
    if (
        r["schema"] != SCHEMA
        or r["status"] != "registered"
        or r["seed"] not in (20262905, 20262906)
        or r["cpu_core"] not in (1, 3)
        or not r["first"]
        <= now
        < r["deadline"]
        <= min(r["first"] + 600, r["operator_end_epoch"], END)
    ):
        raise ValueError("ROOT registered CPU1/3 original600/operator phase")


def check_sources(r):
    stage = Path(r["stage"]).resolve()
    if set(r["runtime_sha256"]) != set(RUNTIME):
        raise ValueError("all19 frozen runtime helpers required")
    inventory = r["source_inventory"]
    if sha(inventory["path"]) != inventory["sha256"]:
        raise ValueError("frozen inventory changed")
    sealed = read(inventory["path"])["runtime_files"]
    if {n: x["sha256"] for n, x in sealed.items()} != r["runtime_sha256"]:
        raise ValueError("exact frozen19 source inventory required")
    for name, digest in r["runtime_sha256"].items():
        if sha(stage / name) != digest:
            raise ValueError("frozen runtime changed: " + name)
    if r["helper_sha256"] != sha(__file__):
        raise ValueError("qualification helper changed")
    for item in r["parent"].values():
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("original parent input changed")
    return stage


def prepare(spec, output):
    if spec["schema"] != "human-prior-own-v1-synthetic-qualification-build-seal":
        raise ValueError("ROOT observed new qualification build seal")
    r = {
        k: spec[k]
        for k in (
            "seed",
            "cpu_core",
            "first",
            "deadline",
            "operator_end_epoch",
            "stage",
            "parent",
            "source_inventory",
        )
    }
    r.update(
        schema=SCHEMA,
        status="registered",
        helper_sha256=sha(__file__),
        runtime_sha256={n: sha(Path(r["stage"]) / n) for n in RUNTIME},
        output=str(spec["output"]),
        synthetic_only=True,
        real_collection_qualified=False,
    )
    clock(r, time.time())
    check_sources(r)
    if not Path(r["output"]).resolve().is_relative_to("/dev/shm"):
        raise ValueError("qualification output RAM only")
    write(output, r)
    return r


def process(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[1]), fields[19], fields[0]
    except (OSError, ValueError):
        return None


def descendants(root):
    found = {root}
    records = {}
    for _ in range(4):
        for p in Path("/proc").iterdir():
            if not p.name.isdigit():
                continue
            pid = int(p.name)
            info = process(pid)
            if info and (pid in found or info[0] in found):
                found.add(pid)
                records[pid] = info[1]
    return records


def stop_owned(records):
    # Never inspect argv/environment; PID/startticks avoids signaling reused IDs.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, ticks in records.items():
            state = process(pid)
            if state and state[1] == ticks and state[2] != "Z":
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, sig)
        if sig == signal.SIGTERM:
            time.sleep(0.3)


def run(r):
    clock(r, time.time())
    stage = check_sources(r)
    os.sched_setaffinity(0, {r["cpu_core"]})
    sys.path.insert(0, str(stage))
    from parent_bridge import validate_metadata
    from zero_parent import make_seal

    seal = make_seal(r["parent"]["initialization_result"])
    parent_contract = validate_metadata(seal)
    sys.path.insert(0, parent_contract["core_source_repo"] + "/src")
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    budget = CgroupMemoryBudget(15 * 2**30)
    out = Path(r["output"])
    out.mkdir(parents=True, exist_ok=False)
    commands = []
    result = {
        "schema": "human-prior-own-v1-synthetic-qualification-result",
        "status": "FAILED-preserved",
    }

    def guard():
        clock(r, time.time())
        budget.check()
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("workspace floor256MiB")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 128 * 2**20:
            raise RuntimeError("qualification RAM128MiB ceiling")

    def child(name, cmd):
        guard()
        owned = {}
        with (out / (name + ".log")).open("xb") as log:
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "CUDA_VISIBLE_DEVICES": "",
                },
            )
            ticks = process(proc.pid)[1]
            try:
                while proc.poll() is None:
                    owned.update(descendants(proc.pid))
                    guard()
                    time.sleep(0.05)
                if proc.returncode:
                    raise RuntimeError(name + " actual subprocess failed")
                guard()
            except BaseException:
                owned.update(descendants(proc.pid))
                stop_owned(owned)
                raise
            finally:
                commands.append(
                    dict(
                        name=name,
                        command=cmd,
                        pid=proc.pid,
                        startticks=ticks,
                        returncode=proc.poll(),
                        finished=time.time(),
                        log=ref(out / (name + ".log")),
                        descendant_startticks={str(k): v for k, v in owned.items()},
                    )
                )

    try:
        guard()
        seal_path = out / "parent-seal.json"
        write(seal_path, seal)
        admission_clock = out / "parent-admission-clock.json"
        write(
            admission_clock,
            dict(
                schema="human-prior-own-parent-readonly-admission-clock-v1",
                first=r["first"],
                deadline=r["deadline"],
                operator_end_epoch=r["operator_end_epoch"],
                helper_sha256=sha(stage / "admit_parent.py"),
                seal_sha256=sha(seal_path),
            ),
        )
        admission = out / "parent-admission.json"
        child(
            "admit-literal-zero-parent",
            [
                sys.executable,
                str(stage / "admit_parent.py"),
                "--seal",
                str(seal_path),
                "--clock",
                str(admission_clock),
                "--output",
                str(admission),
            ],
        )
        # Legal synthetic histories, no model forward or production labels in builder.
        import chess
        import model

        boards = [chess.Board(), chess.Board()]
        boards[1].push_uci("e2e4")
        rows = [
            dict(
                indices=model.board_indices(boards[i % 2]),
                prior_logit=0.125 if i % 2 == 0 else -0.125,
                target=0.2 if i % 2 == 0 else -0.2,
            )
            for i in range(1024)
        ]
        data = out / "synthetic-data.json"
        write(
            data,
            dict(schema="own-kingbucket-sparse-training-data-v1", phase="own-learning", rows=rows),
        )
        provenance = out / "synthetic-provenance.json"
        write(
            provenance,
            dict(
                schema="NNUE-v3-synthetic-qualification-data",
                synthetic_only=True,
                rows=1024,
                unique_histories=2,
                history_uci=[[], ["e2e4"]],
                actual_own_labels_used=False,
                real_collection_receipts_forged=False,
                parent_weights_only=seal["parent_candidate"],
            ),
        )
        from native import MATH

        c = dict(
            phase="own-learning",
            updates=64,
            seed=r["seed"],
            math=MATH,
            generation=1,
            parent_generation=0,
            lineage_origin=parent_contract["lineage_origin"],
            search_helper=parent_contract["search_helper"],
            feature_schema=parent_contract["feature_schema"],
            dataset_sha256=sha(data),
            source_sha256={
                str(stage / n): sha(stage / n) for n in ("model.py", "native.py", "train.py")
            },
            execution_helpers_sha256={
                str(stage / n): sha(stage / n)
                for n in RUNTIME
                if n not in ("model.py", "native.py", "train.py")
            },
            execution_scope_schema="human-prior-own-execution-contract-v1",
            execution_mode="proof",
            original_first_epoch=r["first"],
            original_deadline_epoch=r["deadline"],
            operator_end_epoch=r["operator_end_epoch"],
            synthetic_only=True,
            bootstrap_candidate_path=seal["parent_candidate"]["path"],
            bootstrap_candidate_sha256=seal["parent_candidate"]["sha256"],
            parent_candidate=seal["parent_candidate"],
            parent_admission_seal=ref(seal_path),
            parent_admission_result=ref(admission),
            target_provenance_path=str(provenance),
            target_provenance_sha256=sha(provenance),
            prior_helper_path=parent_contract["prior_helper_path"],
            prior_helper_sha256=parent_contract["prior_helper_sha256"],
            inference_source_sha256=parent_contract["inference_source_sha256"],
            core_source_repo=parent_contract["core_source_repo"],
            core_source_commit=parent_contract["core_source_commit"],
        )
        cp = out / "synthetic-contract.json"
        write(cp, c)
        proof = out / "proof-registration.json"
        write(
            proof,
            dict(
                schema="human-prior-own-training-orchestration-v1",
                status="registered",
                mode="proof",
                seed=r["seed"],
                first=r["first"],
                deadline=r["deadline"],
                operator_end_epoch=r["operator_end_epoch"],
                cpu_core=r["cpu_core"],
                output=str(out / "proof"),
                parent_candidate=seal["parent_candidate"],
                contract=ref(cp),
                dataset=ref(data),
                train=ref(stage / "train.py"),
                native=ref(stage / "native.py"),
                inputs=dict(
                    parent_seal=ref(seal_path),
                    parent_admission=ref(admission),
                    provenance=ref(provenance),
                    qualification_registration=ref(r["_registration_path"]),
                ),
                source_sha256={str(stage / n): sha(stage / n) for n in RUNTIME},
                synthetic_only=True,
            ),
        )
        child(
            "whole-pause-freshresume-native-proof",
            [sys.executable, str(stage / "prove.py"), "--registration", str(proof)],
        )
        actual = read(out / "proof/result.json")
        if (
            actual["status"] != "PASS-own-NNUE-fixed-phase-and-fresh-native-loads-not-strength"
            or actual["full_payload_bits_equal"] is not True
            or actual["own_updates"] != 8
            or [x["step"] for x in actual["native_payloads"]] != [0, 8, 0, 4, 4, 8]
            or any(x["actual_fresh_process"] is not True for x in actual["native_payloads"])
        ):
            raise ValueError("actual full native whole/resume plus six fresh loads")
        guard()
        check_sources(r)
        result.update(
            status="PASS-human-prior-synthetic-parent-bridge-and-native-not-collection-or-strength",
            admission=ref(admission),
            proof=ref(out / "proof/result.json"),
            synthetic_contract=ref(cp),
            full_payload_bits_equal=True,
            fresh_strict_loads=6,
            synthetic_optimizer_updates=16,
            unique_synthetic_histories=2,
            real_collection_qualified=False,
            real_generation_fit=False,
        )
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        result.update(
            seed=r["seed"],
            first=r["first"],
            deadline=r["deadline"],
            operator_end_epoch=r["operator_end_epoch"],
            finished=time.time(),
            commands=commands,
            qualification_registration=ref(r["_registration_path"]),
            parent_original_clocks_unchanged=True,
            runtime_sha256=r["runtime_sha256"],
            helper_sha256=r["helper_sha256"],
        )
        write(out / "result.json", result)


def observed(a):
    paths = dict(initialization_result=str(a.initialization_result.resolve()))
    return dict(
        schema="human-prior-own-v1-synthetic-qualification-build-seal",
        seed=a.seed,
        cpu_core=a.cpu_core,
        first=a.first,
        deadline=a.deadline,
        operator_end_epoch=a.operator_end,
        stage=str(a.stage.resolve()),
        source_inventory=ref(a.inventory),
        parent={k: ref(v) for k, v in paths.items()},
        output=str(a.proof_output.resolve()),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--spec", type=Path)
    p.add_argument("--prepare-observed", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--registration", type=Path)
    p.add_argument("--seed", type=int)
    p.add_argument("--cpu-core", type=int)
    p.add_argument("--stage", type=Path)
    p.add_argument("--inventory", type=Path)
    p.add_argument("--initialization-result", type=Path)
    p.add_argument("--proof-output", type=Path)
    for name in ("first", "deadline", "operator-end"):
        p.add_argument("--" + name, type=float)
    a = p.parse_args()
    if a.prepare_observed:
        prepare(observed(a), a.output)
    elif a.prepare:
        prepare(read(a.spec), a.output)
    else:
        r = read(a.registration)
        r["_registration_path"] = str(a.registration.resolve())
        run(r)
