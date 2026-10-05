"""ROOT-only fixedFINAL16384 offline proof/freshfit; default read-only preflight.

No partial dataset, parameter selection, automatic retry, or qualified optimizer
reuse. Each phase observes ONE clock before publication and ownership writes.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from freeze_launch import (
    ACTUAL,
    CORE,
    HARD_END,
    OPERATOR_HELPER,
    OPERATOR_SHA,
    SEEDS,
    H,
    append_catalog,
    cleanup_created_owner,
    proc,
    publish,
    sha,
    verify_proofs,
)

FIVE_HELPERS = frozenset({"train_v3.py", "learner.py", "journal_v3.py", "value.py", "runtime.py"})
PRODUCTION_REGISTRATION = ACTUAL / "production-registration"


def external_helpers():
    """Independent proof owner is external to the unchanged five-file learner closure."""
    paths = [
        H / "qualify_offline_v3.py", Path(__file__), Path(__file__).with_name("freeze_launch.py")
    ]
    return {str(path): sha(path) for path in paths}


def immutable_final_data():
    sys.path.insert(0, str(H))
    from journal_v3 import read, replay
    from learner import prepare

    actor_proofs, helpers, _ = verify_proofs()
    rows = {}
    for seed in SEEDS:
        configpath = PRODUCTION_REGISTRATION / f"{seed}-production-config.json"
        config = json.loads(configpath.read_text())
        manifestpath = PRODUCTION_REGISTRATION / f"{seed}-epoch-manifest.json"
        manifest = json.loads(manifestpath.read_text())
        if (
            manifest["config"]["path"] != str(configpath)
            or manifest["config"]["sha256"] != sha(configpath)
            or manifest["helper_sha256"] != helpers
        ):
            raise ValueError("exactqualifiedproductiondata config/source helper differs")
        path = Path(f"/dev/shm/harbichess-classical-own-{seed}-20261005/actions-00016384.json.gz")
        if not path.is_file():
            raise FileNotFoundError(f"fixedFINAL16384 not yetavailable seed{seed}")
        production_result = path.parent / "result.json"
        parent = json.loads(production_result.read_text())
        state = read(path)
        if (
            state["actions"] != 16384
            or state["config"] != config
            or config["seed"] != seed
            or config["max_actions"] != 16384
        ):
            raise ValueError("not exactfixedFINAL16384 same-seed data")
        replay(state, config)
        # Deterministic feature/prior consistency and label admission only; no fitting.
        train, val, receipt, dataset_sha = prepare(
            path, config, config["excluded_training_position_keys"]
        )
        updates = min(1024, 4 * receipt["training_rows"] // 256)
        if updates < 8:
            raise ValueError("realderivedbudget cannot support whole8 native proof")
        if (
            parent["status"] != "PASS-fixed16384-classical-own-data-not-trained"
            or parent["final_journal_sha256"] != sha(path)
            or parent["original_first"] != config["original_first_epoch"]
            or parent["original_deadline"] != config["original_deadline_epoch"]
        ):
            raise ValueError("produceractualcompletion not bound to finaljournal")
        rows[seed] = dict(
            journal_path=path,
            journal_sha256=sha(path),
            config_path=configpath,
            config_sha256=sha(configpath),
            source=config["source_commit"],
            receipt=receipt,
            dataset_sha256=dataset_sha,
            derived_updates=updates,
            production_result_path=production_result,
            production_result_sha256=sha(production_result),
            protected=config["excluded_training_position_keys"],
        )
    if rows[SEEDS[0]]["protected"] != rows[SEEDS[1]]["protected"]:
        raise ValueError("bothsameprotected rootset required")
    return rows, helpers


def verify_offline_proofs(rows, helpers):
    result = {}
    for seed in SEEDS:
        path = Path(f"/dev/shm/harbichess-classical-offline-proof-{seed}-20261005/result.json")
        q = json.loads(path.read_text())
        if (
            q["schema"] != "classical-own-offline-native-qualification-v3"
            or q["status"] != "PASS-offline-whole8-pause4-freshresume8-not-strength"
            or not q["full_native_bytes_equal"]
        ):
            raise ValueError("actualbothoffline8/4/fresh8proofPASS required")
        if (
            q["original_deadline"] != q["original_first"] + 600
            or q["finished_epoch"] > q["original_deadline"]
        ):
            raise ValueError("offlineprooforiginal600violated")
        checkpoints = q["strict_fresh_native_loads"]
        if len(checkpoints) != 6 or [r["step"] for r in checkpoints] != [0, 4, 8, 0, 4, 8]:
            raise ValueError("ALLsixfreshnative0/4/8 loads required")
        for item in checkpoints:
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("immutablefullnative bytes changed")
            n = json.loads(Path(item["path"]).read_text())
            c = n["contract"]
            if (
                n["schema"] != "classical-own-offline-native-v3"
                or n["step"] != item["step"]
                or c["protocol_sha256"] != q["protocol_sha256"]
                or c["original_deadline"] != q["original_deadline"]
                or c["dataset_sha256"] != rows[seed]["dataset_sha256"]
                or c["journal_sha256"] != rows[seed]["journal_sha256"]
                or c["config_sha256"] != rows[seed]["config_sha256"]
                or c["source_commit"] != rows[seed]["source"]
                or c["seed"] != seed
                or c["updates"] != rows[seed]["derived_updates"]
            ):
                raise ValueError("actualwholeoffline input/nativebind differs")
        if Path(checkpoints[2]["path"]).read_bytes() != Path(checkpoints[5]["path"]).read_bytes():
            raise ValueError("whole/freshresume8 actualJSON bytes mismatch")
        protocolpath = ACTUAL / "offline-qualification-registration" / "protocol.json"
        if sha(protocolpath) != q["protocol_sha256"]:
            raise ValueError("actualproofprotocol SHA differs")
        protocol = json.loads(protocolpath.read_text())
        lineage = json.loads(protocolpath.with_name("lineage.json").read_text())
        if lineage["external_helper_sha256"] != external_helpers():
            raise ValueError("proof qualifier/launcher/support bytes differ")
        if protocol["helper_sha256"] != {n: helpers[n] for n in FIVE_HELPERS}:
            raise ValueError("proof/current5helperbytes differ")
        result[seed] = dict(path=str(path), sha256=sha(path), actualproof=q)
    return result


def protocol_for(rows, helpers, deadline, phase):
    return dict(
        schema="classical-own-offline-protocol-v3",
        status="registered-own-offline-native-proof"
        if phase == "proof"
        else "registered-fresh-own-offline-fit-not-strength",
        seeds=list(SEEDS),
        final_actions=16384,
        source_commit=rows[SEEDS[0]]["source"],
        inputs={
            str(s): dict(
                journal_sha256=rows[s]["journal_sha256"], config_sha256=rows[s]["config_sha256"]
            )
            for s in SEEDS
        },
        deadline_by_seed={str(s): deadline for s in SEEDS},
        protected_position_keys=rows[SEEDS[0]]["protected"],
        objective=dict(terminal_weight=0.75, search_weight=0.25, prior_l2=0.01),
        optimizer=dict(
            lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, batch=256, slots=4, max_updates=1024
        ),
        helper_sha256={n: helpers[n] for n in sorted(FIVE_HELPERS)},
        minimum_distinct_trajectories=16,
        minimum_known_rows=1024,
        candidate_choice="onlyfixedderivedfinalupdateonFINAL16384;freshAdam/theta0mainfit;novalidationselection",
        phase=phase,
        hard_end=HARD_END,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["proof", "fit"], required=True)
    p.add_argument("--registration", type=Path, required=True)
    p.add_argument(
        "--guardian-catalog",
        type=Path,
        default=Path("/workspace/work/harbichess/continuation-20261005/operator-catalog.json"),
    )
    p.add_argument("--guardian-pid", type=int, default=991477)
    p.add_argument("--guardian-start-ticks", type=int, required=True)
    p.add_argument("--wait-seconds", type=float, default=0.0)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if a.wait_seconds < 0:
        raise ValueError("nonnegativeboundedwait required")
    wait_start = time.time()
    while True:
        try:
            rows, helpers = immutable_final_data()
            break
        except FileNotFoundError:
            if time.time() - wait_start >= a.wait_seconds or time.time() >= HARD_END:
                raise
            time.sleep(0.5)
    proofs = verify_offline_proofs(rows, helpers) if a.phase == "fit" else {}
    guardian = proc(a.guardian_pid)
    if (
        guardian is None
        or guardian["state"] == "Z"
        or guardian["start_ticks"] != a.guardian_start_ticks
    ):
        raise ValueError("exactliveguardian required")
    guardian_argv = Path(f"/proc/{a.guardian_pid}/cmdline").read_bytes().split(b"\0")
    if (
        str(a.guardian_catalog).encode() not in guardian_argv
        or str(int(HARD_END)).encode() not in guardian_argv
    ):
        raise ValueError("guardian actualcatalog/hard08 differs")
    if sha(OPERATOR_HELPER) != OPERATOR_SHA:
        raise ValueError("scopedcleanup helper differs")
    if a.registration.exists():
        raise FileExistsError(a.registration)
    outputs = {
        s: Path(
            f"/dev/shm/harbichess-classical-offline-proof-{s}-20261005"
            if a.phase == "proof"
            else f"/dev/shm/harbichess-classical-fit-{s}-20261005"
        )
        for s in SEEDS
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("freshproof/freshfitoutputs required")
    if not a.execute:
        print(
            json.dumps(
                dict(
                    status="validated-read-only-no-fit-or-clock",
                    phase=a.phase,
                    rows={
                        str(s): {
                            k: v
                            for k, v in rows[s].items()
                            if k not in ("journal_path", "config_path", "production_result_path")
                        }
                        for s in SEEDS
                    },
                    actualofflineproofs={str(s): p["sha256"] for s, p in proofs.items()},
                )
            )
        )
        return
    catalogsha = sha(a.guardian_catalog)
    # Newprospectivephase clock does NOT reset producerclock or inherit proof optimizer.
    first = time.time()
    deadline = first + (600 if a.phase == "proof" else 1800)
    if deadline > HARD_END:
        raise ValueError("phasecannotfit hard08")
    a.registration.mkdir(parents=True, exist_ok=False)
    protocol = protocol_for(rows, helpers, deadline, a.phase)
    publish(
        a.registration / "common-original-clock.json",
        dict(
            first=first,
            deadline=deadline,
            seconds=deadline - first,
            phase=a.phase,
            hard_end=HARD_END,
        ),
    )
    publish(a.registration / "protocol.json", protocol)
    publish(
        a.registration / "data-admission.json",
        {
            str(s): {k: str(v) if isinstance(v, Path) else v for k, v in rows[s].items()}
            for s in SEEDS
        },
    )
    publish(
        a.registration / "lineage.json",
        dict(
            phase=a.phase,
            producer_registration=str(PRODUCTION_REGISTRATION),
            actor_helper_closure=helpers,
            actualofflineproofs={
                str(s): {"path": p["path"], "sha256": p["sha256"]} for s, p in proofs.items()
            },
            launcher_sha256=sha(__file__),
            external_helper_sha256=external_helpers(),
            fresh_main_Adam_theta_zero=True,
            no_teacher_labels=True,
        ),
    )
    owners = {s: a.registration / f"{s}-owner-process.json" for s in SEEDS}
    publish(
        a.registration / "guardian-append.json",
        append_catalog(a.guardian_catalog, owners.values(), catalogsha),
    )
    processes = []
    identities = {}
    launched = {}
    try:
        from runtime import guard

        for s in SEEDS:
            command = [
                sys.executable,
                str(H / ("qualify_offline_v3.py" if a.phase == "proof" else "train_v3.py")),
                "--protocol",
                str(a.registration / "protocol.json"),
                "--protocol-sha256",
                sha(a.registration / "protocol.json"),
                "--config",
                str(rows[s]["config_path"]),
                "--journal",
                str(rows[s]["journal_path"]),
                "--source-repo",
                str(CORE),
                "--seed",
                str(s),
                "--cpu-core",
                "1" if s == SEEDS[0] else "2",
                "--output",
                str(outputs[s]),
            ]
            command += (
                ["--first", str(first), "--deadline", str(deadline)]
                if a.phase == "proof"
                else ["--deadline", str(deadline)]
            )
            guard(deadline, CORE, a.registration)
            with (
                (a.registration / f"{s}.stdout.log").open("xb") as out,
                (a.registration / f"{s}.stderr.log").open("xb") as err,
            ):
                env = os.environ.copy()
                env["PYTHONPATH"] = str(CORE / "src")
                env["OMP_NUM_THREADS"] = "1"
                env["MKL_NUM_THREADS"] = "1"
                child = subprocess.Popen(
                    command, cwd=CORE, env=env, stdout=out, stderr=err, start_new_session=True
                )
            processes.append(child)
            identity = proc(child.pid)
            if identity is None or identity["state"] == "Z":
                raise RuntimeError("ownedphase endedbeforeidentityreceipt")
            identities[child.pid] = dict(pid=child.pid, start_ticks=identity["start_ticks"])
            record = dict(
                pid=child.pid,
                start_ticks=identity["start_ticks"],
                pgid=identity["pgid"],
                command=command,
                started_epoch=time.time(),
                original_first_epoch=first,
                original_deadline_epoch=deadline,
                cpu_core=1 if s == SEEDS[0] else 2,
                GPU_used=False,
                protocol_sha256=sha(a.registration / "protocol.json"),
                phase=a.phase,
            )
            publish(owners[s], record)
            launched[s] = record
        publish(
            a.registration / "launch-result.json",
            dict(
                status="launched-offline-proof-not-strength"
                if a.phase == "proof"
                else "launched-fresh-fixed-final-offline-fit-not-strength",
                first=first,
                deadline=deadline,
                owners={str(s): r for s, r in launched.items()},
            ),
        )
        print(
            json.dumps(
                dict(
                    status="launched",
                    phase=a.phase,
                    first=first,
                    deadline=deadline,
                    owners={str(s): r for s, r in launched.items()},
                )
            )
        )
    except BaseException as exc:
        stopped = []
        for child in processes:
            identity = identities.get(child.pid)
            if identity is None:
                found = proc(child.pid)
                if found:
                    identity = dict(pid=child.pid, start_ticks=found["start_ticks"])
            if identity:
                stopped.extend(cleanup_created_owner(identity))
        publish(
            a.registration / "launch-failure.json",
            dict(
                status="failed-preserved-originalphaseclock-no-retry",
                error=repr(exc),
                first=first,
                deadline=deadline,
                stopped_only_created=stopped,
            ),
        )
        raise


if __name__ == "__main__":
    main()
