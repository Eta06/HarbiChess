"""ROOT-owned freeze/launch proposal; default validates only, never launches.

Execute samples ONE real clock before any registration writes; collection fixed
16384/seed,7200s, exactsame-seed proof with only4literal transfers. No fit/matches.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SEEDS = (20262905, 20262906)
HARD_END = 1791273600.0
SOURCE = "6fcc8b476d25495d1c9c413e55b2c7ba4794013e"
H = Path("/workspace/HarbiChess/experiments/ufuk/cpu-classical-own-v1")
ACTUAL = Path("/workspace/work/harbichess/cpu-classical-own-v1-actual")
CORE = Path("/workspace/work/harbichess/cpu-additive-source-6fcc8b4")
OPERATOR_HELPER = Path(
    "/workspace/HarbiChess/experiments/ufuk/cpu-budget-search-v1/operator_guard.py"
)
OPERATOR_SHA = "c0eae6ba73e2e8b1598a2f19b81d11a77e3c34abfbd893b6f50121e607bf08c6"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canon(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def publish(path, x):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("xb") as f:
        f.write(canon(x))
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(temp, path)
    finally:
        temp.unlink()


def proc(pid):
    try:
        w = Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2].split()
        return dict(pid=pid, state=w[0], start_ticks=int(w[19]), pgid=int(w[2]))
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return None


def append_catalog(path, owners, expected_sha):
    """ROOT serializes catalog changes; reject any unobserved concurrent update."""
    if sha(path) != expected_sha:
        raise ValueError("guardian catalog changed before append")
    current = json.loads(Path(path).read_text())
    if set(current) != {"owner_receipts"}:
        raise ValueError("guardian catalog schema differs")
    if any(str(p) in current["owner_receipts"] for p in owners):
        raise ValueError("owner path already registered")
    updated = {"owner_receipts": current["owner_receipts"] + [str(p) for p in owners]}
    tmp = Path(path).with_name(Path(path).name + ".classical-new.tmp")
    with tmp.open("xb") as f:
        f.write(canon(updated))
        f.flush()
        os.fsync(f.fileno())
    if sha(path) != expected_sha:
        raise ValueError("concurrent guardian catalog update")
    os.replace(tmp, path)
    return {
        "old_sha256": expected_sha,
        "new_sha256": sha(path),
        "appended": [str(p) for p in owners],
    }


def cleanup_created_owner(identity):
    # The existing owner guard freezes the exactPID/startticks root, snapshots
    # descendants, and signals only matching identities, including newgroups.
    if sha(OPERATOR_HELPER) != OPERATOR_SHA:
        raise ValueError("owned cleanup helper bytes differ")
    spec = importlib.util.spec_from_file_location("classical_owned_cleanup", OPERATOR_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.stop_owned(identity)


def load_helpers():
    sys.path.insert(0, str(H))
    from journal_v3 import digest, read, replay
    from qualify_actor_v3 import HELPER_CLOSURE
    from transfer import validate_transfer

    return digest, read, replay, validate_transfer, HELPER_CLOSURE


def verify_proofs():
    digest, read, replay, validate_transfer, closure = load_helpers()
    helper_sha = {n: sha(H / n) for n in sorted(closure)}
    result = {}
    for seed in SEEDS:
        configpath = ACTUAL / f"qualification-registration/{seed}-qualification-config.json"
        config = json.loads(configpath.read_text())
        proofpath = Path(f"/dev/shm/harbichess-classical-qual-{seed}-20261005/result.json")
        q = json.loads(proofpath.read_text())
        if (
            q["schema"] != "classical-own-realCPU-actor-qualification-v3"
            or q["status"]
            != "PASS-new-classical-actor-native-whole9-pause4-freshresume9-not-strength"
        ):
            raise ValueError("bothactualclassicalPASS required")
        if (
            q["config_sha256"] != sha(configpath)
            or config["seed"] != seed
            or q["helper_sha256"] != helper_sha
        ):
            raise ValueError("exactsame-seed proof/helper bindings differ")
        if (
            q["source_commit"] != config["source_commit"]
            or q["source_commit"] != SOURCE
            or config["max_actions"] != 9
        ):
            raise ValueError("source/max9 differs")
        if (
            q["first"] != config["original_first_epoch"]
            or q["deadline"] != config["original_deadline_epoch"]
            or q["deadline"] != q["first"] + 600
            or not q["first"] <= q["finished_epoch"] <= q["deadline"] <= HARD_END
        ):
            raise ValueError("actual original600 qualification bounds differ")
        if [r["actions"] for r in q["all_journals"]] != [9, 4, 9] or not q[
            "whole_resumed_gzip_exact"
        ]:
            raise ValueError("all3proof native records required")
        for row in q["all_journals"]:
            if sha(row["path"]) != row["sha256"]:
                raise ValueError("proof nativeartifact changed")
            state = read(row["path"])
            if (
                state["config"] != config
                or state["config_sha256"] != digest(config)
                or state["actions"] != row["actions"]
            ):
                raise ValueError("native fullconfig/cursor differs")
            replay(state, config)
        if (
            Path(q["all_journals"][0]["path"]).read_bytes()
            != Path(q["all_journals"][2]["path"]).read_bytes()
        ):
            raise ValueError("actual freshresume fullbytes mismatch")
        result[seed] = dict(config=config, config_path=configpath, proof=q, proof_path=proofpath)
    return result, helper_sha, validate_transfer


def plan_seed(seed, first, end, proof, helpers, registration, output):
    c = proof["config"] | dict(
        max_actions=16384,
        epoch_id=f"classical-own-fixed16384-seed-{seed}",
        original_first_epoch=first,
        original_deadline_epoch=end,
    )
    differences = {
        k: {"proof": proof["config"][k], "production": c[k]}
        for k in proof["config"]
        if proof["config"][k] != c[k]
    }
    transfer = dict(
        proof_config_sha256=sha(proof["config_path"]), allowed_difference_values=differences
    )
    configpath = registration / f"{seed}-production-config.json"
    manifest = dict(
        schema="classical-own-epoch-controller-manifest-v3",
        source_repo=str(CORE),
        cpu_core=1 if seed == SEEDS[0] else 2,
        config={"path": str(configpath), "sha256": "FILLED_AFTER_IMMUTABLE_CONFIG_PUBLICATION"},
        model={"path": str(H / "prior.json"), "sha256": sha(H / "prior.json")},
        search_helper={"path": str(H / "arena/search.py"), "sha256": sha(H / "arena/search.py")},
        value_helper={"path": str(H / "value.py"), "sha256": sha(H / "value.py")},
        helper_sha256=helpers,
        qualification_config={
            "path": str(proof["config_path"]),
            "sha256": sha(proof["config_path"]),
        },
        qualification_transfer=transfer,
    )
    return c, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration", type=Path, required=True)
    p.add_argument("--output-prefix", default="/dev/shm/harbichess-classical-own")
    p.add_argument(
        "--guardian-catalog",
        type=Path,
        default=Path("/workspace/work/harbichess/continuation-20261005/operator-catalog.json"),
    )
    p.add_argument("--guardian-pid", type=int, default=991477)
    p.add_argument("--guardian-start-ticks", type=int, required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    proofs, helpers, validate_transfer = verify_proofs()
    if sha(OPERATOR_HELPER) != OPERATOR_SHA:
        raise ValueError("pinned scoped cleanup helper differs")
    g = proc(a.guardian_pid)
    if g is None or g["state"] == "Z" or g["start_ticks"] != a.guardian_start_ticks:
        raise ValueError("exactaliveguardian required")
    guardian_argv = Path(f"/proc/{a.guardian_pid}/cmdline").read_bytes().split(b"\0")
    if (
        str(a.guardian_catalog).encode() not in guardian_argv
        or str(int(HARD_END)).encode() not in guardian_argv
    ):
        raise ValueError("guardian actualcatalog/hard08 differs")
    if a.registration.exists():
        raise FileExistsError(a.registration)
    outputs = {s: Path(f"{a.output_prefix}-{s}-20261005") for s in SEEDS}
    if any(p.exists() for p in outputs.values()):
        raise FileExistsError("newRAMoutputs required")
    catalogsha = sha(a.guardian_catalog)
    if not a.execute:
        print(
            json.dumps(
                dict(
                    status="validated-actual-bothPASS-no-clock-or-files-or-processes-created",
                    source=SOURCE,
                    helpers=helpers,
                    guardian=g,
                    proofs={str(s): sha(proofs[s]["proof_path"]) for s in SEEDS},
                    planned_outputs={str(s): str(v) for s, v in outputs.items()},
                )
            )
        )
        return
    # ONE real clock before ALL publication, metadatawrites, owners and process launches.
    first = time.time()
    end = first + 7200
    if end > HARD_END:
        raise ValueError("original7200 cannot fit hard08")
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=CORE, text=True
    ).strip() != SOURCE or subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=CORE, text=True
    ):
        raise ValueError("clean core differs")
    a.registration.mkdir(parents=True, exist_ok=False)
    publish(
        a.registration / "common-original-clock.json",
        dict(
            schema="classical-own-production-common-original-clock-v3",
            first=first,
            deadline=end,
            seconds=7200,
            hard_end=HARD_END,
            observed_before_registration=True,
        ),
    )
    owners = {s: a.registration / f"{s}-production-owner-process.json" for s in SEEDS}
    rows = {}
    processes = []
    identities = {}
    try:
        for seed in SEEDS:
            c, m = plan_seed(seed, first, end, proofs[seed], helpers, a.registration, outputs[seed])
            validate_transfer(
                proofs[seed]["config"],
                c,
                m["qualification_transfer"],
                proofs[seed]["proof"],
                sha(proofs[seed]["config_path"]),
            )
            configpath = Path(m["config"]["path"])
            publish(configpath, c)
            m["config"]["sha256"] = sha(configpath)
            manifestpath = a.registration / f"{seed}-epoch-manifest.json"
            publish(manifestpath, m)
            cmd = [
                sys.executable,
                str(H / "run_epoch_v3.py"),
                "--manifest",
                str(manifestpath),
                "--manifest-sha256",
                sha(manifestpath),
                "--qualification",
                str(proofs[seed]["proof_path"]),
                "--qualification-sha256",
                sha(proofs[seed]["proof_path"]),
                "--first",
                str(first),
                "--output",
                str(outputs[seed]),
            ]
            rows[seed] = dict(
                command=cmd,
                config_path=str(configpath),
                config_sha256=sha(configpath),
                manifest_path=str(manifestpath),
                manifest_sha256=sha(manifestpath),
                proof_result_sha256=sha(proofs[seed]["proof_path"]),
            )
        publish(
            a.registration / "code-provenance.json",
            dict(
                source_core=SOURCE,
                helper_sha256=helpers,
                stage_inventory_sha256=sha(H / "inventory.json"),
                proposal_derivation_sha256=sha(H / "DERIVATION.txt"),
                launcher_sha256=sha(__file__),
                owned_cleanup_helper_sha256=OPERATOR_SHA,
                actual_proofs={
                    str(s): {
                        "path": str(proofs[s]["proof_path"]),
                        "sha256": sha(proofs[s]["proof_path"]),
                    }
                    for s in SEEDS
                },
                scope="freshfrozen-human-prior-own-data-only;notlearnedcandidate/strength",
            ),
        )
        append = append_catalog(a.guardian_catalog, owners.values(), catalogsha)
        publish(a.registration / "guardian-append.json", append)
        from runtime import guard

        for seed in SEEDS:
            guard(end, CORE, a.registration)
            out = (a.registration / f"{seed}-production.stdout.log").open("xb")
            err = (a.registration / f"{seed}-production.stderr.log").open("xb")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(CORE / "src")
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            child = subprocess.Popen(
                rows[seed]["command"],
                cwd=CORE,
                env=env,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            processes.append(child)
            out.close()
            err.close()
            identity = proc(child.pid)
            if identity is None or identity["state"] == "Z":
                raise RuntimeError("owner terminated beforeidentitypublication")
            identities[child.pid] = dict(pid=child.pid, start_ticks=identity["start_ticks"])
            receipt = dict(
                pid=child.pid,
                start_ticks=identity["start_ticks"],
                pgid=identity["pgid"],
                command=rows[seed]["command"],
                started_epoch=time.time(),
                original_first_epoch=first,
                original_deadline_epoch=end,
                cpu_core=1 if seed == SEEDS[0] else 2,
                GPU_used=False,
                config_sha256=rows[seed]["config_sha256"],
                manifest_sha256=rows[seed]["manifest_sha256"],
                helper_sha256=helpers["run_epoch_v3.py"],
            )
            publish(owners[seed], receipt)
            rows[seed]["owner"] = receipt
        publish(
            a.registration / "launch-result.json",
            dict(
                status="launched-own-data-collection-not-training-or-strength",
                common_first=first,
                original_deadline=end,
                seeds={str(k): v for k, v in rows.items()},
            ),
        )
        print(
            json.dumps(
                dict(
                    status="launched-own-data-collection",
                    first=first,
                    end=end,
                    owners={str(s): rows[s]["owner"] for s in SEEDS},
                    registration=str(a.registration),
                )
            )
        )
    except BaseException as exc:
        stopped = []
        for child in processes:
            identity = identities.get(child.pid)
            if identity is None:
                actual = proc(child.pid)
                if actual is not None:
                    identity = dict(pid=child.pid, start_ticks=actual["start_ticks"])
            if identity is not None:
                stopped.extend(cleanup_created_owner(identity))
        publish(
            a.registration / "launch-failure.json",
            dict(
                status="failed-preserved-originalclock-no-retry",
                error=repr(exc),
                first=first,
                deadline=end,
                stopped_only_created_processes=stopped,
                rows={str(k): v for k, v in rows.items()},
            ),
        )
        raise


if __name__ == "__main__":
    main()
