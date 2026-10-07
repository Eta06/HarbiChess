"""Metadata-only ROOT phase wiring; no imports of Torch, model, engine or trainer."""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

END = 1791448916.685839
SEEDS = (20262905, 20262906)
OFFSETS = (0, 204, 409, 614, 819, 1023)
AUDIT_SHA = "2082c379ac0dc0540e3e758ce70da09c20d972a267eee057a8b95cd2609a491e"
RAM = Path("/dev/shm")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(262144), b""):
            h.update(chunk)
    return h.hexdigest()


def ref(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=sha(p))


def read(path):
    if Path(path).stat().st_size > 8 * 2**20:
        raise ValueError("metadata JSON8MiB cap")
    return json.loads(Path(path).read_bytes())


def pinned(r):
    if sha(r["path"]) != r["sha256"]:
        raise ValueError("sealed metadata/artifact changed")
    return Path(r["path"])


def collection(registration):
    reg = read(registration)
    if (
        reg["schema"] != "own-nnue-ownq-collection-registration-v2"
        or reg["status"] != "registered"
        or reg["seed"] not in SEEDS
    ):
        raise ValueError("actual collection registration-v2")
    root = Path(reg["output_path"])
    receipt = read(root / "receipt.json")
    if (
        receipt["schema"] != "own-nnue-ownq-collection-receipt-v2"
        or receipt["status"] != "PASS-exact-row-budget"
        or receipt["registration_sha256"] != sha(registration)
        or receipt["seed"] != reg["seed"]
        or receipt["teacher_labels_used"] is not False
        or receipt["train_rows"] != 1024
        or not reg["original_first_epoch"]
        <= receipt["finished_epoch"]
        <= receipt["original_deadline_epoch"]
        == reg["original_deadline_epoch"]
        <= reg["operator_end_epoch"]
        <= END
    ):
        raise ValueError("completed immutable actual collection1024")
    ids = receipt["training_row_ids"]
    if (
        len(ids) != 1024
        or len(set(ids)) != 1024
        or not all(isinstance(i, str) for i in ids)
        or receipt["periodic_independent_search_rows"] != [ids[i] for i in OFFSETS]
    ):
        raise ValueError("ordered1024 string row identifiers")
    events = ref(root / "events.jsonl")
    if events["sha256"] != receipt["events_sha256"]:
        raise ValueError("closed event SHA")
    chunks = {}
    for chunk in receipt["alias_chunks"]:
        name = chunk["file"]
        if Path(name).name != name or name in chunks:
            raise ValueError("exact unique regular sidecar names")
        r = ref(root / name)
        if r["sha256"] != chunk["sha256"] or Path(r["path"]).stat().st_size != chunk["bytes"]:
            raise ValueError("complete raw sidecar SHA/bytes")
        chunks[name] = r
    return reg, receipt, events, chunks


def require_audits(manifest_path, registrations):
    manifest = read(manifest_path)
    if manifest["schema"] != "NNUE-own-collection-replay-audit-set-v2":
        raise ValueError("ROOT twelve-packet audit manifest")
    if manifest["helper"]["sha256"] != AUDIT_SHA:
        raise ValueError("only corrected string-ID helper is eligible")
    pinned(manifest["helper"])
    audits = manifest["audits"]
    if sorted(a["seed"] for a in audits) != list(SEEDS):
        raise ValueError("BOTH six-packet audits mandatory before any proof/fit")
    for audit in audits:
        seed = audit["seed"]
        reg, receipt, events, _ = collection(registrations[str(seed)])
        result, clock = read(pinned(audit["result"])), read(pinned(audit["clock"]))
        if (
            result["schema"] != "NNUE-own-collection-six-root-audit-result-v2"
            or result["status"]
            != "PASS-six-actual-chronological-parent-search-packets-and-full-alias-traces"
            or result["row_identifier_repair"] != "root-id:local-ply-mapping-v2"
            or result["seed"] != seed
            or result["helper_sha256"] != AUDIT_SHA
            or result["clock_sha256"] != audit["clock"]["sha256"]
            or clock["schema"] != "NNUE-own-collection-six-root-audit-clock-v2"
            or clock["helper_sha256"] != AUDIT_SHA
            or clock["registration_sha256"] != result["registration_sha256"]
            or clock["receipt_sha256"] != result["receipt_sha256"]
            or clock["events_sha256"] != result["events_sha256"]
            or result["registration_sha256"] != sha(registrations[str(seed)])
            or result["receipt_sha256"] != sha(Path(reg["output_path"]) / "receipt.json")
            or result["events_sha256"] != events["sha256"]
            or [p["row_id"] for p in result["packets"]]
            != receipt["periodic_independent_search_rows"]
            or any(result[k] != 0 for k in ("new_training_rows", "new_games", "optimizer_updates"))
            or (result["first"], result["deadline"]) != (clock["first"], clock["deadline"])
            or not result["first"]
            < result["finished"]
            <= result["deadline"]
            <= min(result["first"] + 600, reg["operator_end_epoch"])
        ):
            raise ValueError("actual complete12 source/packet/clock qualifications")
    return ref(manifest_path)


def checked_clock(phase, first, deadline, operator):
    cap = 1800 if phase == "fit" else 600
    if (
        not all(type(x) in (int, float) and math.isfinite(x) for x in (first, deadline, operator))
        or not first <= time.time() < deadline <= min(first + cap, operator)
        or operator > END
    ):
        raise ValueError("explicit ROOT-observed current phase clock")


def prepare(a):
    checked_clock(a.phase, a.first, a.deadline, a.operator_end)
    if a.seed not in SEEDS:
        raise ValueError("fixed seed")
    reg, receipt, events, chunks = collection(a.registration)
    if reg["seed"] != a.seed or reg["operator_end_epoch"] != a.operator_end:
        raise ValueError("same-seed ROOT window")
    training = Path(a.training).resolve()
    for name in ("model.py", "native.py", "train.py", "convert.py", "contracts.py", "prove.py"):
        if not (training / name).is_file():
            raise ValueError("complete frozen training helper inventory")
    data = RAM / "harbichess-NNUE-own-data-v2" / str(a.seed)
    if a.phase == "convert":
        producer = Path(a.producer).resolve()
        for name, digest in reg["producer_source_sha256"].items():
            if sha(producer / name) != digest:
                raise ValueError("exact frozen producer directory")
        h = reg["parent_helpers"]
        if (
            sha(Path(h["directory"]) / "model.py") != h["model_sha256"]
            or sha(h["prior_path"]) != h["prior_sha256"]
        ):
            raise ValueError("exact parent features/prior")
        seal = dict(
            schema="NNUE-own1024-dataset-conversion-seal-v2",
            status="registered",
            first=a.first,
            deadline=a.deadline,
            operator_end_epoch=a.operator_end,
            registration=ref(a.registration),
            receipt=ref(Path(reg["output_path"]) / "receipt.json"),
            events=events,
            alias_chunks=chunks,
            producer_directory=str(Path(a.producer).resolve()),
            features=ref(Path(h["directory"]) / "model.py"),
            prior=ref(h["prior_path"]),
            core_repo=reg["core_repo"],
        )
        return {"convert-seal.json": seal}, [
            [
                sys.executable,
                str(training / "convert.py"),
                "--seal",
                str(Path(a.output) / "convert-seal.json"),
                "--output",
                str(data),
            ]
        ]
    if not a.audits or not a.registrations or not a.teacher_registry or not a.teacher_admission:
        raise ValueError("both collection replay audits and original teacher refs required")
    audit_set = require_audits(a.audits, read(a.registrations))
    teachers = read(a.teacher_registry)["teachers"]
    teacher = teachers[str(a.seed)]
    if teacher["candidate"]["sha256"] != reg["parent_candidate"]["sha256"]:
        raise ValueError("named actual teacher parent")
    for r in teacher.values():
        if isinstance(r, dict) and "path" in r and "sha256" in r:
            pinned(r)
    if type(a.cpu_core) is not int or a.cpu_core < 0:
        raise ValueError("explicit ROOT CPU core required")
    mode = "proof" if a.phase == "proof" else "fresh-fit"
    audits_metadata = read(a.audits)
    audit_inputs = dict(
        collection_replay_audit_set=audit_set, collection_replay_helper=audits_metadata["helper"]
    )
    for item in audits_metadata["audits"]:
        for key in ("result", "clock"):
            audit_inputs[f"collection_replay_{item['seed']}_{key}"] = item[key]
    build = dict(
        schema="NNUE-own-contract-build-seal-v2",
        status="registered",
        seed=a.seed,
        mode=mode,
        first=a.first,
        deadline=a.deadline,
        operator_end_epoch=a.operator_end,
        dataset=ref(data / "dataset.json"),
        provenance=ref(data / "provenance.json"),
        teacher=teacher,
        teacher_admission=ref(a.teacher_admission),
        collection_replay_audit_set=audit_set,
        teacher_registry=ref(a.teacher_registry),
        collection_replay_results=read(a.audits)["audits"],
        metadata_spec_helper=ref(__file__),
    )
    if mode == "fresh-fit":
        if not a.own_proof_result or not a.own_proof_contract:
            raise ValueError("actual own proof required before fresh64")
        build.update(
            own_proof_result=ref(a.own_proof_result), own_proof_contract=ref(a.own_proof_contract)
        )
    contract = RAM / "harbichess-NNUE-own-contracts-v2" / str(a.seed) / (a.phase + ".json")
    plan = dict(
        schema="NNUE-own-registry-plan-v2",
        seed=a.seed,
        mode=mode,
        first=a.first,
        deadline=a.deadline,
        operator_end_epoch=a.operator_end,
        cpu_core=a.cpu_core,
        training=str(training),
        contract_path=str(contract),
        dataset=build["dataset"],
        parent_candidate=reg["parent_candidate"],
        collection_replay_audit_set=audit_set,
        collection_replay_results=build["collection_replay_results"],
        inputs=dict(build_seal=ref_pending(Path(a.output) / "build-seal.json"), **audit_inputs),
        output=str(
            RAM
            / ("harbichess-NNUE-own-proof-v2" if mode == "proof" else "harbichess-NNUE-own-fit-v2")
            / str(a.seed)
        ),
    )
    commands = [
        [
            sys.executable,
            str(training / "contracts.py"),
            "--seal",
            str(Path(a.output) / "build-seal.json"),
            "--output",
            str(contract),
        ],
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--phase",
            "registry",
            "--plan",
            str(Path(a.output) / "registry-plan.json"),
            "--output",
            str(Path(a.output).with_name(Path(a.output).name + "-registry")),
        ],
    ]
    return {"build-seal.json": build, "registry-plan.json": plan}, commands


def ref_pending(path):
    return dict(path=str(path), sha256="PENDING-unpublished-build-seal")


def registry(plan_path):
    plan = read(plan_path)
    checked_clock(
        "fit" if plan["mode"] == "fresh-fit" else "proof",
        plan["first"],
        plan["deadline"],
        plan["operator_end_epoch"],
    )
    contract = ref(plan["contract_path"])
    c = read(contract["path"])
    if (
        c["seed"],
        c["execution_mode"],
        c["original_first_epoch"],
        c["original_deadline_epoch"],
        c["operator_end_epoch"],
        c["dataset_sha256"],
    ) != (
        plan["seed"],
        plan["mode"],
        plan["first"],
        plan["deadline"],
        plan["operator_end_epoch"],
        plan["dataset"]["sha256"],
    ):
        raise ValueError("strict builder contract matches ROOT plan")
    training = Path(plan["training"])
    inputs = plan["inputs"]
    inputs["build_seal"] = ref(inputs["build_seal"]["path"])
    for r in inputs.values():
        pinned(r)
    r = {
        k: plan[k]
        for k in (
            "seed",
            "mode",
            "first",
            "deadline",
            "operator_end_epoch",
            "cpu_core",
            "output",
            "dataset",
            "parent_candidate",
        )
    }
    r.update(
        metadata_spec_helper=ref(__file__),
        schema="NNUE-own-training-orchestration-v2",
        status="registered",
        contract=contract,
        train=ref(training / "train.py"),
        native=ref(training / "native.py"),
        inputs=inputs,
        source_sha256={
            str(training / n): sha(training / n)
            for n in ("model.py", "native.py", "train.py", "convert.py", "contracts.py", "prove.py")
        },
        collection_replay_audit_set=plan["collection_replay_audit_set"],
    )
    return r


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["convert", "proof", "fit", "registry"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--plan", type=Path)
    for name in (
        "registration",
        "registrations",
        "training",
        "producer",
        "audits",
        "teacher-registry",
        "teacher-admission",
        "own-proof-result",
        "own-proof-contract",
    ):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--seed", type=int)
    p.add_argument("--cpu-core", type=int)
    for name in ("first", "deadline", "operator-end"):
        p.add_argument("--" + name, type=float)
    a = p.parse_args()
    if not (
        a.output.resolve().is_relative_to("/dev/shm")
        or a.output.resolve().is_relative_to("/workspace/work/harbichess")
    ):
        raise ValueError("only unpublished RAM/scratch specs")
    if a.output.exists():
        raise ValueError("publish once")
    if a.phase == "registry":
        documents = {"registration.json": registry(a.plan)}
        commands = [
            [
                sys.executable,
                str(Path(read(a.plan)["training"]) / "prove.py"),
                "--registration",
                str(a.output / "registration.json"),
            ]
        ]
    else:
        documents, commands = prepare(a)
    a.output.mkdir(parents=True, exist_ok=False)
    for name, document in documents.items():
        (a.output / name).write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    (a.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
