"""Synthetic metadata only; no model import, search, training or child launch."""

import copy
import json
from pathlib import Path

import pytest
import specs


@pytest.fixture
def evidence(tmp_path):
    def write(path, x):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(x))
        return specs.ref(path)

    helper = Path(
        "/workspace/work/harbichess/continuation-20261007/"
        "nnue-strength-v2/audit_collection_six_v2.py"
    )
    assert specs.sha(helper) == specs.AUDIT_SHA
    registrations, audits = {}, []
    for seed in specs.SEEDS:
        root = tmp_path / str(seed)
        ids = [f"root-{i // 16}:{i % 16}" for i in range(1024)]
        reg = dict(
            schema="own-nnue-ownq-collection-registration-v2",
            status="registered",
            seed=seed,
            output_path=str(root),
            original_first_epoch=10,
            original_deadline_epoch=100,
            operator_end_epoch=specs.END,
        )
        registration = write(root / "reg.json", reg)
        registrations[str(seed)] = registration["path"]
        events = root / "events.jsonl"
        events.write_bytes(b"synthetic fixture opaque event marker\n")
        receipt = dict(
            schema="own-nnue-ownq-collection-receipt-v2",
            status="PASS-exact-row-budget",
            registration_sha256=registration["sha256"],
            seed=seed,
            train_rows=1024,
            teacher_labels_used=False,
            finished_epoch=90,
            original_deadline_epoch=100,
            training_row_ids=ids,
            periodic_independent_search_rows=[ids[i] for i in specs.OFFSETS],
            events_sha256=specs.sha(events),
            alias_chunks=[],
        )
        rr = write(root / "receipt.json", receipt)
        clock = dict(
            schema="NNUE-own-collection-six-root-audit-clock-v2",
            first=110,
            deadline=210,
            helper_sha256=specs.AUDIT_SHA,
            registration_sha256=registration["sha256"],
            receipt_sha256=rr["sha256"],
            events_sha256=specs.sha(events),
        )
        cr = write(root / "clock.json", clock)
        result = dict(
            schema="NNUE-own-collection-six-root-audit-result-v2",
            status="PASS-six-actual-chronological-parent-search-packets-and-full-alias-traces",
            row_identifier_repair="root-id:local-ply-mapping-v2",
            seed=seed,
            helper_sha256=specs.AUDIT_SHA,
            clock_sha256=cr["sha256"],
            registration_sha256=registration["sha256"],
            receipt_sha256=rr["sha256"],
            events_sha256=specs.sha(events),
            first=110,
            deadline=210,
            finished=200,
            packets=[dict(row_id=ids[i]) for i in specs.OFFSETS],
            new_training_rows=0,
            new_games=0,
            optimizer_updates=0,
        )
        result_ref = write(root / "audit.json", result)
        audits.append(dict(seed=seed, result=result_ref, clock=cr))
    manifest = dict(
        schema="NNUE-own-collection-replay-audit-set-v2", helper=specs.ref(helper), audits=audits
    )
    path = tmp_path / "audit-set.json"
    write(path, manifest)
    return path, registrations, manifest, write


def test_both_six_string_id_audits_required(evidence):
    path, registrations, manifest, write = evidence
    assert specs.require_audits(path, registrations) == specs.ref(path)
    missing = dict(manifest, audits=manifest["audits"][:1])
    write(path, missing)
    with pytest.raises(ValueError, match="BOTH"):
        specs.require_audits(path, registrations)


@pytest.mark.parametrize(
    "field,value",
    [
        ("row_id", 204),
        ("status", "FAILED-preserved"),
        ("helper_sha256", "0" * 64),
        ("optimizer_updates", 1),
        ("receipt_sha256", "0" * 64),
    ],
)
def test_resealed_audit_mutation_rejects(evidence, field, value):
    path, registrations, manifest, write = evidence
    changed = copy.deepcopy(manifest)
    result_path = Path(changed["audits"][0]["result"]["path"])
    result = specs.read(result_path)
    if field == "row_id":
        result["packets"][1]["row_id"] = value
    else:
        result[field] = value
    changed["audits"][0]["result"] = write(result_path, result)
    write(path, changed)
    with pytest.raises(ValueError):
        specs.require_audits(path, registrations)


def test_declared_no_model_imports_or_launches():
    import ast

    tree = ast.parse(Path(specs.__file__).read_text())
    names = {n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)}
    assert not names & {"torch", "subprocess", "chess", "native", "model"}
    assert specs.OFFSETS == (0, 204, 409, 614, 819, 1023)


def test_registry_binds_existing_contract_and_final_output_without_model(tmp_path):
    import time

    training = tmp_path / "training"
    training.mkdir()
    for name in ("model.py", "native.py", "train.py", "convert.py", "contracts.py", "prove.py"):
        (training / name).write_text("# metadata fixture only\n")
    first = time.time() - 1
    dataset = tmp_path / "data.json"
    dataset.write_bytes(b"synthetic-data")
    build = tmp_path / "build.json"
    build.write_bytes(b"synthetic-build-seal")
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"synthetic-audit-set")
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            dict(
                seed=20262905,
                execution_mode="proof",
                original_first_epoch=first,
                original_deadline_epoch=first + 60,
                operator_end_epoch=specs.END,
                dataset_sha256=specs.sha(dataset),
            )
        )
    )
    plan = dict(
        seed=20262905,
        mode="proof",
        first=first,
        deadline=first + 60,
        operator_end_epoch=specs.END,
        cpu_core=4,
        training=str(training),
        contract_path=str(contract),
        dataset=specs.ref(dataset),
        parent_candidate=dict(path="synthetic-parent", sha256="0" * 64),
        output="/dev/shm/harbichess-NNUE-own-proof-v2/20262905",
        collection_replay_audit_set=specs.ref(manifest),
        inputs=dict(
            build_seal=dict(path=str(build), sha256="PENDING"),
            collection_replay_audit_set=specs.ref(manifest),
        ),
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    result = specs.registry(path)
    assert result["contract"] == specs.ref(contract)
    assert result["inputs"]["build_seal"] == specs.ref(build)
    assert result["output"] == plan["output"]
    assert result["schema"] == "NNUE-own-training-orchestration-v2"
    c = specs.read(contract)
    c["dataset_sha256"] = "0" * 64
    contract.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        specs.registry(path)
