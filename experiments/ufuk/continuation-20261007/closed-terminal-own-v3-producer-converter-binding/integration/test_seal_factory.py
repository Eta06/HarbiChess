import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import seal_factory  # noqa: E402


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def ref(path):
    return seal_factory.ref(path)


def test_conversion_spec_and_build_seal_are_explicit_and_pinned(tmp_path):
    output = tmp_path / "producer-output"
    output.mkdir()
    helper = tmp_path / "parent"
    for name in ("model.py", "native.py", "evaluator.py", "prior.py", "ext.so"):
        (helper / name).parent.mkdir(parents=True, exist_ok=True)
        (helper / name).write_bytes(name.encode())
    parent_candidate = tmp_path / "candidate.pt"
    parent_candidate.write_bytes(b"candidate")
    parent_contract = put(tmp_path / "parent-contract.json", {"seed": 20262905})
    registration = {
        "schema": "own-nnue-closed-terminal-collection-registration-v1",
        "status": "registered",
        "seed": 20262905,
        "output_path": str(output),
        "operator_end_epoch": 500,
        "parent_candidate": {**ref(parent_candidate), "contract_sha256": "d" * 64},
        "parent_helpers": {
            "directory": str(helper),
            "model_sha256": seal_factory.sha(helper / "model.py"),
            "native_sha256": seal_factory.sha(helper / "native.py"),
            "evaluator_path": str(helper / "evaluator.py"),
            "evaluator_sha256": seal_factory.sha(helper / "evaluator.py"),
            "extension_path": str(helper / "ext.so"),
            "extension_sha256": seal_factory.sha(helper / "ext.so"),
            "prior_path": str(helper / "prior.py"),
            "prior_sha256": seal_factory.sha(helper / "prior.py"),
        },
    }
    registration_path = put(tmp_path / "registration.json", registration)
    events = output / "events.jsonl"
    events.write_bytes(b"{}\n")
    receipt = {
        "schema": "own-nnue-closed-terminal-collection-receipt-v1",
        "status": "PASS-exact-closed-terminal-row-budget",
        "seed": 20262905,
        "registration_sha256": seal_factory.sha(registration_path),
        "events_bytes": events.stat().st_size,
        "events_sha256": seal_factory.sha(events),
        "alias_chunks": [],
    }
    receipt_path = put(output / "receipt.json", receipt)
    conv_spec = seal_factory.make_conversion_spec(
        registration_path, receipt_path, first=100, deadline=200, core_repo=tmp_path
    )
    assert conv_spec["registration"]["sha256"] == seal_factory.sha(registration_path)
    assert conv_spec["events"]["sha256"] == receipt["events_sha256"]
    spec_path = put(tmp_path / "converter-seal.json", conv_spec)
    dataset = put(output / "dataset.json", {"rows": [1]})
    provenance = put(
        output / "provenance.json",
        {
            "collection_receipt_sha256": conv_spec["receipt"]["sha256"],
            "dataset_sha256": seal_factory.sha(dataset),
            "parent_candidate": {
                "path": str(parent_candidate),
                "sha256": seal_factory.sha(parent_candidate),
            },
        },
    )
    result = put(
        output / "result.json",
        {
            "status": "PASS-own1024-closed-terminal-fullhistory-conversion-not-strength",
            "dataset_sha256": seal_factory.sha(dataset),
            "provenance_sha256": seal_factory.sha(provenance),
        },
    )
    seal = seal_factory.make_seal(
        mode="proof",
        first=200,
        deadline=300,
        operator_end_epoch=500,
        seed=20262905,
        conversion_spec=spec_path,
        conversion_result=result,
        parent_contract=parent_contract,
        core_source_repo=tmp_path,
        core_source_commit="0" * 40,
        feature_schema="fixture",
    )
    assert seal["first"] == 200 and seal["deadline"] == 300
    assert seal["source_inputs"]["collection_receipt"] == conv_spec["receipt"]["sha256"]
    assert seal["parent_candidate"]["sha256"] == seal_factory.sha(parent_candidate)
    assert seal["inference_source_sha256"][str(helper / "model.py")] == seal_factory.sha(
        helper / "model.py"
    )


def test_seal_rejects_wrong_collection_seed(tmp_path):
    spec_path = put(tmp_path / "spec.json", {"schema": "wrong"})
    result_path = put(tmp_path / "result.json", {})
    with pytest.raises(ValueError, match="converter result"):
        seal_factory.make_seal(
            mode="proof",
            first=1,
            deadline=2,
            operator_end_epoch=3,
            seed=20262905,
            conversion_spec=spec_path,
            conversion_result=result_path,
            parent_contract=put(tmp_path / "parent.json", {"seed": 20262905}),
            core_source_repo=tmp_path,
            core_source_commit="0" * 40,
            feature_schema="fixture",
        )


def test_typed_child_record_requires_real_proof_fit_payload_chain(tmp_path):
    seed = 20262905
    proof_seal = put(tmp_path / "proof-seal.json", {"mode": "proof"})
    fit_seal = put(tmp_path / "fit-seal.json", {"mode": "fresh-fit"})
    proof_payloads = []
    for index, step in enumerate((0, 8, 0, 4, 4, 8)):
        path = tmp_path / f"proof-{index}.pt"
        path.write_bytes(f"proof-{index}".encode())
        proof_payloads.append(
            {
                "path": str(path),
                "sha256": seal_factory.sha(path),
                "step": step,
                "actual_fresh_process": True,
            }
        )
    fit_payloads = []
    for index, step in enumerate((0, 64)):
        path = tmp_path / f"fit-{index}.pt"
        path.write_bytes(f"fit-{index}".encode())
        fit_payloads.append(
            {
                "path": str(path),
                "sha256": seal_factory.sha(path),
                "step": step,
                "actual_fresh_process": True,
            }
        )
    proof_registration = put(tmp_path / "proof-reg.json", {})
    fit_registration = put(tmp_path / "fit-reg.json", {})
    proof_contract = put(
        tmp_path / "proof-contract.json",
        {"contract_build_seal_sha256": seal_factory.sha(proof_seal)},
    )
    helper_paths = {}
    for name in ("model", "native", "contract", "convert", "contract_builder"):
        helper_path = tmp_path / f"helper-{name}.py"
        helper_path.write_text(name)
        helper_paths[name] = helper_path
    closure = {
        seal_factory.ref(path)["path"]: seal_factory.sha(path) for path in helper_paths.values()
    }
    contract = put(
        tmp_path / "contract.json",
        {
            "seed": seed,
            "phase": "own-closed-terminal-learning-v1",
            "execution_mode": "fresh-fit",
            "contract_build_seal_sha256": seal_factory.sha(fit_seal),
            "source_sha256": closure,
        },
    )
    proof_result = put(
        tmp_path / "proof-result.json",
        {
            "status": "PASS-closed-terminal-fixed-phase-and-fresh-native-loads-not-strength",
            "mode": "proof",
            "seed": seed,
            "own_updates": 8,
            "contract_sha256": seal_factory.sha(proof_contract),
            "registration_sha256": seal_factory.sha(proof_registration),
            "native_payloads": proof_payloads,
        },
    )
    fit_result = put(
        tmp_path / "fit-result.json",
        {
            "status": "PASS-closed-terminal-fixed-phase-and-fresh-native-loads-not-strength",
            "mode": "fresh-fit",
            "seed": seed,
            "own_updates": 64,
            "contract_sha256": seal_factory.sha(contract),
            "registration_sha256": seal_factory.sha(fit_registration),
            "native_payloads": fit_payloads,
        },
    )
    inputs = {}
    for name in ("dataset", "target", "candidate", "receipt", "collection-reg", "events"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        inputs[name] = path
    record = seal_factory.make_child_record(
        seed=seed,
        contract=contract,
        proof_contract=proof_contract,
        dataset=inputs["dataset"],
        target_provenance=inputs["target"],
        candidate=inputs["candidate"],
        proof_result=proof_result,
        fit_result=fit_result,
        proof_registration=proof_registration,
        fit_registration=fit_registration,
        collection_receipt=inputs["receipt"],
        collection_registration=inputs["collection-reg"],
        events=inputs["events"],
        proof_build_seal=proof_seal,
        fit_build_seal=fit_seal,
        variant_helpers=helper_paths,
    )
    assert record["initial"]["path"] == fit_payloads[0]["path"]
    assert record["native"]["path"] == fit_payloads[1]["path"]
    assert set(record["contract_build_seals"]) == {"proof", "fit"}


def test_audit_set_is_mc_only_and_binds_audit_helper(tmp_path):
    audit_helper = HERE / "audit_six.py"
    entries = []
    for seed in (20262905, 20262906):
        clock = put(tmp_path / f"clock-{seed}.json", {"first": 10, "deadline": 20})
        result = put(
            tmp_path / f"audit-{seed}.json",
            {
                "schema": "NNUE-own-closed-terminal-six-search-audit-v1",
                "status": (
                    "PASS-six-actual-chronological-terminal-eligible-search-packets-"
                    "and-complete-alias-traces"
                ),
                "seed": seed,
                "helper_sha256": seal_factory.sha(audit_helper),
                "clock_sha256": seal_factory.sha(clock),
                "packets": [{}] * 6,
                "new_games": 0,
                "new_training_rows": 0,
                "optimizer_updates": 0,
            },
        )
        entries.append({"seed": seed, "result": result, "clock": clock})
    audit_set = seal_factory.make_audit_set(entries)
    assert audit_set["schema"] == "NNUE-own-closed-terminal-replay-audit-set-v1"
    assert audit_set["helper"]["sha256"] == seal_factory.sha(audit_helper)
    bad = json.loads(Path(entries[0]["result"]).read_text())
    bad["schema"] = "NNUE-own-collection-six-root-audit-result-v2"
    put(entries[0]["result"], bad)
    with pytest.raises(ValueError, match="MC six-packet"):
        seal_factory.make_audit_set(entries)
