"""Adversarial adapter contracts; synthetic receipts are NOT native/CUDA qualification."""

import copy
import json
import sys
from pathlib import Path

import own5_eligibility as eligibility
import pytest
from own5_adapter_controls import HELPERS, SOURCE, validate_spec
from own5_audit_support import publish, sha


def controlled_manifest():
    cfg = json.loads(
        Path(
            "/workspace/work/harbichess/search-acting-v2-readiness/v2-development-config.json"
        ).read_text()
    )
    cfg["seed"] = 20261525
    return dict(
        schema="ufuk-search-acting-formal5-audit-manifest-v1",
        status="frozen-before-formal-execution",
        qualification_ledger_slot=5,
        source_commit=SOURCE,
        fixed_epochs=8,
        neural_audit_epochs=[1, 2, 3, 4, 6, 8],
        neural_witness_K=8,
        original_training_started_epoch=100.0,
        original_training_deadline_epoch=1000.0,
        whole_training_seconds=900.0,
        whole_audit_seconds=1200.0,
        frozen_config=cfg,
        inputs={
            k: {} for k in ("initial_weights", "book", "experiment_config", "protocol")
        },
        helper_sha256={k: "a" * 64 for k in HELPERS},
    )


def test_control_manifest_rejects_old_methods_unfrozen_controls_and_resets():
    valid = controlled_manifest()
    validate_spec(valid)
    for field, value in [
        ("source_commit", "2312652dc52a894e9726f48321117cf114270355"),
        ("qualification_ledger_slot", 4),
        ("status", "DRAFT"),
        ("neural_witness_K", 9),
        ("fixed_epochs", True),
        ("original_training_deadline_epoch", 1001.0),
    ]:
        changed = copy.deepcopy(valid)
        changed[field] = value
        with pytest.raises(ValueError):
            validate_spec(changed)
    changed = copy.deepcopy(valid)
    changed["frozen_config"]["objective"]["behavior_kl_stop"] = 1.0
    with pytest.raises(ValueError):
        validate_spec(changed)
    changed = copy.deepcopy(valid)
    changed["inputs"].pop("protocol")
    with pytest.raises(ValueError):
        validate_spec(changed)


def test_completed_receipt_publication_never_overwrites(tmp_path):
    path = tmp_path / "receipt.json"
    publish(path, {"status": "original"})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        publish(path, {"status": "different"})
    assert path.read_bytes() == before
    assert not path.with_name(".receipt.json.tmp").exists()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return dict(path=str(path), sha256=sha(path))


def synthetic_eligibility_fixture(tmp_path):
    # Hand-authored consistency fixture exercises fail-closed joins, not a valid model.
    rows = []
    files = (
        "model.safetensors",
        "base.safetensors",
        "behavior.safetensors",
        "training.pt",
        "actor.json",
        "last-frozen-epoch.json.gz",
    )
    for seed in (20261525, 20261526):
        run = tmp_path / str(seed) / "run"
        native = run / "checkpoints/epoch-00000008"
        state = dict(
            epoch=8,
            fresh_transitions=8 * 32768,
            actor_steps=8 * 256,
            search_rngs=[[]] * 128,
            pending_search_schedule="closed-empty",
            replay_buffer="closed-empty",
            optimizer_accepted_updates=1,
            sample_chain_sha256="a" * 64,
        )
        native.mkdir(parents=True)
        for name in files:
            (native / name).write_bytes(name.encode())
        write(native / "actor.json", state)
        artifacts = {name: sha(native / name) for name in files}
        write(
            native / "checkpoint.json",
            dict(
                schema="torch-search-acting-native-cuda-v2",
                run_config=dict(schema="torch-fresh-sparse-search-acting-v2"),
                source_commit=SOURCE,
                state=state,
                artifacts=artifacts,
            ),
        )
        write(
            run / "metadata.json",
            dict(
                schema="search-acting-supervised-run-v2", absolute_deadline_epoch=1000.0
            ),
        )
        compared = {"journal/epoch-00000002.json.gz": "journal"}
        compared.update({f"checkpoints/epoch-00000002/{name}": name for name in files})
        for relative, payload in compared.items():
            target = run / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload.encode())
        compared = {rel: sha(run / rel) for rel in compared}
        receipts = {}
        for index in range(9):
            path = tmp_path / str(seed) / f"audit-{index}.json"
            write(path, dict(epoch=index))
            receipts[str(path)] = sha(path)
        full = dict(
            schema="ufuk-search-acting-allnative-data-audit-v2",
            status="pass-all-fixed-search-acting-v2-native-own-data-search-ledgers",
            source_commit=SOURCE,
            seed=seed,
            fixed_epochs=8,
            finished_epoch=1200.0,
            deadline_epoch=1300.0,
            audited_native_checkpoints=9,
            independently_replayed_fresh_transitions=8 * 32768,
            prescribed_neural_roots_verified=108,
            raw_actor_packet_roots_verified=108,
            neural_search_roots_recomputed=120,
            schedule_and_all128search_rngs_verified=True,
            actual_mixed_mu_replayed=True,
            raw_policy_KL_reference="KL(frozen-raw-pi||current-raw-pi)",
            neural_witness_K=8,
            sample_chain_sha256="a" * 64,
            optimizer_updates_performed_by_this_audit=0,
            manifest_sha256="b" * 64,
            audit_sha256="c" * 64,
            immutable_percheckpoint_audit_receipt_sha256=receipts,
        )
        replay = dict(
            schema="ufuk-search-acting-independent-freshCLI-replay-v2",
            status="pass-exact-search-acting-v2-native2-all-six-payloads-and-journal",
            source_commit=SOURCE,
            seed=seed,
            duplicate_fresh_presentations=32768,
            original_deadline_epoch=1000.0,
            started_epoch=1100.0,
            audit_deadline_epoch=1700.0,
            finished_epoch=1200.0,
            compared_sha256=compared,
            manifest_sha256="b" * 64,
            script_sha256="d" * 64,
            original_run=str(run),
        )
        rows.append(
            dict(
                seed=seed,
                run=str(run),
                full_search_acting_audit=write(
                    tmp_path / str(seed) / "full.json", full
                ),
                fresh_search_acting_replay=write(
                    tmp_path / str(seed) / "replay.json", replay
                ),
            )
        )
    return dict(
        qualification_ledger_slot=5,
        source_commit=SOURCE,
        fixed_epochs=8,
        seeds=rows,
        audit_helper_sha256="c" * 64,
        replay_helper_sha256="d" * 64,
    )


def invoke(tmp_path, manifest, monkeypatch):
    path = tmp_path / "manifest.json"
    write(path, manifest)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eligibility",
            "--manifest",
            str(path),
            "--output",
            str(tmp_path / "eligible.json"),
        ],
    )
    eligibility.main()


def test_new_v2_join_and_same_data_mu_raw_reference_proof_required(
    tmp_path, monkeypatch
):
    manifest = synthetic_eligibility_fixture(tmp_path)
    invoke(tmp_path, manifest, monkeypatch)
    assert (
        json.loads((tmp_path / "eligible.json").read_text())["status"]
        == "eligible-both-fixedSEARCH_ACTINGv2-for-preregistered-strength-only"
    )
    (tmp_path / "eligible.json").unlink()
    for key, bad in [
        ("schema", "ufuk-ownsearch-allnative-data-audit-v1"),
        ("actual_mixed_mu_replayed", False),
        ("raw_policy_KL_reference", "KL(actual-mu||current-raw-pi)"),
        ("prescribed_neural_roots_verified", 18),
        ("audited_native_checkpoints", 8),
    ]:
        row = manifest["seeds"][0]["full_search_acting_audit"]
        original = json.loads(Path(row["path"]).read_text())
        changed = dict(original)
        changed[key] = bad
        row.update(write(Path(row["path"]), changed))
        with pytest.raises(AssertionError):
            invoke(tmp_path, manifest, monkeypatch)
        row.update(write(Path(row["path"]), original))
    row = manifest["seeds"][0]["fresh_search_acting_replay"]
    original = json.loads(Path(row["path"]).read_text())
    changed = copy.deepcopy(original)
    changed["compared_sha256"].pop("checkpoints/epoch-00000002/training.pt")
    row.update(write(Path(row["path"]), changed))
    with pytest.raises(AssertionError):
        invoke(tmp_path, manifest, monkeypatch)


def test_epoch1_report_guard_rejects_shadowed_collection_index_and_wrong_source():
    from own5_adapter_controls import validate_epoch_report

    validate_epoch_report(dict(epoch=1, source_commit=SOURCE), 1)
    for report in [
        dict(epoch=32, source_commit=SOURCE),
        dict(epoch=True, source_commit=SOURCE),
        dict(epoch=1, source_commit="a" * 40),
    ]:
        with pytest.raises(ValueError):
            validate_epoch_report(report, 1)


def test_same_production_packet_checks_reject_mover_mu_and_storage_mutations():
    from types import SimpleNamespace

    import own5_audit_core as core
    import torch

    from harbichess.training.torch_search_acting_learner import tensor_bits_equal

    raw = (0.2, 0.3, 0.5)
    row = SimpleNamespace(
        policy=raw,
        base_policy=raw,
        online_pre_wdl=raw,
        base_wdl=raw,
        behavior_policy=(0.8, 0.1, 0.1),
    )
    core.compare_actor_packet(row, raw, raw, raw, raw)
    with pytest.raises(AssertionError):
        core.compare_actor_packet(row, row.behavior_policy, raw, raw, raw)
    core.compare_terminal_packet((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    with pytest.raises(AssertionError):
        core.compare_terminal_packet((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    core.compare_search_packet(
        {"search_policy": [0.2, 0.8]},
        {"search_policy": [0.2, 0.8]},
        lambda x: json.dumps(x),
    )
    with pytest.raises(AssertionError):
        core.compare_search_packet(
            {"search_policy": [0.8, 0.2]},
            {"search_policy": [0.2, 0.8]},
            lambda x: json.dumps(x),
        )
    core.compare_frozen_storage(
        torch.tensor([0.0]), torch.tensor([0.0]), tensor_bits_equal
    )
    with pytest.raises(AssertionError):
        core.compare_frozen_storage(
            torch.tensor([-0.0]), torch.tensor([0.0]), tensor_bits_equal
        )


def test_fullauditor_receipt_binding_checks_actual_core_return_before_publication(
    monkeypatch,
):
    from types import SimpleNamespace

    import own5_full_audit as full

    def correct(*args):
        return SimpleNamespace(epoch=1), dict(epoch=1, source_commit=SOURCE)

    monkeypatch.setattr(full, "audit_epoch", correct)
    assert full.checked_audit_epoch(None, None, None, None, 1, True)[1]["epoch"] == 1

    def shadowed(*args):
        return SimpleNamespace(epoch=1), dict(epoch=32, source_commit=SOURCE)

    monkeypatch.setattr(full, "audit_epoch", shadowed)
    with pytest.raises(ValueError):
        full.checked_audit_epoch(None, None, None, None, 1, True)
