"""Root-invoked fixed-E prospective config factory; never chooses E or queries strength."""

import argparse
import copy
import json
from pathlib import Path

from ownv1_audit_support import publish, sha
from ownv1_strength_config import BOOKS, SEEDS
from ownv1_training_controller import validate_mc_barrier

SOURCE = "a278bba67bce962cb9294f0d24040e02e9acf1f4"


def schedules(epochs):
    assert type(epochs) is int and epochs >= 8
    selected = [1, 2, max(3, round(epochs / 4)), round(epochs / 2), round(3 * epochs / 4), epochs]
    assert len(set(selected)) == 6 and selected == sorted(selected)
    return selected


def configs(epochs, template=None):
    schedules(epochs)
    root = Path(__file__).parent
    draft = json.loads(Path(template or root / "protocol-DRAFT.json").read_text())
    values = copy.deepcopy(draft["configs"])
    for seed in SEEDS:
        config = values[str(seed)]
        assert config["seed"] == seed and config["learning_rate"] == 0.000025
    return values


def check_training_evidence(profile, qualification):
    assert profile["source_commit"] == qualification["source_commit"] == SOURCE
    assert profile["status"] == "pass-one-development-epoch-and-readonly-audit"
    assert profile["finished_epoch"] <= profile["absolute_deadline_epoch"]
    assert qualification["status"] == (
        "pass-actualCUDA-E1-full-data-original-groups-raw-packets-and-targeted-mutations"
    )
    assert qualification["audit_report"]["epoch"] == 1
    assert qualification["audit_report"]["raw_actor_replayed"] == 32768
    assert qualification["audit_report"]["optimizer_committed"] > 0
    assert qualification["audit_report"]["raw_actor_packet_roots_verified"] == 18
    assert all(qualification["targeted_actual_data_mutations_rejected"].values())
    assert qualification["optimizer_updates_performed_by_qualification"] == 0


def actual_model_changed(run):
    import numpy as np
    from safetensors.numpy import load_file

    before = load_file(str(Path(run) / "checkpoints/epoch-00000000/model.safetensors"))
    after = load_file(str(Path(run) / "checkpoints/epoch-00000001/model.safetensors"))
    assert set(before) == set(after)
    for key in before:
        assert before[key].shape == after[key].shape and before[key].dtype == after[key].dtype
    changed = [
        key
        for key in before
        if not np.array_equal(
            np.ascontiguousarray(before[key]).view(np.uint8),
            np.ascontiguousarray(after[key]).view(np.uint8),
        )
    ]
    assert changed, "Retained optimizer counts alone do not prove model storage changed"
    return changed


def main():
    p = argparse.ArgumentParser()
    for name in (
        "template",
        "development-run",
        "output",
        "profile-receipt",
        "auditor-receipt",
        "mc-completion-barrier",
        "weights",
        "training-book",
        "book-20261425",
        "book-20261426",
        "helpers",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "template-sha256",
        "profile-receipt-sha256",
        "auditor-receipt-sha256",
        "mc-completion-barrier-sha256",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--fixed-epochs", type=int, required=True)
    p.add_argument("--whole-training-seconds", type=int, required=True)
    p.add_argument("--whole-audit-seconds", type=int, required=True)
    p.add_argument("--earliest-training-epoch", type=float, required=True)
    p.add_argument("--posttraining-reserve-seconds", type=int, required=True)
    p.add_argument("--freeze", action="store_true")
    a = p.parse_args()
    assert sha(a.template) == a.template_sha256
    epochs = a.fixed_epochs
    selected = schedules(epochs)
    assert 0 < a.whole_training_seconds <= a.whole_audit_seconds
    # Preserve every declared whole-stage ceiling, both fresh replays, and parallel seeds.
    minimum_reserve = 1200 + 120 + 120 + 60 + 180 + 7300 + 120 + 120 + 180 + 120
    assert a.posttraining_reserve_seconds >= minimum_reserve
    assert (
        a.earliest_training_epoch + a.whole_audit_seconds + a.posttraining_reserve_seconds
        < 1791170400
    )
    evidence = {}
    for name in ("profile_receipt", "auditor_receipt", "mc_completion_barrier"):
        path = getattr(a, name)
        assert sha(path) == getattr(a, name + "_sha256")
        evidence[name] = json.loads(path.read_text())
    check_training_evidence(evidence["profile_receipt"], evidence["auditor_receipt"])
    validate_mc_barrier(evidence["mc_completion_barrier"], a.earliest_training_epoch)
    for relative, digest in evidence["profile_receipt"]["audit_result"][
        "native_artifact_sha256"
    ].items():
        assert sha(a.development_run / relative) == digest
    changed_model_tensors = actual_model_changed(a.development_run)
    assert sha(a.weights) == "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
    for seed in SEEDS:
        assert sha(getattr(a, "book_" + str(seed))) == BOOKS[seed]
    helpers = {path.name: sha(path) for path in sorted(a.helpers.glob("ownv1_*.py"))}
    required = {
        "ownv1_full_audit.py",
        "ownv1_audit_support.py",
        "ownv1_training_controller.py",
        "ownv1_audit_controller.py",
        "ownv1_eligibility.py",
        "ownv1_fresh_cli_replay.py",
        "ownv1_baseline.py",
        "ownv1_final_arms.py",
        "ownv1_strength_analysis.py",
        "ownv1_parity.py",
        "ownv1_latency.py",
        "ownv1_all_gates.py",
        "ownv1_strength_config.py",
        "ownv1_strength_runtime.py",
        "ownv1_posttraining.py",
    }
    cohort_helper = a.helpers / "own45_cohort.py"
    if cohort_helper.is_file():
        helpers[cohort_helper.name] = sha(cohort_helper)
    assert required <= set(helpers)
    a.output.mkdir(parents=True, exist_ok=False)
    draft = json.loads(a.template.read_text())
    registration = {
        **draft,
        "schema": "ufuk-ownsearch-method4-prospective-v1",
        "status": "frozen-before-formal-execution" if a.freeze else "NOT_FROZEN_DO_NOT_EXECUTE",
        "source_commit": SOURCE,
        "earliest_training_epoch": a.earliest_training_epoch,
        "fixed_epochs": epochs,
        "whole_training_seconds_per_seed": a.whole_training_seconds,
        "whole_audit_seconds_from_originalfirstclock": a.whole_audit_seconds,
        "configs": configs(epochs, a.template),
        "factory_template_sha256": a.template_sha256,
        "neural_audit_epochs": selected,
        "posttraining_reserve_seconds": a.posttraining_reserve_seconds,
        "infrastructure_profile_pass": True,
        "development_model_storage_changed_tensor_names": changed_model_tensors,
        "development_model_sha256": {
            str(epoch): sha(
                a.development_run / "checkpoints" / f"epoch-{epoch:08d}" / "model.safetensors"
            )
            for epoch in (0, 1)
        },
        "helper_sha256": helpers,
        "controller_sha256": helpers["ownv1_training_controller.py"],
        "mc_completion_barrier": evidence["mc_completion_barrier"],
        "prospective_evidence_sha256": {name: getattr(a, name + "_sha256") for name in evidence},
    }
    registration.pop("blockers", None)
    inputs = {"seeds": {}}
    for seed, config in registration["configs"].items():
        path = a.output / f"config-{seed}.json"
        publish(path, config)
        inputs["seeds"][seed] = {
            "initial_weights": {"path": str(a.weights.resolve()), "sha256": sha(a.weights)},
            "book": {"path": str(a.training_book.resolve()), "sha256": sha(a.training_book)},
            "experiment_config": {"path": str(path.resolve()), "sha256": sha(path)},
        }
    input_path = a.output / "three-input-manifest.json"
    publish(input_path, inputs)
    registration["three_input_manifest_sha256"] = sha(input_path)
    publish(a.output / "registration.json", registration)
    bindings = {
        "schema": "ufuk-ownv1-strength-bindings-v1",
        "status": registration["status"],
        "qualification_ledger_slot": 4,
        "MAX_families": 8,
        "seeds": list(SEEDS),
        "books_sha256": {str(k): v for k, v in BOOKS.items()},
        "source_commit": SOURCE,
        "native_schema": "torch-ownsearch-native-cuda-v1",
        "fixed_epochs": epochs,
        "games_per_arm": 96,
        "arm_wall_seconds": 3600,
        "runtime_scope": "same-A100-CPU-one-thread-Torch2.11-all-six-arms",
        "helper_sha256": helpers,
    }
    publish(a.output / "strength-bindings.json", bindings)
    publish(
        a.output / "factory-receipt.json",
        {
            "status": registration["status"],
            "factory_sha256": sha(__file__),
            "fixed_epochs_explicit_root_argument": epochs,
            "registration_sha256": sha(a.output / "registration.json"),
            "strength_bindings_sha256": sha(a.output / "strength-bindings.json"),
            "note": (
                "No strength packets read. Native protocol is fourth input; "
                "no registration self-hash."
            ),
        },
    )


if __name__ == "__main__":
    main()
