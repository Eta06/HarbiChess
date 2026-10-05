"""Root-invoked fixed-E prospective config factory; never chooses E or queries strength."""

import argparse
import copy
import json
from pathlib import Path

from own7_audit_support import publish, sha
from own7_infrastructure_evidence import check_cli_receipt, check_unit_receipt
from own7_strength_config import BOOKS, SEEDS
from own7_training_controller import validate_mc_barrier

SOURCE = "c022bc1605b44c3089439da5c6efb7bd4db4ff81"


def schedules(epochs):
    assert type(epochs) is int and epochs >= 8
    selected = [
        1,
        2,
        max(3, round(epochs / 4)),
        round(epochs / 2),
        round(3 * epochs / 4),
        epochs,
    ]
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


def check_ancestor_evidence(ancestor, helpers):
    assert ancestor["source_commit"] == "428a30f5e1658f3cf159844db547ff0147ade5a9"
    if ancestor["status"] == "failed-preserved":
        # Historical failure is ancestry only, never a qualified428 E1 proof.
        assert ancestor["schema"] == (
            "owned900-search-acting-v2-fullshape-development-v2-originalclock-bindingrepair"
        )
        assert ancestor["original900_budget_reset"] is False
        assert ancestor["original_started_epoch"] == 1791163947.632503
        assert ancestor["absolute_deadline_epoch"] == 1791164847.632503
        assert ancestor["absolute_deadline_epoch"] == ancestor["original_started_epoch"] + 900
        recorded = ancestor["frozen_helper_sha256"]
        for name in ("own6_audit_core.py", "own6_adapter_controls.py"):
            matching = [digest for path, digest in recorded.items() if Path(path).name == name]
            assert matching == [sha(Path(helpers) / name)]
        return False
    assert ancestor["qualified_production_core_sha256"] == sha(Path(helpers) / "own6_audit_core.py")
    assert ancestor["qualified_report_guard_sha256"] == sha(
        Path(helpers) / "own6_adapter_controls.py"
    )
    return True


def check_training_evidence(profile, qualification):
    assert profile["source_commit"] == qualification["source_commit"] == SOURCE
    assert profile["schema"] == "owned900-search-acting-v3-fullshape-development-v1"
    assert qualification["schema"] == ("ufuk-search-acting-v3-E1-fullchronological-audit-result-v1")
    assert "A100" in qualification["actual_device_name"]
    assert qualification["torch_version"] == "2.11.0+cu130"
    assert qualification["finished_epoch"] <= qualification["absolute_deadline_epoch"]
    assert qualification["new_selfplay_transitions_generated"] == 0
    assert (
        profile["status"]
        == "pass-one-search-acting-v3-development-epoch-and-fullchronological-audit"
    )
    assert profile["finished_epoch"] <= profile["absolute_deadline_epoch"]
    assert qualification["status"] == (
        "pass-actualCUDA-search-acting-v3-E1-all-data-FIRST8-LAST8-original128-masks-raw-packets-and-mutations"
    )
    assert type(qualification["audit_report"]["epoch"]) is int
    assert qualification["audit_report"]["epoch"] == 1
    assert qualification["audit_report"]["independently_replayed_mate_certificate_roots"] > 0
    assert (
        qualification["targeted_actual_data_mutations_rejected"]["actual_certificate_inventory"]
        is True
    )
    assert (
        qualification["audit_report"]["exact_certificate_inventory_and_fullhistory_rules"] is True
    )
    assert qualification["audit_report"]["certified_root_visit_budget_verified"] is True
    assert (
        qualification["audit_report"]["certified_root_full_legal_policy_support_verified"] is True
    )
    assert qualification["audit_report"]["neural_witness_K"] == 8
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
        "book-20261725",
        "book-20261726",
        "helpers",
        "books-provenance",
        "qualified-ancestor-helpers",
        "ancestor-auditor-receipt",
        "curriculum-qualification-run",
        "failed4-dependency-supplement",
        "curriculum-provenance",
        "curriculum-qualification",
        "unit59-receipt",
        "unit-case-inventory",
        "original45-cohort",
        "terminal45-barrier-config",
        "cohort67",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "template-sha256",
        "profile-receipt-sha256",
        "auditor-receipt-sha256",
        "mc-completion-barrier-sha256",
        "books-provenance-sha256",
        "curriculum-provenance-sha256",
        "curriculum-qualification-sha256",
        "unit59-receipt-sha256",
        "unit-case-inventory-sha256",
        "ancestor-auditor-receipt-sha256",
        "failed4-dependency-supplement-sha256",
        "original45-cohort-sha256",
        "terminal45-barrier-config-sha256",
        "cohort67-sha256",
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
    assert sha(a.original45_cohort) == a.original45_cohort_sha256
    original45_cohort = json.loads(a.original45_cohort.read_text())
    assert original45_cohort["slots"] == [4, 5]
    epochs = a.fixed_epochs
    selected = schedules(epochs)
    assert epochs == 8
    assert a.whole_training_seconds == 6000 and a.whole_audit_seconds == 9000
    assert (
        sha(a.training_book) == "1a5ca17664a828d669e58cf2bd5d9eeb7f980b83f20d3a4031d36e4ff0930ccb"
    )
    assert sha(a.curriculum_provenance) == a.curriculum_provenance_sha256
    assert sha(a.curriculum_qualification) == a.curriculum_qualification_sha256
    curriculum_qualification = json.loads(a.curriculum_qualification.read_text())
    check_cli_receipt(curriculum_qualification, a.curriculum_qualification_run, SOURCE, sha)
    assert sha(a.unit59_receipt) == a.unit59_receipt_sha256
    assert sha(a.unit_case_inventory) == a.unit_case_inventory_sha256
    cases = json.loads(a.unit_case_inventory.read_text())
    assert len(cases) == 59
    check_unit_receipt(json.loads(a.unit59_receipt.read_text()), SOURCE, cases)
    from own7_schedule_v3 import AUDIT_CUTOFF, LATEST_LATENCY_START, verify_previous

    assert a.earliest_training_epoch < AUDIT_CUTOFF < LATEST_LATENCY_START < 1791180000
    terminal45_binding = {
        "terminal45_barrier_config": str(a.terminal45_barrier_config.resolve()),
        "terminal45_barrier_config_sha256": a.terminal45_barrier_config_sha256,
    }
    verify_previous(terminal45_binding, sha)
    assert sha(a.cohort67) == a.cohort67_sha256
    cohort67 = json.loads(a.cohort67.read_text())
    assert cohort67["schema"] == "prospective-own67-latency-final-cohort-v1"
    assert cohort67["slots"] == [6, 7] and cohort67["latency_order"] == [6, 7]
    assert cohort67["admission7_deadline_epoch"] == 1791167100
    assert cohort67["prelatency_deadline_epoch"] == 1791173460
    assert cohort67["latest_latency_start_epoch"] == 1791174000
    assert cohort67["helper_sha256"] == sha(a.helpers / "own67_cohort.py")

    # Actual hard cutoffs can make this attempt INCOMPLETE; no promise that maxima fit.
    assert a.posttraining_reserve_seconds > 0
    evidence = {}
    for name in ("profile_receipt", "auditor_receipt", "mc_completion_barrier"):
        path = getattr(a, name)
        assert sha(path) == getattr(a, name + "_sha256")
        evidence[name] = json.loads(path.read_text())
    check_training_evidence(evidence["profile_receipt"], evidence["auditor_receipt"])
    validate_mc_barrier(evidence["mc_completion_barrier"], a.earliest_training_epoch)
    for relative, digest in evidence["auditor_receipt"]["audit_report"][
        "native_artifact_sha256"
    ].items():
        assert sha(a.development_run / relative) == digest
    changed_model_tensors = actual_model_changed(a.development_run)
    assert sha(a.weights) == "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
    assert sha(a.books_provenance) == a.books_provenance_sha256
    provenance = json.loads(a.books_provenance.read_text())
    assert provenance["qualification_ledger_slot"] == 7
    assert provenance["status"] == "pass-roots-only-no-games"
    assert provenance["all_96_records_independently_replayed"] is True
    assert provenance["root_overlap"] == provenance["source_overlap"] == 0
    assert provenance["unique_source_records_across_both"] == 96
    assert set(provenance["selection_to_training_seed"].values()) == set(SEEDS)
    BOOKS.clear()
    BOOKS.update(
        {
            seed: provenance["book_sha256"][selection]
            for selection, seed in provenance["selection_to_training_seed"].items()
        }
    )
    for selection, seed in provenance["selection_to_training_seed"].items():
        assert provenance["root_counts"][selection] == 48
        assert BOOKS[seed] == provenance["book_sha256"][selection]
    assert len(set(BOOKS.values())) == 2
    for seed in SEEDS:
        assert sha(getattr(a, "book_" + str(seed))) == BOOKS[seed]
    assert sha(a.ancestor_auditor_receipt) == a.ancestor_auditor_receipt_sha256
    ancestor = json.loads(a.ancestor_auditor_receipt.read_text())
    ancestor_profile_qualified = check_ancestor_evidence(ancestor, a.qualified_ancestor_helpers)
    assert evidence["auditor_receipt"]["qualified_production_core_sha256"] == sha(
        a.helpers / "own7_audit_core.py"
    )
    assert evidence["auditor_receipt"]["qualified_report_guard_sha256"] == sha(
        a.helpers / "own7_adapter_controls.py"
    )
    assert sha(a.failed4_dependency_supplement) == a.failed4_dependency_supplement_sha256
    scheduling_repair = json.loads(a.failed4_dependency_supplement.read_text())
    assert (
        scheduling_repair["schema"] == "own5-failed4-dependency-analysis-v3-control-supplement-v1"
    )
    assert scheduling_repair["qualification_ledger_slot"] == 5
    helpers = {path.name: sha(path) for path in sorted(a.helpers.glob("own7_*.py"))}
    helpers["own67_cohort.py"] = sha(a.helpers / "own67_cohort.py")
    helpers["qualify_source_cli.py"] = sha(a.helpers / "qualify_source_cli.py")
    required = {
        "own7_full_audit.py",
        "own7_audit_support.py",
        "own7_training_controller.py",
        "own7_audit_controller.py",
        "own7_eligibility.py",
        "own7_fresh_cli_replay.py",
        "own7_baseline.py",
        "own7_final_arms.py",
        "own7_strength_analysis.py",
        "own7_parity.py",
        "own7_latency.py",
        "own7_all_gates.py",
        "own7_strength_config.py",
        "own7_strength_runtime.py",
        "own7_posttraining.py",
    }
    # Method7 does not rewrite or join the immutable original45 cohort.
    assert required <= set(helpers)
    a.output.mkdir(parents=True, exist_ok=False)
    draft = json.loads(a.template.read_text())
    registration = {
        **draft,
        "schema": "ufuk-search-acting-method7-prospective-v1",
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
        "scheduling_version": "prospective-own7-terminal45-scheduling-v3",
        "absolute_audit_cutoff_epoch": AUDIT_CUTOFF,
        "latest_latency_start_epoch": LATEST_LATENCY_START,
        "terminal45_barrier_binding": terminal45_binding,
        "cohort67": {
            "manifest": str(a.cohort67.resolve()),
            "manifest_sha256": a.cohort67_sha256,
            "helper_sha256": helpers["own67_cohort.py"],
        },
        "infrastructure_profile_pass": True,
        "development_model_storage_changed_tensor_names": changed_model_tensors,
        "development_model_sha256": {
            str(epoch): sha(
                a.development_run / "checkpoints" / f"epoch-{epoch:08d}" / "model.safetensors"
            )
            for epoch in (0, 1)
        },
        "helper_sha256": helpers,
        "own_search_ledger_schema": "pre-action-masked-search-behavior-v3",
        "controller_sha256": helpers["own7_training_controller.py"],
        "curriculum_book_sha256": sha(a.training_book),
        "protection_original45_manifest": terminal45_binding,
        "curriculum_provenance_sha256": a.curriculum_provenance_sha256,
        "source7_real_e8_CLI600_sha256": a.curriculum_qualification_sha256,
        "source7_unit59_case_inventory_sha256": a.unit_case_inventory_sha256,
        "actualc022_CUDA59_receipt_sha256": a.unit59_receipt_sha256,
        "ancestor_auditor_receipt_sha256": a.ancestor_auditor_receipt_sha256,
        "ancestor_profile_qualified": ancestor_profile_qualified,
        "qualified_ancestor_source_commit": "428a30f5e1658f3cf159844db547ff0147ade5a9",
        "qualified_ancestor_core_sha256": sha(a.qualified_ancestor_helpers / "own6_audit_core.py"),
        "qualified_ancestor_guard_sha256": sha(
            a.qualified_ancestor_helpers / "own6_adapter_controls.py"
        ),
        "mc_completion_barrier": evidence["mc_completion_barrier"],
        "prospective_evidence_sha256": {name: getattr(a, name + "_sha256") for name in evidence},
    }
    registration["strength"]["frozen_books"] = {str(k): v for k, v in BOOKS.items()}
    registration["book_selection_seeds"] = list(map(int, provenance["selection_to_training_seed"]))
    registration["books_provenance_sha256"] = a.books_provenance_sha256
    registration.pop("blockers", None)
    inputs = {"seeds": {}}
    for seed, config in registration["configs"].items():
        path = a.output / f"config-{seed}.json"
        publish(path, config)
        inputs["seeds"][seed] = {
            "initial_weights": {
                "path": str(a.weights.resolve()),
                "sha256": sha(a.weights),
            },
            "book": {
                "path": str(a.training_book.resolve()),
                "sha256": sha(a.training_book),
            },
            "experiment_config": {"path": str(path.resolve()), "sha256": sha(path)},
        }
    input_path = a.output / "three-input-manifest.json"
    publish(input_path, inputs)
    registration["three_input_manifest_sha256"] = sha(input_path)
    publish(a.output / "registration.json", registration)
    bindings = {
        "schema": "ufuk-own7-strength-bindings-v1",
        "status": registration["status"],
        "qualification_ledger_slot": 7,
        "MAX_families": 8,
        "seeds": list(SEEDS),
        "books_sha256": {str(k): v for k, v in BOOKS.items()},
        "books_provenance_sha256": a.books_provenance_sha256,
        "source_commit": SOURCE,
        "native_schema": "torch-search-acting-native-cuda-v2",
        "fixed_epochs": epochs,
        "games_per_arm": 96,
        "arm_wall_seconds": 3600,
        "runtime_scope": "same-A100-CPU-one-thread-Torch2.11-all-six-arms",
        "helper_sha256": helpers,
        "own_search_ledger_schema": "pre-action-masked-search-behavior-v3",
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
