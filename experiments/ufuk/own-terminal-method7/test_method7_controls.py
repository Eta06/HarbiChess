import ast
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
BASE = (
    Path(__import__("os").environ.get("UFUK_REPOSITORY", "/workspace/HarbiChess"))
    / "experiments/ufuk/own-terminal-method6"
)
SOURCE = "c022bc1605b44c3089439da5c6efb7bd4db4ff81"

for parent in ROOT.parents:
    if (parent / "pyproject.toml").is_file() and (
        parent / "experiments/ufuk/own-terminal-method6"
    ).is_dir():
        BASE = parent / "experiments/ufuk/own-terminal-method6"
        break


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_source_epoch_seed_and_same_training_hyperparameters():
    draft = json.loads((ROOT / "protocol-DRAFT.json").read_text())
    base = json.loads((BASE / "protocol-DRAFT.json").read_text())
    assert draft["source_commit"] == SOURCE and draft["fixed_epochs"] == 8
    assert draft["status"].startswith("UNFROZEN")
    for seed in (20261725, 20261726):
        cfg = copy.deepcopy(draft["configs"][str(seed)])
        cfg["seed"] = seed - 100
        assert cfg == base["configs"][str(seed - 100)]
    assert load("own7_adapter_controls").SEEDS == (20261725, 20261726)


def test_actual_baseline_and_final_argparse_do_not_use_missing_runtime_seeds(monkeypatch, capsys):
    for name in ("own7_baseline", "own7_final_arms"):
        module = load(name)
        assert module.SEEDS == (20261725, 20261726)
        monkeypatch.setattr(sys, "argv", [name, "--help"])
        with pytest.raises(SystemExit) as ended:
            module.main()
        assert ended.value.code == 0
        assert "20261725" in capsys.readouterr().out


def test_source7_training_evidence_requires_actual_certificate_and_old_source_rejected():
    factory = load("own7_formal_config_factory")
    q = {
        "source_commit": SOURCE,
        "schema": "ufuk-search-acting-v3-E1-fullchronological-audit-result-v1",
        "status": (
            "pass-actualCUDA-search-acting-v3-E1-all-data-FIRST8-LAST8-"
            "original128-masks-raw-packets-and-mutations"
        ),
        "actual_device_name": "NVIDIA A100",
        "torch_version": "2.11.0+cu130",
        "finished_epoch": 100,
        "absolute_deadline_epoch": 200,
        "new_selfplay_transitions_generated": 0,
        "optimizer_updates_performed_by_qualification": 0,
        "targeted_actual_data_mutations_rejected": {
            "raw": True,
            "actual_certificate_inventory": True,
        },
        "audit_report": {
            "epoch": 1,
            "neural_witness_K": 8,
            "raw_actor_replayed": 32768,
            "optimizer_committed": 1,
            "raw_actor_packet_roots_verified": 18,
            "independently_replayed_mate_certificate_roots": 1,
            "exact_certificate_inventory_and_fullhistory_rules": True,
            "certified_root_visit_budget_verified": True,
            "certified_root_full_legal_policy_support_verified": True,
        },
    }
    profile = {
        "source_commit": SOURCE,
        "schema": "owned900-search-acting-v3-fullshape-development-v1",
        "status": "pass-one-search-acting-v3-development-epoch-and-fullchronological-audit",
        "finished_epoch": 100,
        "absolute_deadline_epoch": 200,
    }
    factory.check_training_evidence(profile, q)
    for patch in (
        {"source_commit": "428a30f5e1658f3cf159844db547ff0147ade5a9"},
        {"audit_report": {**q["audit_report"], "epoch": True}},
        {
            "audit_report": {
                **q["audit_report"],
                "independently_replayed_mate_certificate_roots": 0,
            }
        },
        {"targeted_actual_data_mutations_rejected": {"actual_certificate_inventory": False}},
    ):
        with pytest.raises(AssertionError):
            factory.check_training_evidence(profile, {**q, **patch})


def test_actual_unit_inventory_old428_or_skip_cannot_qualify_source7():
    module = load("own7_infrastructure_evidence")
    cases = [f"test-case-{i}" for i in range(59)]
    receipt = {
        "schema": "ufuk-clean-producer-actualCUDA-unit-suite-v1",
        "status": "pass-actualCUDA59-no-skip",
        "source_commit": SOURCE,
        "cases": cases,
        "tests_passed": 59,
        "tests_skipped": 0,
        "tests_failed": 0,
        "actualCUDA": True,
        "source_clean": True,
        "device_name": "NVIDIA A100",
        "torch_version": "2.11.0+cu130",
    }
    module.check_unit_receipt(receipt, SOURCE, cases)
    for patch in (
        {"source_commit": "428a30f5e1658f3cf159844db547ff0147ade5a9"},
        {"tests_skipped": 1},
        {"cases": cases[:-1]},
        {"actualCUDA": False},
    ):
        with pytest.raises(AssertionError):
            module.check_unit_receipt({**receipt, **patch}, SOURCE, cases)


def test_helper_manifest_inventory_matches_certificate_guard_six_names():
    owner = load("own7_owner_config_factory")
    guard = load("own7_adapter_controls")
    assert set(owner.AUDIT_HELPERS) == guard.HELPERS
    for name in owner.AUDIT_HELPERS:
        assert (ROOT / name).is_file()
    factory = ast.unparse(ast.parse((ROOT / "own7_formal_config_factory.py").read_text()))
    assert "helpers['qualify_source_cli.py']" in factory
    assert "qualified_production_core_sha256" in factory
    assert "own6_audit_core.py" in factory  # Honest separate428 ancestry.


def test_strength_statistical_functions_and_threshold_function_unchanged():
    def functions(path):
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name != "main"
        }

    assert functions(ROOT / "own7_strength_analysis.py") == functions(
        BASE / "own6_strength_analysis.py"
    )
    assert (
        functions(ROOT / "own7_all_gates.py")["require_strength_gates"]
        == functions(BASE / "own6_all_gates.py")["require_strength_gates"]
    )


def test_new_cutoffs_and_physical_phase_cap():
    schedule = load("own7_schedule_v3")
    assert schedule.AUDIT_CUTOFF == 1791172200 and schedule.LATEST_LATENCY_START == 1791174000
    assert schedule.audit_deadline(1791164000, 9000) == 1791172200
    deadline = schedule.phase_deadline(schedule.LATEST_LATENCY_START + 180, 7300)
    assert deadline == 1791180000 - 1
    assert (
        float(schedule.bind_argument_deadline(["--deadline-epoch", 1791180000], deadline)[1])
        == deadline
    )


def test_receipt_slot7_and_native_v2_ledger_v3_are_honest():
    for name in (
        "own7_strength_analysis.py",
        "own7_parity.py",
        "own7_latency.py",
        "own7_all_gates.py",
    ):
        text = (ROOT / name).read_text()
        assert "qualification_ledger_slot=6" not in text
        assert "qualification_ledger_slot=7" in text
    guard = load("own7_adapter_controls")
    assert guard.NATIVE_SCHEMA == "torch-search-acting-native-cuda-v2"
    assert "pre-action-masked-search-behavior-v3" in (ROOT / "own7_audit_core.py").read_text()
    assert "actual_certificate_inventory" in (ROOT / "own7_qualify_e1.py").read_text()


def test_actual_owner_factory_cli_has_no_output_sha_and_binds_six_audit_helpers(
    monkeypatch, capsys
):
    owner = load("own7_owner_config_factory")
    monkeypatch.setattr(sys, "argv", ["own7_owner_config_factory", "--help"])
    with pytest.raises(SystemExit) as stopped:
        owner.main()
    assert stopped.value.code == 0
    helptext = capsys.readouterr().out
    assert "--output-sha256" not in helptext and "--paths-config-sha256" in helptext
    text = (ROOT / "own7_owner_config_factory.py").read_text()
    assert "ufuk-search-acting-formal7-audit-manifest-v1" in text
    assert "own7-common-original-firstclock-v1" in text


def test_every_post_launch_runtime_call_binds_actual_signature():
    import inspect

    tree = ast.parse((ROOT / "own7_posttraining.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    definition = next(
        n for n in ast.walk(main) if isinstance(n, ast.FunctionDef) and n.name == "launch"
    )
    scope = {}
    copy_definition = copy.deepcopy(definition)
    copy_definition.body = [ast.Pass()]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[copy_definition], type_ignores=[])),
            "actual-launch-signature",
            "exec",
        ),
        scope,
    )
    signature = inspect.signature(scope["launch"])
    calls = [
        n
        for n in ast.walk(main)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "launch"
    ]
    assert calls
    for call in calls:
        signature.bind(*[object() for _ in call.args], **{kw.arg: object() for kw in call.keywords})


def test_actual_failed428_ancestor_never_becomes_qualified_and_clock_is_not_reset(tmp_path):
    factory = load("own7_formal_config_factory")
    for name in ("own6_audit_core.py", "own6_adapter_controls.py"):
        (tmp_path / name).write_bytes(name.encode())
    ancestor = {
        "source_commit": "428a30f5e1658f3cf159844db547ff0147ade5a9",
        "status": "failed-preserved",
        "schema": "owned900-search-acting-v2-fullshape-development-v2-originalclock-bindingrepair",
        "original900_budget_reset": False,
        "original_started_epoch": 1791163947.632503,
        "absolute_deadline_epoch": 1791164847.632503,
        "frozen_helper_sha256": {
            str(tmp_path / name): factory.sha(tmp_path / name)
            for name in ("own6_audit_core.py", "own6_adapter_controls.py")
        },
    }
    assert factory.check_ancestor_evidence(ancestor, tmp_path) is False
    for mutation in (
        {"original900_budget_reset": True},
        {"absolute_deadline_epoch": 1791164848.632503},
        {"source_commit": factory.SOURCE},
    ):
        with pytest.raises(AssertionError):
            factory.check_ancestor_evidence({**ancestor, **mutation}, tmp_path)
    (tmp_path / "own6_audit_core.py").write_bytes(b"changed")
    with pytest.raises(AssertionError):
        factory.check_ancestor_evidence(ancestor, tmp_path)
