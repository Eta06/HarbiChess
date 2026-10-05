import ast
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
REPO = Path.cwd()
ANCESTOR = REPO / "experiments/ufuk/search-acting-method5"


def load(name):
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    if name == "own6_protect_original45_v2":
        # Exact historical guardian needs its exact historical sibling at import time.
        original = sys.modules.get("own6_previous_methods_barrier")
        historical = load("own6_previous_methods_barrier_v2")
        sys.modules["own6_previous_methods_barrier"] = historical
        try:
            spec.loader.exec_module(module)
        finally:
            if original is None:
                sys.modules.pop("own6_previous_methods_barrier", None)
            else:
                sys.modules["own6_previous_methods_barrier"] = original
    else:
        spec.loader.exec_module(module)
    return module


def test_qualified_native_audit_core_all_functions_unchanged():
    def functions(path):
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef)
        }

    assert functions(ROOT / "own6_audit_core.py") == functions(ANCESTOR / "own5_audit_core.py")


def test_exact_ancestor_hyperparameters_only_seed_changes():
    draft = json.loads((ROOT / "protocol-DRAFT.json").read_text())
    ancestor = json.loads((ANCESTOR / "frozen/registration.json").read_text())
    for old_seed, new_seed in ((20261525, 20261625), (20261526, 20261626)):
        actual = copy.deepcopy(draft["configs"][str(new_seed)])
        actual["seed"] = old_seed
        assert actual == ancestor["configs"][str(old_seed)]
    assert draft["fixed_epochs"] == 8
    assert draft["whole_training_seconds_per_seed"] == 6000
    assert draft["whole_audit_seconds_from_originalfirstclock"] == 9000


def test_dynamic_books_strict_full_provenance_binding():
    mod = load("own6_strength_config")
    q = {
        "schema": "ufuk-own6-strength-bindings-v1",
        "status": "frozen-before-formal-execution",
        "qualification_ledger_slot": 6,
        "MAX_families": 8,
        "seeds": [20261625, 20261626],
        "books_sha256": {"20261625": "a" * 64, "20261626": "b" * 64},
        "books_provenance_sha256": "c" * 64,
        "fixed_epochs": 8,
        "source_commit": mod.SOURCE,
        "native_schema": "torch-search-acting-native-cuda-v2",
        "games_per_arm": 96,
        "arm_wall_seconds": 3600,
        "runtime_scope": "same-A100-CPU-one-thread-Torch2.11-all-six-arms",
    }
    assert mod.validate_config(q) == q
    assert mod.BOOKS == {20261625: "a" * 64, 20261626: "b" * 64}
    for change in (
        {"qualification_ledger_slot": 5},
        {"seeds": [20261525, 20261526]},
        {"books_sha256": {"20261625": "a" * 64, "20261626": "a" * 64}},
        {"source_commit": "a" * 40},
    ):
        with pytest.raises(AssertionError):
            mod.validate_config({**q, **change})


def test_new_real06_deadlines_and_native_v2_schema():
    for p in ROOT.glob("own6_*.py"):
        assert "1791170400" not in p.read_text()
    train = load("own6_training_controller")
    reg = {
        "earliest_training_epoch": 100,
        "whole_training_seconds_per_seed": 6000,
        "whole_audit_seconds_from_originalfirstclock": 9000,
        "posttraining_reserve_seconds": 10020,
        "scheduling_version": "prospective-own6-terminal45-scheduling-v3",
        "absolute_audit_cutoff_epoch": 1791170700,
        "latest_latency_start_epoch": 1791172800,
    }
    train.validate_training_clock(reg, 101, 6101, 102)
    with pytest.raises(AssertionError):
        train.validate_training_clock(reg, 103, 6103, 102)


def test_owner_output_has_no_input_sha_flag():
    text = (ROOT / "own6_owner_config_factory.py").read_text()
    assert '"output",' in text
    assert '"paths-config",' in text
    assert '"--output-sha256"' not in text
    assert "own6-common-original-firstclock-v1" in text and "== [6]" in text


def test_actual_post_launch_signature_binding_and_old45_barrier():
    tree = ast.parse((ROOT / "own6_posttraining.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    launch = next(
        n for n in ast.walk(main) if isinstance(n, ast.FunctionDef) and n.name == "launch"
    )
    required = len(launch.args.args) - len(launch.args.defaults)
    for call in ast.walk(main):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "launch"
        ):
            assert required <= len(call.args) <= len(launch.args.args)
    text = (ROOT / "own6_posttraining.py").read_text()
    assert "wait_previous(c, wait_json, sha)" in text
    assert "own45_cohort" not in text
    assert "LATEST_LATENCY_START" in text
    for name in (
        "own5_final_arms",
        "ownv1_final_arms",
        "torch_search_acting_run",
        "portable_arena",
    ):
        assert name in text


def test_no_inferential_count_or_bootstrap_change():
    analysis = load("own6_strength_analysis")
    assert analysis.REPLICATES == 50000 and analysis.CONFIDENCE == 0.9875
    assert '"primary_comparisons": 4' in (ROOT / "own6_strength_analysis.py").read_text()
    assert 'strength["primary_comparisons"] == 4' in (ROOT / "own6_all_gates.py").read_text()


def test_protection_descendants_cross_groups_and_pid_reuse_isolation():
    guard = load("own6_protect_original45_v2")
    owners = [{"pid": 100, "startticks": 1000}]
    table = {
        100: {"pid": 100, "startticks": 1000, "ppid": 1, "pgid": 100},
        101: {"pid": 101, "startticks": 1001, "ppid": 100, "pgid": 101},
        102: {"pid": 102, "startticks": 1002, "ppid": 101, "pgid": 102},
        200: {"pid": 200, "startticks": 2000, "ppid": 1, "pgid": 200},
    }
    tracked = guard.own_descendants(owners, table)
    assert set(tracked) == {100, 101, 102}
    changed = copy.deepcopy(table)
    changed[101]["startticks"] = 9999
    signals = []
    guard.stop_targets(tracked, changed, lambda pid, sig: signals.append((pid, sig)), 15)
    assert {pid for pid, sig in signals} == {100, 102}
    assert all(pid != 200 for pid, sig in signals)
    assert guard.own_descendants([{"pid": 100, "startticks": 9999}], table) == {}


def test_protection_disappeared_child_does_not_interrupt_cleanup():
    guard = load("own6_protect_original45_v2")
    tracked = {100: {"pid": 100, "startticks": 1}, 101: {"pid": 101, "startticks": 2}}
    signals = []

    def send(pid, sig):
        if pid == 100:
            raise ProcessLookupError()
        signals.append(pid)

    assert guard.stop_targets(tracked, tracked, send, 15) == [101]
    assert signals == [101]


def test_actual_baseline_main_help_and_all_runtime_exports(monkeypatch, capsys):
    baseline = load("own6_baseline")
    monkeypatch.setattr(sys, "argv", ["own6_baseline.py", "--help"])
    with pytest.raises(SystemExit) as exitcode:
        baseline.main()
    assert exitcode.value.code == 0
    assert "20261625" in capsys.readouterr().out
    tree = ast.parse((ROOT / "own6_baseline.py").read_text())
    attributes = {
        n.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "c"
    }
    runtime = load("own6_strength_runtime")
    assert all(hasattr(runtime, name) for name in attributes)
    assert runtime.HARD_DEADLINE == 1791180000


def test_actual_final_arm_main_help(monkeypatch, capsys):
    final = load("own6_final_arms")
    monkeypatch.setattr(sys, "argv", ["own6_final_arms.py", "--help"])
    with pytest.raises(SystemExit) as exitcode:
        final.main()
    assert exitcode.value.code == 0
    assert "20261625" in capsys.readouterr().out


def test_protection_prunes_foreign_owned_branch_before_tracking_or_failure_cleanup():
    mod = load("own6_protect_original45_v2")
    table = {
        10: {"pid": 10, "startticks": 1, "ppid": 1, "pgid": 10, "state": "S"},
        20: {"pid": 20, "startticks": 2, "ppid": 10, "pgid": 20, "state": "S"},
        30: {"pid": 30, "startticks": 3, "ppid": 20, "pgid": 30, "state": "S"},
        40: {"pid": 40, "startticks": 4, "ppid": 10, "pgid": 40, "state": "S"},
    }
    tracked = mod.own_descendants([table[10]], table, {(20, 2)})
    assert set(tracked) == {10, 40}
    sent = []
    mod.stop_targets(tracked, table, lambda pid, sig: sent.append(pid), 15)
    assert sent == [10, 40]


def test_new428_cli_receipt_binds_both_payload_sets_and_rejects_ancestor_source(tmp_path):
    import hashlib

    import pytest

    mod = load("own6_infrastructure_evidence")

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    paths = {f"journal/epoch-{i:08d}.json.gz" for i in (1, 2)}
    paths |= {f"checkpoints/epoch-00000002/{name}" for name in mod.PAYLOADS}
    receipt = {
        "schema": "actual-CUDA-search-acting-CLI-qualification-v2",
        "status": "pass",
        "source_commit": mod.SOURCE,
        "input_sha256": {"weights": mod.E8, "book": mod.BOOK},
        "both_final_full_native_freshprocess_strictload": True,
        "per_run_epochs": 2,
        "finished_epoch": 10,
        "absolute_deadline_epoch": 11,
        "last_epoch_policy_target_rows": 1,
        "last_epoch_known_terminal_rows": 2,
        "last_epoch_UNKNOWN_excluded_value_rows": 3,
        "byte_exact_artifacts": {},
    }
    for arm in ("whole", "split"):
        for relative in paths:
            path = tmp_path / arm / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic-controller-hash-probe-" + relative.encode())
            receipt["byte_exact_artifacts"][relative] = sha(path)
        native = {
            "schema": "torch-search-acting-native-cuda-v2",
            "source_commit": mod.SOURCE,
            "runtime": {"torch": "2.11.0+cu130"},
            "inputs": {"book": {"sha256": mod.BOOK}},
            "artifacts": {
                name: sha(tmp_path / arm / "checkpoints/epoch-00000002" / name)
                for name in mod.PAYLOADS
            },
        }
        (tmp_path / arm / "checkpoints/epoch-00000002/checkpoint.json").write_text(
            json.dumps(native)
        )
    mod.check_cli_receipt(receipt, tmp_path, sha)
    receipt["source_commit"] = "4515a7c0dda3b4f9615c2fc78a47c872ab14699d"
    with pytest.raises(AssertionError):
        mod.check_cli_receipt(receipt, tmp_path, sha)
    receipt["source_commit"] = mod.SOURCE
    (tmp_path / "split/checkpoints/epoch-00000002/actor.json").write_bytes(b"mutation")
    with pytest.raises(AssertionError):
        mod.check_cli_receipt(receipt, tmp_path, sha)
