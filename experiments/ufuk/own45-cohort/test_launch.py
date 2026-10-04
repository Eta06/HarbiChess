"""Pure launch construction and actual argparse/AST contract checks; no child jobs."""

import ast
import inspect
from pathlib import Path

import launch_cohort as launch
import pytest

REPO = Path(__file__).parents[1]


def allowed(filename):
    tree = ast.parse(filename.read_text())
    values = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            values.update(
                a.value
                for a in node.args
                if isinstance(a, ast.Constant) and isinstance(a.value, str)
            )
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Tuple | ast.List)
            and any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument"
                for n in ast.walk(node)
            )
        ):
            values.update(
                "--" + v.value
                for v in node.iter.elts
                if isinstance(v, ast.Constant) and isinstance(v.value, str)
            )
    return values


def test_placeholders_rejected_before_clock_and_legitimate_template_filename_allowed():
    launch.no_placeholders({"template": "/staged/helpers/protocol-DRAFT.json"})
    for bad in (
        "ROOT_SOURCE",
        "/content/ROOT_PENDING/file.json",
        "NOT_FROZEN_DO_NOT_EXECUTE",
    ):
        with pytest.raises(AssertionError):
            launch.no_placeholders({"nested": [bad]})


@pytest.mark.parametrize(
    "slot,folder", [(4, "ownsearch-method4"), (5, "search-acting-method5")]
)
def test_factory_command_all_actual_required_flags_and_same_clock(slot, folder):
    names = (
        "template",
        "template-sha256",
        "development-run",
        "profile-receipt",
        "profile-receipt-sha256",
        "auditor-receipt",
        "auditor-receipt-sha256",
        "mc-completion-barrier",
        "mc-completion-barrier-sha256",
        "weights",
        "training-book",
    )
    args = {name: "synthetic-no-execution" for name in names}
    args.update({"book-" + str(seed): "synthetic-book" for seed in launch.SEEDS[slot]})
    if slot == 5:
        args.update(
            {"books-provenance": "synthetic", "books-provenance-sha256": "synthetic"}
        )
    method = {
        "slot": slot,
        "helper_subdir": "helpers" + str(slot),
        "factory_args": args,
        "fixed_epochs": 24 if slot == 4 else 8,
        "whole_training_seconds": 10800,
        "whole_audit_seconds": 12600,
        "posttraining_reserve_seconds": 10020,
    }
    argv = launch.factory_command("/python", "/stage", method, 101.25)
    flags = {x for x in argv if x.startswith("--")}
    declared = allowed(
        REPO / folder / (launch.PREFIX[slot] + "_formal_config_factory.py")
    )
    assert flags <= declared
    assert (
        "--freeze" in flags
        and argv[argv.index("--earliest-training-epoch") + 1] == "101.25"
    )
    assert flags - {"--freeze"} == declared - {"--freeze"}


def test_all_actual_launcher_calls_bind_no_positional_forbidden_string_regression():
    tree = ast.parse(Path(launch.__file__).read_text())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "launch"
    )
    sig = inspect.Signature(
        [
            inspect.Parameter(a.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for a in fn.args.args
        ]
    )
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "launch"
    ]
    assert len(calls) == 5
    for call in calls:
        sig.bind(*[None] * len(call.args), **{k.arg: None for k in call.keywords})


def test_baseline_actual_API_uses_registered_weights_root_first_clock():
    tree = ast.parse(Path(launch.__file__).read_text())
    statements = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.List)
    ]
    command = next(
        n.value
        for n in statements
        if any(
            isinstance(e, ast.Call)
            and isinstance(e.func, ast.Name)
            and e.func.id == "helper"
            and isinstance(e.args[-1], ast.Constant)
            and e.args[-1].value == "baseline"
            for e in n.value.elts
        )
    )
    flags = {
        e.value
        for e in command.elts
        if isinstance(e, ast.Constant)
        and isinstance(e.value, str)
        and e.value.startswith("--")
    }
    for slot, folder in [(4, "ownsearch-method4"), (5, "search-acting-method5")]:
        declared = allowed(REPO / folder / (launch.PREFIX[slot] + "_baseline.py"))
        assert (
            flags - {"--qualification-config", "--qualification-config-sha256"}
            <= declared
        )
    assert {
        "--started-epoch",
        "--registration",
        "--registration-sha256",
        "--weights",
        "--root",
    } <= flags
    assert not {"--initial", "--run-dir", "--source-commit"} & flags


def test_actual_frozen4_reuse_keeps_six_hashes_earliest_and_rejects_budget_helper_change():
    import json

    tree = ast.parse(Path(launch.__file__).read_text())
    preflight = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "preflight"
    )
    condition = next(
        n
        for n in ast.walk(preflight)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and isinstance(n.test.left, ast.Constant)
        and n.test.left.value == "frozen_registration_directory"
        and any(isinstance(b, ast.Assign) for b in n.body)
    )
    code = compile(
        ast.Module(body=condition.body, type_ignores=[]),
        "<actual-frozen-validation>",
        "exec",
    )
    frozen = REPO / "ownsearch-method4/frozen"
    registration = json.loads((frozen / "registration.json").read_text())
    inventory = {p.name: launch.sha(p) for p in frozen.glob("*.json")}
    assert len(inventory) == 6
    method = {
        "frozen_registration_directory": str(frozen),
        "frozen_registration_sha256": inventory,
        "helper_subdir": "helpers4",
        "fixed_epochs": 24,
        "whole_training_seconds": 10800,
        "whole_audit_seconds": 12600,
        "posttraining_reserve_seconds": 10020,
    }
    names = {
        "helpers4/" + name: {"sha256": digest}
        for name, digest in registration["helper_sha256"].items()
    }
    scope = {
        "method": method,
        "Path": Path,
        "sha": launch.sha,
        "json": json,
        "slot": 4,
        "SOURCES": launch.SOURCES,
        "names": names,
        "now": registration["earliest_training_epoch"] + 1,
    }
    exec(code, scope)
    assert (
        inventory["registration.json"]
        == "df8f7459aa36fcea89fadd8685ecfed55eb361a3eb7333977440e45af4c5ae3e"
    )
    with pytest.raises(AssertionError):
        exec(
            code, {**scope, "method": {**method, "posttraining_reserve_seconds": 10021}}
        )
    name = next(iter(names))
    with pytest.raises(AssertionError):
        exec(code, {**scope, "names": {**names, name: {"sha256": "0" * 64}}})
    with pytest.raises(AssertionError):
        exec(code, {**scope, "now": registration["earliest_training_epoch"] - 1})
