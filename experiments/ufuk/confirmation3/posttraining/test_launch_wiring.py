"""Validate every real coordinator launch against its real callable signature."""

import ast
import inspect
from pathlib import Path

import pytest


def tree_and_signature():
    tree = ast.parse(Path(__file__).with_name("orchestrate.py").read_text())
    definition = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "launch"
    )
    required = len(definition.args.args) - len(definition.args.defaults)
    signature = inspect.Signature([
        inspect.Parameter(
            arg.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=inspect.Parameter.empty if i < required else None,
        )
        for i, arg in enumerate(definition.args.args)
    ])
    return tree, signature


def calls(tree):
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "launch"
    ]


def bind(signature, call):
    assert not any(isinstance(arg, ast.Starred) for arg in call.args)
    assert all(keyword.arg is not None for keyword in call.keywords)
    return signature.bind(
        *[object() for _ in call.args],
        **{keyword.arg: object() for keyword in call.keywords},
    )


def test_all_actual_stage_calls_bind_and_pass_argument_lists():
    tree, signature = tree_and_signature()
    stage_calls = calls(tree)
    assert {call.args[1].value for call in stage_calls} == {
        "a100-mc-fresh-cli-replay.py", "a100-mc-fixed-candidate-eligibility-v2-neural.py",
        "a100-mc-final-portable-parity.py", "a100-mc-strength-latency.py",
        "a100-mc-final-two-arm-controller.py", "a100-mc-strength-analysis.py",
        "a100-mc-all-gates-qualification.py",
    }
    for call in stage_calls:
        bind(signature, call)
        assert isinstance(call.args[2], ast.List)


def test_original_extra_quiescence_strings_fail_real_signature():
    tree, signature = tree_and_signature()
    replay = next(
        call for call in calls(tree)
        if call.args[1].value == "a100-mc-fresh-cli-replay.py"
    )
    replay.args[2:2] = [ast.Constant(value="harbichess.training.torch_online_run")] * 9
    with pytest.raises(TypeError):
        bind(signature, replay)
