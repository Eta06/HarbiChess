"""Execute the actual certificate loop and actual report epoch expression, without NN."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "epoch_index,row_index", [(1, 15), (1, 32767), (2, 127), (8, 32767), (8, 728)]
)
def test_actual_certificate_loop_preserves_native_epoch(epoch_index, row_index):
    source = ast.parse(Path(__file__).with_name("own8_audit_core.py").read_text())
    function = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "audit_epoch"
    )
    loop = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Attribute)
        and isinstance(node.iter.func.value, ast.Name)
        and node.iter.func.value.id == "by_index"
    )
    report = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "report" for t in node.targets)
    )
    epoch = next(keyword.value for keyword in report.value.keywords if keyword.arg == "epoch")
    actions = [None] * (row_index + 1)
    actions[row_index] = SimpleNamespace(
        transition=SimpleNamespace(pre="fullhistory"), legal_actions=(1, 2)
    )
    namespace = dict(
        index=epoch_index,
        by_index={row_index: {"certified_losing_actions": []}},
        epoch=SimpleNamespace(actions=actions),
        final=SimpleNamespace(actors=SimpleNamespace(rules="rules")),
        config=SimpleNamespace(
            actors=SimpleNamespace(claim_draw=True), search=SimpleNamespace(simulations=16)
        ),
        certificate_roots=0,
        certificate_moves=0,
        loss_roots=0,
        loss_actions=0,
        audit_certified_root_receipt=lambda *args, **kwargs: 1,
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])),
            "actual-source8-certificate-loop",
            "exec",
        ),
        namespace,
    )
    value = eval(compile(ast.Expression(epoch), "actual-source8-report-epoch", "eval"), namespace)
    assert value == epoch_index and namespace["certificate_moves"] == 1
