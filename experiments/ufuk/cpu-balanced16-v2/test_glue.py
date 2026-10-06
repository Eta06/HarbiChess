"""Synthetic-only native/schema/source wiring controls; no real-data SGD or searches."""

import ast
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
from learner_v2 import MATH, Learner, parent
from support import canonical
from transform import group_digest, model_dict, summary

HERE = Path(__file__).parent


def fixture():
    groups = {
        str(i): [{"features": [1.0] + [0.0] * 15, "target": j % 2} for j in range(8)]
        for i in range(16)
    }
    record = summary(groups)
    record["training_group_sha256"] = group_digest(groups)
    contract = dict(
        schema="own-balanced-risk-learning-contract-v2",
        seed=20262905,
        updates=48,
        math=MATH,
        transform=record,
        inputs={},
        source_commit="synthetic-only",
        helper_sha256={},
        first_epoch=1,
        deadline_epoch=901,
    )
    return groups, contract


def test_synthetic_whole_pause_fresh_native_and_restored_private_gradient():
    rng = random.getstate()
    gradient = parent.row_gradient
    try:
        groups, c = fixture()
        whole = Learner(c["seed"], c)
        whole.advance(groups, 8)
        split = Learner(c["seed"], c)
        split.advance(groups, 4)
        resume = Learner(c["seed"], c, json.loads(canonical(split.native())))
        resume.advance(groups, 8)
        assert canonical(whole.native()) == canonical(resume.native())
        assert parent.row_gradient is gradient
        assert whole.native()["model"]["schema"].endswith("risk-model-v2")
    finally:
        random.setstate(rng)


def test_v1_native_and_changed_transform_rejected():
    rng = random.getstate()
    try:
        groups, c = fixture()
        state = Learner(c["seed"], c).native()
        state["schema"] = "own-selective-quiescence-effort-native-v1"
        with pytest.raises(ValueError):
            Learner(c["seed"], c, state)
        bad = dict(c, transform=dict(c["transform"], training_group_sha256="changed"))
        learner = Learner(c["seed"], bad)
        with pytest.raises(ValueError):
            learner.advance(groups, 8)
    finally:
        random.setstate(rng)


def test_cli_parser_and_schema_wiring_no_compute():
    for name in ["train.py", "qualify.py", "fit.py", "register.py", "architecture.py"]:
        completed = subprocess.run(
            [sys.executable, str(HERE / name), "--help"], capture_output=True, timeout=10
        )
        assert completed.returncode == 0, completed.stderr.decode()
    tree = ast.parse((HERE / "fit.py").read_text())
    strings = {
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    assert {"--audit-only", "initial.json", "native.json"} <= strings
    module = ast.parse((HERE / "register.py").read_text())
    assert module
    assert (
        model_dict([0.0] * 16)["score_semantics"]
        == "balanced-class-risk-logit-not-calibrated-posterior"
    )
