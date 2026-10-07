import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from model import FEATURE_SCHEMA
from native import MATH, bits_equal
from support import namespace, ref

HERE = Path(__file__).parent


def test_actual_phase_math_fresh_subprocess_resume_and_six_opens(tmp_path):
    c = dict(
        phase="antisymmetric-procedural-own-learning-v4",
        updates=64,
        math=MATH,
        feature_schema=FEATURE_SCHEMA,
        seed=51,
        synthetic_only=True,
        zero_parent={},
        initializer="literalzero-antisymmetric-new-Adam-RNG-v1",
    )
    cp = tmp_path / "c.json"
    cp.write_text(json.dumps(c))
    rp = tmp_path / "rows.json"
    rp.write_text(
        json.dumps(
            [
                dict(
                    indices=[5, 77, 801],
                    opposite_indices=[30, 90, 999],
                    prior_logit=0.1,
                    target=0.4,
                )
            ]
        )
    )
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def run(name, stop, resume=None, audit=False):
        out = tmp_path / (name + ".pt")
        argv = [
            sys.executable,
            str(HERE / "synthetic_cli.py"),
            "--contract",
            str(cp),
            "--rows",
            str(rp),
            "--output",
            str(out),
            "--stop",
            str(stop),
        ]
        if resume:
            argv += ["--resume", str(resume)]
        if audit:
            argv += ["--audit-only"]
        subprocess.run(argv, env=env, check=True, capture_output=True, timeout=15)
        return out

    z = run("zero", 0)
    w = run("whole", 8)
    p = run("pause", 4)
    r = run("resume", 8, p)

    def load(path):
        return torch.load(path, weights_only=False, map_location="cpu")

    assert bits_equal(load(w), load(r))
    for i, path in enumerate([z, w, z, p, p, r]):
        assert bits_equal(load(path), load(run("fresh" + str(i), 0, path, True)))
    assert load(w)["model"]["head.bias"].reshape(-1).view(torch.uint8).count_nonzero() == 0
    assert not bits_equal(load(z)["model"], load(w)["model"])


def test_complete_namespace_pin_and_changed_helper_rejection(tmp_path):
    source = tmp_path / "dummy.py"
    source.write_text("VALUE=42\n")
    inv = tmp_path / "inventory.json"
    inv.write_text(json.dumps({"files": {"dummy.py": ref(source)["sha256"]}}))
    with namespace(tmp_path, [ref(inv)]) as files:
        assert files["dummy.py"] == ref(source)["sha256"]
    source.write_text("VALUE=43\n")
    with pytest.raises(ValueError, match="SHA"), namespace(tmp_path, [ref(inv)]):
        pass


def test_advance_body_identical_and_synthetic_not_actual_admission(tmp_path):
    old = Path(
        "/workspace/work/harbichess/continuation-20261007/antisymmetric-human-prior-own-v1/native.py"
    )

    def advance(path):
        return next(
            n
            for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.FunctionDef) and n.name == "advance"
        )

    assert ast.dump(advance(old)) == ast.dump(advance(HERE / "native.py"))
    from admission import phase

    c = tmp_path / "c.json"
    c.write_text(json.dumps({"synthetic_only": True}))
    r = tmp_path / "r.json"
    r.write_text("{}")
    with pytest.raises(ValueError, match="synthetic"):
        phase(ref(r), ref(c), "proof")


def test_metadata_geometry_rejects_real_executor_collision(tmp_path):
    from prove import validate_output_geometry

    actual_output = Path("/dev/shm/antisymmetric-fixture-never-created-output")
    reg = dict(
        output=str(actual_output),
        contract=dict(path=str(tmp_path / "c.json")),
        dataset=dict(
            path="/dev/shm/harbichess-human-randomstarts-forensic-data-v4/20262905/dataset.json"
        ),
        train=dict(path=str(HERE / "train.py")),
        native=dict(path=str(HERE / "native.py")),
    )
    validate_output_geometry(reg)
    bad = dict(reg, contract=dict(path=str(actual_output / "contract.json")))
    with pytest.raises(ValueError, match="OUTSIDE"):
        validate_output_geometry(bad)
    assert not actual_output.exists()


def test_profile_receipt_clock_fields_match_actual_producer():
    producer = ast.parse((HERE / "qualify_profile.py").read_text())
    consumer = ast.parse((HERE / "run_known160.py").read_text())
    producer_fields = {
        keyword.arg
        for node in ast.walk(producer)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
    }
    consumer_fields = {
        node.slice.value
        for node in ast.walk(consumer)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "q"
        and isinstance(node.slice, ast.Constant)
    }
    assert consumer_fields <= producer_fields
    assert "finished" in consumer_fields
    assert "finished_epoch" not in consumer_fields


def test_original_search_different_paths_exactbytes_and_changedsource_rejected(tmp_path):
    from contracts import exact_search_bindings

    original = HERE.parent / "antisymmetric-procedural-own-v3-production/search.py"
    copied = tmp_path / "different-legal-copy.py"
    copied.write_bytes(original.read_bytes())
    result = exact_search_bindings(ref(original), ref(copied))
    assert result["collection"]["path"] != result["literalzero"]["path"]
    assert result["paths_may_differ"] is True
    copied.write_bytes(original.read_bytes() + b"\n# changed source\n")
    with pytest.raises(ValueError, match="de53"):
        exact_search_bindings(ref(original), ref(copied))


@pytest.mark.parametrize("seed", [20262905, 20262906])
def test_actual_failed_contract_metadata_now_binds_both_real_search_paths(monkeypatch, seed):
    import contracts
    from support import read

    old = HERE.parent / f"antisymmetric-root-v3-proof-{seed}/build-seal.json"
    spec = __import__("json").loads(old.read_bytes())
    spec["schema"] = "antisymmetric-forensic-training-build-seal-v4"
    paired, provenance = read(spec["dataset"]), read(spec["provenance"])
    monkeypatch.setattr(contracts, "derive", lambda *args: (paired, provenance))
    monkeypatch.setattr(contracts, "admit", lambda binding, sid: ({}, read(binding["contract"])))
    c = contracts.build(spec)
    assert c["contract_schema"] == "antisymmetric-byte-bound-training-contract-v4"
    assert c["phase"] == "antisymmetric-procedural-own-learning-v4"
    assert c["input_data_phase"] == paired["phase"] == "antisymmetric-procedural-own-learning-v3"
    assert c["search_bindings"]["collection"]["path"] != c["search_bindings"]["literalzero"]["path"]
    assert (
        c["search_bindings"]["collection"]["sha256"]
        == c["search_bindings"]["literalzero"]["sha256"]
    )
    assert c["zero_profile"] == spec["zero_profile"]
    assert "parent_bridge.py" in c["source_refs"] and "initialize.py" not in c["source_refs"]
