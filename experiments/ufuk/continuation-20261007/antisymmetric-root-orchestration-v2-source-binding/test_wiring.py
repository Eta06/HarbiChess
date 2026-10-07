"""Actual accepted ancestor refs / new output namespace; metadata only."""

from pathlib import Path

import execute
import pytest

ROOT = Path("/workspace/work/harbichess/continuation-20261007")


@pytest.mark.parametrize("seed", execute.SEEDS)
def test_passed_ancestors_reused_failed_proof_preserved_newoutput(seed, tmp_path):
    stage = execute.stage(ROOT)
    assert stage.name == "antisymmetric-procedural-own-v4-source-binding"
    for phase in ["zero", "convert", "zero-profile"]:
        assert execute.completed(ROOT, phase, seed)["status"].startswith("PASS-actual-")
    inputs = execute.inputs(ROOT, seed, tmp_path)
    assert inputs["conversion"]["common_dataset"]["path"].endswith(f"{seed}/dataset.json")
    assert (
        execute.output("convert", seed).parent.parent.name
        == "harbichess-antisymmetric-procedural-v3"
    )
    assert (
        execute.output("proof-v2", seed).parent.parent.name
        == "harbichess-antisymmetric-procedural-v4"
    )
    assert not execute.records(ROOT, "proof-v2", seed).exists()
    old = ROOT / f"antisymmetric-root-v3-proof-{seed}/result.json"
    assert execute.read(old)["status"] == "FAILED-preserved"
    q = execute.protocol(ROOT, stage, "zero-unused", 1791400000.0, 1791448916.685839)
    zero_contract = execute.read(execute.zero_binding(ROOT, seed)["contract"]["path"])
    assert q["original_search"] == zero_contract["search_helper"]
    assert q["schema"] == "antisymmetric-procedural-known160-protocol-v4"
