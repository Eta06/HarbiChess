"""Synthetic phase/reader contracts only; no actual model inference or search."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from native_admission import check_phase

BASE = Path("/workspace/work/harbichess/cpu-ownq-C18-arena-v2-proposal")


def test_search_and_tournament_are_original_bytes():
    here = Path(__file__).parent
    for name in ["search.py", "tournament.py"]:
        assert (here / name).read_bytes() == (BASE / name).read_bytes()
    assert (
        hashlib.sha256((here / "search.py").read_bytes()).hexdigest()
        == "de53c14728a67b7772f18b396ac8ef5c35a144d4e4e616fec40099cd461a6670"
    )


def fixture():
    reg = dict(
        schema="ownq-coarse80-root-phase-v2",
        phase="proof",
        first_epoch=100,
        deadline_epoch=700,
        input_pins={},
        seeds=[5, 6],
        contract_closure=dict(
            phase="proof",
            first_epoch=100,
            deadline_epoch=700,
            pins={},
            native_contract_version="ROOT-phase-bound-v2-not-old-validator-v1",
        ),
    )
    result = dict(
        schema="ownq-coarse80-realdata-phase-result-v1",
        phase="proof",
        status="PASS-proof",
        registration_sha256="0" * 64,
        first_epoch=100,
        deadline_epoch=700,
        finished_epoch=699,
        rows=[{"seed": 5}, {"seed": 6}],
        strength_success_claimed=False,
    )
    return reg, result


def test_actual_phase_clock_no_old_future_validator():
    reg, result = fixture()
    check_phase(reg, result, "proof", "PASS-proof", "0" * 64)
    for key, value in [
        ("first_epoch", 101),
        ("deadline_epoch", 701),
        ("finished_epoch", 701),
        ("registration_sha256", "1" * 64),
        ("phase", "fit"),
        ("strength_success_claimed", True),
    ]:
        changed = copy.deepcopy(result)
        changed[key] = value
        with pytest.raises(ValueError):
            check_phase(reg, changed, "proof", "PASS-proof", "0" * 64)


def test_closure_clock_and_inventory_are_not_unbound_metadata():
    reg, result = fixture()
    for key, value in [
        ("phase", "fit"),
        ("first_epoch", 99),
        ("deadline_epoch", 701),
        ("pins", {"extra": "0" * 64}),
        ("native_contract_version", "v1"),
    ]:
        changed = copy.deepcopy(reg)
        changed["contract_closure"][key] = value
        with pytest.raises(ValueError):
            check_phase(changed, result, "proof", "PASS-proof", "0" * 64)


def test_draft_never_inherits_old_C18_clock_or_models():
    p = json.loads(Path(__file__).with_name("protocol-DRAFT.json").read_bytes())
    for key in [
        "original_first_epoch",
        "original_deadline_epoch",
        "profile_first_epoch",
        "profile_deadline_epoch",
        "audit_deadline_epoch",
    ]:
        assert p[key] is None
    assert p["total_games"] == 160 and p["search_nodes"] == 512 and p["quiescence_plies"] == 2
    assert len(p["tasks"]) == 5
    assert p["fast_reader"]["path"].endswith("fast_evaluator80_v1.py")
    assert (
        p["compiled_binary"]["sha256"]
        == "0c522abc7220ce748008c41fe7520b48c0c22a71d69f92486c70bf610b84f173"
    )
    assert all(r["learned"]["accepted_updates"] == 64 for r in p["models"].values())
