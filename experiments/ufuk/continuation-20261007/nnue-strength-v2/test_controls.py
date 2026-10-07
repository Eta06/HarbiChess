"""Pure metadata/bootstrap/CLI wiring only; no actual NN forward or search."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from admission import SEARCH_SHA, proof_fit_contracts, sha
from confirmation_stats import ALPHA_FAMILY, COMPARISONS, TAIL, evaluate_seed, paired_arrays

HERE = Path(__file__).parent


def test_search_identity_five_arms_and_exact_multiplicity():
    assert sha(HERE / "search.py") == SEARCH_SHA
    assert (HERE / "tournament.py").read_bytes() == Path(
        "/workspace/HarbiChess/experiments/ufuk/cpu-classical-own-v1/arena/tournament.py"
    ).read_bytes()
    p = json.loads((HERE / "protocol-DRAFT.json").read_bytes())
    assert p["total_games"] == 2 * 5 * 8 * 2 == 160
    assert set(map(tuple, p["tasks"])) == {
        ("learned", "e8"),
        ("learned", "parent"),
        ("learned", "SF512"),
        ("e8", "SF512"),
        ("parent", "SF512"),
    }
    assert p["ROOToperator_end_epoch"] == 1791448916.685839
    q = json.loads((HERE / "confirmation-DRAFT.json").read_bytes())
    assert q["total_games"] == 2 * 5 * 48 * 2 == 960
    assert ALPHA_FAMILY == 0.00625 and COMPARISONS == 8 and TAIL == 0.00078125
    assert 1 - 2 * TAIL == 0.9984375


def test_all_contract_fields_and_exact_proof_binding():
    ref = dict(path="/future/proof.json", sha256="proofSHA")
    proof_contract_ref = dict(path="/future/proof-contract.json", sha256="contractSHA")
    p = dict(
        phase="own-learning",
        updates=64,
        seed=20262905,
        all_unanticipated_field="keep",
        original_first_epoch=1,
        original_deadline_epoch=601,
        inference_source_sha256={"model": "source"},
        execution_mode="proof",
        own_phase_proof=None,
        contract_build_seal=dict(mode="proof", first=1, deadline=601, immutable_dataset="keep"),
    )
    f = dict(
        p,
        original_first_epoch=2,
        original_deadline_epoch=1802,
        execution_mode="fresh-fit",
        own_phase_proof=ref,
        inference_source_sha256={"model": "source", ref["path"]: ref["sha256"]},
        contract_build_seal=dict(
            mode="fresh-fit",
            first=2,
            deadline=1802,
            immutable_dataset="keep",
            own_proof_result=ref,
            own_proof_contract=proof_contract_ref,
        ),
    )
    proof_fit_contracts(p, f, ref, proof_contract_ref)
    for k, v in [
        ("seed", 20262906),
        ("all_unanticipated_field", "drop"),
        ("phase", "teacher-bootstrap"),
    ]:
        bad = dict(f)
        bad[k] = v
        with pytest.raises(ValueError):
            proof_fit_contracts(p, bad, ref, proof_contract_ref)
    bad = copy.deepcopy(f)
    bad["inference_source_sha256"]["hidden"] = "extra"
    with pytest.raises(ValueError):
        proof_fit_contracts(p, bad, ref, proof_contract_ref)


def games(score, root_ids):
    return [
        dict(root_id=r, candidate_color=c, score=score, termination="stalemate")
        for r in root_ids
        for c in ["white", "black"]
    ]


def test_paired_sets_strict_and_caps_are_unknown_adverse():
    ids = [str(i) for i in range(48)]
    rows = games(0.5, ids)
    assert all(paired_arrays(rows, ids) == 0.5)
    rows[0]["termination"] = "max_plies"
    assert paired_arrays(rows, ids, adverse=True)[0] == 0.25
    assert paired_arrays(rows, ids, adverse=True, baseline=True)[0] == 0.75
    with pytest.raises(ValueError):
        paired_arrays([*rows, rows[0]], ids)
    with pytest.raises(ValueError):
        paired_arrays(rows[:-1], ids)


def test_parent_comparator_cannot_be_replaced_by_E8():
    ids = [str(i) for i in range(48)]
    arms = {
        k: games(v, ids)
        for k, v in {
            "learned-e8": 1.0,
            "learned-parent": 0.5,
            "learned-SF512": 0.5,
            "e8-SF512": 0.0,
            "parent-SF512": 0.5,
        }.items()
    }
    result = evaluate_seed(arms, ids, 20262905)
    assert not result["passed"]
    assert not result["analyses"][0]["gates"]["parent_direct_gt_060"]
    assert not result["analyses"][0]["gates"]["SF_gain_parent_gt_0"]


def test_profile_named_to_avoid_stdlib_shadow_and_parser_no_execution():
    assert not (HERE / "profile.py").exists()
    for name in ["run.py", "qualify_profile.py", "audit_known160.py"]:
        p = subprocess.run(
            [sys.executable, str(HERE / name), "--help"], capture_output=True, timeout=10
        )
        assert p.returncode == 0, p.stderr.decode()


def test_actual_v2_fresh_log_wire_and_clock_tampering(tmp_path):
    from admission import own_receipt

    contract_path = tmp_path / "contract.json"
    c = dict(original_first_epoch=100, original_deadline_epoch=1900)
    contract_path.write_text(json.dumps(c))
    contract_ref = dict(path=str(contract_path), sha256=sha(contract_path))
    payloads = []
    commands = []
    for step in [0, 64]:
        path = tmp_path / f"native{step}.pt"
        path.write_bytes(b"synthetic-byte-fixture-only")
        log = tmp_path / f"audit{step}.log"
        log.write_text(json.dumps(dict(status="PASS-strict-native-readonly", step=step)))
        payloads.append(
            dict(
                path=str(path),
                sha256=sha(path),
                step=step,
                actual_fresh_process=True,
                log_path=str(log),
            )
        )
        commands.append(
            dict(
                command=[
                    "train.py",
                    "--contract",
                    str(contract_path),
                    "--stop",
                    str(step),
                    "--resume",
                    str(path),
                    "--resume-sha256",
                    sha(path),
                    "--audit-only",
                ],
                returncode=0,
                log_sha256=sha(log),
            )
        )
    result = dict(
        status="PASS-own-NNUE-fixed-phase-and-fresh-native-loads-not-strength",
        contract_sha256=contract_ref["sha256"],
        raw_zip_identity_claimed=False,
        first=100,
        deadline=1900,
        finished=200,
        commands=commands,
        native_payloads=payloads,
    )
    assert len(own_receipt(result, contract_ref, c, [0, 64], 2000)) == 2
    for field, value in [("finished", 1901), ("deadline", 1901)]:
        bad = copy.deepcopy(result)
        bad[field] = value
        with pytest.raises(ValueError):
            own_receipt(bad, contract_ref, c, [0, 64], 2000)
    bad = copy.deepcopy(result)
    bad["native_payloads"][0]["actual_fresh_process"] = False
    with pytest.raises(ValueError):
        own_receipt(bad, contract_ref, c, [0, 64], 2000)


def test_actual_build_seal_unknown_and_clock_fields_cannot_drift():
    ref = dict(path="/future/proof.json", sha256="SHA")
    cp = dict(path="/future/contract.json", sha256="contractSHA")
    p = dict(
        seed=1,
        original_first_epoch=1,
        original_deadline_epoch=601,
        execution_mode="proof",
        own_phase_proof=None,
        inference_source_sha256={"model": "source"},
        contract_build_seal=dict(mode="proof", first=1, deadline=601, protected_aliases="exact"),
    )
    f = dict(
        p,
        original_first_epoch=2,
        original_deadline_epoch=1802,
        execution_mode="fresh-fit",
        own_phase_proof=ref,
        inference_source_sha256={"model": "source", ref["path"]: ref["sha256"]},
        contract_build_seal=dict(
            mode="fresh-fit",
            first=2,
            deadline=1802,
            protected_aliases="exact",
            own_proof_result=ref,
            own_proof_contract=cp,
        ),
    )
    proof_fit_contracts(p, f, ref, cp)
    bad = copy.deepcopy(f)
    bad["contract_build_seal"]["protected_aliases"] = "other"
    with pytest.raises(ValueError):
        proof_fit_contracts(p, bad, ref, cp)
    bad = copy.deepcopy(f)
    bad["contract_build_seal"]["first"] = 3
    with pytest.raises(ValueError):
        proof_fit_contracts(p, bad, ref, cp)
