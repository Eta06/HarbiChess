"""Pure four-input/version/receipt tests; no search, fitting, or matches."""

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "balanced_arena_admission_test", HERE / "native_admission.py"
)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def put(path, data):
    path.write_text(json.dumps(data))
    return dict(path=str(path), sha256=native.sha(path))


def fixture(tmp):
    data = {"dataset": put(tmp / "data.json", {}), "split": put(tmp / "split.json", {})}
    proof = put(tmp / "proof.json", {})
    helpers = {"synthetic": "exact"}
    profile_reg = dict(
        seed=20262905,
        source_commit="synthetic",
        helper_sha256=helpers,
        first_epoch=1,
        deadline_epoch=601,
        inputs={},
    )
    registration = put(tmp / "profile-reg.json", profile_reg)
    profile = put(
        tmp / "profile.json",
        dict(
            status="PASS-architecture-only-not-strength",
            finished=3,
            deadline=601,
            median_ratio=1.01,
            packets=[dict(role=r) for r in ["new-zero"] * 12 + ["old-q2"] * 12],
        ),
    )
    binding = dict(
        profile, registration_path=registration["path"], registration_sha256=registration["sha256"]
    )
    fit = dict(
        seed=20262905,
        source_commit="synthetic",
        helper_sha256=helpers,
        inputs=dict(data, native_qualification=proof, profile_qualification=profile),
    )
    return fit, dict(inputs=data), proof, binding


def test_all_four_real_binding_roles_retained(tmp_path):
    rows = fixture(tmp_path)
    native.validate_fit_qualifications(*rows)
    rows[0]["inputs"].pop("profile_qualification")
    with pytest.raises(ValueError):
        native.validate_fit_qualifications(*rows)


def test_profile_source_and_original_clock_mutations_reject(tmp_path):
    rows = fixture(tmp_path)
    path = Path(rows[3]["registration_path"])
    reg = json.loads(path.read_bytes())
    reg["source_commit"] = "other"
    path.write_text(json.dumps(reg))
    rows[3]["registration_sha256"] = native.sha(path)
    with pytest.raises(ValueError):
        native.validate_fit_qualifications(*rows)
    reg["source_commit"] = "synthetic"
    path.write_text(json.dumps(reg))
    rows[3]["registration_sha256"] = native.sha(path)
    path = Path(rows[3]["path"])
    profile = json.loads(path.read_bytes())
    profile["finished"] = 602
    path.write_text(json.dumps(profile))
    rows[3]["sha256"] = native.sha(path)
    rows[0]["inputs"]["profile_qualification"]["sha256"] = native.sha(path)
    with pytest.raises(ValueError):
        native.validate_fit_qualifications(*rows)


def test_search_tournament_statistics_frozen_and_zero_schema():
    original = Path("/workspace/work/harbichess/cpu-selective-q-arena-v2-proposal")
    for name in ["search.py", "tournament.py", "audit_arena_v2.py", "audit_support.py"]:
        assert (HERE / name).read_bytes() == (original / name).read_bytes()
    packet = json.loads((HERE / "newzero.json").read_bytes())
    assert packet["schema"] == "own-selective-quiescence-balanced-risk-model-v2"
    assert packet["weights"] == [0.0] * 16
    assert packet["score_semantics"] == "balanced-class-risk-logit-not-calibrated-posterior"
