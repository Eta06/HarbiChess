"""Imports, static legal ordering and source/admission guards only; no real search."""

import importlib.util
import json
import sys
from pathlib import Path

import chess
import pytest

ROOT = Path(__file__).parent
SOURCE = Path("/workspace/work/harbichess/cpu-own-deeper-action-proposal")
sys.path.insert(0, str(SOURCE))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def test_same_engine_zero_prior_static_order_no_search():
    value = module("synthetic_arena_value", ROOT / "arena/value.py")
    search = module("synthetic_arena_search", ROOT / "arena/search.py")
    p = json.loads((ROOT / "arena/protocol-DRAFT.json").read_bytes())
    prior = p["models"]["20262905"]["prior"]["path"]
    evaluator = value.MixedValue(prior, p)
    engine = search.BudgetSearch(evaluator, nodes=512, quiescence_plies=2, max_depth=8)
    board = chess.Board()
    moves = list(board.legal_moves)
    scores = {m: 0 for m in moves}
    assert engine.engine.root_order(board, moves, scores) == sorted(moves, key=lambda m: m.uci())
    assert evaluator.ordering_weights == [0.0] * 280


def test_learned_model_does_not_change_prior_evaluator(tmp_path):
    import action_model

    value = module("synthetic_arena_value2", ROOT / "arena/value.py")
    p = json.loads((ROOT / "arena/protocol-DRAFT.json").read_bytes())
    path = tmp_path / "synthetic-model.json"
    path.write_text(json.dumps(action_model.model_dict([0.01] * 280)))
    evaluator = value.MixedValue(path, p)
    assert evaluator.kind == "learned280-ordering-frozen-prior18-value"
    assert list(evaluator.value.__self__.theta) == [0.0] * 18


def test_unknown_or_nonzero_prior_schema_rejected(tmp_path):
    value = module("synthetic_arena_value3", ROOT / "arena/value.py")
    p = json.loads((ROOT / "arena/protocol-DRAFT.json").read_bytes())
    prior = json.loads(Path(p["models"]["20262905"]["prior"]["path"]).read_bytes())
    prior["theta"][0] = 1
    path = tmp_path / "wrong-prior.json"
    path.write_text(json.dumps(prior))
    with pytest.raises(ValueError):
        value.MixedValue(path, p)


def test_synthetic_native_proof_actual_six_subprocess_loads():
    p = json.loads((ROOT / "synthetic-native-proof-final.json").read_bytes())
    assert len(p["native_checks"]) == 3 and all(
        r["all_native_bytes_equal"] for r in p["native_checks"]
    )
    assert len([r for r in p["receipts"] if r["name"].startswith("freshload-")]) == 6
    assert p["real_data_used"] is False


def test_frozen_math_contract_budget_truth():
    from action_learner import MATH, Learner

    assert MATH["max_updates"] == 1024  # inherited UNUSED metadata; cannot advertise changed source
    with pytest.raises(ValueError):
        Learner(1, {"updates": 1024})
    assert Learner(1, {"updates": 64}).step == 0


def test_profile_order_bias_is_order_only_and_counts_extra_refs():
    source = (ROOT / "arena/profile.py").read_text()
    assert "r != reference" not in source
    assert "actual_search_calls=32" in source
    assert "float(r.value).hex()" in source
    student = (SOURCE / "student_search.py").read_text()
    assert "scores[m] + correction[m]" in student
    assert "iteration[move] = value" in student  # correction never modifies backed-up value


def test_strict_offline_native_rejects_adam_corruption_and_restores_global_rng(tmp_path):
    import hashlib
    import random

    from action_learner import Learner

    admission = module("synthetic_native_admission", ROOT / "arena/native_admission.py")
    p = json.loads((ROOT / "arena/protocol-DRAFT.json").read_bytes())
    contract = dict(
        updates=64,
        seed=20262905,
        source_commit=p["source_commit"],
        protocol_sha256="synthetic-fit",
        labels_sha256="synthetic-labels",
        producer_registration_sha256="synthetic-producer",
        deadline=1,
    )
    native = Learner(20262905, contract).native()
    native["step"] = 64  # STRUCTURAL synthetic state only, not64fit proof
    model = tmp_path / "candidate.json"
    model.write_text(json.dumps(native["model"]))
    path = tmp_path / "native.json"
    path.write_text(json.dumps(native))

    def sha(f):
        return hashlib.sha256(f.read_bytes()).hexdigest()

    p["models"]["20262905"]["learned"].update(
        path=str(model), sha256=sha(model), native_path=str(path), native_sha256=sha(path)
    )
    p["ACTION_fit_inputs"] = {
        "20262905": {
            k: contract[k]
            for k in ["protocol_sha256", "labels_sha256", "producer_registration_sha256"]
        }
    }
    global_before = random.getstate()
    assert admission.admit(p, 20262905)["original_fit_deadline_preserved"] == 1
    assert random.getstate() == global_before
    native["v"][0] = -1
    path.write_text(json.dumps(native))
    p["models"]["20262905"]["learned"]["native_sha256"] = sha(path)
    with pytest.raises(ValueError):
        admission.admit(p, 20262905)
    assert random.getstate() == global_before
