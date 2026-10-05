"""Actual trained PST static calls only; no search, games, SGD, NN or CUDA."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import chess
import pytest

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("actual_pst_arena_value", ROOT / "value.py")
v = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v
spec.loader.exec_module(v)
p = json.loads((ROOT / "protocol-TEMPLATE.json").read_text())


def boards():
    board = chess.Board()
    out = [board.copy(stack=True)]
    for move in ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]:
        board.push_uci(move)
        out.append(board.copy(stack=True))
    out.append(board.mirror())
    out.append(chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1"))
    return out


def test_both_real_fixed_final_PST_bitexact_optimized_and_terminal_packets():
    module = v.pst_module(p)
    fast = v.optimized_type(module)
    for seed in [20262905, 20262906]:
        packet = json.loads(Path(p["models"][str(seed)]["learned"]["path"]).read_text())
        original = module.load_pst(p["models"][str(seed)]["learned"]["path"])
        optimized = fast(original.linear.theta, original.pst_theta)
        assert any(packet["pst_theta"])
        for board in boards():
            before = board.fen(), tuple(board.move_stack)
            assert original.nonterminal(board) == optimized.nonterminal(board)
            assert original(board) == optimized(board)
            assert before == (board.fen(), tuple(board.move_stack))


def test_zero_PST_exact_SAME_humanprior_and_reflection():
    module = v.pst_module(p)
    fast = v.optimized_type(module)()
    for board in boards():
        assert fast.nonterminal(board) == module.ClassicalValue().nonterminal(board)
        assert fast(board) == module.ClassicalValue()(board)
        assert fast.nonterminal(board) == fast.nonterminal(board.mirror())


def test_actual_full_native0_final_contract_and_candidate_admission():
    spec = importlib.util.spec_from_file_location("native_admission", ROOT / "native_admission.py")
    module = importlib.util.module_from_spec(spec)
    # Native module imports the explicit adapter, not original generic value.
    original = sys.modules.get("value")
    sys.modules["value"] = v
    try:
        spec.loader.exec_module(module)
        assert module.verify_pst_fit(p)
        bad = copy.deepcopy(p)
        bad["models"]["20262905"]["learned"]["accepted_updates"] -= 1
        with pytest.raises(ValueError):
            module.verify_pst_fit(bad)
    finally:
        if original is None:
            sys.modules.pop("value", None)
        else:
            sys.modules["value"] = original


def test_newadapter_explicit_schema_dispatch_and_reject_other_json(tmp_path):
    prior = p["models"]["20262905"]["prior"]["path"]
    assert v.MixedValue(prior, p).kind.startswith("explicit-untrained-humanprior18")
    candidate = p["models"]["20262905"]["learned"]["path"]
    assert v.MixedValue(candidate, p).kind.startswith("explicit-trained-PST242")
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(dict(schema="other")))
    with pytest.raises(ValueError):
        v.MixedValue(f, p)
