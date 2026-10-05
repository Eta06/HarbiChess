"""Static evaluator/native dispatch only, no search, SGD, games or NN queries."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import chess
import pytest

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("actual_quiet_arena_value", ROOT / "value.py")
v = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v
spec.loader.exec_module(v)
p = json.loads((ROOT / "protocol-TEMPLATE.json").read_text())


def test_learned_ordering_does_not_modify_fixed_humanprior_evaluator():
    prior = v.MixedValue(p["models"]["20262905"]["prior"]["path"], p)
    board = chess.Board()
    positions = [board.copy(stack=True)]
    for move in ["e2e4", "e7e5", "g1f3", "b8c6"]:
        board.push_uci(move)
        positions.append(board.copy(stack=True))
    for seed in [20262905, 20262906]:
        learned = v.MixedValue(p["models"][str(seed)]["learned"]["path"], p)
        assert any(learned.ordering_weights) and not any(prior.ordering_weights)
        for position in positions:
            assert learned(position) == prior(position)
            assert learned(position) == learned(position.mirror())


def test_all_actual_native_storage_contracts_and_corrupt_endpoint_rejection():
    spec = importlib.util.spec_from_file_location(
        "quiet_native_admission", ROOT / "native_admission.py"
    )
    m = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("value")
    sys.modules["value"] = v
    try:
        spec.loader.exec_module(m)
        assert m.verify_quiet_fit(p)
        bad = copy.deepcopy(p)
        bad["models"]["20262905"]["learned"]["accepted_updates"] -= 1
        with pytest.raises(ValueError):
            m.verify_quiet_fit(bad)
        bad = copy.deepcopy(p)
        bad["models"]["20262906"]["prior"]["sha256"] = "0" * 64
        with pytest.raises(ValueError):
            m.verify_quiet_fit(bad)
    finally:
        if previous is None:
            sys.modules.pop("value", None)
        else:
            sys.modules["value"] = previous


def test_adapter_binds_original_search_and_qualified_ordering_without_running_it():
    spec = importlib.util.spec_from_file_location("quiet_arena_search", ROOT / "search.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    assert v.sha(ROOT / "search_base.py") == m.BASE_SHA
    prior = v.MixedValue(p["models"]["20262905"]["prior"]["path"], p)
    search = m.BudgetSearch(prior)
    assert search.ordering_weights == [0.0] * 268
    assert search.node_budget == 512 and search.qdepth == 2 and search.max_depth == 8
