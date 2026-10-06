"""Disclosed synthetic evaluators/data only; no actual NN/search qualification."""

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import chess
import pytest
from labels import admission, collect, digest, history, restore
from learner import Learner
from model import features, model_dict
from search import BASE_SHA, search_type
from support import sha

BASE = Path("/workspace/HarbiChess/experiments/ufuk/cpu-classical-own-v1/arena/search.py")
assert sha(BASE) == BASE_SHA
spec = importlib.util.spec_from_file_location("pure_base", BASE)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
Selective = search_type(base.BudgetSearch, base.BudgetExhausted)


def test_budget_alllegal_botharms_and_history():
    board = chess.Board()
    before = history(board)
    for w in [[0.0] * 16, [0.1] * 16]:
        s = Selective(lambda b: 0.0, nodes=64, quiescence_plies=2, max_depth=8, weights=w)
        r = s.search(board)
        assert r.move in board.legal_moves and r.root_actions == 20 and r.nodes <= 64
        assert history(board) == before and all(x["nodes"] <= 64 for x in s.extension_receipts)


def test_checked_horizon_censors_no_static_value():
    b = chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
    s = Selective(
        lambda _: pytest.fail("checked horizon must not static evaluate"),
        nodes=64,
        quiescence_plies=2,
        max_depth=8,
    )
    with pytest.raises(base.BudgetExhausted):
        s.extra(b, -math.inf, math.inf, 0, 0, 0)
    assert s.nodes == 1


def test_terminal_mover_and_claimhistory():
    mate = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
    r = Selective(
        lambda _: pytest.fail("terminal evaluator"), nodes=64, quiescence_plies=2, max_depth=8
    ).search(mate)
    assert r.value == -2 and r.move is None
    b = chess.Board()
    for u in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        b.push_uci(u)
    restored = restore(history(b))
    assert restored.can_claim_threefold_repetition()
    assert (
        Selective(
            lambda _: pytest.fail("draw evaluator"), nodes=64, quiescence_plies=2, max_depth=8
        )
        .search(restored)
        .value
        == 0
    )


def test_chronological_leaves_counterfactuals_censored_not_negative():
    rows = [
        dict(row_id=f"{i:04}", trajectory_id=str(i % 16), history=history(chess.Board()))
        for i in range(1024)
    ]
    result = collect(rows, base.BudgetSearch, base.BudgetExhausted, lambda _: 0.0, root_count=2)
    assert [r["row_id"] for r in result["roots"]] == ["0000", "0001"]
    assert len(result["leaves"]) <= 8
    for r in result["leaves"]:
        assert r["additional_nodes"] <= 64 and digest(r["history"]) == r["history_sha256"]
        assert r["target"] in [None, 0, 1]
    with pytest.raises(ValueError, match="INCOMPLETE"):
        admission(result, set(str(i) for i in range(16)))


def test_state16_reflection_dimensions_and_tamper():
    x = features(chess.Board(), 0.0)
    b = chess.Board()
    b.push_uci("e2e4")
    assert features(b, 0.2) == features(b.mirror(), 0.2)
    assert len(x) == 16 and all(math.isfinite(a) for a in x)
    n = Learner(1, {"updates": 48}).native()
    n["v"][0] = -1
    with pytest.raises(ValueError):
        Learner(1, {"updates": 48}, n)
    assert model_dict([0.0] * 16)["evaluator"] == "unchanged-humanprior18"


def test_synthetic_fresh_process8_4_resume8_fullbytes(tmp_path):
    code = """import json,sys
from learner import Learner
from support import canonical
state=None if sys.argv[2]=='-' else json.loads(open(sys.argv[2]).read())
groups={str(i):[{'features':[1.]+[float(i%2)]*15,'target':i%2}] for i in range(16)}
x=Learner(7,{'updates':48,'synthetic_fixture':True},state);x.advance(groups,int(sys.argv[1]));open(sys.argv[3],'wb').write(canonical(x.native()))
"""

    def run(stop, resume, out):
        subprocess.run(
            [sys.executable, "-c", code, str(stop), str(resume), str(out)],
            cwd=Path(__file__).parent,
            check=True,
            timeout=10,
        )

    w, s, v = [tmp_path / n for n in ["w", "s", "v"]]
    run(8, "-", w)
    run(4, "-", s)
    run(8, s, v)
    assert w.read_bytes() == v.read_bytes()


def test_literal_boundary_censor_and_fullhistory_packet_reject():
    import copy

    b = chess.Board()
    h = history(b)
    rows = []
    for game in range(16):
        for i in range(4):
            rows.append(
                dict(
                    history=h,
                    history_sha256=digest(h),
                    features=features(b, 0.0),
                    static=0.0,
                    reference_mover_value=0.2,
                    target=1,
                    status="completed",
                    trajectory_id=str(game),
                    additional_nodes=1,
                    leaf_ordinal=i,
                )
            )
    data = {"leaves": rows}
    groups = set(str(i) for i in range(16))
    assert len(admission(data, groups)) == 16
    bad = copy.deepcopy(data)
    bad["leaves"][0]["reference_mover_value"] = 0.10
    with pytest.raises(ValueError, match="boundary"):
        admission(bad, groups)
    bad = copy.deepcopy(data)
    bad["leaves"][0].update(status="censored", reference_mover_value=None, target=0)
    with pytest.raises(ValueError, match="censor"):
        admission(bad, groups)
    bad = copy.deepcopy(data)
    bad["leaves"][0]["features"][0] = 2
    with pytest.raises(ValueError, match="feature"):
        admission(bad, groups)


def test_exact_local64_and_global_consume_ceilings():
    s = Selective(lambda _: 0.0, nodes=128, quiescence_plies=2, max_depth=8)
    for _ in range(64):
        s.consume()
    with pytest.raises(base.BudgetExhausted):
        s.extra(chess.Board(), -math.inf, math.inf, 2, 0, 0)
    assert s.nodes == 64
    t = Selective(lambda _: 0.0, nodes=1, quiescence_plies=2, max_depth=8)
    t.consume()
    with pytest.raises(base.BudgetExhausted):
        t.consume()
    assert t.nodes == 1
