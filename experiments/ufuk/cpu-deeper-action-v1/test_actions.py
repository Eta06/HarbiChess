"""Static/gradient/native-serialization fixtures only; no search or Adam updates."""

import ast
import json
from pathlib import Path

import action_model as model
import chess
import pytest
from action_learner import Learner, row_gradient
from student_search import search_type


def test_legal_allaction_context_capture_ep_promotion():
    cases = [
        ("4k3/8/8/8/3p4/8/3R4/4K3 w - - 0 1", "d2d4", True, False),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", True, False),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n", False, True),
    ]
    for fen, text, capture, promotion in cases:
        board = chess.Board(fen)
        move = chess.Move.from_uci(text)
        before = board.fen()
        features = model.features(board, move)
        assert any(268 <= i < 274 for i in features) == capture
        assert any(274 <= i < 280 for i in features) == promotion
        assert board.fen() == before and all(0 <= i < 280 for i in features)


def test_legal_ce_not_value_or_visit_target():
    grad = row_gradient([0.0] * 280, {"features": [(1, 7), (2, 7)], "target": 0})
    assert grad[1] == -0.5 and grad[2] == 0.5 and grad[7] == 0


def test_native0_serialized_full_rng_and_corruption_rejection():
    contract = {"updates": 64, "source": "synthetic-no-fit"}
    original = Learner(7, contract)
    native = json.loads(json.dumps(original.native()))
    restored = Learner(7, contract, native)
    assert json.dumps(original.native(), sort_keys=True) == json.dumps(
        restored.native(), sort_keys=True
    )
    native["v"][0] = -1
    with pytest.raises(ValueError):
        Learner(7, contract, native)


def test_fixed64_contract_rejects_extended_budget():
    with pytest.raises(ValueError):
        Learner(1, {"updates": 65})


def test_root_correction_is_bounded_and_zero_delegates():
    class Base:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, board):
            return ("original-packet", -0.0, 512, 301, 2, 20)

    cls = search_type(Base, object, Exception, model)
    board = chess.Board()
    a = cls(None)
    assert a.search(board) == Base().search(board)
    moves = sorted(board.legal_moves, key=lambda m: m.uci())
    weights = [0.0] * 280
    for i in model.features(board, moves[-1]):
        weights[i] = 5
    a = cls(None, ordering_weights=weights)
    scores = {m: float(i) / 10 for i, m in enumerate(moves)}
    assert a.root_order(board, moves, scores) == sorted(moves, key=lambda m: (-scores[m], m.uci()))
    p = model.probabilities(weights, [model.features(board, m) for m in moves])
    delta = [model.EPSILON * (x - 1 / len(moves)) for x in p]
    assert max(delta) - min(delta) <= model.EPSILON


def test_original_selection_helper_unchanged():
    import hashlib

    raw = Path(__file__).with_name("select_roots.py").read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "ca571105a075bf23f5992a9a5856ab1d4dc3f7e4f43ffd9f69391e5faca7a17c"
    )


def test_search_ast_only_root_expression_changes():
    before = next(
        n
        for n in ast.walk(
            ast.parse(
                Path(
                    "/workspace/HarbiChess/experiments/ufuk/cpu-classical-own-v1/arena/search.py"
                ).read_text()
            )
        )
        if isinstance(n, ast.FunctionDef) and n.name == "search"
    )
    after = next(
        n
        for n in ast.walk(ast.parse(Path(__file__).with_name("student_search.py").read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "search"
    )
    after.body = after.body[1:]

    class Normalize(ast.NodeTransformer):
        def visit_Name(self, n):
            if n.id == "Result":
                n.id = "result_type"
            if n.id == "BudgetExhausted":
                n.id = "budget_exhausted"
            return n

        def visit_For(self, n):
            self.generic_visit(n)
            if (
                isinstance(n.iter, ast.Call)
                and isinstance(n.iter.func, ast.Name)
                and n.iter.func.id == "sorted"
                and n.iter.keywords
            ):
                n.iter = ast.parse("self.root_order(board,moves,scores)", mode="eval").body
            return n

    assert ast.dump(Normalize().visit(before), include_attributes=False) == ast.dump(
        after, include_attributes=False
    )


def test_tactical_and_terminal_range_slots_fixed():
    class Base:
        def __init__(self, *args, **kwargs):
            pass

    board = chess.Board("4k3/8/8/8/3p4/8/3R4/4K3 w - - 0 1")
    moves = sorted(board.legal_moves, key=lambda m: m.uci())
    weights = [0.25] * 280
    search = search_type(Base, object, Exception, model)(None, ordering_weights=weights)
    scores = {m: 0.0 for m in moves}
    terminal_like = next(m for m in moves if not board.is_capture(m))
    scores[terminal_like] = 1.9
    old = sorted(moves, key=lambda m: (-scores[m], m.uci()))
    new = search.root_order(board, moves, scores)
    for i, m in enumerate(old):
        if board.is_capture(m) or m.promotion or abs(scores[m]) > 1:
            assert new[i] == m
    assert set(new) == set(old)


def test_training_root_admission_ast_unchanged():
    original = Path(
        "/workspace/work/harbichess/cpu-own-search-reanalysis-proposal/run_reanalysis.py"
    )
    changed = Path(__file__).with_name("produce_actions.py")

    def group_ast(path):
        return next(
            n
            for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.FunctionDef) and n.name == "_position_groups"
        )

    assert ast.dump(group_ast(original), include_attributes=False) == ast.dump(
        group_ast(changed), include_attributes=False
    )


def test_fake_producer_black_mover_alllegal_context_and_reference():
    from types import SimpleNamespace

    from action_labels import run_selected

    row = {
        "row_id": "synthetic:0",
        "trajectory_id": "synthetic",
        "root_fen": chess.STARTING_FEN,
        "prefix_uci": ["e2e4"],
    }

    class FakePlanner:
        def search(self, b):
            return SimpleNamespace(
                move=chess.Move.from_uci("e7e5"),
                nodes=len(list(b.legal_moves)) + 1,
                evaluations=len(list(b.legal_moves)),
                completed_depth=1,
                root_actions=len(list(b.legal_moves)),
                value=-0.2,
            )

    first = run_selected([row], FakePlanner, expected_count=1, state_features=lambda b: [0.0] * 18)[
        0
    ]
    assert first["root_mover"] == "black" and first["raw_q_mover"] == -0.2
    assert (
        first["actions"][first["target"]] == "e7e5"
        and len(first["features"]) == first["root_actions"]
    )
    reference = {"synthetic:0": dict(first)}
    run_selected(
        [row],
        FakePlanner,
        expected_count=1,
        state_features=lambda b: [0.0] * 18,
        reference=reference,
    )
    reference["synthetic:0"]["raw_q_mover"] = 0.2
    with pytest.raises(ValueError):
        run_selected(
            [row],
            FakePlanner,
            expected_count=1,
            state_features=lambda b: [0.0] * 18,
            reference=reference,
        )
